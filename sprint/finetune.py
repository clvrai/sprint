from collections import defaultdict
import queue
import time
import json
import numpy as np
from tqdm import tqdm
import torch
import os
from torch import nn
import argparse
import random
import wandb
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import signal
import sys

from sprint.utils.wandb_info import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from sprint.alfred.models.nn.resnet import Resnet
from sprint.models.sprint_model import SPRINTETIQLModel
from sprint.models.actionable_models import AMETIQLModel
from sprint.eval import setup_mp, multiprocess_rollout, get_list_of_checkpoints
from sprint.utils.utils import (
    AttrDict,
    str2bool,
    send_to_device_if_not_none,
    cleanup_mp,
    load_goal_states,
)
from sprint.utils.data_utils import CombinedDataset
from sprint.dataloaders.sprint_dataloader import RLBuffer, collate_func


EVAL_INTERVAL = 10000


def train_epoch(
    agent_model,
    dataloader,
    epoch,
    device,
):
    agent_model.train()
    running_tracker_dict = defaultdict(list)
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_i, data_dict in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            frames = send_to_device_if_not_none(data_dict, "skill_feature", device)
            action = send_to_device_if_not_none(data_dict, "low_action", device).long()
            obj_id = send_to_device_if_not_none(data_dict, "object_ids", device).long()
            interact_mask = send_to_device_if_not_none(
                data_dict, "valid_interact", device
            )
            lengths_frames = send_to_device_if_not_none(
                data_dict, "feature_length", device
            )
            lengths_lang = send_to_device_if_not_none(data_dict, "token_length", device)
            lang = send_to_device_if_not_none(data_dict, "ann_token", device).int()
            rewards = send_to_device_if_not_none(data_dict, "reward", device)
            terminals = send_to_device_if_not_none(data_dict, "terminal", device)
            goal_frames = send_to_device_if_not_none(data_dict, "goal_feature", device)
            summary_lang_list = data_dict["summary_token_list"]
            summary_lengths_lang = send_to_device_if_not_none(
                data_dict, "summary_token_length", device
            )

            loss_info = agent_model.train_offline_from_batch(
                frames,
                lang,
                action,
                obj_id,
                lengths_frames,
                lengths_lang,
                interact_mask,
                rewards,
                terminals,
                state_goal=goal_frames,
                state_text_summaries_list=summary_lang_list,
                lengths_state_text_summaries=summary_lengths_lang,
            )
            tepoch.set_postfix(
                loss=loss_info["policy_total_loss"],
                vf_loss=loss_info["vf_loss"],
            )
            for k, v in loss_info.items():
                running_tracker_dict[k].append(v)
    eval_metrics = {}
    for k, v in running_tracker_dict.items():
        eval_metrics[f"train_{k}"] = np.mean(v)
    return eval_metrics


def multithread_dataset_aggregation(
    result_queue,
    rollout_returns,
    subgoal_successes,
    rollout_gifs,
    video_captions,
    extra_info,
    dataset,
    config,
    num_env_samples_list,
    eval,
):
    # asynchronously collect results from result_queue and adds to the replay buffer
    num_env_samples = 0
    num_finished_tasks = 0
    num_rollouts = (
        config.num_eval_tasks if eval else config.num_rollouts_per_epoch
    )
    with tqdm(total=num_rollouts) as pbar:
        while num_finished_tasks < num_rollouts:
            result = result_queue.get()
            if not eval:
                dataset.add_traj_to_buffer(
                    result["obs"],
                    result["acs"],
                    result["obj_acs"],
                    result["rews"],
                    result["dones"],
                    result["chained_language_instruction"],
                    result["goal_state"] if "goal_state" in result else None,
                )
            rollout_returns.append(result["rews"].sum().item())
            subgoal_successes.append(result["dones"][-1])
            if result["video_frames"] is not None:
                rollout_gifs.append(result["video_frames"])
            video_captions.append(result["video_caption"])
            extra_info["skill_lengths"].append(result["skill_length"])
            num_env_samples += result["rews"].shape[0]
            num_finished_tasks += 1
            pbar.update(1)
            if eval:
                pbar.set_description(
                    "EVAL: Finished %d/%d rollouts" % (num_finished_tasks, num_rollouts)
                )
            else:
                pbar.set_description(
                    "TRAIN: Finished %d/%d rollouts"
                    % (num_finished_tasks, num_rollouts)
                )
    num_env_samples_list.append(num_env_samples)


def main(config):
    config.eval_json = (
        f"{os.environ['SPRINT']}/sprint/rollouts/{config.env_type}_ann_human.json"
    )
    model_checkpoint_dir = config.model_checkpoint_dir
    list_of_checkpoints, _ = get_list_of_checkpoints(model_checkpoint_dir)

    print(f"Loading from checkpoint {list_of_checkpoints[-1]}...")
    checkpoint = torch.load(list_of_checkpoints[-1], map_location="cpu")
    print("Loaded!")

    old_config = checkpoint["config"]
    # overwrite only certain aspects of the config
    for key in vars(old_config):
        if (
            key not in vars(config)
            or vars(config)[key] is None
            and vars(old_config)[key] is not None
            and key != "experiment_name"
        ):
            vars(config)[key] = vars(old_config)[key]
    goal_states = None
    if "actionable_models" in vars(old_config) and old_config.actionable_models:
        goal_states = load_goal_states(
            config
        )  # for actionable models goal specification

    run = wandb.init(
        resume=config.experiment_name,
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        notes=config.notes,
        config=config,
        group=config.run_group,
    )
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(
        os.path.join(
            config.save_dir,
            (
                config.experiment_name
                if config.experiment_name is not None
                else wandb.run.name
            ),
        ),
        exist_ok=True,
    )

    device = torch.device(config.gpus[0])
    if vars(config)["model"] == "sprint":
        agent_model = SPRINTETIQLModel(config)
        goal_states = None
    elif vars(config)["model"] == "am":
        agent_model = AMETIQLModel(config)
        goal_states = load_goal_states(
            config
        )  # for actionable models goal specification

    drop_old_data = True
    if config.old_new_sampling:
        drop_old_data = False
    dataset = RLBuffer(
        path=config.data_dir,
        split="train",
        use_full_skill=False,
        max_skill_length=config.max_skill_length,
        drop_old_data=drop_old_data,
        sample_primitive_skill=old_config.sample_primitive_skill,
        use_llm_labels=old_config.use_llm_labels,
        max_size=config.max_buffer_size,
    )

    if not config.drop_old_data:
        print(
            f"Not dropping old data, use_llm_labels={old_config.use_llm_labels}, sample_primitive_skill={old_config.sample_primitive_skill}"
        )

    # if loading an already partially-finetuned RL model, then start from the same num samples and restore the replay buffer
    num_env_samples = 0
    start_epoch = 0
    if "num_env_samples" in checkpoint:
        num_env_samples = checkpoint["num_env_samples"]
        start_epoch = checkpoint["epoch"]
        dataset.rl_buffer = checkpoint["saved_buffer"]
        print(f"Restoring from epoch {start_epoch}, num_env_samples {num_env_samples}")

    if config.equal_traj_type_sampling or config.old_new_sampling:
        dataset_failure = RLBuffer(
            path=config.data_dir,
            split="train",
            use_full_skill=False,
            max_skill_length=config.max_skill_length,
            drop_old_data=True,  # always drop old data for failure dataset, since the regular datset may contain the old data which is all successful demos
            sample_primitive_skill=old_config.sample_primitive_skill,
            use_llm_labels=old_config.use_llm_labels,
            max_size=config.max_buffer_size,
        )
        # weigh both equally
        dataset_random_sampler = torch.utils.data.WeightedRandomSampler(
            weights=[0.5, 0.5],
            num_samples=config.num_updates_per_epoch * config.batch_size,
            replacement=True,
        )
        if config.equal_traj_type_sampling and config.old_new_sampling:
            dataset_failure_2 = RLBuffer(
                path=config.data_dir,
                split="train",
                use_full_skill=False,
                max_skill_length=config.max_skill_length,
                drop_old_data=True,  # always drop old data for failure dataset, since the regular datset may contain the old data which is all success
                sample_primitive_skill=old_config.sample_primitive_skill,
                use_llm_labels=old_config.use_llm_labels,
                max_size=config.max_buffer_size,
            )
            # this represents the RL replay buffer and it allows for equal success/fail trajectory sampling
            dataset_failure = CombinedDataset(
                dataset_failure,
                dataset_failure_2,
                config.count_partial_success,
                old_new_sampling=False,
                first_dataset_ratio=0.5,
            )
        dataset = CombinedDataset(
            dataset,
            dataset_failure,
            config.count_partial_success,
            old_new_sampling=config.old_new_sampling,
            first_dataset_ratio=config.old_data_ratio,
        )
    else:
        dataset_random_sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=True,
            num_samples=config.num_updates_per_epoch * config.batch_size,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=dataset_random_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_func,
    )

    if len(config.gpus) > 1:
        print(f"-----Using {len(config.gpus)} GPUs-----")
        agent_model = nn.DataParallel(
            agent_model,
            device_ids=config.gpus,
        )
    agent_model.to(device)

    agent_model.load_from_checkpoint(checkpoint)
    if config.new_lr is not None:
        agent_model.set_lr(config.new_lr)

    if not config.use_auxiliary_objectives:
        # disable SPRINT chaining objectives as we're training on new data
        agent_model.chain_multi_trajectory = False
    # enable training with advantage in case using L-BC or some other baseline that doesn't train with the offline RL objective during pre-training
    agent_model.train_with_advantage = True

    resnet_args = AttrDict()
    resnet_args.visual_model = "resnet18"
    resnet_args.gpu = config.gpus[0]
    resnet = Resnet(resnet_args, eval=True, use_conv_feat=True)

    # multiprocessed rollout setup
    task_queue = queue.SimpleQueue()
    result_queue = queue.SimpleQueue()

    processes, num_eval_tasks = setup_mp(
        result_queue,
        task_queue,
        agent_model,
        resnet,
        config,
        device,
        goal_states,
    )
    config.num_eval_tasks = num_eval_tasks

    def signal_handler(sig, frame):
        print("SIGINT received. Exiting...closing all processes first")
        cleanup_mp(task_queue, processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    epoch = start_epoch
    if epoch == 0:
        log_dict = {}
        eval_metrics, _ = multiprocess_rollout(
            task_queue,
            result_queue,
            config,
            0,
            dataset,
            eval=True,
            make_video=True,
            dataset_agg_func=multithread_dataset_aggregation,
        )
        for k, v in eval_metrics.items():
            log_dict[f"eval_{k}"] = v
        wandb.log(
            log_dict,
            step=num_env_samples,
        )
    while num_env_samples < config.total_env_steps:
        epoch += 1
        log_dict = {}
        eval_metrics = None
        decayed_epsilon = max(
            config.epsilon * (config.linear_decay_epsilon**epoch),
            config.min_epsilon,
        )
        if num_env_samples == 0:
            # collect warmup samples
            old_num_rollouts_per_epoch = config.num_rollouts_per_epoch
            config.num_rollouts_per_epoch = 50
            result_metrics, new_env_samples = multiprocess_rollout(
                task_queue,
                result_queue,
                config,
                decayed_epsilon,  # no epsilon
                dataset,
                eval=False,
                make_video=False,
                dataset_agg_func=multithread_dataset_aggregation,
            )
            config.num_rollouts_per_epoch = old_num_rollouts_per_epoch
        else:
            result_metrics, new_env_samples = multiprocess_rollout(
                task_queue,
                result_queue,
                config,
                decayed_epsilon,  # no epsilon
                dataset,
                eval=False,
                make_video=False,
                dataset_agg_func=multithread_dataset_aggregation,
            )
        for k, v in result_metrics.items():
            log_dict[f"rollout_{k}"] = v
        num_env_samples += new_env_samples

        # train policy on a few updates
        training_metrics = train_epoch(
            agent_model,
            dataloader,
            epoch,
            device,
        )
        log_dict.update(training_metrics)
        log_dict["epoch"] = epoch

        # Eval
        if (
            num_env_samples % EVAL_INTERVAL
            < (num_env_samples - new_env_samples) % EVAL_INTERVAL
        ) or num_env_samples >= config.total_env_steps:
            eval_metrics, _ = multiprocess_rollout(
                task_queue,
                result_queue,
                config,
                0,
                dataset,
                eval=True,
                make_video=True,
                dataset_agg_func=multithread_dataset_aggregation,
            )
        if eval_metrics:
            for k, v in eval_metrics.items():
                log_dict[f"eval_{k}"] = v
        wandb.log(
            log_dict,
            step=num_env_samples,
        )

        model_state_dict = agent_model.get_all_state_dicts()
        model_state_dict.update(
            dict(
                epoch=epoch,
                num_env_samples=num_env_samples,
                saved_buffer=dataset.rl_buffer,
            )
        )
        if epoch % 5 == 0 or num_env_samples >= config.total_env_steps:
            torch.save(
                model_state_dict,
                os.path.join(
                    config.save_dir,
                    (
                        config.experiment_name
                        if config.experiment_name
                        else wandb.run.name
                    ),
                    f"rl_finetune_model.pth",
                ),
            )

    cleanup_mp(task_queue, processes)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune an ALFRED offline RL model on existing low-level, language annotated skills"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/px_llama_13b/px_llama_13b",
        help="Parent directory containing the dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="sprint_saved_rl_models/",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--new_lr",
        type=float,
        default=None,
        help="finetuning learning rate, if none uses the original.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="which gpus. pass in as comma separated string to use DataParallel on multiple GPUs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loading"
    )
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=4,
        help="number of parallel workers for policy rollouts",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="name of the experiment for logging on WandB",
    )
    parser.add_argument(
        "--notes", type=str, default="", help="optional notes for logging on WandB"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        required=True,
        choices=["eval_instruct", "eval_length", "eval_scene"],
        help="alfred environment to use",
    )
    parser.add_argument(
        "--specific_task",
        type=int,
        default=None,
        help="if specified, only train on this task index",
    )
    parser.add_argument(
        "--subgoal_type",
        type=str,
        default="failure",
        choices=["success", "failure"],
        help="whether to sample successful or failed subgoals for training",
    )
    parser.add_argument(
        "--max_skill_length",
        type=int,
        default=None,
        help="length of context window. use this to overwrite.",
    )
    parser.add_argument(
        "--total_env_steps",
        type=int,
        default=50000,
        help="number of timesteps to train for",
    )
    parser.add_argument(
        "--num_updates_per_epoch",
        type=int,
        default=16,
        help="number of updates per epoch",
    )
    parser.add_argument(
        "--num_rollouts_per_epoch",
        type=int,
        default=10,
        help="number of env rollouts per epoch when training online",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="quantile to use for quantile regression. defaults to None so no overwriting",
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default=None,
        required=True,
        help="path to load the model from to finetune from",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.00,
        help="starting epsilon for epsilon-greedy action sampling",
    )
    parser.add_argument(
        "--linear_decay_epsilon",
        type=float,
        default=1.0,
        help="epsilon decay rate for linear decay. Multipled per epoch",
    )
    parser.add_argument(
        "--min_epsilon",
        type=float,
        default=0.00,
        help="minimum epsilon for epsilon-greedy action sampling, after decay",
    )
    parser.add_argument(
        "--deterministic_action",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to use deterministic action sampling during training",
    )
    parser.add_argument(
        "--advantage_temp",
        type=float,
        default=None,
        help="temperature for computing the advantage. setting this overrides the inherited value",
    )
    parser.add_argument(
        "--run_group",
        type=str,
        default=None,
        help="group to run the experiment in. If None, no group is used",
    )
    parser.add_argument(
        "--drop_old_data",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to drop the old data after loading checkpoint",
    )
    parser.add_argument(
        "--scene_type",
        type=str,
        default="valid_unseen",
        # default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen"],
        help="which type of scenes to sample from/evaluate on",
    )
    parser.add_argument(
        "--equal_traj_type_sampling",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to sample trajectories of equal type (success/failure), 50/50, for training. Helps with sparse-reward RL.",
    )
    parser.add_argument(
        "--count_partial_success",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to count partial success as success for success/fail buffer sampling",
    )
    parser.add_argument(
        "--fractional_rewards",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to use fractional rewards or total sum",
    )
    parser.add_argument(
        "--old_new_sampling",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to sample old and new trajectories simultaneously for training. Empirically req'd to get transformer-based online RL to work",
    )
    parser.add_argument(
        "--old_data_ratio",
        type=float,
        default=0.7,
        help="ratio of old data to sample for training",
    )
    parser.add_argument(
        "--use_auxiliary_objectives",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to use auxiliary objectives from SPRINT (chaining) for online RL",
    )
    parser.add_argument(
        "--max_buffer_size",
        type=float,
        default=float("inf"),
        help="maximum size of the replay buffer",
    )
    config = parser.parse_args()
    config.gpus = [int(gpu) for gpu in config.gpus.strip().split(",")]
    mp.set_sharing_strategy(
        "file_system"
    )  # to ensure the too many open files error doesn't happen

    main(config)
