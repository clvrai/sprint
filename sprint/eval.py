from collections import defaultdict
import queue
import glob
import time
import json
import numpy as np
import regex as re
import torch
import os
import argparse
import random
import wandb
import threading
import signal
import sys

from sprint.utils.wandb_info import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME
from sprint.models.sprint_model import SPRINTETIQLModel
from sprint.models.actionable_models import AMETIQLModel
from sprint.utils.utils import AttrDict, str2bool, load_goal_states, cleanup_mp
from sprint.rollouts.rollout import run_policy_multi_process
from torch import nn
from tqdm import tqdm
from sprint.alfred.models.nn.resnet import Resnet

COMPOSITE_EPOCH_TO_STEP_RATIO = 2881
PRIMITIVE_EPOCH_TO_STEP_RATIO = 1027


def setup_mp(
    result_queue,
    task_queue,
    agent_model,
    resnet,
    config,
    device,
    goal_states,
):
    num_workers = config.num_rollout_workers
    workers = []
    # start workers
    worker_target = run_policy_multi_process
    for _ in range(num_workers):
        worker = threading.Thread(
            target=worker_target,
            args=(
                result_queue,
                task_queue,
                agent_model,
                resnet,
                device,
                config.max_skill_length,
                goal_states,
                config.env_type,
                config.eval_json,
                config.specific_task,
            ),
        )
        worker.daemon = True  # kills thread/process when parent thread terminates
        worker.start()
        time.sleep(0.5)
        workers.append(worker)
        num_tasks = result_queue.get()
    return workers, num_tasks


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
    # asynchronously collect results from result_queue
    num_env_samples = 0
    num_finished_tasks = 0
    num_rollouts = (
        config.num_eval_tasks if eval else config.num_rollouts_per_epoch
    )
    with tqdm(total=num_rollouts) as pbar:
        while num_finished_tasks < num_rollouts:
            result = result_queue.get()
            rollout_returns.append(result["rews"].sum().item())
            subgoal_successes.append(result["dones"][-1])
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


def multiprocess_rollout(
    task_queue,
    result_queue,
    config,
    epsilon,
    dataset,
    eval,
    make_video,
    dataset_agg_func,
):
    rollout_returns = []
    subgoal_successes = []
    rollout_gifs = []
    video_captions = []
    extra_info = defaultdict(list)
    num_rollouts = (
        config.num_eval_tasks if eval else config.num_rollouts_per_epoch
    )
    # create tasks for MP Queue

    args_func = lambda subgoal: (
        True if eval else config.deterministic_action,
        True if eval else False,  # log to video
        epsilon,
        subgoal if eval else None,
    )

    for subgoal in range(num_rollouts):
        task_queue.put(args_func(subgoal))

    num_env_samples_list = []  # use list for thread safety
    dataset_agg_func(
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
    )

    num_env_samples = num_env_samples_list[0]
    # aggregate metrics
    rollout_metrics = dict(
        average_return=np.mean(rollout_returns),
        subgoal_success=np.mean(subgoal_successes),
    )
    for key, value in extra_info.items():
        rollout_metrics[key] = np.mean(value)

    # generate per-length return and subgoal success data
    skill_lengths = extra_info["skill_lengths"]
    per_number_return = defaultdict(list)
    per_number_success = defaultdict(list)
    for skill_length, success, rollout_return in zip(
        skill_lengths, subgoal_successes, rollout_returns
    ):
        per_number_return[skill_length].append(rollout_return)
        per_number_success[skill_length].append(success)
    for num_attempts, returns in per_number_return.items():
        rollout_metrics[f"length_{num_attempts}_return"] = np.mean(returns)
        rollout_metrics[f"length_{num_attempts}_success"] = np.mean(
            per_number_success[num_attempts]
        )

    if make_video:
        # sort both rollout_gifs and video_captions by the caption so that we have a consistent ordering
        rollout_gifs, video_captions = zip(
            *sorted(zip(rollout_gifs, video_captions), key=lambda x: x[1])
        )
        for i, (gif, caption) in enumerate(zip(rollout_gifs, video_captions)):
            rollout_metrics["videos_%d" % i] = wandb.Video(
                gif, caption=caption, fps=3, format="mp4"
            )
    return rollout_metrics, num_env_samples


def get_list_of_checkpoints(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    new_files_to_add = []
    checkpoint_files += new_files_to_add
    checkpoint_files.sort(
        key=lambda f: int(re.search(r"\d+.pth", f).group().split(".pth")[0])
    )
    epochs = [
        int(re.search(r"\d+.pth", f).group().split(".pth")[0]) for f in checkpoint_files
    ]
    print(epochs)
    return checkpoint_files, epochs


def main(config):
    config.eval_json = (
        f"{os.environ['SPRINT']}/sprint/rollouts/{config.env_type}_ann_human.json"
    )
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    run = wandb.init(
        resume=config.experiment_name,
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        notes=config.notes,
        config=config,
        group=config.run_group,
    )
    model_checkpoint_dir = config.model_checkpoint_dir
    print(model_checkpoint_dir)
    config.use_llm = False  # set use llm to false so it doesn't load the LLM
    list_of_checkpoints, list_of_epochs = get_list_of_checkpoints(model_checkpoint_dir)
    # load one of the checkpoints' configs

    checkpoint = torch.load(list_of_checkpoints[-1], map_location="cpu")
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

    # step ratio just makes the x-axis consistent on wandb, not important
    step_ratio = None
    if vars(old_config)["sample_primitive_skill"]:
        step_ratio = PRIMITIVE_EPOCH_TO_STEP_RATIO
    else:
        step_ratio = COMPOSITE_EPOCH_TO_STEP_RATIO

    device = torch.device(config.gpus[0])

    if vars(config)["model"] == "sprint":
        agent_model = SPRINTETIQLModel(config)
        goal_states = None
    elif vars(config)["model"] == "am":
        agent_model = AMETIQLModel(config)
        goal_states = load_goal_states(
            config
        )  # for actionable models goal specification

    if len(config.gpus) > 1:
        print(f"-----Using {len(config.gpus)} GPUs-----")
        agent_model = nn.DataParallel(
            agent_model,
            device_ids=config.gpus,
        )
    agent_model.to(device)

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

    # add a 0-data eval
    if config.start_epoch < 0:
        list_of_epochs.insert(0, -1)
        list_of_checkpoints.insert(0, None)
    for checkpoint_path, epoch in zip(list_of_checkpoints, list_of_epochs):
        if epoch < config.start_epoch or (epoch + 1) % config.eval_interval != 0:
            continue
        print(f"-----Loading checkpoint {checkpoint_path}-----")
        if checkpoint_path is not None:
            checkpoint = torch.load(os.path.join(checkpoint_path), map_location="cpu")
            agent_model.load_from_checkpoint(checkpoint)
        # if no checkpoint just do the eval anyway for the 0-epoch eval
        # Eval
        log_dict = {}
        eval_metrics, _ = multiprocess_rollout(
            task_queue,
            result_queue,
            config,
            0,
            None,
            eval=True,
            make_video=True,  # epoch == list_of_epochs[-1],  # only make video for last epoch
            dataset_agg_func=multithread_dataset_aggregation,
        )
        for k, v in eval_metrics.items():
            log_dict[f"eval_{k}"] = v
        wandb.log(
            log_dict,
            step=(epoch + 1) * step_ratio,
        )

    cleanup_mp(task_queue, processes)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Finetune models on an existing dataset of annotated multi-length skills."
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
        "--num_rollout_workers",
        type=int,
        default=4,
        help="number of workers for policy rollouts",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        required=True,
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
        "--model_checkpoint_dir",
        type=str,
        default=None,
        required=True,
        help="path to load the model checkpoints from to for eval",
    )
    parser.add_argument(
        "--run_group",
        type=str,
        default=None,
        help="group to run the experiment in. If None, no group is used",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=-1,
        help="epoch to start eval from",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="interval between epochs to eval",
    )
    parser.add_argument(
        "--use_summarized_instruction",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="if enabled, uses the summarized annotation. Otherwise concats everything together.",
    )
    parser.add_argument(
        "--specific_task",
        type=int,
        default=None,
        help="if specified, only eval on this task index",
    )
    config = parser.parse_args()
    config.gpus = [int(gpu) for gpu in config.gpus.strip().split(",")]
    base_config = parser.parse_args()
    base_config.gpus = [int(gpu) for gpu in base_config.gpus.strip().split(",")]

    config = AttrDict()

    # HYPER PARAMETERS
    # batch size
    # optimizer type, must be in ('adam', 'adamw')
    config.optimizer = "adamw"
    # L2 regularization weight
    config.weight_decay = 0.33
    # num epochs
    config.epochs = 27
    # learning rate settings
    config.lr = {
        # learning rate initial value
        "init": 1e-4,
        # lr scheduler type: {'linear', 'cosine', 'triangular', 'triangular2'}
        "profile": "linear",
        # (LINEAR PROFILE) num epoch to adjust learning rate
        "decay_epoch": 10,
        # (LINEAR PROFILE) scaling multiplier at each milestone
        "decay_scale": 0.1,
        # (COSINE & TRIANGULAR PROFILE) learning rate final value
        "final": 1e-5,
        # (TRIANGULAR PROFILE) period of the cycle to increase the learning rate
        "cycle_epoch_up": 0,
        # (TRIANGULAR PROFILE) period of the cycle to decrease the learning rate
        "cycle_epoch_down": 0,
        # warm up period length in epochs
        "warmup_epoch": 0,
        # initial learning rate will be divided by this value
        "warmup_scale": 1,
    }
    # weight of action loss
    config.action_loss_wt = 1.0
    # weight of object loss
    config.object_loss_wt = 1.0
    # weight of subgoal completion predictor
    config.subgoal_aux_loss_wt = 0.1
    # weight of progress monitor
    config.progress_aux_loss_wt = 0.1
    # maximizing entropy loss (by default it is off)
    config.entropy_wt = 0.0

    # TRANSFORMER settings
    # size of transformer embeddings
    config.demb = 768
    # number of heads in multi-head attention
    config.encoder_heads = 12
    # number of layers in transformer encoder
    config.encoder_layers = 2
    # how many previous actions to use as input
    config.num_input_actions = 1
    # which encoder to use for language encoder (by default no encoder)
    config.encoder_lang = {
        "shared": True,
        "layers": 2,
        "pos_enc": True,
        "instr_enc": False,
    }
    # which decoder to use for the speaker model
    config.decoder_lang = {
        "layers": 2,
        "heads": 12,
        "demb": 768,
        "dropout": 0.1,
        "pos_enc": True,
    }
    # do not propagate gradients to the look-up table and the language encoder
    config.detach_lang_emb = False

    # DROPOUTS
    config.dropout = {
        # dropout rate for language (goal + instr)
        "lang": 0.0,
        # dropout rate for Resnet feats
        "vis": 0.3,
        # dropout rate for processed lang and visual embeddings
        "emb": 0.0,
        # transformer model specific dropouts
        "transformer": {
            # dropout for transformer encoder
            "encoder": 0.1,
            # remove previous actions
            "action": 0.0,
        },
    }

    # ENCODINGS
    config.enc = {
        # use positional encoding
        "pos": True,
        # use learned positional encoding
        "pos_learn": False,
        # use learned token ([WORD] or [IMG]) encoding
        "token": False,
        # dataset id learned encoding
        "dataset": False,
    }
    config.update(vars(base_config))
    main(config)
