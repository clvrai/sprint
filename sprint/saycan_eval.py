from collections import defaultdict
import json
import copy
import queue
import time
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
import argparse
import random
import wandb
from sprint.alfred.models.nn.resnet import Resnet
import torch.multiprocessing as mp
import threading
import signal
import sys
import revtok
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


from sprint.models.sprint_model import SPRINTETIQLModel
from sprint.eval import get_list_of_checkpoints
from sprint.models.saycan_llm import SaycanPlanner
from sprint.utils.utils import (
    generate_primitive_skill_list_from_eval_skill_info_list,
    AttrDict,
    str2bool,
    cleanup_mp,
)
from sprint.utils.utils import make_primitive_annotation_eval_dataset
from sprint.utils.data_utils import process_annotation
from sprint.rollouts.saycan_rollout import run_policy_multi_process

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WANDB_ENTITY_NAME = "clvr"
WANDB_PROJECT_NAME = "p-bootstrap-llm"


class ETLanguageEncoder:
    def __init__(self, vocab_word):
        self.vocab_word = vocab_word

    def encode(self, annotations, convert_to_tensor=False):
        if convert_to_tensor:
            return pad_sequence(
                [process_annotation(a, self.vocab_word).long() for a in annotations],
                batch_first=True,
                padding_value=0,
            )
        return [process_annotation(a, self.vocab_word).long() for a in annotations]


def setup_mp(
    result_queue,
    task_queue,
    saycan_planner,
    sentence_embedder,
    agent_model,
    resnet,
    config,
    device,
    subgoal_pool,
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
                saycan_planner,
                sentence_embedder,
                agent_model,
                resnet,
                device,
                subgoal_pool,
                config.max_skill_length,
                goal_states,
            ),
        )
        worker.daemon = True  # kills thread/process when parent thread terminates
        worker.start()
        time.sleep(0.5)
        workers.append(worker)
    return workers


def setup_mp(
    result_queue,
    task_queue,
    agent_model,
    saycan_planner,
    sentence_encoder,
    resnet,
    config,
    device,
):
    num_workers = config.num_rollout_workers
    workers = []
    # start workers
    worker_class = threading.Thread
    worker_target = run_policy_multi_process
    for _ in range(num_workers):
        worker = worker_class(
            target=worker_target,
            args=(
                result_queue,
                task_queue,
                agent_model,
                saycan_planner,
                sentence_encoder,
                resnet,
                config,
                device,
            ),
        )
        worker.daemon = True  # kills thread/process when parent thread terminates
        worker.start()
        time.sleep(0.5)
        workers.append(worker)
    return workers


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
        config.num_subgoals_in_pool if eval else config.num_rollouts_per_epoch
    )
    with tqdm(total=num_rollouts) as pbar:
        while num_finished_tasks < num_rollouts:
            result = result_queue.get()
            rollout_returns.append(result["rews"].sum().item())
            subgoal_successes.append(result["dones"][-1])
            rollout_gifs.append(result["video_frames"])
            video_captions.append(result["video_caption"])
            extra_info["skill_lengths"].append(result["skill_length"])
            if "completed_skills" in result:
                extra_info["completed_skills"].append(result["completed_skills"])
            if "predicted_skills" in result:
                extra_info["predicted_skills"].append(result["predicted_skills"])
            if "high_level_skill" in result:
                extra_info["high_level_skill"].append(result["high_level_skill"])
            if "ground_truth_sequence" in result:
                extra_info["ground_truth_sequence"].append(
                    result["ground_truth_sequence"]
                )
            num_env_samples += result["obs"].shape[0]
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
    num_rollouts = config.num_subgoals_in_pool if eval else config.num_rollouts_per_epoch

    # create tasks for thread/process Queue
    args_func = lambda subgoal: (
        config.env_type,
        config.num_subgoals_in_pool,
        True if eval else config.deterministic_action,
        True if eval else False,  # log to video
        epsilon,
        subgoal if eval else None,
    )

    for i in range(num_rollouts):
        eval_list = eval_skill_info_list
        eval_skill_index = i
        task_queue.put(args_func(eval_skill_index, eval_list))

    num_env_samples_list = []  # use list for thread safety
    multithread_dataset_aggregation(
        result_queue,
        rollout_returns,
        subgoal_successes,
        rollout_gifs,
        video_captions,
        extra_info,
        None,
        config,
        num_env_samples_list,
        eval,
    )

    num_env_samples = num_env_samples_list[0]
    # make a WandB table for the high level skill, ground truth sequence, predicted, completed skills
    saycan_completed_skill_data = []
    keys = [
        "high_level_skill",
        "ground_truth_sequence",
        "predicted_skills",
        "completed_skills",
    ]
    for i in range(len(extra_info["high_level_skill"])):
        saycan_completed_skill_data.append([])
        for key in keys:
            saycan_completed_skill_data[-1].append(extra_info[key][i])

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
    table = wandb.Table(
        columns=[
            "High Level Skill",
            "Ground Truth Sequence",
            "Predicted Skills",
            "Completed Skills",
        ],
        data=saycan_completed_skill_data,
    )
    rollout_metrics["evaluation_table"] = table
    renamed_metrics = {}
    for key in rollout_metrics:
        renamed_metrics[f"{rollout_mode}_{key}"] = rollout_metrics[key]
    return renamed_metrics, num_env_samples


def main(config):
    model_checkpoint_dir = config.model_checkpoint_dir
    print(model_checkpoint_dir)
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

    run = wandb.init(
        # resume=config.experiment_name,  # "allow",
        # name=config.experiment_name,
        id=config.experiment_name,
        name=config.experiment_name,
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        notes=config.notes,
        config=config,
        group=config.run_group,
    )
    seed = config.seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = False
        random.seed(seed)
    torch.backends.cudnn.benchmark = False  # uses a lot of gpu memory if True

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
    agent_model = SPRINTETIQLModel(config)
    goal_states = None
    if len(config.gpus) > 1:
        print(f"-----Using {len(config.gpus)} GPUs-----")
        agent_model = nn.DataParallel(
            agent_model,
            device_ids=config.gpus,
        )
    agent_model.to(device)

    agent_model.load_from_checkpoint(checkpoint)

    # print(agent_model)

    resnet_args = AttrDict()
    resnet_args.visual_model = "resnet18"
    resnet_args.gpu = config.gpus[0]
    resnet = Resnet(resnet_args, eval=True, use_conv_feat=True)
    sentence_encoder = ETLanguageEncoder(agent_model.vocab_word)

    saycan_planner = SaycanPlanner(config)

    # multiprocessed rollout setup
    task_queue = queue.SimpleQueue()
    result_queue = queue.SimpleQueue()

    with open(config.eval_json, "r") as f:
        eval_skill_info_list = json.load(f)

    # sort skill info list by num_primitive_skills, descending, for faster evaluation with multiple threads
    # eval_skill_info_list.sort(
    #    key=lambda x: sum(
    #        [
    #            len(primitive_skill["api_action"])
    #            for primitive_skill in x["primitive_skills"]
    #        ]
    #    ),
    #    reverse=True,
    # )
    eval_skill_info_list.sort(key=lambda x: len(x["primitive_skills"]), reverse=True)
    with open(f"scene_sampling/{config.scene_type}_scene.json", "r") as f:
        all_scenes_json = json.load(f)
    floorplan_set = set()
    all_floorplans = [
        all_scenes_json[x["primitive_skills"][0]["scene_index"]]["scene_num"]
        for x in eval_skill_info_list
    ]
    unique_floorplans = [
        x for x in all_floorplans if not (x in floorplan_set or floorplan_set.add(x))
    ]

    if config.eval_per_task_in_json:
        eval_skill_info_list = [
            x for x in eval_skill_info_list if len(x["primitive_skills"])
        ]

        sorted_task_names = [
            dict(
                task=task["task"],
                starting_subgoal_id=min(task["subgoal_ids"]),
                repeat_id=task["repeat_id"],
            )
            for task in eval_skill_info_list
        ]
    else:
        if config.specific_task is not None:
            eval_skill_info_list = eval_skill_info_list[
                config.specific_task : config.specific_task + 1
            ]
        # unique so we can select a specific env
        sorted_task_names = [
            count[0]
            for count in Counter(
                [skill_info["task"] for skill_info in eval_skill_info_list]
            ).most_common()
        ]

        eval_skill_info_list = [
            skill_info
            for skill_info in eval_skill_info_list
            if skill_info["task"] in sorted_task_names
        ]

    config.num_subgoals_in_pool = len(sorted_task_names)
    # step by step evaluation skill info list
    primitive_eval_skill_info_list = make_primitive_annotation_eval_dataset(
        eval_skill_info_list
    )
    print(
        f"Evaluating on {len(sorted_task_names)} tasks. Total {len(eval_skill_info_list)} skills"
    )

    primitive_skills_to_use = []
    task_lengths = None
    if config.eval_per_task_in_json:
        task_lengths = []
        for task in primitive_eval_skill_info_list:
            primitive_skills_to_use.append(
                generate_primitive_skill_list_from_eval_skill_info_list([task])
            )
            task_lengths.append(len(task["primitive_skills"]))
    else:
        primitive_skills_to_use = [
            generate_primitive_skill_list_from_eval_skill_info_list(
                primitive_eval_skill_info_list
            )
        ]
    if not config.use_only_task_primitive_skills:
        # aggregate all primitive skills if they have the same floorplan
        floorplan_per_task = []
        for task in eval_skill_info_list:
            floorplan_per_task.append(
                all_scenes_json[task["primitive_skills"][0]["scene_index"]]["scene_num"]
            )
        aggregated_tasks = {}
        for task, fp in zip(eval_skill_info_list, floorplan_per_task):
            if fp not in aggregated_tasks:
                aggregated_tasks[fp] = copy.deepcopy(task)
            else:
                aggregated_tasks[fp]["primitive_skills"].extend(
                    task["primitive_skills"]
                )
        # now separate by floorplan
        all_primitive_skills_per_floorplan = {
            fp: generate_primitive_skill_list_from_eval_skill_info_list(
                [aggregated_tasks[fp]]
            )
            for fp in aggregated_tasks
        }
        # now add the primitive skills for each task
        primitive_skills_to_use = []
        for fp in floorplan_per_task:
            primitive_skills_to_use.append(all_primitive_skills_per_floorplan[fp])
    print(
        f"Evaluating on {len(sorted_task_names)} tasks. Total {[len(primitive_skills_to_use[i]) for i in range(len(primitive_skills_to_use))]} skills"
    )
    processes = setup_mp(
        result_queue,
        task_queue,
        agent_model,
        saycan_planner,
        sentence_encoder,
        resnet,
        config,
        device,
        eval_skill_info_list,
        goal_states,
    )

    def signal_handler(sig, frame):
        print("SIGINT received. Exiting...closing all processes first")
        cleanup_mp(task_queue, processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    eval_metrics, _ = multiprocess_rollout(
        task_queue,
        result_queue,
        config,
        0,
        rollout_mode="fixed_eval",
        eval_skill_info_list=eval_skill_info_list,
    )
    wandb.log(
        eval_metrics,
        step=0,
    )
    cleanup_mp(task_queue, processes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Autonomous exploration of an ALFRED IQL offline RL model for new language annotated skills"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Parent directory containing the dataset",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="which gpus. pass in as comma separated string to use DataParallel on multiple GPUs",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for initialization"
    )
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=3,
        help="number of workers for policy rollouts",
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
        "--scene_type",
        type=str,
        default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen"],
        help="which type of scenes to sample from/evaluate on",
    )
    parser.add_argument(
        "--eval_json",
        type=str,
        default="scene_sampling/bootstrap_valid_seen-40_ann_human.json",
        help="path to the json file containing the evaluation scenes and skills",
    )
    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to use automatic mixed precision. set default to false to disable nans during online training.",
    )
    parser.add_argument(
        "--eval_per_task_in_json",
        type=str2bool,
        default=True,
        const=True,
        nargs="?",
        help="whether to evaluate each task in the json file separately.",
    )
    parser.add_argument(
        "--use_only_task_primitive_skills",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
        help="whether to use only the given eval primitive skills for chaining",
    )
    # LLM arguments
    parser.add_argument(
        "--llm_model",
        type=str,
        # default="decapoda-research/llama-13b-hf",
        default="decapoda-research/llama-7B-hf",
        help="which model to use for the large language model. ",
        choices=[
            "None",
            "decapoda-research/llama-13b-hf",
            "decapoda-research/llama-7B-hf",
        ],
    )
    parser.add_argument(
        "--llm_gpus",
        type=str,
        default="0",
        help="comma separated list of which gpus to use for the large language model",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=1,
        help="num concurrent queries for the LLM",
    )
    parser.add_argument(
        "--specific_task",
        type=int,
        default=None,
        help="if specified, only train on this subgoal index",
    )  # for reporting the finetuning exps
    config = parser.parse_args()
    config.gpus = [int(gpu) for gpu in config.gpus.strip().split(",")]
    config.llm_gpus = [int(gpu) for gpu in config.llm_gpus.strip().split(",")]
    config.use_pretrained_lang = False
    if config.experiment_name is None and config.run_group is not None:
        config.experiment_name = f"{config.run_group}_{config.seed}"
    mp.set_sharing_strategy(
        "file_system"
    )  # to ensure the too many open files error doesn't happen with the dataloader
    main(config)
