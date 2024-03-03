import os
import json
import pprint
from tqdm import tqdm
import numpy as np
import argparse
import pickle
from datasets.large_language_model import LargeLanguageModel
from sprint.utils.utils import process_skill_strings


long_tasks_path = "eval_length_data.json"
with open(long_tasks_path, "r") as f:
    eval_length_data = json.load(f)

split_path = "splits/oct21.json"
with open(split_path) as f:
    splits = json.load(f)
    pprint.pprint({k: len(v) for k, v in splits.items()})

data_type = "train"
data = splits[data_type]


problem_data_path = "problem_data.json"


def generate_language_combination(data):
    problem_data = []
    combination_skill = dict()
    combination_skill["name_subgoal"] = []
    combination_skill["lang_combination"] = []

    for trial in tqdm(data):
        task = trial["task"]
        repeat_id = trial["repeat_idx"]
        file_path = os.path.join(
            "data",
            "json_2.1.0_merge_goto",
            data_type,
            task,
            "augmented_traj_data_new.json",
        )
        with open(file_path, "r") as f:
            traj_data = json.load(f)
        if traj_data["plan"]["high_pddl"][-1]["discrete_action"]["action"] == "NoOp":
            traj_data["plan"]["high_pddl"].pop(-1)
        num_skill = len(traj_data["plan"]["high_pddl"])
        num_lang = len(traj_data["turk_annotations"]["anns"][repeat_id]["high_descs"])

        if num_skill == num_lang:
            for i in range(num_skill - 1):
                for j in range(i + 1, num_skill):

                    if i == 0 and j == (num_skill - 1) and task in eval_length_data:
                        continue
                    else:
                        language = traj_data["turk_annotations"]["anns"][repeat_id][
                            "high_descs"
                        ][i : j + 1]
                        # language = " ".join(language)

                        subgoal_ids = np.arange(i, j + 1)
                        subgoal_ids = subgoal_ids.tolist()
                        subgoal_ids = [str(i) for i in subgoal_ids]
                        subgoal_ids = "+".join(subgoal_ids)
                        name = (
                            task + "-ridx=" + str(repeat_id) + "-subgoal=" + subgoal_ids
                        )

                        combination_skill["name_subgoal"].append(name)
                        combination_skill["lang_combination"].append(
                            process_skill_strings(language)
                        )
    return combination_skill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_model",
        type=str,
        default="decapoda-resaerch/llama-13b-hf",
        help="which model to use for the large language model. For optimal performance, use GPT-J-6B or bigger. For speed and decent performance, opt-2.7b is fine.",
        choices=[
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "EleutherAI/gpt-j-6B",
            "facebook/opt-13b",
            "EleutherAI/gpt-neox-20b",
            "facebook/opt-30b",
            "facebook/opt-66b",
            "decapoda-research/llama-13b-hf",
            "decapoda-research/llama-7b-hf",
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
        default=100,
        help="batch size for the large language model",
    )
    parser.add_argument(
        "--llm_max_new_tokens",
        type=int,
        default=30,
        help="max number of new tokens to sample from the large language model",
    )
    config = parser.parse_args()
    print(config.llm_model.split("/")[1])
    config.llm_gpus = [int(gpu) for gpu in config.llm_gpus.split(",")]
    skill_combination = generate_language_combination(data)
    language_combination = skill_combination["lang_combination"]
    llm = LargeLanguageModel(config)
    summaries, logprobs = llm.get_summaries_and_logprobs(language_combination)

    skill_combination["summaries"] = summaries
    skill_combination["logprobs"] = logprobs

    with open(
        f"sam_{config.llm_model.split('/')[1]}_summarized_language_annotations.pkl",
        "wb",
    ) as handle:
        pickle.dump(skill_combination, handle, protocol=pickle.HIGHEST_PROTOCOL)


main()
