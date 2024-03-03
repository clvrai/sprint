import os
import json
import pprint
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sprint.datasets.pp_loader_old import iterater
import argparse
import pyxis as px
import pickle
import copy


def process_skill_strings(strings):
    processed_strings = []
    for string in strings:
        string = string.strip().lower()
        if string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings


###### use for evaluation ######
eval_length_data_path = "eval_length_data.json"
with open(eval_length_data_path, "r") as f:
    eval_length_data = json.load(f)


incorrect_sizes = []


def process_stuff(dataset, config):
    # language_model = SentenceTransformer(
    #    "sentence-transformers/all-mpnet-base-v2", device=config.device
    # )
    batch_size = 10
    # dataset: {"repeat_idx ,"task"} * 21k
    # Namespace(data_path='./data/json_2.1.0_merge_goto', save_path='./data/json_2.1.0_merge_goto', device='cuda:0', dataset_type='train')
    dataset = [trial for trial in dataset if trial["repeat_idx"] == 0]

    if config.dataset_type == "train":
        summary_path = "sam_opt_summarized_next_skill_annotation.pkl"
    else:
        summary_path = (
            "sam_opt_summarized_next_skill_annotation" + config.dataset_type + ".pkl"
        )
    with open(summary_path, "rb") as f:
        language_summary = pickle.load(f)

    loader = iterater(
        dataset,
        batch_size,
        config.data_path,
        set_type=config.dataset_type,
        load_mask=False,
    )
    with px.Writer(
        dirpath=os.path.join(
            config.data_path,
            (
                f"px_llama_13b_{config.dataset_type}/"
                if "valid" in config.dataset_type
                else "px_llama_13b"
            ),
        ),
        map_size_limit=400000,
        ram_gb_limit=10,
    ) as db:

        for batch, feat in loader:

            skill_dataset = []
            for batch_index in range(len(batch)):

                # get how many annotations
                num_annotations = len(batch[batch_index]["turk_annotations"]["anns"])
                trial_name = "/".join(feat["root"][batch_index].split("/")[-2:])
                num_skill = len(
                    batch[batch_index]["turk_annotations"]["anns"][0]["high_descs"]
                )
                num_ann = len(batch[batch_index]["plan"]["high_pddl"])

                if num_skill == num_ann:  # check num of skill equal num of annotation

                    skill_to_save = {}
                    # find same trial name with all repeat id and subgoal combination
                    skill_num = num_skill
                    same_trial_combinations_index = [
                        language_summary["name_subgoal"].index(traj_name)
                        for traj_name in language_summary["name_subgoal"]
                        if "-".join(traj_name.split("-")[0:-2]) == trial_name
                    ]  # same trial with different repeat id and subgoal combination
                    action_switch = feat["action_switch_point"][batch_index]
                    action_frame_index = copy.deepcopy(action_switch)
                    action_frame_index.append(
                        feat["resnet_features"][batch_index].shape[0] - 1
                    )  # we need to save the last frame as well (we need goal state)
                    traj_resnet_feature = feat["resnet_features"][batch_index][
                        action_frame_index
                    ].numpy()  # state number is 1 more than action number

                    # import pdb ; pdb.set_trace()
                    # get all skill combinations
                    lang_combinations = [
                        language_summary["lang_combination"][i]
                        for i in same_trial_combinations_index
                    ]
                    lang_summaries = [
                        language_summary["summaries"][i]
                        for i in same_trial_combinations_index
                    ]
                    lang_logprobs = [
                        language_summary["logprobs"][i]
                        for i in same_trial_combinations_index
                    ]
                    lang_ridx = [
                        int(language_summary["name_subgoal"][i].split("=")[-2][0])
                        for i in same_trial_combinations_index
                    ]
                    lang_subgoals = [
                        (language_summary["name_subgoal"][i].split("=")[-1])
                        for i in same_trial_combinations_index
                    ]

                    # next_skill_latent
                    lang_next_skill_latent = [
                        language_summary["encoded_next_skills"][i]
                        for i in same_trial_combinations_index
                    ]

                    # import pdb ; pdb.set_trace()
                    # add primitive skills to dataset

                    # num_of_repeat = len(batch[batch_index]["turk_annotations"]["anns"] )
                    # for repeat_index in range(num_of_repeat):
                    #     for subgoal_index in range(num_skill):
                    #         primitive_ann = batch[batch_index]["turk_annotations"]["anns"][0]['high_descs'][subgoal_index]
                    #         lang_combinations.append([primitive_ann])
                    #         lang_summaries.append(primitive_ann)
                    #         lang_logprobs.append(torch.tensor(0))
                    #         lang_ridx.append(repeat_index)
                    #         lang_subgoals.append(str(subgoal_index))

                    lang_subgoals = [subgoal.split("+") for subgoal in lang_subgoals]

                    for p in range(len(lang_subgoals)):
                        combine = lang_subgoals[p]
                        combine = [int(i) for i in combine]
                        lang_subgoals[p] = combine
                    # lang_embeddings = language_model.encode(process_skill_strings(lang_instruction))
                    # load other stuff
                    skill_switch_point = feat["skill_switch_point"][batch_index]
                    action_low = feat["action_low"][batch_index]
                    path = feat["root"][batch_index]
                    action_low_valid_interact = feat["action_low_valid_interact"][
                        batch_index
                    ]
                    traj_object_ids = feat["object_ids"][batch_index]
                    rewards = feat["rewards"][batch_index]

                    lang_low_action = []
                    lang_valid_interact = []
                    lang_object_ids = []
                    lang_rewards = []
                    lang_encode_cat_language = []
                    lang_encode_summary = []

                    # process all languages
                    for combination_id in range(len(lang_combinations)):
                        combination = lang_combinations[combination_id]
                        language = " ".join(combination)
                        lang_combinations[combination_id] = language
                        lang_encode_cat_language.append(language)

                    lang_summaries = [
                        summary[0] if (type(summary) == list) else summary
                        for summary in lang_summaries
                    ]
                    # lang_encode_cat_language = language_model.encode(
                    #    process_skill_strings(lang_encode_cat_language)
                    # )
                    # lang_encode_summary = language_model.encode(
                    #    process_skill_strings(lang_summaries)
                    # )

                    # process actions and objects
                    # import pdb ; pdb.set_trace()
                    # import pdb ; pdb.set_trace()
                    for subgoal_pair_id in range(len(lang_subgoals)):
                        skills = lang_subgoals[subgoal_pair_id]
                        skill_action_start = skill_switch_point[skills[0]]
                        if skills[-1] + 1 == skill_num:
                            # skills.append(skills[-1] + 1)

                            skill_action = action_low[skill_action_start:]
                            skill_interact = action_low_valid_interact[
                                skill_action_start:
                            ]
                            object_ids = traj_object_ids[skill_action_start:]
                            skill_rews = rewards[skill_action_start:]

                        else:
                            skill_action = action_low[
                                skill_action_start : skill_switch_point[skills[-1] + 1]
                            ]
                            # if len(skills) == 1:
                            #     print(subgoal_pair_id, "skill_action", skill_action)
                            skill_interact = action_low_valid_interact[
                                skill_action_start : skill_switch_point[skills[-1] + 1]
                            ]
                            object_ids = traj_object_ids[
                                skill_action_start : skill_switch_point[skills[-1] + 1]
                            ]
                            skill_rews = rewards[
                                skill_action_start : skill_switch_point[skills[-1] + 1]
                            ]
                        skill_action = np.array(skill_action)
                        # change all actions that are after "end_action" to their label minus 1
                        skill_action[skill_action > 2] = (
                            skill_action[skill_action > 2] - 1
                        )
                        lang_low_action.append(skill_action)
                        lang_valid_interact.append(skill_interact)
                        lang_object_ids.append(object_ids)
                        lang_rewards.append(skill_rews)

                    # a traj data
                    # traj_resnet_feature (num_action + 1, 512, 7, 7))
                    # lang_encode_cat_language (list, length is number of combination) each element is (768)
                    # lang_encode_summary (list, length is number of combination) each element is (768)
                    # lang_low_action (list, length is number of combination) each element np.array action
                    # lang_valid_interact (list, length is number of combination) each element np.array (0, 1)
                    # lang_object_ids (list, length is number of combination) each element object name
                    # lang_rewards (list, length is number of combination)
                    lang_logprobs = np.array([prob.numpy() for prob in lang_logprobs])
                    lang_next_skill_latent = np.array(lang_next_skill_latent)
                    # lang_next_skill_latent = np.array([ latent.cpu().numpy() for latent in lang_next_skill_latent])

                    for i in range(len(lang_subgoals)):
                        subgoals = lang_subgoals[i]
                        subgoals = [str(i) for i in subgoals]
                        lang_subgoals[i] = "+".join(subgoals)

                    for i in range(len(lang_low_action)):
                        low_actions = lang_low_action[i]
                        low_actions = [str(i) for i in low_actions]
                        lang_low_action[i] = "+".join(low_actions)

                    for i in range(len(lang_valid_interact)):
                        low_interact = lang_valid_interact[i]
                        low_interact = [str(i) for i in low_interact]
                        lang_valid_interact[i] = "+".join(low_interact)

                    skill_to_save["next_skill_latent"] = lang_next_skill_latent
                    skill_to_save["traj_resnet_feature"] = traj_resnet_feature
                    skill_to_save["lang_encode_cat_language"] = lang_encode_cat_language
                    skill_to_save["lang_encode_summary"] = lang_encode_summary
                    skill_to_save["lang_low_action"] = lang_low_action
                    skill_to_save["lang_valid_interact"] = lang_valid_interact
                    skill_to_save["lang_object_ids"] = lang_object_ids
                    skill_to_save["lang_combinations"] = lang_combinations
                    skill_to_save["lang_summaries"] = lang_summaries
                    skill_to_save["lang_subgoals"] = lang_subgoals
                    skill_to_save["lang_logprobs"] = lang_logprobs
                    skill_to_save["lang_ridx"] = np.array(lang_ridx)
                    skill_to_save["skill_switch_point"] = skill_switch_point

                skill_dataset.append(skill_to_save)

            skill_dataset_dict = {}
            for skill in skill_dataset:
                for key, value in skill.items():
                    if key not in skill_dataset_dict:
                        skill_dataset_dict[key] = []
                    skill_dataset_dict[key].append(value)
            np_array_version = []
            for key, value in skill_dataset_dict.items():
                if key in ["lang_object_ids"]:
                    new_value = []
                    for i, traj in enumerate(value):
                        serialized = pickle.dumps(traj)
                        new_value.append(serialized)
                    value = new_value
                    np_array_version.append(key)
                    np_array_version.append(np.asarray(value, dtype="S"))
                # elif key in ["lang_combinations", "lang_summaries", "lang_subgoals", "lang_low_action", "lang_valid_interact"]:
                #     np_array_version.append(key)
                #     np_array_version.append(np.asarray(value, dtype="S"))

                else:
                    np_array_version.append(key)
                    np_array_version.append(np.asarray(value))
                # print(*np_array_version)
            db.put_samples(*np_array_version)

        print(incorrect_sizes)
        incorrect_size_path_set = set([tup[0] for tup in incorrect_sizes])
        print(f"{len(incorrect_size_path_set)} Incorrect sizes")
        print(f"{incorrect_size_path_set}")


def print_db_length(config):
    with px.Reader(
        os.path.join(
            # config.save_path, f"px_alfred_data_rgb_segmentation_{config.dataset_type}"
            config.data_path,
            f"px_llama_13b{'_' + config.dataset_type if 'valid' in config.dataset_type else ''}",
        )
    ) as db:
        print(
            f"Length of the {config.dataset_type} DB is {len(db)}, saved at {config.data_path}"
        )
        print(db[-1])
        print(db[-1]["next_skill_latent"].shape)
        print(db[-1]["lang_logprobs"].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/json_2.1.0_merge_goto")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["train", "valid_seen", "valid_unseen"],
        default="train",
    )
    config = parser.parse_args()

    train_data_path = os.path.join(config.data_path, "preprocess/", config.dataset_type)
    split_path = "splits/oct21.json"
    pp_folder = "pp"
    vocab = torch.load(os.path.join(config.data_path, "pp.vocab"))
    with open(split_path) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})
    train = splits["train"]
    valid_seen = splits["valid_seen"]
    valid_unseen = splits["valid_unseen"]

    dataset_type_to_data = dict(
        train=train,
        valid_seen=valid_seen,
        valid_unseen=valid_unseen,
    )

    process_stuff(dataset_type_to_data[config.dataset_type], config)
    print_db_length(config)
