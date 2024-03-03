from sprint.alfred.gen.constants import VISIBILITY_DISTANCE
import os
import random
import torch
import pickle
import torch
import numpy as np
import sprint.alfred.gen.constants as constants
import random
import re
import argparse
import copy
import torch
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __deepcopy__(self, memo):
        return AttrDict(copy.deepcopy(dict(self), memo))


####### General definitions/constants for discrete actions in the environment #####
visibility_distance = constants.VISIBILITY_DISTANCE

interactive_actions = [
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
]
knives = ["ButterKnife", "Knife"]
##########


def load_goal_states(config):
    # loading goal states for actionable models as it is final image-conditioned
    goal_states = None
    if config.model == "am":
        if config.env_type == "eval_instruct":
            goal_states_path = "am_eval_instruct_goal_states.pkl"
        elif config.env_type == "eval_length":
            goal_states_path = "am_eval_length_goal_states.pkl"
        else:
            goal_states_path = "am_eval_unseen_goal_states.pkl"
        goal_states_path = f"{os.environ['SPRINT']}/sprint/rollouts/{goal_states_path}"
        with open(goal_states_path, "rb") as f:
            goal_states = pickle.load(f)
    return goal_states


def send_to_device_if_not_none(data_dict, entry_name, device):
    # helper function to send torch tensor to device if it is not None
    if entry_name not in data_dict or data_dict[entry_name] is None:
        return None
    else:
        return data_dict[entry_name].to(device)


def load_object_class(vocab_obj, object_name):
    """
    load object classes for interactive actions
    """
    if object_name is None:
        return 0
    object_class = object_name.split("|")[0]
    return vocab_obj.word2index(object_class)


def extract_item(possible_tensor):
    # if it is a tensor then extract the item otherwise just return it
    if isinstance(possible_tensor, torch.Tensor):
        return possible_tensor.item()
    return possible_tensor


def str2bool(v):
    # used for parsing boolean arguments
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def generate_invalid_action_mask_and_objects(env, visible_object, vocab_obj, vocab):
    # we will first filter out all the interact actions that are not available
    def filter_objects(condition):
        condition_input = condition
        if condition == "toggleon" or condition == "toggleoff":
            condition = "toggleable"

        if condition == "openable" or condition == "closeable":
            condition = "openable"

        visible_candidate_objects = [
            obj for obj in visible_object if obj[condition] == True
        ]

        candidate_obj_type = [
            vis_obj["objectId"].split("|")[0] for vis_obj in visible_candidate_objects
        ]

        remove_indicies = []

        if condition_input == "toggleon":
            if "Faucet" in candidate_obj_type:
                # SinkBasin: Sink|+03.08|+00.89|+00.09|SinkBasin
                visible_object_name = [
                    obj["objectId"].split("|")[-1] for obj in visible_object
                ]
                if "SinkBasin" not in visible_object_name:
                    remove_indicies.append(candidate_obj_type.index("Faucet"))

            for i, obj in enumerate(visible_candidate_objects):
                if (
                    obj["isToggled"] == True
                    and obj["objectId"].split("|")[0] in candidate_obj_type
                ):
                    remove_indicies.append(i)

        elif condition_input == "toggleoff":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isToggled"] == False:
                    remove_indicies.append(i)

        elif condition_input == "openable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isOpen"] == True or obj["isToggled"] == True:
                    remove_indicies.append(i)

        elif condition_input == "closeable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isOpen"] == False:
                    remove_indicies.append(i)

        elif condition_input == "receptacle":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["openable"] == True and obj["isOpen"] == False:
                    remove_indicies.append(i)

        elif condition_input == "sliceable":
            for i, obj in enumerate(visible_candidate_objects):
                if obj["isSliced"] == True:
                    remove_indicies.append(i)

        remove_indicies = set(remove_indicies)
        filtered_candidate_obj_type = [
            j for i, j in enumerate(candidate_obj_type) if i not in remove_indicies
        ]
        filtered_visible_candidate_objects = [
            j
            for i, j in enumerate(visible_candidate_objects)
            if i not in remove_indicies
        ]

        candidate_obj_type_id = [
            vocab_obj.word2index(candidate_obj_type_use)
            for candidate_obj_type_use in filtered_candidate_obj_type
            if candidate_obj_type_use in vocab_obj.to_dict()["index2word"]
        ]
        candidate_obj_type_id = np.array(list(set(candidate_obj_type_id)))
        return filtered_visible_candidate_objects, candidate_obj_type_id

    pickupable_object_names, pickupable_objects = filter_objects("pickupable")
    openable_object_names, openable_objects = filter_objects("openable")
    sliceable_object_names, sliceable_objects = filter_objects("sliceable")
    closeable_object_names, closeable_objects = filter_objects("closeable")
    receptacle_object_names, receptacle_objects = filter_objects("receptacle")

    toggleon_object_names, toggleon_objects = filter_objects("toggleon")
    toggleoff_object_names, toggleoff_objects = filter_objects("toggleoff")

    # generate invalid mask
    invalid_action_mask = []
    if (
        len(pickupable_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) > 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("PickupObject") - 2)
    if len(openable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("OpenObject") - 2)
    if (
        len(sliceable_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 2)
    if len(closeable_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("CloseObject") - 2)
    if (
        len(receptacle_objects) == 0
        or len(env.last_event.metadata["inventoryObjects"]) == 0
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("PutObject") - 2)
    if len(toggleon_objects) == 0:
        invalid_action_mask.append(vocab["action_low"].word2index("ToggleObjectOn") - 2)
    if len(toggleoff_objects) == 0:
        invalid_action_mask.append(
            vocab["action_low"].word2index("ToggleObjectOff") - 2
        )
    if (
        len(env.last_event.metadata["inventoryObjects"]) > 0
        and env.last_event.metadata["inventoryObjects"][0]["objectId"].split("|")[0]
        not in knives
    ):
        invalid_action_mask.append(vocab["action_low"].word2index("SliceObject") - 2)

    # <<stop>> action needs to be invalid
    invalid_action_mask.append(vocab["action_low"].word2index("<<stop>>") - 2)
    invalid_action_mask = list(set(invalid_action_mask))

    ret_dict = dict(
        pickupable_object_names=pickupable_object_names,
        pickupable_objects=pickupable_objects,
        openable_object_names=openable_object_names,
        openable_objects=openable_objects,
        sliceable_object_names=sliceable_object_names,
        sliceable_objects=sliceable_objects,
        closeable_object_names=closeable_object_names,
        closeable_objects=closeable_objects,
        receptacle_object_names=receptacle_object_names,
        receptacle_objects=receptacle_objects,
        toggleon_object_names=toggleon_object_names,
        toggleon_objects=toggleon_objects,
        toggleoff_object_names=toggleoff_object_names,
        toggleoff_objects=toggleoff_objects,
    )
    return invalid_action_mask, ret_dict


def process_skill_strings(strings):
    # process strings to all be proper sentences with punctuation and capitalization.
    if not isinstance(strings, list):
        strings = [strings]
    processed_strings = []
    for string in strings:
        if isinstance(string, list):
            # artifact of bug in the data
            string = string[0]
        string = string.strip().lower()
        string = re.sub(" +", " ", string)  # remove extra spaces
        if len(string) > 0 and string[-1] not in ["?", ".", "!"]:
            string = string + "."
        processed_strings.append(string.capitalize())
    return processed_strings


def compute_distance(agent_position, object):
    # computes xyz distance to an object
    agent_location = np.array(
        [agent_position["x"], agent_position["y"], agent_position["z"]]
    )
    object_location = np.array(
        [object["position"]["x"], object["position"]["y"], object["position"]["z"]]
    )

    distance = np.linalg.norm(agent_location - object_location)

    return distance


def compute_visibility_based_on_distance(agent_position, object, visibility_distance):
    # directly rewritten from C++ code here https://github.com/allenai/ai2thor/blob/f39ae981646d689047ba7006cb9c1dc507a247ff/unity/Assets/Scripts/BaseFPSAgentController.cs#L2628
    # used to figure out if an object is visible
    is_visible = True
    x_delta = object["position"]["x"] - agent_position["x"]
    y_delta = object["position"]["y"] - agent_position["y"]
    z_delta = object["position"]["z"] - agent_position["z"]
    if abs(x_delta) > visibility_distance:
        is_visible = False
    elif abs(y_delta) > visibility_distance:
        is_visible = False
    elif abs(z_delta) > visibility_distance:
        is_visible = False
    elif (
        x_delta * x_delta + z_delta * z_delta
        > visibility_distance * visibility_distance
    ):
        is_visible = False
    return is_visible


def mask_and_resample(action_probs, action_mask, deterministic, take_rand_action):
    # mask the action probabilities with the action mask (don't allow invalid actions)
    # then use those probabilities to produce an action
    action_probs[0, action_mask] = 0
    if torch.all(action_probs[0] == 0):
        # set the indicies NOT in action mask to 0
        action_mask_complement = np.ones(action_probs.shape[1], dtype=bool)
        action_mask_complement[action_mask] = False
        action_probs[0, action_mask_complement] = 1
    logprobs = torch.log(action_probs)
    logprobs[0, action_mask] = -100
    if deterministic:
        chosen_action = torch.argmax(action_probs)
    else:
        dist = torch.distributions.Categorical(logits=logprobs)
        chosen_action = dist.sample()
    if take_rand_action:
        action_mask_complement = np.ones(action_probs.shape[1], dtype=bool)
        # anything that doesn't get masked out by action_mask is in action_mask_complement
        action_mask_complement[action_mask] = False
        # set uniform probability for all valid actions
        # logprobs[0, action_mask_complement] = 0
        action_probs[0, action_mask_complement] = 1
        # sample uniformly
        dist = torch.distributions.Categorical(action_probs)
        chosen_action = dist.sample()
    return chosen_action


def get_action_from_agent(
    model,
    feat,
    vocab,
    vocab_obj,
    env,
    deterministic,
    epsilon,
    ret_value,
):
    take_rand_action = random.random() < epsilon

    action_out, object_pred_id, value = model.step(feat, ret_value=ret_value)

    action_out = torch.softmax(action_out, dim=1)

    object_pred_prob = torch.softmax(object_pred_id, dim=1)

    agent_position = env.last_event.metadata["agent"]["position"]

    visible_object = [
        obj
        for obj in env.last_event.metadata["objects"]
        if (
            obj["visible"] == True
            and compute_visibility_based_on_distance(
                agent_position, obj, VISIBILITY_DISTANCE
            )
        )
    ]
    invalid_action_mask, ret_dict = generate_invalid_action_mask_and_objects(
        env, visible_object, vocab_obj, vocab
    )
    # choose the action after filtering with the mask
    chosen_action = mask_and_resample(
        action_out, invalid_action_mask, deterministic, take_rand_action
    )
    string_act = vocab["action_low"].index2word(chosen_action + 2)
    assert string_act != "<<stop>>", breakpoint()
    if string_act not in interactive_actions:
        return string_act, None, value
    object_pred_prob = object_pred_prob.squeeze(0).cpu().detach().numpy()
    # otherwise, we need to choose the closest visible object for our action
    string_act_to_object_list_map = dict(
        PickupObject=(
            ret_dict["pickupable_object_names"],
            ret_dict["pickupable_objects"],
        ),
        OpenObject=(ret_dict["openable_object_names"], ret_dict["openable_objects"]),
        SliceObject=(ret_dict["sliceable_object_names"], ret_dict["sliceable_objects"]),
        CloseObject=(ret_dict["closeable_object_names"], ret_dict["closeable_objects"]),
        PutObject=(ret_dict["receptacle_object_names"], ret_dict["receptacle_objects"]),
        ToggleObjectOn=(
            ret_dict["toggleon_object_names"],
            ret_dict["toggleon_objects"],
        ),
        ToggleObjectOff=(
            ret_dict["toggleoff_object_names"],
            ret_dict["toggleoff_objects"],
        ),
    )

    candidate_object_names, candidate_object_ids = string_act_to_object_list_map[
        string_act
    ]
    prob_dict = {}
    for id in candidate_object_ids:
        if take_rand_action:
            prob_dict[id] = 1
        else:
            prob_dict[id] = object_pred_prob[id]
    prob_value = prob_dict.values()
    if deterministic:
        max_prob = max(prob_value)
        choose_id = [k for k, v in prob_dict.items() if v == max_prob][0]
    else:
        # sample from the object prob distribution
        object_probs = torch.tensor(list(prob_value), dtype=torch.float32)
        if torch.all(object_probs == 0):
            object_probs = torch.ones_like(object_probs)
        choose_id = torch.multinomial(object_probs, 1)[0].item()
        choose_id = list(prob_dict.keys())[choose_id]

    # choose the closest object
    object_type = vocab_obj.index2word(choose_id)
    candidate_objects = [
        obj
        for obj in candidate_object_names
        if obj["objectId"].split("|")[0] == object_type
    ]
    # object type
    agent_position = env.last_event.metadata["agent"]["position"]
    min_distance = float("inf")
    for ava_object in candidate_objects:
        obj_agent_dist = compute_distance(agent_position, ava_object)
        if obj_agent_dist < min_distance:
            min_distance = obj_agent_dist
            output_object = ava_object["objectId"]
    return string_act, output_object, value


def cleanup_mp(task_queue, processes):
    # generate termination signal for each worker
    for _ in range(len(processes)):
        task_queue.put(None)

    # wait for workers to terminate
    for worker in processes:
        worker.join()


def make_primitive_annotation_eval_dataset(eval_list: list[dict]):
    """
    Make a dataset for evaluation of primitive annotations, used for SayCan
    """
    new_eval_dataset = []
    for eval_dict in eval_list:
        eval_dict_copy = eval_dict.copy()
        annotations = []
        for primitive_skill in eval_dict_copy["primitive_skills"]:
            annotations.append(primitive_skill["annotations"])
        annotations = process_skill_strings(annotations)
        eval_dict_copy["annotation"] = " ".join(annotations)
        new_eval_dataset.append(eval_dict_copy)
    return new_eval_dataset


def generate_primitive_skill_list_from_eval_skill_info_list(
    primitive_eval_skill_info_list,
):
    primitive_skills_to_use = []
    for skill_info in primitive_eval_skill_info_list:
        primitive_skills_to_use.extend(
            [primitive_skill for primitive_skill in skill_info["primitive_skills"]]
        )
    for primitive_skill in primitive_skills_to_use:
        primitive_skill["api_actions"] = primitive_skill[
            "api_action"
        ]  # relabeling since online_reward.py expects api_actions

    def tuplify_dict_of_dicts(d):
        to_tuplify = []
        for k in sorted(d):
            if isinstance(d[k], dict):
                to_tuplify.append((k, tuplify_dict_of_dicts(d[k])))
            elif isinstance(d[k], list):
                inner_tuplify = []
                for item in d[k]:
                    if isinstance(item, list):
                        inner_tuplify.append(tuple(item))
                    else:
                        inner_tuplify.append(item)
                to_tuplify.append(tuple(inner_tuplify))
            else:
                to_tuplify.append((k, d[k]))
        return tuple(to_tuplify)

    # now remove duplicate primitive skills which is a list of dicts of inner dicts
    primitive_skill_set = set()
    unique_primitive_skills_to_use = []
    for primitive_skill in primitive_skills_to_use:
        if tuplify_dict_of_dicts(primitive_skill) not in primitive_skill_set:
            primitive_skill_set.add(tuplify_dict_of_dicts(primitive_skill))
            unique_primitive_skills_to_use.append(primitive_skill)
    return unique_primitive_skills_to_use
