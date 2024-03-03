import json
import pprint
import os
import progressbar
import torch
from vocab import Vocab
import copy
import revtok
from tqdm import tqdm
import random
import argparse
from sprint.utils.data_utils import remove_spaces_and_lower


# data_path = "json_2.1.0_merge_goto"


# from sprint.alfred.gen.utils.py_util import remove_spaces_and_lower
# from sprint.alfred.gen.utils.game_util import sample_templated_task_desc_from_traj_data


# preprocess and save


def sample_templated_task_desc_from_traj_data(traj_data):
    pddl_params = traj_data["pddl_params"]
    goal_str = traj_data["task_type"]
    if pddl_params["object_sliced"]:
        goal_str += "_slice"
    template = random.choice(glib.gdict[goal_str]["templates"])
    obj = pddl_params["object_target"].lower()
    recep = pddl_params["parent_target"].lower()
    toggle = pddl_params["toggle_target"].lower()
    mrecep = pddl_params["mrecep_target"].lower()
    filled_in_str = template.format(obj=obj, recep=recep, toggle=toggle, mrecep=mrecep)
    return filled_in_str


def fix_missing_high_pddl_end_action(ex):
    """
    appends a terminal action to a sequence of high-level actions
    """
    if ex["plan"]["high_pddl"][-1]["planner_action"]["action"] != "End":
        ex["plan"]["high_pddl"].append(
            {
                "discrete_action": {"action": "NoOp", "args": []},
                "planner_action": {"value": 1, "action": "End"},
                "high_idx": len(ex["plan"]["high_pddl"]),
            }
        )
    return ex


def remove_extra_high_pddl_end_action(ex):
    """
    appends a terminal action to a sequence of high-level actions
    """
    if ex["plan"]["high_pddl"][-1]["planner_action"]["action"] == "End":
        ex["plan"]["high_pddl"].pop(-1)
    return ex


def has_interaction(action):
    """
    check if low-level action is interactive
    """
    non_interact_actions = [
        "MoveAhead",
        "Rotate",
        "Look",
        "<<stop>>",
        "<<pad>>",
        "<<seg>>",
    ]
    if any(a in action for a in non_interact_actions):
        return False
    else:
        return True


class Dataset(object):
    def __init__(self, args):
        self.vocab = {
            "action_low": Vocab(["<<pad>>", "<<seg>>", "<<stop>>"]),
            "action_high": Vocab(["<<pad>>", "<<seg>>", "<<stop>>"]),
        }
        self.pframe = 300
        self.args = args
        self.data_path = args.json_dir
        self.preprocessed_folder = args.preprocessed_dir
        if not os.path.isdir(self.preprocessed_folder):
            os.makedirs(self.preprocessed_folder)
        split_path = "splits/oct21.json"

        with open(split_path) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

    def preprocess_splits(self):
        """
        saves preprocessed data as jsons in specified folder
        """
        # splits = json.load(open(self.split_path))
        for k, d in self.splits.items():
            # if k == "train":
            if k in ["train", "valid_seen"]:
                d = d[:]
                for task in progressbar.progressbar(d):
                    #                     print(task['task'])
                    # TODO: this used to be augmented_traj_data.json
                    json_path = os.path.join(
                        self.data_path, k, task["task"], "augmented_traj_data_new.json"
                    )
                    if not os.path.exists(json_path):
                        continue
                    # json_path = os.path.join(
                    #     self.data_path, k, task["task"], "augmented_traj_data_new.json"
                    # )
                    self.task = task
                    with open(json_path) as f:
                        ex = json.load(f)
                    r_idx = task["repeat_idx"]
                    traj = ex.copy()
                    traj["root"] = os.path.join(self.data_path, task["task"])
                    traj["split"] = k
                    traj["repeat_idx"] = r_idx
                    #                     traj['num'] = {}
                    self.process_language(ex, traj, r_idx)
                    self.process_actions(ex, traj)

                    #                     print(traj['images'])

                    preprocessed_folder = os.path.join(
                        self.preprocessed_folder, k, task["task"]
                    )
                    if not os.path.isdir(preprocessed_folder):
                        os.makedirs(preprocessed_folder)

                    # save preprocessed json
                    preprocessed_json_path = os.path.join(
                        preprocessed_folder, "ann_%d.json" % r_idx
                    )
                    with open(preprocessed_json_path, "w") as f_new:
                        json.dump(traj, f_new, sort_keys=True, indent=4)
        # save vocab in dout path
        #         vocab_dout_path = os.path.join(self.args.dout, '%s.vocab' % self.args.pp_folder)
        #         torch.save(self.vocab, vocab_dout_path)

        # save vocab in data path
        vocab_data_path = os.path.join(self.preprocessed_folder, "pp.vocab")
        torch.save(self.vocab, vocab_data_path)

    def process_language(self, ex, traj, r_idx, use_templated_goals=False):
        # goal instruction
        if use_templated_goals:
            task_desc = sample_templated_task_desc_from_traj_data(traj)
        else:
            task_desc = ex["turk_annotations"]["anns"][r_idx]["task_desc"]

        # step-by-step instructions
        high_descs = ex["turk_annotations"]["anns"][r_idx]["high_descs"]

        traj["ann"] = {
            "instr": [revtok.tokenize(remove_spaces_and_lower(x)) for x in high_descs]
            + [["<<stop>>"]],
            "repeat_idx": r_idx,
        }

        #         print(len(traj['ann']['instr']))
        #         print(type(high_descs[0]))

        #         for x in high_descs:
        #             print(x)
        #         print([x ])

        #         traj['num'] = {}
        #         traj['num']['lang_instr'] = [self.numericalize(self.vocab['word'], x, train=True) for x in traj['ann']['instr']]

        # numericalize language only store with sentences
        traj["num"] = {}
        traj["num"]["lang_instr"] = [x for x in high_descs]  # + ["stop task"]

    #         print(traj['num']['lang_instr'])

    @staticmethod
    def numericalize(vocab, words, train=True):
        """
        converts words to unique integers
        """
        return vocab.word2index([w.strip().lower() for w in words], train=train)

    def merge_last_two_low_actions(self, conv):
        """
        combines the last two action sequences into one sequence
        """
        extra_seg = copy.deepcopy(conv["num"]["action_low"][-2])
        for sub in extra_seg:
            sub["high_idx"] = conv["num"]["action_low"][-3][0]["high_idx"]
            conv["num"]["action_low"][-3].append(sub)
        del conv["num"]["action_low"][-2]
        conv["num"]["action_low"][-1][0]["high_idx"] = (
            len(conv["plan"]["high_pddl"]) - 1
        )

    def process_actions(self, ex, traj):
        # fix_missing_high_pddl_end_action(ex)
        remove_extra_high_pddl_end_action(ex)

        end_action = {
            "api_action": {"action": "NoOp"},
            "discrete_action": {"action": "<<stop>>", "args": {}},
            "high_idx": ex["plan"]["high_pddl"][-1]["high_idx"],
        }
        # init action_low and action_high
        num_hl_actions = len(ex["plan"]["high_pddl"])
        traj["num"]["action_low"] = [
            list() for _ in range(num_hl_actions)
        ]  # temporally aligned with HL actions
        traj["num"]["action_high"] = []
        low_to_high_idx = []

        # for a in (ex['plan']['low_actions'] + [end_action]):
        for a in ex["plan"]["low_actions"]:
            high_idx = a["high_idx"]
            low_to_high_idx.append(high_idx)

            #         print(len(low_to_high_idx),a)
            # low-level action (API commands)
            traj["num"]["action_low"][high_idx].append(
                {
                    "high_idx": a["high_idx"],
                    "action": self.vocab["action_low"].word2index(
                        a["discrete_action"]["action"], train=True
                    ),
                    "action_high_args": a["discrete_action"]["args"],
                }
            )
            # low-level bounding box (not used in the model)
            if "bbox" in a["discrete_action"]["args"]:
                xmin, ymin, xmax, ymax = [
                    float(x) if x != "NULL" else -1
                    for x in a["discrete_action"]["args"]["bbox"]
                ]
                traj["num"]["action_low"][high_idx][-1]["centroid"] = [
                    (xmin + (xmax - xmin) / 2) / self.pframe,
                    (ymin + (ymax - ymin) / 2) / self.pframe,
                ]
            else:
                traj["num"]["action_low"][high_idx][-1]["centroid"] = [-1, -1]

            # low-level interaction mask (Note: this mask needs to be decompressed)
            if "mask" in a["discrete_action"]["args"]:
                mask = a["discrete_action"]["args"]["mask"]
                # also save the object ID
                if "receptacleObjectId" in a["api_action"]:
                    object_id = a["api_action"]["receptacleObjectId"]
                else:
                    object_id = a["api_action"]["objectId"]
            else:
                mask = None
                object_id = None
            traj["num"]["action_low"][high_idx][-1]["mask"] = mask
            traj["num"]["action_low"][high_idx][-1]["object_id"] = object_id

            # interaction validity

            valid_interact = 1 if has_interaction(a["discrete_action"]["action"]) else 0
            traj["num"]["action_low"][high_idx][-1]["valid_interact"] = valid_interact
        # low to high idx
        traj["num"]["low_to_high_idx"] = low_to_high_idx
        # high-level actions
        for a in ex["plan"]["high_pddl"]:
            traj["num"]["action_high"].append(
                {
                    "high_idx": a["high_idx"],
                    "action": self.vocab["action_high"].word2index(
                        a["discrete_action"]["action"], train=True
                    ),
                    "action_high_args": self.numericalize(
                        self.vocab["action_high"], a["discrete_action"]["args"]
                    ),
                }
            )

        # check alignment between step-by-step language and action sequence segments
        action_low_seg_len = len(traj["num"]["action_low"])

        lang_instr_seg_len = len(traj["num"]["lang_instr"])

        seg_len_diff = action_low_seg_len - lang_instr_seg_len
        if seg_len_diff != 0:
            # if seg_len_diff == 1:
            #     import pdb ; pdb.set_trace()
            # if seg_len_diff != 1:
            #     import pdb ; pdb.set_trace()
            # assert seg_len_diff == 1  # sometimes the alignment is off by one  ¯\_(ツ)_/¯
            self.merge_last_two_low_actions(traj)

            # fix last two for low_to_high_idx and action_high from merge (from: https://github.com/askforalfred/alfred/issues/84)
            traj["num"]["low_to_high_idx"][-1] = traj["num"]["action_low"][-1][0][
                "high_idx"
            ]
            traj["num"]["low_to_high_idx"][-2] = traj["num"]["action_low"][-2][0][
                "high_idx"
            ]
            traj["num"]["action_high"][-1]["high_idx"] = traj["num"]["action_high"][-2][
                "high_idx"
            ]
            traj["num"]["action_high"][-2]["high_idx"] = traj["num"]["action_high"][-3][
                "high_idx"
            ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ALFRED IQL offline RL model on a fixed dataset of low-level, language annotated skills"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="./json_2.1.0_merge_goto",
        help="json file directory",
    )

    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default="./preprocess",
        help="preprocessed data directory",
    )

    args = parser.parse_args()

    dataset = Dataset(args)
    dataset.preprocess_splits()
