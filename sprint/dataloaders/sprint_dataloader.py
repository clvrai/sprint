import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
from sprint.utils.data_utils import process_annotation, load_object_class
from sprint.dataloaders.base_dataloader import CustomDataset


class SPRINTDataset(CustomDataset):
    def __init__(
        self,
        path,
        data_type,
        sample_primitive_skill,
        max_skill_length,
        use_full_skill=False,
    ):

        self.path = path
        self.data_type = data_type
        self.sample_primitive_skill = sample_primitive_skill
        self.use_full_skill = use_full_skill
        self.data = self.load_pyxis()
        self.vocab_obj = torch.load(
            f"{os.environ['SPRINT']}/sprint/models/obj_cls.vocab"
        )
        self.vocab_ann = torch.load(
            f"{os.environ['SPRINT']}/sprint/models/sprint_human.vocab"
        )
        self.max_skill_length = max_skill_length
        self.include_list_dict = [
            "lang_low_action",
            "lang_object_ids",
            "lang_valid_interact",
            "lang_subgoals",
            "lang_combinations",
            "lang_summaries",
            "lang_ridx",
        ]

        self.include_all_dict = [
            "traj_resnet_feature",
            "skill_switch_point",
        ]

        # sampling primitive skill means we only sample the primitive skills
        if not self.sample_primitive_skill:
            pkl_name = "SPRINT_composite_skill_set_" + data_type
        else:
            pkl_name = "SPRINT_primitive_skill_set_" + data_type
        pkl_name += ".pkl"
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                (self.single_sample_trajectory_dict,) = pickle.load(f)
        else:
            self.create_single_sample_trajectory_dict()
            # in case another process takes over and saves it first
            if not os.path.exists(pkl_name):
                with open(pkl_name, "wb") as f:
                    pickle.dump(
                        (self.single_sample_trajectory_dict,),
                        f,
                    )

    def create_single_sample_trajectory_dict(self):
        print("generating pickle file")
        single_sample_trajectory_dict = {}
        total_samples = 0

        if not self.sample_primitive_skill:
            for i in tqdm(range(len(self.data))):

                num_skill = self.data[i]["lang_ridx"].shape[0]

                for j in range(num_skill):
                    single_sample_trajectory_dict[total_samples] = (i, j)
                    total_samples += 1

        else:
            for i in tqdm(range(len(self.data))):
                num_skill = self.data[i]["lang_ridx"].shape[0]
                for j in range(num_skill):
                    if len(self.data[i]["lang_subgoals"][j].split("+")) == 1:
                        single_sample_trajectory_dict[total_samples] = (i, j)
                        total_samples += 1
        self.single_sample_trajectory_dict = single_sample_trajectory_dict

    def get_data_from_pyxis(self, i, j):
        data_dict = self.data[i]
        if "lang_object_ids" in data_dict:
            data_dict["lang_object_ids"] = pickle.loads(data_dict["lang_object_ids"])

        traj_dict = {}
        for key, value in data_dict.items():
            if key in self.include_list_dict:
                traj_dict[key] = value[j]
            elif key in self.include_all_dict:
                traj_dict[key] = value

        skill_dict = {}
        subgoal_idx = [int(x) for x in traj_dict["lang_subgoals"].split("+")]

        # process resnet feature
        skill_start_index = subgoal_idx[0]
        skill_end_index = subgoal_idx[-1]

        start_index = data_dict["skill_switch_point"][skill_start_index]

        if skill_end_index == len(data_dict["skill_switch_point"]) - 1:
            skill_feature = data_dict["traj_resnet_feature"][start_index:]

        else:
            end_index = (
                data_dict["skill_switch_point"][skill_end_index + 1] + 1
            )  # goal state include the last state of skill combination
            skill_feature = data_dict["traj_resnet_feature"][start_index:end_index]
        skill_feature = torch.from_numpy(
            skill_feature
        ).float()  # this includes the last state
        low_action = np.asarray(
            [float(action) for action in traj_dict["lang_low_action"].split("+")]
        )
        low_action = torch.from_numpy(low_action).float()
        object_ids = traj_dict["lang_object_ids"]
        object_ids = np.asarray(
            [load_object_class(self.vocab_obj, ids) for ids in object_ids]
        )
        object_ids = torch.from_numpy(object_ids).float()
        valid_interact = np.asarray(
            [int(va) for va in traj_dict["lang_valid_interact"].split("+")]
        )
        valid_interact = torch.from_numpy(valid_interact)

        annotation = traj_dict["lang_combinations"]
        ann_token = process_annotation(
            annotation, self.vocab_ann, train="train" in self.data_type
        )
        summary_annotation = traj_dict["lang_summaries"]
        summary_token = process_annotation(
            summary_annotation, self.vocab_ann, train="train" in self.data_type
        )

        if self.use_full_skill:
            start = 0
        else:
            start = random.randint(0, len(low_action) - 1)
        skill_dict["skill_feature"] = skill_feature[
            start : start + self.max_skill_length + 1
        ]  # shape:  batch x seq_len x 512 x 7 x 7

        # for actionable models to get the last 5 images as the goal frame
        skill_dict["goal_feature"] = skill_feature[
            -5:
        ]  # shape: batch x 5 x 512 x 7 x 7

        skill_dict["low_action"] = (
            low_action[start : start + self.max_skill_length] - 1
        )  # shape: batch x action_len
        rewards = torch.zeros(skill_dict["low_action"].shape[0])
        if start + self.max_skill_length >= len(skill_dict["low_action"]):
            rewards[-1] = 1
        assert len(skill_dict["skill_feature"]) == len(skill_dict["low_action"]) + 1
        skill_dict["object_ids"] = object_ids[
            start : start + self.max_skill_length
        ]  # shape: batch x object
        skill_dict["valid_interact"] = valid_interact[
            start : start + self.max_skill_length
        ]
        skill_dict["annotation"] = annotation
        skill_dict["summary_annotation"] = summary_token
        skill_dict["summary_token_length"] = summary_token.shape[0]
        skill_dict["ann_token"] = ann_token
        skill_dict["token_length"] = ann_token.shape[0]  # token length
        # skill_dict["skill_length"] = skill_dict["low_action"].shape[0]
        skill_dict["terminal"] = skill_dict["reward"] = rewards
        skill_dict["feature_length"] = skill_dict["skill_feature"].shape[
            0
        ]  # feature length one more than low action number
        return skill_dict


def collate_func(batch_dic):
    batch_len = len(batch_dic)  # size

    skill_feature = []
    goal_feature = []
    annotations = []
    low_action = []
    object_ids = []
    valid_interact = []
    ann_token = []
    summary_ann_token = []
    summary_token_length = []
    feature_length = []
    token_length = []
    reward = []
    terminal = []
    for i in range(batch_len):
        dic = batch_dic[i]
        skill_feature.append(dic["skill_feature"])
        low_action.append(dic["low_action"])
        object_ids.append(dic["object_ids"])
        valid_interact.append(dic["valid_interact"])
        ann_token.append(dic["ann_token"])
        summary_ann_token.append(dic["summary_annotation"])
        goal_feature.append(dic["goal_feature"])
        feature_length.append(dic["feature_length"])
        token_length.append(dic["token_length"])
        annotations.append(dic["annotation"])
        reward.append(dic["reward"])
        terminal.append(dic["terminal"])
        summary_token_length.append(dic["summary_token_length"])

    res = {}
    res["skill_feature"] = pad_sequence(
        skill_feature, batch_first=True, padding_value=0
    )
    res["goal_feature"] = pad_sequence(goal_feature, batch_first=True, padding_value=0)
    # pad one more to do parallel curr state/next state value computation and match length with skill feature
    low_actions_padded = pad_sequence(low_action, batch_first=True, padding_value=0)
    res["low_action"] = torch.cat(
        (low_actions_padded, torch.zeros((batch_len,)).unsqueeze(1)), dim=1
    )
    obj_ids_padded = pad_sequence(object_ids, batch_first=True, padding_value=0)
    res["object_ids"] = torch.cat(
        (obj_ids_padded, torch.zeros((batch_len,)).unsqueeze(1)), dim=1
    )

    res["valid_interact"] = pad_sequence(
        valid_interact, batch_first=True, padding_value=0
    )
    res["ann_token"] = pad_sequence(ann_token, batch_first=True, padding_value=0)
    res["feature_length"] = torch.tensor(np.asarray(feature_length))
    res["token_length"] = torch.tensor(np.asarray(token_length))
    res["summary_token_length"] = torch.tensor(np.asarray(summary_token_length))
    res["summary_token"] = pad_sequence(
        summary_ann_token, batch_first=True, padding_value=0
    )
    res["summary_token_list"] = summary_ann_token
    res["ann_token_list"] = ann_token
    res["annotation"] = annotations
    res["reward"] = pad_sequence(reward, batch_first=True, padding_value=-1)
    res["terminal"] = pad_sequence(terminal, batch_first=True, padding_value=-1)

    return res


class RLBuffer(SPRINTDataset):
    def __init__(
        self,
        path,
        split,
        drop_old_data,
        use_full_skill,
        max_skill_length,
        sample_primitive_skill,
        use_llm_labels,  # for old data
        max_size=float("inf"),
    ):
        self.drop_old_data = drop_old_data
        super(RLBuffer, self).__init__(
            path,
            split,
            sample_primitive_skill=sample_primitive_skill,
            max_skill_length=max_skill_length,
            use_full_skill=use_full_skill,
        )
        self.rl_buffer = []
        self.use_llm_labels_for_old_data = use_llm_labels
        self.max_size = max_size
        if hasattr(self, "drop_old_data") and self.drop_old_data:  # for RL buffer
            self.data = None
        else:
            self.data = self.load_pyxis()

    def __len__(self):
        if self.drop_old_data:
            return len(self.rl_buffer)
        else:
            return len(self.single_sample_trajectory_dict) + len(self.rl_buffer)

    def __getitem__(self, index):
        if self.drop_old_data:
            return self.get_data_from_RL_buffer(index)
        else:
            if index < len(self.single_sample_trajectory_dict):
                data = super(SPRINTDataset, self).__getitem__(index)
                if self.use_llm_labels_for_old_data:
                    data["ann_token"] = data["summary_annotation"]
                    data["token_length"] = data["summary_token_length"]
                return data
            else:
                return self.get_data_from_RL_buffer(
                    index - len(self.single_sample_trajectory_dict)
                )

    def add_traj_to_buffer(
        self,
        frames,
        actions,
        obj_acs,
        rewards,
        terminals,
        language,
        goal_frames=None,
    ):
        assert len(actions) == len(frames) - 1
        assert len(actions) == len(obj_acs)
        self.rl_buffer.append(
            {
                "frames": frames,
                "actions": actions,
                "obj_acs": obj_acs,
                "rewards": rewards,
                "terminals": terminals,
                "language": (
                    language.squeeze(0) if len(language.shape) > 1 else language
                ),
                "goal_frames": goal_frames.squeeze(0) if goal_frames is not None else None,
            }
        )
        if len(self.rl_buffer) > self.max_size:
            # replace the oldest one
            self.rl_buffer.pop(0)

    def get_data_from_RL_buffer(self, i):
        traj_dict = self.rl_buffer[i]
        skill_dict = {}
        if len(traj_dict["actions"]) > self.max_skill_length:
            start = random.randint(0, len(traj_dict["actions"]) - self.max_skill_length)
            skill_dict["skill_feature"] = traj_dict["frames"][
                start : start + self.max_skill_length + 1
            ]
            skill_dict["low_action"] = traj_dict["actions"][
                start : start + self.max_skill_length
            ]
            skill_dict["object_ids"] = traj_dict["obj_acs"][
                start : start + self.max_skill_length
            ]
            skill_dict["valid_interact"] = (
                traj_dict["obj_acs"][start : start + self.max_skill_length] != 0
            )
            skill_dict["reward"] = traj_dict["rewards"][
                start : start + self.max_skill_length
            ]
            skill_dict["terminal"] = traj_dict["terminals"][
                start : start + self.max_skill_length
            ]

        else:
            skill_dict["skill_feature"] = traj_dict["frames"]
            skill_dict["low_action"] = traj_dict["actions"]
            skill_dict["object_ids"] = traj_dict["obj_acs"]
            skill_dict["valid_interact"] = traj_dict["obj_acs"] != 0
            skill_dict["reward"] = traj_dict["rewards"]
            skill_dict["terminal"] = traj_dict["terminals"]
        # for Actionable Models
        skill_dict["goal_feature"] = (
            traj_dict["goal_frames"]
            if traj_dict["goal_frames"] is not None
            else torch.empty_like(skill_dict["skill_feature"][0:1])
        )

        skill_dict["ann_token"] = traj_dict["language"]
        skill_dict["token_length"] = traj_dict["language"].shape[0]  # token length
        skill_dict["feature_length"] = skill_dict["skill_feature"].shape[
            0
        ]  # feature length one more than low action number
        skill_dict["annotation"] = traj_dict["language"]  # for compatibility
        skill_dict["summary_annotation"] = traj_dict[
            "language"
        ]  # for auxiliary (SPRINT) objectives during online rl
        skill_dict["summary_token_length"] = traj_dict["language"].shape[
            0
        ]  # for auxiliary (SPRINT) objectives during online rl
        return skill_dict


# saving the updated vocab in case there are new words
# TODO: REMEMBER TO DO THIS!
if __name__ == "__main__":
    """
    commenting this out for now


    path = "./data/px_llama_13b/px_llama_13b"
    dataset = ETIQLFeeder(path, "train", False, 5)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_func,
        # pin_memory=True,
    )
    for item in tqdm(dataloader):

        pass
    torch.save(dataset.vocab_ann, "ET/sprint_human.vocab")
    """
