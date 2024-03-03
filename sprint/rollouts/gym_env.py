# gym interface for ALFRED-RL env
import gym
from sprint.alfred.env.thor_env import ThorEnv
from sprint.utils.utils import AttrDict, process_skill_strings
import torch
import numpy as np
import os
import json
import random

DEFAULT_NUM_STEPS = 30
STEP_RATIO = 2


class ALFREDRLEnv(gym.Env):

    def __init__(
        self,
        data_path,
        reward_config_path,
        task_suite,
        eval_json_path,
        vocab_path,
        vocab_obj_path,
        specific_task=None,
    ):
        """_summary_

        Args:
            data_path (str): path where ALFRED data files are saved
            reward_config_path (str): path where the reward config file is saved
            task_suite (str): which task suite to use
            eval_json_path (str): where the json file for the eval path file is located
            vocab_path (str): where the vocab file is located for action idx to strings
            vocab_obj_path (str): where the vocab file is located for object idx to strings
            specific_task (int, optional): if not None, will reset the environment to this specific task
        """
        self.data_path = data_path
        self.task_args = AttrDict()
        self.task_args.reward_config = reward_config_path
        self.eval_json_path = eval_json_path

        if task_suite == "eval_instruct":
            json_path_name = "train"
        elif task_suite == "eval_length":
            json_path_name = "train"
        else:
            json_path_name = "valid_unseen"

        self.json_path_name = json_path_name
        self.num_subgoals_to_complete = 0
        self.max_steps = 0
        self.curr_step = 0
        self.obs = None
        self.lang_instruction = None
        self.curr_subgoal_idx = None
        self.curr_task_idx = None
        self.first_subgoal_idx = None
        self.vocab = torch.load(vocab_path)
        self.vocab_obj = torch.load(vocab_obj_path)
        self._specific_task = specific_task
        self._thor_env = ThorEnv()
        self.subgoal_pool = self._load_task_pool()

    @property
    def num_tasks(self):
        return len(self.subgoal_pool)

    def _load_task_pool(self):
        if os.path.exists(self.eval_json_path):
            with open(self.eval_json_path, "r") as f:
                eval_skill_info_list = json.load(f)
        else:
            raise ValueError(f"Could not find the eval json at {self.eval_json_path}")

        # sort both skill info lists by num_primitive_skills, descending, for faster evaluation with multiple threads
        eval_skill_info_list.sort(
            key=lambda x: len(x["primitive_skills"]), reverse=True
        )
        if self._specific_task is not None:
            eval_skill_info_list = [eval_skill_info_list[self._specific_task]]
        return eval_skill_info_list

    def reset(self, specific_task=None):
        first_subgoal_idx, traj_data, REPEAT_ID, specific_task = self._sample_task(
            specific_task
        )
        curr_task = self.subgoal_pool[specific_task]
        self.num_subgoals_to_complete = len(curr_task["primitive_skills"])
        expert_init_actions = [
            a["discrete_action"]
            for a in traj_data["plan"]["low_actions"]
            if a["high_idx"] < first_subgoal_idx
        ]
        num_primitive_steps_in_task = sum(
            [
                len(primitive_skill["api_action"])
                for primitive_skill in curr_task["primitive_skills"]
            ]
        )
        self.max_steps = num_primitive_steps_in_task * STEP_RATIO
        self._setup_scene(traj_data)
        t = 0

        # set the initial state of the environment to the target initial state of the sampled task
        while t < len(expert_init_actions):
            action = expert_init_actions[t]
            compressed_mask = (
                action["args"]["mask"] if "mask" in action["args"] else None
            )
            mask = (
                self._thor_env.decompress_mask(compressed_mask)
                if compressed_mask is not None
                else None
            )
            success, _, _, err, _ = self._thor_env.va_interact(
                action["action"], interact_mask=mask, smooth_nav=True, debug=False
            )
            t += 1
            if not success:
                print(
                    "Failed to execute expert action when initializing ALFRED env, retrying"
                )
                return self.reset(specific_task)
            _, _ = (
                self._thor_env.get_transition_reward()
            )  # advances the reward function
        curr_frame = np.uint8(self.last_event.frame)
        curr_lang_instruction = process_skill_strings(
            self.subgoal_pool[specific_task]["annotation"]
        )[0]
        self.lang_instruction = curr_lang_instruction
        self.obs = curr_frame
        self.curr_subgoal_idx = first_subgoal_idx
        self.first_subgoal_idx = first_subgoal_idx
        self.curr_task_idx = specific_task
        self.curr_step = 0
        # obs, info
        return self.obs, self._build_info()

    def _build_info(self):
        return AttrDict(
            task=self.curr_task_idx,
            subgoal=self.curr_subgoal_idx,
            timeout=self.curr_step >= self.max_steps,
            lang_instruction=self.lang_instruction,
        )

    def cleanup(self):
        self._thor_env.stop()

    def _sample_task(
        self,
        specific_task=None,
    ):
        if specific_task is None:
            specific_task = random.randint(0, len(self.subgoal_pool) - 1)
        log = self.subgoal_pool[specific_task]

        task = log["task"]
        REPEAT_ID = log["repeat_id"]
        eval_idx = log["subgoal_ids"][0]
        json_path = os.path.join(
            self.data_path, self.json_path_name, task, "ann_%d.json" % REPEAT_ID
        )
        with open(json_path) as f:
            traj_data = json.load(f)
        return eval_idx, traj_data, REPEAT_ID, specific_task

    def _setup_scene(self, traj_data):
        """
        intialize the scene and agent from the task info
        """
        # scene setup
        scene_num = traj_data["scene"]["scene_num"]
        object_poses = traj_data["scene"]["object_poses"]
        dirty_and_empty = traj_data["scene"]["dirty_and_empty"]
        object_toggles = traj_data["scene"]["object_toggles"]

        scene_name = "FloorPlan%d" % scene_num
        self._thor_env.reset(scene_name)
        self._thor_env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        self._thor_env.step(dict(traj_data["scene"]["init_action"]))

        # setup task for reward
        self._thor_env.set_task(traj_data, self.task_args, reward_type="dense")

    def step(self, action_dict: dict):
        # action_dict is an ordered dict with keys "action" and "object"
        action, obj_action = action_dict["action"], action_dict["object"]
        # if action and obj_action are already given as strings, just use directly
        if not isinstance(action, str):
            action = self.vocab[action + 3]  # + 3 offset for the 3 unused actions
            obj_action = self.vocab_obj[obj_action]
        try:
            _, _ = self._thor_env.to_thor_api_exec(action, obj_action, smooth_nav=True)
        except Exception as e:
            # ignore failed action execution
            pass
        self.curr_step += 1
        self.obs = np.uint8(self.last_event.frame)
        _, _ = self._thor_env.get_transition_reward()  # advances the reward function
        # subgoal_idx from thor_env starts at -1 and iterates up
        new_subgoal_idx = self._thor_env.get_subgoal_idx() + 1
        rew = 0
        if new_subgoal_idx > self.curr_subgoal_idx:
            self.curr_subgoal_idx = new_subgoal_idx
            rew = 1
        done = new_subgoal_idx == self.num_subgoals_to_complete + self.first_subgoal_idx
        info = self._build_info()
        return self.obs, rew, done, info

    @property
    def last_event(self):
        return self._thor_env.last_event

    @property
    def action_space(self):
        return gym.spaces.Dict(
            {
                "action": gym.spaces.Discrete(len(self.vocab) - 3),
                "object": gym.spaces.Discrete(len(self.vocab_obj)),
            }
        )

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
