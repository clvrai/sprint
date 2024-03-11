import numpy as np
import torch
import os

import revtok
from sprint.rollouts.gym_env import ALFREDRLEnv
from sprint.utils.data_utils import (
    remove_spaces_and_lower,
    numericalize,
    process_annotation,
)
from sprint.utils.utils import (
    get_action_from_agent,
    load_object_class,
)

path = "."

import sys
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(path, "gen"))
sys.path.append(os.path.join(path, "models"))
sys.path.append(os.path.join(path, "models", "eval"))
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

vocab_path = f"{os.environ['SPRINT']}/sprint/models/low_level_actions.vocab"
vocab_obj_path = f"{os.environ['SPRINT']}/sprint/models/obj_cls.vocab"

DATA_PATH = (
    f"{os.environ['SPRINT']}/sprint/alfred/data/json_2.1.0_merge_goto/preprocess"
)
REWARD_CONFIG_PATH = f"{os.environ['SPRINT']}/sprint/alfred/models/config/rewards.json"


def run_policy(
    env: ALFREDRLEnv,
    model,
    visual_preprocessor,
    device,
    max_skill_length,
    goal_states,
    deterministic,
    log_video,
    epsilon,
    selected_specific_subgoal=None,
):
    # actually does the rollout. This function is a bit tricky to integrate with ALFRED's required code.
    model.eval()
    ob, info = env.reset(selected_specific_subgoal)
    # build the features to give as input to the actual model
    feat = {}
    # initialize frames and action buffers for the transformer
    ob = visual_preprocessor.featurize([Image.fromarray(ob)], batch=1)
    feat["frames_buffer"] = ob.unsqueeze(0).to(device)
    feat["action_traj"] = torch.zeros(1, 0).long().to(device)
    feat["object_traj"] = torch.zeros(1, 0).long().to(device)

    chained_subgoal_instr = info.lang_instruction
    actually_selected_subgoal = info.task
    # subgoal info

    ann_l = revtok.tokenize(remove_spaces_and_lower(chained_subgoal_instr))
    ann_l = [w.strip().lower() for w in ann_l]
    ann_token = numericalize(model.vocab_word, ann_l, train=False)
    ann_token = torch.tensor(ann_token).long()
    feat["language_ann"] = ann_token.to(device).unsqueeze(0)
    if goal_states is not None:
        task_name = env.subgoal_pool[actually_selected_subgoal]["task"]
        subgoal_str = list()
        for p in range(env.first_subgoal_idx, (env.num_subgoals_to_complete + env.first_subgoal_idx)):
            subgoal_str.append(str(p))
        trial_name = task_name + "-" + "_".join(subgoal_str)
        task_index = goal_states[env.json_path_name]["trial_name"].index(trial_name)
        trial_goal = goal_states[env.json_path_name]["goal_state"][task_index]
        feat["state_goal"] = trial_goal.reshape([1, 5, 512, 7, 7]).to(device)

    obs = []
    acs = []
    obj_acs = []
    dones = []
    env_rewards = []
    str_act = []
    video_frames = []
    value_predict = []
    done, timeout = False, False
    while not (done or timeout):
        obs.append(ob.cpu().detach().squeeze(1))
        video_frames.append(env.obs)

        (action, output_object, _) = get_action_from_agent(
            model,
            feat,
            env.vocab,
            env.vocab_obj,
            env,
            deterministic=deterministic,
            epsilon=epsilon,
            ret_value=False,
        )
        value_output = None
        if value_output != None:
            value_output = value_output.squeeze().cpu().detach().numpy()
            value_predict.append(value_output)

        action_dict = dict(action=action, object=output_object)
        next_ob, rew, done, info = env.step(action_dict)
        timeout = info.timeout
        next_ob = Image.fromarray(next_ob)
        next_ob = (
            visual_preprocessor.featurize([next_ob], batch=1)
            .unsqueeze(0)
            .to(device)
        )
        ob = next_ob
        feat["frames_buffer"] = torch.cat(
            [feat["frames_buffer"], next_ob], dim=1
        ).to(device)
        # - 2 because ET dataloader had a -1 for padding reasons on the action, and we did - 1 when processing ALFRED actions to get rid of
        # the extraneous END action
        tensor_action = torch.tensor(
            env.vocab["action_low"].word2index(action) - 2
        ).to(device)
        feat["action_traj"] = torch.cat(
            [feat["action_traj"], tensor_action.unsqueeze(0).unsqueeze(0)],
            dim=1,
        ).to(device)
        obj_index = load_object_class(env.vocab_obj, output_object)
        feat["object_traj"] = torch.cat(
            [
                feat["object_traj"],
                torch.tensor(obj_index).unsqueeze(0).unsqueeze(0).to(device),
            ],
            dim=1,
        )
        feat["frames_buffer"] = feat["frames_buffer"][:, -max_skill_length:]
        feat["action_traj"] = feat["action_traj"][:, -max_skill_length + 1 :]
        feat["object_traj"] = feat["object_traj"][:, -max_skill_length + 1 :]

        env_rewards.append(rew)

        acs.append(tensor_action.cpu())
        str_act.append(
            dict(
                action=action,
                object=(
                    output_object.split("|")[0]
                    if output_object is not None
                    else None
                ),
            )
        )
        obj_acs.append(obj_index)

    subgoal_last_frame_video = env.obs
    video_frames.append(subgoal_last_frame_video)
    (*_,) = get_action_from_agent(
        model,
        feat,
        env.vocab,
        env.vocab_obj,
        env,
        deterministic=deterministic,
        epsilon=epsilon,
        ret_value=False,
    )
    obs.append(ob.cpu().detach().squeeze(1))  # last next obs
    value_output = None
    if value_output != None:
        value_output = value_output.squeeze().cpu().detach().numpy()
        value_predict.append(value_output)

    if log_video:
        value_font = ImageFont.truetype("FreeMono.ttf", 20)
        action_font = ImageFont.truetype("FreeMono.ttf", 14)
        gif_logs = []
        for frame_number in range(len(video_frames)):
            img = video_frames[frame_number]
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            if len(value_predict) != 0:
                value_log = value_predict[frame_number]
                # draw.text(
                #    (1, 280),
                #    "Value: %.3f" % (value_log),
                #    fill=(255, 255, 255),
                #    font=value_font,
                # )
            if frame_number != 0:
                reward_log = env_rewards[frame_number - 1]
                # draw.text(
                #    (1, 260),
                #    "Reward: %.1f" % (reward_log),
                #    fill=(255, 255, 255),
                #    font=value_font,
                # )
                return_log = sum(env_rewards[0:frame_number])
                # draw.text(
                #    (150, 260),
                #    "Return: %.1f" % (return_log),
                #    fill=(255, 255, 255),
                #    font=value_font,
                # )
            if frame_number != len(video_frames) - 1:
                action_log, object_log = (
                    str_act[frame_number]["action"],
                    str_act[frame_number]["object"],
                )
                # draw.text(
                #    (1, 1),
                #    f"Action: {action_log}\nObject: {str(object_log)}",
                #    fill=(255, 255, 255),
                #    font=action_font,
                # )

            log_images = np.array(img)
            gif_logs.append(log_images)

        video_frames = np.asarray(gif_logs)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))

    rewards = torch.tensor(env_rewards, dtype=torch.float)
    dones = torch.zeros(len(rewards))
    dones[-1] = done
    vid_caption = f"{chained_subgoal_instr[0] if isinstance(chained_subgoal_instr, list) else chained_subgoal_instr}: {'SUCCESS' if done else 'FAIL'}. Return: {rewards.sum()}/{env.num_subgoals_to_complete}."
    return dict(
        obs=torch.cat(obs),
        acs=torch.tensor(acs),
        obj_acs=torch.tensor(obj_acs),
        rews=rewards,
        dones=dones,
        video_frames=video_frames if log_video else None,
        video_caption=vid_caption,
        chained_language_instruction=process_annotation(
            chained_subgoal_instr, model.vocab_word, train=False
        ).long(),
        skill_length=env.num_subgoals_to_complete,
        goal_state=feat["state_goal"].cpu() if goal_states is not None else None,
    )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    offline_rl_model,
    resnet,
    device,
    max_skill_length,
    goal_states,
    task_suite,
    eval_json_path,
    specific_task,
):
    env = ALFREDRLEnv(
        DATA_PATH,
        REWARD_CONFIG_PATH,
        task_suite,
        eval_json_path,
        vocab_path,
        vocab_obj_path,
        specific_task,
    )
    num_eval_tasks = env.num_tasks
    # put the number of tasks into the return queue to tell the calling script thing how many rollouts to perform for evaluation
    ret_queue.put(num_eval_tasks)
    while True:
        task_args = task_queue.get()
        if task_args is None:
            break
        with torch.no_grad():
            ret_queue.put(
                run_policy(
                    env,
                    offline_rl_model,
                    resnet,
                    device,
                    max_skill_length,
                    goal_states,
                    *task_args,
                )
            )
    env.cleanup()
