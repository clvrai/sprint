import numpy as np
from sprint.alfred.env.thor_env import ThorEnv
from sprint.rollouts.rollout import sample_task, setup_scene
import torch
from sprint.models.saycan_llm import SaycanPlanner
import revtok
from sprint.utils.data_utils import (
    remove_spaces_and_lower,
    numericalize,
    process_annotation,
)
import os
import sys
from utils import generate_video

path = "."
sys.path.append(os.path.join(path))
sys.path.append(os.path.join(path, "gen"))
sys.path.append(os.path.join(path, "models"))
sys.path.append(os.path.join(path, "models", "eval"))
from sprint.utils.utils import (
    load_object_class,
    AttrDict,
    get_action_from_agent,
    process_skill_strings,
)

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

vocab = torch.load(f"{os.environ['SPRINT']}/sprint/models/low_level_actions.vocab")
vocab_obj = torch.load(f"{os.environ['SPRINT']}/sprint/models/obj_cls.vocab")

DATA_PATH = (
    f"{os.environ['SPRINT']}/sprint/alfred/data/json_2.1.0_merge_goto/preprocess"
)
VISUAL_MODEL = "resnet18"
REWARD_CONFIG = f"{os.environ['SPRINT']}/sprint/alfred/models/config/rewards.json"
DEFAULT_NUM_STEPS = 30
EVAL_STEP_RATIO = 2
TRAIN_STEP_RATIO = 2

global_task_args = AttrDict()
global_task_args.reward_config = REWARD_CONFIG
global_task_args.visual_model = VISUAL_MODEL


def get_next_skill_from_saycan(
    model,
    saycan_planner: SaycanPlanner,
    sentence_embedder,
    high_level_skill: str,
    primitive_skill_annotations: list[str],
    already_completed_skills: list[str],
    feat: dict,
    device,
):
    llm_logprobs = saycan_planner.get_saycan_logprobs(
        already_completed_skills,
        primitive_skill_annotations,
        [high_level_skill],
    )
    # get value logprobs
    primitive_embeddings = sentence_embedder.encode(
        primitive_skill_annotations, convert_to_tensor=True
    )
    values = []
    for primitive_embedding in primitive_embeddings:
        primitive_embedding = primitive_embedding.to(device)
        feat["language_ann"] = primitive_embedding.unsqueeze(0)
        *_, value = model.step(feat, ret_value=True)
        values.append(value.unsqueeze(0))
    values = torch.cat(values, dim=0)
    values = torch.clamp(values, min=0, max=1).cpu()
    # combine LLM and values
    llm_probs = torch.exp(llm_logprobs)
    combined_affordance_probs = llm_probs * values
    # now take the argmax
    next_skill_idx = torch.argmax(combined_affordance_probs).item()
    feat["language_ann"] = sentence_embedder.encode(
        primitive_skill_annotations[next_skill_idx : next_skill_idx + 1],
        convert_to_tensor=True,
    ).to(
        device
    )  # re-encode the selected skill so there's no padding
    return primitive_skill_annotations[next_skill_idx]


def run_policy(
    env,
    model,
    saycan_planner,
    sentence_embedder,
    visual_preprocessor,
    device,
    subgoal_pool,
    max_skill_length,
    goal_states,
    eval_split,
    num_subgoals_in_pool,
    deterministic,
    log_video,
    epsilon,
    selected_specific_subgoal=None,
    eval=True,
):
    # actually does the rollout. This function is a bit tricky to integrate with ALFRED's required code.
    model.eval()
    if eval_split == "eval_instruct":
        json_path_name = "train"
    elif eval_split == "eval_length":
        json_path_name = "train"
    else:
        json_path_name = "valid_unseen"
    with torch.no_grad():
        eval_idx, traj_data, r_idx, actually_selected_subgoal = sample_task(
            json_path_name,
            subgoal_pool,
            num_subgoals_in_pool,
            selected_specific_subgoal,
        )
        num_subgoals_to_complete = len(
            subgoal_pool[actually_selected_subgoal]["primitive_skills"]
        )
        num_primitive_steps_in_task = sum(
            [
                len(primitive_skill["api_action"])
                for primitive_skill in subgoal_pool[actually_selected_subgoal][
                    "primitive_skills"
                ]
            ]
        )
        if eval:
            MAX_STEPS = num_primitive_steps_in_task * EVAL_STEP_RATIO
        else:
            MAX_STEPS = num_primitive_steps_in_task * TRAIN_STEP_RATIO

        setup_scene(env, traj_data, r_idx, global_task_args)

        expert_init_actions = [
            a["discrete_action"]
            for a in traj_data["plan"]["low_actions"]
            if a["high_idx"] < eval_idx
        ]
        all_instructions = process_skill_strings(
            [
                subgoal_pool[actually_selected_subgoal]["primitive_skills"][0][
                    "annotations"
                ]
            ]
            + [subgoal_pool[actually_selected_subgoal]["annotation"]]
        )
        primitive_skills = all_instructions[:-1]
        # subgoal info
        chained_subgoal_instr = all_instructions[-1]

        ann_l = revtok.tokenize(remove_spaces_and_lower(chained_subgoal_instr))
        ann_l = [w.strip().lower() for w in ann_l]
        ann_token = numericalize(model.vocab_word, ann_l, train=False)
        ann_token = torch.tensor(ann_token).long()
        feat = {}
        feat["language_ann"] = ann_token.to(device).unsqueeze(0)
        if goal_states is not None:
            task_name = subgoal_pool[actually_selected_subgoal]["task"]
            subgoal_str = list()
            for p in range(eval_idx, (eval_idx + num_subgoals_to_complete)):
                subgoal_str.append(str(p))
            trial_name = task_name + "-" + "_".join(subgoal_str)

            task_index = goal_states[json_path_name]["trial_name"].index(trial_name)
            trial_goal = goal_states[json_path_name]["goal_state"][task_index]
            feat["state_goal"] = trial_goal.reshape([1, 5, 512, 7, 7]).to(device)

        completed_eval_idx = eval_idx + num_subgoals_to_complete - 1
        done = 0
        t = 0

        obs = []
        acs = []
        obj_acs = []
        dones = []
        env_rewards = []
        str_act = []
        video_frames = []
        completed_skills = []
        predicted_skills = []
        value_predict = []
        while not done:
            # break if max_steps reached
            if t >= MAX_STEPS + len(expert_init_actions):
                break

            if (len(expert_init_actions) == 0 and t == 0) or (
                len(expert_init_actions) != 0 and t == len(expert_init_actions)
            ):
                curr_image = Image.fromarray(np.uint8(env.last_event.frame))

                curr_frame = (
                    visual_preprocessor.featurize([curr_image], batch=1)
                    .unsqueeze(0)
                    .to(device)
                )
                feat["frames_buffer"] = curr_frame.to(device)
                feat["action_traj"] = torch.zeros(1, 0).long().to(device)
                feat["object_traj"] = torch.zeros(1, 0).long().to(device)
                # get first saycan planner action
                saycan_selected_next_skill = get_next_skill_from_saycan(
                    model,
                    saycan_planner,
                    sentence_embedder,
                    chained_subgoal_instr,
                    primitive_skills,
                    completed_skills,
                    feat,
                    device,
                )
                predicted_skills.append(saycan_selected_next_skill)

            if t < len(expert_init_actions):
                # get expert action

                action = expert_init_actions[t]
                # print("expert_action", action)
                compressed_mask = (
                    action["args"]["mask"] if "mask" in action["args"] else None
                )
                mask = (
                    env.decompress_mask(compressed_mask)
                    if compressed_mask is not None
                    else None
                )

                success, _, _, err, _ = env.va_interact(
                    action["action"], interact_mask=mask, smooth_nav=True, debug=False
                )
                if not success:
                    print("expert initialization failed, re-sampling")
                    return run_policy(
                        env,
                        model,
                        visual_preprocessor,
                        device,
                        subgoal_pool,
                        max_skill_length,
                        eval_split,
                        num_subgoals_in_pool,
                        deterministic,
                        log_video,
                        epsilon,
                        selected_specific_subgoal,
                        eval,
                    )
                _, _ = env.get_transition_reward()
            else:
                obs.append(curr_frame.cpu().detach().squeeze(1))
                video_frames.append(np.uint8(env.last_event.frame))

                (action, output_object, _) = get_action_from_agent(
                    model,
                    feat,
                    vocab,
                    vocab_obj,
                    env,
                    deterministic=deterministic,
                    epsilon=epsilon,
                    ret_value=False,
                )
                value_output = None
                if value_output != None:
                    value_output = value_output.squeeze().cpu().detach().numpy()
                    value_predict.append(value_output)
                try:
                    _, _ = env.to_thor_api_exec(action, output_object, smooth_nav=True)
                except Exception as e:
                    # if there's an exception from running, then we'll just try again.
                    # print(e)
                    # record this exception in exceptions.txt
                    with open("exceptions.txt", "a") as f:
                        f.write(str(e) + "\n")

                next_frame = np.uint8(env.last_event.frame)
                next_frame = Image.fromarray(next_frame)
                next_frame = (
                    visual_preprocessor.featurize([next_frame], batch=1)
                    .unsqueeze(0)
                    .to(device)
                )
                curr_frame = next_frame

                feat["frames_buffer"] = torch.cat(
                    [feat["frames_buffer"], next_frame], dim=1
                ).to(device)
                # - 2 because ET dataloader had a -1 for padding reasons on the action, and we did - 1 when processing ALFRED actions to get rid of
                # the extraneous END action
                tensor_action = torch.tensor(
                    vocab["action_low"].word2index(action) - 2
                ).to(device)
                feat["action_traj"] = torch.cat(
                    [feat["action_traj"], tensor_action.unsqueeze(0).unsqueeze(0)],
                    dim=1,
                ).to(device)
                obj_index = load_object_class(vocab_obj, output_object)
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

                t_success = env.last_event.metadata["lastActionSuccess"]
                if not t_success:
                    # env error logging just in case, but we'll just continue
                    # these aren't critical errors, just sim errors
                    err = env.last_event.metadata["errorMessage"]
                    exception_string = f"Failed to execute action {action}. Object: {output_object}. Error: {err}"
                    with open("exceptions.txt", "a") as f:
                        f.write(exception_string + "\n")

                # MUST call get_transition_reward to update the environment
                _, _ = env.get_transition_reward()
                curr_subgoal_idx = env.get_subgoal_idx()
                partial_success = 0
                if curr_subgoal_idx == completed_eval_idx:
                    done = 1
                    partial_success = 1
                elif curr_subgoal_idx == eval_idx:
                    eval_idx += 1
                    partial_success = 1
                    completed_skills.append(saycan_selected_next_skill)
                    feat["frames_buffer"] = next_frame.unsqueeze(0).to(device)
                    feat["action_traj"] = torch.zeros(1, 0).long().to(device)
                    feat["object_traj"] = torch.zeros(1, 0).long().to(device)
                    saycan_selected_next_skill = get_next_skill_from_saycan(
                        model,
                        saycan_planner,
                        sentence_embedder,
                        chained_subgoal_instr,
                        primitive_skills,
                        completed_skills,
                        feat,
                        device,
                    )
                    predicted_skills.append(saycan_selected_next_skill)
                env_rewards.append(partial_success)

            t = t + 1
        subgoal_last_frame_video = np.uint8(env.last_event.frame)
        video_frames.append(subgoal_last_frame_video)
        (*_,) = get_action_from_agent(
            model,
            feat,
            vocab,
            vocab_obj,
            env,
            deterministic=deterministic,
            epsilon=epsilon,
            ret_value=False,
        )
        obs.append(next_frame.cpu().detach().squeeze(1))  # next obs
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
        vid_caption = f"{chained_subgoal_instr[0] if isinstance(chained_subgoal_instr, list) else chained_subgoal_instr}: {'SUCCESS' if done else 'FAIL'}. Return: {rewards.sum()}/{num_subgoals_to_complete}."
        ground_truth_sequence = " ".join(primitive_skills)
        return dict(
            completed_skills=" ".join(completed_skills),
            predicted_skills=" ".join(predicted_skills),
            ground_truth_sequence=ground_truth_sequence,
            high_level_skill=chained_subgoal_instr,
            rews=rewards,
            dones=dones,
            video_frames=video_frames if log_video else None,
            video_caption=vid_caption,
            chained_language_instruction=process_annotation(
                chained_subgoal_instr, model.vocab_word, train=False
            ).long(),
            skill_length=num_subgoals_to_complete,
        )


def run_policy_multi_process(
    ret_queue,
    task_queue,
    offline_rl_model,
    resnet,
    device,
    subgoal_pool,
    max_skill_length,
    goal_states,
):
    env = ThorEnv()
    while True:
        task_args = task_queue.get()
        if task_args is None:
            break
        ret_queue.put(
            run_policy(
                env,
                offline_rl_model,
                resnet,
                device,
                subgoal_pool,
                max_skill_length,
                goal_states,
                *task_args,
            )
        )
    env.stop()
