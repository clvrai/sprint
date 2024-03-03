import torch
import collections
import numpy as np
from sprint.models.sprint_model import ETIQLPolicy, ETIQLCritics, SPRINTETIQLModel
from sprint.utils.model_utils import soft_update_from_to

import os
import copy


class AMETIQLPolicy(ETIQLPolicy):
    """
    transformer IQL Policy that doesn't use language
    """
    def embed_lang(self, lang_pad, vocab):
        """
        let "lang" be the image goal state
        """
        frame_lengths = torch.tensor([lang_pad.shape[1]]).repeat(lang_pad.shape[0]).long()
        return self.embed_frames(lang_pad)[0], frame_lengths

class AMETIQLCritics(ETIQLCritics):
    def embed_lang(self, lang_pad, vocab):
        """
        let "lang" be the image goal state
        """
        frame_lengths = torch.tensor([lang_pad.shape[1]]).repeat(lang_pad.shape[0]).long()
        return self.embed_frames(lang_pad)[0], frame_lengths


class AMETIQLModel(SPRINTETIQLModel):
    
    def __init__(self, args, pad=0, seg=1):
        """
        AM transformer IQL agent
        """
        super(SPRINTETIQLModel, self).__init__()
        self.vocab_word = torch.load(
            #os.path.join(f"{os.environ['SPRINT']}/sprint/models/sprint_human.vocab")
            os.path.join(f"{os.environ['SPRINT']}/sprint/models/human.vocab")
        )  # vocab file for language annotations, not needed for this model but used for consistency
        self.vocab_word["word"].name = "word"
        self.vocab_word["action_low"] = torch.load(f"{os.environ['SPRINT']}/sprint/models/low_level_actions.vocab")[
            "action_low"
        ]  # our custom vocab
        self.policy = AMETIQLPolicy(args, self.vocab_word, pad, seg)

        self.critics = AMETIQLCritics(args, self.vocab_word, pad, seg)
        self.target_critics = copy.deepcopy(self.critics)
        self.training_steps = 0
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.use_amp = args.use_amp
        self.args = args
        self.pad = pad
        self.seg = seg
        self.train_with_advantage = args.train_with_advantage
    
    def accum_gradient_from_batch(
        self,
        frames,
        goal_frame,
        action,
        obj_id,
        lengths_frames,
        interact_mask,
        rewards,
        terminals,
        accum_factor,
        eval,
    ):
        combined_outs = {}
        policy_outs = self.policy.forward(
            vocab=self.vocab_word["word"],
            lang=goal_frame,
            frames=frames,
            lengths_frames=lengths_frames,
            length_frames_max=lengths_frames.max().item(),
            action=action,
        )
        combined_outs.update(policy_outs)
        critic_outs = self.critics.forward(
            vocab=self.vocab_word["word"],
            lang=goal_frame,
            frames=frames,
            lengths_frames=lengths_frames,
            length_frames_max=lengths_frames.max().item(),
            action=action,
            forward_v=True,
            object=obj_id,
        )
        combined_outs.update(critic_outs)
        with torch.no_grad():
            target_critic_outs = self.target_critics.forward(
                vocab=self.vocab_word["word"],
                lang=goal_frame,
                frames=frames,
                lengths_frames=lengths_frames,
                length_frames_max=lengths_frames.max().item(),
                action=action,
                forward_v=False,
                object=obj_id,
            )
        gt_dict = {
            "action": action[:, :-1],
            "object": obj_id[:, :-1],
            "action_valid_interact": interact_mask,
        }

        metrics = collections.defaultdict(list)
        # compute losses
        combined_outs.update({"target_" + k: v for k, v in target_critic_outs.items()})
        target_q_pred = torch.min(
            combined_outs["target_q1_val"], combined_outs["target_q2_val"]
        ).detach()
        combined_outs["target_q_pred"] = target_q_pred
        # advantage weights for IQL policy training
        advantage = target_q_pred - combined_outs["v_val"].detach()

        exp_advantage = torch.exp(advantage * self.args.advantage_temp)
        if self.args.clip_score is not None:
            exp_advantage = torch.clamp(exp_advantage, max=self.args.clip_score)

        if self.train_with_advantage:
            weights = exp_advantage.detach().squeeze(-1)
        else:
            weights = torch.ones_like(exp_advantage).detach().squeeze(-1)
        self.training_steps += 1
        policy_losses = self.policy.compute_batch_loss(combined_outs, weights, gt_dict)

        critic_losses = self.critics.compute_batch_loss(
            combined_outs, rewards, terminals
        )
        for key, value in policy_losses.items():
            policy_losses[key] = value * accum_factor
        for key, value in critic_losses.items():
            critic_losses[key] = value * accum_factor
        metrics.update(policy_losses)
        metrics.update(critic_losses)
        metrics["exp_advantage"] = exp_advantage.mean().item()
        metrics["exp_advantage_max"] = exp_advantage.max().item()
        metrics["exp_advantage_min"] = exp_advantage.min().item()

        # compute metrics
        self.policy.compute_metrics(
            combined_outs, gt_dict, metrics,
        )
        self.critics.compute_metrics(
            critic_outs, gt_dict, metrics,
        )
        for key, value in metrics.items():
            if type(value) is torch.Tensor:
                metrics[key] = torch.mean(value).item()
            else:
                metrics[key] = np.mean(value)
        for key, value in combined_outs.items():
            if value is not None:
                metrics[key] = value.mean().item()
        losses = {}
        losses.update(policy_losses)
        losses.update(critic_losses)
        return metrics, losses

    def train_offline_from_batch(
        self,
        frames,
        lang, # not used for image goal conditioned model 
        action,
        obj_id,
        lengths_frames,
        lengths_lang, # not used for image goal conditioned model
        interact_mask,
        rewards,
        terminals,
        state_goal=None,  # for AM and VF queries on terminal state
        state_text_instructions_list=None,  # for cross-traj chaining (for sprint only)
        state_text_summaries=None,  # for cross-traj chaining (for sprint only)
        state_text_summaries_list=None,  # for cross-traj chaining (for sprint only)
        lengths_state_text_summaries=None,  # for cross-traj chaining (for sprint only)
        eval=False,
    ):
        ret_dicts = []
        accum_factor = 0.5 # performing 2 updates, each with equal weights
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # in trajectory
            ret_dicts.append(
                self.accum_gradient_from_batch(
                    frames,
                    state_goal,
                    action,
                    obj_id,
                    lengths_frames,
                    interact_mask.view(-1),
                    rewards,
                    terminals,
                    accum_factor,
                    eval,
                )
            )

            # cross trajectory chaining for actionable models 
            if not eval:
                chain_frames_lengths = []
                chain_terminals = terminals.clone()
                chain_actions = action.clone()
                chain_obj_ids = obj_id.clone()
                chain_interact_mask = interact_mask.clone()
                for i in range(lengths_frames.shape[0]):
                    length = torch.randint(
                        2, lengths_frames[i] + 1, (1,), device=lengths_frames.device
                    )
                    # we're randomly sampling lengths and then setting those as terminal
                    # - 2 because the last frame is the next obs, and the index is 0-based
                    chain_terminals[i, length - 2] = 1
                    # set all the terminals in front of that to the terminal padding value
                    chain_terminals[i, length - 1 :] = -1
                    # set all the actions in front of that to the action padding value
                    chain_actions[i, length - 1 :] = self.pad
                    # set all the obj_ids in front of that to the obj_id padding value
                    chain_obj_ids[i, length - 1 :] = self.pad
                    # set all the interact_mask in front of that to the interact_mask padding value
                    chain_interact_mask[i, length - 1 :] = self.pad
                    chain_frames_lengths.append(length)
                chain_frames_lengths = torch.cat(chain_frames_lengths)
                max_length = torch.max(chain_frames_lengths).item()
                chain_terminals = chain_terminals[:, : max_length - 1]
                chain_frames = frames[:, :max_length]
                chain_actions = chain_actions[:, :max_length]
                chain_obj_ids = chain_obj_ids[:, :max_length]
                chain_interact_mask = chain_interact_mask[:, : max_length - 1]

                cross_traj_goals = torch.roll(
                    state_goal, -1, 0
                )  # random matching by shifting
                with torch.no_grad():
                    prev_training = self.training
                    self.eval()
                    chain_q_1, chain_q_2 = self.target_critics.get_sequential_q(
                        # chain_rewards = self.critics.get_sequential_v(
                        vocab=self.critics.vocab_word["word"],
                        lang=cross_traj_goals,
                        frames=chain_frames,
                        lengths_frames=chain_frames_lengths,
                        length_frames_max=max_length,
                        action=chain_actions,
                        object=chain_obj_ids,
                    )

                    chain_rewards = torch.minimum(chain_q_1, chain_q_2)
                    chain_rewards = torch.clamp(
                        chain_rewards * chain_terminals, min=0, max=1
                    )  # to match with Actionable Models: label only last state
                    self.train(prev_training)
                ret_dicts.append(
                    self.accum_gradient_from_batch(
                        chain_frames,
                        cross_traj_goals,
                        chain_actions,
                        chain_obj_ids,
                        chain_frames_lengths,
                        chain_interact_mask.reshape(-1),
                        chain_rewards,
                        chain_terminals,
                        accum_factor,
                        eval,
                    )
                )

        merged_ret_dict = {}
        for k in ret_dicts[0][0].keys():
            if "loss" in k:
                merged_ret_dict[k] = np.sum([ret_dict[0][k] for ret_dict in ret_dicts])
            else:
                merged_ret_dict[k] = np.mean([ret_dict[0][k] for ret_dict in ret_dicts])
        merged_loss_dict = {}
        for k in ret_dicts[0][1].keys():
            merged_loss_dict[k] = sum([ret_dict[1][k] for ret_dict in ret_dicts])

        if not eval:
            self.policy.perform_model_update(merged_loss_dict, self.grad_scaler)
            self.critics.perform_model_update(merged_loss_dict, self.grad_scaler)
            self.grad_scaler.update()
            # soft target adjustment
            soft_update_from_to(
                self.critics, self.target_critics, self.args.soft_target_tau
            )
        return merged_ret_dict

    def step(self, input_dict, ret_value):
        """
        forward the model for a single time-step (used for real-time execution during eval)
        """
        input_dict["language_ann"] = input_dict["state_goal"]
        return super().step(input_dict, ret_value)
