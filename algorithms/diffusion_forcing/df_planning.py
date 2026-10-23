import collections
import json
import os
import pickle
import random as random_module
from random import random
from typing import Optional, Any

from omegaconf import DictConfig
import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from PIL import Image

from .df_base import DiffusionForcingBase
from utils.logging_utils import (
    make_random_start_plans_images,
    make_trajectory_images,
    make_trajectory_images_with_nodes,
)
from .high_level_tree_node import HighLevelTreeNode
from .low_level_tree_node import LowLevelTreeNode
from utils.controller_assets import antmaze_controller_dir
from utils.graph_routes import shortest_simple_routes

SUB_GOAL_INTERVAL = 10  # plan steps between DQL controller subgoals
OVERRUN_BONUS = 0.05  # above accumulated novelty (~0.01 per level), below most in-budget solutions

class DiffusionForcingPlanning(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        self.env_id = cfg.env_id
        self.dataset = cfg.dataset
        self.action_dim = len(cfg.action_mean)
        self.observation_dim = len(cfg.observation_mean)
        self.use_reward = cfg.use_reward
        self.unstacked_dim = self.observation_dim + self.action_dim + int(self.use_reward)
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len
        self.n_tokens = self.episode_len // cfg.frame_stack + 1
        self.gamma = cfg.gamma
        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.observation_mean = np.array(cfg.observation_mean[: self.observation_dim])
        self.observation_std = np.array(cfg.observation_std[: self.observation_dim])
        self.action_mean = np.array(cfg.action_mean[: self.action_dim])
        self.action_std = np.array(cfg.action_std[: self.action_dim])
        self.open_loop_horizon = cfg.open_loop_horizon
        self.padding_mode = cfg.padding_mode
        self.interaction_seed = cfg.interaction_seed
        self.task_id = cfg.task_id
        self.val_max_steps = cfg.val_max_steps
        configured_plan_max = cfg.get("plan_length_max", None)
        self.has_explicit_plan_length_max = configured_plan_max is not None
        self.plan_length_max = (min(self.val_max_steps - 1, int(configured_plan_max))
                                if self.has_explicit_plan_length_max else self.val_max_steps)
        if self.plan_length_max < 1:
            raise ValueError("plan_length_max must be at least 1")
        self.threshold_for_warp = cfg.threshold_for_warp
        self.low_level_search = cfg.low_level_search
        self.simulation_low_level_search = cfg.get("simulation_low_level_search", "best-of-n")
        if self.simulation_low_level_search not in ("best-of-n", "mctd"):
            raise ValueError("simulation_low_level_search should be best-of-n or mctd")
        self.mctd_max_search_num = cfg.mctd_max_search_num
        self.num_samples_best_of_n = cfg.num_samples_best_of_n
        self.high_level_search = cfg.high_level_search
        self.num_graph_nodes = cfg.num_graph_nodes
        self.graph_max_steps = cfg.graph_max_steps
        self.graph_max_search_num = cfg.graph_max_search_num
        self.parts_mctd_max_steps = cfg.parts_mctd_max_steps
        # One guidance set per child of a tree node; the list length is the number of children per node.
        self.children_guidance_sets = [list(g) for g in cfg.children_guidance_sets]
        self.cmctd_threshold_for_reaching = cfg.cmctd_threshold_for_reaching
        self.composer_plan_rank = int(cfg.get("composer_plan_rank", 1))
        self.parallel_search_num = int(cfg.get("parallel_search_num", 1))
        # parallel_num sets the total parallelism: the composer (outer tree) then selects
        # parallel_num / mctd_node_parallel_num nodes per iteration (the MCTD node planner's own
        # parallelism is mctd_node_parallel_num; best-of-N counts as 1).
        total_parallel = cfg.get("parallel_num", None)
        if total_parallel is not None:
            node_parallel = int(cfg.get("mctd_node_parallel_num", 1)) if self.low_level_search == "mctd" else 1
            if int(total_parallel) % node_parallel != 0:
                raise ValueError("parallel_num must be a multiple of mctd_node_parallel_num")
            self.parallel_search_num = int(total_parallel) // node_parallel
        self.virtual_visit_weight = float(cfg.get("virtual_visit_weight", 1.0))
        if self.parallel_search_num < 1:
            raise ValueError("parallel_search_num must be at least 1")
        self.compress_still_steps = bool(cfg.get("compress_still_steps", False))
        self.still_step_threshold = float(cfg.get("still_step_threshold", 0.03))
        self.simulation_overrun_ratio = float(cfg.get("simulation_overrun_ratio", 0.0))
        self.simulation_jumpy_interval = int(cfg.get("simulation_jumpy_interval", 1))
        self.early_stopping_condition = cfg.get("early_stopping_condition", "achieved")
        if self.early_stopping_condition not in ("achieved", "solved"):
            raise ValueError("early_stopping_condition must be 'achieved' or 'solved'")
        self.novelty_radius = float(cfg.get("novelty_radius", 1.0))
        self.novelty_recent_arclength = float(cfg.get("novelty_recent_arclength", 2.0))
        if self.composer_plan_rank < 1:
            raise ValueError("composer_plan_rank must be at least 1")
        super().__init__(cfg)
        self.plot_end_points = cfg.plot_start_goal and self.guidance_scale != 0

    def _build_model(self):
        mean = list(self.observation_mean) + list(self.action_mean)
        std = list(self.observation_std) + list(self.action_std)
        if self.use_reward:
            mean += [self.reward_mean]
            std += [self.reward_std]
        self.cfg.data_mean = np.array(mean).tolist()
        self.cfg.data_std = np.array(std).tolist()
        super()._build_model()

    def _preprocess_batch(self, batch):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]

        observations = observations[..., : self.observation_dim]
        actions = actions[..., : self.action_dim]

        if (n_frames - 1) % self.frame_stack != 0:
            raise ValueError("Number of frames - 1 must be divisible by frame stack size")

        nonterminals = torch.cat([torch.ones_like(nonterminals[:, : self.frame_stack]), nonterminals[:, :-1]], dim=1)
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1)
        bundles = self._normalize_x(self.make_bundle(observations, actions, rewards))  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = 0  # zero out actions and rewards after normalization
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b t c)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = None

        return bundles, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

        n_tokens, batch_size = xs.shape[:2]

        weights = masks.float()
        if not self.causal:
            # manually mask out entries to train for varying length
            random_terminal = torch.randint(2, n_tokens + 1, (batch_size,), device=self.device)
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[:, :n_tokens].bool()
            random_terminal = repeat(random_terminal, "b t -> (t fs) b", fs=self.frame_stack)
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs, masks=masks))

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            o, a, r = self.split_bundle(xs_pred)
            trajectory = o.detach().cpu().numpy()[:-1, :8]  # last observation is dummy, sample 8
            images = make_trajectory_images(self.env_id, trajectory, trajectory.shape[1], None, None, False)
            for i, img in enumerate(images):
                self.log_image(
                    f"training_visualization/sample_{i}",
                    Image.fromarray(img),
                )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        horizon = self.episode_len
        # Check the configuration is runnable
        assert self.low_level_search in ("best-of-n", "mctd"), "low_level_search should be best-of-n or mctd"
        search = self.high_level_search
        if search == "online-composer":
            self.online_composer(batch_size, conditions, horizon, namespace)
        elif search == "distributed-composer":
            self.distributed_composer(batch_size, conditions, horizon, namespace)
        elif search == "preplan-composer":
            self.preplan_composer(batch_size, conditions, horizon, namespace)
        elif search == "replay-plan":
            self.replay_plan(namespace)
        elif search == "replan":
            self.replan(conditions, horizon, namespace)
        elif search == "preplan-edges":
            self.preplan_edges(conditions, horizon, namespace)
        else:
            raise ValueError(f"Unknown high_level_search: {self.high_level_search}")

    def plan(self, start: torch.Tensor, goal: torch.Tensor, horizon: int, conditions: Optional[Any] = None,
        guidance_scale: Optional[Any] = None, noise_level: Optional[torch.Tensor] = None, plan: Optional[torch.Tensor] = None):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan

        batch_size = start.shape[0]

        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                dist = nn.functional.mse_loss(pred, target, reduction="none")  # (t fs) b c

                # guidance weight for observation and action
                weight = np.array(
                    [20] * (self.frame_stack)  # conditoning (aka reconstruction guidance)

                    + [1 for _ in range(horizon)]  # try to reach the goal at any horizon
                    #+ [0 for _ in range(horizon-1)] + [1]  # Diffuer guidance
                    + [0] * (h_padded - horizon)  # don't guide padded entries due to horizon % frame_stack != 0
                )
                # mathematically, one may also try multiplying weight by sqrt(alpha_cum)
                # this means you put higher weight to less noisy terms
                # which might be better but we haven't tried yet
                weight = torch.from_numpy(weight).float().to(self.device)
                
                dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                dist_o = dist_o[:, :, : 2]
                dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=1).sqrt()
                dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = dist_o
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]

                if type(guidance_scale) == torch.Tensor:
                    episode_return = -(dist * weight * guidance_scale.unsqueeze(0).unsqueeze(-1)).mean() * 1000 * dist.shape[1] / 16
                else:
                    episode_return = -(dist * weight).mean() * 1000 * dist.shape[1] / 16 # considering the batch size
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError("reward guidance is not supported")

            if type(guidance_scale) == torch.Tensor:
                return episode_return
            else:
                return guidance_scale * episode_return

        if type(guidance_scale) == torch.Tensor:
            guidance_fn = goal_guidance
        else:
            guidance_fn = goal_guidance if guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        if noise_level is None:
            scheduling_matrix = self._generate_scheduling_matrix(plan_tokens)
        else: # if noise_level is given, use it
            scheduling_matrix = noise_level
        if plan is None:
            chunk = torch.randn((plan_tokens, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        else: # if plan is given, use it
            chunk = plan
            chunk = rearrange(chunk, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)
        pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0)

        plan_hist = [plan.detach().clone()[: self.n_tokens - pad_tokens]]
        stabilization = 0
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            to_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m + 1],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
            from_noise_levels = repeat(from_noise_levels, "t -> t b", b=batch_size)
            to_noise_levels = repeat(to_noise_levels, "t -> t b", b=batch_size)
            plan[1 : self.n_tokens - pad_tokens] = self.diffusion_model.sample_step(
                plan, conditions, from_noise_levels, to_noise_levels, guidance_fn=guidance_fn
            )[1 : self.n_tokens - pad_tokens]
            plan_hist.append(plan.detach().clone()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack)
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]

        return plan_hist

    def _score_candidates(self, o_tensor, plan_histories, starts, goals, horizon):
        """Score every candidate plan (the best-of-N scoring rule).

        o_tensor: (t, m, b, c) unnormalized candidate observations. Returns (m, b) arrays of the
        values, the statuses (Warp / Achieved / Close / NotReached) and the first goal-reaching steps.
        Candidates are checked in time order: a jump larger than threshold_for_warp (from the start for
        the first point) rejects the candidate as Warp, reaching the goal scores (horizon - t) / horizon,
        and an unreached candidate is scored by its novel arclength. The checks are batched per item (on the CPU)."""
        num_samples, T = o_tensor.shape[1], o_tensor.shape[0]
        obs = o_tensor.detach().float().cpu()  # many small ops per item: faster on the CPU than on the GPU
        dev = obs.device
        xy = obs[..., :2]
        start_xy = torch.as_tensor(np.asarray(starts)[:, :2], dtype=torch.float32, device=dev)
        goal_xy = torch.as_tensor(np.asarray(goals)[:, :2], dtype=torch.float32, device=dev)
        prev = torch.cat([start_xy[None, None].expand(1, num_samples, -1, -1), xy[:-1]], dim=0)
        warp = (xy - prev).norm(dim=-1) > self.threshold_for_warp  # (t, m, b)
        reach = (xy - goal_xy[None, None]).norm(dim=-1) < 0.5

        def first(mask):  # first time index where mask holds, T if never
            return torch.where(mask.any(0), mask.float().argmax(0), torch.full_like(mask[0], T, dtype=torch.long))

        first_warp, first_reach = first(warp), first(reach)  # (m, b)
        is_warp = (first_warp < T) & (first_warp <= first_reach)  # the warp check comes first at a step
        is_reached = first_reach < first_warp
        values = torch.where(is_reached, (horizon - first_reach).float() / horizon, torch.zeros_like(xy[0, ..., 0]))
        is_close = torch.zeros_like(is_warp)
        steps_all = (xy[1:] - xy[:-1]).norm(dim=-1)  # (t-1, m, b)
        arclength_all = torch.cat([torch.zeros_like(steps_all[:1]), steps_all.cumsum(0)], dim=0)  # (t, m, b)
        dist = lambda x, y: torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
        for i in range(obs.shape[2]):
            unreached = torch.nonzero(~(is_warp[:, i] | is_reached[:, i])).flatten()
            if len(unreached) == 0:
                continue
            cand = obs[:, unreached, i].transpose(0, 1)  # (n, t, c)
            cand_xy = cand[..., :2]
            if len(plan_histories[i]) != 0:
                history = torch.as_tensor(np.asarray(plan_histories[i]), dtype=torch.float32)
                # Label revisits of the branch history (diagnostic only; not rejected)
                to_history = dist(cand, history[None, :, :cand.shape[-1]].expand(len(unreached), -1, -1))
                is_close[unreached, i] = (to_history.min(dim=-1).values[:, 20:] < 1.0).any(dim=1)
                memory = history[:, :2]
            else:
                memory = start_xy[i:i + 1]
            # Novel arclength (see _novel_arclength), batched over the unreached candidates
            to_memory = dist(cand_xy, memory[None].expand(len(unreached), -1, -1)).min(dim=-1).values  # (n, t)
            arclength = arclength_all[:, unreached, i].T  # (n, t)
            earlier = arclength[:, None, :] <= arclength[:, :, None] - self.novelty_recent_arclength
            to_self = torch.where(earlier, dist(cand_xy, cand_xy), torch.full_like(earlier, float("inf"), dtype=torch.float32))
            novel = torch.minimum(to_memory, to_self.min(dim=-1).values) >= self.novelty_radius
            novel_length = (steps_all[:, unreached, i].T * novel[:, 1:]).sum(dim=1)
            # 0.001 per unit of new ground for short segments, saturating smoothly below 0.01
            values[unreached, i] = 0.01 * (1 - torch.exp(-novel_length / 10))
        values = values.cpu().numpy().astype(np.float64)
        is_warp, is_reached, is_close = is_warp.cpu().numpy(), is_reached.cpu().numpy(), is_close.cpu().numpy()
        first_reach = first_reach.cpu().numpy()
        status = np.where(is_warp, "Warp", np.where(is_reached, "Achieved", np.where(is_close, "Close", "NotReached")))
        return values, status, first_reach


    def _finalize_candidate(self, plan, status, value, first_reach, history_len, max_steps):
        """Cut an Achieved plan at the goal and rescore it on the branch's step budget; compress still steps."""
        if status == "Achieved":
            plan = plan[:int(first_reach) + 1]
            if history_len != 0:
                value = (max_steps - (history_len + int(first_reach))) / max_steps
        if self.compress_still_steps:
            plan = self._drop_still_steps(plan)
            if status == "Achieved":
                value = (max_steps - (history_len + len(plan) - 1)) / max_steps
        return plan, value

    def _select_candidates(self, o_tensor, plan_histories, starts, goals, horizon, max_steps, sample_guidance, rank=1):
        """Score candidate plans and pick one per item (the best-of-N selection rule).

        o_tensor: (t, m, b, c) unnormalized candidate observations; sample_guidance: (m, b). With rank k > 1 and
        candidates that reach the goal, the k-th best of those (the longest if fewer) is picked instead."""
        values, status, first_reach = self._score_candidates(o_tensor, plan_histories, starts, goals, horizon)
        traj_list, info_debugging_list, solved_list, value_list = [], [], [], []
        for i in range(o_tensor.shape[2]):
            _values = values[:, i]
            _info_for_debugging = status[:, i].tolist()
            # Zero is also a legitimate score for a valid, unreached candidate.
            # Exclude Warp (discontinuous) candidates explicitly instead of relying on scores.
            # Close (revisiting) candidates stay selectable at zero value: waiting or
            # backtracking segments are valid, and rejecting them can exhaust the tree.
            valid_indices = [s for s, st in enumerate(_info_for_debugging)
                             if st != "Warp"]
            if not valid_indices:
                traj_list.append(o_tensor[:0, 0, i])
                info_debugging_list.append("NoValidCandidate")
                value_list.append(0.0)
                solved_list.append(False)
                continue
            # Ties (e.g. every candidate revisits) go to the stronger goal guidance, then the lower index.
            _best_idx = max(valid_indices, key=lambda k: (_values[k], sample_guidance[k, i], -k))
            reached = [k for k in valid_indices if _info_for_debugging[k] == "Achieved"]
            if rank > 1 and reached:
                reached.sort(key=lambda k: (_values[k], sample_guidance[k, i], -k), reverse=True)
                _best_idx = reached[min(rank, len(reached)) - 1]
            _o_tensor_best, value = self._finalize_candidate(
                o_tensor[:, _best_idx, i], _info_for_debugging[_best_idx], _values[_best_idx],
                first_reach[_best_idx, i], len(plan_histories[i]), max_steps)
            traj_list.append(_o_tensor_best)
            info_debugging_list.append(_info_for_debugging[_best_idx])
            solved_list.append(_info_for_debugging[_best_idx] == "Achieved")
            value_list.append(value)

        return traj_list, info_debugging_list, value_list, solved_list

    def best_of_n(self, plan_histories, starts, goals, norm_starts, norm_goals, horizon, conditions, max_steps, guidance_scales=None,
                  noise_level=None):
        for i in range(len(plan_histories)):
            if len(plan_histories[i]) != 0 and not isinstance(plan_histories[i], np.ndarray):
                plan_histories[i] = torch.cat([plan.detach().cpu() for plan in plan_histories[i]], dim=0)
                plan_histories[i] = plan_histories[i].detach().cpu().numpy()
        if guidance_scales is None:
            guidance_scales = [[0, 0.1, 0.5, 1, 2] for _ in range(len(plan_histories))] # default guidance scales
        for i in range(len(guidance_scales)):
            if len(guidance_scales[i]) == 0:
                guidance_scales[i] = [0, 0.1, 0.5, 1, 2] # default guidance scales

        num_samples = self.num_samples_best_of_n
        rs_norm_starts = norm_starts[None].repeat(num_samples, 1, 1) # (m b c)
        rs_norm_goals = norm_goals[None].repeat(num_samples, 1, 1) # (m b c)
        rs_guidance_scales = []
        for i in range(num_samples):
            _rs_guidance_scales = []
            for j in range(len(guidance_scales)):
                _rs_guidance_scales.append(guidance_scales[j][i % len(guidance_scales[j])])
            rs_guidance_scales.append(_rs_guidance_scales)
        rs_guidance_scales = torch.tensor(rs_guidance_scales).to(self.device)
        rs_norm_starts = rs_norm_starts.reshape(-1, rs_norm_starts.shape[-1]) # (m*b c)
        rs_norm_goals = rs_norm_goals.reshape(-1, rs_norm_goals.shape[-1]) # (m*b c)
        rs_guidance_scales = rs_guidance_scales.reshape(-1) # (m*b)
        sample_guidance = rs_guidance_scales.reshape(num_samples, -1).detach().cpu().numpy() # (m b)
        rs_plan_hist = self.plan(rs_norm_starts, rs_norm_goals, self.episode_len, conditions, guidance_scale=rs_guidance_scales,
                                 noise_level=noise_level)
        rs_plan_hist = self._unnormalize_x(rs_plan_hist)  # (m t b c)
        rs_plans = rs_plan_hist[-1]  # (t b c)
        o_tensor, _, _ = self.split_bundle(rs_plans)
        o_tensor = o_tensor[:-1] # last observation is dummy
        o_tensor = o_tensor.reshape(o_tensor.shape[0], num_samples, -1, o_tensor.shape[-1]) # (t m b c)
        traj_list, info_debugging_list, value_list, solved_list = self._select_candidates(
            o_tensor, plan_histories, starts, goals, horizon, max_steps, sample_guidance)

        return traj_list, info_debugging_list, value_list, solved_list

    def _drop_still_steps(self, plan):
        """Drop points that move less than still_step_threshold from the last kept point.

        The first and last points are kept, so segments still join exactly; the route
        is unchanged while waiting no longer spends the plan-length budget."""
        if len(plan) <= 2:
            return plan
        xy = plan[:, :2].detach().cpu().numpy()
        keep, last = [0], xy[0]
        for t in range(1, len(xy) - 1):
            if np.linalg.norm(xy[t] - last) >= self.still_step_threshold:
                keep.append(t)
                last = xy[t]
        keep.append(len(xy) - 1)
        return plan[keep]

    def _novel_arclength(self, plan, memory):
        """Path length of the plan steps that land outside the already visited region.

        A point is visited if it lies within ``novelty_radius`` of the branch history
        (``memory``) or of the plan's own points at least ``novelty_recent_arclength``
        earlier along the plan. Waiting, backtracking and loops therefore earn nothing,
        and a dead end is rewarded only the first time it is entered.
        """
        xy = np.asarray(plan)[:, :2]
        memory = np.asarray(memory)[:, :2]
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arclength = np.concatenate([[0.], np.cumsum(steps)])
        to_memory = np.linalg.norm(xy[:, None] - memory[None], axis=-1).min(axis=1)
        to_self = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        earlier = arclength[None, :] <= arclength[:, None] - self.novelty_recent_arclength
        to_self = np.where(earlier, to_self, np.inf).min(axis=1)
        novel = np.minimum(to_memory, to_self) >= self.novelty_radius
        return float((steps * novel[1:]).sum())

    def mctd_segments(self, plan_histories, starts, goals, norm_starts, norm_goals, horizon, conditions, max_steps,
                      guidance_sets, settings=None):
        """Node planner that searches the denoising process of each segment (MCTD) instead of best-of-N.

        One inner tree per item; the inner tree partially denoises the segment ``mctd_node_denoising_steps``
        levels per depth, and an inner node has ``mctd_node_child_repeat`` child slots per guidance value of
        the item. The inner search is Parallel MCTD (Fast-MCTD, Sec. 4.1), like the outer tree: every iteration each unfinished
        tree claims up to ``mctd_node_parallel_num`` child slots by UCT with redundancy-aware selection
        (``virtual_visit_weight``), all claimed children of all trees are denoised together (one diffusion
        batch per inner depth), completed with jumpy denoising (every ``mctd_node_jumpy_interval``-th
        level) and scored with the best-of-N rule, and the trees are updated afterwards (max backup).
        Each child spends one unit of the per-tree budget ``mctd_node_search_num``; a tree stops when a
        child reaches the goal. Each segment is the best of its tree's completed plans. ``settings``
        overrides these mctd_node_* values (entries that are None keep the configured value)."""
        settings = settings or {}
        opt = lambda key, default: settings[key] if settings.get(key) is not None else self.cfg.get(key, default)
        steps_per_depth = int(opt("mctd_node_denoising_steps", 20))
        jumpy = int(opt("mctd_node_jumpy_interval", 10))
        budget = int(opt("mctd_node_search_num", 4))
        exp_weight = float(opt("mctd_node_exp_weight", 0.1))
        parallel = max(1, int(opt("mctd_node_parallel_num", 1)))
        # Child slots per guidance value: independent samples of the same guidance at every inner node.
        child_repeat = max(1, int(opt("mctd_node_child_repeat", 1)))
        max_batch = opt("mctd_node_max_batch", None)  # None: one diffusion batch per inner depth
        max_batch = None if max_batch is None else int(max_batch)
        virtual_weight = self.virtual_visit_weight
        n_items = len(plan_histories)
        guidance_sets = [list(g) if len(g) else [0, 0.1, 0.5, 1, 2] for g in guidance_sets]
        plan_histories = list(plan_histories)
        for i in range(n_items):
            if len(plan_histories[i]) != 0 and not isinstance(plan_histories[i], np.ndarray):
                plan_histories[i] = torch.cat([plan.detach().cpu() for plan in plan_histories[i]], dim=0).numpy()
        schedule = self._generate_scheduling_matrix(int(np.ceil(self.episode_len / self.frame_stack)))
        last = schedule.shape[0] - 1
        terminal_depth = int(np.ceil(last / steps_per_depth))

        def row(depth):  # schedule row an inner node at this depth has been denoised to
            return min(depth * steps_per_depth, last)

        def chunked_plan(b_starts, b_goals, gs, noise_level, plan):
            """Final denoised plans (t, B, c), in diffusion batches of at most mctd_node_max_batch items."""
            size = max_batch or len(b_starts)
            out = []
            for k in range(0, len(b_starts), size):
                out.append(self.plan(b_starts[k:k + size], b_goals[k:k + size], self.episode_len, conditions,
                                     guidance_scale=gs[k:k + size], noise_level=noise_level,
                                     plan=None if plan is None else plan[:, k:k + size])[-1])
            return torch.cat(out, dim=1)

        # Every inner node's children take the guidance values (guidance may change between denoising
        # stages); the value order repeats: g0, g1, .., g0, g1, ..
        roots = [LowLevelTreeNode(0, list(guidance_sets[i]) * child_repeat) for i in range(n_items)]
        completed = [[] for _ in range(n_items)]
        completed_guidance = [[] for _ in range(n_items)]
        spent = [0] * n_items
        done = [False] * n_items

        while not all(done):
            # Selection: each unfinished tree claims up to `parallel` child slots
            claims = []  # (item, parent, child)
            for i in range(n_items):
                if done[i]:
                    continue
                for _ in range(min(parallel, budget - spent[i])):
                    parent = roots[i].select(terminal_depth, exp_weight, virtual_weight)
                    if parent is None:
                        break
                    claims.append((i, parent, parent.claim_child()))
                if not any(c[0] == i for c in claims):
                    done[i] = True  # tree fully expanded or budget spent
            if not claims:
                break
            # Expansion + rollout, one diffusion batch per inner depth
            for d in sorted({child.depth for _, _, child in claims}):
                group = [c for c in claims if c[2].depth == d]
                idx = [i for i, _, _ in group]
                gs = torch.tensor([child.guidance for _, _, child in group], dtype=torch.float32, device=self.device)
                parent_plan = None if d == 1 else torch.cat([parent.plan for _, parent, _ in group], dim=1)
                b_starts, b_goals = norm_starts[idx], norm_goals[idx]
                child_plans = chunked_plan(b_starts, b_goals, gs, schedule[row(d - 1):row(d) + 1], parent_plan)
                # child_plans: (t, B, c), normalized, denoised to row(d)
                if row(d) < last:
                    rollout_levels = np.concatenate([schedule[row(d)::jumpy], schedule[-1:]], axis=0)
                    full = chunked_plan(b_starts, b_goals, gs, rollout_levels, child_plans)
                else:
                    full = child_plans
                o_tensor, _, _ = self.split_bundle(self._unnormalize_x(full))
                o_tensor = o_tensor[:-1]  # last observation is dummy
                # Score each child on its own (as a single best-of-N candidate), one scoring call per item
                columns = collections.defaultdict(list)
                for col, (i, _, _) in enumerate(group):
                    columns[i].append(col)
                for i, cols in columns.items():
                    values, status, first_reach = self._score_candidates(
                        o_tensor[:, cols, None], [plan_histories[i]], starts[i:i + 1], goals[i:i + 1], horizon)
                    for n, col in enumerate(cols):
                        child = group[col][2]
                        completed[i].append(o_tensor[:, col]); completed_guidance[i].append(child.guidance)
                        child.plan = child_plans[:, col:col + 1]
                        if status[n, 0] == "Warp":
                            child.info, child.score = "NoValidCandidate", 0.0
                            continue
                        _, value = self._finalize_candidate(o_tensor[:, col], status[n, 0], values[n, 0],
                                                            first_reach[n, 0], len(plan_histories[i]), max_steps)
                        child.info, child.score = str(status[n, 0]), float(value)
            # Delayed tree update (max backup)
            for i, _, child in claims:
                child.backpropagate()
                spent[i] += 1
                if child.info == "Achieved":
                    done[i] = True

        cands = [torch.stack(completed[i], dim=1)[:, :, None] for i in range(n_items)]  # (t, M, 1, c)
        out = ([], [], [], [])
        for i in range(n_items):
            res = self._select_candidates(cands[i], [plan_histories[i]], starts[i:i + 1], goals[i:i + 1], horizon,
                                          max_steps, np.array(completed_guidance[i], dtype=np.float32)[:, None],
                                          rank=int(settings.get("select_rank") or 1))
            for lst, v in zip(out, res):
                lst.append(v[0])
        return out

    def expand_via_stitching(self, nodes, starts, goals, horizon, conditions, max_steps):
        """Expansion (Alg. 1): stitch one new plan onto each expanded node (all nodes in one batch)."""
        if not nodes:
            return [], [], [], []
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        guidance_scales, current_steps, plan_histories, names = [], [], [], []
        for node in nodes:
            guidance_scales.append(node.guidance_set)
            current_steps.append(node.plan_length() if self.compress_still_steps
                                 else node.depth * self.episode_len - self.episode_len)
            plan_histories.append(node.plan_history.copy())
            names.append(node.name)
        plans = [[] for _ in range(len(nodes))]
        total_values = [0 for _ in range(len(nodes))]
        normalized_starts = ((torch.from_numpy(starts).to(self.device) - obs_mean[None]) / obs_std[None])
        normalized_goals = ((torch.from_numpy(goals).to(self.device) - obs_mean[None]) / obs_std[None])
        if self.low_level_search == "best-of-n":
            _plan_list, _local_name_list, _local_value_list, _solved_list = self.best_of_n(
                plan_histories.copy(), starts, goals, normalized_starts, normalized_goals, horizon, conditions, max_steps, guidance_scales
            )
        elif self.low_level_search == "mctd":
            _plan_list, _local_name_list, _local_value_list, _solved_list = self.mctd_segments(
                [h.copy() for h in plan_histories], starts, goals, normalized_starts, normalized_goals, horizon, conditions,
                max_steps, guidance_scales
            )
        else:
            raise ValueError("Unknown low level search method")
        for i in range(len(nodes)):
            plans[i] = _plan_list[i]
            plan_histories[i].append(_plan_list[i])
            total_values[i] += _local_value_list[i]
            names[i] += "-" + _local_name_list[i]
        solveds = _solved_list
        for i in range(len(nodes)):
            if solveds[i]:
                success_steps = current_steps[i] + len(_plan_list[i])
                total_values[i] = (max_steps - success_steps) / max_steps
        return total_values, names, plans, solveds

    def _jumpy_noise_level(self):
        """Denoising schedule that keeps every simulation_jumpy_interval-th level (and the last one)."""
        schedule = self._generate_scheduling_matrix(int(np.ceil(self.episode_len / self.frame_stack)))
        rows = list(range(0, schedule.shape[0], self.simulation_jumpy_interval))
        if rows[-1] != schedule.shape[0] - 1:
            rows.append(schedule.shape[0] - 1)
        return schedule[rows]

    def simulate_with_fast_replanning(self, nodes, starts, goals, horizon, conditions, max_steps):
        """Simulation (Alg. 1): complete each node with fast replanning and score it.

        With simulation_jumpy_interval > 1 the replanning uses jumpy denoising (Sec. 3.1). With
        simulation_low_level_search "mctd" each replanning step is the MCTD node planner instead of
        best-of-N (its rollouts use mctd_node_jumpy_interval)."""
        if not nodes:
            return [], [], [], []
        sim_noise_level = self._jumpy_noise_level() if self.simulation_jumpy_interval > 1 else None
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        guidance_scales, current_steps, plan_histories, names, remaining_iters = [], [], [], [], []
        for node in nodes:
            guidance_scales.append(node.guidance_set)
            if self.compress_still_steps:
                # Budget by points; the iteration cap is the usual one (simulation_overrun_ratio extends it).
                current_steps.append(node.plan_length())
                remaining_iters.append(max(0, max_steps // self.episode_len - node.depth))
            else:
                current_steps.append(node.depth * self.episode_len - self.episode_len)
                remaining_iters.append(max_steps // self.episode_len - node.depth)
            plan_histories.append(node.plan_history.copy())
            names.append(node.local_name)
        # Simulation may run past max_steps (compressed mode only): reaching the goal there
        # still marks the branch as leading to the goal, without counting as a solution.
        sim_max_steps = max_steps
        if self.compress_still_steps and self.simulation_overrun_ratio > 0:
            sim_max_steps = int(max_steps * (1 + self.simulation_overrun_ratio))
            remaining_iters = [int(r * (1 + self.simulation_overrun_ratio)) for r in remaining_iters]
        plans = [[] for _ in range(len(nodes))]
        total_values = [0 for _ in range(len(nodes))]
        not_done_node_index_list = list(range(len(nodes)))
        solveds = [False for _ in range(len(nodes))]
        for _iter in range(max(remaining_iters)):

            not_done_node_index_list = [
                i for i in not_done_node_index_list
                if remaining_iters[i] >= _iter + 1 and not solveds[i]
                and (not self.compress_still_steps
                     or current_steps[i] + sum(len(plan) for plan in plans[i]) < sim_max_steps)
            ]

            if len(not_done_node_index_list) == 0:
                break
            
            # Make _starts, _goals, _guidance_scales
            _plan_histories, _starts, _goals, _guidance_scales = [], [], [], []
            for i in range(len(not_done_node_index_list)):
                not_done_index = not_done_node_index_list[i]
                _plan_histories.append(plan_histories[not_done_index].copy())
                if _iter == 0:
                    _starts.append(torch.from_numpy(starts[not_done_index]).to(self.device))
                else:
                    _starts.append(plans[not_done_index][-1][-1])
                _goals.append(torch.from_numpy(goals[not_done_index]).to(self.device))
                _guidance_scales.append(guidance_scales[not_done_index])
            _starts = torch.stack(_starts, dim=0)
            normalized_starts = ((_starts - obs_mean[None]) / obs_std[None])
            _goals = torch.stack(_goals, dim=0)
            normalized_goals = ((_goals - obs_mean[None]) / obs_std[None])
            _starts = _starts.detach().cpu().numpy()
            _goals = _goals.detach().cpu().numpy()

            if self.simulation_low_level_search == "best-of-n":
                _plan_list, _local_name_list, _local_value_list, _solved_list = self.best_of_n(
                    _plan_histories, _starts, _goals, normalized_starts, normalized_goals, horizon, conditions, max_steps, _guidance_scales,
                    noise_level=sim_noise_level
                )
            elif self.simulation_low_level_search == "mctd":
                _plan_list, _local_name_list, _local_value_list, _solved_list = self.mctd_segments(
                    _plan_histories, _starts, _goals, normalized_starts, normalized_goals, horizon, conditions, max_steps,
                    _guidance_scales, settings={key: self.cfg.get(f"simulation_{key}") for key in (
                        "mctd_node_search_num", "mctd_node_parallel_num", "mctd_node_child_repeat")}
                )
            else:
                raise ValueError("Unknown simulation low level search method")

            # Update statistics
            rejected_indices = set()
            for i in range(len(not_done_node_index_list)):
                not_done_index = not_done_node_index_list[i]
                if len(_plan_list[i]) == 0:
                    rejected_indices.add(not_done_index)
                    names[not_done_index] += "-NoValidCandidate"
                    continue
                plans[not_done_index].append(_plan_list[i])
                plan_histories[not_done_index].append(_plan_list[i])
                total_values[not_done_index] += _local_value_list[i]
                names[not_done_index] += "-" + _local_name_list[i]

            # Check the solved nodes
            for i in range(len(not_done_node_index_list)):
                not_done_index = not_done_node_index_list[i]
                if not_done_index in rejected_indices:
                    continue
                if _solved_list[i]:
                    if self.compress_still_steps:
                        success_steps = current_steps[not_done_index] + sum(len(plan) for plan in plans[not_done_index])
                    else:
                        success_steps = current_steps[not_done_index] + len(_plan_list[i]) + int((len(plans[not_done_index])-1)*self.episode_len)
                    if self.compress_still_steps and success_steps > max_steps:
                        # Reached the goal only beyond the budget: stop here; not a solution.
                        rejected_indices.add(not_done_index)
                        names[not_done_index] += "-Overrun"
                        continue
                    solveds[not_done_index] = True
                    total_values[not_done_index] = (max_steps - success_steps) / max_steps
            not_done_node_index_list = [
                i for i in not_done_node_index_list if i not in rejected_indices
            ]
        out_plans = []
        for i in range(len(nodes)):
            if len(plans[i]) != 0:
                out_plans.append(torch.cat(plans[i], dim=0))
            else:
                out_plans.append([])
        return total_values, names, out_plans, solveds

    @staticmethod
    def _waypoint_graph(waypoint_plans):
        """Undirected connectivity graph over [start, centers..., goal] (Distributed Composer).

        waypoint_plans[i][j] is the shortest plan found from tree origin i (node i) to
        waypoint j (node j + 1). Every recorded connection becomes an edge, whichever
        tree found it; when both directions are found, the shorter plan is kept.
        """
        num_nodes = len(waypoint_plans) + 1
        graph_costs = np.zeros((num_nodes, num_nodes)) + np.inf
        graph_paths = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i, plans_from_origin in enumerate(waypoint_plans):
            for j, _plan in enumerate(plans_from_origin):
                a, b = i, j + 1
                if len(_plan) == 0 or a == b:
                    continue
                if len(_plan.shape) == 1:
                    _plan = _plan.reshape(1, -1)
                if len(_plan) < graph_costs[a, b]:
                    graph_costs[a, b] = graph_costs[b, a] = len(_plan)
                    graph_paths[a][b] = _plan
                    graph_paths[b][a] = _plan[::-1]
        return graph_costs, graph_paths

    @staticmethod
    def _retain_valid_expansions(selected_nodes, expanded_nodes, batch_indices,
                                 values, names, plans, solveds):
        """Remove rejected samples before histories, caches or bridges see them."""
        keep = []
        for i, plan in enumerate(plans):
            if len(plan):
                keep.append(i)
            else:
                expanded_nodes[i].set_local_name(names[i])
                expanded_nodes[i].reject()
        return tuple([items[i] for i in keep] for items in (
            selected_nodes, expanded_nodes, batch_indices, values, names, plans, solveds))

    def plan_tree_search(self, starts, goals, horizon, conditions, max_steps=1000, max_search_num=10, early_stopping=False, waypoints=None,
                         parallel_search_num=None, plan_rank=1):
        """Plan-level tree search (Alg. 1), one tree per start/goal pair.

        Used by Online Composer with a single pair. With ``waypoints`` it runs the
        Distributed Composer loop (Alg. 2): trees grow in parallel, connect when a plan
        reaches another origin, and the shortest path over the connectivity graph
        gives the synthesized plan. With ``plan_rank`` k > 1 (without waypoints) a tree keeps searching until it
        has solutions in k branches (subtrees of the nodes at depth plan_rank_branch_depth; a solved branch is
        closed) and returns the longest of them, i.e. the k-th shortest (composer_plan_rank for Online)."""
        # Distributed: complete start-to-goal plans (routes over the connectivity graph and direct plans);
        # the search stops once composer_plan_rank of them are found.
        candidate_count = self.composer_plan_rank if waypoints is not None else 0
        candidates = []
        candidate_keys = set()

        def add_candidate(candidate_plan, route, search_count):
            candidate_plan = np.asarray(candidate_plan)
            if len(candidate_plan) >= self.val_max_steps or len(candidate_plan) > self.plan_length_max:
                return
            key = (candidate_plan.shape, candidate_plan.tobytes())
            if key in candidate_keys:
                return
            candidate_keys.add(key)
            candidates.append((candidate_plan.copy(), tuple(route), search_count))
            print(f"Candidate {len(candidates)} (target {candidate_count}): "
                  f"route={route}, length={len(candidate_plan)}, search={search_count}")

        terminal_depth = max_steps // self.episode_len
        if self.compress_still_steps:
            terminal_depth *= 3  # safety cap; the point budget is checked in select_expandable
        children_guidance_sets = [list(g) for g in self.children_guidance_sets]
        start_root_nodes = [HighLevelTreeNode('0', 0, None, children_guidance_sets, []) for _ in range(len(starts))]
        # Search
        search_num = 0
        best_solved_plans = [[] for _ in range(len(starts))]
        best_lengths = np.zeros(len(starts)) + np.inf
        solved_samples = []
        ranked = plan_rank > 1 and waypoints is None
        # ranked: one solution per branch (the node at depth plan_rank_branch_depth); a solved branch is closed
        # so the search moves to other branches. tree_solutions[b] maps a branch to its (length, plan).
        branch_depth = int(self.cfg.get("plan_rank_branch_depth", 2)) if ranked else None
        tree_solutions = [{} for _ in range(len(starts))]

        def record_solution(b, plan, length, label, node=None):
            """Keep a solution; returns True once tree b has enough solutions to stop."""
            if ranked:
                if length < max_steps and node is not None:
                    branch = node
                    while branch.depth > branch_depth:
                        branch = branch._parent_node
                    key = branch.name
                    if key not in tree_solutions[b] or length < tree_solutions[b][key][0]:
                        tree_solutions[b][key] = (length, plan)
                    branch.invalid = True  # close the branch; its value stays for the backup
                    print(f"{label} with length {length} in branch {key} "
                          f"({len(tree_solutions[b])}/{plan_rank} branches)")
                return len(tree_solutions[b]) >= plan_rank
            if length < best_lengths[b] and length < max_steps:
                best_solved_plans[b], best_lengths[b] = plan, length
                print(f"{label} with length {length}")
                return True
            return False
        # early_stopping_condition "solved" (Online only): rollout solutions are kept as a fallback
        # and do not end the search; only a plan reached by tree expansion does (as in MCTD).
        rollout_only = self.early_stopping_condition == "solved" and waypoints is None
        # Distributed with "solved": connections and direct plans found only by rollouts do not make
        # candidates; they are kept for a fallback route when no tree-plan route exists.
        tree_only = self.early_stopping_condition == "solved" and waypoints is not None
        fallback_plans = [[] for _ in range(len(starts))]
        fallback_lengths = np.zeros(len(starts)) + np.inf
        if waypoints is not None:
            waypoints = torch.from_numpy(waypoints).to(self.device)
            waypoint_plans = [[[] for _ in range(len(waypoints))] for _ in range(len(starts))] # (n+1, n+1, c), n+1: start, n+1: waypoints
            rollout_waypoint_plans = [[[] for _ in range(len(waypoints))] for _ in range(len(starts))]  # tree_only: every connection
            rollout_direct_candidates = []  # tree_only: start-tree plans completed by a rollout

        # Parallel search (Fast-MCTD, Sec. 4.1): each iteration selects up to K nodes per
        # tree with redundancy-aware selection, expands and simulates them as one batch, and
        # updates the tree afterwards. K = 1 is the sequential search.
        if parallel_search_num is None:
            parallel_search_num = self.parallel_search_num
        while search_num < max_search_num:
            ##################################
            # Tree Search from Start
            ##################################
            # Selection (the child slot is claimed immediately; its plan is generated below)
            selected_nodes, selected_batch_idxes, expanded_nodes = [], [], []
            for i, start_root_node in enumerate(start_root_nodes):
                if early_stopping and waypoints is None and i in solved_samples:
                    continue
                for _ in range(min(parallel_search_num, max_search_num - search_num)):
                    selected_node = start_root_node.select_expandable(
                        terminal_depth, exp_weight=0.01, virtual_visit_weight=self.virtual_visit_weight,
                        max_plan_length=max_steps if self.compress_still_steps else None)
                    if selected_node is None:
                        break
                    selected_node.add_virtual_visit()
                    selected_nodes.append(selected_node)
                    selected_batch_idxes.append(i)
                    expanded_nodes.append(selected_node.expand())
            for selected_node in selected_nodes:
                selected_node.clear_virtual_visits()
            if len(selected_nodes) == 0:
                search_num += parallel_search_num
                print(f"Search Count: {search_num}")
                continue
            # Expansion
            _starts = []
            for i, expanded_node in enumerate(expanded_nodes):
                # Planning the partial trajectories
                _start = expanded_node.get_start()
                if _start is None:
                    _start = starts[selected_batch_idxes[i]].reshape(1, -1)
                else:
                    _start = _start.detach().cpu().numpy().reshape(1, -1)
                _starts.append(_start)
            _starts = np.concatenate(_starts, axis=0)
            _goals = np.concatenate([goals[i].reshape(1, -1) for i in selected_batch_idxes], axis=0)
            local_values, local_names, plans, solveds = self.expand_via_stitching(expanded_nodes, _starts, _goals, horizon, conditions, max_steps)
            (selected_nodes, expanded_nodes, selected_batch_idxes, local_values,
             local_names, plans, solveds) = self._retain_valid_expansions(
                selected_nodes, expanded_nodes, selected_batch_idxes,
                local_values, local_names, plans, solveds)
            if not expanded_nodes:
                search_num += parallel_search_num
                continue

            if waypoints is not None:
                for _idx, _plan in enumerate(plans):
                    if "Warp" in local_names[_idx]:
                        continue
                    _batch_idx = selected_batch_idxes[_idx]
                    _diff = torch.norm(waypoints[:, None] - _plan[None], dim=-1) # (n+1, l, c)
                    for _i, _waypoint in enumerate(waypoints):
                        if _batch_idx == _i+1:
                            continue
                        _min_idx = torch.argmin(_diff[_i])
                        if _diff[_i, _min_idx] < self.cmctd_threshold_for_reaching:
                            __plan = torch.cat([plan.detach() for plan in expanded_nodes[_idx].plan_history] + [_plan[:_min_idx].cpu(), waypoints[_i][None].cpu()], axis=0)
                            __plan = __plan.cpu().numpy()
                            for _target in (waypoint_plans, rollout_waypoint_plans):
                                if len(_target[_batch_idx][_i]) == 0 or len(_target[_batch_idx][_i]) > len(__plan):
                                    _target[_batch_idx][_i] = __plan

            for i, expanded_node in enumerate(expanded_nodes):
                expanded_node.set_plan_history(plans[i])
                expanded_node.set_local_name(local_names[i])
                expanded_node.add_local_value(local_values[i])
                if solveds[i]:
                    _plan = torch.cat([plan.detach() for plan in expanded_node.plan_history], axis=0)
                    _length = len(_plan)
                    if waypoints is not None and selected_batch_idxes[i] == 0:
                        add_candidate(np.concatenate([_plan.cpu().numpy(), goals[0][None]], axis=0),
                                      ("direct",), search_num + 1)
                    if record_solution(selected_batch_idxes[i], _plan, _length, "Solved Plan", expanded_node):
                        solved_samples.append(selected_batch_idxes[i])
            # Simulation - value estimation
            _starts = []
            for i, expanded_node in enumerate(expanded_nodes):
                _starts.append(expanded_node.get_start().detach().cpu().numpy().reshape(1, -1))
            _starts = np.concatenate(_starts, axis=0)
            _goals = np.concatenate([goals[i].reshape(1, -1) for i in selected_batch_idxes], axis=0)
            values, value_names, value_plans, value_solveds = self.simulate_with_fast_replanning(expanded_nodes, _starts, _goals, horizon, conditions, max_steps)
            #value_plans, value_solveds = [], [] # NOTE: For ablation study of value estimation

            if waypoints is not None:
                for _idx, _plan in enumerate(value_plans):
                    if "Warp" in value_names[_idx]:
                        continue
                    if len(_plan) == 0:
                        continue
                    _batch_idx = selected_batch_idxes[_idx]
                    _diff = torch.norm(waypoints[:, None] - _plan[None], dim=-1) # (n+1, l, c)
                    for _i, _waypoint in enumerate(waypoints):
                        if _batch_idx == _i+1:
                            continue
                        _min_idx = torch.argmin(_diff[_i])
                        if _diff[_i, _min_idx] < self.cmctd_threshold_for_reaching:
                            __plan = torch.cat([plan.detach() for plan in expanded_nodes[_idx].plan_history] + [_plan[:_min_idx].cpu(), waypoints[_i][None].cpu()], axis=0)
                            __plan = __plan.cpu().numpy()
                            for _target in ((rollout_waypoint_plans,) if tree_only else (waypoint_plans, rollout_waypoint_plans)):
                                if len(_target[_batch_idx][_i]) == 0 or len(_target[_batch_idx][_i]) > len(__plan):
                                    _target[_batch_idx][_i] = __plan

            for i, value_plan in enumerate(value_plans):
                if len(value_plan) != 0:
                    expanded_nodes[i].set_local_name(value_names[i])
                    expanded_nodes[i].set_value_estimation_plan(value_plan)
            for i, value_solved in enumerate(value_solveds):
                if value_solved:
                    expanded_nodes[i].set_value(values[i])
                    _plan = torch.cat([plan.detach() for plan in expanded_nodes[i].plan_history], axis=0).to(self.device)
                    _plan = torch.cat([_plan, value_plans[i]], axis=0)
                    _length = len(_plan)
                    if waypoints is not None and selected_batch_idxes[i] == 0:
                        _direct = np.concatenate([_plan.detach().cpu().numpy(), goals[0][None]], axis=0)
                        if tree_only:
                            rollout_direct_candidates.append((_direct, ("direct-value",), search_num + 1))
                        else:
                            add_candidate(_direct, ("direct-value",), search_num + 1)
                    if rollout_only:
                        if _length < fallback_lengths[selected_batch_idxes[i]] and _length < max_steps:
                            fallback_plans[selected_batch_idxes[i]] = _plan
                            fallback_lengths[selected_batch_idxes[i]] = _length
                            print(f"Value Solved Plan with length {_length} (kept as fallback; search continues)")
                    elif record_solution(selected_batch_idxes[i], _plan, _length, "Value Solved Plan", expanded_nodes[i]):
                        solved_samples.append(selected_batch_idxes[i])
                elif value_names[i].endswith("-Overrun"):
                    # The rollout reached the goal only past max_steps: prefer this branch over
                    # unreached ones so the tree refines it into a solution within budget.
                    expanded_nodes[i].set_value(expanded_nodes[i].local_value + OVERRUN_BONUS)
                else:
                    expanded_nodes[i].set_value(expanded_nodes[i].local_value)
            # Backpropagation
            for i, selected_node in enumerate(selected_nodes):
                selected_nodes[i].backpropagate()

            if waypoints is not None:
                # Routes from the start (node 0) to the goal (the last node) over the connectivity graph
                graph_costs, graph_paths = self._waypoint_graph(waypoint_plans)
                graph_costs[graph_costs == np.inf] = 0
                distances = dijkstra(graph_costs, indices=0)
                if distances[len(starts)] != np.inf:
                    for route in shortest_simple_routes(graph_costs, 0, len(starts), candidate_count,
                                                        max_cost=self.plan_length_max):
                        route_plan = np.concatenate(
                            [graph_paths[route[k]][route[k+1]] for k in range(len(route)-1)]
                            + [goals[0][None]], axis=0)
                        add_candidate(route_plan, route, search_num + 1)
                if len(candidates) >= candidate_count:
                    break

            search_num += parallel_search_num
            print(f"Search Count: {search_num}")

        if ranked:
            for b in range(len(starts)):
                if tree_solutions[b]:
                    order = sorted(tree_solutions[b].values(), key=lambda item: item[0])
                    best_lengths[b], best_solved_plans[b] = order[min(plan_rank, len(order)) - 1]
                    print(f"Selected solution {min(plan_rank, len(order))}/{len(order)} by length "
                          f"(requested rank {plan_rank}): length {best_lengths[b]}")
        if rollout_only:
            for b in range(len(starts)):
                if len(best_solved_plans[b]) == 0 and len(fallback_plans[b]) != 0:
                    print(f"No expansion reached the goal; using the rollout solution with length {fallback_lengths[b]}")
                    best_solved_plans[b], best_lengths[b] = fallback_plans[b], fallback_lengths[b]
        if tree_only and not candidates:
            # No route of tree plans: fall back to routes that also use rollout connections.
            graph_costs, graph_paths = self._waypoint_graph(rollout_waypoint_plans)
            graph_costs[graph_costs == np.inf] = 0
            for route in shortest_simple_routes(graph_costs, 0, len(starts), candidate_count,
                                                max_cost=self.plan_length_max):
                add_candidate(np.concatenate([graph_paths[route[k]][route[k+1]] for k in range(len(route)-1)]
                                             + [goals[0][None]], axis=0), route, search_num)
            for candidate in rollout_direct_candidates:
                add_candidate(*candidate)
            candidates.sort(key=lambda item: len(item[0]))
            del candidates[candidate_count:]
            print(f"No tree-plan route; using {len(candidates)} rollout fallback candidate(s)")
        if waypoints is not None:
            return candidates
        return best_solved_plans, best_lengths

    def _make_env_and_agent(self):
        """Create the OGBench task env, its low-level controller (AntMaze only) and the start/goal."""
        import ogbench

        env, _, _ = ogbench.make_env_and_datasets(self.dataset)
        agent = self.get_agent(env) if "antmaze" in self.env_id else None
        initial_obs, info = env.reset(seed=self.interaction_seed, options=dict(task_id=self.task_id))
        start = initial_obs.copy()[:self.observation_dim].reshape(1, -1).astype(np.float32) # (1, c)
        goal = info['goal'][:self.observation_dim].reshape(1, -1).astype(np.float32) # (1, c)
        return env, agent, initial_obs, start, goal

    def _load_cluster_centers(self):
        path = f"graphs/{self.dataset}_cluster_centers{self.num_graph_nodes}.pkl"
        if not os.path.exists(path):
            raise ValueError(f"Cluster centers for {self.dataset} with {self.num_graph_nodes} nodes are not found, please run the clustering first from the dataset script in datasets folder.")
        with open(path, "rb") as f:
            centers = pickle.load(f)
        return np.array(centers)[:, :self.observation_dim]

    @staticmethod
    def _seed_controller(seed):
        random_module.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _log_no_plan(self, namespace, max_steps):
        episode_reward = np.zeros(1)
        episode_reward_if_stay = np.zeros(1)
        first_reach = np.zeros(1) + max_steps
        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())
        self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0))

    def replan(self, conditions=None, horizon=None, namespace="validation"):
        """Replanning baseline without search: segments from the node planner (low_level_search).

        With replan_open_loop (the release setting) the segments are chained from the end of the plan so
        far without execution, up to the goal or plan_length_max, and the whole plan is then executed open
        loop, like the composers' plans. Otherwise (closed loop, as MCTD-Replanning) each segment is
        executed and the next one is planned from where the agent is, with the executed trajectory as the
        plan history (novelty and step budget)."""
        horizon = self.episode_len if horizon is None else horizon
        env, agent, initial_obs, start, goal = self._make_env_and_agent()
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        max_steps = self.plan_length_max
        settings = {"select_rank": self.cfg.get("replan_plan_rank")}
        # The planner samples every guidance value of the children guidance sets
        guidance = [float(g) for g in sorted({g for gs in self.children_guidance_sets for g in gs})]

        def plan_segment(executed):
            position = executed[-1:].astype(np.float32)
            history = executed[:-1] if len(executed) > 1 else []
            norm_start = (torch.from_numpy(position).to(self.device) - obs_mean[None]) / obs_std[None]
            norm_goal = (torch.from_numpy(goal).to(self.device) - obs_mean[None]) / obs_std[None]
            if self.low_level_search == "mctd":
                plans, names, _, _ = self.mctd_segments([history], position, goal, norm_start, norm_goal, horizon,
                                                        conditions, max_steps, [list(guidance)], settings=settings)
            else:
                plans, names, _, _ = self.best_of_n([history], position, goal, norm_start, norm_goal, horizon,
                                                    conditions, max_steps, [list(guidance)])
            print(f"Replan at {len(executed) - 1} steps: {names[0]}")
            if len(plans[0]) == 0:  # every candidate warped: keep the current plan
                return None
            plan = plans[0].detach().cpu().numpy()
            return plan.reshape(plan.shape[0], 1, plan.shape[1])

        if self.cfg.get("replan_open_loop", False):
            # Open loop: plan segment after segment from the end of the plan so far (no execution in
            # between) until a segment reaches the goal or the plan-point cap, then execute the whole plan.
            planned = start.astype(np.float32)
            while len(planned) - 1 < max_steps:
                segment = plan_segment(planned)
                if segment is None:  # every candidate warped
                    break
                planned = np.concatenate([planned, segment[:, 0]], axis=0)
                if np.linalg.norm(planned[-1, :2] - goal[0, :2]) < 0.5:
                    break
            if len(planned) == 1:
                self._log_no_plan(namespace, max_steps)
                return
            plan = np.concatenate([planned[1:max_steps + 1], goal], axis=0)
            self.interaction(env, plan.reshape(plan.shape[0], 1, plan.shape[1]), initial_obs, start, goal, agent, namespace)
            return
        plan = plan_segment(start.astype(np.float32))
        if plan is None:
            self._log_no_plan(namespace, max_steps)
            return
        self.interaction(env, plan, initial_obs, start, goal, agent, namespace, replanner=plan_segment)

    def online_composer(self, batch_size: int, conditions=None, horizon=None, namespace="validation"):
        """Online Composer (Sec. 3.1, Alg. 1)."""
        horizon = self.episode_len if horizon is None else horizon
        
        ###############################
        # Planning with High-level Tree Search
        ###############################

        env, agent, initial_obs, start, goal = self._make_env_and_agent()

        max_steps = self.plan_length_max
        max_search_num = self.mctd_max_search_num
        best_solved_plans, _ = self.plan_tree_search(start, goal, horizon, conditions,
                                                               max_steps=max_steps, max_search_num=max_search_num,
                                                               early_stopping=True, plan_rank=self.composer_plan_rank)

        if len(best_solved_plans[0]) == 0: # Failed to find the plan
            self._log_no_plan(namespace, max_steps)
            return

        plan = np.concatenate([best_solved_plans[0].detach().cpu().numpy(), goal], axis=0)

        # Visualization
        plan = plan.reshape(plan.shape[0], 1, plan.shape[1])
        image = make_trajectory_images(self.env_id, plan, 1, start, goal, self.plot_end_points)[0]
        self.log_image(f"{namespace}_plan/plan", Image.fromarray(image))


        ###############################
        # Interaction
        ###############################
        self.interaction(env, plan, initial_obs, start, goal, agent)

    def replay_plan(self, namespace="validation"):
        """Run a saved plan from its exact recorded physical state with a new controller RNG seed."""
        import ogbench

        if self.cfg.get("replay_plan_path") is None:
            raise ValueError("replay-plan requires algorithm.replay_plan_path")
        if self.cfg.get("controller_seed") is None:
            raise ValueError("replay-plan requires algorithm.controller_seed")

        env, _, _ = ogbench.make_env_and_datasets(self.dataset)
        agent = self.get_agent(env) if "antmaze" in self.env_id else None  # PointMaze uses the PD controller
        env.reset(seed=self.interaction_seed, options=dict(task_id=self.task_id))
        with np.load(self.cfg.replay_plan_path) as saved:
            plan = saved["plan"].copy()
            recorded_obs = saved["initial_obs"].copy()
            start = saved["start"].copy()
            goal = saved["goal"].copy()

        base_env = env.unwrapped
        nq = base_env.model.nq
        nv = base_env.model.nv
        if recorded_obs.shape != (nq + nv,):
            raise ValueError(f"Saved observation shape {recorded_obs.shape} does not match qpos/qvel {(nq, nv)}")
        base_env.set_state(recorded_obs[:nq], recorded_obs[nq:])
        base_env.set_goal(goal_xy=goal[0])
        initial_obs = base_env.get_ob()
        if not np.allclose(initial_obs, recorded_obs, rtol=0, atol=1e-9):
            raise RuntimeError("Could not restore the recorded initial Ant state")
        if not np.allclose(initial_obs[:2], start[0], rtol=0, atol=1e-5):
            raise RuntimeError("Saved plan start differs from the restored Ant state")

        controller_seed = int(self.cfg.controller_seed)
        self._seed_controller(controller_seed)
        print(f"Replaying fixed plan ({len(plan)} points) from recorded Ant state; "
              f"controller_seed={controller_seed}")
        self.interaction(env, plan, initial_obs, start, goal, agent, namespace)

    def distributed_composer(self, batch_size: int, conditions=None, horizon=None, namespace="validation"):
        """Distributed Composer (Sec. 3.2, Alg. 2) over the cluster-center origins."""
        horizon = self.episode_len if horizon is None else horizon
        
        ###############################
        # Parallel Planning with High-level Tree Search
        ###############################

        env, agent, initial_obs, start, goal = self._make_env_and_agent()

        # Waypoints are cluster centers of the dataset trajectories
        num_clusters = self.num_graph_nodes
        centers = self._load_cluster_centers()

        starts = np.concatenate([start, centers], axis=0) # (n+1, c)
        goals = np.repeat(goal, num_clusters+1, axis=0) # (n+1, c)
        waypoints = np.concatenate([centers, goal], axis=0) # (n+1, c)

        max_steps = self.parts_mctd_max_steps
        max_search_num = self.mctd_max_search_num
        # One tree per origin, all in one batch; each expands distributed_parallel_num nodes per iteration.
        candidates = self.plan_tree_search(starts, goals, horizon, conditions,
                                           max_steps=max_steps, max_search_num=max_search_num,
                                           early_stopping=True, waypoints=waypoints,
                                           parallel_search_num=int(self.cfg.get("distributed_parallel_num") or 5))
        candidates = sorted(candidates, key=lambda item: len(item[0]))[:self.composer_plan_rank]
        if not candidates:
            print("Distributed Composer found no complete plan")
            self.log(f"{namespace}/first_reach", self.val_max_steps)
            self.log(f"{namespace}/success_rate", 0.0)
            return
        selected_rank = len(candidates)
        plan, route, search_count = candidates[-1]
        print(f"Distributed Composer selected rank {selected_rank}/{len(candidates)} "
              f"(requested {self.composer_plan_rank}), route={route}, length={len(plan)}")
        capture_dir = os.environ.get("MCTD_PLAN_CAPTURE_DIR")
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
            with open(os.path.join(capture_dir, "plan_selection.json"), "w") as handle:
                json.dump({"requested_rank": self.composer_plan_rank,
                           "selected_rank": selected_rank,
                           "available_candidates": len(candidates),
                           "candidate_lengths": [len(item[0]) for item in candidates],
                           "route": route, "search_count": search_count}, handle, indent=2)

        # Visualization
        plan = plan.reshape(plan.shape[0], 1, plan.shape[1])
        image = make_trajectory_images(self.env_id, plan, 1, start, goal, self.plot_end_points)[0]
        self.log_image(f"{namespace}_plan/plan", Image.fromarray(image))


        ###############################
        # Interaction
        ###############################
        self.interaction(env, plan, initial_obs, start, goal, agent)

    def _preplan_edge_budget(self):
        """(plan-length cap, search budget) of one Preplan graph edge; unset values keep the original
        per-map values (Giant 100 points / 10 searches, otherwise 200 / 20)."""
        giant = "giant" in self.dataset
        max_steps = self.cfg.get("preplan_edge_max_steps") or (100 if giant else 200)
        max_search_num = self.cfg.get("preplan_edge_max_search_num") or (10 if giant else 20)
        return int(max_steps), int(max_search_num)

    def preplan_edges(self, conditions=None, horizon=None, namespace="validation"):
        """Search Preplan graph edges one waypoint pair at a time and save each as edge_{i}_{j}.json.

        Runs the pairs (i < j) whose index % m == k for preplan_edge_shard = [k, m], each with the
        Online search (its full parallelism, stopping when a tree plan reaches the waypoint), so the
        shards can run on different GPUs; scripts/build_preplan_graph.py collects the files into
        the graph pickles. Existing files are skipped."""
        import json
        import time
        horizon = self.episode_len if horizon is None else horizon
        centers = self._load_cluster_centers()
        out_dir = self.cfg.get("preplan_edge_dir") or os.path.join(os.environ.get("MCTD_PLAN_CAPTURE_DIR", "."), "edges")
        os.makedirs(out_dir, exist_ok=True)
        shard, shards = (list(self.cfg.get("preplan_edge_shard") or [0, 1]) + [1])[:2]
        max_steps, max_search_num = self._preplan_edge_budget()
        edge_warp = self.cfg.get("preplan_edge_threshold_for_warp")
        if edge_warp is not None:
            self.threshold_for_warp = float(edge_warp)  # this mode only builds edges
        pairs = [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]
        for index, (i, j) in enumerate(pairs):
            path = os.path.join(out_dir, f"edge_{i}_{j}.json")
            if index % int(shards) != int(shard) or os.path.exists(path):
                continue
            began = time.time()
            plans, _ = self.plan_tree_search(centers[i:i + 1], centers[j:j + 1], horizon, conditions, max_steps=max_steps,
                                             max_search_num=max_search_num, early_stopping=True)
            plan = plans[0].detach().cpu().numpy() if len(plans[0]) else np.zeros((0, centers.shape[1]))
            record = dict(i=i, j=j, found=bool(len(plan)), length=len(plan), plan=plan.tolist(), max_steps=max_steps,
                          max_search_num=max_search_num, seconds=round(time.time() - began, 2), dataset=self.dataset)
            with open(path + ".tmp", "w") as f:
                json.dump(dict(record, threshold_for_warp=self.threshold_for_warp), f)
            os.replace(path + ".tmp", path)
            print(f"Preplan edge {i}-{j}: found={record['found']} length={record['length']} ({record['seconds']}s)", flush=True)

    def preplan_composer(self, batch_size: int, conditions=None, horizon=None, namespace="validation"):
        """Preplan Composer (Sec. 3.3, Alg. 3) on the prebuilt waypoint plan graph."""
        horizon = self.episode_len if horizon is None else horizon
        
        ###############################
        # Prebuilt plan graph over the waypoints (cluster centers of the dataset)
        ###############################
        num_clusters = self.num_graph_nodes
        centers = self._load_cluster_centers()
        # Waypoint centers always come from graphs/; the edge graph may live elsewhere (graph_dir).
        graph_dir = self.cfg.get("graph_dir", None) or "graphs"
        costs_path = os.path.join(graph_dir, f"{self.dataset}_graph_costs{num_clusters}.pkl")
        paths_path = os.path.join(graph_dir, f"{self.dataset}_graph_paths{num_clusters}.pkl")
        if not (os.path.exists(costs_path) and os.path.exists(paths_path)):
            raise FileNotFoundError(f"Preplan graph {costs_path} not found; build it with high_level_search=preplan-edges "
                                    "and scripts/build_preplan_graph.py (see README)")
        with open(costs_path, "rb") as f:
            graph_costs = pickle.load(f)
        with open(paths_path, "rb") as f:
            graph_paths = pickle.load(f)

        # visualization
        image = make_random_start_plans_images(self.env_id, centers, graph_costs, graph_paths)[0]
        self.log_image(f"{namespace}_graph_building/full", Image.fromarray(image))

        ###############################
        # Connect Start and Goal to Graph and Find the Shortest Path
        ###############################

        env, agent, initial_obs, start, goal = self._make_env_and_agent()

        starts, goals = [], []
        # Start to centers and goal
        for i in range(num_clusters):
            starts.append(start)
            goals.append(centers[i].reshape(1, -1)[:, :self.observation_dim])
        starts.append(start)
        goals.append(goal)
        # Start and centers to goal
        for i in range(num_clusters):
            starts.append(centers[i].reshape(1, -1)[:, :self.observation_dim])
            goals.append(goal)

        max_steps = self.graph_max_steps
        max_search_num = self.graph_max_search_num
        # The connection trees run in one batch; each expands this many nodes per iteration and stops once solved.
        connect_parallel = int(self.cfg.get("preplan_connect_parallel_num") or 5)
        neighbor_k = self.cfg.get("preplan_neighbor_k", 5)
        if neighbor_k:
            # Start connects to its k nearest of (centers, goal); goal to its k nearest of (centers, start).
            _xy = centers[:, :2]
            _s, _g = np.asarray(start).reshape(-1)[:2], np.asarray(goal).reshape(-1)[:2]
            _start_near = set(np.argsort(np.linalg.norm(np.vstack([_xy, _g]) - _s, axis=-1))[:neighbor_k])
            _goal_near = set(np.argsort(np.linalg.norm(np.vstack([_xy, _s]) - _g, axis=-1))[:neighbor_k])
            selected = [i for i in range(num_clusters) if i in _start_near]
            if num_clusters in _start_near or num_clusters in _goal_near:
                selected.append(num_clusters)  # start <-> goal
            selected += [num_clusters + 1 + i for i in range(num_clusters) if i in _goal_near]
            print(f"Preplan connects start to {sorted(int(i) for i in _start_near)} and goal to "
                  f"{sorted(int(i) for i in _goal_near)} (index {num_clusters} = the other end)")
            _plans, _ = self.plan_tree_search([starts[i] for i in selected], [goals[i] for i in selected],
                                              horizon, conditions, max_steps=max_steps, max_search_num=max_search_num,
                                              early_stopping=True, parallel_search_num=connect_parallel)
            best_solved_plans = [[] for _ in range(len(starts))]
            for i, plan_i in zip(selected, _plans):
                best_solved_plans[i] = plan_i
        else:
            best_solved_plans, _ = self.plan_tree_search(starts, goals, horizon, conditions, max_steps=max_steps,
                                                         max_search_num=max_search_num, early_stopping=True,
                                                         parallel_search_num=connect_parallel)

        # New graph build including start and goal
        new_graph_costs = np.zeros((num_clusters+2, num_clusters+2)) + np.inf
        new_graph_paths = [[[None] for _ in range(num_clusters+2)] for _ in range(num_clusters+2)]
        new_graph_costs[:num_clusters, :num_clusters] = graph_costs
        for i in range(num_clusters):
            for j in range(num_clusters):
                new_graph_paths[i][j] = graph_paths[i][j]
        for i in range(num_clusters): # start to centers
            if len(best_solved_plans[i]) > 0:
                new_graph_costs[num_clusters, i] = len(best_solved_plans[i])
                new_graph_costs[i, num_clusters] = len(best_solved_plans[i])
                new_graph_paths[num_clusters][i] = best_solved_plans[i].detach().cpu().numpy()
                new_graph_paths[i][num_clusters] = best_solved_plans[i].flip(0).detach().cpu().numpy()
        if len(best_solved_plans[num_clusters]) > 0: # start to goal
            new_graph_costs[num_clusters, num_clusters+1] = len(best_solved_plans[num_clusters])
            new_graph_costs[num_clusters+1, num_clusters] = len(best_solved_plans[num_clusters])
            new_graph_paths[num_clusters][num_clusters+1] = best_solved_plans[num_clusters].detach().cpu().numpy()
            new_graph_paths[num_clusters+1][num_clusters] = best_solved_plans[num_clusters].flip(0).detach().cpu().numpy()
        for i in range(num_clusters): # centers to goal
            if len(best_solved_plans[i+num_clusters+1]) > 0:
                new_graph_costs[i, num_clusters+1] = len(best_solved_plans[i+num_clusters+1])
                new_graph_costs[num_clusters+1, i] = len(best_solved_plans[i+num_clusters+1])
                new_graph_paths[i][num_clusters+1] = best_solved_plans[i+num_clusters+1].detach().cpu().numpy()
                new_graph_paths[num_clusters+1][i] = best_solved_plans[i+num_clusters+1].flip(0).detach().cpu().numpy()

        # Select the requested length-ranked simple route from start to goal.
        new_graph_costs[new_graph_costs == np.inf] = 0
        routes = list(shortest_simple_routes(new_graph_costs, num_clusters,
                                            num_clusters + 1, self.composer_plan_rank,
                                            max_cost=(self.plan_length_max
                                                      if self.has_explicit_plan_length_max else None)))
        if not routes:
            print("Preplan Composer found no complete route")
            self.log(f"{namespace}/first_reach", self.val_max_steps)
            self.log(f"{namespace}/success_rate", 0.0)
            return
        route_lengths = [int(sum(new_graph_costs[a, b] for a, b in zip(route[:-1], route[1:])))
                         for route in routes]
        path = routes[-1]
        plan = np.concatenate([new_graph_paths[a][b] for a, b in zip(path[:-1], path[1:])]
                              + [goal], axis=0)
        print(f"Preplan Composer selected rank {len(routes)}/{len(routes)} "
              f"(requested {self.composer_plan_rank}), route={path}, length={len(plan)}")
        capture_dir = os.environ.get("MCTD_PLAN_CAPTURE_DIR")
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
            with open(os.path.join(capture_dir, "plan_selection.json"), "w") as handle:
                json.dump({"requested_rank": self.composer_plan_rank,
                           "selected_rank": len(routes),
                           "available_candidates": len(routes),
                           "candidate_lengths": route_lengths,
                           "route": path}, handle, indent=2)
        plan = plan.reshape(plan.shape[0], 1, plan.shape[1])

        # Visualization
        image = make_trajectory_images_with_nodes(self.env_id, plan, centers, 1, start, goal, self.plot_end_points)[0]
        self.log_image(f"{namespace}_plan/plan", Image.fromarray(image))


        ###############################
        # Interaction
        ###############################

        self.interaction(env, plan, initial_obs, start, goal, agent)

    def pad_init(self, x, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            x[: self.frame_stack - 1] = 0
        elif self.padding_mode != "same":
            raise ValueError("init_pad must be 'zero' or 'same'")
        if batch_first:
            x = rearrange(x, "fs b ... -> b fs ...")

        return x

    def split_bundle(self, bundle):
        if self.use_reward:
            return torch.split(bundle, [self.observation_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [self.observation_dim, self.action_dim], -1)
            return o, a, None

    def make_bundle(
        self,
        obs: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if action is not None and valid_value is not None:
            valid_value = action
        if reward is not None and valid_value is not None:
            valid_value = reward
        if valid_value is None:
            raise ValueError("At least one of obs, action, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise_levels = super()._generate_noise_levels(xs, masks)
        _, batch_size, *_ = xs.shape

        # first frame is almost always known, this reflect that
        if random() < 0.5:
            noise_levels[0] = torch.randint(0, self.timesteps // 4, (batch_size,), device=xs.device)

        return noise_levels

    def get_agent(self, env):
        if "antmaze" not in self.env_id:
            raise ValueError(f"Environment {self.env_id} not supported")
        from dql.main_Antmaze import hyperparameters
        from dql.agents.ql_diffusion import Diffusion_QL as Agent
        params = hyperparameters[self.dataset]
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        agent = Agent(
            env_name=self.env_id,
            state_dim=state_dim*2,
            action_dim=action_dim,
            max_action=max_action,
            device=0,
            discount=0.99,
            tau=0.005,
            max_q_backup=params["max_q_backup"],
            beta_schedule="vp",
            n_timesteps=5,
            eta=params["eta"],
            lr=params["lr"],
            lr_decay=False,
            lr_maxt=params["num_epochs"],
            grad_norm=params["gn"],
            goal_dim=2,
            lcb_coef=4.0,
        )
        directory = antmaze_controller_dir(
            self.dataset, self.cfg.get("controller_checkpoint_dir", None))
        agent.load_model(str(directory), id=int(self.cfg.get("controller_checkpoint_epoch", 200)))
        return agent

    def interaction(self, env, plan, initial_obs, start, goal, agent, namespace="validation", replanner=None):
        """Execute a plan in the environment. With ``replanner`` (closed-loop replan), once the plan has been
        executed it is replaced by replanner(executed positions) and the controller restarts on it."""
        if self.cfg.get("controller_seed") is not None:
            controller_seed = int(self.cfg.controller_seed)
            self._seed_controller(controller_seed)
            print(f"Low-level controller seed: {controller_seed}")
        plan_capture_dir = os.environ.get("MCTD_PLAN_CAPTURE_DIR")
        captured_observations = [np.asarray(initial_obs).copy()] if plan_capture_dir else None
        if plan_capture_dir:
            os.makedirs(plan_capture_dir, exist_ok=True)
            np.savez(
                os.path.join(plan_capture_dir, "plan.npz"),
                plan=plan,
                initial_obs=initial_obs,
                start=start,
                goal=goal,
            )

        trajectory = []  # actual trajectory
        episode_reward = np.zeros(1)
        episode_reward_if_stay = np.zeros(1)
        reached = np.zeros(1, dtype=bool)
        first_reach = np.zeros(1)
        obs = initial_obs.copy()[None]
        origin_obs_numpy = initial_obs.copy()
        plan = torch.from_numpy(plan).float()
        use_controller = "antmaze" in self.env_id

        if use_controller:
            # The DQL controller tracks plan waypoints spaced SUB_GOAL_INTERVAL steps apart.
            if plan.shape[0] > SUB_GOAL_INTERVAL:
                sub_goal = plan[SUB_GOAL_INTERVAL, :, :self.observation_dim].detach().cpu().numpy()
                sub_goal_step = SUB_GOAL_INTERVAL
            else:
                sub_goal = plan[-1, :, :self.observation_dim].detach().cpu().numpy()
                sub_goal_step = plan.shape[0]
        plan_t0 = 0  # step at which the current plan started
        replans = [plan.numpy().copy()] if replanner is not None else None
        for t in range(self.open_loop_horizon):
            if replanner is not None and t > 0 and t - plan_t0 >= plan.shape[0]:
                executed = np.stack([np.asarray(o).reshape(-1)[:self.observation_dim]
                                     for o in [initial_obs] + [b[0, :self.observation_dim].numpy() for b in trajectory]])
                new_plan = replanner(executed)
                if new_plan is not None:
                    plan = torch.from_numpy(new_plan).float()
                    replans.append(new_plan.copy())
                    plan_t0 = t
                    if use_controller:
                        sub_goal_step = min(SUB_GOAL_INTERVAL, plan.shape[0] - 1)
                        sub_goal = plan[sub_goal_step, :, :self.observation_dim].detach().cpu().numpy()
            tt = t - plan_t0  # time index into the current plan
            if use_controller:
                if type(obs) == torch.Tensor:
                    obs_numpy = obs.detach().cpu().numpy()
                else:
                    obs_numpy = obs
                if np.linalg.norm(obs_numpy[0, :self.observation_dim] - sub_goal[0, :self.observation_dim]) < 1.0:
                    print(f"sub_goal_step {sub_goal_step} achieved at step {t}")
                    sub_goal_step += SUB_GOAL_INTERVAL
                    if plan.shape[0] - sub_goal_step <= 0:
                        sub_goal = plan[-1, :, :self.observation_dim].detach().cpu().numpy()
                    else:
                        sub_goal = plan[sub_goal_step, :, :self.observation_dim].detach().cpu().numpy()
                assert obs_numpy.shape[0] == 1, f"Batch size must be 1 for AntMaze, got {obs_numpy.shape[0]}"
                action = agent.sample_action(origin_obs_numpy, sub_goal)
                action = torch.from_numpy(action).float().reshape(1, -1)
            else:
                # PointMaze: PD controller on the planned positions.
                if tt == 0:
                    plan_vel = plan[tt, :, :self.observation_dim] - obs[:, :self.observation_dim]
                else:
                    if tt < plan.shape[0]:
                        plan_vel = plan[tt, :, :self.observation_dim] - plan[tt - 1, :, :self.observation_dim]
                    else:
                        plan_vel = 0
                if tt < plan.shape[0]:
                    action = 12.5 * (plan[tt, :, :self.observation_dim] - obs[:, :self.observation_dim]) + 1.2 * (plan_vel - obs[:, self.observation_dim:])
                else:
                    action = 12.5 * (plan[-1, :, :self.observation_dim] - obs[:, :self.observation_dim]) + 1.2 * (plan_vel - obs[:, self.observation_dim:])
            action = torch.clip(action, -1, 1).detach().cpu()
            origin_obs_numpy, reward, terminated, truncated, info = env.step(np.nan_to_num(action.numpy())[0])
            if captured_observations is not None:
                captured_observations.append(np.asarray(origin_obs_numpy).copy())

            reached = np.logical_or(reached, reward >= 1.0)
            episode_reward += reward
            episode_reward_if_stay += np.where(~reached, reward, 1)
            first_reach += ~reached

            if terminated:
                break

            obs, reward = [torch.from_numpy(item).float() for item in [origin_obs_numpy, np.array(reward)]]
            obs = obs[None]
            bundle = self.make_bundle(obs, action, reward[..., None])
            trajectory.append(bundle)
            origin_obs_numpy = origin_obs_numpy[None]

        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())
        self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0))

        # Visualization
        trajectory = torch.stack(trajectory)
        if captured_observations is not None:
            np.savez(
                os.path.join(plan_capture_dir, "execution.npz"),
                observations=np.stack(captured_observations),
                first_reach=first_reach,
                success=episode_reward >= 1.0,
            )
            if replans is not None:
                np.savez(os.path.join(plan_capture_dir, "replans.npz"), **{f"plan_{k}": p for k, p in enumerate(replans)})
        image = make_trajectory_images(self.env_id, trajectory, 1, initial_obs[:self.observation_dim][None], goal, self.plot_end_points)[0]
        self.log_image(f"{namespace}_interaction/real_trajectory", Image.fromarray(image))
