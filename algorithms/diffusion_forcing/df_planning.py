import os
from random import random
from typing import Optional, Any

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from PIL import Image

from .df_base import DiffusionForcingBase
from utils.logging_utils import make_trajectory_images
from .cube_tree_node import CubeTreeNode

SUB_GOAL_INTERVAL = 10  # plan steps between the controller's subgoals
CUBE_OBS_START = 19  # OGBench cube observation: 19 arm entries, then 9 per cube (position, quaternion, yaw)
CUBE_OBS_SIZE = 9
GUIDANCE_SCALES = [0, 0.1, 0.5, 1, 2]  # guidance scales of the node planner's samples, in turn
CUBE_PLAN_MAX_ATTEMPTS = 10  # best-of-N replanning attempts for one cube move before its branch is pruned
UNPLANNABLE = object()  # _plan_cube_moves result of a move without a plan
SINGLE_CUBE_CONTROLLER = "cube-single-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|3|1.0|False|cql_antmaze|0.2|4.0|10"


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
        self.threshold_for_warp = cfg.threshold_for_warp
        self.mctd_max_search_num = cfg.mctd_max_search_num
        self.num_samples_best_of_n = cfg.num_samples_best_of_n
        self.high_level_search = cfg.high_level_search
        self.cube_viz = cfg.cube_viz
        self.cube_interval_between_plans = cfg.cube_interval_between_plans
        self.cube_print_out = cfg.cube_print_out
        self.cube_plan_max_steps = cfg.cube_plan_max_steps
        self.cube_goal_threshold = cfg.cube_goal_threshold
        self.cube_num_temporary_positions = cfg.cube_num_temporary_positions
        self.cube_plan_cache = cfg.cube_plan_cache
        self.parallel_search_num = cfg.parallel_search_num
        self.virtual_visit_weight = cfg.virtual_visit_weight
        self.cube_reach_threshold = cfg.cube_reach_threshold
        self.cube_subgoal_threshold = cfg.cube_subgoal_threshold
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
        if self.high_level_search == "online-composer":
            self.plan_and_execute(self.cube_tree_search, conditions, namespace)
        elif self.high_level_search == "replan":
            self.plan_and_execute(self.cube_replan, conditions, namespace)
        elif self.high_level_search == "replay-plan":
            self.replay_plan(namespace)
        else:
            raise ValueError(f"Unknown high_level_search: {self.high_level_search}")

    def plan(self, start: torch.Tensor, goal: torch.Tensor, horizon: int, conditions: Optional[Any] = None,
             guidance_scale: Optional[Any] = None):
        """Sample plans from normalized starts and goals (b, obs_dim) with goal guidance (a scale, or a scale per
        item); returns the plan history (m, t, b, c) whose last entry is the fully denoised plan."""
        batch_size = start.shape[0]

        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if self.use_reward:
                raise NotImplementedError("reward guidance is not supported")
            # sparse / no reward setting, guide with goal like diffuser
            target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
            dist = nn.functional.mse_loss(pred, target, reduction="none")  # (t fs) b c

            # guidance weight for observation and action
            weight = np.array(
                [20] * (self.frame_stack)  # conditoning (aka reconstruction guidance)
                + [1 for _ in range(horizon)]  # try to reach the goal at any horizon
                + [0] * (h_padded - horizon)  # don't guide padded entries due to horizon % frame_stack != 0
            )
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
                return -(dist * weight * guidance_scale.unsqueeze(0).unsqueeze(-1)).mean() * 1000 * dist.shape[1] / 16
            episode_return = -(dist * weight).mean() * 1000 * dist.shape[1] / 16  # considering the batch size
            return guidance_scale * episode_return

        if type(guidance_scale) == torch.Tensor:
            guidance_fn = goal_guidance
        else:
            guidance_fn = goal_guidance if guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        scheduling_matrix = self._generate_scheduling_matrix(plan_tokens)
        chunk = torch.randn((plan_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
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

    def _count(self, key, n=1):
        """Search statistics of the current task (logged as validation/search_<key>)."""
        if not hasattr(self, "search_stats"):
            self.search_stats = {}
        self.search_stats[key] = self.search_stats.get(key, 0) + n

    def best_of_n(self, plan_histories, starts, goals, horizon, conditions, max_steps, reach_dims):
        """Node planner: sample num_samples_best_of_n plans per item (the guidance scales in turn) and keep the best.

        A plan step longer than threshold_for_warp rejects the plan (Warp); reaching the goal within
        cube_reach_threshold on reach_dims[i] (the moved cube's entries) scores (horizon - t) / horizon, or
        (max_steps - total length) / max_steps after earlier plans (Achieved); an unreached plan that revisits
        the branch history scores 0 (Close), otherwise 0.001 per unit of displacement, at most 0.01 (NotReached).
        Returns the best plans, their values and whether each item has a sample that reached its goal."""
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        histories = [torch.cat([plan.detach().cpu() for plan in history], dim=0).numpy() if len(history) else []
                     for history in plan_histories]
        num_samples = self.num_samples_best_of_n
        norm_starts = (torch.from_numpy(starts).to(self.device) - obs_mean[None]) / obs_std[None]
        norm_goals = (torch.from_numpy(goals).to(self.device) - obs_mean[None]) / obs_std[None]
        rs_norm_starts = norm_starts[None].repeat(num_samples, 1, 1).reshape(-1, norm_starts.shape[-1])  # (m*b c)
        rs_norm_goals = norm_goals[None].repeat(num_samples, 1, 1).reshape(-1, norm_goals.shape[-1])  # (m*b c)
        rs_guidance_scales = torch.tensor([[GUIDANCE_SCALES[i % len(GUIDANCE_SCALES)]] * len(starts)
                                           for i in range(num_samples)]).to(self.device).reshape(-1)  # (m*b)
        self._count("diffusion_calls")
        self._count("sampled_segments", len(rs_norm_starts))
        rs_plan_hist = self.plan(rs_norm_starts, rs_norm_goals, self.episode_len, conditions, guidance_scale=rs_guidance_scales)
        rs_plan_hist = self._unnormalize_x(rs_plan_hist)  # (m t b c)
        o_tensor, _, _ = self.split_bundle(rs_plan_hist[-1])
        o_tensor = o_tensor[:-1]  # last observation is dummy
        o_tensor = o_tensor.reshape(o_tensor.shape[0], num_samples, -1, o_tensor.shape[-1])  # (t m b c)
        o = o_tensor.detach().cpu().numpy()
        plans, values, solveds = [], [], []
        for i in range(o.shape[2]):
            _values = np.zeros(num_samples)
            _status = ["NotReached"] * num_samples
            _terminated_steps = [o.shape[0]] * num_samples
            _o = o[:, :, i, :]
            _reach = reach_dims[i]
            for s in range(num_samples):
                for t in range(o.shape[0]):
                    previous = starts[i] if t == 0 else _o[t - 1, s]
                    if np.linalg.norm(_o[t, s] - previous, axis=-1) > self.threshold_for_warp:
                        _status[s] = "Warp"
                        break
                    if np.linalg.norm(_o[t, s, _reach] - goals[i, _reach], axis=-1) < self.cube_reach_threshold:
                        _values[s] = (horizon - t) / horizon
                        _status[s] = "Achieved"
                        _terminated_steps[s] = t
                        break
                if _status[s] == "NotReached":  # a small value for moving away from the branch history
                    if len(histories[i]) != 0:
                        # The closest distance between each plan point and the branch history
                        min_diffs = np.linalg.norm(histories[i][None] - _o[:, s:(s+1)], axis=-1).min(axis=1)
                        if np.any(min_diffs[20:] < 1.0):
                            _status[s] = "Close"
                    if _status[s] == "NotReached":
                        _values[s] = min(np.linalg.norm(_o[0, s] - _o[-1, s], axis=-1) * 0.001, 0.01)
            best = np.argmax(_values)
            plan, value = o_tensor[:, best, i], _values[best]
            if _status[best] == "Achieved":
                plan = plan[:_terminated_steps[best] + 1]
                if len(histories[i]) != 0:
                    value = (max_steps - (len(histories[i]) + _terminated_steps[best])) / max_steps
            plans.append(plan)
            values.append(value)
            solveds.append("Achieved" in _status)
        return plans, values, solveds

    def replan(self, starts, goals, conditions, max_steps, reach_dims):
        """Best-of-N replanning of each start/goal pair (all pairs in one batch): the best of num_samples_best_of_n
        plan segments, then again from its end, until a segment reaches the goal (reach_dims[i]: the moved cube's
        entries) or max_steps plan points. Returns the chained plan of each pair, [] if none reaches the goal."""
        histories = [[] for _ in range(len(starts))]
        current = [starts[i].reshape(1, -1) for i in range(len(starts))]
        plans = [[] for _ in range(len(starts))]
        active = list(range(len(starts)))
        while active:
            segments, _, solveds = self.best_of_n([list(histories[i]) for i in active],
                                                  np.concatenate([current[i] for i in active], axis=0),
                                                  np.stack([goals[i] for i in active]), self.episode_len, conditions,
                                                  max_steps, [reach_dims[i] for i in active])
            still = []
            for n, i in enumerate(active):
                histories[i].append(segments[n].detach().cpu())
                if sum(len(segment) for segment in histories[i]) >= max_steps:
                    continue  # no plan
                if solveds[n]:
                    plans[i] = torch.cat(histories[i], dim=0)
                else:
                    current[i] = segments[n][-1].detach().cpu().numpy().reshape(1, -1)
                    still.append(i)
            active = still
        return plans

    def _cube_temporary_positions(self, goal, num_cubes):
        """Temporary positions per cube (where a cube can be put aside), drawn once per search at random on the
        table, 0.4 away from every goal and from each other."""
        temporary_positions = [[] for _ in range(num_cubes)]
        for cube_idx in range(num_cubes):
            for i in range(self.cube_num_temporary_positions):
                while True:
                    _position = np.array(list(np.random.uniform(-1, 1, size=2)) + [0.2])
                    found = True
                    for cube_idx_2 in range(num_cubes):
                        if np.linalg.norm(goal[:, 3*cube_idx_2:3*cube_idx_2+3] - _position) < 0.4:
                            found = False
                            break
                    if not found:
                        continue
                    for cube_idx_2 in range(num_cubes):
                        for position in temporary_positions[cube_idx_2]:
                            if np.linalg.norm(position - _position) < 0.4:
                                found = False
                                break
                    if not found:
                        continue
                    temporary_positions[cube_idx].append(_position)
                    break
        return temporary_positions

    def _cube_children_specs(self, state, goal, num_cubes, temporary_positions):
        """Children of a cube tree node, [cube index, target position], for every cube not at its goal: move it to
        its goal, or to one of its temporary positions."""
        children_node_specs = []
        for cube_idx in range(num_cubes):
            start_cube = state[0, 3*cube_idx:3*cube_idx+3]
            goal_cube = goal[0, 3*cube_idx:3*cube_idx+3]
            if np.linalg.norm(start_cube - goal_cube) < self.cube_goal_threshold:  # already solved
                continue
            for i in range(1 + self.cube_num_temporary_positions):
                if i == 0:  # idx=0 is the original goal
                    children_node_specs.append([cube_idx, goal[:, 3*cube_idx:3*cube_idx+3]])
                else:  # a temporary position
                    children_node_specs.append([cube_idx, temporary_positions[cube_idx][i-1]])
        return children_node_specs

    def _cube_value(self, state, goal, num_cubes):
        """Value of a state: the share of the cubes at their goal; -inf for a physically unrealistic state (a cube
        in the air or two cubes at one position)."""
        goal_threshold = self.cube_goal_threshold
        solved_cube_num = 0
        for cube_idx in range(num_cubes):
            if np.linalg.norm(state[0, 3*cube_idx:3*cube_idx+3] - goal[0, 3*cube_idx:3*cube_idx+3]) < goal_threshold:
                solved_cube_num += 1
        value = solved_cube_num / num_cubes
        # A cube off the table (0.4 or higher) must rest on another cube, 0.5 lower within the
        # threshold (each cube checks the cube right below it, so a tower of any height is checked level by level)
        for i in range(num_cubes):
            below_cube_z = state[0, 3*i+2] - 0.5
            if below_cube_z < -0.1:  # on the table
                continue
            if not any(abs(state[0, 3*j+2] - below_cube_z) < goal_threshold
                       and np.linalg.norm(state[0, 3*j:3*j+3] - state[0, 3*i:3*i+3], axis=-1) < 0.6
                       for j in range(num_cubes) if j != i):
                value = -np.inf
                break
        # Two cubes at one position
        for i in range(num_cubes):
            for j in range(num_cubes):
                if i == j:
                    continue
                if np.linalg.norm(state[0, 3*i:3*i+3] - state[0, 3*j:3*j+3]) < goal_threshold:
                    value = -np.inf
                    break
        return value

    def _cube_move_value(self, start, plan, cube, goal, num_cubes):
        """Value estimation of a cube tree node from its move: -inf when the moved cube has another cube on top at
        the start (an obstructed grasp), which prunes the branch; otherwise the value of the state the move
        reaches (_cube_value)."""
        moved = start[0, 3*cube:3*cube+3]
        for j in range(num_cubes):
            if j == cube:
                continue
            other = start[0, 3*j:3*j+3]
            # A cube resting on the moved cube (0.5 above, as in the stacking check of _cube_value)
            if abs(other[2] - (moved[2] + 0.5)) < self.cube_goal_threshold and np.linalg.norm(other - moved) < 0.6:
                return -np.inf
        return self._cube_value(plan[-1][None], goal, num_cubes)

    def _plan_cube_moves(self, moves, conditions, plan_cache):
        """Plans of cube moves [(start, cube index, target position)] while the other cubes stay, by best-of-N
        replanning of all the moves in one batch; an identical move (cached, or repeated in the batch) reuses its
        plan. A move's plan is None when the cube is already there, UNPLANNABLE when CUBE_PLAN_MAX_ATTEMPTS
        replannings find no plan."""
        results = [None] * len(moves)
        searches = []  # distinct moves to plan: dict(cube, start and goal of the cube, first move, members, plan)
        goals = []
        for m, (start, cube, target) in enumerate(moves):
            goal = start.copy()
            goal[:, 3*cube:3*cube+3] = target
            goals.append(goal)
            if np.linalg.norm(start - goal) < 0.01:  # nothing to move
                continue
            _start, _goal = start[:, 3*cube:3*cube+3].copy(), goal[:, 3*cube:3*cube+3].copy()
            if self.cube_plan_cache:
                cached = [plan for cached_start, cached_goal, plan in plan_cache.get(cube, [])
                          if np.linalg.norm(cached_start - _start) < 0.01 and np.linalg.norm(cached_goal - _goal) < 0.01]
                if cached:
                    self._count("cache_hits")
                    results[m] = cached[0].copy()
                    if self.cube_print_out:
                        print(f"Found cached plan for cube {cube} from {_start} to {_goal}")
                    continue
                same = next((q for q in searches if q["cube"] == cube and np.linalg.norm(q["start"] - _start) < 0.01
                             and np.linalg.norm(q["goal"] - _goal) < 0.01), None)
                if same is not None:
                    same["members"].append(m)
                    continue
            searches.append(dict(cube=cube, start=_start, goal=_goal, first=m, members=[m], plan=None))

        for _ in range(CUBE_PLAN_MAX_ATTEMPTS):
            todo = [q for q in searches if q["plan"] is None]
            if not todo:
                break
            self._count("planned_moves", len(todo))
            plans = self.replan(np.concatenate([moves[q["first"]][0] for q in todo], axis=0),
                                np.concatenate([goals[q["first"]] for q in todo], axis=0), conditions,
                                self.cube_plan_max_steps, [slice(3*q["cube"], 3*q["cube"]+3) for q in todo])
            for q, plan in zip(todo, plans):
                if len(plan) > 0:
                    q["plan"] = plan
                else:
                    print(f"No plan found for cube {q['cube']} from {moves[q['first']][0]} to {goals[q['first']]}, retry...")

        for q in searches:
            if q["plan"] is None:
                for m in q["members"]:
                    results[m] = UNPLANNABLE
                continue
            plan = np.concatenate([q["plan"].numpy(), goals[q["first"]]], axis=0)
            if self.cube_print_out:
                print(f"Found plan: {len(plan) - 1}")
            if self.cube_plan_cache:
                plan_cache.setdefault(q["cube"], []).append([q["start"], q["goal"], plan.copy()])
            for m in q["members"]:
                results[m] = plan.copy()

        for m, (start, cube, _) in enumerate(moves):  # the other cubes stay where they are
            if results[m] is not None and results[m] is not UNPLANNABLE:
                results[m][:, :3*cube] = start[0, :3*cube]
                results[m][:, 3*cube+3:] = start[0, 3*cube+3:]
        return results

    def cube_tree_search(self, start, goal, conditions=None):
        """Object-level tree search: expanding a node plans one cube move (to the cube's goal or to a temporary
        position) with best-of-N replanning, and simulation estimates the value of the state it reaches. Each
        iteration expands up to parallel_search_num nodes (redundancy-aware selection with virtual visits, as
        Fast-MCTD) whose moves are planned in one batch. Returns the plans and moved cube indices of the first solution, or [], []."""
        assert start.shape[0] == 1, "Batch size must be 1 for the cube tree search"
        num_cubes = start.shape[1] // 3
        plan_cache = {}  # cube index -> [start, goal, plan] of the planned moves
        temporary_positions = self._cube_temporary_positions(goal, num_cubes)

        root_node = CubeTreeNode(name="0", depth=0, parent_node=None, node_spec=None)
        root_node.set_value(0)
        root_node.set_children_node_specs(self._cube_children_specs(start, goal, num_cubes, temporary_positions))

        search_num = 0
        while search_num < self.mctd_max_search_num:
            # Selection: up to parallel_search_num nodes, each claiming its next child
            selected, expanded = [], []
            for _ in range(min(self.parallel_search_num, self.mctd_max_search_num - search_num)):
                selected_node = root_node.select_expandable(virtual_visit_weight=self.virtual_visit_weight)
                if selected_node is None:  # every branch is pruned or waiting for its evaluation
                    break
                selected_node.add_virtual_visit()
                selected.append(selected_node)
                expanded.append(selected_node.expand())
            for selected_node in selected:
                selected_node.clear_virtual_visits()
            if not selected:
                break
            self._count("iterations")
            self._count("expanded_nodes", len(expanded))
            # Expansion: the new children's moves, planned in one batch to bring each cube to its target
            moves = []
            for selected_node, expanded_node in zip(selected, expanded):
                expansion_start = start.copy() if selected_node.get_plan() is None else selected_node.get_plan()[-1][None]
                target_cube_idx, target_cube_goal = expanded_node.get_node_spec()
                if self.cube_print_out:
                    print(f"Search {search_num}: {selected_node} -> cube {target_cube_idx} to {target_cube_goal}")
                moves.append((expansion_start, target_cube_idx, target_cube_goal))
            plans = self._plan_cube_moves(moves, conditions, plan_cache)
            for expanded_node, (expansion_start, cube, _), plan in zip(expanded, moves, plans):
                if plan is None or plan is UNPLANNABLE:  # nothing to move, or no plan: prune the branch
                    expanded_node.set_value(-np.inf)
                    continue
                expanded_node.set_plan(plan)
                # Simulation: value estimation of the move and the state it reaches (no rollout); -inf prunes the branch
                value = self._cube_move_value(expansion_start, plan, cube, goal, num_cubes)
                expanded_node.set_value(value)
                if value != -np.inf:
                    expanded_node.set_children_node_specs(self._cube_children_specs(plan[-1][None], goal, num_cubes, temporary_positions))
            for expanded_node in expanded:
                if expanded_node.get_value() == 1:  # solved
                    plans, target_cube_indexes = [], []
                    node = expanded_node
                    while not node.is_root_node():
                        plans.append(node.get_plan())
                        target_cube_indexes.append(node.get_node_spec()[0])
                        node = node.get_parent_node()
                    return plans[::-1], target_cube_indexes[::-1]
            # Backpropagation
            for selected_node in selected:
                selected_node.backpropagate()
            search_num += len(selected)
        return [], []

    def cube_replan(self, start, goal, conditions=None):
        """Replan baseline (no search and no value estimation): from the current state, plan every cube move (each
        cube not at its goal, to its goal or to a temporary position) with best-of-N replanning in one batch, commit
        to the move whose plan reaches its target in the fewest points, and plan again from its end, until every
        cube is at its goal. Returns the plans and moved cube indices, or [], [] when no move is plannable or the
        plans pass val_max_steps points."""
        assert start.shape[0] == 1, "Batch size must be 1 for the cube replanning"
        num_cubes = start.shape[1] // 3
        plan_cache = {}
        temporary_positions = self._cube_temporary_positions(goal, num_cubes)
        state, plans, target_cube_indexes, num_points = start.copy(), [], [], 0
        while True:
            specs = self._cube_children_specs(state, goal, num_cubes, temporary_positions)
            if not specs:  # every cube is at its goal
                return plans, target_cube_indexes
            self._count("iterations")
            self._count("expanded_nodes", len(specs))
            moves = self._plan_cube_moves([(state.copy(), cube, target) for cube, target in specs], conditions, plan_cache)
            candidates = [k for k, plan in enumerate(moves) if plan is not None and plan is not UNPLANNABLE]
            if not candidates:
                return [], []
            chosen = min(candidates, key=lambda k: len(moves[k]))  # the shortest plan (the first of equal ones)
            num_points += len(moves[chosen])
            if num_points > self.val_max_steps:
                return [], []
            if self.cube_print_out:
                print(f"Move {len(plans)}: cube {specs[chosen][0]} to {specs[chosen][1]} ({len(moves[chosen])} points)")
            plans.append(moves[chosen])
            target_cube_indexes.append(specs[chosen][0])
            state = moves[chosen][-1][None].copy()

    def _make_env_and_agent(self):
        """Create the OGBench task env, the single-cube controller and the cube start/goal positions."""
        import ogbench

        env, _, _ = ogbench.make_env_and_datasets(self.dataset)
        agent = self.get_agent(env)
        initial_obs, info = env.reset(seed=self.interaction_seed, options=dict(task_id=self.task_id))
        num_cubes = self._num_cubes(initial_obs)
        start = np.concatenate([initial_obs.copy()[self._cube_slice(i, 3)] for i in range(num_cubes)], axis=-1).reshape(1, -1).astype(np.float32)  # (1, c)
        goal = np.concatenate([info['goal'][self._cube_slice(i, 3)] for i in range(num_cubes)], axis=-1).reshape(1, -1).astype(np.float32)  # (1, c)
        return env, agent, initial_obs, start, goal

    @staticmethod
    def _num_cubes(obs):
        return (obs.shape[-1] - CUBE_OBS_START) // CUBE_OBS_SIZE

    @staticmethod
    def _cube_slice(cube_idx, size=CUBE_OBS_SIZE):
        """Slice of one cube's entries in the OGBench observation (size 3: its position)."""
        return slice(CUBE_OBS_START + cube_idx * CUBE_OBS_SIZE, CUBE_OBS_START + cube_idx * CUBE_OBS_SIZE + size)

    def _log_no_plan(self, namespace):
        print("No plan of the cube moves found")
        self.log(f"{namespace}/episode_reward", 0.0)
        self.log(f"{namespace}/first_reach", float(self.val_max_steps))
        self.log(f"{namespace}/success_rate", 0.0)

    def plan_and_execute(self, search, conditions=None, namespace="validation"):
        """Plan the cube moves with search (the Online Composer's cube_tree_search, or cube_replan), then execute."""
        import time

        env, agent, initial_obs, start, goal = self._make_env_and_agent()
        plan_start_time = time.time()
        self.search_stats = {}
        plans, target_cube_indexes = search(start, goal, conditions)
        self.log(f"{namespace}/plan_time", time.time() - plan_start_time)
        # iterations, expanded nodes (replan: moves planned from each state), moves planned (a retry counts again),
        # plan-cache hits, diffusion calls and sampled segments of the search
        for key, value in sorted(self.search_stats.items()):
            self.log(f"{namespace}/search_{key}", float(value))
        print("Search statistics:", dict(sorted(self.search_stats.items())))
        if not plans:  # no solution
            self._log_no_plan(namespace)
            return

        plans = [plan.reshape(plan.shape[0], 1, plan.shape[1]) for plan in plans]
        for i in range(len(target_cube_indexes) if self.logger is not None else 0):
            image = make_trajectory_images(self.env_id, plans[i], 1, start, goal, self.plot_end_points)[0]
            self.log_image(f"{namespace}_plan/{i+1}th_plan_for_cube_{target_cube_indexes[i]}", Image.fromarray(image))

        self.interaction(env, plans, target_cube_indexes, initial_obs, start, goal, agent, namespace)

    def replay_plan(self, namespace="validation"):
        """Execute the cube plans saved in replay_plan_path (plan.npz of an earlier run of the same task and seed)
        without a search, with the controller RNG seeded by controller_seed."""
        import random as random_module

        if self.cfg.get("replay_plan_path") is None or self.cfg.get("controller_seed") is None:
            raise ValueError("replay-plan requires algorithm.replay_plan_path and algorithm.controller_seed")
        env, agent, initial_obs, start, goal = self._make_env_and_agent()
        with np.load(self.cfg.replay_plan_path) as saved:
            target_cube_indexes = [int(i) for i in saved["target_cube_indexes"]]
            plans = [saved[f"plan_{k}"] for k in range(len(target_cube_indexes))]
        seed = int(self.cfg.controller_seed)
        random_module.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Replaying {len(plans)} saved cube plans with controller_seed={seed}")
        self.interaction(env, plans, target_cube_indexes, initial_obs, start, goal, agent, namespace)

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
        """The Diffusion-QL controller that moves one cube (input: the arm and that cube's observation) toward a
        cube position: controller_checkpoint_dir (the release presets: the target-cube controller of each multi-cube
        size), or the single-cube controller."""
        from dql.main_Cube import hyperparameters
        from dql.agents.ql_diffusion import Diffusion_QL as Agent
        params = hyperparameters[self.dataset]
        agent = Agent(
            env_name=self.env_id,
            state_dim=(CUBE_OBS_START + CUBE_OBS_SIZE) * 2,
            action_dim=env.action_space.shape[0],
            max_action=float(env.action_space.high[0]),
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
            goal_dim=params["goal_dim"],
            lcb_coef=4.0,
        )
        directory = self.cfg.get("controller_checkpoint_dir") or os.path.join("dql", "results", SINGLE_CUBE_CONTROLLER)
        agent.load_model(os.path.join(os.getcwd(), directory), id=int(self.cfg.get("controller_checkpoint_epoch", 200)))
        return agent

    def _single_cube_obs(self, obs, cube_idx):
        """Controller input: the arm entries and one cube's entries of the (1, d) observation."""
        return np.concatenate([obs[:, :CUBE_OBS_START], obs[:, self._cube_slice(cube_idx)]], axis=-1)

    def interaction(self, env, plans, target_cube_idxes, initial_obs, start, goal, agent, namespace="validation"):
        """Execute the cube plans one after another with the single-cube controller.

        Before each plan the arm moves toward its cube (and after a plan it opens the gripper and lifts);
        the controller tracks subgoals SUB_GOAL_INTERVAL plan steps apart, a plan is done once its cube is
        within 0.2 of the plan's end, and after the last plan the gripper opens until the episode ends."""
        episode_reward = np.zeros(1)
        episode_reward_if_stay = np.zeros(1)
        reached = np.zeros(1, dtype=bool)
        first_reach = np.zeros(1)
        trajectory = []  # executed observations
        origin_obs_numpy = initial_obs.copy()[None]
        plans = [torch.from_numpy(plan).float() for plan in plans]
        num_cubes = self._num_cubes(initial_obs)
        cube_visual_obss = [env.get_pixel_observation()]

        def step(action):
            """One environment step; returns True when the episode ends."""
            nonlocal origin_obs_numpy, episode_reward, episode_reward_if_stay, reached, first_reach
            origin_obs_numpy, reward, terminated, truncated, info = env.step(np.nan_to_num(action)[0])
            if self.cube_viz:
                cube_visual_obss.append(env.get_pixel_observation())
            reached = np.logical_or(reached, reward >= 1.0)
            episode_reward += reward
            episode_reward_if_stay += np.where(~reached, reward, 1)
            first_reach += ~reached
            if terminated or truncated:
                return True
            trajectory.append(origin_obs_numpy.copy())
            origin_obs_numpy = origin_obs_numpy[None]
            return False

        def cube_positions():  # float32, as the positions the plans are compared with
            return np.concatenate([origin_obs_numpy[:, self._cube_slice(i, 3)] for i in range(num_cubes)], axis=-1).astype(np.float32)

        def move_to_cube(cube_idx, release_steps):
            """After release_steps steps of opening the gripper and lifting, move the arm toward the cube (its
            plan's first point), for cube_interval_between_plans steps in total."""
            for release_idx in range(self.cube_interval_between_plans):
                if release_idx < release_steps:
                    action = np.nan_to_num([0, 0, 1.0, 0, -1.0]).reshape(1, -1)  # open the gripper and move up
                else:
                    subgoal = plans[plan_idx][0, :, 3*cube_idx:3*cube_idx+3]
                    action = agent.sample_action(self._single_cube_obs(origin_obs_numpy, cube_idx), subgoal)
                    action[2] = 0.02
                    action = action.reshape(1, -1)
                action = torch.from_numpy(np.asarray(action)).float().reshape(1, -1).numpy()
                if step(action):
                    return True
                nonlocal t
                t += 1
            return False

        def first_subgoal(plan):
            """The subgoal SUB_GOAL_INTERVAL steps into the plan (its end for a shorter plan) and its step."""
            if plan.shape[0] > SUB_GOAL_INTERVAL:
                return plan[SUB_GOAL_INTERVAL, :, :self.observation_dim].detach().cpu().numpy(), SUB_GOAL_INTERVAL
            return plan[-1, :, :self.observation_dim].detach().cpu().numpy(), plan.shape[0]

        plan_idx = 0
        sub_goal, sub_goal_step = first_subgoal(plans[plan_idx])

        t = 0
        terminate = move_to_cube(target_cube_idxes[plan_idx], release_steps=0)
        while not terminate and t < self.open_loop_horizon:
            cube_idx = target_cube_idxes[plan_idx]
            cube = slice(3*cube_idx, 3*cube_idx+3)
            if np.linalg.norm(cube_positions()[0, cube] - sub_goal[0, cube]) < self.cube_subgoal_threshold:
                sub_goal_step += SUB_GOAL_INTERVAL
                if plans[plan_idx].shape[0] - sub_goal_step <= 0:
                    sub_goal = plans[plan_idx][-1, :, :self.observation_dim].detach().cpu().numpy()
                else:
                    sub_goal = plans[plan_idx][sub_goal_step, :, :self.observation_dim].detach().cpu().numpy()
            action = agent.sample_action(self._single_cube_obs(origin_obs_numpy, cube_idx), sub_goal[:, cube])
            action = torch.clip(torch.from_numpy(action).float().reshape(1, -1), -1, 1).numpy()
            if step(action):
                break
            t += 1

            plan_end = plans[plan_idx][-1, :, cube].detach().cpu().numpy()
            if np.linalg.norm(origin_obs_numpy[:, self._cube_slice(cube_idx, 3)] - plan_end) < 0.2:
                print(f"Cube {cube_idx} reached at step {t}")
                plan_idx += 1
                if plan_idx == len(plans):
                    # Every plan is done but the episode continues (the stack is not aligned yet): open the gripper
                    while not step(np.array([[0, 0, 1.0, 0, -1.0]], dtype=np.float32)):
                        t += 1
                    break
                sub_goal, sub_goal_step = first_subgoal(plans[plan_idx])
                terminate = move_to_cube(target_cube_idxes[plan_idx], release_steps=5)

        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())
        self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0))

        # Visualization
        if trajectory and self.logger is not None:
            executed = np.concatenate([np.stack(trajectory)[:, self._cube_slice(i, 3)] for i in range(num_cubes)], axis=-1)[:, None]
            image = make_trajectory_images(self.env_id, executed, 1, start, goal, self.plot_end_points)[0]
            self.log_image(f"{namespace}_interaction/real_trajectory", Image.fromarray(image))
        if self.cube_viz and self.logger is not None:
            import wandb
            visual_obss = np.array(cube_visual_obss).transpose(0, 3, 1, 2)  # (t, c, h, w)
            self.logger.experiment.log({
                "validation_interaction_video/video": wandb.Video(visual_obss, fps=24),
                "trainer/global_step": self.global_step,
            })
        capture_dir = os.environ.get("MCTD_PLAN_CAPTURE_DIR")
        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)
            np.savez(os.path.join(capture_dir, "plan.npz"), target_cube_indexes=np.array(target_cube_idxes),
                     **{f"plan_{k}": p.numpy() for k, p in enumerate(plans)})
            np.savez(os.path.join(capture_dir, "execution.npz"), observations=np.stack([initial_obs] + trajectory),
                     first_reach=first_reach, success=episode_reward >= 1.0)
            if self.cube_viz:  # the rendered execution, every second step
                frames = [Image.fromarray(np.asarray(frame)) for frame in cube_visual_obss[::2]]
                frames[0].save(os.path.join(capture_dir, "execution.gif"), save_all=True, append_images=frames[1:],
                               duration=40, loop=0)
