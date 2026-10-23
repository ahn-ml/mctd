"""Loop implementation of DiffusionForcingPlanning._select_candidates (before it was batched on the GPU).

Kept as the reference the batched implementation must match."""

import numpy as np


def select_candidates_reference(self, o_tensor, plan_histories, starts, goals, horizon, max_steps, sample_guidance):
    """Score candidate plans and pick one per item (the best-of-N selection rule).

    o_tensor: (t, m, b, c) unnormalized candidate observations; sample_guidance: (m, b)."""
    num_samples = o_tensor.shape[1]
    o = o_tensor.detach().cpu().numpy()
    # Heuristic for selecting the best plan
    traj_list, info_debugging_list, solved_list, value_list = [], [], [], []
    for i in range(o.shape[2]):
        _values = np.zeros(num_samples)
        _info_for_debugging = ["NotReached"] * num_samples
        _terminated_steps = [o.shape[0]] * num_samples
        _o = o[:, :, i, :]
        _start = starts[i]
        _goal = goals[i]
        for s in range(num_samples):
            for t in range(o.shape[0]):
                if t == 0:
                    pos_diff = np.linalg.norm(_o[t, s, :2] - _start[:2], axis=-1)
                else:
                    pos_diff = np.linalg.norm(_o[t, s, :2] - _o[t - 1, s, :2], axis=-1)
                if pos_diff > self.threshold_for_warp:
                    _values[s] = 0
                    _info_for_debugging[s] = "Warp"
                    break
                if np.linalg.norm(_o[t, s, :2] - _goal[:2], axis=-1) < 0.5:
                    _values[s] = (horizon - t) / horizon
                    _info_for_debugging[s] = "Achieved"
                    _terminated_steps[s] = t
                    break
            # Unreached candidates are scored by the new ground they cover (map- and goal-free).
            if _info_for_debugging[s] == "NotReached":
                if len(plan_histories[i]) != 0:
                    # Label revisits of the branch history (diagnostic only; not rejected)
                    __o = _o[:, s:(s+1)].copy()
                    diffs = plan_histories[i][None] - __o
                    min_diffs = np.linalg.norm(diffs, axis=-1).min(axis=1)
                    if np.any(min_diffs[20:] < 1.0):
                        _info_for_debugging[s] = "Close"
                memory = plan_histories[i] if len(plan_histories[i]) != 0 else _start[None, :_o.shape[-1]]
                # 0.001 per unit of new ground for short segments, saturating smoothly below 0.01
                _values[s] = 0.01 * (1 - np.exp(-self._novel_arclength(_o[:, s], memory) / 10))
        # Zero is also a legitimate score for a valid, unreached candidate.
        # Exclude Warp (discontinuous) candidates explicitly instead of relying on scores.
        # Close (revisiting) candidates stay selectable at zero value: waiting or
        # backtracking segments are valid, and rejecting them can exhaust the tree.
        valid_indices = [s for s, status in enumerate(_info_for_debugging)
                         if status != "Warp"]
        if not valid_indices:
            traj_list.append(o_tensor[:0, 0, i])
            info_debugging_list.append("NoValidCandidate")
            value_list.append(0.0)
            solved_list.append(False)
            continue
        # Ties (e.g. every candidate revisits) go to the stronger goal guidance, then the lower index.
        _best_idx = max(valid_indices, key=lambda k: (_values[k], sample_guidance[k, i], -k))
        _o_tensor_best = o_tensor[:, _best_idx, i]
        value = _values[_best_idx]
        if _info_for_debugging[_best_idx] == "Achieved":
            terminated_step = _terminated_steps[_best_idx]
            _o_tensor_best = _o_tensor_best[:terminated_step+1]
            if len(plan_histories[i]) != 0:
                success_step = len(plan_histories[i]) + terminated_step
                value = (max_steps - success_step) / max_steps
        if self.compress_still_steps:
            _o_tensor_best = self._drop_still_steps(_o_tensor_best)
            if _info_for_debugging[_best_idx] == "Achieved":
                value = (max_steps - (len(plan_histories[i]) + len(_o_tensor_best) - 1)) / max_steps
        traj_list.append(_o_tensor_best)
        info_debugging_list.append(_info_for_debugging[_best_idx])
        solved_list.append(_info_for_debugging[_best_idx] == "Achieved")
        value_list.append(value)

    return traj_list, info_debugging_list, value_list, solved_list
