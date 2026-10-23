"""CPU test of the target-cube sampler of the single-cube controller."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dql.utils.data_sampler import TargetCubeSampler  # noqa: E402


class TargetCubeSamplerTests(unittest.TestCase):
    def test_the_moved_cube_is_the_target(self):
        n, arm = 40, 19
        obs = np.zeros((n, arm + 2 * 9), dtype=np.float32)
        obs[:, :arm] = np.arange(arm)  # arm entries
        obs[:, arm + 9:arm + 12] = np.linspace(0, 3, n)[:, None]  # cube 1 moves, cube 0 stays at 0
        data = dict(observations=obs, next_observations=np.roll(obs, -1, axis=0),
                    actions=np.ones((n, 5), dtype=np.float32), terminals=np.eye(1, n, n - 1)[0])
        sampler = TargetCubeSampler(data, 'cpu', K=10, p=0.2)
        torch.manual_seed(0)
        np.random.seed(0)
        state, action, next_state, reward, not_done, goal = sampler.sample(256)
        self.assertEqual((state.shape[1], goal.shape[1], sampler.state_dim), (28, 3, 56))
        moved = goal[:, 0] > state[:, arm]  # the goal is ahead: a moving cube was chosen
        self.assertGreater(int(moved.sum()), 0)
        cube_pos = state[moved, arm:arm + 3]  # the chosen cube's position: cube 1 (x = y = z), not cube 0 (all 0)
        self.assertTrue(torch.allclose(cube_pos[:, 0], cube_pos[:, 1]) and bool((goal[moved] > 0).all()))
        self.assertTrue(torch.equal(state[:, :arm], torch.arange(arm).float().expand(256, arm)))
        self.assertTrue(bool((reward <= -1.0 + 1e-6).all()) and bool((reward > -2.0).all()))  # -2 plus a reach term in (0, 1]

if __name__ == '__main__':
    unittest.main()
