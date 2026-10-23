"""GPU smoke check for independent-stream DQL action batching."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from dql.agents.ql_diffusion import Diffusion_QL


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA")
    agent = Diffusion_QL(
        env_name="antmaze-large-v0", state_dim=58, action_dim=8,
        max_action=1.0, device=0, discount=0.99, tau=0.005,
        beta_schedule="vp", n_timesteps=5, goal_dim=2,
    )
    state = np.zeros((1, 29), dtype=np.float32)
    goal = np.ones((1, 2), dtype=np.float32)
    torch.manual_seed(317)
    single = agent.sample_action(state, goal)
    generator = torch.Generator(device="cuda:0").manual_seed(317)
    batched = agent.sample_action_batch(state, goal, [generator])[0]
    if not np.array_equal(single, batched):
        raise AssertionError(f"single and B=1 batch differ: {single} vs {batched}")
    generators = [torch.Generator(device="cuda:0").manual_seed(seed)
                  for seed in (11, 12)]
    actions = agent.sample_action_batch(np.repeat(state, 2, axis=0),
                                        np.repeat(goal, 2, axis=0), generators)
    assert actions.shape == (2, 8)
    print("DQL batch sampling smoke passed (B=1 exact; B=2 shape)")


if __name__ == "__main__":
    main()
