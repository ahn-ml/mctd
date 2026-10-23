import numpy as np
import torch
from omegaconf import DictConfig
import ogbench


class OGCubeOfflineRLDataset(torch.utils.data.Dataset):
    """OGBench cube play datasets as windows of cube positions (3 per cube), for the diffusion planner."""

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.env_id = cfg.env_id
        self.dataset_name = cfg.dataset
        self.n_frames = cfg.episode_len + 1
        self.gamma = cfg.gamma
        self.split = split
        self.jump = 1 if self.split != "training" else cfg.jump

        self.dataset = self.get_dataset()
        # Observation: the cube positions (19 arm entries, then 9 per cube: position, quaternion, yaw)
        num_blocks = (self.dataset["observations"].shape[1] - 19) // 9
        self.dataset["observations"] = np.concatenate(
            [self.dataset["observations"][:, 19 + i*9:19 + i*9 + 3] for i in range(num_blocks)], axis=-1)
        sample_length = 1001  # OGBench episode length
        self.n_frames = min(self.n_frames, sample_length)

        raw_observations = np.reshape(self.dataset["observations"], (-1, sample_length, self.dataset["observations"].shape[-1]))
        raw_actions = np.reshape(self.dataset["actions"], (-1, sample_length, self.dataset["actions"].shape[-1]))
        raw_terminals = np.zeros((raw_observations.shape[0], sample_length))  # not used in training and validation
        raw_terminals[:, -1] = 1
        raw_rewards = np.copy(raw_terminals)
        raw_values = self.compute_value(raw_rewards) * (1 - self.gamma) * 4 - 1

        # Windows of episode_len steps (sliding by one step)
        self.observations, self.actions, self.rewards, self.values = [], [], [], []
        for i in range(raw_observations.shape[0]):
            for j in range(sample_length - self.n_frames + 1):
                self.observations.append(raw_observations[i, j:j+self.n_frames:self.jump])
                self.actions.append(raw_actions[i, j:j+self.n_frames:self.jump])
                self.rewards.append(raw_rewards[i, j:j+self.n_frames:self.jump])
                self.values.append(raw_values[i, j:j+self.n_frames:self.jump])
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.values = np.array(self.values)
        print(f"{self.dataset_name} ({self.split}): {self.observations.shape[0]} windows of {self.observations.shape[1]} steps")

        self.total_samples = len(self.observations)

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(reward.shape[1] - 2, -1, -1):
            value[:, i] += self.gamma * value[:, i + 1]
        return value

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.observations[idx]).float() # (episode_len, obs_dim)
        action = torch.from_numpy(self.actions[idx]).float() # (episode_len, act_dim)
        reward = torch.from_numpy(self.rewards[idx]).float() # (episode_len,)
        done = np.zeros(len(observation), dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)
        return observation, action, reward, nonterminal

    def get_dataset(self):
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(self.dataset_name, compact_dataset=True)
        env.reset()  # kept: part of the random-state sequence of the released runs
        if self.split == "training":
            return train_dataset
        else:
            return val_dataset
