# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
import pdb
from pathlib import Path
import json
import re
import os
import cv2

from torchvision.transforms.functional import affine
from torchvision.transforms import ColorJitter
from torch.utils.data import IterableDataset


class Data_Sampler(object):
    def __init__(
        self,
        data,
        device,
        reward_tune="no",
        K=4,
        p=0.25,
    ):
        self.state = torch.from_numpy(data["observations"]).float().to(device)
        self.action = torch.from_numpy(data["actions"]).float().to(device)
        self.next_state = torch.from_numpy(data["next_observations"]).float().to(device)
        #reward = torch.from_numpy(data["rewards"]).view(-1, 1).float().to(device)
        reward = torch.zeros(self.state.shape[0], 1).to(device)
        if "timeouts" not in data:
            self.not_done = 1.0 - torch.from_numpy(data["terminals"]).view(
                -1, 1
            ).float().to(device)
        else:
            self.not_done = 1.0 - torch.from_numpy(data["timeouts"]).view(
                -1, 1
            ).float().to(device)

        self.K = K
        not_done_seq = torch.stack(
            [
                self.not_done[i : i + self.K + 1]
                for i in range(self.not_done.shape[0] - self.K - 1)
            ],
            dim=0,
        ).squeeze(-1)
        self.not_done_seq = torch.where(
            torch.cumprod(not_done_seq, dim=-1)[:, -2] == 1
        )[0]
        self.reward_tune = reward_tune

        self.size = self.not_done_seq.shape[0]
        self.state_dim = self.state.shape[1]
        self.state_dim = self.state_dim * 2
        self.action_dim = self.action.shape[1]

        self.device = device
        self.p = p

        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (reward - 0.25) * 2.0
        self.reward = reward.to(device)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        ind = self.not_done_seq[ind]

        min_window_size = 0
        max_window_size = np.random.default_rng().geometric(p=self.p) * self.K
        goal_idx = torch.randint(
            min_window_size, max_window_size, size=(batch_size,), device=self.device
        )

        goal_idx = torch.minimum(
            goal_idx.new_ones(goal_idx.shape) * self.size, ind + goal_idx
        )
        goal = self.state[goal_idx]

        s_feat = self.state[ind + 1][:, :2]
        g_feat = goal[:, :2]
        r_g = (-torch.linalg.norm(g_feat - s_feat, dim=-1, keepdims=True)).exp()
        not_done = (r_g != 1.0).float() * self.not_done[ind]

        reward = self.reward[ind] + r_g

        actions = self.action[ind]
        return (
            self.state[ind],
            actions,
            self.next_state[ind],
            reward,
            not_done,
            goal,
        )


def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(
        torch.tensor(trajs_rt)
    )
    reward /= rt_max - rt_min
    reward *= 1000.0
    return reward


class Normalizer:
    """
    parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class LimitsNormalizer(Normalizer):
    """
    maps [ xmin, xmax ] to [ -1, 1 ]
    """

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        """
        x : [ -1, 1 ]
        """
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.0

        return x * (self.maxs - self.mins) + self.mins


class Transform:
    def __init__(self, max_translate=6, brightness=0.1, contrast=0.1, hue=0.02):
        self.color_jitter = ColorJitter(
            brightness=brightness, contrast=contrast, hue=hue
        )
        self.max_translate = max_translate

    def __call__(self, image, eval=False):
        if not eval:
            translate = list(
                np.random.randint(-self.max_translate, self.max_translate + 1, size=2)
            )
            image = affine(image, angle=0, translate=translate, scale=1, shear=[0])
            image = self.color_jitter(image)
        image = 2.0 * image / 255.0 - 1.0
        return image


def arctanh_actions(actions, *args, **kwargs):
    epsilon = 1e-4
    assert (
        actions.min() >= -1 and actions.max() <= 1
    ), f"applying arctanh to actions in range [{actions.min()}, {actions.max()}]"
    actions = torch.clamp(actions, -1 + epsilon, 1 - epsilon)
    actions = torch.arctanh(actions)
    return actions


class TargetCubeSampler(object):
    """Samples of a single-cube controller from a multi-cube cube play dataset.

    Each sample takes the cube that moves most between the state and its goal state (a random cube if none
    moves): the state is the arm entries (19) and that cube's entries (9), the goal is that cube's future
    position, and the goal-reaching reward uses that cube's position. The goal is a future state of the same
    episode, geometric(p) * K steps ahead at most, as in Data_Sampler."""

    ARM, CUBE = 19, 9

    def __init__(self, data, device, reward_tune="cql_antmaze", K=10, p=0.2):
        obs = torch.from_numpy(data["observations"]).float()
        self.num_cubes = (obs.shape[1] - self.ARM) // self.CUBE
        self.obs = obs.to(device)
        self.next_obs = torch.from_numpy(data["next_observations"]).float().to(device)
        self.action = torch.from_numpy(data["actions"]).float().to(device)
        self.pos = torch.stack([self.obs[:, self.ARM + i*self.CUBE:self.ARM + i*self.CUBE + 3]
                                for i in range(self.num_cubes)], dim=1)  # (N, cubes, 3)
        n = self.obs.shape[0]
        # Last index of each step's episode (terminals mark the episode ends; the data ends an episode too)
        ends = np.nonzero(np.asarray(data["terminals"]).reshape(-1) > 0)[0]
        ends = np.unique(np.append(ends, n - 1))
        self.episode_end = torch.from_numpy(ends[np.searchsorted(ends, np.arange(n))]).to(device)
        self.valid = torch.nonzero(torch.arange(n, device=device) < self.episode_end).squeeze(-1)
        base = torch.zeros(n, 1, device=device)
        self.base_reward = (base - 0.5) * 4.0 if reward_tune == "cql_antmaze" else base
        self.not_done = torch.ones(n, 1, device=device)
        self.K, self.p, self.device = K, p, device
        self.size = len(self.valid)
        self.state_dim = (self.ARM + self.CUBE) * 2  # the agent's convention: state and goal slots
        self.action_dim = self.action.shape[1]
        self.goal_ready = True  # the goals are already the target cube's positions

    def sample(self, batch_size):
        ind = self.valid[torch.randint(0, self.size, (batch_size,), device=self.device)]
        window = int(np.random.default_rng().geometric(p=self.p) * self.K)
        goal_idx = torch.minimum(ind + torch.randint(0, window, (batch_size,), device=self.device),
                                 self.episode_end[ind])
        moved = torch.linalg.norm(self.pos[goal_idx] - self.pos[ind], dim=-1)  # (B, cubes)
        cube = torch.where(moved.max(dim=1).values > 0.02, moved.argmax(dim=1),
                           torch.randint(0, self.num_cubes, (batch_size,), device=self.device))
        rows = torch.arange(batch_size, device=self.device)
        cols = self.ARM + cube[:, None] * self.CUBE + torch.arange(self.CUBE, device=self.device)[None]

        def single(obs, idx):
            return torch.cat([obs[idx, :self.ARM], obs[idx[:, None], cols]], dim=-1)

        state, next_state = single(self.obs, ind), single(self.next_obs, ind)
        goal = self.pos[goal_idx, cube]
        next_pos = self.next_obs[ind[:, None], self.ARM + cube[:, None] * self.CUBE + torch.arange(3, device=self.device)[None]]
        r_g = (-torch.linalg.norm(next_pos - goal, dim=-1, keepdim=True)).exp()
        not_done = (r_g != 1.0).float() * self.not_done[ind]
        reward = self.base_reward[ind] + r_g
        return state, self.action[ind], next_state, reward, not_done, goal
