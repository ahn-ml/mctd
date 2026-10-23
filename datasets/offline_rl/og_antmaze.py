from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import urllib
import os
import ogbench
class OGAntMazeOfflineRLDataset(torch.utils.data.Dataset):
    '''
    '''
        
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        #self.save_dir = cfg.save_dir # Using default save_dir, "~/.ogbench/data"
        self.env_id = cfg.env_id
        self.dataset_name = cfg.dataset
        self.n_frames = cfg.episode_len + 1
        self.gamma = cfg.gamma
        self.split = split
        #Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()
        # Use [position, velocity] as observation
        #self.dataset["observations"] = np.concatenate([self.dataset["qpos"], self.dataset["qvel"]], axis=-1)
        if "navigate" in self.dataset_name:
            if "giant" in self.dataset_name:
                sample_length = 2001
            else:
                sample_length = 1001
        elif "stitch" in self.dataset_name:
            sample_length = 201
        elif "explore" in self.dataset_name:
            sample_length = 501
        else:
            raise ValueError(f"Invalid Dataset {self.dataset}")

        if self.n_frames > sample_length:
            self.n_frames = sample_length
        #assert self.n_frames <= sample_length, f"Episode length {self.n_frames} is greater than sample length {sample_length}"

        # Dataset Statistics
        print(f"Dataset: {self.dataset_name}")
        print(f"Total samples: {len(self.dataset['observations'])}, Subtrajectory length: {sample_length}")
        obs_mean = np.mean(self.dataset["observations"], axis=0)
        obs_std = np.std(self.dataset["observations"], axis=0)
        print(f"Observation shape: {self.dataset['observations'].shape}")
        print(f"observation_mean: [{','.join(np.array(obs_mean,dtype=str))}]")
        print(f"observation_std:  [{','.join(np.array(obs_std,dtype=str))}]")
        act_mean = np.mean(self.dataset["actions"], axis=0)
        act_std = np.std(self.dataset["actions"], axis=0)
        print(f"Action shape: {self.dataset['actions'].shape}")
        print(f"action_mean: [{','.join(np.array(act_mean,dtype=str))}]")
        print(f"action_std:  [{','.join(np.array(act_std,dtype=str))}]")

        # Dataset Reshaping
        raw_observations = np.reshape(self.dataset["observations"], (-1, sample_length, self.dataset["observations"].shape[-1]))
        raw_actions = np.reshape(self.dataset["actions"], (-1, sample_length, self.dataset["actions"].shape[-1]))
        raw_terminals = np.zeros((raw_observations.shape[0], sample_length)) # This will not be used in training and validation
        raw_terminals[:, -1] = 1
        raw_rewards = np.copy(raw_terminals)
        raw_values = self.compute_value(raw_rewards) * (1 - self.gamma) * 4 - 1

        # Dataset Preprocessing (Collecting episode_len trajectories in sliding window manner)
        self.observations, self.actions, self.rewards, self.values = [], [], [], []
        for i in range(raw_observations.shape[0]):
            for j in range(sample_length - self.n_frames + 1):
                self.observations.append(raw_observations[i, j:j+self.n_frames])
                self.actions.append(raw_actions[i, j:j+self.n_frames])
                self.rewards.append(raw_rewards[i, j:j+self.n_frames])
                self.values.append(raw_values[i, j:j+self.n_frames])
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.values = np.array(self.values)
        
        # Preprocessed Dataset Statistics
        print(f"Preprocessed Dataset Statistics")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")
        print(f"Reward shape: {self.rewards.shape}")
        print(f"Value shape: {self.values.shape}")

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
        value = torch.from_numpy(self.values[idx]).float() # (episode_len,)
        done = np.zeros(len(observation), dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)
        return observation, action, reward, nonterminal

    def get_dataset(self):
        _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            self.dataset_name,
            #self.save_dir, # Using default save_dir, "~/.ogbench/data"
            compact_dataset=True,
        )
        if self.split == "training":
            return train_dataset
        else:
            return val_dataset

if __name__ == "__main__":
    from unittest.mock import MagicMock
    import os
    import matplotlib.pyplot as plt
    import gymnasium 
    os.chdir("../..")
    cfg = MagicMock()
    dataset_list = [
        #"antmaze-medium-navigate-v0",
        #"antmaze-medium-stitch-v0",
        #"antmaze-medium-explore-v0",
        #"antmaze-large-navigate-v0",
        #"antmaze-large-stitch-v0",
        #"antmaze-large-explore-v0",
        #"antmaze-giant-navigate-v0",
        #"antmaze-giant-stitch-v0",
        #"antmaze-teleport-navigate-v0",
        #"antmaze-teleport-stitch-v0",
        #"antmaze-teleport-explore-v0",

        #"antmaze-medium-stitch-v0",
        "antmaze-large-stitch-v0",
        #"antmaze-giant-stitch-v0",
    ]
    for dataset in dataset_list:
        cfg.dataset = dataset
        #cfg.save_dir = ".ogbench/datasets" # Using default save_dir, "~/.ogbench/data"
        cfg.env_id = "-".join(dataset.split("-")[:2] + ["v0"])
        if "navigate" in cfg.dataset:
            if "giant" in cfg.dataset:
                cfg.episode_len = 1000
            else:
                cfg.episode_len = 500
        elif "stitch" in cfg.dataset:
            cfg.episode_len = 100
        elif "explore" in cfg.dataset:
            cfg.episode_len = 250
        cfg.gamma = 1.0
        dataset = OGAntMazeOfflineRLDataset(cfg)

    ############################################################
    # Doing Clustering and Visualization
    num_clusters = 40
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(dataset.dataset["observations"])
    cluster_centers = kmeans.cluster_centers_
    # nearest positions
    diffs = cluster_centers.reshape(-1, 1, dataset.dataset["observations"].shape[-1]) - dataset.dataset["observations"].reshape(1, -1, dataset.dataset["observations"].shape[-1])
    distances = np.linalg.norm(diffs, axis=-1)
    nearest_indices = np.argmin(distances, axis=1)
    centers = dataset.dataset["observations"][nearest_indices]

    # near centers merging
    for _ in range(10):
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                if np.linalg.norm(cluster_centers[i,:2] - cluster_centers[j,:2]) < 0.1:
                    centers[i] = (centers[i] + centers[j]) / 2
                    centers = np.delete(centers, j, axis=0)
                    num_clusters -= 1
                    break
        print(len(centers))

    # Visualization
    def convert_maze_string_to_grid(maze_string):
        lines = maze_string.split("\\")
        grid = [line[1:-1] for line in lines]
        return grid[1:-1]
    if "medium" in cfg.dataset:
        maze_string = "########\\#OO#OOO#\\#OOOO#O#\\###O#OO#\\##OOOO##\\#OO#O#O#\\#OO#OOO#\\########"
    elif "large" in cfg.dataset:
        maze_string = "#########\\#OOOOO#O#\\#O#O#OOO#\\#O#O#####\\#OOO#OOO#\\###O###O#\\#OOOOOOO#\\#O###O###\\#OOO#OOO#\\#O#O#O#O#\\#OOOOO#O#\\#########"
    elif "giant" in cfg.dataset:
        maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    elif "teleport" in cfg.dataset:
        maze_string = "#########\\#O##OOOO#\\#OOOO##O#\\#O##O##O#\\#OO#OOOO#\\#OO######\\##OOOOOO#\\#O#O###O#\\##OOOOOO#\\#OOO#####\\#O#OOOOO#\\#########"
    grid = convert_maze_string_to_grid(maze_string)
    plt.figure()
    plt.scatter(centers[:, 0]/4+1, centers[:, 1]/4+1, cmap="Reds")
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                plt.gca().add_patch(square)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("lightgray")
    plt.gca().set_axisbelow(True)
    plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
    plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
    plt.xlim([0.5, len(grid) + 0.5])
    plt.ylim([0.5, len(grid[0]) + 0.5])
    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    plt.grid(True, color="white", which="minor", linewidth=4)
    plt.gca().spines["top"].set_linewidth(4)
    plt.gca().spines["right"].set_linewidth(4)
    plt.gca().spines["bottom"].set_linewidth(4)
    plt.gca().spines["left"].set_linewidth(4)
    plt.show()
    save_path = Path(f"./datasets/offline_rl/viz")
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"{cfg.dataset}-Cluster{num_clusters}.png")
    file_save_path = Path(f"./graphs/{cfg.dataset}_cluster_centers{num_clusters}.pkl")
    import pickle
    with open(file_save_path, "wb") as f:
        pickle.dump(centers, f)
    print("Done.");exit(1)

    ## Dataset Visualization
    #obs = dataset.dataset["observations"]
    #def convert_maze_string_to_grid(maze_string):
    #    lines = maze_string.split("\\")
    #    grid = [line[1:-1] for line in lines]
    #    return grid[1:-1]
    #if "medium" in cfg.dataset:
    #    maze_string = "########\\#OO#OOO#\\#OOOO#O#\\###O#OO#\\##OOOO##\\#OO#O#O#\\#OO#OOO#\\########"
    #elif "large" in cfg.dataset:
    #    maze_string = "#########\\#OOOOO#O#\\#O#O#OOO#\\#O#O#####\\#OOO#OOO#\\###O###O#\\#OOOOOOO#\\#O###O###\\#OOO#OOO#\\#O#O#O#O#\\#OOOOO#O#\\#########"
    #elif "giant" in cfg.dataset:
    #    maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    #elif "teleport" in cfg.dataset:
    #    maze_string = "#########\\#O##OOOO#\\#OOOO##O#\\#O##O##O#\\#OO#OOOO#\\#OO######\\##OOOOOO#\\#O#O###O#\\##OOOOOO#\\#OOO#####\\#O#OOOOO#\\#########"
    #grid = convert_maze_string_to_grid(maze_string)
    #plt.figure()
    #steps = 2000
    #plt.scatter(obs[:steps, 0]/4+1, obs[:steps, 1]/4+1, c=np.arange(len(obs[:steps])), cmap="Reds")
    #for i, row in enumerate(grid):
    #    for j, cell in enumerate(row):
    #        if cell == "#":
    #            square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
    #            plt.gca().add_patch(square)
    #plt.gca().set_aspect("equal", adjustable="box")
    #plt.gca().set_facecolor("lightgray")
    #plt.gca().set_axisbelow(True)
    #plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
    #plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
    #plt.xlim([0.5, len(grid) + 0.5])
    #plt.ylim([0.5, len(grid[0]) + 0.5])
    #plt.tick_params(
    #    axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    #)
    #plt.grid(True, color="white", which="minor", linewidth=4)
    #plt.gca().spines["top"].set_linewidth(4)
    #plt.gca().spines["right"].set_linewidth(4)
    #plt.gca().spines["bottom"].set_linewidth(4)
    #plt.gca().spines["left"].set_linewidth(4)
    #plt.show()
    #save_path = Path(f"./datasets/offline_rl/viz")
    #save_path.mkdir(parents=True, exist_ok=True)
    #plt.savefig(save_path / f"{cfg.dataset}.png")
    #print("Done.");exit(1)