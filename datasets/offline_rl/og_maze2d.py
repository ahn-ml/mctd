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
class OGMaze2dOfflineRLDataset(torch.utils.data.Dataset):
    '''
    Offline RL dataset for 2D maze environments from OG-Bench.
    - Dataset: pointmaze-medium-navigate-v0
        - Total samples: 1000100, Subtrajectory length: 1001
        - Observation shape: (1000100, 2)
        - Observation mean: [1.0233182e+01  9.5702305e+00  1.5680445e-04 -3.4479308e-05]
        - Observation std:  [5.8341284  5.1942716  0.03949062 0.03930504]
        - Action shape: (1000100, 2)
        - Action mean: [0.00319824 -0.00255336]
        - Action std:  [0.705255  0.7001879]
    - Dataset: pointmaze-medium-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [9.5560627e+00  1.0342426e+01  5.5203458e-05 -3.1391672e-05]
        - Observation std:  [5.79129    5.055657   0.02651537 0.0255206]
        - Action shape: (1005000, 2)
        - Action mean: [-0.01176269  0.00561046]
        - Action std:  [0.7088544 0.7083117]
    - Dataset: pointmaze-large-navigate-v0
        - Total samples: 1000100, Subtrajectory length: 1001
        - Observation shape: (1000100, 2)
        - Observation mean: [1.6903608e+01  1.1005926e+01 -1.5858751e-04 -9.3383751e-05]
        - Observation std:  [10.277822    7.008117    0.03491852  0.03474263]
        - Action shape: (1000100, 2)
        - Action mean: [-0.01136365 -0.00160795]
        - Action std:  [0.70731187 0.6887115]
    - Dataset: pointmaze-large-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [1.8227280e+01 1.0928875e+01 5.9924867e-05 1.7535901e-07]
        - Observation std:  [11.593134    7.2386537   0.02676597  0.02496127]
        - Action shape: (1005000, 2)
        - Action mean: [0.00297679 -0.02221723]
        - Action std:  [0.70413345 0.70931256]
    - Dataset: pointmaze-giant-navigate-v0
        - Total samples: 1000500, Subtrajectory length: 2001
        - Observation shape: (1000500, 2)
        - Observation mean: [2.51892452e+01  1.73160381e+01 -1.02453356e-04 -9.73671558e-05]
        - Observation std:  [14.725286   11.648896    0.04031942  0.03863621]
        - Action shape: (1000500, 2)
        - Action mean: [-0.00830816  0.00012539]
        - Action std:  [0.7027886 0.6976617]
    - Dataset: pointmaze-giant-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [2.6124264e+01  1.7934097e+01 -4.2366974e-05 -1.6943675e-05]
        - Observation std:  [16.770395   11.946014    0.02510506  0.02465044]
        - Action shape: (1005000, 2)
        - Action mean: [-0.00038941 -0.00806763]
        - Action std:  [0.70801765 0.70562637]
    - Dataset: pointmaze-teleport-navigate-v0
        - Total samples: 1000100, Subtrajectory length: 1001
        - Observation shape: (1000500, 2)
        - Observation mean: [1.9765738e+01  5.7362795e+00 -1.6682481e-05 -5.1102510e-05]
        - Observation std:  [9.971918   6.44103    0.02823308 0.02780383]
        - Action shape: (1000100, 2)
        - Action mean: [-0.05035752 -0.02637289]
        - Action std:  [0.71056426 0.69628423]
    - Dataset: pointmaze-teleport-stitch-v0
        - Total samples: 1005000, Subtrajectory length: 201
        - Observation shape: (1005000, 4)
        - Observation mean: [2.0528719e+01  9.8234177e+00  6.3715575e-05 -6.4280961e-05]
        - Observation std:  [11.406025    7.509961    0.02334532  0.02215875]
        - Action shape: (1005000, 2)
        - Action mean: [-0.04648644 -0.06475478]
        - Action std:  [0.6943906  0.71181977]

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
        self.dataset["observations"] = np.concatenate([self.dataset["qpos"], self.dataset["qvel"]], axis=-1)
        if "navigate" in self.dataset_name:
            if "giant" in self.dataset_name:
                sample_length = 2001
            else:
                sample_length = 1001
        elif "stitch" in self.dataset_name:
            sample_length = 201
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
        print(f"Observation mean: {obs_mean}")
        print(f"Observation std:  {obs_std}")
        act_mean = np.mean(self.dataset["actions"], axis=0)
        act_std = np.std(self.dataset["actions"], axis=0)
        print(f"Action shape: {self.dataset['actions'].shape}")
        print(f"Action mean: {act_mean}")
        print(f"Action std:  {act_std}")

        # Dataset Reshaping
        raw_observations = self.reshaped_obs = np.reshape(self.dataset["observations"], (-1, sample_length, self.dataset["observations"].shape[-1]))
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
    #cfg.dataset = "pointmaze-medium-navigate-v0"
    cfg.dataset =  "pointmaze-medium-stitch-v0"
    #cfg.dataset = "pointmaze-large-navigate-v0"
    #cfg.dataset =  "pointmaze-large-stitch-v0"
    #cfg.dataset = "pointmaze-giant-navigate-v0"
    #cfg.dataset =  "pointmaze-giant-stitch-v0"
    #cfg.dataset = "pointmaze-teleport-navigate-v0"
    #cfg.dataset =  "pointmaze-teleport-stitch-v0"
    #cfg.save_dir = ".ogbench/datasets" # Using default save_dir, "~/.ogbench/data"
    cfg.env_id = "pointmaze-giant-v0"
    if "navigate" in cfg.dataset:
        cfg.episode_len = 1000
    elif "stitch" in cfg.dataset:
        cfg.episode_len = 100
    cfg.gamma = 1.0
    dataset = OGMaze2dOfflineRLDataset(cfg)

    ############################################################
    # Doing Clustering and Visualization
    num_clusters = 10
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

    # add start and goal nodes
    env, _, _ = ogbench.make_env_and_datasets(cfg.dataset)
    start, info = env.reset(options=dict(task_id=1))
    goal = info['goal']
    #start = start[:2]
    #goal = goal[:2]
    centers = np.concatenate([centers, start[None], goal[None]], axis=0)
    num_clusters += 2

    # path finding
    threshold = 0.1
    obs = dataset.reshaped_obs
    paths = {}
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            paths[(i, j)] = []
    for i in range(obs.shape[0]):
        _obs = obs[i]
        for path_id, path in paths.items():
            st, ed = path_id
            st_center = centers[st]
            ed_center = centers[ed]
            if np.min(np.linalg.norm(_obs - st_center[None], axis=-1)) < threshold and np.min(np.linalg.norm(_obs - ed_center[None], axis=-1)) < threshold:
                _near_st_idx = np.argmin(np.linalg.norm(_obs - st_center[None], axis=-1))
                _near_ed_idx = np.argmin(np.linalg.norm(_obs - ed_center[None], axis=-1))
                if _near_st_idx < _near_ed_idx:
                    _path = _obs[_near_st_idx:_near_ed_idx+1]
                else:
                    _path = _obs[_near_ed_idx:_near_st_idx+1][::-1]
                if len(path) == 0:
                    paths[path_id] = _path
                else:
                    if len(_path) < len(path):
                        paths[path_id] = _path

    # Dijkstra's algorithm from start to goal
    import networkx as nx
    G = nx.Graph()
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            G.add_edge(i, j, weight=len(paths[(i, j)]))
    start_idx = len(centers) - 2
    goal_idx = len(centers) - 1
    path = nx.dijkstra_path(G, start_idx, goal_idx)

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

    #plt.scatter(centers[:, 0]/4+1, centers[:, 1]/4+1, cmap="Reds")
    #for path_id, path in paths.items():
    #    if len(path) > 0:
    #        plt.plot(path[:, 0]/4+1, path[:, 1]/4+1, c="blue")

    # only visualize the path from Dijkstra's algorithm
    path_nodes = path[1:-1]
    plt.scatter(centers[path_nodes, 0]/4+1, centers[path_nodes, 1]/4+1, cmap="Reds")
    for i in range(len(path_nodes) - 1):
        plt.plot([centers[path_nodes[i], 0]/4+1, centers[path_nodes[i+1], 0]/4+1], [centers[path_nodes[i], 1]/4+1, centers[path_nodes[i+1], 1]/4+1], c="blue")

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
    #file_save_path = Path(f"./graphs/{cfg.dataset}_cluster_centers{num_clusters}.pkl")
    #import pickle
    #with open(file_save_path, "wb") as f:
    #    pickle.dump(centers, f)
    #file_save_path = Path(f"./graphs/{cfg.dataset}_cluster_centers{num_clusters}_paths.pkl")
    #import pickle
    #with open(file_save_path, "wb") as f:
    #    pickle.dump(paths, f)
    print("Done.");exit(1)

    ## Dataset Visualization
    #obs = dataset.dataset["observations"]
    #def convert_maze_string_to_grid(maze_string):
    #    lines = maze_string.split("\\")
    #    grid = [line[1:-1] for line in lines]
    #    return grid[1:-1]
    #if "giant" in cfg.dataset:
    #    maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    #grid = convert_maze_string_to_grid(maze_string)
    #plt.figure()
    #steps = 2002
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

    #o, a, r, n = dataset.__getitem__(10)
    #print(o.shape, a.shape, r.shape, n.shape)
    #plt.figure()
    #plt.scatter(o[:, 0], o[:, 1], c=np.arange(len(o)), cmap="Reds")
    #def convert_maze_string_to_grid(maze_string):
    #    lines = maze_string.split("\\")
    #    grid = [line[1:-1] for line in lines]
    #    return grid[1:-1]
    #import os
    
    #os.environ['MUJOCO_GL'] = 'egl'
    #if 'SLURM_STEP_GPUS' in os.environ:
    #    os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']
    #ogbench.locomaze.maze.make_maze_env('point','maze',maze_type='giant')
    maze_string = "################\\#O#OOOOOO##OOOO#\\#O#O##O#O#OO##O#\\#OOO#OO#OOO#OOO#\\#O###O######O#O#\\#OOO#OOO#OOOO#O#\\###O#O#OO#O#O###\\#OOO#OO#OOO#OOO#\\#O#O#O######O#O#\\#O###OOO#OOO##O#\\#OOOOO#OOO#OOOO#\\################"
    grid = convert_maze_string_to_grid(maze_string)