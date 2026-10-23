# Compositional Monte Carlo Tree Diffusion (C-MCTD): Robot Arm Cube Manipulation

This repository provides the **official PyTorch implementation** of Compositional Monte Carlo Tree Diffusion (C-MCTD) for the robot arm cube manipulation tasks of [OGBench](https://seohong.me/projects/ogbench/) (single, double, triple and quadruple cubes). The maze tasks are released in a separate branch.

## Compositional Monte Carlo Tree Diffusion (C-MCTD)

Author: Jaesik Yoon, Hyeonseo Cho, Sungjin Ahn

_Published in Advances in Neural Information Processing Systems (NeurIPS) 2025, Spotlight._

[[Paper]](https://arxiv.org/abs/2510.21361)

![C-MCTD](./assets/C-MCTD_overview.png)

C-MCTD searches over compositions of complete plans instead of within a single trajectory. On the cube tasks it runs the **Online Composer with a plan cache** (Sec. 5.3 of the paper) over cube moves:

- **Object-level tree**: expanding a node plans one cube move, to the cube's goal or to a temporary position (drawn once per search on the table, away from the goals, so that a cube can be put aside, e.g. to swap two cubes). Simulation estimates the node's value without a rollout: the share of the cubes at their goal, while a physically unrealistic move prunes the branch (grasping a cube with another cube on top, a cube left in the air, two cubes at one position). The tree expands 4 nodes per iteration with redundancy-aware selection (virtual visits, as in Fast-MCTD) and plans their moves in one batch.
- **Cube-move planning**: each cube move is planned with best-of-N replanning, while the other cubes stay where they are (object-wise planning): the best of 50 diffusion-generated plan segments, then again from its end until the cube reaches its target (at most 500 plan points for double and 1000 for triple and quadruple).
- **Plan cache**: the plan of an identical cube move (same cube, start and target) is reused across the object-level tree.

The plans are executed one cube after another with a Diffusion-QL controller that sees the arm and the cube it moves. From the double-cube tasks on, the controller follows the object-wise planning: it is trained on the multi-cube data with the cube that moves always placed as its (first and only) cube, so that it controls whichever cube the current plan moves.

This release updates the implementation of the paper: the cube moves are planned with best-of-N replanning, the object-level tree is searched in parallel with an extended value estimation, and the multi-cube tasks use target-cube controllers.

## Installation

We recommend using Docker to set up the environment for reproducibility.

**1. Download MuJoCo binaries:**
Our Docker setup requires the MuJoCo 2.1.0 binaries. Please download them from this [link](https://drive.google.com/drive/folders/1gwXsIzpTILXG6kZv1EDrLBeEKYpPAl-G?usp=drive_link) and place the `mujoco210` directory into `./dockerfile/mujoco/`. The final path should look like `./dockerfile/mujoco/mujoco210`.

**2. Build the Docker image:**

From the root of the repository, build the image with your user and group IDs so that files written to the mounted repository stay writable:

```bash
docker build -t cmctd --build-arg UID=$(id -u) --build-arg GID=$(id -g) dockerfile
```

> **Note on the environment:** The Dockerfile installs a customized version of the [OGBench](https://seohong.me/projects/ogbench/) benchmark (`dockerfile/ogbench-1.0.1-py3-none-any.whl`). Stock OGBench is not interchangeable with it.

**3. Run commands in the container:**
`scripts/docker_run.sh` runs a command inside the image with the repository mounted at `/workspace` and the OGBench dataset cache (`~/.ogbench/data`, downloaded on first use) mounted inside the container. `CMCTD_IMAGE`, `CMCTD_GPUS` (a GPU id or `none`) and `CMCTD_DATASETS` override the image, GPU and dataset cache. To compare your environment with the one used for the reported results:

```bash
CMCTD_GPUS=none bash scripts/docker_run.sh python3 scripts/check_environment.py
```

Logging to Weights & Biases is optional and disabled by default (`wandb.mode=disabled`).

## Evaluation

### 1. Download Pre-trained Models

Download the pre-trained models from this [link](https://drive.google.com/drive/folders/1ncik6zfrWamvA9e7kOVR_iHy3943_jVq) and extract the archives from the repository root; they keep the paths expected by the code (`ASSET_MANIFEST.json` lists every file with its SHA-256).

| Archive | Contents |
|---|---|
| [`cmctd_cube_planner_weights.tar.gz`](https://drive.google.com/file/d/1_X34ixxxTxHzKzL6IpOgGihdIcri44wy/view) | Diffusion planners of the four task sizes |
| [`cmctd_cube_single_controller.tar.gz`](https://drive.google.com/file/d/1y8wMA3LzXJTHNzRCsErQ92r122APU2cd/view) | Single-cube Diffusion-QL controller (epoch 200), used on the single-cube tasks |
| Target-cube controllers: [Double](https://drive.google.com/file/d/1G5a_4k2luTiNe-lZVmintE2A6mWoANSR/view), [Triple](https://drive.google.com/file/d/1LcwmdU5eoRMrctWd44vA9jqmm2hv2HgD/view), [Quadruple](https://drive.google.com/file/d/1jVk7t8Lxw08bFeh_6EYqAPke0A_GQZPq/view) | Diffusion-QL controllers of the multi-cube tasks (epoch 600), trained on the arm and the cube that moves |

The planners are extracted to `outputs/downloaded/jaesikyoon/jsik_c_mctd/<id>/model.ckpt` and the controllers to `dql/results/`:

| Task | Single | Double | Triple | Quadruple |
|---|---|---|---|---|
| Planner | `x3rlj069` | `xn2tbc3m` | `wouak5pu` | `b8i3jzfw` |

### 2. Run the Evaluation

```bash
bash scripts/docker_run.sh python3 scripts/cmctd.py evaluate \
  --size double --task 1 --seed 0 \
  --checkpoint outputs/downloaded/jaesikyoon/jsik_c_mctd/xn2tbc3m/model.ckpt \
  --output outputs/eval/cube-double-task1-seed0
```

- `--size`: `single`, `double`, `triple` or `quadruple`; `--task`: 1–5; `--seed`: planner and environment seed.
- `--composer`: `online` (default) or `replan`, the baseline without search: every cube move from the current state is planned with the same best-of-N replanning, the move with the shortest plan to its target is taken without a value estimate, and the moves are chained from its end until every cube is at its goal.
- `--output` must be a new directory. It stores the Hydra configuration, the cube plans (`plan.npz`) and the executed trajectory (`execution.npz`). A case where no plan is found counts as a failure. `--override algorithm.cube_viz=True` also renders the execution to `execution.gif`.
- `--override KEY=VALUE` passes additional Hydra overrides and `--dry-run` prints the command without running it.

The settings are the defaults in `configurations/algorithm/df_planning.yaml` (cube tree budget 500, best-of-50 replanning per cube move, one temporary position per cube, goal threshold 0.15) plus the per-size presets in `configurations/release/cube_presets.json` (horizons of 200 / 500 / 1000 / 1000 steps and the controller of each size).

The reported results aggregate tasks 1–5 and seeds 0–9: the success rate is averaged over the five tasks for each seed, then averaged over the ten seeds.

A saved plan can be re-executed with another controller seed to separate plan quality from controller variance (a dropped or knocked-over cube). For example, after evaluating the triple task 4 with seed 0 into `outputs/eval/cube-triple-task4-seed0`:

```bash
bash scripts/docker_run.sh python3 scripts/cmctd.py evaluate \
  --size triple --task 4 --seed 0 \
  --checkpoint outputs/downloaded/jaesikyoon/jsik_c_mctd/wouak5pu/model.ckpt \
  --output outputs/eval/replay-seed1 \
  --override algorithm.high_level_search=replay-plan \
  --override algorithm.replay_plan_path=/workspace/outputs/eval/cube-triple-task4-seed0/plan.npz \
  --override algorithm.controller_seed=1
```

### 3. Results

Success rate (%) over tasks 1–5 and seeds 0–9.

| Method | Single | Double | Triple | Quadruple |
|---|---|---|---|---|
| Replan | 98 | 50 | 58 | 32 |
| **C-MCTD** (Online Composer with Plan Cache) | **100** | **100** | **98** | **91** |

In this environment the outcome of a plan also depends on the controller: a sound plan can fail when a cube is dropped or knocked over, and a wrong plan can still succeed when the arm happens to push a cube into place. To account for this, the plan of each failed run is executed again with ten controller seeds, and the table reports the median success rate over these ten rounds.

## Training

### Diffusion planner

```bash
bash scripts/docker_run.sh python3 scripts/cmctd.py train-planner \
  --size double --seed 0 --output outputs/train/cube-double
```

The planners are trained on the cube play datasets (cube positions only). Checkpoints are written under `<output>/hydra/checkpoints/`, and the defaults are in `configurations/experiment/exp_planning.yaml`.

### Low-level controller

```bash
bash scripts/docker_run.sh python3 scripts/cmctd.py train-controller --size triple --seed 0
```

For `--size single`, this trains the single-cube Diffusion-QL controller on `cube-single-play-v0`. For the other sizes, it trains a target-cube controller on the play dataset of that size (600 epochs): following the object-wise planning, the cube that moves (the one that moves most in the sample's window) is always placed as the first cube of the controller input, next to the arm, with its future position as the goal. The controllers are written under `dql/results/`. To evaluate with another controller, pass `--override algorithm.controller_checkpoint_dir=<directory>` (and `algorithm.controller_checkpoint_epoch` if needed).

## Tests

```bash
CMCTD_GPUS=none bash scripts/docker_run.sh python3 -m unittest discover -s tests
```

The unit tests run on the CPU in about a second and need neither the model weights nor the OGBench environment: the cube tree search, its value estimation and the replan baseline with a stub planner (`tests/test_cube_search.py`), the CLI and the presets (`tests/test_release_cli.py`), and the target-cube sampler of the controller training (`tests/test_target_cube_sampler.py`).

## References

If you find our work useful, please consider citing:

```bibtex
@inproceedings{yoon2025compositional,
  title={Compositional Monte Carlo Tree Diffusion for Extendable Planning},
  author={Yoon, Jaesik and Cho, Hyeonseo and Ahn, Sungjin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgement

This implementation builds on [MCTD](https://github.com/ahn-ml/mctd) and [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing). The low-level controller is based on [Diffusion-QL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL).

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.
