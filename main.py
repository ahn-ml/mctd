"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.

Main file for the project. This will create and run new experiments and load local checkpoints.
Borrowed part of the code from David Charatan and wandb.
"""

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero


def run_local(cfg: DictConfig):
    validation_seed = cfg.experiment.validation.get("seed", None)
    if validation_seed is not None:
        random.seed(validation_seed)
        np.random.seed(validation_seed)
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)

    # delay some imports in case they are not needed
    from experiments import build_experiment

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to:"), output_dir)
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(output_dir, target_is_directory=True)

    # Optional logging with wandb (disabled by default).
    if cfg.wandb.mode != "disabled":
        from utils.wandb_utils import SpaceEfficientWandbLogger

        offline = cfg.wandb.mode != "online"
        logger = SpaceEfficientWandbLogger(
            name=f"{cfg.name} ({output_dir.parent.name}/{output_dir.name})",
            save_dir=str(output_dir),
            offline=offline,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            project=cfg.wandb.project or Path(__file__).parent.name,
            log_model="all" if not offline else False,
            config=OmegaConf.to_container(cfg),
        )
    else:
        logger = None

    # Load a local checkpoint
    checkpoint_path = cfg.get("load", None)
    if checkpoint_path and is_rank_zero:
        print(f"Will load checkpoint from {checkpoint_path}")

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    if "name" not in cfg:
        raise ValueError("must specify a name for the run with command line argument '+name=[name]'")
    load = cfg.get("load", None)
    if load and not Path(load).is_file():
        raise FileNotFoundError(f"load must be a local checkpoint file, got {load}")
    run_local(cfg)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
