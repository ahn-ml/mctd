from datasets import OGMaze2dOfflineRLDataset, OGAntMazeOfflineRLDataset
from algorithms.diffusion_forcing import DiffusionForcingPlanning
from .exp_base import BaseLightningExperiment


class PlanningExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_planning=DiffusionForcingPlanning,
    )

    compatible_datasets = dict(
        og_maze2d_medium_stitch=OGMaze2dOfflineRLDataset,
        og_maze2d_large_stitch=OGMaze2dOfflineRLDataset,
        og_maze2d_giant_stitch=OGMaze2dOfflineRLDataset,
        og_antmaze_medium_stitch=OGAntMazeOfflineRLDataset,
        og_antmaze_large_stitch=OGAntMazeOfflineRLDataset,
        og_antmaze_giant_stitch=OGAntMazeOfflineRLDataset,
    )
