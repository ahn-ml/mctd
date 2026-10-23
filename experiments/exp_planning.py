from datasets import OGCubeOfflineRLDataset
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
        og_cube_single_play=OGCubeOfflineRLDataset,
        og_cube_double_play=OGCubeOfflineRLDataset,
        og_cube_triple_play=OGCubeOfflineRLDataset,
        og_cube_quadruple_play=OGCubeOfflineRLDataset,
    )
