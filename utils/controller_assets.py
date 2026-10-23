"""Resolve the Navigate-trained controllers used by AntMaze composers."""
from pathlib import Path


def antmaze_controller_dir(dataset, directory=None):
    if directory is not None:
        return Path(directory).expanduser()
    fields = dataset.split('-')
    if len(fields) != 4 or fields[0] != 'antmaze' or fields[1] not in ('medium', 'large', 'giant'):
        raise ValueError(f'No released controller for {dataset}; set algorithm.controller_checkpoint_dir')
    folder = (f'antmaze-{fields[1]}-navigate-v0|exp|diffusion-ql|T-5|lr_decay|'
              'ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10')
    return Path('dql/results') / folder
