"""Portable single-process training/evaluation; no SSH or W&B account required."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMPOSERS = {'online': 'online-composer', 'distributed': 'distributed-composer',
             'preplan': 'preplan-composer', 'replan': 'replan'}


def planner_config(maze, size, seed):
    import yaml
    dataset = f"og_{'maze2d' if maze == 'pointmaze' else 'antmaze'}_{size}_stitch"
    stats = yaml.safe_load((ROOT / 'configurations/dataset' / f'{dataset}.yaml').read_text())
    return {
        'experiment': 'exp_planning', 'algorithm': 'df_planning', 'dataset': dataset,
        'dataset.action_mean': [], 'dataset.action_std': [],
        'dataset.observation_mean': stats['observation_mean'][:2],
        'dataset.observation_std': stats['observation_std'][:2],
        'dataset.episode_len': 100, 'experiment.validation.seed': seed,
        'wandb.mode': 'disabled',
    }


def evaluation_config(maze, size, composer, task, seed):
    config = planner_config(maze, size, seed)
    presets = json.loads((ROOT / f'configurations/release/{maze}_presets.json').read_text())
    config.update(presets['common'])
    config.update(presets[size]['common'])
    config.update(presets[size][composer])
    if maze == 'pointmaze':
        config['algorithm.plan_length_max'] = 1000
    config.update({
        'algorithm.high_level_search': COMPOSERS[composer],
        'algorithm.open_loop_horizon': 1000, 'algorithm.val_max_steps': 1000,
        'algorithm.num_samples_best_of_n': 50,
        'algorithm.task_id': task, 'experiment.validation.seed': seed,
        'experiment.tasks': ['validation'], 'experiment.validation.batch_size': 1,
        'wandb.mode': 'disabled',
        '+name': f'{maze}-{size}-{composer}-task{task}-seed{seed}',
    })
    config.pop('load', None)
    return config


def hydra_value(value):
    if isinstance(value, str) and value.startswith('['):
        value = json.loads(value)  # Historical jobs encode arrays as strings.
    return json.dumps(value, separators=(',', ':'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train-planner', 'evaluate', 'train-controller'])
    parser.add_argument('--maze', choices=['pointmaze', 'antmaze'], default='pointmaze')
    parser.add_argument('--size', choices=['medium', 'large', 'giant'], default='medium')
    parser.add_argument('--composer', choices=COMPOSERS, default='online')
    parser.add_argument('--task', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job', type=Path, help='Exact flat Hydra JSON config for evaluation')
    parser.add_argument('--override', action='append', default=[], help='Repeatable Hydra override')
    parser.add_argument('--dry-run', action='store_true', help='Print command without writing or running')
    args = parser.parse_args(argv)
    if args.mode == 'evaluate' and args.checkpoint is None:
        parser.error('evaluate requires --checkpoint (local model.ckpt, not a W&B ID)')
    if args.output is None:
        parser.error('--output is required; use a new directory to preserve existing weights/results')
    if args.mode == 'train-controller' and (args.maze != 'antmaze' or args.job or args.override or args.checkpoint):
        parser.error('train-controller requires --maze antmaze; use dql.main_Antmaze for custom flags')
    if args.job and args.mode != 'evaluate':
        parser.error('--job is only supported for evaluate')
    return args


def main(argv=None):
    args = parse_args(argv)
    env = dict(os.environ)
    if args.mode == 'train-controller':
        command = [sys.executable, '-m', 'dql.main_Antmaze', '--env_name',
                   f'antmaze-{args.size}-navigate-v0', '--seed', str(args.seed), '--lr_decay']
        if args.output:
            command += ['--dir', str(args.output.resolve())]
    else:
        if args.mode == 'train-planner':
            config = planner_config(args.maze, args.size, args.seed)
            config.update({'experiment.tasks': ['training'],
                           '+name': f'{args.maze}-{args.size}-planner-seed{args.seed}'})
        elif args.job:
            config = json.loads(args.job.read_text())
            config.update({'wandb.mode': 'disabled', 'experiment.tasks': ['validation']})
        else:
            config = evaluation_config(args.maze, args.size, args.composer, args.task, args.seed)
        if args.checkpoint:
            if not args.dry_run and not args.checkpoint.is_file():
                raise FileNotFoundError(args.checkpoint)
            config['load'] = str(args.checkpoint.resolve())
        config['hydra.run.dir'] = str(args.output.resolve() / 'hydra')
        command = [sys.executable, '-m', 'main']
        command += [f'{key}={hydra_value(value)}' for key, value in config.items()]
        command += args.override
        if args.mode == 'evaluate':
            env['MCTD_PLAN_CAPTURE_DIR'] = str(args.output.resolve())
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'launch.json').write_text(json.dumps({
            'command': command, 'cwd': str(ROOT),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        }, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
