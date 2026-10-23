"""Portable single-process training/evaluation on the OGBench cube tasks; no SSH or W&B account required."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIZES = ('single', 'double', 'triple', 'quadruple')


def planner_config(size, seed):
    return {
        'experiment': 'exp_planning', 'algorithm': 'df_planning', 'dataset': f'og_cube_{size}_play',
        'experiment.validation.seed': seed, 'wandb.mode': 'disabled',
    }


COMPOSERS = {'online': 'online-composer', 'replan': 'replan'}


def evaluation_config(size, task, seed, composer='online'):
    config = planner_config(size, seed)
    presets = json.loads((ROOT / 'configurations/release/cube_presets.json').read_text())
    config.update(presets[size])
    config.update({
        'algorithm.high_level_search': COMPOSERS[composer],
        'algorithm.task_id': task, 'experiment.validation.seed': seed,
        'experiment.tasks': ['validation'], 'experiment.validation.batch_size': 1,
        '+name': f'cube-{size}-{composer}-task{task}-seed{seed}',
    })
    return config


def hydra_value(value):
    return json.dumps(value, separators=(',', ':'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train-planner', 'evaluate', 'train-controller'])
    parser.add_argument('--size', choices=SIZES, default='single')
    parser.add_argument('--composer', choices=list(COMPOSERS), default='online',
                        help='online: Online Composer with the plan cache; replan: the baseline without search')
    parser.add_argument('--task', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--override', action='append', default=[], help='Repeatable Hydra override')
    parser.add_argument('--dry-run', action='store_true', help='Print command without writing or running')
    args = parser.parse_args(argv)
    if args.mode == 'evaluate' and args.checkpoint is None:
        parser.error('evaluate requires --checkpoint (local model.ckpt)')
    if args.mode != 'train-controller' and args.output is None:
        parser.error('--output is required; use a new directory to preserve existing weights/results')
    if args.mode == 'train-controller' and (args.override or args.checkpoint):
        parser.error('train-controller takes --size and --seed only; use dql.main_Cube for custom flags')
    return args


def main(argv=None):
    args = parse_args(argv)
    env = dict(os.environ)
    if args.mode == 'train-controller':
        # dql.main_Cube writes to dql/results/: the single-cube controller of the single tasks, and a target-cube
        # controller (the arm and the cube it moves, from the multi-cube data) for each larger size
        if args.size == 'single':
            command = [sys.executable, '-m', 'dql.main_Cube', '--env_name', 'cube-single-play-v0',
                       '--goal_dim', '3', '--seed', str(args.seed), '--lr_decay']
        else:
            command = [sys.executable, '-m', 'dql.main_Cube', '--env_name', f'cube-{args.size}-play-v0',
                       '--target_cube', '--seed', str(args.seed), '--num_epochs', '600']
    else:
        if args.mode == 'train-planner':
            config = planner_config(args.size, args.seed)
            config.update({'experiment.tasks': ['training'], '+name': f'cube-{args.size}-planner-seed{args.seed}'})
        else:
            config = evaluation_config(args.size, args.task, args.seed, args.composer)
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
