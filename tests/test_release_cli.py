"""CPU-only tests: compose every public preset without loading data or models."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import cmctd


class ReleaseTests(unittest.TestCase):
    def test_controller_paths(self):
        from utils.controller_assets import antmaze_controller_dir
        for size in ('medium', 'large', 'giant'):
            for dataset in ('stitch', 'navigate'):
                self.assertEqual(str(antmaze_controller_dir(f'antmaze-{size}-{dataset}-v0')),
                                 f'dql/results/antmaze-{size}-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10')
        self.assertEqual(antmaze_controller_dir('antmaze-medium-stitch-v0', '/tmp/new-controller'),
                         Path('/tmp/new-controller'))

    def test_all_presets_compose(self):
        from hydra import compose, initialize_config_dir
        for maze in ('pointmaze', 'antmaze'):
            for size in ('medium', 'large', 'giant'):
                configs = [cmctd.planner_config(maze, size, 3)]
                configs += [cmctd.evaluation_config(maze, size, c, 2, 3) for c in cmctd.COMPOSERS]
                for config in configs:
                    with self.subTest(maze=maze, size=size, method=config.get('algorithm.high_level_search')):
                        with initialize_config_dir(config_dir=str(cmctd.ROOT / 'configurations'), version_base=None):
                            cfg = compose(config_name='config', overrides=[
                                f'{key}={cmctd.hydra_value(value)}' for key, value in config.items()])
                        self.assertEqual(len(cfg.dataset.observation_mean), 2)
                        self.assertEqual(list(cfg.dataset.action_mean), [])
                        self.assertEqual(cfg.experiment.validation.seed, 3)
                        self.assertEqual(cfg.wandb.mode, 'disabled')

    def test_ant_protocol(self):
        rank = {'medium': 1, 'large': 1, 'giant': 5}
        warp = {'medium': 0.5, 'large': 0.5, 'giant': 1.0}
        for size in ('medium', 'large', 'giant'):
            for composer in cmctd.COMPOSERS:
                config = cmctd.evaluation_config('antmaze', size, composer, 1, 0)
                self.assertEqual(config['algorithm.plan_length_max'], 1000 if composer in ('online', 'replan') else 800)
                self.assertEqual(config['algorithm.composer_plan_rank'], rank[size])
                self.assertEqual(config['algorithm.replan_plan_rank'], rank[size])
                self.assertEqual(config['algorithm.threshold_for_warp'], warp[size])
                self.assertEqual(config['algorithm.controller_seed'], 0)
                self.assertNotIn('algorithm.cmctd_guidance_set', config)
        giant = cmctd.evaluation_config('antmaze', 'giant', 'preplan', 1, 0)
        self.assertEqual(giant['algorithm.num_graph_nodes'], 70)
        self.assertEqual(giant['algorithm.preplan_edge_threshold_for_warp'], 0.5)
        self.assertTrue(cmctd.evaluation_config('antmaze', 'large', 'replan', 1, 0)['algorithm.replan_open_loop'])

    def test_release_defaults(self):
        import yaml
        defaults = yaml.safe_load((cmctd.ROOT / 'configurations/algorithm/df_planning.yaml').read_text())
        self.assertEqual(defaults['low_level_search'], 'mctd')
        self.assertEqual(defaults['children_guidance_sets'], [[0, 0, 0.5], [0, 0.5, 1], [0.5, 1], [0.5, 1, 1.5], [1, 1.5, 2]])
        self.assertEqual(defaults['early_stopping_condition'], 'solved')
        self.assertEqual(defaults['parallel_num'], 300)
        self.assertTrue(defaults['compress_still_steps'])

    def test_pointmaze_plan_cap(self):
        for size in ('medium', 'large', 'giant'):
            for composer in cmctd.COMPOSERS:
                config = cmctd.evaluation_config('pointmaze', size, composer, 1, 0)
                self.assertEqual(config['algorithm.plan_length_max'], 1000)
                self.assertEqual(config['algorithm.val_max_steps'], 1000)

    def test_pointmaze_presets(self):
        warp = {'medium': 0.5, 'large': 0.5, 'giant': 1.0}
        budget = {'medium': 1000, 'large': 1000, 'giant': 2000}
        for size in ('medium', 'large', 'giant'):
            for composer in cmctd.COMPOSERS:
                config = cmctd.evaluation_config('pointmaze', size, composer, 1, 0)
                self.assertEqual(config['algorithm.threshold_for_warp'], warp[size])
                self.assertEqual(config['algorithm.high_level_search'], cmctd.COMPOSERS[composer])
            self.assertEqual(cmctd.evaluation_config('pointmaze', size, 'online', 1, 0)['algorithm.mctd_max_search_num'],
                             budget[size])
        self.assertEqual(list(cmctd.COMPOSERS), ['online', 'distributed', 'preplan', 'replan'])

    def test_dry_run_does_not_write_or_launch(self):
        with tempfile.TemporaryDirectory() as folder, patch('scripts.cmctd.subprocess.run') as run:
            output = Path(folder) / 'new'
            with contextlib.redirect_stdout(io.StringIO()):
                cmctd.main(['train-planner', '--output', str(output), '--dry-run'])
            self.assertFalse(output.exists())
            run.assert_not_called()

    def test_controller_uses_navigate(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            cmctd.main(['train-controller', '--maze', 'antmaze', '--size', 'large',
                        '--output', '/tmp/controller-test', '--dry-run'])
        self.assertIn('antmaze-large-navigate-v0', stream.getvalue())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch('scripts.cmctd.subprocess.run') as run:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                cmctd.main(['train-planner', '--output', folder])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
