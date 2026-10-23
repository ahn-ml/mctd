"""CPU tests of the release CLI and presets."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cmctd  # noqa: E402


class ReleaseTests(unittest.TestCase):
    def test_presets_set_the_horizons(self):
        horizon = {'single': 200, 'double': 500, 'triple': 1000, 'quadruple': 1000}
        for size in cmctd.SIZES:
            config = cmctd.evaluation_config(size, 2, 3)
            self.assertEqual(config['dataset'], f'og_cube_{size}_play')
            self.assertEqual(config['algorithm.val_max_steps'], horizon[size])
            self.assertEqual(config['algorithm.open_loop_horizon'], horizon[size])
            self.assertEqual(config['algorithm.cube_plan_max_steps'], horizon[size])
            self.assertEqual(config['dataset.episode_len'], 200 if size == 'single' else 500)
            self.assertEqual((config['algorithm.task_id'], config['experiment.validation.seed']), (2, 3))

    def test_release_defaults(self):
        import yaml
        defaults = yaml.safe_load((ROOT / 'configurations/algorithm/df_planning.yaml').read_text())
        self.assertEqual(defaults['high_level_search'], 'online-composer')
        self.assertEqual((defaults['mctd_max_search_num'], defaults['num_samples_best_of_n']), (500, 50))
        self.assertTrue(defaults['cube_plan_cache'])
        self.assertEqual((defaults['parallel_search_num'], defaults['virtual_visit_weight']), (4, 1.0))
        self.assertEqual(defaults['cube_num_temporary_positions'], 1)

    def test_all_presets_compose(self):
        from hydra import compose, initialize_config_dir
        for size in cmctd.SIZES:
            config = cmctd.evaluation_config(size, 1, 0)
            with self.subTest(size=size), initialize_config_dir(config_dir=str(ROOT / 'configurations'), version_base=None):
                cfg = compose(config_name='config', overrides=[f'{k}={cmctd.hydra_value(v)}' for k, v in config.items()])
                self.assertEqual(len(cfg.dataset.observation_mean), 3 * (cmctd.SIZES.index(size) + 1))
                self.assertEqual(cfg.wandb.mode, 'disabled')

    def test_dry_run_prints_the_command(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            cmctd.main(['evaluate', '--size', 'double', '--task', '4', '--seed', '7', '--checkpoint', '/tmp/model.ckpt',
                        '--output', '/tmp/cmctd-dry-run', '--dry-run'])
        command = stream.getvalue()
        self.assertIn('og_cube_double_play', command)
        self.assertIn('algorithm.task_id=4', command)
        self.assertFalse(Path('/tmp/cmctd-dry-run').exists())

    def test_composer_selects_the_search(self):
        self.assertEqual(cmctd.evaluation_config('triple', 1, 0)['algorithm.high_level_search'], 'online-composer')
        self.assertEqual(cmctd.evaluation_config('triple', 1, 0, 'replan')['algorithm.high_level_search'], 'replan')

    def test_presets_use_the_target_cube_controllers(self):
        self.assertNotIn('algorithm.controller_checkpoint_dir', cmctd.evaluation_config('single', 1, 0))
        for size in ('double', 'triple', 'quadruple'):
            config = cmctd.evaluation_config(size, 1, 0)
            self.assertTrue(config['algorithm.controller_checkpoint_dir'].startswith(f'dql/results/cube-{size}-play-v0|'))
            self.assertTrue(config['algorithm.controller_checkpoint_dir'].endswith('|target_cube'))
            self.assertEqual(config['algorithm.controller_checkpoint_epoch'], 600)

    def test_train_controller_per_size(self):
        for size, flags in (('single', ['cube-single-play-v0', '--lr_decay']), ('triple', ['cube-triple-play-v0', '--target_cube'])):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                cmctd.main(['train-controller', '--size', size, '--dry-run'])
            for flag in flags:
                self.assertIn(flag, stream.getvalue())

    def test_presets_cover_every_size(self):
        presets = json.loads((ROOT / 'configurations/release/cube_presets.json').read_text())
        self.assertEqual([k for k in presets if not k.startswith('_')], list(cmctd.SIZES))


if __name__ == '__main__':
    unittest.main()
