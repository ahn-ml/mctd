"""CPU regressions for rejected candidates, using production planning methods.

Extract methods via AST to avoid importing training/model dependencies or
loading a checkpoint; diffusion outputs are supplied by small test fixtures.
"""

import ast
import collections
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from utils.graph_routes import shortest_simple_routes  # noqa: E402
NODE_FILE = ROOT / 'algorithms/diffusion_forcing/high_level_tree_node.py'
spec = importlib.util.spec_from_file_location('tested_tree_node', NODE_FILE)
node_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node_module)
HighLevelTreeNode = node_module.HighLevelTreeNode
LOW_NODE_FILE = ROOT / 'algorithms/diffusion_forcing/low_level_tree_node.py'
spec = importlib.util.spec_from_file_location('tested_low_level_tree_node', LOW_NODE_FILE)
low_node_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(low_node_module)
LowLevelTreeNode = low_node_module.LowLevelTreeNode
PLANNER_FILE = ROOT / 'algorithms/diffusion_forcing/df_planning.py'
source = ast.parse(PLANNER_FILE.read_text())
planner_class = next(n for n in source.body if isinstance(n, ast.ClassDef)
                     and n.name == 'DiffusionForcingPlanning')
methods = {
    'best_of_n', '_select_candidates', '_score_candidates', '_finalize_candidate', 'mctd_segments', '_novel_arclength', '_drop_still_steps', '_jumpy_noise_level', 'expand_via_stitching', 'simulate_with_fast_replanning',
    '_retain_valid_expansions', '_waypoint_graph', 'plan_tree_search',
}
planner_class.bases = []
planner_class.body = [n for n in planner_class.body
                      if isinstance(n, ast.FunctionDef) and n.name in methods]
namespace = dict(np=np, torch=torch, os=os, dijkstra=dijkstra, collections=collections,
                 shortest_simple_routes=shortest_simple_routes,
                 HighLevelTreeNode=HighLevelTreeNode, LowLevelTreeNode=LowLevelTreeNode, OVERRUN_BONUS=0.05)
exec(compile(ast.Module(body=[planner_class], type_ignores=[]), str(PLANNER_FILE), 'exec'),
     namespace)
Planner = namespace['DiffusionForcingPlanning']


def planner(bank):
    """bank is [candidate, time, xy], excluding the dummy last observation."""
    bank = torch.as_tensor(bank, dtype=torch.float32)
    obj = Planner()
    obj.device = torch.device('cpu')
    obj.observation_dim = 2
    obj.data_mean = torch.zeros(2)
    obj.data_std = torch.ones(2)
    obj.num_samples_best_of_n = len(bank)
    obj.episode_len = bank.shape[1]
    obj.threshold_for_warp = 0.5
    obj.low_level_search = 'best-of-n'
    obj.parallel_search_num = 1
    obj.children_guidance_sets = [[0, 1], [0, 1]]
    obj.compress_still_steps = False
    obj.still_step_threshold = 0.03
    obj.simulation_overrun_ratio = 0.0
    obj.simulation_jumpy_interval = 1
    obj.simulation_low_level_search = 'best-of-n'
    obj.early_stopping_condition = 'achieved'
    obj.novelty_radius = 1.0
    obj.novelty_recent_arclength = 2.0
    obj.virtual_visit_weight = 1.0
    obj.composer_plan_rank = 1
    obj.plan_length_max = obj.val_max_steps = 100
    obj._unnormalize_x = lambda x: x
    obj.split_bundle = lambda x: (x, None, None)
    padded = torch.cat([bank, bank[:, -1:]], dim=1)
    # one copy of the bank per batch item, laid out (sample, item) as best_of_n expects
    obj.plan = lambda start, *args, **kwargs: padded.repeat_interleave(
        max(1, len(start) // len(bank)), dim=0).transpose(0, 1)[None]
    return obj


def root():
    return HighLevelTreeNode('0', 0, None, [[0, 1], [0, 1]], [])


def sample(obj, history=None, goal=(10., 0.)):
    starts = np.array([[0., 0.]])
    goals = np.array([goal])
    return obj.best_of_n([history or []], starts, goals, torch.tensor(starts),
                         torch.tensor(goals), 100, None, 100)


class RejectedCandidateTests(unittest.TestCase):
    def test_all_warp_returns_no_candidate(self):
        obj = planner([[[2, 0], [2.1, 0]], [[3, 0], [3.1, 0]]])
        plans, names, values, solved = sample(obj)
        self.assertEqual(len(plans[0]), 0)
        self.assertEqual(names, ['NoValidCandidate'])
        self.assertEqual(values, [0.0])
        self.assertEqual(solved, [False])

    def test_valid_zero_score_beats_rejected_zero_score(self):
        obj = planner([[[2, 0], [2.1, 0]], [[0, 0], [0, 0]]])
        plans, names, values, solved = sample(obj)
        self.assertEqual(names, ['NotReached'])
        self.assertTrue(torch.equal(plans[0], torch.zeros(2, 2)))
        self.assertEqual(values, [0.0])
        self.assertEqual(solved, [False])

    def test_close_candidates_remain_selectable_at_zero_value(self):
        obj = planner(np.zeros((2, 24, 2)))
        plans, names, values, solved = sample(obj, [torch.zeros(24, 2)])
        self.assertEqual(len(plans[0]), 24)
        self.assertEqual(names, ['Close'])
        self.assertEqual(values, [0.0])
        self.assertEqual(solved, [False])

    def test_progressing_candidate_beats_close_candidate(self):
        moving = np.stack([np.linspace(0, 4.6, 24), np.zeros(24)], axis=1)
        obj = planner(np.stack([np.zeros((24, 2)), moving]))
        plans, names, values, _ = sample(obj, [torch.zeros(24, 2)])
        self.assertEqual(names, ['NotReached'])
        self.assertTrue(torch.allclose(plans[0], torch.as_tensor(moving, dtype=torch.float32)))
        self.assertGreater(values[0], 0)

    def test_valid_solution_still_selected_and_truncated(self):
        obj = planner([[[2, 0], [2.1, 0]], [[0.2, 0], [0.4, 0]]])
        plans, names, values, solved = sample(obj, goal=(0.8, 0.))
        self.assertEqual(names, ['Achieved'])
        self.assertEqual(solved, [True])
        self.assertEqual(len(plans[0]), 2)
        self.assertGreater(values[0], 0)

    def test_rejected_branch_never_expands_again(self):
        r = root()
        bad = r.expand()
        bad.reject()
        good = r.expand()
        good.set_value(0.2)
        r.backpropagate()
        self.assertIs(r.select_expandable(3), good)
        with self.assertRaises(ValueError):
            bad.expand()
        good.reject()
        self.assertIsNone(r.select_expandable(3))

    def test_filter_preserves_batch_identity_and_no_invalid_history(self):
        roots = [root(), root()]
        nodes = [r.expand() for r in roots]
        result = Planner._retain_valid_expansions(
            roots, nodes, [3, 9], [0., 0.1], ['NoValidCandidate', 'NotReached'],
            [torch.empty(0, 2), torch.zeros(2, 2)], [False, False])
        self.assertEqual(result[2], [9])
        self.assertIs(result[1][0], nodes[1])
        self.assertTrue(nodes[0].invalid)
        self.assertEqual(nodes[0].plan_history, [])

    def test_value_rollout_stops_on_rejection(self):
        obj = planner([[[2, 0], [2.1, 0]]])
        node = root().expand()
        node.set_local_name('NotReached')
        node.set_plan_history(torch.zeros(2, 2))
        values, names, plans, solved = obj.simulate_with_fast_replanning(
            [node], np.zeros((1, 2)), np.array([[10., 0.]]), 100, None, 8)
        self.assertEqual(plans, [[]])
        self.assertEqual(solved, [False])
        self.assertIn('NoValidCandidate', names[0])

    def _mctd_planner(self, bank, budget):
        obj = planner(bank)
        obj.cfg = {'mctd_node_search_num': budget, 'mctd_node_denoising_steps': 20, 'mctd_node_jumpy_interval': 10}
        obj.frame_stack = 10
        obj.episode_len = 100
        obj._generate_scheduling_matrix = lambda tokens: np.arange(59, -1, -1)[:, None].repeat(tokens, 1)
        calls = []
        plan = obj.plan
        obj.plan = lambda *a, **k: (calls.append(1), plan(*a, **k))[1]
        return obj, calls

    def test_mctd_node_planner_returns_a_goal_reaching_segment(self):
        obj, calls = self._mctd_planner([[[0.2, 0], [0.4, 0]]], budget=4)
        starts, goals = np.array([[0., 0.]]), np.array([[0.8, 0.]])
        plans, names, values, solved = obj.mctd_segments([[]], starts, goals, torch.tensor(starts), torch.tensor(goals),
                                                         100, None, 100, [[0.5, 1]])
        self.assertEqual(names, ['Achieved'])
        self.assertEqual(solved, [True])
        self.assertEqual(len(calls), 2)  # one inner expansion (partial denoising + jumpy rollout), then it stops

    def test_mctd_node_planner_respects_its_budget(self):
        obj, calls = self._mctd_planner([[[0.1, 0], [0.2, 0]]], budget=3)
        starts, goals = np.array([[0., 0.]]), np.array([[10., 0.]])
        plans, names, values, solved = obj.mctd_segments([[]], starts, goals, torch.tensor(starts), torch.tensor(goals),
                                                         100, None, 100, [[0.5, 1]])
        self.assertEqual(solved, [False])
        self.assertLessEqual(len(calls), 3 * 2)

    def test_mctd_node_planner_batches_all_trees(self):
        obj, calls = self._mctd_planner([[[0.2, 0], [0.4, 0]]], budget=3)
        sizes = []
        plan = obj.plan
        obj.plan = lambda start, *a, **k: (sizes.append(len(start)), plan(start, *a, **k))[1]
        starts, goals = np.array([[0., 0.], [0., 0.]]), np.array([[0.8, 0.], [10., 0.]])
        plans, names, values, solved = obj.mctd_segments([[], []], starts, goals, torch.tensor(starts),
                                                         torch.tensor(goals), 100, None, 100, [[0.5, 1], [0.5, 1]])
        self.assertEqual(solved, [True, False])
        self.assertEqual(sizes[:2], [2, 2])  # first iteration: one child of each tree in one batch
        self.assertTrue(all(n == 1 for n in sizes[2:]))  # then only the unsolved tree

    def test_mctd_node_planner_is_parallel_mctd(self):
        obj, calls = self._mctd_planner([[[0.1, 0], [0.2, 0]]], budget=5)
        obj.cfg['mctd_node_parallel_num'] = 2
        obj.virtual_visit_weight = 1.0
        claims = []  # (batch size, guidance) of each partial-denoising call
        plan = obj.plan
        obj.plan = lambda start, *a, noise_level=None, guidance_scale=None, **k: (
            len(noise_level) >= 20 and claims.append((len(start), guidance_scale.tolist())),
            plan(start, *a, noise_level=noise_level, guidance_scale=guidance_scale, **k))[1]
        starts, goals = np.array([[0., 0.]]), np.array([[10., 0.]])
        obj.mctd_segments([[]], starts, goals, torch.tensor(starts), torch.tensor(goals), 100, None, 100, [[0.5, 1]])
        self.assertEqual(claims[0], (2, [0.5, 1]))  # both child slots of the root in one batch
        self.assertEqual(sum(n for n, _ in claims), 5)  # the budget counts children
        self.assertEqual([n for n, _ in claims], [2, 2, 1])

    def test_mctd_node_planner_repeats_child_slots_per_guidance(self):
        obj, calls = self._mctd_planner([[[0.1, 0], [0.2, 0]]], budget=4)
        obj.cfg.update(mctd_node_parallel_num=4, mctd_node_child_repeat=2)
        obj.virtual_visit_weight = 1.0
        claims = []
        plan = obj.plan
        obj.plan = lambda start, *a, noise_level=None, guidance_scale=None, **k: (
            len(noise_level) >= 20 and claims.append(guidance_scale.tolist()),
            plan(start, *a, noise_level=noise_level, guidance_scale=guidance_scale, **k))[1]
        starts, goals = np.array([[0., 0.]]), np.array([[10., 0.]])
        obj.mctd_segments([[]], starts, goals, torch.tensor(starts), torch.tensor(goals), 100, None, 100, [[0.5, 1]])
        self.assertEqual(claims, [[0.5, 1, 0.5, 1]])  # the root's four slots in one batch

    def test_simulation_can_use_the_mctd_node_planner(self):
        obj, calls = self._mctd_planner([[[0.2, 0], [0.4, 0]]], budget=4)
        obj.simulation_low_level_search = 'mctd'
        obj.cfg['simulation_mctd_node_search_num'] = 2
        seen = []
        segments = obj.mctd_segments
        obj.mctd_segments = lambda *a, settings=None, **k: (seen.append(settings), segments(*a, settings=settings, **k))[1]
        obj.best_of_n = lambda *a, **k: self.fail('simulation used best-of-N')
        root = HighLevelTreeNode('root', 0, None, [[0.5, 1]], [], guidance_set=[0.5, 1])
        node = HighLevelTreeNode('n', 1, root, [[0.5, 1]], [], guidance_set=[0.5, 1])
        node.local_name = 'n'
        with contextlib.redirect_stdout(io.StringIO()):
            values, names, plans, solveds = obj.simulate_with_fast_replanning(
                [node], np.array([[0., 0.]]), np.array([[0.8, 0.]]), 100, None, 200)
        self.assertEqual(solveds, [True])
        self.assertEqual(seen[0]['mctd_node_search_num'], 2)
        self.assertIsNone(seen[0]['mctd_node_parallel_num'])  # falls back to the expansion value

    def test_mctd_node_settings_override_the_config(self):
        obj, calls = self._mctd_planner([[[0.1, 0], [0.2, 0]]], budget=10)
        starts, goals = np.array([[0., 0.]]), np.array([[10., 0.]])
        obj.mctd_segments([[]], starts, goals, torch.tensor(starts), torch.tensor(goals), 100, None, 100, [[0.5, 1]],
                          settings={'mctd_node_search_num': 1, 'mctd_node_parallel_num': None})
        self.assertEqual(len(calls), 2)  # one child: partial denoising + rollout

    def test_mctd_node_planner_splits_large_diffusion_batches(self):
        obj, calls = self._mctd_planner([[[0.1, 0], [0.2, 0]]], budget=4)
        obj.cfg.update(mctd_node_parallel_num=4, mctd_node_child_repeat=2, mctd_node_max_batch=3)
        sizes = []
        plan = obj.plan
        obj.plan = lambda start, *a, **k: (sizes.append(len(start)), plan(start, *a, **k))[1]
        starts, goals = np.array([[0., 0.], [0., 0.]]), np.array([[10., 0.], [10., 0.]])
        obj.mctd_segments([[], []], starts, goals, torch.tensor(starts), torch.tensor(goals), 100, None, 100,
                          [[0.5, 1], [0.5, 1]])
        self.assertEqual(sizes[:3], [3, 3, 2])  # 8 root children (2 trees x 4 slots) in batches of at most 3
        self.assertTrue(max(sizes) <= 3)

    def test_candidate_rank_picks_the_kth_best_goal_reaching_candidate(self):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        # three candidates reach the goal (1.2, 0) at steps 1, 2 and 3, one does not
        bank = torch.tensor([[[0.4, 0], [0.8, 0], [0.8, 0], [0.8, 0]], [[0.3, 0], [0.6, 0], [0.9, 0], [0.9, 0]],
                             [[0.2, 0], [0.4, 0], [0.6, 0], [0.8, 0]], [[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]]])
        o = bank.transpose(0, 1)[:, :, None]  # (t, m, 1, c)
        g = np.zeros((4, 1))
        args = (o, [[]], np.zeros((1, 2)), np.array([[1.2, 0.]]), 4, 100, g)
        best = obj._select_candidates(*args)[0][0]
        second = obj._select_candidates(*args, rank=2)[0][0]
        fifth = obj._select_candidates(*args, rank=5)[0][0]
        self.assertEqual(len(best), 2)  # reaches the goal at step 1
        self.assertEqual(len(second), 3)
        self.assertEqual(len(fifth), 4)  # fewer than five reach it: the longest reaching one

    def test_batched_candidate_scoring_matches_the_loop_reference(self):
        from reference_select_candidates import select_candidates_reference
        rng = np.random.default_rng(0)
        for trial in range(6):
            obj = planner([[[0.1, 0], [0.2, 0]]])
            obj.compress_still_steps = bool(trial % 2)
            t, m, b = 30, 12, 3
            steps = rng.normal(0, 0.2, (t, m, b, 2))
            steps[rng.random((t, m, b)) < 0.01] += 1.0  # some warps
            o = torch.tensor(np.cumsum(steps, 0), dtype=torch.float32)
            starts = np.zeros((b, 2))
            goals = rng.normal(0, 1.5, (b, 2))
            histories = [np.cumsum(rng.normal(0, 0.2, (25, 2)), 0) if (trial + i) % 3 else [] for i in range(b)]
            guidance = rng.choice([0., 0.5, 1.], (m, b))
            new = obj._select_candidates(o, [h if len(h) else [] for h in histories], starts, goals, t, 200, guidance)
            ref = select_candidates_reference(obj, o, [h if len(h) else [] for h in histories], starts, goals, t, 200,
                                              guidance)
            self.assertEqual(new[1], ref[1])
            self.assertEqual(new[3], ref[3])
            np.testing.assert_allclose(new[2], ref[2], rtol=1e-5, atol=1e-7)
            for a, r in zip(new[0], ref[0]):
                self.assertTrue(torch.equal(a, r))

    def test_rejected_simulation_keeps_only_preceding_valid_prefix(self):
        obj = planner([[[0, 0], [0, 0]]])
        node = root().expand()
        node.set_local_name('NotReached')
        node.set_plan_history(torch.zeros(2, 2))
        calls = []
        prefix = torch.tensor([[0.1, 0.], [0.2, 0.]])

        def sample_then_reject(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return [prefix], ['NotReached'], [0.01], [False]
            return [torch.empty(0, 2)], ['NoValidCandidate'], [0.], [False]

        obj.best_of_n = sample_then_reject
        _, _, plans, solved = obj.simulate_with_fast_replanning(
            [node], np.zeros((1, 2)), np.array([[10., 0.]]), 100, None, 10)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(plans[0], prefix))
        self.assertEqual(solved, [False])

    def test_rejected_plans_cannot_make_waypoint_edges(self):
        obj = planner([[[2, 0], [2.1, 0]]])
        graphs = []
        waypoint_graph = obj._waypoint_graph
        obj._waypoint_graph = lambda plans: (graphs.append(plans), waypoint_graph(plans))[1]
        with contextlib.redirect_stdout(io.StringIO()):
            candidates = obj.plan_tree_search(
                np.array([[0., 0.]]), np.array([[4., 0.]]), 100, None,
                max_steps=4, max_search_num=10, waypoints=np.array([[2., 0.]]))
        self.assertTrue(all(plans == [[[]]] for plans in graphs))
        self.assertEqual(candidates, [])

class ParallelSearchTests(unittest.TestCase):
    def test_sequential_selection_matches_uct_without_virtual_visits(self):
        r = root()
        a, b = r.expand(), r.expand()
        a.set_value(0.5); b.set_value(0.1)
        r.backpropagate()
        self.assertIs(r.select_expandable(3, exp_weight=0.01), a)

    def test_pending_children_are_skipped_within_a_batch(self):
        r = root()
        r.expand(), r.expand()  # both claimed in this batch, no value yet
        self.assertIsNone(r.select_expandable(3, exp_weight=0.01, virtual_visit_weight=1.0))

    def test_virtual_visits_spread_selections_and_are_cleared(self):
        r = root()
        a, b = r.expand(), r.expand()
        a.set_value(0.2); b.set_value(0.19)
        a.visit_count = b.visit_count = 1
        first = r.select_expandable(3, exp_weight=1.0, virtual_visit_weight=10.0)
        self.assertIs(first, a)
        first.add_virtual_visit()
        self.assertIs(r.select_expandable(3, exp_weight=1.0, virtual_visit_weight=10.0), b)
        first.clear_virtual_visits()
        self.assertEqual((a.virtual_visit_count, r.virtual_visit_count), (0, 0))

    def test_parallel_batch_expands_several_nodes_per_iteration(self):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        obj.parallel_search_num = 2
        batch_sizes = []

        def rollout(nodes, starts, goals, *args):
            batch_sizes.append(len(nodes))
            return ([0.] * len(nodes), ['NotReached'] * len(nodes),
                    [torch.tensor(start[None] + 0.1, dtype=torch.float32) for start in starts],
                    [False] * len(nodes))

        obj.expand_via_stitching = rollout
        obj.simulate_with_fast_replanning = lambda nodes, *args: (
            [0.] * len(nodes), ['Test'] * len(nodes), [[] for _ in nodes], [False] * len(nodes))
        with contextlib.redirect_stdout(io.StringIO()):
            obj.plan_tree_search(np.array([[0., 0.]]), np.array([[10., 0.]]), 100, None,
                                 max_steps=6, max_search_num=4)
        self.assertEqual(batch_sizes[0], 2)  # both root children in the first batch
        self.assertEqual(sum(batch_sizes), 4)  # budget counts selected nodes


    def _rollout_solves(self, condition):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        obj.early_stopping_condition = condition
        calls = []

        def expand(nodes, starts, goals, *args):
            calls.append(len(nodes))
            return ([0.] * len(nodes), ['NotReached'] * len(nodes),
                    [torch.tensor(start[None] + 0.1, dtype=torch.float32) for start in starts],
                    [False] * len(nodes))

        obj.expand_via_stitching = expand
        obj.simulate_with_fast_replanning = lambda nodes, *args: (
            [0.5] * len(nodes), ['Test'] * len(nodes), [torch.zeros(3, 2) for _ in nodes], [True] * len(nodes))
        with contextlib.redirect_stdout(io.StringIO()):
            plans, lengths = obj.plan_tree_search(np.array([[0., 0.]]), np.array([[10., 0.]]), 100, None,
                                                  max_steps=100, max_search_num=4, early_stopping=True)
        return calls, plans, lengths

    def _distributed_rollout_solves(self, condition):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        obj.early_stopping_condition = condition
        obj.cmctd_threshold_for_reaching = 1.0
        calls = []

        def expand(nodes, starts, goals, *args):
            calls.append(len(nodes))
            return ([0.] * len(nodes), ['NotReached'] * len(nodes),
                    [torch.tensor(start[None] + 0.1, dtype=torch.float32) for start in starts],
                    [False] * len(nodes))

        obj.expand_via_stitching = expand
        obj.simulate_with_fast_replanning = lambda nodes, *args: (
            [0.5] * len(nodes), ['Test'] * len(nodes), [torch.zeros(3, 2) for _ in nodes], [True] * len(nodes))
        with contextlib.redirect_stdout(io.StringIO()):
            candidates = obj.plan_tree_search(
                np.array([[0., 0.], [5., 5.]]), np.array([[10., 0.], [10., 0.]]), 100, None,
                max_steps=100, max_search_num=4, early_stopping=True, waypoints=np.array([[5., 5.], [10., 0.]]))
        return calls, candidates

    def test_distributed_takes_rollout_plans_when_achieved(self):
        calls, candidates = self._distributed_rollout_solves('achieved')
        self.assertEqual(len(calls), 1)  # the first rollout solution ends the search
        self.assertEqual(candidates[0][1], ('direct-value',))

    def test_distributed_keeps_rollout_plans_as_a_fallback_when_solved(self):
        calls, candidates = self._distributed_rollout_solves('solved')
        self.assertGreater(len(calls), 1)  # the search went on
        self.assertEqual(len(candidates), 1)  # and fell back to a rollout candidate

    def test_online_rank_takes_the_kth_shortest_over_distinct_branches(self):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        obj.early_stopping_condition = 'solved'
        obj.cfg = {'plan_rank_branch_depth': 1}
        obj.children_guidance_sets = [[0], [1], [2], [3]]
        lengths = iter([3, 5, 4, 9, 9, 9, 9, 9])
        branches = []

        def expand(nodes, starts, goals, *args):
            branches.extend(n.name.split('-')[1] for n in nodes)
            plans = [torch.zeros(next(lengths), 2) for _ in nodes]
            return [0.] * len(nodes), ['Achieved'] * len(nodes), plans, [True] * len(nodes)

        obj.expand_via_stitching = expand
        obj.simulate_with_fast_replanning = lambda nodes, *args: (
            [0.] * len(nodes), ['Test'] * len(nodes), [torch.zeros(0, 2) for _ in nodes], [False] * len(nodes))
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            plans, found = obj.plan_tree_search(np.array([[0., 0.]]), np.array([[10., 0.]]), 100, None,
                                                max_steps=100, max_search_num=10, early_stopping=True, plan_rank=3)
        self.assertEqual(len(branches), 3)  # three root children, each solved once
        self.assertEqual(len(set(branches)), 3)  # a solved branch is closed, so each expansion is a new branch
        self.assertEqual(found[0], 5)  # the 3rd shortest of 3, 5, 4
        self.assertIn('3/3 branches', log.getvalue())

    def test_rollout_solution_ends_search_when_achieved(self):
        calls, plans, _ = self._rollout_solves('achieved')
        self.assertEqual(sum(calls), 1)
        self.assertEqual(len(plans[0]), 4)

    def test_rollout_solution_is_only_a_fallback_when_solved(self):
        calls, plans, lengths = self._rollout_solves('solved')
        self.assertGreater(sum(calls), 1)  # the search went on after the rollout reached the goal
        self.assertEqual(len(plans[0]), lengths[0])  # and fell back to the rollout solution

    def test_jumpy_schedule_skips_levels_and_keeps_the_last(self):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        obj.episode_len, obj.frame_stack, obj.simulation_jumpy_interval = 100, 10, 10
        full = np.arange(60)[:, None].repeat(10, 1)
        obj._generate_scheduling_matrix = lambda tokens: full
        jumpy = obj._jumpy_noise_level()
        self.assertEqual(jumpy[:, 0].tolist(), [0, 10, 20, 30, 40, 50, 59])

    def test_children_guidance_sets_set_the_branching_factor(self):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        obj.parallel_search_num = 8
        obj.children_guidance_sets = [[0, 0.1], [0.1, 0.5], [0.5, 1], [1, 2], [2, 3], [3, 5]]
        seen = []

        def rollout(nodes, starts, goals, *args):
            seen.append([n.guidance_set for n in nodes])
            return ([0.] * len(nodes), ['NotReached'] * len(nodes),
                    [torch.tensor(start[None] + 0.1, dtype=torch.float32) for start in starts],
                    [False] * len(nodes))

        obj.expand_via_stitching = rollout
        obj.simulate_with_fast_replanning = lambda nodes, *args: (
            [0.] * len(nodes), ['Test'] * len(nodes), [[] for _ in nodes], [False] * len(nodes))
        with contextlib.redirect_stdout(io.StringIO()):
            obj.plan_tree_search(np.array([[0., 0.]]), np.array([[10., 0.]]), 100, None,
                                 max_steps=6, max_search_num=8)
        self.assertEqual(len(seen[0]), 6)  # all six root children in the first batch
        self.assertEqual(sorted(map(tuple, seen[0])), sorted(map(tuple, obj.children_guidance_sets)))

    def test_rejecting_a_node_while_its_sibling_is_pending(self):
        r = root()
        a, b = r.expand(), r.expand()  # both claimed in one parallel batch
        a.reject()                     # b has no value yet
        self.assertEqual(r.value, 0.0)
        b.set_value(0.3)
        r.backpropagate()
        self.assertEqual(r.value, 0.3)

    def test_parallel_batch_survives_a_rejected_expansion(self):
        obj = planner([[[0.1, 0], [0.2, 0]]])
        obj.parallel_search_num = 2
        calls = []

        def rollout(nodes, starts, goals, *args):
            calls.append(len(nodes))
            plans = [torch.empty(0, 2) if k == 0 else torch.tensor(start[None] + 0.1, dtype=torch.float32)
                     for k, start in enumerate(starts)]
            return [0.] * len(nodes), ['NoValidCandidate'] + ['NotReached'] * (len(nodes) - 1), plans, [False] * len(nodes)

        obj.expand_via_stitching = rollout
        obj.simulate_with_fast_replanning = lambda nodes, *args: (
            [0.] * len(nodes), ['Test'] * len(nodes), [[] for _ in nodes], [False] * len(nodes))
        with contextlib.redirect_stdout(io.StringIO()):
            plans, _ = obj.plan_tree_search(np.array([[0., 0.]]), np.array([[10., 0.]]), 100, None,
                                            max_steps=6, max_search_num=6)
        self.assertEqual(calls[0], 2)
        self.assertEqual(plans, [[]])


class StillStepTests(unittest.TestCase):
    def test_default_keeps_waiting_points(self):
        waiting = np.concatenate([np.zeros((10, 2)), np.stack([np.linspace(0, 2, 14), np.zeros(14)], 1)])
        obj = planner(waiting[None])
        plans, *_ = sample(obj)
        self.assertEqual(len(plans[0]), 24)

    def test_waiting_points_dropped_and_endpoints_kept(self):
        waiting = np.concatenate([np.zeros((10, 2)), np.stack([np.linspace(0, 2, 14), np.zeros(14)], 1)])
        obj = planner(waiting[None])
        obj.compress_still_steps = True
        plans, names, values, _ = sample(obj)
        self.assertEqual(len(plans[0]), 14)  # first point + 13 moving points
        self.assertTrue(torch.equal(plans[0][0], torch.zeros(2)))
        self.assertTrue(torch.allclose(plans[0][-1], torch.tensor([2., 0.])))
        self.assertEqual(names, ['NotReached'])

    def test_achieved_value_counts_kept_points(self):
        plan = np.concatenate([np.zeros((10, 2)), np.stack([np.linspace(0, 1, 14), np.zeros(14)], 1)])
        obj = planner(plan[None])
        obj.compress_still_steps = True
        history = [torch.zeros(24, 2) + torch.tensor([-5., 0.])]
        plans, names, values, solved = sample(obj, history, goal=(0.9, 0.))
        self.assertEqual(names, ['Achieved'])
        self.assertEqual(values[0], (100 - (24 + len(plans[0]) - 1)) / 100)
        self.assertLess(len(plans[0]), 14)

    def test_point_budget_makes_long_branches_terminal(self):
        r = HighLevelTreeNode('0', 0, None, [[0, 1]], [])
        child = r.expand()
        child.set_plan_history(torch.zeros(30, 2))
        child.set_value(0.1)
        r.backpropagate()
        self.assertIsNone(r.select_expandable(10, exp_weight=0.01, max_plan_length=30))
        self.assertIs(r.select_expandable(10, exp_weight=0.01, max_plan_length=31), child)


    def _rollout(self, max_steps):
        line = np.stack([np.linspace(0.1, 2.4, 24), np.zeros(24)], 1)
        obj = planner(line[None])
        obj.compress_still_steps = True
        obj.simulation_overrun_ratio = 0.5
        node = root().expand()
        node.set_local_name('NotReached')
        node.set_plan_history(torch.zeros(90, 2))
        return obj.simulate_with_fast_replanning(
            [node], np.zeros((1, 2)), np.array([[2.3, 0.]]), 100, None, max_steps)

    def test_rollout_reaching_goal_past_budget_is_overrun_not_solution(self):
        values, names, plans, solved = self._rollout(100)
        self.assertEqual(solved, [False])
        self.assertTrue(names[0].endswith('-Overrun'))

    def test_rollout_reaching_goal_within_budget_is_solution(self):
        values, names, plans, solved = self._rollout(200)
        self.assertEqual(solved, [True])
        self.assertGreater(values[0], 0)


class NoveltyScoreTests(unittest.TestCase):
    def setUp(self):
        self.obj = planner([[[0., 0.]]])


    def resampling_planner(self, banks):
        obj = planner(banks[0])
        calls = []

        def plan(*args, **kwargs):
            bank = torch.as_tensor(banks[min(len(calls), len(banks) - 1)], dtype=torch.float32)
            calls.append(1)
            return torch.cat([bank, bank[:, -1:]], dim=1).transpose(0, 1)[None]
        obj.plan = plan
        return obj, calls

    def test_no_resampling_by_default(self):
        obj, calls = self.resampling_planner([np.zeros((2, 60, 2)), np.zeros((2, 60, 2))])
        obj.best_of_n([[]], np.array([[0., 0.]]), np.array([[50., 50.]]),
                      torch.zeros(1, 2), torch.zeros(1, 2), 100, None, 100)
        self.assertEqual(len(calls), 1)


    def test_ties_go_to_the_stronger_goal_guidance(self):
        # default guidance set [0, 0.1, 0.5, 1, 2]: sample k uses guidance[k % 5]
        history = [torch.zeros(24, 2)]
        waiting = [np.zeros((24, 2)) + 0.01 * k for k in range(3)]  # all revisit, all score zero
        obj = planner(np.stack(waiting))
        plans, names, values, _ = obj.best_of_n([history], np.array([[0., 0.]]), np.array([[20., 20.]]),
                                                torch.zeros(1, 2), torch.zeros(1, 2), 100, None, 100)
        self.assertEqual(values, [0.0])
        self.assertTrue(torch.allclose(plans[0], torch.as_tensor(waiting[2], dtype=torch.float32)))

    def test_longer_new_ground_wins_beyond_the_old_cap(self):
        # 12 and 20 units of new ground both scored the old cap (0.01); now the longer one wins
        short, long_ = self.line((0, 0), (0, 12), 60), self.line((0, 0), (20, 0), 60)
        obj = planner(np.stack([short, long_]))
        plans, names, values, _ = obj.best_of_n([[]], np.array([[0., 0.]]), np.array([[50., 50.]]),
                                                torch.zeros(1, 2), torch.zeros(1, 2), 100, None, 100)
        self.assertEqual(names, ['NotReached'])
        self.assertTrue(torch.allclose(plans[0], torch.as_tensor(long_, dtype=torch.float32)))
        self.assertLess(values[0], 0.01)

    def line(self, a, b, n=41):
        return np.stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)], axis=1)

    def test_waiting_earns_nothing(self):
        plan = np.zeros((30, 2)) + np.random.default_rng(0).normal(0, 0.05, (30, 2))
        self.assertEqual(self.obj._novel_arclength(plan, np.zeros((1, 2))), 0.0)

    def test_backtracking_along_history_earns_nothing(self):
        history = self.line((0, 0), (8, 0))
        self.assertEqual(self.obj._novel_arclength(self.line((8, 0), (0, 0)), history), 0.0)

    def test_new_ground_earns_its_length_beyond_the_radius(self):
        novel = self.obj._novel_arclength(self.line((0, 0), (0, 8)), self.line((-8, 0), (0, 0)))
        self.assertAlmostEqual(novel, 7.0, delta=0.25)

    def test_returning_over_own_path_earns_nothing_extra(self):
        out_and_back = np.concatenate([self.line((0, 0), (0, 6)), self.line((0, 6), (0, 0))[1:]])
        once = self.obj._novel_arclength(self.line((0, 0), (0, 6)), np.zeros((1, 2)))
        # Only the turn-around within novelty_recent_arclength / 2 can count again.
        extra = self.obj._novel_arclength(out_and_back, np.zeros((1, 2))) - once
        self.assertGreaterEqual(extra, 0.0)
        self.assertLessEqual(extra, self.obj.novelty_recent_arclength / 2)

    def test_moving_close_candidate_beats_waiting_close_candidate(self):
        history = [torch.as_tensor(self.line((0, 0), (0, 5), 24), dtype=torch.float32)]
        waiting = np.tile([[0., 5.]], (24, 1))
        # explores new ground, then ends near the history (Close) but still earns its outbound novelty
        backtrack_then_turn = np.concatenate([self.line((0, 5), (4, 5), 16), self.line((4, 5), (0.5, 5), 9)[1:]])
        obj = planner(np.stack([waiting, backtrack_then_turn]))
        plans, names, values, _ = obj.best_of_n([history], np.array([[0., 5.]]), np.array([[20., 20.]]),
                                                torch.zeros(1, 2), torch.zeros(1, 2), 100, None, 100)
        self.assertEqual(names, ['Close'])
        self.assertTrue(torch.allclose(plans[0], torch.as_tensor(backtrack_then_turn, dtype=torch.float32)))
        self.assertGreater(values[0], 0)


class WaypointGraphTests(unittest.TestCase):
    """Nodes: 0 = start, 1..n = centers, n + 1 = goal; waypoint j is node j + 1."""

    def plans(self, n, found):
        table = [[np.zeros((0, 2)) for _ in range(n + 1)] for _ in range(n + 1)]
        for (i, j), length in found.items():
            table[i][j] = np.arange(2 * length, dtype=float).reshape(length, 2)
        return table

    def test_higher_tree_reaching_lower_waypoint_makes_an_edge(self):
        # tree of center 2 (node 2) reaches center 1 (waypoint 0 -> node 1)
        costs, paths = Planner._waypoint_graph(self.plans(2, {(2, 0): 5}))
        self.assertEqual(costs[2, 1], 5)
        self.assertEqual(costs[1, 2], 5)
        self.assertEqual(len(paths[1][2]), 5)

    def test_shorter_direction_is_kept(self):
        costs, paths = Planner._waypoint_graph(self.plans(2, {(1, 1): 9, (2, 0): 4}))
        self.assertEqual(costs[1, 2], 4)
        self.assertTrue(np.array_equal(paths[2][1], self.plans(2, {(2, 0): 4})[2][0]))

    def test_route_through_backward_connection(self):
        # start -> center 2, center 2 -> center 1 (found by the higher tree), center 1 -> goal
        costs, _ = Planner._waypoint_graph(self.plans(2, {(0, 1): 3, (2, 0): 4, (1, 2): 5}))
        costs[costs == np.inf] = 0
        distances, _ = dijkstra(costs, return_predecessors=True, indices=0)
        self.assertEqual(distances[3], 12)


class LowLevelTreeNodeTests(unittest.TestCase):
    def test_slots_fill_before_descending(self):
        root = LowLevelTreeNode(0, [0.5, 1, 0.5, 1])
        claimed = [root.select(3, 0.1, 1.0).claim_child() for _ in range(4)]
        self.assertEqual([c.guidance for c in claimed], [0.5, 1, 0.5, 1])
        self.assertEqual(root.virtual, 4)
        self.assertIsNone(root.select(3, 0.1, 1.0))  # every child is still in the batch

    def test_backpropagate_takes_the_max_and_clears_virtual_visits(self):
        root = LowLevelTreeNode(0, [0, 1])
        a, b = root.claim_child(), root.claim_child()
        a.score, b.score = 0.2, 0.7
        a.backpropagate(); b.backpropagate()
        self.assertEqual((root.value, root.visits, root.virtual), (0.7, 2, 0))
        self.assertIs(root.select(3, 0.0, 1.0).parent, root)  # descends into the better child
        self.assertIs(root.select(3, 0.0, 1.0), b)


if __name__ == '__main__':
    unittest.main()
