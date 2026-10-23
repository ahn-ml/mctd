"""CPU tests of the cube object-level search: tree nodes, value rules, children, plan cache and retries."""
import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CubeTreeNode = load_module('tested_cube_tree_node', ROOT / 'algorithms/diffusion_forcing/cube_tree_node.py').CubeTreeNode
PLANNER_FILE = ROOT / 'algorithms/diffusion_forcing/df_planning.py'
source = ast.parse(PLANNER_FILE.read_text())
planner_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DiffusionForcingPlanning')
methods = {'_count', '_cube_temporary_positions', '_cube_children_specs', '_cube_value', '_cube_move_value', '_plan_cube_moves', 'replan', 'cube_tree_search',
           'cube_replan'}
planner_class.bases = []
planner_class.body = [n for n in planner_class.body if isinstance(n, ast.FunctionDef) and n.name in methods]
constants = [n for n in source.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in ('CUBE_PLAN_MAX_ATTEMPTS', 'UNPLANNABLE') for t in n.targets)]
namespace = dict(np=np, torch=torch, CubeTreeNode=CubeTreeNode)
exec(compile(ast.Module(body=constants + [planner_class], type_ignores=[]), str(PLANNER_FILE), 'exec'), namespace)
Planner, UNPLANNABLE = namespace['DiffusionForcingPlanning'], namespace['UNPLANNABLE']
CUBE_PLAN_MAX_ATTEMPTS = namespace['CUBE_PLAN_MAX_ATTEMPTS']


def planner(temporary_positions=0):
    obj = Planner()
    obj.cube_goal_threshold = 0.15
    obj.cube_num_temporary_positions = temporary_positions
    obj.cube_plan_cache = True
    obj.cube_print_out = False
    obj.parallel_search_num = 1
    obj.virtual_visit_weight = 1.0
    obj.cube_plan_max_steps = 200
    obj.mctd_max_search_num = 20
    obj.episode_len = 100
    obj.calls = []
    obj.reach_dims = []

    def straight_plans(starts, goals, conditions, max_steps, reach_dims):
        """Every move is planned with a straight plan of 5 points."""
        obj.calls.append(len(starts))
        obj.reach_dims.append(reach_dims)
        return [torch.from_numpy(np.linspace(starts[k], goals[k], 5, dtype=np.float32)) for k in range(len(starts))]
    obj.replan = straight_plans
    return obj


def state(*cubes):
    return np.array([np.concatenate(cubes)], dtype=np.float32)


TABLE = 0.2  # cube height on the table
ON_ONE = 0.7  # resting on one cube (0.5 above)


class CubeTreeNodeTests(unittest.TestCase):
    @staticmethod
    def root_with_two_children():
        root = CubeTreeNode('0', 0, None)
        root.set_value(0)
        root.set_children_node_specs([[0, None], [1, None]])
        a, b = root.expand(), root.expand()
        for child in (a, b):
            child.set_children_node_specs([[2, None]])  # each child has a move left
        return root, a, b

    def test_select_skips_pruned_children_and_returns_none_when_all_are(self):
        root, a, b = self.root_with_two_children()
        a.set_value(-np.inf); b.set_value(0.5)
        self.assertIs(root.select_expandable(), b)
        b.set_value(-np.inf)
        self.assertIsNone(root.select_expandable())

    def test_select_skips_children_waiting_for_their_evaluation(self):
        root, a, b = self.root_with_two_children()
        a.set_value(0.5)  # b is still in the batch
        self.assertIs(root.select_expandable(), a)
        a.set_value(None)
        self.assertIsNone(root.select_expandable())

    def test_select_falls_back_to_the_next_branch_with_an_expandable_node(self):
        root, a, b = self.root_with_two_children()
        a.set_value(0.9); b.set_value(0.1)
        a.set_children_node_specs([[2, None]])
        a.expand()  # a's only child waits for its evaluation
        self.assertIs(root.select_expandable(), b)

    def test_virtual_visits_steer_the_selection_to_another_branch(self):
        root, a, b = self.root_with_two_children()
        a.set_value(0.5); b.set_value(0.5)
        self.assertIs(root.select_expandable(), a)
        a.add_virtual_visit()
        self.assertIs(root.select_expandable(), b)
        a.clear_virtual_visits()
        self.assertIs(root.select_expandable(), a)

    def test_backpropagate_takes_the_best_child(self):
        root = CubeTreeNode('0', 0, None)
        root.set_value(0)
        root.set_children_node_specs([[0, None], [1, None]])
        a = root.expand(); a.set_value(0.5)
        root.backpropagate()
        self.assertEqual(root.get_value(), 0.5)  # not fully expanded: max of its own value and the children
        b = root.expand(); b.set_value(-np.inf)
        root.backpropagate()
        self.assertEqual((root.get_value(), root.get_visit_count()), (0.5, 2))


class CubeValueTests(unittest.TestCase):
    def test_value_is_the_share_of_solved_cubes(self):
        obj = planner()
        goal = state([0, 0, TABLE], [1, 1, TABLE])
        self.assertEqual(obj._cube_value(state([0, 0, TABLE], [-1, 1, TABLE]), goal, 2), 0.5)
        self.assertEqual(obj._cube_value(goal, goal, 2), 1.0)

    def test_stacked_cube_is_valid_and_floating_cube_is_pruned(self):
        obj = planner()
        goal = state([0, 0, TABLE], [0, 0, ON_ONE])
        self.assertEqual(obj._cube_value(state([0, 0, TABLE], [0, 0, ON_ONE]), goal, 2), 1.0)
        self.assertEqual(obj._cube_value(state([0, 0, TABLE], [1, 1, ON_ONE]), goal, 2), -np.inf)

    def test_a_four_cube_tower_is_valid_at_any_top_height(self):
        obj = planner()
        for top in (1.397, 1.42):  # the goal of the quadruple stacking task varies around 1.4
            tower = state([0, 0, TABLE], [0, 0, 0.6], [0, 0, 1.0], [0.01, 0, top])
            self.assertEqual(obj._cube_value(tower, tower, 4), 1.0)
        self.assertEqual(obj._cube_value(state([0, 0, TABLE], [0, 0, 0.6], [1, 1, 1.0]), state([0, 0, TABLE], [0, 0, 0.6], [0, 0, 1.0]), 3), -np.inf)

    def test_two_cubes_at_one_position_are_pruned(self):
        obj = planner()
        goal = state([0, 0, TABLE], [1, 1, TABLE])
        self.assertEqual(obj._cube_value(state([0, 0, TABLE], [0.05, 0, TABLE]), goal, 2), -np.inf)


class CubeMoveValueTests(unittest.TestCase):
    def test_move_value_is_the_value_of_the_reached_state(self):
        obj = planner()
        start, goal = state([0, 0, TABLE], [1, 1, TABLE]), state([-1, -1, TABLE], [1, 1, TABLE])
        plan = np.linspace(start[0], goal[0], 5)
        self.assertEqual(obj._cube_move_value(start, plan, 0, goal, 2), 1.0)

    def test_moving_a_cube_with_another_on_top_is_pruned(self):
        obj = planner()
        start = state([0, 0, TABLE], [0, 0, ON_ONE])  # cube 1 rests on cube 0
        goal = state([1, 1, TABLE], [0, 0, TABLE])
        plan = np.linspace(start[0], state([1, 1, TABLE], [0, 0, ON_ONE])[0], 5)
        self.assertEqual(obj._cube_move_value(start, plan, 0, goal, 2), -np.inf)


class CubeChildrenTests(unittest.TestCase):
    def test_children_move_the_unsolved_cubes_to_the_goal_then_the_temporary_positions(self):
        obj = planner(temporary_positions=1)
        goal = state([0, 0, TABLE], [1, 1, TABLE])
        temporary_positions = [[np.array([0.5, -0.5, TABLE])], [np.array([-0.5, 0.5, TABLE])]]
        specs = obj._cube_children_specs(state([0, 0, TABLE], [-1, -1, TABLE]), goal, 2, temporary_positions)
        self.assertEqual([s[0] for s in specs], [1, 1])  # cube 0 is already solved
        np.testing.assert_allclose(specs[0][1], goal[:, 3:6])
        np.testing.assert_allclose(specs[1][1], temporary_positions[1][0])

    def test_temporary_positions_keep_away_from_goals_and_each_other(self):
        np.random.seed(0)
        obj = planner(temporary_positions=2)
        goal = state([0, 0, TABLE], [0.5, 0.5, TABLE])
        temporary_positions = obj._cube_temporary_positions(goal, 2)
        points = [p for per_cube in temporary_positions for p in per_cube]
        self.assertEqual(len(points), 4)
        for k, p in enumerate(points):
            self.assertTrue(all(np.linalg.norm(goal[:, 3*i:3*i+3] - p) >= 0.4 for i in range(2)))
            self.assertTrue(all(np.linalg.norm(p - q) >= 0.4 for q in points[:k]))


class CubeMoveTests(unittest.TestCase):
    def test_move_keeps_the_shortest_plan_and_fixes_the_other_cubes(self):
        obj = planner()
        start = state([0, 0, TABLE], [1, 1, TABLE])
        plan = obj._plan_cube_moves([(start, 1, np.array([[-1, 1, TABLE]], dtype=np.float32))], None, {})[0]
        self.assertEqual(len(plan), 5 + 1)  # the plan and the goal
        np.testing.assert_allclose(plan[:, :3], np.repeat(start[:, :3], len(plan), axis=0))
        np.testing.assert_allclose(plan[-1, 3:], [-1, 1, TABLE])

    def test_identical_move_comes_from_the_cache(self):
        obj = planner()
        cache = {}
        start = state([0, 0, TABLE], [1, 1, TABLE])
        target = np.array([[-1, 1, TABLE]], dtype=np.float32)
        first = obj._plan_cube_moves([(start, 1, target)], None, cache)[0]
        moved_other = state([0.5, 0, TABLE], [1, 1, TABLE])  # same move of cube 1, cube 0 elsewhere
        second = obj._plan_cube_moves([(moved_other, 1, target)], None, cache)[0]
        self.assertEqual(obj.calls, [1])
        np.testing.assert_allclose(second[:, 3:], first[:, 3:])
        np.testing.assert_allclose(second[:, :3], np.repeat(moved_other[:, :3], len(second), axis=0))

    def test_reach_check_looks_at_the_moved_cube(self):
        obj = planner()
        obj._plan_cube_moves([(state([0, 0, TABLE], [1, 1, TABLE]), 1, np.array([[-1, 1, TABLE]], dtype=np.float32))], None, {})
        self.assertEqual(obj.reach_dims, [[slice(3, 6)]])

    def test_moves_are_planned_in_one_batch_and_repeats_once(self):
        obj = planner()
        start = state([0, 0, TABLE], [1, 1, TABLE])
        a, b = np.array([[-1, 1, TABLE]], dtype=np.float32), np.array([[1, -1, TABLE]], dtype=np.float32)
        plans = obj._plan_cube_moves([(start, 1, a), (start, 0, b), (start, 1, a)], None, {})
        self.assertEqual(obj.calls, [2])  # two distinct moves, one batch
        np.testing.assert_allclose(plans[0], plans[2])
        np.testing.assert_allclose(plans[1][-1, :3], b[0])

    def test_no_move_needs_no_plan(self):
        obj = planner()
        start = state([0, 0, TABLE])
        self.assertIsNone(obj._plan_cube_moves([(start, 0, start[:, :3])], None, {})[0])

    def test_unplannable_move_gives_up_after_the_attempts(self):
        obj = planner()
        def no_plan(starts, *a, **k):
            obj.calls.append(1)
            return [[] for _ in starts]
        obj.replan = no_plan
        start = state([0, 0, TABLE])
        self.assertIs(obj._plan_cube_moves([(start, 0, np.array([[1, 1, TABLE]], dtype=np.float32))], None, {})[0], UNPLANNABLE)
        self.assertEqual(len(obj.calls), CUBE_PLAN_MAX_ATTEMPTS)


class ReplanTests(unittest.TestCase):
    def test_replanning_chains_segments_until_one_reaches_the_goal(self):
        obj = Planner()
        obj.episode_len = 100
        calls = []

        def best_of_n(histories, starts, goals, *args):
            """Segments of 4 points halfway to the goal; the second segment of an item reaches it."""
            calls.append(len(starts))
            plans = [torch.from_numpy(np.linspace(starts[i], goals[i] if histories[i] else (starts[i] + goals[i]) / 2, 4,
                                                  dtype=np.float32)) for i in range(len(starts))]
            return plans, [0.0] * len(starts), [bool(h) for h in histories]
        obj.best_of_n = best_of_n
        plan = obj.replan(state([0, 0, TABLE]), state([1, 1, TABLE]), None, 200, [slice(0, 3)])[0]
        self.assertEqual(calls, [1, 1])
        self.assertEqual(len(plan), 8)
        np.testing.assert_allclose(plan[-1], [1, 1, TABLE])

    def test_replanning_gives_up_at_the_plan_cap(self):
        obj = Planner()
        obj.episode_len = 100
        obj.best_of_n = lambda histories, starts, goals, *a: ([torch.zeros(4, 3)], [0.0], [False])
        self.assertEqual(obj.replan(state([0, 0, TABLE]), state([1, 1, TABLE]), None, 10, [slice(0, 3)]), [[]])


class CubeTreeSearchTests(unittest.TestCase):
    def test_search_moves_each_cube_to_its_goal(self):
        np.random.seed(0)
        obj = planner()
        start = state([0, 0, TABLE], [1, 1, TABLE])
        goal = state([-1, -1, TABLE], [-1, 1, TABLE])
        plans, cubes = obj.cube_tree_search(start, goal)
        self.assertEqual(sorted(cubes), [0, 1])
        np.testing.assert_allclose(plans[-1][-1], goal[0])

    def test_parallel_search_expands_several_nodes_per_iteration(self):
        np.random.seed(0)
        obj = planner()
        obj.parallel_search_num = 4
        start = state([0, 0, TABLE], [1, 1, TABLE], [-1, 1, TABLE])
        goal = state([-1, -1, TABLE], [1, -1, TABLE], [0, 1, TABLE])
        plans, cubes = obj.cube_tree_search(start, goal)
        self.assertEqual(sorted(cubes), [0, 1, 2])
        self.assertEqual(obj.calls[0], 3)  # the three first moves (one per cube) in one batch
        stats = obj.search_stats
        self.assertEqual(stats["planned_moves"], 3)  # each cube's move once; the later moves come from the cache
        self.assertEqual(stats["expanded_nodes"], stats["planned_moves"] + stats["cache_hits"])
        np.testing.assert_allclose(plans[-1][-1], goal[0])

    def test_search_puts_a_blocking_cube_aside(self):
        np.random.seed(0)
        obj = planner(temporary_positions=1)
        start = state([0, 0, TABLE], [1, 1, TABLE])
        goal = state([1, 1, TABLE], [0, 0, TABLE])  # swap: each goal is taken by the other cube
        plans, cubes = obj.cube_tree_search(start, goal)
        self.assertEqual(len(cubes), 3)  # one cube to its temporary position, then both to their goals
        np.testing.assert_allclose(plans[-1][-1], goal[0])

    def test_search_without_any_plan_returns_nothing(self):
        np.random.seed(0)
        obj = planner()
        obj.replan = lambda starts, *a, **k: [[] for _ in starts]
        self.assertEqual(obj.cube_tree_search(state([0, 0, TABLE]), state([1, 1, TABLE])), ([], []))


class CubeReplanBaselineTests(unittest.TestCase):
    def test_replan_commits_to_the_shortest_plan_each_time(self):
        obj = planner()
        obj.val_max_steps = 1000

        def plans_by_cube(starts, goals, conditions, max_steps, reach_dims):
            """Cube 1's moves take 3 points, cube 0's 6."""
            return [torch.from_numpy(np.linspace(starts[k], goals[k], 3 if reach.start == 3 else 6, dtype=np.float32))
                    for k, reach in enumerate(reach_dims)]
        obj.replan = plans_by_cube
        start = state([0, 0, TABLE], [1, 1, TABLE])
        goal = state([-1, -1, TABLE], [-1, 1, TABLE])
        plans, cubes = obj.cube_replan(start, goal)
        self.assertEqual(cubes, [1, 0])
        np.testing.assert_allclose(plans[1][0], plans[0][-1])  # the second move starts where the first ends
        np.testing.assert_allclose(plans[-1][-1], goal[0])

    def test_replan_does_not_estimate_values(self):
        obj = planner()
        obj.val_max_steps = 1000
        start = state([0, 0, TABLE], [1, 1, TABLE])
        goal = state([1, 1, TABLE], [0, 0, TABLE])  # swap: each goal is taken by the other cube
        plans, cubes = obj.cube_replan(start, goal)
        self.assertEqual(cubes, [0, 1])  # cube 0 into cube 1's place: the search would prune it
        self.assertEqual(obj._cube_value(plans[0][-1][None], goal, 2), -np.inf)

    def test_replan_fails_without_a_plannable_move_or_past_the_step_limit(self):
        obj = planner()
        obj.val_max_steps = 1000
        start, goal = state([0, 0, TABLE], [1, 1, TABLE]), state([-1, -1, TABLE], [-1, 1, TABLE])
        obj.replan = lambda starts, *a, **k: [[] for _ in starts]
        self.assertEqual(obj.cube_replan(start, goal), ([], []))
        obj = planner()
        obj.val_max_steps = 8  # two moves of 5 + 1 points
        self.assertEqual(obj.cube_replan(start, goal), ([], []))


if __name__ == '__main__':
    unittest.main()
