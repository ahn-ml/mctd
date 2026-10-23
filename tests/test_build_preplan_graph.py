"""The Preplan edge collector writes the graph pickles Preplan loads."""
import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_preplan_graph  # noqa: E402


class BuildPreplanGraphTests(unittest.TestCase):
    def test_collects_edges_both_ways(self):
        with tempfile.TemporaryDirectory() as folder:
            edges = Path(folder) / 'shard0'
            edges.mkdir()
            for i, j, found, length in ((0, 1, True, 3), (0, 2, False, 0), (1, 2, True, 3)):
                plan = [[0, 0], [1, 1], [2, 2]] if found else []
                (edges / f'edge_{i}_{j}.json').write_text(json.dumps(dict(
                    i=i, j=j, found=found, length=length, plan=plan, dataset='pointmaze-medium-stitch-v0')))
            build_preplan_graph.main(['--edges', folder, '--nodes', '3', '--out', str(Path(folder) / 'graph')])
            costs = pickle.loads((Path(folder) / 'graph/pointmaze-medium-stitch-v0_graph_costs3.pkl').read_bytes())
            paths = pickle.loads((Path(folder) / 'graph/pointmaze-medium-stitch-v0_graph_paths3.pkl').read_bytes())
        self.assertEqual(costs[0, 1], 3)
        self.assertEqual(costs[1, 0], 3)
        self.assertTrue(np.isinf(costs[0, 2]))
        self.assertEqual(paths[1][0].tolist(), [[2, 2], [1, 1], [0, 0]])
        self.assertEqual(paths[0][2], [None])

    def test_length_cap(self):
        costs, _ = build_preplan_graph.build([dict(i=0, j=1, found=True, length=3, plan=[[0, 0]] * 3)], 2, max_steps=2)
        self.assertTrue(np.isinf(costs).all())


if __name__ == '__main__':
    unittest.main()
