"""Compare Yen against exhaustive enumeration on small graphs."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.graph_routes import shortest_simple_routes


def exhaustive(costs, source, target, limit, cap):
    complete = []

    def visit(route, cost):
        if cap is not None and cost >= cap:
            return
        if route[-1] == target:
            complete.append((cost, route))
            return
        for nxt, weight in enumerate(costs[route[-1]]):
            if weight > 0 and nxt not in route:
                visit(route + (nxt,), cost + weight)

    visit((source,), 0)
    return [route for _, route in sorted(complete)[:max(0, limit)]]


class RouteTests(unittest.TestCase):
    def test_exhaustive_equivalence(self):
        rng = random.Random(817)
        for case in range(300):
            n = rng.randrange(2, 8)
            costs = [[rng.randrange(1, 5) if i != j and rng.random() < .55 else 0
                      for j in range(n)] for i in range(n)]
            if case % 2:
                for i in range(n):
                    for j in range(i):
                        costs[i][j] = costs[j][i]
            for cap in (None, 0, 1, 4, 5, 6, 12):
                for limit in (0, 1, 5, 20):
                    for target in (0, n - 1):
                        with self.subTest(case=case, cap=cap, limit=limit, target=target):
                            self.assertEqual(
                                list(shortest_simple_routes(costs, 0, target, limit, cap)),
                                exhaustive(costs, 0, target, limit, cap))

    def test_dense_unreachable(self):
        costs = [[int(i != j and i < 71 and j < 71) for j in range(72)]
                 for i in range(72)]
        self.assertEqual(list(shortest_simple_routes(costs, 0, 71, 5)), [])

    def test_dense_ties(self):
        costs = [[int(i != j) for j in range(72)] for i in range(72)]
        self.assertEqual(list(shortest_simple_routes(costs, 0, 71, 5)),
                         [(0, 71), (0, 1, 71), (0, 2, 71), (0, 3, 71), (0, 4, 71)])


if __name__ == '__main__':
    unittest.main()
