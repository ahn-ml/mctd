"""Collect per-pair Preplan edges (edge_{i}_{j}.json from high_level_search=preplan-edges) into graph pickles.

Writes <out>/<dataset>_graph_costs<n>.pkl and _graph_paths<n>.pkl in the format Preplan loads:
costs[i, j] = plan length (inf when there is no edge), paths[i][j] = the plan, paths[j][i] = it reversed.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def build(records, num_nodes, max_steps=None):
    costs = np.zeros((num_nodes, num_nodes)) + np.inf
    paths = [[[None] for _ in range(num_nodes)] for _ in range(num_nodes)]
    for r in records:
        if not r['found'] or (max_steps is not None and r['length'] > max_steps):
            continue
        i, j, plan = r['i'], r['j'], np.asarray(r['plan'], dtype=np.float32)
        costs[i, j] = costs[j, i] = len(plan)
        paths[i][j], paths[j][i] = plan, plan[::-1].copy()
    return costs, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edges', type=Path, required=True, help='directory (searched recursively) of edge_{i}_{j}.json')
    parser.add_argument('--nodes', type=int, required=True, help='number of waypoints (10 / 30 / 70)')
    parser.add_argument('--out', type=Path, required=True, help='graph directory (Preplan graph_dir)')
    parser.add_argument('--max-steps', type=int, default=None, help='drop edges longer than this')
    args = parser.parse_args(argv)
    records = [json.loads(p.read_text()) for p in sorted(args.edges.rglob('edge_*_*.json'))]
    datasets = {r['dataset'] for r in records}
    if len(datasets) != 1:
        raise ValueError(f'edges from {len(datasets)} datasets: {sorted(datasets)}')
    dataset = datasets.pop()
    expected = {(i, j) for i in range(args.nodes) for j in range(i + 1, args.nodes)}
    found = {(r['i'], r['j']) for r in records}
    if found - expected:
        raise ValueError(f'edges outside {args.nodes} nodes: {sorted(found - expected)[:5]}')
    costs, paths = build(records, args.nodes, args.max_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    for kind, value in (('costs', costs), ('paths', paths)):
        target = args.out / f'{dataset}_graph_{kind}{args.nodes}.pkl'
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(pickle.dumps(value))
    edges = int(np.isfinite(costs).sum() // 2)
    print(f'{dataset}: {len(records)}/{len(expected)} pairs searched, {edges} edges '
          f'(missing pairs {len(expected - found)}), mean length {costs[np.isfinite(costs)].mean() if edges else 0:.1f} -> {args.out}')


if __name__ == '__main__':
    main()
