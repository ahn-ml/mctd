"""Memory-bounded, deterministic k-shortest loopless graph routes."""

import heapq
import math


def shortest_simple_routes(costs, source, target, limit, max_cost=None):
    """Yield Yen routes sorted by (cost, node tuple), with a strict cost cap.

    Positive finite entries are edges; zero denotes no edge. Dijkstra retains
    only the best label per node instead of enumerating all partial routes.
    """
    if limit <= 0:
        return
    adjacency = [tuple((v, w) for v, w in enumerate(row)
                       if w > 0 and math.isfinite(w)) for row in costs]

    def shortest(start, blocked_nodes, blocked_edges, initial_cost, root):
        route = root + (start,)
        best = {start: (initial_cost, route)}
        queue = [(initial_cost, route)]
        while queue:
            cost, path = heapq.heappop(queue)
            node = path[-1]
            if best[node] != (cost, path):
                continue
            if max_cost is not None and cost >= max_cost:
                continue
            if node == target:
                return cost, path
            for nxt, weight in adjacency[node]:
                if nxt in blocked_nodes or (node, nxt) in blocked_edges:
                    continue
                candidate = (cost + weight, path + (nxt,))
                if max_cost is not None and candidate[0] >= max_cost:
                    continue
                if nxt not in best or candidate < best[nxt]:
                    best[nxt] = candidate
                    heapq.heappush(queue, candidate)
        return None

    first = shortest(source, set(), set(), 0, ())
    if first is None:
        return
    accepted = [first[1]]
    seen = {first[1]}
    candidates = []
    yield first[1]
    while len(accepted) < limit:
        previous = accepted[-1]
        root_cost = 0
        for index in range(len(previous) - 1):
            root = previous[:index + 1]
            blocked_edges = {(path[index], path[index + 1])
                             for path in accepted
                             if len(path) > index + 1 and path[:index + 1] == root}
            candidate = shortest(root[-1], set(root[:-1]), blocked_edges,
                                 root_cost, root[:-1])
            if candidate is not None and candidate[1] not in seen:
                seen.add(candidate[1])
                heapq.heappush(candidates, candidate)
            root_cost += costs[previous[index]][previous[index + 1]]
        if not candidates:
            return
        _, route = heapq.heappop(candidates)
        accepted.append(route)
        yield route
