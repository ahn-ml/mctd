# Tests

CPU unit tests; they run in about a second and need neither the model weights nor the OGBench environment:

- `test_cube_search.py`: the cube tree search with a stub planner (tree nodes and the parallel selection, the value estimation, the children and temporary positions, the cube-move planning with the plan cache, best-of-N replanning) and the replan baseline.
- `test_release_cli.py`: the release CLI (`scripts/cmctd.py`) and the per-size presets.
- `test_target_cube_sampler.py`: the target-cube sampler of the controller training.

```bash
python3 -m unittest discover -s tests
```
