# Scripts

- `cmctd.py`: planner and controller training, and composer evaluation with the release presets (see the root README).
- `build_preplan_graph.py`: collects the per-pair Preplan edges (`high_level_search=preplan-edges`) into the graph pickles.
- `docker_run.sh`: runs a command inside the Docker image with the repository mounted at `/workspace`.
- `check_environment.py`: compares package versions and the custom OGBench build with the tested runtime.
