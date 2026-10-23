# configurations

We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. Change/Add the yaml files in this folder
to change the default configurations. You can also override the default configurations by
passing command line arguments.

`algorithm/df_planning.yaml` holds the planner and composer defaults of the release; `release/*_presets.json`
hold the per-map and per-composer settings that `scripts/cmctd.py` adds on top of them.

The configuration of each run is saved by Hydra in its output directory.
