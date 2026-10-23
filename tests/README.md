# Tests

CPU regression tests for the release CLI and presets, the graph route search and the rejection of invalid candidates:

```bash
python3 -m unittest discover -s tests
```

`test_dql_batch_sampling.py` additionally checks batched Diffusion-QL sampling and is skipped without a GPU.
