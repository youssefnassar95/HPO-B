# Benchmark slice for the sweep runner

This folder holds the part of HPO-B that the sweep runner needs, so that a sweep
can run without downloading the full benchmark.

- `hpob-data/meta-test-dataset.json` and `hpob-data/bo-initializations.json` are
  search space `5971` taken unchanged from the meta-test split of the HPO-B
  meta-dataset.
- `surrogates/` has one XGBoost surrogate per dataset of that search space and
  the `summary-stats.json` that `HPOBHandler.evaluate_continuous` uses to
  normalize responses. They were trained on the slice above with
  `continuous/build_sweep_slice.py`; they are not the surrogates published with
  HPO-B, so incumbent values differ from the ones in `results/`.

The HPO-B meta-dataset is distributed under a CC-BY license:

> Sebastian Pineda Arango, Hadi S. Jomaa, Martin Wistuba, Josif Grabocka.
> "HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML."
> NeurIPS 2021 Datasets and Benchmarks Track.
