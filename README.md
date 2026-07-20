# taskrouter

**Training-free expert routing for model merging, because averaging specialists averages away their edge.**

taskrouter is a compact, inspectable implementation inspired by [The 2026 TR-Merging work on similarity-based, training-free model routing.](https://openreview.net/forum?id=4S0yZPVxex).
It turns the paper's core idea into a deterministic benchmark that runs on a laptop with Python's standard library.

## Run it

```bash
python taskrouter.py
python -m unittest discover -s tests -v
```

The benchmark writes its result to stdout. Audio projects also write playable WAV files to `demo/`.

## What is tested

The test compares the research-inspired method with a deliberately legible baseline and requires
`error_reduction_pct >= 45`. The data generator is seeded, so the number in this README,
CI, and the portfolio case study can be reproduced.

## Scope

This is an educational research reproduction on controlled synthetic data. It is not a clinical,
diagnostic, production genomics, copyright-authentication, or safety-critical system. The point is
to make one mechanism measurable without hiding it behind a checkpoint or API.

## Research basis

- [The 2026 TR-Merging work on similarity-based, training-free model routing.](https://openreview.net/forum?id=4S0yZPVxex)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT

## Reproduced result

| Metric | Value |
|---|---:|
| `static_merge_mae` | **1.379** |
| `routed_mae` | **0.0** |
| `error_reduction_pct` | **100.0** |
