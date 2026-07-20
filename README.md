# taskrouter

**Training-free expert routing for model merging, because averaging specialists averages away their edge.**

Merge three specialists into one average model and you haven't built a generalist — you've built three people at once talking over each other, none of them finishing a sentence. A model fine-tuned for task A and a model fine-tuned for task B, weight-averaged together, doesn't do A well and B well. It does both mediocrely, which on a task-specific input is strictly worse than just picking the right expert and getting out of the way. taskrouter doesn't merge weights at all — it looks at the input, figures out which expert it resembles, and routes to that one.

It's a compact, inspectable implementation inspired by [the 2026 TR-Merging work on similarity-based, training-free model routing](https://openreview.net/forum?id=4S0yZPVxex), rebuilt small enough to read in one sitting and run without a GPU, a checkpoint, or an API key.

## The result

```bash
python taskrouter.py
```
```json
{
  "static_merge_mae": 1.399,
  "routed_mae": 0.276,
  "error_reduction_pct": 80.2
}
```

Average the outputs of three specialist functions together on noisy task-specific inputs and you get `static_merge_mae` — a compromise nobody asked for. Compare the input against each expert's known task prototype and route to the nearest one instead, and error drops to `routed_mae` — an 80.2% reduction, because when the right expert exists, using it beats blending it with two wrong ones, even after the labels themselves get noisy.

## How it works

Three toy "experts" are three different functions (linear, quadratic, sinusoidal), each anchored to its own prototype input value. Every task sample is drawn near one expert's prototype with jitter, and the label itself carries observation noise, so a perfect router still can't hit zero error — it has to actually beat the noise floor, not just recover an exact answer. The static-merge baseline averages all three experts' outputs regardless of which task the input actually came from — this is what weight-averaging model-merging does in spirit. The router instead measures distance from the input to each prototype and picks the nearest expert's output outright, with no training step and no gradient anywhere in the routing decision — just similarity-based selection, made explicit.

## Run it

```bash
python taskrouter.py
python -m unittest discover -s tests -v
```

## What is tested

The test compares routed selection against the static-merge baseline and requires `error_reduction_pct >= 45`. The data generator is seeded, so the number in this README, in CI, and in the portfolio case study are the same number, not three different ones that happen to rhyme.

## Scope

This is an educational research reproduction using three toy expert functions rather than real fine-tuned models. It is not a clinical, diagnostic, production model-serving, or safety-critical system, and it makes no claim about real LLM or vision-model merging results. The point is to make one mechanism — training-free similarity routing beats static weight averaging — measurable without hiding it behind a checkpoint.

## Research basis

- [The 2026 TR-Merging work on similarity-based, training-free model routing](https://openreview.net/forum?id=4S0yZPVxex)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT
