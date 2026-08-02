"""Stress-test taskrouter's claim under genuine routing ambiguity.

This is not a bug fix -- taskrouter.py's mechanism and published numbers
are correct and reproduce exactly. This checks whether the claim
generalizes past the one easy scenario the benchmark ships with, where
routing happens to be correct 100% of the time by construction."""
import json
import random
import statistics as st

from adversarial import HOLDOUT_SEEDS, STRESS_JITTER, TUNING_SEEDS
from taskrouter import EXPERTS, PROTOS


def run_at_jitter(seed, jitter):
    rng = random.Random(seed)
    static = route = correct = 0
    for _ in range(240):
        task = rng.randrange(3)
        x = PROTOS[task] + rng.uniform(-jitter, jitter)
        y = EXPERTS[task](x) + rng.gauss(0, .35)
        merged = sum(f(x) for f in EXPERTS) / len(EXPERTS)
        chosen = min(range(3), key=lambda k: abs(x - PROTOS[k]))
        correct += chosen == task
        static += abs(merged - y)
        route += abs(EXPERTS[chosen](x) - y)
    return {
        "static_merge_mae": round(static / 240, 3),
        "routed_mae": round(route / 240, 3),
        "error_reduction_pct": round(100 * (1 - route / static), 1),
        "routing_accuracy": round(correct / 240, 3),
    }


def summarize(seeds, jitter):
    results = [run_at_jitter(seed, jitter) for seed in seeds]
    reductions = [r["error_reduction_pct"] for r in results]
    accuracies = [r["routing_accuracy"] for r in results]
    return {
        "n": len(seeds),
        "jitter": jitter,
        "mean_reduction_pct": round(st.mean(reductions), 1),
        "min_reduction_pct": min(reductions),
        "mean_routing_accuracy": round(st.mean(accuracies), 3),
    }


def main():
    print("taskrouter eval_v2: routing under genuine ambiguity (not a bug fix, a robustness check)")
    print(f"published scenario (jitter=.45): {run_at_jitter(23, .45)}")
    for label, seeds in (("tuning", TUNING_SEEDS), ("holdout", HOLDOUT_SEEDS)):
        print(f"\n{label} at STRESS_JITTER={STRESS_JITTER} ({len(seeds)} seeds):")
        print(json.dumps(summarize(seeds, STRESS_JITTER), indent=2))


if __name__ == "__main__":
    main()
