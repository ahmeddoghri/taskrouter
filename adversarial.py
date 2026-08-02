"""Adversarial seeds for the routing-ambiguity stress test.

taskrouter.py's published scenario uses jitter=.45 with prototypes spaced
2.0 apart, so every task sample stays well inside its own prototype's
region -- nearest-prototype routing is correct 100% of the time, every
seed (verified directly: routing_accuracy == 1.0 for every seed tested).
That's not a tautology (routed_mae still depends on genuine label noise),
but it does mean the published result never has to face real ambiguity
about which expert to route to.

STRESS_JITTER pushes samples out far enough that prototype regions overlap
(jitter=1.5 vs. half-spacing=1.0), producing genuine routing errors
(~77% accuracy, not 100%) -- a harder, still-plausible scenario for
similarity-based routing.

TUNING_SEEDS: used to characterize the stress-test result.
HOLDOUT_SEEDS: disjoint, evaluated exactly once after characterization.
"""

STRESS_JITTER = 1.5

TUNING_SEEDS = list(range(1, 41))
HOLDOUT_SEEDS = list(range(1000, 1030))
