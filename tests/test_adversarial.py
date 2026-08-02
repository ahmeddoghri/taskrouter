import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import taskrouter
from adversarial import HOLDOUT_SEEDS, STRESS_JITTER, TUNING_SEEDS
from eval_v2 import run_at_jitter, summarize


class AdversarialTest(unittest.TestCase):
    def test_holdout_disjoint_from_tuning(self):
        self.assertTrue(set(TUNING_SEEDS).isdisjoint(HOLDOUT_SEEDS))

    def test_original_benchmark_still_reproduces_exactly(self):
        result = taskrouter.run()
        self.assertEqual(result["static_merge_mae"], 1.399)
        self.assertEqual(result["routed_mae"], 0.276)
        self.assertEqual(result["error_reduction_pct"], 80.2)

    def test_published_scenario_has_perfect_routing_but_is_not_a_tautology(self):
        """At the published jitter (.45), routing is correct 100% of the
        time on every seed, since regions never overlap (jitter < half the
        2.0 prototype spacing). This isn't circular -- routed_mae still
        depends on genuine label noise -- but it never faces ambiguity."""
        accuracies = [run_at_jitter(seed, 0.45)["routing_accuracy"] for seed in TUNING_SEEDS]
        self.assertEqual(set(accuracies), {1.0})

    def test_stress_jitter_introduces_genuine_routing_ambiguity(self):
        accuracies = [run_at_jitter(seed, STRESS_JITTER)["routing_accuracy"] for seed in TUNING_SEEDS]
        self.assertTrue(any(a < 1.0 for a in accuracies))
        self.assertGreater(len(set(accuracies)), 1)

    def test_routing_advantage_generalizes_under_ambiguity_on_tuning_seeds(self):
        result = summarize(TUNING_SEEDS, STRESS_JITTER)
        self.assertGreater(result["min_reduction_pct"], 40)

    def test_routing_advantage_generalizes_under_ambiguity_on_frozen_holdout_seeds(self):
        result = summarize(HOLDOUT_SEEDS, STRESS_JITTER)
        self.assertGreater(result["min_reduction_pct"], 40)

    def test_original_module_untouched(self):
        import inspect

        source = inspect.getsource(taskrouter.run)
        self.assertIn("rng.uniform(-.45,.45)", source)

    def test_report_is_reproducible(self):
        a = summarize(TUNING_SEEDS[:5], STRESS_JITTER)
        b = summarize(TUNING_SEEDS[:5], STRESS_JITTER)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
