import unittest
import taskrouter

class BenchmarkTest(unittest.TestCase):
    def test_research_method_clears_baseline(self):
        result = taskrouter.run()
        self.assertGreaterEqual(result["error_reduction_pct"], 45)

if __name__ == "__main__":
    unittest.main()
