import unittest

from tal.stats import mean_ci, paired_bootstrap_drop, rate_stats, wilson_interval


class StatsTest(unittest.TestCase):
    def test_wilson_interval_edges(self):
        low, high = wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.25)
        low, high = wilson_interval(20, 20)
        self.assertEqual(high, 1.0)
        self.assertGreater(low, 0.75)
        empty = wilson_interval(0, 0)
        self.assertEqual((empty.low, empty.high), (0.0, 1.0))

    def test_rate_stats(self):
        stats = rate_stats(3, 4)
        self.assertEqual(stats.n, 4)
        self.assertEqual(stats.successes, 3)
        self.assertAlmostEqual(stats.rate, 0.75)
        self.assertLess(stats.ci.low, 0.75)
        self.assertGreater(stats.ci.high, 0.75)

    def test_paired_drop_perfect_removal(self):
        delta = paired_bootstrap_drop([True] * 30, [False] * 30)
        self.assertEqual(delta.n, 30)
        self.assertEqual(delta.drop, 1.0)
        self.assertEqual(delta.ci.low, 1.0)
        self.assertEqual(delta.ci.high, 1.0)
        self.assertEqual(delta.p_drop_le_zero, 0.0)

    def test_paired_drop_no_change(self):
        delta = paired_bootstrap_drop([True, False] * 10, [True, False] * 10)
        self.assertEqual(delta.drop, 0.0)
        self.assertLessEqual(delta.ci.low, 0.0)
        self.assertGreaterEqual(delta.ci.high, 0.0)
        self.assertGreater(delta.p_drop_le_zero, 0.2)

    def test_mean_ci_constant(self):
        interval = mean_ci([0.5, 0.5, 0.5, 0.5])
        self.assertEqual(interval.low, 0.5)
        self.assertEqual(interval.high, 0.5)


if __name__ == "__main__":
    unittest.main()
