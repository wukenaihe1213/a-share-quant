import unittest

import pandas as pd

from src.market_scan import build_history_pool


class MarketScanPoolTests(unittest.TestCase):

    def test_history_pool_unions_trend_momentum_rebound_and_liquidity_buckets(self):
        spot = pd.DataFrame([
            {"code": "600001", "name": "Liquid", "price": 10, "amount": 900_000_000, "pct_60d": 5, "change_pct": 0, "volume_ratio": 0.9, "turnover_pct": 0.5},
            {"code": "000002", "name": "Trend", "price": 10, "amount": 200_000_000, "pct_60d": 35, "change_pct": 1, "volume_ratio": 1.0, "turnover_pct": 1.0},
            {"code": "300003", "name": "Momentum", "price": 10, "amount": 180_000_000, "pct_60d": 4, "change_pct": 3, "volume_ratio": 2.0, "turnover_pct": 2.0},
            {"code": "688004", "name": "Rebound", "price": 10, "amount": 150_000_000, "pct_60d": -12, "change_pct": 2, "volume_ratio": 1.5, "turnover_pct": 1.2},
            {"code": "830005", "name": "Active", "price": 10, "amount": 100_000_000, "pct_60d": 1, "change_pct": 0.5, "volume_ratio": 0.9, "turnover_pct": 8.0},
            {"code": "600006", "name": "ST Excluded", "price": 10, "amount": 800_000_000, "pct_60d": 20, "change_pct": 5, "volume_ratio": 3, "turnover_pct": 5},
        ])
        pool, counts = build_history_pool(spot, max_history_stocks=5)
        self.assertEqual(len(pool), 5)
        self.assertNotIn("600006", set(pool["code"]))
        self.assertIn("300003", set(pool["code"]))
        self.assertIn("688004", set(pool["code"]))
        self.assertEqual(
            counts["history_pool_selection_mode"],
            "multi_strategy_bucket_union_with_board_interleave",
        )
        self.assertGreaterEqual(len(counts["history_pool_bucket_selected_counts"]), 4)

    def test_partial_intraday_turnover_is_not_a_hard_liquidity_cutoff(self):
        spot = pd.DataFrame([
            {"code": "000001", "name": "Alpha", "price": 10.0, "amount": 10_000_000, "pct_60d": float("nan")},
            {"code": "000002", "name": "Beta", "price": 20.0, "amount": 20_000_000, "pct_60d": float("nan")},
            {"code": "000003", "name": "ST Gamma", "price": 30.0, "amount": 30_000_000, "pct_60d": float("nan")},
            {"code": "000004", "name": "Bad Price", "price": 0.0, "amount": 40_000_000, "pct_60d": float("nan")},
            {"code": "000005", "name": "Delta", "price": 15.0, "amount": 5_000_000, "pct_60d": float("nan")},
        ])

        pool, counts = build_history_pool(spot, max_history_stocks=3)

        self.assertEqual(len(pool), 3)
        self.assertEqual(counts["eligible_for_history_scan"], 3)
        self.assertEqual(counts["excluded_st_or_delisting"], 1)
        self.assertEqual(counts["excluded_invalid_price"], 1)
        self.assertEqual(counts["spot_60d_available_count"], 0)
        self.assertEqual(
            counts["history_pool_selection_mode"],
            "multi_strategy_bucket_union_with_board_interleave",
        )
        self.assertEqual(set(pool["code"]), {"000001", "000002", "000005"})


if __name__ == "__main__":
    unittest.main()
