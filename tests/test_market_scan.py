import unittest

import pandas as pd

from src.market_scan import build_history_pool, merge_tencent_quote_fields


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

    def test_tencent_supplement_fills_only_missing_metrics_and_preserves_primary_price(self):
        spot = pd.DataFrame([
            {
                "code": "600000", "name": "浦发银行", "price": 10.10,
                "change_pct": 1.2, "amount": 100_000_000,
                "volume": 10_000, "volume_ratio": float("nan"),
                "turnover_pct": float("nan"),
            },
            {
                "code": "000001", "name": "平安银行", "price": 12.30,
                "change_pct": -0.2, "amount": 200_000_000,
                "volume": 20_000, "volume_ratio": 1.8,
                "turnover_pct": 2.1,
            },
        ])
        quotes = [
            {
                "code": "600000", "price": 10.12, "change_pct": 1.4,
                "amount": 110_000_000, "volume": 11_000,
                "volume_ratio": 1.5, "turnover_pct": 0.7,
                "quote_timestamp": "20261125143000",
            },
            {
                "code": "000001", "price": 12.28, "change_pct": -0.3,
                "amount": 210_000_000, "volume": 21_000,
                "volume_ratio": 2.0, "turnover_pct": 2.2,
                "quote_timestamp": "20261125143000",
            },
        ]

        merged, stats = merge_tencent_quote_fields(spot, quotes)

        first = merged.loc[merged["code"] == "600000"].iloc[0]
        second = merged.loc[merged["code"] == "000001"].iloc[0]
        self.assertEqual(first["price"], 10.10)
        self.assertEqual(first["change_pct"], 1.2)
        self.assertEqual(first["volume_ratio"], 1.5)
        self.assertEqual(first["turnover_pct"], 0.7)
        self.assertEqual(first["volume_ratio_source"], "tencent_qt_gtimg")
        self.assertEqual(first["turnover_pct_source"], "tencent_qt_gtimg")
        self.assertEqual(first["tencent_quote_timestamp"], "20261125143000")
        self.assertEqual(second["price"], 12.30)
        self.assertEqual(second["volume_ratio"], 1.8)
        self.assertEqual(second["turnover_pct"], 2.1)
        self.assertEqual(second["volume_ratio_source"], "akshare_primary")
        self.assertEqual(stats["returned_symbols"], 2)
        self.assertEqual(stats["volume_ratio_filled"], 1)
        self.assertEqual(stats["turnover_pct_filled"], 1)

    def test_tencent_supplement_unavailable_keeps_missing_metrics_null(self):
        spot = pd.DataFrame([
            {
                "code": "600000", "name": "浦发银行", "price": 10.10,
                "change_pct": 1.2, "amount": 100_000_000,
                "volume_ratio": float("nan"), "turnover_pct": float("nan"),
            },
        ])
        merged, stats = merge_tencent_quote_fields(spot, [])
        self.assertTrue(pd.isna(merged.loc[0, "volume_ratio"]))
        self.assertTrue(pd.isna(merged.loc[0, "turnover_pct"]))
        self.assertEqual(stats["status"], "unavailable")
        self.assertIsNone(merged.loc[0, "volume_ratio_source"])
        self.assertFalse(bool(merged.loc[0, "tencent_quote_available"]))

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
