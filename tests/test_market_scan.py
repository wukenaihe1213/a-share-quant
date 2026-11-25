import unittest

import pandas as pd

from src.market_scan import build_history_pool


class MarketScanPoolTests(unittest.TestCase):
    def test_partial_intraday_turnover_is_not_a_hard_liquidity_cutoff(self):
        spot = pd.DataFrame([
            {"code": "000001", "name": "Alpha", "price": 10.0, "amount": 10_000_000, "pct_60d": None},
            {"code": "000002", "name": "Beta", "price": 20.0, "amount": 20_000_000, "pct_60d": None},
            {"code": "000003", "name": "ST Gamma", "price": 30.0, "amount": 30_000_000, "pct_60d": None},
            {"code": "000004", "name": "Bad Price", "price": 0.0, "amount": 40_000_000, "pct_60d": None},
            {"code": "000005", "name": "Delta", "price": 15.0, "amount": 5_000_000, "pct_60d": None},
        ])

        pool, counts = build_history_pool(spot, max_history_stocks=3)

        self.assertEqual(len(pool), 3)
        self.assertEqual(counts["eligible_for_history_scan"], 3)
        self.assertEqual(counts["excluded_st_or_delisting"], 1)
        self.assertEqual(counts["excluded_invalid_price"], 1)
        self.assertEqual(counts["spot_60d_available_count"], 0)
        self.assertEqual(
            counts["history_pool_selection_mode"],
            "spot_amount_rank_only_no_hard_cutoff",
        )
        self.assertEqual(set(pool["code"]), {"000001", "000002", "000005"})


if __name__ == "__main__":
    unittest.main()
