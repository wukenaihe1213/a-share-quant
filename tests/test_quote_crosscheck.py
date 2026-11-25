import unittest

import pandas as pd

from src.quote_crosscheck import (
    classify_price_difference,
    compare_pair,
    parse_tencent_timestamp,
)


class QuoteCrosscheckTests(unittest.TestCase):
    def test_price_difference_thresholds(self):
        self.assertEqual(classify_price_difference(0.10), "normal")
        self.assertEqual(classify_price_difference(0.20), "normal")
        self.assertEqual(classify_price_difference(0.35), "review")
        self.assertEqual(classify_price_difference(0.50), "review")
        self.assertEqual(classify_price_difference(0.80), "abnormal")
        self.assertEqual(classify_price_difference(None), "unavailable")

    def test_compare_pair_calculates_price_and_amount_differences(self):
        left = pd.DataFrame([
            {"code": "600000", "name": "浦发银行", "price": 10.00, "change_pct": 1.0, "amount": 100_000_000},
            {"code": "000001", "name": "平安银行", "price": 12.00, "change_pct": 2.0, "amount": 200_000_000},
            {"code": "300750", "name": "宁德时代", "price": 200.00, "change_pct": -1.0, "amount": 300_000_000},
        ])
        right = pd.DataFrame([
            {"code": "600000", "name": "浦发银行", "price": 10.01, "change_pct": 1.1, "amount": 101_000_000},
            {"code": "000001", "name": "平安银行", "price": 12.00, "change_pct": 2.0, "amount": 199_000_000},
            {"code": "300750", "name": "宁德时代", "price": 201.20, "change_pct": -0.8, "amount": 310_000_000},
        ])

        summary, details = compare_pair("eastmoney", left, "sina", right, 3.0)

        self.assertEqual(summary["common_count"], 3)
        self.assertEqual(summary["collection_time_gap_seconds"], 3.0)
        self.assertAlmostEqual(
            float(details.loc[details["code"] == "600000", "price_diff_pct"].iloc[0]),
            abs(10.00 - 10.01) / 10.005 * 100,
            places=4,
        )
        self.assertEqual(
            details.loc[details["code"] == "000001", "price_status"].iloc[0],
            "normal",
        )
        self.assertIn("amount_diff_pct", details.columns)

    def test_tencent_timestamp_parser(self):
        parsed = parse_tencent_timestamp("20261125143000")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.strftime("%Y-%m-%d %H:%M:%S"), "2026-11-25 14:30:00")
        self.assertIsNone(parse_tencent_timestamp("not-a-time"))


if __name__ == "__main__":
    unittest.main()
