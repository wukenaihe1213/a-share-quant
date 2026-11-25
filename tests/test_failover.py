import unittest
from unittest.mock import patch

import pandas as pd

import src.market_scan as market_scan


class FailoverTests(unittest.TestCase):
    def test_spot_falls_through_to_tencent_when_akshare_sources_fail(self):
        fake_quotes = pd.DataFrame([
            {
                "code": f"{index:06d}",
                "name": f"Stock{index}",
                "price": 10.0,
                "change_pct": 1.0,
                "amount": 100_000_000.0,
                "volume": 100_000.0,
                "turnover_pct": 1.0,
                "volume_ratio": 1.1,
                "pct_60d": float("nan"),
                "industry": None,
            }
            for index in range(1000)
        ])
        source_name = "tencent_qt_gtimg_baostock_universe"

        with patch.object(
            market_scan, "interface_alternatives",
            return_value=["stock_zh_a_spot_sina"]
        ), patch.object(
            market_scan.ak, "stock_zh_a_spot_em",
            side_effect=RuntimeError("injected Eastmoney outage"), create=True
        ), patch.object(
            market_scan.ak, "stock_zh_a_spot_sina",
            side_effect=RuntimeError("injected Sina outage"), create=True
        ), patch.object(
            market_scan.ak, "stock_zh_a_spot",
            side_effect=RuntimeError("injected AKShare generic spot outage"), create=True
        ), patch.object(
            market_scan, "fetch_tencent_spot_universe",
            return_value=(fake_quotes, source_name, ["mock Tencent coverage: 100%"])
        ) as tencent:
            frame, source, errors = market_scan.fetch_spot_universe()

        self.assertEqual(source, source_name)
        self.assertEqual(len(frame), 1000)
        self.assertGreaterEqual(len(errors), 3)
        self.assertIn("injected Eastmoney outage", " ".join(errors))
        self.assertIn("injected Sina outage", " ".join(errors))
        self.assertIn("injected AKShare generic spot outage", " ".join(errors))
        tencent.assert_called_once()

    def test_history_falls_back_to_baostock_when_akshare_history_fails(self):
        raw = pd.DataFrame([
            {
                "date": "2026-11-20",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 100000,
                "amount": 1020000,
            },
            {
                "date": "2026-11-23",
                "open": 10.2,
                "high": 10.6,
                "low": 10.1,
                "close": 10.4,
                "volume": 110000,
                "amount": 1144000,
            },
        ])

        with patch.object(
            market_scan, "interface_alternatives", return_value=[]
        ), patch.object(
            market_scan.ak, "stock_zh_a_hist_tx",
            side_effect=RuntimeError("injected Tencent-history outage"), create=True
        ), patch.object(
            market_scan.ak, "stock_zh_a_hist_sina",
            side_effect=RuntimeError("injected Sina-history outage"), create=True
        ), patch.object(
            market_scan.ak, "stock_zh_a_daily",
            side_effect=RuntimeError("injected daily-history outage"), create=True
        ), patch.object(
            market_scan, "fetch_baostock_history", return_value=raw
        ) as baostock:
            frame, source, eastmoney_limited = market_scan.fetch_history(
                "600000", "20261101", "20261125", skip_eastmoney=True
            )

        self.assertEqual(source, "baostock_query_history_k_data_plus")
        self.assertEqual(len(frame), 2)
        self.assertEqual(float(frame.iloc[-1]["close"]), 10.4)
        self.assertFalse(eastmoney_limited)
        baostock.assert_called_once_with("600000", "20261101", "20261125")


if __name__ == "__main__":
    unittest.main()
