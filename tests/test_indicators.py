import unittest
from datetime import date, timedelta

import pandas as pd

from src.indicators import calculate_indicators


class IndicatorTests(unittest.TestCase):
    def make_history(self, count: int = 40) -> pd.DataFrame:
        today = date(2026, 11, 24)
        rows = []
        for index in range(count):
            close = 10.0 + index * 0.08
            day = today - timedelta(days=(count - index))
            rows.append({
                "date": day.isoformat(),
                "open": close - 0.03,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": 1000 + index * 10,
                "amount": 200_000_000 + index * 1_000_000,
            })
        return pd.DataFrame(rows)

    def test_returns_moving_averages_and_no_buyable_without_intraday(self):
        history = self.make_history()
        result = calculate_indicators(
            history=history,
            current_price=13.20,
            current_change_pct=1.0,
            current_date=date(2026, 11, 24),
        )
        self.assertIsNotNone(result["ma5"])
        self.assertIsNotNone(result["ma10"])
        self.assertIsNotNone(result["ma20"])
        self.assertGreater(result["return_5d_pct"], 0)
        self.assertGreater(result["return_20d_pct"], 0)
        self.assertFalse(result["buyable"])
        self.assertFalse(result["watchlist_eligible"])
        self.assertEqual(result["intraday_confirmation"], "not_checked")
        self.assertIn(result["daily_setup_tier"], ("B", "C"))

    def test_calculates_60_day_return_when_history_is_long_enough(self):
        result = calculate_indicators(
            history=self.make_history(70),
            current_price=16.0,
            current_change_pct=0.5,
            current_date=date(2026, 11, 24),
        )
        self.assertIsNotNone(result["return_60d_pct"])
        self.assertTrue(result["trend_60d_up"])

    def test_current_day_bar_is_excluded(self):
        history = self.make_history()
        current_day = pd.DataFrame([{
            "date": "2026-11-24",
            "open": 999.0,
            "high": 999.0,
            "low": 999.0,
            "close": 999.0,
            "volume": 999999,
            "amount": 999999999999,
        }])
        combined = pd.concat([history, current_day], ignore_index=True)
        result_with_current = calculate_indicators(
            combined, 13.20, 1.0, date(2026, 11, 24)
        )
        result_without_current = calculate_indicators(
            history, 13.20, 1.0, date(2026, 11, 24)
        )
        self.assertEqual(result_with_current["ma20"], result_without_current["ma20"])

    def test_rejects_insufficient_history(self):
        with self.assertRaises(ValueError):
            calculate_indicators(
                self.make_history(20), 12.0, 0.5, date(2026, 11, 24)
            )


if __name__ == "__main__":
    unittest.main()
