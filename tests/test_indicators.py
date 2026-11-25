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
        self.assertTrue(result["trend_up"])

    def test_rejects_short_term_rebound_when_60_day_trend_is_down(self):
        history = self.make_history(70)
        # Keep the latest 20 sessions rising, but make the 60-session anchor
        # materially higher so this is a short-term rebound in a wider downtrend.
        history.loc[:9, ["open", "high", "low", "close"]] *= 2
        result = calculate_indicators(
            history=history,
            current_price=16.0,
            current_change_pct=0.5,
            current_date=date(2026, 11, 24),
        )
        self.assertGreater(result["return_20d_pct"], 0)
        self.assertLess(result["return_60d_pct"], 0)
        self.assertFalse(result["trend_60d_up"])
        self.assertFalse(result["trend_up"])
        self.assertFalse(result["watchlist_eligible"])

    def test_marks_partial_but_structurally_sound_setup_as_watchlist_b(self):
        today = date(2026, 11, 24)
        rows = []
        for index in range(70):
            close = 10.0 + index * 0.03
            high = close + 0.10
            volume = 1000
            if index >= 60:
                tail = [
                    (12.0, 12.1), (12.2, 12.3), (12.5, 12.6),
                    (12.8, 12.9), (13.1, 13.2), (13.4, 13.5),
                    (13.2, 13.3), (13.0, 13.1), (12.9, 13.0),
                    (12.95, 13.0),
                ]
                close, high = tail[index - 60]
            if index >= 67:
                volume = 500
            rows.append({
                "date": (today - timedelta(days=70-index)).isoformat(),
                "open": close - 0.03,
                "high": high,
                "low": close - 0.10,
                "close": close,
                "volume": volume,
                "amount": 200_000_000,
            })
        result = calculate_indicators(
            history=pd.DataFrame(rows),
            current_price=13.0,
            current_change_pct=0.5,
            current_date=today,
        )
        self.assertTrue(result["trend_up"])
        self.assertTrue(result["pullback_ok"])
        self.assertTrue(result["volume_shrinking"])
        self.assertTrue(result["support_near"])
        self.assertEqual(result["daily_setup_tier"], "B")
        self.assertTrue(result["watchlist_eligible"])
        self.assertFalse(result["buyable"])

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
