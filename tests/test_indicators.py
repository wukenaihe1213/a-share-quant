import unittest
from datetime import date, timedelta

import pandas as pd

from src.indicators import calculate_indicators, calculate_position_plan


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

    def make_bottom_history(self) -> pd.DataFrame:
        today = date(2026, 11, 24)
        tail = [10.00, 9.95, 9.85, 9.75, 9.72, 9.76, 9.82, 9.95, 10.10, 10.20]
        rows = []
        for index in range(70):
            if index < 50:
                close = 20.0 - index * 0.18
            elif index < 60:
                close = 11.18 - (index - 49) * 0.11
            else:
                close = tail[index - 60]
            volume = 700 if index >= 67 else 1500
            rows.append({
                "date": (today - timedelta(days=70 - index)).isoformat(),
                "open": close - 0.04,
                "high": close + 0.12,
                "low": close - 0.10,
                "close": close,
                "volume": volume,
                "amount": 200_000_000,
            })
        return pd.DataFrame(rows)

    def test_negative_60_day_return_is_not_a_hard_veto_for_trend_base(self):
        history = self.make_history(70)
        # Make the 60-day anchor materially higher while keeping the latest
        # 20 sessions rising. A negative 60-day return is a score input, not a veto.
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
        self.assertTrue(result["trend_up"])
        self.assertIn(result["trend_pullback_tier"], ("B", "C", "N"))
        self.assertFalse(result["buyable"])

    def test_trend_pullback_watchlist_and_no_buyable_without_intraday(self):
        result = calculate_indicators(
            history=self.make_history(40),
            current_price=13.20,
            current_change_pct=1.0,
            current_date=date(2026, 11, 24),
        )
        self.assertIsNotNone(result["ma5"])
        self.assertIsNotNone(result["ma10"])
        self.assertIsNotNone(result["ma20"])
        self.assertGreater(result["return_5d_pct"], 0)
        self.assertGreater(result["return_20d_pct"], 0)
        self.assertIn(result["daily_setup_tier"], ("B", "C", "N"))
        self.assertFalse(result["buyable"])
        self.assertEqual(result["intraday_confirmation"], "not_checked")

    def test_bottom_reversal_qualifies_while_60_day_trend_is_down(self):
        result = calculate_indicators(
            history=self.make_bottom_history(),
            current_price=10.35,
            current_change_pct=1.2,
            current_date=date(2026, 11, 24),
        )
        self.assertLess(result["return_20d_pct"], 0)
        self.assertLess(result["return_60d_pct"], 0)
        self.assertTrue(result["bottom_prior_decline"])
        self.assertTrue(result["bottom_near_20d_low"])
        self.assertTrue(result["bottom_near_60d_low"])
        self.assertTrue(result["bottom_stabilization"])
        self.assertEqual(result["bottom_reversal_tier"], "B")
        self.assertEqual(result["candidate_strategy"], "bottom_reversal")
        self.assertFalse(result["buyable"])

    def test_falling_knife_below_recent_low_is_not_bottom_reversal_candidate(self):
        result = calculate_indicators(
            history=self.make_bottom_history(),
            current_price=9.40,
            current_change_pct=-4.8,
            current_date=date(2026, 11, 24),
        )
        self.assertFalse(result["bottom_stabilization"])
        self.assertFalse(result["bottom_reversal_tier"] == "B")
        self.assertEqual(result["bottom_reversal_tier"], "N")

    def test_overheated_runup_is_not_trend_pullback_b_tier(self):
        history = self.make_history(70)
        history.loc[60:, ["open", "high", "low", "close"]] *= 1.4
        ma5 = float(history["close"].tail(5).mean())
        history.loc[65:69, "high"] = history.loc[65:69, "close"] + 0.05
        history.loc[68, "high"] = ma5 * 1.05
        history.loc[67:69, "volume"] = 500
        result = calculate_indicators(
            history=history,
            current_price=ma5 * 0.99,
            current_change_pct=0.5,
            current_date=date(2026, 11, 24),
        )
        self.assertGreater(result["runup_10d_pct"], 30)
        self.assertFalse(result["runup_acceptable"])
        self.assertNotEqual(result["trend_pullback_tier"], "B")

    def test_trend_strategy_does_not_promote_a_no_pullback_runup_to_b(self):
        history = self.make_history(70)
        current_price = float(history["high"].tail(5).max()) * 1.01
        result = calculate_indicators(
            history=history,
            current_price=current_price,
            current_change_pct=0.5,
            current_date=date(2026, 11, 24),
        )
        self.assertEqual(result["pullback_depth_pct"], 0.0)
        self.assertNotEqual(result["trend_pullback_tier"], "B")

    def test_severe_medium_term_decline_is_not_bottom_reversal_b_tier(self):
        history = self.make_bottom_history()
        history.loc[10:20, "high"] *= 1.5
        result = calculate_indicators(
            history=history,
            current_price=10.35,
            current_change_pct=1.2,
            current_date=date(2026, 11, 24),
        )
        self.assertTrue(result["bottom_severe_decline"])
        self.assertNotEqual(result["bottom_reversal_tier"], "B")

    def test_unaffordable_one_lot_is_not_b_tier(self):
        history = self.make_bottom_history()
        history[["open", "high", "low", "close"]] *= 4
        result = calculate_indicators(
            history=history,
            current_price=41.40,
            current_change_pct=1.2,
            current_date=date(2026, 11, 24),
        )
        self.assertFalse(result["affordable"])
        self.assertGreater(result["one_lot_cost_cny"], 3500)
        self.assertNotEqual(result["daily_setup_tier"], "B")

    def test_atr14_is_reported_as_volatility_feature(self):
        result = calculate_indicators(
            history=self.make_history(40),
            current_price=13.20,
            current_change_pct=1.0,
            current_date=date(2026, 11, 24),
        )
        self.assertGreater(result["atr14"], 0)
        self.assertGreater(result["atr_pct"], 0)
        self.assertIsInstance(result["atr_suitable"], bool)

    def test_position_plan_respects_100_share_lots_cash_and_five_percent_risk(self):
        plan = calculate_position_plan(current_price=12.0)
        self.assertEqual(plan["budget_per_position_cny"], 3500.0)
        self.assertEqual(plan["lot_size_shares"], 100)
        self.assertEqual(plan["affordable_shares"], 200)
        self.assertEqual(plan["estimated_position_value_cny"], 2400.0)
        self.assertEqual(plan["max_loss_per_trade_pct"], 5.0)
        self.assertEqual(plan["estimated_loss_at_max_risk_cny"], 120.0)
        self.assertEqual(plan["estimated_profit_at_target_min_cny"], 72.0)
        self.assertEqual(plan["estimated_profit_at_target_max_cny"], 120.0)

    def test_position_plan_rejects_price_when_one_lot_exceeds_budget(self):
        plan = calculate_position_plan(current_price=40.0)
        self.assertFalse(plan["affordable"])
        self.assertEqual(plan["affordable_shares"], 0)

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
