from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def calculate_position_plan(
    current_price: float,
    account_capital_cny: float = 10_000.0,
    cash_reserve_pct: float = 30.0,
    max_positions: int = 2,
    single_position_budget_cap_cny: float = 10_000.0,
    max_loss_per_trade_pct: float = 5.0,
    target_profit_min_pct: float = 3.0,
    target_profit_max_pct: float = 5.0,
    lot_size: int = 100,
) -> dict[str, Any]:
    """Create an illustrative A-share position plan using whole 100-share lots."""
    if current_price is None or current_price <= 0:
        raise ValueError("current_price must be a positive number")
    if account_capital_cny <= 0 or max_positions < 1 or lot_size < 1:
        raise ValueError("Capital, max_positions and lot_size must be positive")
    if single_position_budget_cap_cny <= 0:
        raise ValueError("single_position_budget_cap_cny must be positive")
    if not 0 <= cash_reserve_pct < 100:
        raise ValueError("cash_reserve_pct must be in [0, 100)")
    if not 0 < max_loss_per_trade_pct <= 100:
        raise ValueError("max_loss_per_trade_pct must be in (0, 100]")

    investable_capital = account_capital_cny * (1 - cash_reserve_pct / 100)
    # Single-position scenario: the position may use the available investment
    # capital, subject to the user's per-stock ceiling. If two positions are
    # held simultaneously, the illustrative equal-split budget is shown too.
    per_position_budget = min(single_position_budget_cap_cny, investable_capital)
    two_position_budget = investable_capital / max_positions
    one_lot_cost = current_price * lot_size
    lots = int((per_position_budget + 1e-9) / one_lot_cost)
    shares = lots * lot_size
    two_position_lots = int((two_position_budget + 1e-9) / one_lot_cost)
    shares_if_two_positions = two_position_lots * lot_size
    position_value = shares * current_price

    return {
        "account_capital_cny": round(account_capital_cny, 2),
        "cash_reserve_pct": round(cash_reserve_pct, 2),
        "max_positions": int(max_positions),
        "single_position_budget_cap_cny": round(single_position_budget_cap_cny, 2),
        "investable_capital_after_cash_reserve_cny": round(investable_capital, 2),
        "budget_per_position_cny": round(per_position_budget, 2),
        "budget_if_two_positions_cny": round(two_position_budget, 2),
        "lot_size_shares": int(lot_size),
        "one_lot_cost_cny": round(one_lot_cost, 2),
        "affordable_shares": int(shares),
        "affordable_shares_if_two_positions": int(shares_if_two_positions),
        "estimated_position_value_cny": round(position_value, 2),
        "estimated_position_value_if_two_positions_cny": round(
            shares_if_two_positions * current_price, 2
        ),
        "max_loss_per_trade_pct": round(max_loss_per_trade_pct, 2),
        "estimated_loss_at_max_risk_cny": round(
            position_value * max_loss_per_trade_pct / 100, 2
        ),
        "estimated_profit_at_target_min_cny": round(
            position_value * target_profit_min_pct / 100, 2
        ),
        "estimated_profit_at_target_max_cny": round(
            position_value * target_profit_max_pct / 100, 2
        ),
        "affordable": bool(shares >= lot_size),
        "affordable_if_two_positions": bool(shares_if_two_positions >= lot_size),
        "fees_included": False,
        "note": (
            "Illustrative sizing only; actual fill price, fees, slippage and gap risk "
            "can change the result. A 5% stop is not guaranteed to execute at that loss."
        ),
    }


def calculate_indicators(
    history: pd.DataFrame,
    current_price: float,
    current_change_pct: float | None,
    current_date: date,
    average_amount_floor: float = 50_000_000,
    account_capital_cny: float = 10_000.0,
    cash_reserve_pct: float = 30.0,
    max_positions: int = 2,
    single_position_budget_cap_cny: float = 10_000.0,
    max_loss_per_trade_pct: float = 5.0,
    current_volume_ratio: float | None = None,
    current_turnover_pct: float | None = None,
    current_amount_cny: float | None = None,
) -> dict[str, Any]:
    """Score six independent short-term setups using completed daily bars + live snapshot.

    Signals are research/watchlist scores, not forecasts or executable buy orders.
    Intraday minute-bar confirmation is not implemented, so buyable remains false.
    """
    required = {"date", "high", "low", "close", "volume", "amount"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"History is missing canonical columns: {sorted(missing)}")

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame.loc[frame["date"].dt.date < current_date]
    for column in ("high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["high", "low", "close", "volume", "amount"])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if len(frame) < 25:
        raise ValueError(f"At least 25 completed daily bars are required; got {len(frame)}")
    if current_price is None or current_price <= 0:
        raise ValueError("current_price must be a positive number")

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    amount = frame["amount"]

    try:
        current_volume_ratio = (
            float(current_volume_ratio)
            if current_volume_ratio is not None and pd.notna(current_volume_ratio)
            else None
        )
    except (TypeError, ValueError):
        current_volume_ratio = None
    try:
        current_turnover_pct = (
            float(current_turnover_pct)
            if current_turnover_pct is not None and pd.notna(current_turnover_pct)
            else None
        )
    except (TypeError, ValueError):
        current_turnover_pct = None
    try:
        current_amount_cny = (
            float(current_amount_cny)
            if current_amount_cny is not None and pd.notna(current_amount_cny)
            else None
        )
    except (TypeError, ValueError):
        current_amount_cny = None

    ma5 = float(close.tail(5).mean())
    ma10 = float(close.tail(10).mean())
    ma20 = float(close.tail(20).mean())
    ma20_previous = float(close.iloc[-25:-5].mean())
    ma5_previous = float(close.iloc[-10:-5].mean())

    return_3d = (current_price / float(close.iloc[-4]) - 1) * 100
    return_5d = (current_price / float(close.iloc[-6]) - 1) * 100
    return_10d = (current_price / float(close.iloc[-11]) - 1) * 100
    return_20d = (current_price / float(close.iloc[-21]) - 1) * 100
    return_60d = (
        (current_price / float(close.iloc[-61]) - 1) * 100
        if len(close) >= 61 else None
    )

    runup_10d = (float(high.tail(10).max()) / float(close.iloc[-11]) - 1) * 100
    high_5d = float(high.tail(5).max())
    five_bar_high_series = high.tail(5).reset_index(drop=True)
    peak_position = int(five_bar_high_series.idxmax())
    days_since_peak = 5 - peak_position
    pullback_pct = (current_price / high_5d - 1) * 100
    pullback_depth = max(0.0, -pullback_pct)

    low_20d = float(low.tail(20).min())
    high_20d = float(high.tail(20).max())
    distance_20d_high = (current_price / high_20d - 1) * 100 if high_20d > 0 else None
    high_60d = float(high.tail(60).max()) if len(frame) >= 60 else None
    low_60d = float(low.tail(60).min()) if len(frame) >= 60 else None
    distance_20d_low = (current_price / low_20d - 1) * 100 if low_20d > 0 else None
    distance_60d_low = (
        (current_price / low_60d - 1) * 100
        if low_60d is not None and low_60d > 0 else None
    )
    drawdown_60d_high = (
        (current_price / high_60d - 1) * 100
        if high_60d is not None and high_60d > 0 else None
    )
    rebound_from_5d_low = (
        (current_price / float(low.tail(5).min()) - 1) * 100
        if float(low.tail(5).min()) > 0 else None
    )

    previous_5_volume = float(volume.iloc[-8:-3].mean())
    recent_3_volume = float(volume.tail(3).mean())
    volume_shrink_ratio = (
        recent_3_volume / previous_5_volume if previous_5_volume > 0 else None
    )
    average_amount_20d = float(amount.tail(20).mean())

    # ATR(14) is used as a volatility suitability/ranking feature, not as a
    # forecast or a guaranteed stop distance.
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = float(true_range.tail(14).mean())
    atr_pct = atr14 / current_price * 100 if current_price > 0 else None
    atr_suitable = bool(atr_pct is not None and 1.2 <= atr_pct <= 6.5)

    investable_capital = account_capital_cny * (1 - cash_reserve_pct / 100)
    budget_per_position = min(single_position_budget_cap_cny, investable_capital)
    budget_if_two_positions = investable_capital / max_positions
    one_lot_cost = current_price * 100
    affordable = bool(one_lot_cost <= budget_per_position + 1e-9)
    affordable_if_two_positions = bool(
        one_lot_cost <= budget_if_two_positions + 1e-9
    )

    ma_values = {"MA5": ma5, "MA10": ma10, "MA20": ma20}
    support_distances = {
        name: abs(current_price / value - 1) * 100
        for name, value in ma_values.items()
        if value > 0
    }
    nearest_support = (
        min(support_distances, key=support_distances.get)
        if support_distances else None
    )
    nearest_support_distance = (
        support_distances.get(nearest_support) if nearest_support else None
    )
    support_near = False
    if nearest_support is not None:
        threshold = 5.0 if nearest_support == "MA20" else 3.0
        support_value = ma_values[nearest_support]
        support_near = (
            nearest_support_distance is not None
            and nearest_support_distance <= threshold
            and current_price >= support_value * 0.97
        )

    ma20_slope_pct = (
        (ma20 / ma20_previous - 1) * 100 if ma20_previous > 0 else None
    )
    ma5_slope_pct = (
        (ma5 / ma5_previous - 1) * 100 if ma5_previous > 0 else None
    )
    trend_60d_up = bool(return_60d is not None and return_60d > 0)
    trend_up = bool(
        return_20d > 0
        and ma20_slope_pct is not None
        and ma20_slope_pct >= -0.5
        and current_price >= ma20 * 0.98
    )

    runup_preferred = 8 <= runup_10d <= 20
    runup_acceptable = 3 <= runup_10d <= 30
    pullback_ok = 1.5 <= pullback_depth <= 8 and 2 <= days_since_peak <= 5
    volume_shrinking = (
        volume_shrink_ratio is not None and volume_shrink_ratio <= 0.90
    )
    liquidity_ok = average_amount_20d >= average_amount_floor
    no_chase = current_change_pct is None or current_change_pct <= 4.0
    not_crashing = current_change_pct is None or current_change_pct >= -5.0
    daily_risk_ok = no_chase and not_crashing

    # Strategy A: an established short-term uptrend that is pulling back.
    trend_score = 0
    trend_reasons: list[str] = []
    trend_missing: list[str] = []
    if return_20d > 0:
        trend_score += 15
        trend_reasons.append("近20日涨幅为正")
    else:
        trend_missing.append("近20日趋势尚未转正")
    if ma20_slope_pct is not None and ma20_slope_pct > 0:
        trend_score += 10
        trend_reasons.append("MA20斜率向上")
    elif ma20_slope_pct is not None and ma20_slope_pct >= -0.5:
        trend_score += 5
        trend_reasons.append("MA20走平，尚未明显转弱")
    else:
        trend_missing.append("MA20斜率偏弱")
    if trend_60d_up:
        trend_score += 5
        trend_reasons.append("60日趋势向上（加分项）")
    elif return_60d is None:
        trend_missing.append("60日数据不足，未计分")
    else:
        trend_missing.append("60日趋势向下，但不单独否决趋势回调策略")
    if runup_preferred:
        trend_score += 15
        trend_reasons.append("近10日上涨波段处于优先区间")
    elif runup_acceptable:
        trend_score += 8
        trend_reasons.append("此前存在可接受的上涨波段")
    else:
        trend_missing.append("此前上涨波段不足或短期涨幅过热")
    if pullback_ok:
        trend_score += 20
        trend_reasons.append("近期回撤幅度和持续时间合适")
    elif 0.5 <= pullback_depth <= 10:
        trend_score += 8
        trend_missing.append("回调幅度或持续时间不完全理想")
    else:
        trend_missing.append("尚未形成明确的短期回调")
    if volume_shrinking:
        trend_score += 15
        trend_reasons.append("回调阶段成交量明显收缩")
    elif volume_shrink_ratio is not None and volume_shrink_ratio <= 1.05:
        trend_score += 8
        trend_missing.append("成交量略有收缩，信号一般")
    else:
        trend_missing.append("回调阶段尚未明显缩量")
    if support_near:
        trend_score += 15
        trend_reasons.append(f"价格接近{nearest_support}支撑")
    else:
        trend_missing.append("价格尚未靠近明确均线支撑")
    if liquidity_ok:
        trend_score += 10
        trend_reasons.append("近20日平均成交额达到流动性门槛")
    else:
        trend_missing.append("近20日平均成交额不足或数据缺失")
    if daily_risk_ok:
        trend_score += 5
    else:
        trend_missing.append("当日涨跌幅过大，暂不作为买入候选")

    trend_base_ok = (
        trend_up and liquidity_ok and daily_risk_ok
    )
    trend_structure_signals = sum((pullback_ok, volume_shrinking, support_near))
    if atr_suitable:
        trend_score += 5
        trend_reasons.append("ATR波动率处于短线观察区间")
    else:
        trend_missing.append("ATR波动率偏低或偏高，短线空间/风险不理想")

    if not affordable:
        trend_missing.append("100股一手超过单只可用预算（账户预留现金后，单只上限不超过10000元）")

    # B tier requires a real 2-5 day pullback and at least one additional
    # confirmation (shrinking volume or nearby support). No-pullback leaders
    # are not promoted into a pullback strategy just because the score is high.
    trend_b_ready = bool(
        trend_base_ok
        and runup_acceptable
        and pullback_ok
        and trend_structure_signals >= 2
        and atr_suitable
        and affordable
        and trend_score >= 60
    )
    trend_c_ready = bool(
        trend_base_ok
        and runup_acceptable
        and 0.8 <= pullback_depth <= 10.0
        and 1 <= days_since_peak <= 5
        and trend_structure_signals >= 1
        and trend_score >= 40
    )
    if trend_b_ready:
        trend_tier = "B"
    elif trend_c_ready:
        trend_tier = "C"
    else:
        trend_tier = "N"

    # Strategy B: a base/reversal setup near a recent low. A falling knife
    # is rejected unless price is holding the 20-day low area and stabilizing.
    near_20d_low = bool(
        distance_20d_low is not None and -3.0 <= distance_20d_low <= 8.0
    )
    near_60d_low = bool(
        distance_60d_low is not None and -3.0 <= distance_60d_low <= 15.0
    )
    near_bottom_zone = near_20d_low or near_60d_low
    base_holds_20d_low = bool(
        distance_20d_low is not None and distance_20d_low >= -1.0
    )
    prior_decline = bool(
        (drawdown_60d_high is not None and drawdown_60d_high <= -8.0)
        or return_20d <= -5.0
    )
    stabilization = bool(
        rebound_from_5d_low is not None
        and rebound_from_5d_low >= 1.5
        and (
            return_3d > 0
            or (current_change_pct is not None and current_change_pct >= 0.5)
            or (ma5_slope_pct is not None and ma5_slope_pct >= 0)
        )
    ) or bool(
        return_3d >= 1.0
        and current_price >= ma5 * 0.98
    )
    volume_bottom_confirmation = bool(
        volume_shrink_ratio is not None
        and (
            volume_shrink_ratio <= 1.05
            or (
                volume_shrink_ratio >= 1.10
                and return_3d > 0
                and (current_change_pct is None or current_change_pct > 0)
            )
        )
    )
    bottom_repair = bool(
        current_price >= ma5 * 0.98
        and (
            (ma5_slope_pct is not None and ma5_slope_pct >= -0.2)
            or return_3d >= 1.0
        )
    )
    severe_decline = bool(
        return_20d <= -25.0
        or (drawdown_60d_high is not None and drawdown_60d_high <= -50.0)
        or (ma20_slope_pct is not None and ma20_slope_pct <= -8.0)
    )

    bottom_score = 0
    bottom_reasons: list[str] = []
    bottom_missing: list[str] = []
    if prior_decline:
        bottom_score += 15
        bottom_reasons.append("此前存在阶段性回撤")
    else:
        bottom_missing.append("尚无明显阶段性下跌或调整")
    if near_20d_low:
        bottom_score += 15
        bottom_reasons.append("价格接近20日区间低位")
    if near_60d_low:
        bottom_score += 20
        bottom_reasons.append("价格接近60日区间低位")
    if base_holds_20d_low:
        bottom_score += 15
        bottom_reasons.append("暂未有效跌破20日区间低位")
    else:
        bottom_missing.append("价格已明显跌破20日区间低位")
    if stabilization:
        bottom_score += 25
        bottom_reasons.append("近几日出现反弹或止跌迹象")
    else:
        bottom_missing.append("尚未出现足够明确的止跌确认")
    if volume_bottom_confirmation:
        bottom_score += 10
        bottom_reasons.append("成交量收缩或反弹时温和放大")
    else:
        bottom_missing.append("成交量尚未确认企稳")
    if liquidity_ok:
        bottom_score += 10
        bottom_reasons.append("近20日平均成交额达到流动性门槛")
    else:
        bottom_missing.append("近20日平均成交额不足或数据缺失")
    if atr_suitable:
        bottom_score += 5
        bottom_reasons.append("ATR波动率处于短线观察区间")
    else:
        bottom_missing.append("ATR波动率偏低或偏高，短线空间/风险不理想")
    if daily_risk_ok:
        bottom_score += 5
    else:
        bottom_missing.append("当日涨跌幅过大，暂不参与")

    if severe_decline:
        bottom_score = max(0, bottom_score - 20)
        bottom_missing.append("中期跌幅或MA20下行斜率过大，防范接飞刀")
    if not bottom_repair:
        bottom_missing.append("尚未站回MA5附近或MA5未出现企稳迹象")
    if not affordable:
        bottom_missing.append("100股一手超过单只可用预算（账户预留现金后，单只上限不超过10000元）")

    bottom_base_ok = (
        prior_decline and near_bottom_zone and base_holds_20d_low
        and liquidity_ok and daily_risk_ok
    )
    bottom_b_ready = bool(
        bottom_base_ok
        and stabilization
        and volume_bottom_confirmation
        and bottom_repair
        and not severe_decline
        and atr_suitable
        and affordable
        and bottom_score >= 70
    )
    bottom_c_ready = bool(
        bottom_base_ok and stabilization and bottom_score >= 45
    )
    if bottom_b_ready:
        bottom_tier = "B"
    elif bottom_c_ready:
        bottom_tier = "C"
    else:
        bottom_tier = "N"


    # Strategy C: breakout from a completed 20-day range with live volume support.
    daily_change = (
        float(current_change_pct)
        if current_change_pct is not None and pd.notna(current_change_pct)
        else None
    )
    volume_ratio_ok = current_volume_ratio is not None
    breakout_near = bool(
        distance_20d_high is not None and distance_20d_high >= -0.5
    )
    breakout_confirmed = bool(
        distance_20d_high is not None and distance_20d_high >= 0.2
    )
    breakout_score = 0
    breakout_reasons: list[str] = []
    breakout_missing: list[str] = []
    if breakout_confirmed:
        breakout_score += 25
        breakout_reasons.append("现价有效突破此前20日高点")
    elif breakout_near:
        breakout_score += 15
        breakout_reasons.append("现价接近此前20日高点，等待突破确认")
    else:
        breakout_missing.append("尚未接近20日区间上沿")
    if daily_change is not None and 0.5 <= daily_change <= 5.5:
        breakout_score += 15
        breakout_reasons.append("当日涨幅适中，暂未明显追高")
    else:
        breakout_missing.append("当日涨幅不在突破观察区间")
    if current_volume_ratio is not None and 1.2 <= current_volume_ratio <= 3.5:
        breakout_score += 20
        breakout_reasons.append("量比显示成交活跃")
    elif current_volume_ratio is not None and 1.0 <= current_volume_ratio < 1.2:
        breakout_score += 8
        breakout_missing.append("量能略有改善但确认不足")
    else:
        breakout_missing.append("缺少有效量能确认或量比数据")
    if return_5d >= 2:
        breakout_score += 10
    if return_10d >= 4:
        breakout_score += 10
    if liquidity_ok:
        breakout_score += 10
        breakout_reasons.append("近20日平均成交额达到流动性门槛")
    else:
        breakout_missing.append("近20日平均成交额不足")
    if atr_suitable:
        breakout_score += 5
    if affordable:
        breakout_score += 5
    breakout_b = bool(
        breakout_confirmed
        and daily_change is not None and 0.5 <= daily_change <= 5.5
        and current_volume_ratio is not None and 1.2 <= current_volume_ratio <= 3.5
        and liquidity_ok and affordable and breakout_score >= 65
    )
    breakout_c = bool(
        breakout_near
        and daily_change is not None and 0 <= daily_change <= 6
        and current_volume_ratio is not None and current_volume_ratio >= 1.0
        and liquidity_ok and breakout_score >= 40
    )
    breakout_tier = "B" if breakout_b else ("C" if breakout_c else "N")

    # Strategy D: continuation of an already-established short-term move.
    momentum_score = 0
    momentum_reasons: list[str] = []
    momentum_missing: list[str] = []
    if return_5d >= 2:
        momentum_score += 15
        momentum_reasons.append("近5日已有正向动量")
    else:
        momentum_missing.append("近5日动量不足")
    if return_10d >= 4:
        momentum_score += 15
        momentum_reasons.append("近10日上涨波段较明显")
    else:
        momentum_missing.append("近10日上涨波段不足")
    if current_price >= ma5:
        momentum_score += 7
    if current_price >= ma10:
        momentum_score += 7
    if current_price >= ma20:
        momentum_score += 6
        momentum_reasons.append("价格位于MA5/10/20上方")
    else:
        momentum_missing.append("价格未能站稳MA20")
    if daily_change is not None and 0.5 <= daily_change <= 5.5:
        momentum_score += 15
        momentum_reasons.append("当日保持温和走强")
    else:
        momentum_missing.append("当日动量偏弱或涨幅过大")
    if current_volume_ratio is not None and 1.1 <= current_volume_ratio <= 3.5:
        momentum_score += 15
        momentum_reasons.append("量能较近期活跃")
    else:
        momentum_missing.append("量能确认不足")
    if liquidity_ok:
        momentum_score += 10
    if atr_suitable:
        momentum_score += 5
    if affordable:
        momentum_score += 5
    momentum_b = bool(
        return_5d >= 2 and return_10d >= 4
        and daily_change is not None and 0.5 <= daily_change <= 5.5
        and current_volume_ratio is not None and 1.1 <= current_volume_ratio <= 3.5
        and current_price >= ma10 and liquidity_ok and affordable
        and momentum_score >= 65
    )
    momentum_c = bool(
        return_5d > 0 and return_10d > 2
        and daily_change is not None and -0.5 <= daily_change <= 5.5
        and current_price >= ma20 * 0.98 and liquidity_ok
        and momentum_score >= 40
    )
    momentum_tier = "B" if momentum_b else ("C" if momentum_c else "N")

    # Strategy E: next-session strength watch. This is not a prediction that
    # tomorrow will rise; it only ranks liquid, active stocks without chasing.
    next_day_score = 0
    next_day_reasons: list[str] = []
    next_day_missing: list[str] = []
    if daily_change is not None and 1.0 <= daily_change <= 5.5:
        next_day_score += 20
        next_day_reasons.append("当日涨幅适中，具备隔日强势观察价值")
    else:
        next_day_missing.append("当日涨幅不符合隔日动量观察区间")
    if current_volume_ratio is not None and 1.3 <= current_volume_ratio <= 3.5:
        next_day_score += 20
        next_day_reasons.append("量比处于活跃但非极端区间")
    elif current_volume_ratio is not None and 1.1 <= current_volume_ratio < 1.3:
        next_day_score += 10
        next_day_missing.append("量比有所改善但不突出")
    else:
        next_day_missing.append("隔日策略缺少量能确认")
    if return_3d > 0:
        next_day_score += 15
        next_day_reasons.append("近3日价格动量为正")
    else:
        next_day_missing.append("近3日动量尚未转正")
    if return_5d > 0:
        next_day_score += 10
    if current_price >= ma5 * 0.99:
        next_day_score += 10
        next_day_reasons.append("价格守住MA5附近")
    else:
        next_day_missing.append("价格仍低于MA5较多")
    if current_price <= ma5 * 1.06:
        next_day_score += 5
    else:
        next_day_missing.append("价格偏离MA5较远，隔日追高风险增加")
    if liquidity_ok:
        next_day_score += 10
        next_day_reasons.append("近20日平均成交额满足流动性要求")
    else:
        next_day_missing.append("流动性不足")
    if atr_suitable:
        next_day_score += 5
    if affordable:
        next_day_score += 5
    next_day_b = bool(
        daily_change is not None and 1.0 <= daily_change <= 5.5
        and current_volume_ratio is not None and 1.3 <= current_volume_ratio <= 3.5
        and return_3d > 0 and return_5d > 0
        and current_price >= ma5 * 0.99 and current_price <= ma5 * 1.06
        and liquidity_ok and affordable and next_day_score >= 65
    )
    next_day_c = bool(
        daily_change is not None and 0.5 <= daily_change <= 6.0
        and current_volume_ratio is not None and current_volume_ratio >= 1.1
        and return_3d >= 0 and liquidity_ok and next_day_score >= 40
    )
    next_day_tier = "B" if next_day_b else ("C" if next_day_c else "N")

    # Strategy F: short-term oversold bounce, distinct from the slower base
    # reversal model. A deep decline alone never qualifies without a rebound.
    oversold_zone = bool(near_20d_low or near_60d_low)
    oversold_score = 0
    oversold_reasons: list[str] = []
    oversold_missing: list[str] = []
    if oversold_zone:
        oversold_score += 20
        oversold_reasons.append("价格位于20/60日区间低位附近")
    else:
        oversold_missing.append("未处于近期低位区域")
    if prior_decline:
        oversold_score += 10
        oversold_reasons.append("此前存在一段调整")
    if rebound_from_5d_low is not None and rebound_from_5d_low >= 2:
        oversold_score += 20
        oversold_reasons.append("较近5日低点已有明显反弹")
    elif rebound_from_5d_low is not None and rebound_from_5d_low >= 1:
        oversold_score += 10
        oversold_missing.append("反弹幅度仍较有限")
    else:
        oversold_missing.append("尚未从近期低点有效反弹")
    if daily_change is not None and 0.5 <= daily_change <= 5.5:
        oversold_score += 15
        oversold_reasons.append("当日出现温和修复")
    else:
        oversold_missing.append("当日修复力度不足或涨幅过大")
    if current_volume_ratio is not None and 1.2 <= current_volume_ratio <= 3.5:
        oversold_score += 15
        oversold_reasons.append("反弹伴随成交活跃")
    else:
        oversold_missing.append("反弹量能确认不足")
    if base_holds_20d_low:
        oversold_score += 10
    if liquidity_ok:
        oversold_score += 10
    if not severe_decline:
        oversold_score += 5
    if affordable:
        oversold_score += 5
    oversold_b = bool(
        oversold_zone and base_holds_20d_low
        and rebound_from_5d_low is not None and rebound_from_5d_low >= 2
        and daily_change is not None and 0.5 <= daily_change <= 5.5
        and current_volume_ratio is not None and 1.2 <= current_volume_ratio <= 3.5
        and liquidity_ok and affordable and not severe_decline
        and oversold_score >= 65
    )
    oversold_c = bool(
        oversold_zone and rebound_from_5d_low is not None and rebound_from_5d_low >= 1
        and daily_change is not None and 0 <= daily_change <= 6
        and liquidity_ok and oversold_score >= 40
    )
    oversold_tier = "B" if oversold_b else ("C" if oversold_c else "N")

    strategy_tiers = {
        "trend_pullback": trend_tier,
        "bottom_reversal": bottom_tier,
        "breakout_start": breakout_tier,
        "momentum_continuation": momentum_tier,
        "next_day_momentum": next_day_tier,
        "oversold_bounce": oversold_tier,
    }
    strategy_scores = {
        "trend_pullback": int(trend_score),
        "bottom_reversal": int(bottom_score),
        "breakout_start": int(breakout_score),
        "momentum_continuation": int(momentum_score),
        "next_day_momentum": int(next_day_score),
        "oversold_bounce": int(oversold_score),
    }
    strategy_labels = {
        "trend_pullback": "趋势回调低吸",
        "bottom_reversal": "阶段性底部反转",
        "breakout_start": "平台突破启动",
        "momentum_continuation": "强势趋势延续",
        "next_day_momentum": "隔日冲高观察",
        "oversold_bounce": "短线超跌反弹",
    }
    strategy_reason_map = {
        "trend_pullback": (trend_reasons, trend_missing),
        "bottom_reversal": (bottom_reasons, bottom_missing),
        "breakout_start": (breakout_reasons, breakout_missing),
        "momentum_continuation": (momentum_reasons, momentum_missing),
        "next_day_momentum": (next_day_reasons, next_day_missing),
        "oversold_bounce": (oversold_reasons, oversold_missing),
    }
    qualified = [
        name for name, tier in strategy_tiers.items() if tier in ("B", "C")
    ]
    if not qualified:
        candidate_strategy = "none"
        strategy_name = "暂未达到两类策略观察条件"
        daily_tier = "N"
        reasons: list[str] = []
        missing_conditions = (
            trend_missing + bottom_missing + breakout_missing
            + momentum_missing + next_day_missing + oversold_missing
        )
    else:
        best_strategy = max(
            qualified,
            key=lambda name: (strategy_tiers[name] == "B", strategy_scores[name]),
        )
        best_tier = strategy_tiers[best_strategy]
        best_score = strategy_scores[best_strategy]
        daily_tier = "B" if any(
            strategy_tiers[name] == "B" for name in strategy_tiers
        ) else "C"
        candidate_strategy = best_strategy
        strategy_name = (
            strategy_labels[best_strategy]
            if len(qualified) == 1
            else "多策略共振：" + "、".join(strategy_labels[name] for name in qualified)
        )
        reasons, missing_conditions = strategy_reason_map[best_strategy]
        # Keep the aggregate score comparable across the two independent models.
        best_score = max(strategy_scores[name] for name in qualified)

    position_plan = calculate_position_plan(
        current_price=current_price,
        account_capital_cny=account_capital_cny,
        cash_reserve_pct=cash_reserve_pct,
        max_positions=max_positions,
        max_loss_per_trade_pct=max_loss_per_trade_pct,
    )

    return {
        "return_3d_pct": round(return_3d, 2),
        "return_5d_pct": round(return_5d, 2),
        "return_10d_pct": round(return_10d, 2),
        "return_20d_pct": round(return_20d, 2),
        "return_60d_pct": round(return_60d, 2) if return_60d is not None else None,
        "trend_60d_up": trend_60d_up,
        "runup_10d_pct": round(runup_10d, 2),
        "pullback_from_5d_high_pct": round(pullback_pct, 2),
        "pullback_depth_pct": round(pullback_depth, 2),
        "days_since_5d_peak": int(days_since_peak),
        "ma5": round(ma5, 3),
        "ma10": round(ma10, 3),
        "ma20": round(ma20, 3),
        "ma20_slope_5d_pct": round(ma20_slope_pct, 2)
        if ma20_slope_pct is not None else None,
        "ma5_slope_5d_pct": round(ma5_slope_pct, 2)
        if ma5_slope_pct is not None else None,
        "nearest_support": nearest_support,
        "nearest_support_distance_pct": round(nearest_support_distance, 2)
        if nearest_support_distance is not None else None,
        "low_20d": round(low_20d, 3),
        "high_20d": round(high_20d, 3),
        "distance_20d_high_pct": round(distance_20d_high, 2)
        if distance_20d_high is not None else None,
        "current_volume_ratio": round(current_volume_ratio, 3)
        if current_volume_ratio is not None else None,
        "current_turnover_pct": round(current_turnover_pct, 3)
        if current_turnover_pct is not None else None,
        "current_amount_cny": round(current_amount_cny, 2)
        if current_amount_cny is not None else None,
        "low_60d": round(low_60d, 3) if low_60d is not None else None,
        "high_60d": round(high_60d, 3) if high_60d is not None else None,
        "distance_20d_low_pct": round(distance_20d_low, 2)
        if distance_20d_low is not None else None,
        "distance_60d_low_pct": round(distance_60d_low, 2)
        if distance_60d_low is not None else None,
        "drawdown_60d_high_pct": round(drawdown_60d_high, 2)
        if drawdown_60d_high is not None else None,
        "rebound_from_5d_low_pct": round(rebound_from_5d_low, 2)
        if rebound_from_5d_low is not None else None,
        "volume_shrink_ratio_3d_vs_prev5d": round(volume_shrink_ratio, 3)
        if volume_shrink_ratio is not None else None,
        "average_amount_20d_cny": round(average_amount_20d, 2),
        "atr14": round(atr14, 4),
        "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
        "atr_suitable": bool(atr_suitable),
        "one_lot_cost_cny": round(one_lot_cost, 2),
        "single_position_budget_cap_cny": round(single_position_budget_cap_cny, 2),
        "budget_per_position_cny": round(budget_per_position, 2),
        "budget_if_two_positions_cny": round(budget_if_two_positions, 2),
        "affordable": bool(affordable),
        "affordable_if_two_positions": bool(affordable_if_two_positions),
        "bottom_repair": bool(bottom_repair),
        "bottom_severe_decline": bool(severe_decline),
        "trend_up": trend_up,
        "runup_preferred": bool(runup_preferred),
        "runup_acceptable": bool(runup_acceptable),
        "pullback_ok": bool(pullback_ok),
        "volume_shrinking": bool(volume_shrinking),
        "support_near": bool(support_near),
        "liquidity_ok": bool(liquidity_ok),
        "no_chase": bool(no_chase),
        "trend_pullback_score": int(trend_score),
        "bottom_reversal_score": int(bottom_score),
        "trend_pullback_tier": trend_tier,
        "bottom_reversal_tier": bottom_tier,
        "breakout_start_score": int(breakout_score),
        "breakout_start_tier": breakout_tier,
        "momentum_continuation_score": int(momentum_score),
        "momentum_continuation_tier": momentum_tier,
        "next_day_momentum_score": int(next_day_score),
        "next_day_momentum_tier": next_day_tier,
        "oversold_bounce_score": int(oversold_score),
        "oversold_bounce_tier": oversold_tier,
        "next_day_strength_watch": bool(next_day_tier in ("B", "C")),
        "candidate_strategies": [name for name in qualified],
        "trend_pullback_structure_signals": int(trend_structure_signals),
        "bottom_near_20d_low": bool(near_20d_low),
        "bottom_near_60d_low": bool(near_60d_low),
        "bottom_prior_decline": bool(prior_decline),
        "bottom_stabilization": bool(stabilization),
        "bottom_volume_confirmation": bool(volume_bottom_confirmation),
        "candidate_strategy": candidate_strategy,
        "strategy_name": strategy_name,
        "strategy_scores": strategy_scores,
        "strategy_tiers": strategy_tiers,
        "score": int(max(trend_score, bottom_score)),
        "daily_setup_tier": daily_tier,
        "watchlist_eligible": bool(daily_tier == "B"),
        "intraday_confirmation": "not_checked",
        "buyable": False,
        "buy_zone": None,
        "take_profit_zone": None,
        "risk_plan": position_plan,
        "reasons": reasons,
        "missing_conditions": missing_conditions,
        "calculation_basis": (
            "Completed daily bars before current_date are used for all historical metrics; "
            "current_price/current_change_pct are the separately retrieved spot snapshot. "
            "Six independent setups are scored: trend pullback, base reversal, breakout, momentum continuation, next-day strength watch and oversold bounce. "
            "Daily snapshot scores are not forecasts; no minute-bar confirmation is available, so buyable remains false. "
            "ATR(14) is a volatility feature, not a return forecast. "
            "B-tier candidates must pass the 100-share single-position budget check; B tier is not an executable buy signal."
        ),
    }
