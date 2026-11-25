from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def calculate_position_plan(
    current_price: float,
    account_capital_cny: float = 10_000.0,
    cash_reserve_pct: float = 30.0,
    max_positions: int = 2,
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
    if not 0 <= cash_reserve_pct < 100:
        raise ValueError("cash_reserve_pct must be in [0, 100)")
    if not 0 < max_loss_per_trade_pct <= 100:
        raise ValueError("max_loss_per_trade_pct must be in (0, 100]")

    investable_capital = account_capital_cny * (1 - cash_reserve_pct / 100)
    per_position_budget = investable_capital / max_positions
    one_lot_cost = current_price * lot_size
    lots = int((per_position_budget + 1e-9) / one_lot_cost)
    shares = lots * lot_size
    position_value = shares * current_price

    return {
        "account_capital_cny": round(account_capital_cny, 2),
        "cash_reserve_pct": round(cash_reserve_pct, 2),
        "max_positions": int(max_positions),
        "budget_per_position_cny": round(per_position_budget, 2),
        "lot_size_shares": int(lot_size),
        "one_lot_cost_cny": round(one_lot_cost, 2),
        "affordable_shares": int(shares),
        "estimated_position_value_cny": round(position_value, 2),
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
    average_amount_floor: float = 100_000_000,
    account_capital_cny: float = 10_000.0,
    cash_reserve_pct: float = 30.0,
    max_positions: int = 2,
    max_loss_per_trade_pct: float = 5.0,
) -> dict[str, Any]:
    """Score trend-pullback and bottom-reversal setups independently.

    All moving-average, price-range and volume indicators use completed daily bars.
    The current spot quote is used only as the latest price/change snapshot. Neither
    strategy can create a buyable signal until intraday confirmation is implemented.
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
    no_chase = current_change_pct is None or current_change_pct <= 4.5
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
    if (
        trend_base_ok and runup_acceptable
        and trend_score >= 55 and trend_structure_signals >= 1
    ):
        trend_tier = "B"
    elif trend_base_ok and trend_score >= 35:
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
    if daily_risk_ok:
        bottom_score += 5
    else:
        bottom_missing.append("当日涨跌幅过大，暂不参与")

    bottom_base_ok = (
        prior_decline and near_bottom_zone and base_holds_20d_low
        and liquidity_ok and daily_risk_ok
    )
    if bottom_base_ok and stabilization and bottom_score >= 65:
        bottom_tier = "B"
    elif bottom_base_ok and bottom_score >= 45:
        bottom_tier = "C"
    else:
        bottom_tier = "N"

    strategy_tiers = {
        "trend_pullback": trend_tier,
        "bottom_reversal": bottom_tier,
    }
    strategy_scores = {
        "trend_pullback": int(trend_score),
        "bottom_reversal": int(bottom_score),
    }
    qualified = [
        name for name, tier in strategy_tiers.items() if tier in ("B", "C")
    ]
    if not qualified:
        candidate_strategy = "none"
        strategy_name = "暂未达到两类策略观察条件"
        daily_tier = "N"
        reasons: list[str] = []
        missing_conditions = trend_missing + bottom_missing
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
        candidate_strategy = (
            "both"
            if all(strategy_tiers[name] in ("B", "C") for name in strategy_tiers)
            else best_strategy
        )
        strategy_name = (
            "趋势回调低吸" if best_strategy == "trend_pullback"
            else "阶段性底部反弹"
        )
        reasons = trend_reasons if best_strategy == "trend_pullback" else bottom_reasons
        missing_conditions = (
            trend_missing if best_strategy == "trend_pullback" else bottom_missing
        )
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
            "Trend-pullback and bottom-reversal scores are independent."
        ),
    }
