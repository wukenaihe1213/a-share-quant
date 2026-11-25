from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def calculate_indicators(
    history: pd.DataFrame,
    current_price: float,
    current_change_pct: float | None,
    current_date: date,
    average_amount_floor: float = 100_000_000,
) -> dict[str, Any]:
    """Calculate trend/pullback metrics from completed daily bars only."""
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
    volume = frame["volume"]
    amount = frame["amount"]

    ma5 = float(close.tail(5).mean())
    ma10 = float(close.tail(10).mean())
    ma20 = float(close.tail(20).mean())
    ma20_previous = float(close.iloc[-25:-5].mean())

    return_5d = (current_price / float(close.iloc[-6]) - 1) * 100
    return_20d = (current_price / float(close.iloc[-21]) - 1) * 100
    return_60d = (
        (current_price / float(close.iloc[-61]) - 1) * 100
        if len(close) >= 61 else None
    )
    runup_10d = (float(high.tail(10).max()) / float(close.iloc[-11]) - 1) * 100

    five_bar_high_series = high.tail(5).reset_index(drop=True)
    five_bar_high = float(five_bar_high_series.max())
    peak_position = int(five_bar_high_series.idxmax())
    days_since_peak = 5 - peak_position
    pullback_pct = (current_price / five_bar_high - 1) * 100
    pullback_depth = max(0.0, -pullback_pct)

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
        threshold = 4.0 if nearest_support == "MA20" else 2.5
        support_value = ma_values[nearest_support]
        support_near = (
            nearest_support_distance is not None
            and nearest_support_distance <= threshold
            and current_price >= support_value * 0.98
        )

    ma20_slope_pct = (
        (ma20 / ma20_previous - 1) * 100 if ma20_previous > 0 else None
    )
    trend_up = (
        return_60d is not None
        and return_60d > 0
        and return_20d > 0
        and ma20_slope_pct is not None
        and ma20_slope_pct > 0
        and current_price >= ma20 * 0.98
    )
    runup_preferred = 8 <= runup_10d <= 20
    pullback_ok = 2 <= pullback_depth <= 8 and 2 <= days_since_peak <= 5
    volume_shrinking = (
        volume_shrink_ratio is not None and volume_shrink_ratio <= 0.85
    )
    liquidity_ok = average_amount_20d >= average_amount_floor
    no_chase = current_change_pct is None or current_change_pct <= 4.5
    not_crashing = current_change_pct is None or current_change_pct >= -5.0

    score = 0
    reasons: list[str] = []
    missing_conditions: list[str] = []

    if trend_up:
        score += 20
        reasons.append("60日、20日趋势向上且MA20斜率为正")
    else:
        missing_conditions.append("60日/20日趋势或MA20斜率未确认")

    if runup_preferred:
        score += 15
        reasons.append("近10日上涨波段处于优先区间")
    elif 5 <= runup_10d <= 25:
        score += 8
        reasons.append("此前存在一定上涨波段，但强度一般")
    else:
        missing_conditions.append("前期上涨波段不足或过热")

    if pullback_ok:
        score += 20
        reasons.append("近2至5个交易日回撤幅度符合观察范围")
    elif 1 <= pullback_depth <= 10:
        score += 8
        missing_conditions.append("回撤幅度或回撤持续时间不完全符合要求")
    else:
        missing_conditions.append("未形成合适的2至5日回调")

    if volume_shrinking:
        score += 15
        reasons.append("近3日均量较此前5日萎缩")
    elif volume_shrink_ratio is not None and volume_shrink_ratio <= 1.0:
        score += 8
        missing_conditions.append("成交量有所收缩，但不够明显")
    else:
        missing_conditions.append("尚未观察到成交量萎缩")

    if support_near:
        score += 15
        reasons.append(f"价格接近{nearest_support}且未明显跌破")
    else:
        missing_conditions.append("尚未接近MA5/MA10/MA20有效支撑")

    if liquidity_ok:
        score += 10
        reasons.append("近20日平均成交额达到1亿元")
    else:
        missing_conditions.append("近20日平均成交额不足1亿元或数据缺失")

    if no_chase and not_crashing:
        score += 5
        reasons.append("当日涨跌幅未触发追高或急跌过滤")
    elif not no_chase:
        missing_conditions.append("当日涨幅偏大，不作为买入候选")
    else:
        missing_conditions.append("当日跌幅过大，暂不参与")

    base_setup = trend_up and liquidity_ok and no_chase and not_crashing
    runup_acceptable = 5 <= runup_10d <= 25
    partial_setup_components = sum((
        runup_acceptable, pullback_ok, volume_shrinking, support_near
    ))
    core_setup = (
        base_setup and pullback_ok and volume_shrinking and support_near
    )

    # B means the medium-term trend/liquidity gates pass and most daily
    # structure signals are present. Minute-bar confirmation is still absent,
    # so this phase must never label a candidate A or buyable.
    tier = (
        "B"
        if base_setup and score >= 60 and partial_setup_components >= 2
        else "C"
    )

    return {
        "return_5d_pct": round(return_5d, 2),
        "return_20d_pct": round(return_20d, 2),
        "return_60d_pct": round(return_60d, 2) if return_60d is not None else None,
        "trend_60d_up": bool(return_60d is not None and return_60d > 0),
        "runup_10d_pct": round(runup_10d, 2),
        "pullback_from_5d_high_pct": round(pullback_pct, 2),
        "pullback_depth_pct": round(pullback_depth, 2),
        "days_since_5d_peak": int(days_since_peak),
        "ma5": round(ma5, 3),
        "ma10": round(ma10, 3),
        "ma20": round(ma20, 3),
        "ma20_slope_5d_pct": round(ma20_slope_pct, 2)
        if ma20_slope_pct is not None else None,
        "nearest_support": nearest_support,
        "nearest_support_distance_pct": round(nearest_support_distance, 2)
        if nearest_support_distance is not None else None,
        "volume_shrink_ratio_3d_vs_prev5d": round(volume_shrink_ratio, 3)
        if volume_shrink_ratio is not None else None,
        "average_amount_20d_cny": round(average_amount_20d, 2),
        "trend_up": bool(trend_up),
        "runup_preferred": bool(runup_preferred),
        "runup_acceptable": bool(runup_acceptable),
        "partial_setup_components": int(partial_setup_components),
        "pullback_ok": bool(pullback_ok),
        "volume_shrinking": bool(volume_shrinking),
        "support_near": bool(support_near),
        "liquidity_ok": bool(liquidity_ok),
        "no_chase": bool(no_chase),
        "score": int(score),
        "daily_setup_tier": tier,
        "watchlist_eligible": bool(tier == "B"),
        "intraday_confirmation": "not_checked",
        "buyable": False,
        "buy_zone": None,
        "take_profit_zone": None,
        "reasons": reasons,
        "missing_conditions": missing_conditions,
        "calculation_basis": (
            "MA and volume metrics use completed daily bars before current_date; "
            "current price/change use the separately retrieved spot snapshot."
        ),
    }
