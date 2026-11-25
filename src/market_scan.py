from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

from src.indicators import calculate_indicators

SHANGHAI = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
JSON_OUTPUT = DATA_DIR / "market_scan.json"
CSV_OUTPUT = DATA_DIR / "candidates.csv"

SPOT_ALIASES = {
    "code": ("代码", "股票代码", "code", "symbol"),
    "name": ("名称", "股票简称", "name"),
    "price": ("最新价", "最新", "现价", "price", "trade"),
    "change_pct": ("涨跌幅", "涨幅", "changepercent", "change_pct"),
    "amount": ("成交额", "amount", "turnoveramount"),
    "volume": ("成交量", "volume"),
    "turnover_pct": ("换手率", "turnover"),
    "volume_ratio": ("量比", "volume_ratio"),
    "pct_60d": ("60日涨跌幅", "60日涨幅", "pct_60d"),
    "industry": ("行业", "所属行业", "industry"),
}
HISTORY_ALIASES = {
    "date": ("日期", "时间", "date", "day"),
    "open": ("开盘", "open"),
    "high": ("最高", "high"),
    "low": ("最低", "low"),
    "close": ("收盘", "close"),
    "volume": ("成交量", "volume"),
    "amount": ("成交额", "amount"),
}


def now_cn() -> datetime:
    return datetime.now(SHANGHAI)


def safe_number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def is_rate_limit(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    tokens = (
        "push2ratelimit", "rate limit", "too many requests", "429",
        "临时限流", "访问过于频繁", "请求过于频繁", "封禁",
    )
    return any(token in message for token in tokens)


def interface_alternatives(interface_name: str) -> list[str]:
    info_func = getattr(ak, "interface_info", None)
    if not callable(info_func):
        return []
    try:
        info = info_func(interface_name)
        if not isinstance(info, dict):
            return []
        alternatives = info.get("alternatives") or []
        names: list[str] = []
        for item in alternatives:
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                name = item.get("name") or item.get("interface")
            else:
                name = None
            if name and name not in names and name != interface_name:
                names.append(name)
        return names
    except Exception as exc:
        print(f"[WARN] Cannot inspect alternatives for {interface_name}: {exc}")
        return []


def invoke_history_function(func, symbol: str, start_date: str, end_date: str):
    kwargs = {
        "symbol": symbol, "period": "daily",
        "start_date": start_date, "end_date": end_date, "adjust": "qfq",
    }
    try:
        signature = inspect.signature(func)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    except (TypeError, ValueError):
        pass
    return func(**kwargs)


def choose_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def clean_code(value: Any) -> str | None:
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value)).zfill(6)
    text = str(value).strip().upper().replace(".0", "")
    match = re.search(r"(\d{6})$", text)
    if not match:
        match = re.search(r"(\d{6})", text)
    return match.group(1) if match else None


def normalize_spot(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Spot endpoint returned an empty DataFrame.")
    columns = {key: choose_column(frame, aliases) for key, aliases in SPOT_ALIASES.items()}
    required_spot_fields = ("code", "name", "price", "change_pct", "amount")
    missing_spot_fields = [key for key in required_spot_fields if columns[key] is None]
    if missing_spot_fields:
        raise ValueError(
            f"Spot schema is missing required fields {missing_spot_fields}. "
            f"Columns received: {list(frame.columns)}"
        )
    normalized = pd.DataFrame()
    for key, column in columns.items():
        normalized[key] = frame[column] if column is not None else None
    normalized["code"] = normalized["code"].map(clean_code)
    normalized["name"] = normalized["name"].astype(str).str.strip()
    for column in (
        "price", "change_pct", "amount", "volume",
        "turnover_pct", "volume_ratio", "pct_60d",
    ):
        normalized[column] = pd.to_numeric(
            normalized[column].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce",
        )
    normalized = normalized.dropna(subset=["code", "price"])
    normalized = normalized.drop_duplicates("code")
    normalized = normalized.loc[normalized["price"] > 0].reset_index(drop=True)
    return normalized


def fetch_spot_universe() -> tuple[pd.DataFrame, str, list[str]]:
    primary = "stock_zh_a_spot_em"
    candidates = [primary] + interface_alternatives(primary)
    for fallback in ("stock_zh_a_spot_sina", "stock_zh_a_spot"):
        if fallback not in candidates:
            candidates.append(fallback)
    errors: list[str] = []
    for interface_name in candidates:
        func = getattr(ak, interface_name, None)
        if not callable(func):
            continue
        try:
            print(f"[INFO] Fetching full A-share spot universe via {interface_name}")
            frame = normalize_spot(func())
            if len(frame) < 1000:
                raise RuntimeError(
                    f"{interface_name} returned only {len(frame)} normalized rows; "
                    "refusing to treat an incomplete market snapshot as full-market data."
                )
            print(f"[OK] Spot universe: {len(frame)} symbols via {interface_name}")
            return frame, interface_name, errors
        except Exception as exc:
            message = f"{interface_name}: {type(exc).__name__}: {exc}"
            errors.append(message[:500])
            print(f"[WARN] {message}")
            if is_rate_limit(exc):
                print(f"[WARN] Rate limit on {interface_name}; trying an alternate source.")
    raise RuntimeError("No full-market spot source succeeded. " + " | ".join(errors[-4:]))


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Historical K-line endpoint returned an empty DataFrame.")
    columns = {key: choose_column(frame, aliases) for key, aliases in HISTORY_ALIASES.items()}
    missing = [
        key for key in ("date", "high", "low", "close", "volume", "amount")
        if columns[key] is None
    ]
    if missing:
        raise ValueError(
            f"History schema missing fields {missing}; columns received: {list(frame.columns)}"
        )
    result = pd.DataFrame()
    for key, column in columns.items():
        result[key] = frame[column] if column is not None else None
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(
        subset=["date", "high", "low", "close", "volume", "amount"]
    )
    return result.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def market_prefixed_symbol(symbol: str) -> str:
    code = str(symbol).strip().zfill(6)
    if code.startswith(("92", "4", "8")):
        return f"bj{code}"
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


def fetch_history(
    symbol: str,
    start_date: str,
    end_date: str,
    skip_eastmoney: bool = False,
) -> tuple[pd.DataFrame, str, bool]:
    primary = "stock_zh_a_hist_em"
    candidates: list[str] = []
    if not skip_eastmoney:
        candidates.append(primary)
        for alternative in interface_alternatives(primary):
            if alternative not in (
                "stock_zh_a_hist_tx", "stock_zh_a_hist_sina", "stock_zh_a_daily"
            ) and alternative not in candidates:
                candidates.append(alternative)
    for fallback in ("stock_zh_a_hist_tx", "stock_zh_a_hist_sina", "stock_zh_a_daily"):
        if fallback not in candidates:
            candidates.append(fallback)

    errors: list[str] = []
    eastmoney_rate_limited = False
    for interface_name in candidates:
        func = getattr(ak, interface_name, None)
        if not callable(func):
            continue
        try:
            request_symbol = (
                market_prefixed_symbol(symbol)
                if "sina" in interface_name or interface_name.endswith("_daily")
                else symbol
            )
            raw = invoke_history_function(
                func, request_symbol, start_date, end_date
            )
            return normalize_history(raw), interface_name, eastmoney_rate_limited
        except Exception as exc:
            errors.append(f"{interface_name}: {type(exc).__name__}: {exc}")
            print(f"[WARN] {symbol} {errors[-1][:300]}")
            if is_rate_limit(exc) and "em" in interface_name:
                eastmoney_rate_limited = True
                print("[WARN] Eastmoney history source is rate-limited; skip it for the rest of this run.")
    raise RuntimeError("All historical sources failed: " + " | ".join(errors[-3:]))


def normalize_boards(frame: pd.DataFrame, board_type: str) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    name_col = choose_column(frame, ("板块名称", "板块", "名称", "name"))
    pct_col = choose_column(frame, ("涨跌幅", "涨幅", "change_pct"))
    amount_col = choose_column(frame, ("成交额", "总成交额", "amount"))
    adv_col = choose_column(frame, ("上涨家数", "上涨数", "advancers"))
    dec_col = choose_column(frame, ("下跌家数", "下跌数", "decliners"))
    lead_col = choose_column(frame, ("领涨股票", "领涨股", "股票名称", "leader"))
    if name_col is None:
        return []
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "board_type": board_type,
            "name": str(row[name_col]),
            "change_pct": safe_number(row[pct_col]) if pct_col else None,
            "amount_cny": safe_number(row[amount_col]) if amount_col else None,
            "advancers": safe_number(row[adv_col]) if adv_col else None,
            "decliners": safe_number(row[dec_col]) if dec_col else None,
            "leader": str(row[lead_col]) if lead_col and pd.notna(row[lead_col]) else None,
        })
    return rows


def fetch_sina_boards() -> tuple[list[dict[str, Any]], list[str]]:
    all_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    func = getattr(ak, "stock_sector_spot", None)
    if not callable(func):
        return [], ["stock_sector_spot: interface unavailable in installed AKShare"]
    for indicator, board_type in (("行业", "industry"), ("概念", "concept")):
        try:
            frame = func(indicator=indicator)
            rows = normalize_boards(frame, board_type)
            for item in rows:
                item["source"] = "sina"
            all_rows.extend(rows)
            print(f"[OK] Sina {board_type} boards: {len(rows)}")
        except Exception as exc:
            errors.append(
                f"stock_sector_spot({indicator}): {type(exc).__name__}: {exc}"[:500]
            )
            print(f"[WARN] Sina {board_type} boards unavailable: {exc}")
    return all_rows, errors


def fetch_board_data(prefer_sina: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    if prefer_sina:
        print("[INFO] Eastmoney is rate-limited/unavailable; using Sina sector data.")
        return fetch_sina_boards()

    all_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    endpoints = (
        ("stock_board_industry_name_em", "industry"),
        ("stock_board_concept_name_em", "concept"),
    )
    for endpoint, board_type in endpoints:
        func = getattr(ak, endpoint, None)
        if not callable(func):
            errors.append(f"{endpoint}: interface unavailable in installed AKShare")
            continue
        try:
            frame = func()
            rows = normalize_boards(frame, board_type)
            for item in rows:
                item["source"] = "eastmoney"
            all_rows.extend(rows)
            print(f"[OK] Eastmoney {board_type} boards: {len(rows)}")
        except Exception as exc:
            errors.append(f"{endpoint}: {type(exc).__name__}: {exc}"[:500])
            print(f"[WARN] Board data unavailable from {endpoint}: {exc}")
            if is_rate_limit(exc):
                print("[WARN] Eastmoney board endpoint is rate-limited; stop querying this source.")
                break

    have_industry = any(x["board_type"] == "industry" for x in all_rows)
    have_concept = any(x["board_type"] == "concept" for x in all_rows)
    if not have_industry or not have_concept:
        fallback_rows, fallback_errors = fetch_sina_boards()
        missing_types = set()
        if not have_industry:
            missing_types.add("industry")
        if not have_concept:
            missing_types.add("concept")
        all_rows.extend(x for x in fallback_rows if x["board_type"] in missing_types)
        errors.extend(fallback_errors)
    return all_rows, errors


def summarize_market(spot: pd.DataFrame) -> dict[str, Any]:
    pct = pd.to_numeric(spot["change_pct"], errors="coerce")
    amount = pd.to_numeric(spot["amount"], errors="coerce")
    return {
        "listed_symbols_in_snapshot": int(len(spot)),
        "advancers": int((pct > 0).sum()),
        "decliners": int((pct < 0).sum()),
        "unchanged": int((pct == 0).sum()),
        "change_pct_median": round(float(pct.median()), 3) if pct.notna().any() else None,
        "total_turnover_cny": round(float(amount.sum()), 2) if amount.notna().any() else None,
        "stocks_turnover_ge_100m": int((amount >= 100_000_000).sum()),
    }


def build_history_pool(
    spot: pd.DataFrame,
    max_history_stocks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose a bounded history-scan pool without treating today's partial turnover as 20-day liquidity."""
    pool = spot.copy()
    pool["name"] = pool["name"].fillna("")
    excluded_st = pool["name"].str.contains(r"ST|退", case=False, regex=True)
    excluded_price = pool["price"].isna() | (pool["price"] <= 0)
    eligible = pool.loc[~(excluded_st | excluded_price)].copy()

    spot_60d_available = int(eligible["pct_60d"].notna().sum())
    positive_60d = eligible.loc[
        eligible["pct_60d"].notna() & (eligible["pct_60d"] > 0)
    ]
    if spot_60d_available:
        primary = positive_60d.sort_values("amount", ascending=False).head(max_history_stocks)
        selection_mode = "spot_60d_positive_then_spot_amount_rank"
    else:
        # Spot amount is only a ranking hint for the bounded request pool.
        # The actual 20-day average-turnover requirement is checked later
        # from completed daily K-lines in calculate_indicators().
        primary = eligible.sort_values("amount", ascending=False).head(max_history_stocks)
        selection_mode = "spot_amount_rank_only_no_hard_cutoff"

    if len(primary) < max_history_stocks:
        remaining = eligible.loc[~eligible["code"].isin(primary["code"])]
        fill = remaining.sort_values("amount", ascending=False).head(
            max_history_stocks - len(primary)
        )
        primary = pd.concat([primary, fill], ignore_index=True)

    counts = {
        "spot_rows": int(len(spot)),
        "excluded_st_or_delisting": int(excluded_st.sum()),
        "excluded_invalid_price": int(excluded_price.sum()),
        "eligible_for_history_scan": int(len(eligible)),
        "spot_60d_available_count": spot_60d_available,
        "positive_60d_in_spot": int(len(positive_60d)) if spot_60d_available else None,
        "history_pool_selection_mode": selection_mode,
        "history_pool_size": int(len(primary)),
    }
    return primary.reset_index(drop=True), counts


def scan(
    max_history_stocks: int = 120,
    min_amount_cny: float = 100_000_000,
    delay_seconds: float = 1.0,
    output_candidates: int = 50,
) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    collected_at = now_cn()
    today = collected_at.date()
    start_date = (today - timedelta(days=150)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    errors: list[str] = []
    spot, spot_source, spot_errors = fetch_spot_universe()
    errors.extend(spot_errors)
    eastmoney_blocked = any("Push2RateLimitError" in item for item in spot_errors)
    boards, board_errors = fetch_board_data(
        prefer_sina=eastmoney_blocked or spot_source == "stock_zh_a_spot_sina"
    )
    errors.extend(board_errors)

    market_summary = summarize_market(spot)
    pool, pool_counts = build_history_pool(
        spot, max_history_stocks=max_history_stocks
    )

    candidates: list[dict[str, Any]] = []
    history_failures = 0
    rate_limited = False

    for index, row in pool.iterrows():
        symbol = str(row["code"])
        if index > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        print(f"[INFO] History scan {index + 1}/{len(pool)}: {symbol} {row['name']}")
        try:
            history, history_source, history_rate_limited = fetch_history(
                symbol, start_date, end_date, skip_eastmoney=eastmoney_blocked
            )
            if history_rate_limited:
                eastmoney_blocked = True
                rate_limited = True
            history = history.loc[history["date"].dt.date < today].reset_index(drop=True)
            metrics = calculate_indicators(
                history=history,
                current_price=float(row["price"]),
                current_change_pct=safe_number(row["change_pct"]),
                current_date=today,
                average_amount_floor=min_amount_cny,
            )
            candidates.append({
                "code": symbol,
                "name": str(row["name"]),
                "industry": str(row["industry"]) if pd.notna(row["industry"]) else None,
                "latest_price": safe_number(row["price"]),
                "daily_change_pct": safe_number(row["change_pct"]),
                "spot_amount_cny": safe_number(row["amount"]),
                "spot_turnover_pct": safe_number(row["turnover_pct"]),
                "spot_volume_ratio": safe_number(row["volume_ratio"]),
                "pct_60d_from_spot": safe_number(row["pct_60d"]),
                "spot_source": spot_source,
                "spot_retrieved_at": collected_at.isoformat(timespec="seconds"),
                "exchange_quote_timestamp": None,
                "history_source": history_source,
                "history_latest_completed_bar": (
                    history["date"].iloc[-1].date().isoformat()
                    if not history.empty else None
                ),
                **metrics,
            })
        except Exception as exc:
            history_failures += 1
            message = f"{symbol}: {type(exc).__name__}: {exc}"
            errors.append(message[:600])
            print(f"[WARN] Skipping {symbol}: {message}")
            if is_rate_limit(exc):
                rate_limited = True
                print("[WARN] Rate limit detected. Stop further history requests to protect the source.")
                break

    tier_rank = {"B": 2, "C": 1, "N": 0}
    candidates.sort(
        key=lambda item: (
            tier_rank.get(item.get("daily_setup_tier"), 0),
            item.get("affordable", False),
            item.get("atr_suitable", False),
            item.get("score", 0),
            max(
                item.get("trend_pullback_score", 0),
                item.get("bottom_reversal_score", 0),
            ),
            item.get("liquidity_ok", False),
            item.get("average_amount_20d_cny") or 0,
        ),
        reverse=True,
    )
    board_industry = [
        x for x in boards
        if x["board_type"] == "industry" and x["change_pct"] is not None
    ]
    board_concept = [
        x for x in boards
        if x["board_type"] == "concept" and x["change_pct"] is not None
    ]
    board_industry.sort(key=lambda x: x["change_pct"], reverse=True)
    board_concept.sort(key=lambda x: x["change_pct"], reverse=True)

    success_count = len(candidates)
    industry_board_count = len(board_industry)
    concept_board_count = len(board_concept)
    status = (
        "ok"
        if success_count >= 20 and industry_board_count > 0 and concept_board_count > 0
        else "partial"
    )
    result = {
        "project": "a-share-quant",
        "schema_version": 1,
        "status": status,
        "collected_at": collected_at.isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "quote_timestamp_note": (
            "AKShare spot endpoints do not guarantee a per-security exchange timestamp. "
            "spot_retrieved_at is collection time only."
        ),
        "market_data": {
            "spot_source": spot_source,
            "spot_quote_timestamp": None,
            "summary": market_summary,
            "industry_board_count": industry_board_count,
            "concept_board_count": concept_board_count,
            "industry_board_source": (
                board_industry[0].get("source") if board_industry else None
            ),
            "concept_board_source": (
                board_concept[0].get("source") if board_concept else None
            ),
            "industry_boards_top10": board_industry[:10],
            "industry_boards_bottom10": list(reversed(board_industry[-10:])),
            "concept_boards_top10": board_concept[:10],
            "concept_boards_bottom10": list(reversed(board_concept[-10:])),
        },
        "scan_parameters": {
            "max_history_stocks": max_history_stocks,
            "min_amount_cny": min_amount_cny,
            "history_request_delay_seconds": delay_seconds,
            "daily_bars_adjustment": "qfq",
            "strategy_mode": "parallel_trend_pullback_and_bottom_reversal",
            "atr14_suitability_pct_range": [1.2, 6.5],
            "candidate_output_limit": output_candidates,
        },
        "risk_management": {
            "account_capital_cny": 10000,
            "cash_reserve_pct": 30,
            "max_positions": 2,
            "max_position_budget_cny": 3500,
            "max_loss_per_trade_pct": 5,
            "take_profit_observation_range_pct": [3, 5],
            "max_holding_trading_days": 10,
            "a_share_lot_size": 100,
            "risk_note": (
                "5% is the planned maximum loss relative to each position's entry value, "
                "not a guaranteed execution price or 5% of total account capital. "
                "Gap risk, slippage and fees can increase actual losses."
            ),
        },
        "coverage": {
            **pool_counts,
            "history_requested": int(len(pool)),
            "history_success": success_count,
            "history_failed": int(history_failures),
            "history_rate_limited": bool(rate_limited),
            "eastmoney_rate_limited": bool(eastmoney_blocked),
            "history_coverage_pct": round(
                success_count / len(pool) * 100, 2
            ) if len(pool) else 0,
            "candidate_count": int(len(candidates)),
            "history_positive_60d": int(
                sum(1 for item in candidates if (item.get("return_60d_pct") or 0) > 0)
            ),
            "trend_pullback_b_count": int(
                sum(1 for item in candidates if item.get("trend_pullback_tier") == "B")
            ),
            "trend_pullback_c_count": int(
                sum(1 for item in candidates if item.get("trend_pullback_tier") == "C")
            ),
            "bottom_reversal_b_count": int(
                sum(1 for item in candidates if item.get("bottom_reversal_tier") == "B")
            ),
            "bottom_reversal_c_count": int(
                sum(1 for item in candidates if item.get("bottom_reversal_tier") == "C")
            ),
            "daily_setup_b_count": int(
                sum(1 for item in candidates if item.get("daily_setup_tier") == "B")
            ),
            "daily_setup_c_count": int(
                sum(1 for item in candidates if item.get("daily_setup_tier") == "C")
            ),
            "daily_setup_n_count": int(
                sum(1 for item in candidates if item.get("daily_setup_tier") == "N")
            ),
            "affordable_candidate_count": int(
                sum(1 for item in candidates if item.get("affordable", False))
            ),
            "atr_suitable_candidate_count": int(
                sum(1 for item in candidates if item.get("atr_suitable", False))
            ),
            "bottom_severe_decline_count": int(
                sum(1 for item in candidates if item.get("bottom_severe_decline", False))
            ),
        },
        "candidates": candidates[:output_candidates],
        "errors": errors[-30:],
        "limitations": [
            "The full-market spot snapshot is scanned, but detailed daily K-lines are fetched only for the liquidity/trend-prioritized history pool.",
            "Trend-pullback and bottom-reversal strategies are scored independently; a negative 60-day return is not a hard veto.",
            "Detailed daily K-lines are calculated for the top-ranked 120-stock history pool by default, not every stock in the 5,000+ spot universe.",
            "No 14:30 minute-bar confirmation is included yet; all candidates have buyable=false.",
            "No precise buy zone or take-profit price is emitted until minute-level confirmation and support validation are implemented.",
            "ATR(14) is used only as a short-term volatility suitability/ranking feature; it is not a prediction of future returns.",
            "B-tier candidates must pass the estimated 100-share affordability check under the configured per-position budget; C-tier candidates may remain unaffordable watch-only names.",
            "The spot endpoint retrieval time is not a guaranteed exchange quote timestamp.",
            "When Eastmoney is rate-limited, Sina sector data is used as a fallback; sector classifications and names may differ by provider.",
            "Sector board rankings are market context and are not treated as proof that an individual stock belongs to a strong trend setup.",
        ],
    }

    JSON_OUTPUT.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if candidates:
        pd.DataFrame(candidates).to_csv(
            CSV_OUTPUT, index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(
            columns=["code", "name", "score", "daily_setup_tier"]
        ).to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    print("[SUMMARY] " + json.dumps(result["coverage"], ensure_ascii=False))
    print(f"[OK] JSON: {JSON_OUTPUT}")
    print(f"[OK] CSV: {CSV_OUTPUT}")
    print(f"[SUMMARY] Scan status: {status}; candidates: {len(candidates)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-market A-share snapshot with parallel trend-pullback and bottom-reversal scans."
    )
    parser.add_argument("--max-history-stocks", type=int, default=120)
    parser.add_argument("--min-amount-cny", type=float, default=100_000_000)
    parser.add_argument("--delay-seconds", type=float, default=1.2)
    parser.add_argument("--output-candidates", type=int, default=50)
    args = parser.parse_args()

    if not 10 <= args.max_history_stocks <= 300:
        parser.error("--max-history-stocks must be between 10 and 300")
    if args.min_amount_cny < 0:
        parser.error("--min-amount-cny cannot be negative")
    if not 0 <= args.delay_seconds <= 10:
        parser.error("--delay-seconds must be between 0 and 10")
    if not 3 <= args.output_candidates <= 200:
        parser.error("--output-candidates must be between 3 and 200")

    try:
        result = scan(
            max_history_stocks=args.max_history_stocks,
            min_amount_cny=args.min_amount_cny,
            delay_seconds=args.delay_seconds,
            output_candidates=args.output_candidates,
        )
        return 0 if result["coverage"]["history_success"] > 0 else 1
    except Exception as exc:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "project": "a-share-quant",
            "status": "error",
            "collected_at": now_cn().isoformat(timespec="seconds"),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        JSON_OUTPUT.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
