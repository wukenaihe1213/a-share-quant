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
from src.news_sentiment import collect_news
from src.tencent_quote import (
    fetch_baostock_history,
    fetch_baostock_universe,
    fetch_tencent_quotes,
    market_code_to_tencent,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
JSON_OUTPUT = DATA_DIR / "market_scan.json"
CSV_OUTPUT = DATA_DIR / "candidates.csv"
NEWS_OUTPUT = DATA_DIR / "news_sentiment.json"

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
    "quote_timestamp": ("行情时间", "报价时间", "更新时间", "quote_timestamp", "timestamp"),
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


def merge_tencent_quote_fields(
    spot: pd.DataFrame,
    tencent_quotes: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing spot metrics from Tencent without replacing primary-source prices."""
    enriched = spot.copy()
    for column in ("volume_ratio", "turnover_pct", "volume", "amount"):
        if column not in enriched.columns:
            enriched[column] = float("nan")
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    quote_frame = pd.DataFrame(tencent_quotes)
    if quote_frame.empty or "code" not in quote_frame.columns:
        for field in ("volume_ratio", "turnover_pct"):
            source_col = f"{field}_source"
            enriched[source_col] = enriched[field].notna().map(
                lambda available: "akshare_primary" if available else None
            )
        enriched["tencent_quote_timestamp"] = None
        enriched["tencent_quote_available"] = False
        return enriched, {
            "status": "unavailable",
            "source": "tencent_qt_gtimg",
            "requested_symbols": int(len(enriched)),
            "returned_symbols": 0,
            "coverage_pct": 0.0,
            "volume_ratio_filled": 0,
            "turnover_pct_filled": 0,
            "volume_ratio_available_after": int(enriched["volume_ratio"].notna().sum()),
            "turnover_pct_available_after": int(enriched["turnover_pct"].notna().sum()),
        }

    quote_frame["code"] = quote_frame["code"].map(clean_code)
    quote_frame = quote_frame.dropna(subset=["code"]).drop_duplicates("code")
    quote_map = quote_frame.set_index("code").to_dict("index")
    spot_codes = enriched["code"].astype(str)
    matched = spot_codes.isin(quote_map)
    enriched["tencent_quote_available"] = matched
    enriched["tencent_quote_timestamp"] = spot_codes.map(
        lambda code: quote_map.get(code, {}).get("quote_timestamp")
    )
    for field in ("price", "change_pct", "amount", "volume", "volume_ratio", "turnover_pct"):
        enriched[f"tencent_{'amount_cny' if field == 'amount' else field}"] = spot_codes.map(
            lambda code: quote_map.get(code, {}).get(field)
        )

    filled_counts: dict[str, int] = {}
    for field in ("volume_ratio", "turnover_pct", "volume", "amount"):
        primary_values = pd.to_numeric(enriched[field], errors="coerce")
        fallback_values = pd.to_numeric(
            enriched[f"tencent_{'amount_cny' if field == 'amount' else field}"],
            errors="coerce",
        )
        fill_mask = primary_values.isna() & fallback_values.notna()
        enriched[field] = primary_values.where(~fill_mask, fallback_values)
        source_col = f"{field}_source"
        enriched[source_col] = None
        enriched.loc[primary_values.notna(), source_col] = "akshare_primary"
        enriched.loc[fill_mask, source_col] = "tencent_qt_gtimg"
        filled_counts[field] = int(fill_mask.sum())

    returned_count = int(matched.sum())
    stats = {
        "status": "ok" if returned_count else "unavailable",
        "source": "tencent_qt_gtimg",
        "requested_symbols": int(len(enriched)),
        "returned_symbols": returned_count,
        "coverage_pct": round(returned_count / max(len(enriched), 1) * 100, 2),
        "volume_ratio_filled": filled_counts["volume_ratio"],
        "turnover_pct_filled": filled_counts["turnover_pct"],
        "volume_filled": filled_counts["volume"],
        "amount_filled": filled_counts["amount"],
        "volume_ratio_available_after": int(enriched["volume_ratio"].notna().sum()),
        "turnover_pct_available_after": int(enriched["turnover_pct"].notna().sum()),
        "quote_timestamp_available": int(enriched["tencent_quote_timestamp"].notna().sum()),
    }
    if returned_count == 0:
        stats["status"] = "unavailable"
    elif returned_count < len(enriched):
        stats["status"] = "partial"
    return enriched, stats


def fetch_tencent_spot_universe() -> tuple[pd.DataFrame, str, list[str]]:
    """Independent live-quote fallback: BaoStock universe + Tencent quote batches."""
    universe = fetch_baostock_universe()
    symbols = [market_code_to_tencent(code) for code in universe["code"].tolist()]
    quotes, quote_errors = fetch_tencent_quotes(symbols, batch_size=80)
    frame = pd.DataFrame(quotes)
    if frame.empty:
        raise RuntimeError("Tencent returned an empty A-share quote snapshot.")

    names = dict(zip(universe["code"].astype(str), universe["name"].astype(str)))
    frame["name"] = frame.apply(
        lambda row: row["name"] if str(row.get("name", "")).strip()
        else names.get(str(row["code"]), ""),
        axis=1,
    )
    frame["industry"] = None
    frame["pct_60d"] = float("nan")
    frame["tencent_quote_timestamp"] = frame.get("quote_timestamp")
    frame["tencent_quote_available"] = True
    for field in ("volume_ratio", "turnover_pct", "volume", "amount"):
        frame[f"{field}_source"] = "tencent_qt_gtimg"
    coverage = len(frame) / max(len(universe), 1)
    if len(frame) < 1000 or coverage < 0.60:
        raise RuntimeError(
            f"Tencent quote coverage is incomplete: {len(frame)}/{len(universe)} "
            f"({coverage:.1%}); refusing to scan a partial market snapshot."
        )

    warnings = [
        f"Tencent quote coverage: {len(frame)}/{len(universe)} ({coverage:.1%})"
    ]
    warnings.extend(quote_errors[:10])
    print(
        f"[OK] Spot universe: {len(frame)} symbols via Tencent "
        f"(BaoStock universe coverage {coverage:.1%})"
    )
    return frame.reset_index(drop=True), "tencent_qt_gtimg_baostock_universe", warnings


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
    try:
        print("[INFO] All AKShare spot sources failed; switching to Tencent live quotes + BaoStock universe.")
        frame, source, fallback_errors = fetch_tencent_spot_universe()
        errors.extend(fallback_errors)
        return frame, source, errors
    except Exception as exc:
        message = f"Tencent/BaoStock fallback: {type(exc).__name__}: {exc}"
        errors.append(message[:600])
        print(f"[FATAL] {message}")
    raise RuntimeError("No full-market spot source succeeded. " + " | ".join(errors[-6:]))


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
    try:
        raw = fetch_baostock_history(symbol, start_date, end_date)
        return normalize_history(raw), "baostock_query_history_k_data_plus", eastmoney_rate_limited
    except Exception as exc:
        errors.append(f"baostock: {type(exc).__name__}: {exc}")
        print(f"[WARN] {symbol} BaoStock history fallback failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("All historical sources failed: " + " | ".join(errors[-4:]))


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
    """Build a diversified multi-signal history pool instead of only taking the biggest stocks."""
    pool = spot.copy()
    pool["name"] = pool["name"].fillna("")
    for column in ("amount", "pct_60d", "change_pct", "volume_ratio", "turnover_pct"):
        if column not in pool.columns:
            pool[column] = float("nan")
        pool[column] = pd.to_numeric(pool[column], errors="coerce")

    excluded_st = pool["name"].str.contains(r"ST|退", case=False, regex=True)
    excluded_price = pool["price"].isna() | (pool["price"] <= 0)
    eligible = pool.loc[~(excluded_st | excluded_price)].copy()

    spot_60d_available = int(eligible["pct_60d"].notna().sum())
    positive_60d = eligible.loc[
        eligible["pct_60d"].notna() & (eligible["pct_60d"] > 0)
    ]

    buckets: dict[str, pd.DataFrame] = {
        "liquidity": eligible.sort_values("amount", ascending=False),
        "trend": (
            positive_60d.sort_values(["pct_60d", "amount"], ascending=[False, False])
            if not positive_60d.empty
            else eligible.loc[eligible["change_pct"] > 0].sort_values(
                ["change_pct", "amount"], ascending=[False, False]
            )
        ),
        "momentum": eligible.loc[
            eligible["change_pct"].between(0.5, 7.0)
            & (eligible["volume_ratio"] >= 1.1)
        ].sort_values(["volume_ratio", "amount"], ascending=[False, False]),
        "rebound": eligible.loc[
            (eligible["pct_60d"].isna() | (eligible["pct_60d"] <= 0))
            & eligible["change_pct"].between(0.5, 6.0)
            & (eligible["volume_ratio"] >= 1.0)
        ].sort_values(["volume_ratio", "amount"], ascending=[False, False]),
        "active": eligible.loc[
            (eligible["turnover_pct"] >= 1.0)
            & eligible["change_pct"].between(-2.0, 6.0)
        ].sort_values(["turnover_pct", "amount"], ascending=[False, False]),
    }
    weights = {
        "liquidity": 0.30,
        "trend": 0.25,
        "momentum": 0.20,
        "rebound": 0.15,
        "active": 0.10,
    }

    def board_group(code: Any) -> str:
        value = str(code).zfill(6)
        if value.startswith(("300", "301", "302", "303")):
            return "chinext"
        if value.startswith(("688", "689")):
            return "star"
        if value.startswith(("4", "8", "92")):
            return "beijing"
        if value.startswith("6"):
            return "shanghai"
        return "shenzhen"

    def interleave_boards(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        work = frame.copy()
        work["_board_group"] = work["code"].map(board_group)
        groups = {
            key: group.drop(columns=["_board_group"]).to_dict("records")
            for key, group in work.groupby("_board_group", sort=True)
        }
        ordered: list[dict[str, Any]] = []
        while any(groups.values()):
            for key in sorted(groups):
                if groups[key]:
                    ordered.append(groups[key].pop(0))
        return ordered

    selected_codes: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    bucket_selected: dict[str, int] = {}
    for bucket_name, weight in weights.items():
        quota = max(1, int(round(max_history_stocks * weight)))
        added = 0
        for row in interleave_boards(buckets[bucket_name]):
            code = str(row["code"])
            if code in selected_codes:
                continue
            selected_codes.add(code)
            selected_rows.append(row)
            added += 1
            if added >= quota or len(selected_rows) >= max_history_stocks:
                break
        bucket_selected[bucket_name] = added
        if len(selected_rows) >= max_history_stocks:
            break

    # Fill unused quota or empty signal buckets with the most liquid remaining names.
    fill_count = 0
    for row in interleave_boards(buckets["liquidity"]):
        if len(selected_rows) >= max_history_stocks:
            break
        code = str(row["code"])
        if code in selected_codes:
            continue
        selected_codes.add(code)
        selected_rows.append(row)
        fill_count += 1

    primary = pd.DataFrame(selected_rows).head(max_history_stocks).reset_index(drop=True)
    counts = {
        "spot_rows": int(len(spot)),
        "excluded_st_or_delisting": int(excluded_st.sum()),
        "excluded_invalid_price": int(excluded_price.sum()),
        "eligible_for_history_scan": int(len(eligible)),
        "spot_60d_available_count": spot_60d_available,
        "positive_60d_in_spot": int(len(positive_60d)) if spot_60d_available else None,
        "history_pool_selection_mode": "multi_strategy_bucket_union_with_board_interleave",
        "history_pool_size": int(len(primary)),
        "history_pool_bucket_selected_counts": bucket_selected,
        "history_pool_liquidity_fill_count": int(fill_count),
    }
    return primary, counts


def scan(
    max_history_stocks: int = 300,
    min_amount_cny: float = 50_000_000,
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

    tencent_enrichment: dict[str, Any] = {
        "status": "skipped",
        "reason": "Primary snapshot already contains volume ratio and turnover rate."
        if (
            spot["volume_ratio"].notna().all()
            and spot["turnover_pct"].notna().all()
        )
        else "Primary snapshot already comes from Tencent."
        if spot_source.startswith("tencent_")
        else "Tencent supplement has not run.",
        "source": "tencent_qt_gtimg",
        "requested_symbols": int(len(spot)),
        "returned_symbols": 0,
        "coverage_pct": None,
        "volume_ratio_filled": 0,
        "turnover_pct_filled": 0,
    }
    missing_ratio = int(spot["volume_ratio"].isna().sum())
    missing_turnover = int(spot["turnover_pct"].isna().sum())
    if not spot_source.startswith("tencent_") and (missing_ratio or missing_turnover):
        try:
            print(
                "[INFO] Supplementing missing volume-ratio/turnover fields "
                f"from Tencent for {len(spot)} A-share symbols."
            )
            symbols = [market_code_to_tencent(code) for code in spot["code"].tolist()]
            tencent_quotes, tencent_errors = fetch_tencent_quotes(
                symbols, batch_size=80, pause_seconds=0.15
            )
            spot, tencent_enrichment = merge_tencent_quote_fields(
                spot, tencent_quotes
            )
            tencent_enrichment["batch_errors"] = tencent_errors[:20]
            errors.extend(
                f"Tencent quote supplement: {message}" for message in tencent_errors[:10]
            )
            print(
                "[OK] Tencent quote supplement: "
                f"{tencent_enrichment['returned_symbols']}/"
                f"{tencent_enrichment['requested_symbols']} symbols; "
                f"filled volume_ratio={tencent_enrichment['volume_ratio_filled']}, "
                f"turnover_pct={tencent_enrichment['turnover_pct_filled']}."
            )
        except Exception as exc:
            tencent_enrichment.update({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            })
            errors.append(
                f"Tencent quote supplement: {type(exc).__name__}: {exc}"[:600]
            )
            print(f"[WARN] Tencent quote supplement failed: {type(exc).__name__}: {exc}")
    boards, board_errors = fetch_board_data(
        prefer_sina=eastmoney_blocked or spot_source == "stock_zh_a_spot_sina"
    )
    errors.extend(board_errors)

    market_summary = summarize_market(spot)
    quote_times = (
        [str(value) for value in spot["quote_timestamp"].dropna().unique()]
        if "quote_timestamp" in spot.columns else []
    )
    spot_quote_timestamp = max(quote_times) if quote_times else None
    tencent_quote_times = (
        [str(value) for value in spot["tencent_quote_timestamp"].dropna().unique()]
        if "tencent_quote_timestamp" in spot.columns else []
    )
    tencent_quote_timestamp_latest = (
        max(tencent_quote_times) if tencent_quote_times else None
    )
    quote_timestamp_note = (
        "Tencent quote timestamp is read from the per-security quote payload; "
        "spot_retrieved_at is the local collection time. "
        "The primary Tencent snapshot supplies both price and technical quote fields."
        if spot_source.startswith("tencent_")
        else (
            "AKShare spot endpoints do not guarantee a per-security exchange timestamp; "
            "spot_retrieved_at is collection time only. "
            "Tencent supplemental metrics retain their own per-security quote timestamp "
            "and do not replace the primary-source price or daily change."
        )
    )
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
                current_volume_ratio=safe_number(row["volume_ratio"]),
                current_turnover_pct=safe_number(row["turnover_pct"]),
                current_amount_cny=safe_number(row["amount"]),
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
                "spot_volume_ratio_source": (
                    str(row["volume_ratio_source"])
                    if "volume_ratio_source" in row.index and pd.notna(row["volume_ratio_source"])
                    else ("tencent_qt_gtimg" if spot_source.startswith("tencent_") else spot_source)
                    if safe_number(row["volume_ratio"]) is not None else None
                ),
                "spot_turnover_pct_source": (
                    str(row["turnover_pct_source"])
                    if "turnover_pct_source" in row.index and pd.notna(row["turnover_pct_source"])
                    else ("tencent_qt_gtimg" if spot_source.startswith("tencent_") else spot_source)
                    if safe_number(row["turnover_pct"]) is not None else None
                ),
                "tencent_quote_timestamp": (
                    str(row["tencent_quote_timestamp"])
                    if "tencent_quote_timestamp" in row.index and pd.notna(row["tencent_quote_timestamp"])
                    else None
                ),
                "tencent_quote_available": (
                    bool(row["tencent_quote_available"])
                    if "tencent_quote_available" in row.index else False
                ),
                "tencent_quote_price": safe_number(row.get("tencent_price")),
                "tencent_quote_change_pct": safe_number(row.get("tencent_change_pct")),
                "pct_60d_from_spot": safe_number(row["pct_60d"]),
                "spot_source": spot_source,
                "spot_retrieved_at": collected_at.isoformat(timespec="seconds"),
                "exchange_quote_timestamp": (
                    str(row["quote_timestamp"])
                    if "quote_timestamp" in row.index and pd.notna(row["quote_timestamp"])
                    else None
                ),
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
                item.get("breakout_start_score", 0),
                item.get("momentum_continuation_score", 0),
                item.get("next_day_momentum_score", 0),
                item.get("oversold_bounce_score", 0),
            ),
            item.get("liquidity_ok", False),
            item.get("average_amount_20d_cny") or 0,
        ),
        reverse=True,
    )

    # News is collected after technical ranking so requests focus on the highest-ranked
    # candidates. It is stored as an independent context/risk dimension and does not
    # change existing strategy scores or ranking until out-of-sample evidence supports it.
    try:
        news_data = collect_news(
            candidates,
            max_stock_symbols=min(20, len(candidates)),
            delay_seconds=0.25,
            now=now_cn(),
        )
    except Exception as exc:
        # News is an optional enrichment: a news-provider failure must never abort
        # the technical market scan or cause stale news to be reused.
        news_data = {
            "schema_version": 1,
            "status": "error",
            "collected_at": now_cn().isoformat(timespec="seconds"),
            "timezone": "Asia/Shanghai",
            "provider_status": {},
            "coverage": {
                "candidate_symbols_requested": min(20, len(candidates)),
                "candidate_symbols_with_recent_news": 0,
                "candidate_symbols_total": len(candidates),
            },
            "market_sentiment": {"sentiment_label": "unavailable"},
            "market_news": [],
            "candidate_news": {},
            "errors": [f"{type(exc).__name__}: {exc}"[:1000]],
            "limitations": ["News collection failed; no news-based conclusion is available."],
        }
    news_by_code = news_data.get("candidate_news", {})
    for candidate in candidates:
        summary = news_by_code.get(str(candidate.get("code", "")).zfill(6), {})
        candidate["news_sentiment_label"] = summary.get("sentiment_label", "not_scanned")
        candidate["news_sentiment_score"] = summary.get("average_keyword_score")
        candidate["news_recent_count_48h"] = summary.get("recent_news_count_48h", 0)
        candidate["news_risk_alert"] = bool(summary.get("risk_alert", False))
        candidate["news_risk_flags"] = summary.get("risk_flags", [])
        candidate["news_latest_at"] = summary.get("latest_news_at")
        candidate["news_sources"] = summary.get("sources", [])
        candidate["news_scanned"] = bool(summary.get("scanned", False))
    print(
        "[NEWS][SUMMARY] "
        + json.dumps(news_data.get("coverage", {}), ensure_ascii=False)
        + f" status={news_data.get('status')}"
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
        "schema_version": 2,
        "status": status,
        "collected_at": collected_at.isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "quote_timestamp_note": quote_timestamp_note,
        "market_data": {
            "spot_source": spot_source,
            "spot_quote_timestamp": spot_quote_timestamp,
            "tencent_supplemental_quote_timestamp_latest": tencent_quote_timestamp_latest,
            "tencent_quote_enrichment": tencent_enrichment,
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
            "strategy_mode": "six_parallel_short_term_strategies",
            "strategy_names": [
                "trend_pullback",
                "bottom_reversal",
                "breakout_start",
                "momentum_continuation",
                "next_day_momentum",
                "oversold_bounce",
            ],
            "atr14_suitability_pct_range": [1.2, 6.5],
            "candidate_output_limit": output_candidates,
        },
        "risk_management": {
            "account_capital_cny": 10000,
            "cash_reserve_pct": 30,
            "max_positions": 2,
            "max_single_position_budget_cap_cny": 10000,
            "effective_single_position_budget_cny": 7000,
            "equal_budget_if_two_positions_cny": 3500,
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
            "breakout_start_b_count": int(
                sum(1 for item in candidates if item.get("breakout_start_tier") == "B")
            ),
            "breakout_start_c_count": int(
                sum(1 for item in candidates if item.get("breakout_start_tier") == "C")
            ),
            "momentum_continuation_b_count": int(
                sum(1 for item in candidates if item.get("momentum_continuation_tier") == "B")
            ),
            "momentum_continuation_c_count": int(
                sum(1 for item in candidates if item.get("momentum_continuation_tier") == "C")
            ),
            "next_day_momentum_b_count": int(
                sum(1 for item in candidates if item.get("next_day_momentum_tier") == "B")
            ),
            "next_day_momentum_c_count": int(
                sum(1 for item in candidates if item.get("next_day_momentum_tier") == "C")
            ),
            "oversold_bounce_b_count": int(
                sum(1 for item in candidates if item.get("oversold_bounce_tier") == "B")
            ),
            "oversold_bounce_c_count": int(
                sum(1 for item in candidates if item.get("oversold_bounce_tier") == "C")
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
        "news_data": news_data,
        "candidates": candidates[:output_candidates],
        "errors": errors[-30:],
        "limitations": [
            "The full-market spot snapshot is scanned, but detailed daily K-lines are fetched only for the liquidity/trend-prioritized history pool.",
            "Six independent short-term strategies are scored; a negative 60-day return is not a hard veto for all strategies.",
            "Detailed daily K-lines are calculated for a diversified 300-stock multi-signal history pool by default, not every stock in the 5,000+ spot universe.",
            "No 14:25-14:30 minute-bar confirmation is included yet; next-day-strength candidates are watchlist signals only and all candidates have buyable=false.",
            "No precise buy zone or take-profit price is emitted until minute-level confirmation and support validation are implemented.",
            "ATR(14) is used only as a short-term volatility suitability/ranking feature; it is not a prediction of future returns.",
            "B-tier candidates must pass the estimated 100-share affordability check under the configured per-position budget; C-tier candidates may remain unaffordable watch-only names.",
            "The spot endpoint retrieval time is not a guaranteed exchange quote timestamp.",
            "When Eastmoney is rate-limited, Sina spot/sector data is used as a fallback; Tencent supplements missing quote metrics where available.",
            "Tencent supplemental volume ratio/turnover metrics retain per-security quote timestamps and do not overwrite primary-source prices or daily change.",
            "A failed or partial Tencent supplement is disclosed in market_data.tencent_quote_enrichment; missing metrics remain null and are not fabricated.",
            "Sector board rankings are market context and are not treated as proof that an individual stock belongs to a strong trend setup.",
            "News headlines and keyword sentiment are an independent context/risk dimension; they do not alter the six technical strategy scores or ranking in this version.",
            "Only the top 20 ranked candidates receive individual news and disclosure lookups; other candidates are explicitly marked not_scanned.",
        ],
    }

    JSON_OUTPUT.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    NEWS_OUTPUT.write_text(
        json.dumps(json_safe(news_data), ensure_ascii=False, indent=2),
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
    print(f"[OK] News JSON: {NEWS_OUTPUT}")
    print(f"[SUMMARY] Scan status: {status}; candidates: {len(candidates)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-market A-share snapshot with six parallel short-term opportunity scans."
    )
    parser.add_argument("--max-history-stocks", type=int, default=300)
    parser.add_argument("--min-amount-cny", type=float, default=50_000_000)
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
