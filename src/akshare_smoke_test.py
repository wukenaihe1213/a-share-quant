from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "smoke_test.json"


def json_value(value: Any) -> Any:
    """Convert pandas/numpy/date values into JSON-safe Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(
        f"Required column not found. Expected one of {names}; "
        f"actual columns: {list(frame.columns)}"
    )


def fetch_with_retry(label: str, func, attempts: int = 3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[INFO] Fetching {label} (attempt {attempt}/{attempts})")
            return func()
        except Exception as exc:
            last_error = exc
            print(f"[WARN] {label} failed: {type(exc).__name__}: {exc}")
            # Eastmoney explicitly warns that retrying during a rate-limit
            # cooldown can extend the block. Stop immediately in that case.
            if type(exc).__name__ == "Push2RateLimitError" or "临时限流" in str(exc):
                print("[ERROR] Rate limit detected; stopping retries to respect the upstream cooldown.")
                raise
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise RuntimeError(
        f"{label} failed after {attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def fetch_spot_snapshot(symbol: str) -> dict[str, Any]:
    """Fetch one stock's quote instead of crawling all 59 market-wide pages."""
    frame = fetch_with_retry(
        f"individual quote for {symbol}",
        lambda: ak.stock_bid_ask_em(symbol=symbol),
        attempts=2,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"AKShare returned an empty quote for {symbol}.")
    if not {"item", "value"}.issubset(frame.columns):
        raise RuntimeError(
            "Unexpected stock_bid_ask_em schema. "
            f"Actual columns: {list(frame.columns)}"
        )

    quote = dict(zip(frame["item"].astype(str), frame["value"]))
    required = ("最新", "涨幅", "金额", "昨收")
    missing = [key for key in required if key not in quote]
    if missing:
        raise RuntimeError(f"Quote response is missing fields: {missing}")

    numeric_fields = (
        "最新", "均价", "涨幅", "涨跌", "总手", "金额", "换手",
        "量比", "最高", "最低", "今开", "昨收", "涨停", "跌停",
    )
    parsed = {key: as_float(quote.get(key)) for key in numeric_fields}

    return {
        "universe_rows": None,
        "target": {
            "code": symbol,
            "name": None,
            "latest_price": parsed["最新"],
            "change_pct": parsed["涨幅"],
            "amount_cny": parsed["金额"],
            "average_price": parsed["均价"],
            "volume_lots": parsed["总手"],
            "turnover_pct": parsed["换手"],
            "volume_ratio": parsed["量比"],
            "high": parsed["最高"],
            "low": parsed["最低"],
            "open": parsed["今开"],
            "previous_close": parsed["昨收"],
        },
        "five_level_quote": {
            str(row["item"]): as_float(row["value"])
            for _, row in frame.iterrows()
        },
        "source": "AKShare stock_bid_ask_em (Eastmoney individual quote)",
        "source_quote_timestamp": None,
        "timestamp_note": (
            "The individual quote response does not expose a reliable exchange "
            "timestamp. collected_at is the retrieval time, not a guaranteed "
            "exchange quote timestamp."
        ),
    }

def fetch_daily_history(symbol: str) -> dict[str, Any]:
    end = datetime.now(SHANGHAI).date()
    start = end - timedelta(days=120)
    frame = fetch_with_retry(
        f"daily history for {symbol}",
        lambda: ak.stock_zh_a_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        ),
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"AKShare returned no daily history for {symbol}.")

    date_col = find_column(frame, ("日期", "date"))
    close_col = find_column(frame, ("收盘", "close"))
    volume_col = find_column(frame, ("成交量", "volume"))
    amount_col = find_column(frame, ("成交额", "amount"))

    records = []
    for _, row in frame.tail(10).iterrows():
        records.append({str(k): json_value(v) for k, v in row.to_dict().items()})

    last = frame.iloc[-1]
    return {
        "symbol": symbol,
        "adjustment": "qfq",
        "rows_returned": int(len(frame)),
        "latest_bar_date": json_value(last[date_col]),
        "latest_close": as_float(last[close_col]),
        "latest_volume": as_float(last[volume_col]),
        "latest_amount_cny": as_float(last[amount_col]),
        "last_10_bars": records,
        "source": "AKShare stock_zh_a_hist_em (Eastmoney source)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test AKShare A-share spot and daily-history connectivity."
    )
    parser.add_argument(
        "--symbol",
        default="000001",
        help="Six-digit A-share code to test (default: 000001, Ping An Bank).",
    )
    args = parser.parse_args()
    # Strip accidental leading backslashes from workflow input before validation.
    symbol = str(args.symbol).strip().lstrip(chr(92)).zfill(6)
    if len(symbol) != 6 or not symbol.isdigit():
        parser.error("symbol must be exactly six digits, e.g. 000001")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(SHANGHAI).isoformat(timespec="seconds")

    result: dict[str, Any] = {
        "project": "a-share-quant",
        "status": "running",
        "collected_at": collected_at,
        "timezone": "Asia/Shanghai",
        "symbol": symbol,
        "akshare_version": getattr(ak, "__version__", "unknown"),
        "data_sources": [],
        "notes": [
            "This is a connectivity test, not a trading signal.",
            "Individual quote collected_at must not be represented as an exchange quote timestamp.",
            "Historical daily bars may not include a completed bar for the current session before market close.",
        ],
    }

    try:
        result["spot"] = fetch_spot_snapshot(symbol)
        result["daily_history"] = fetch_daily_history(symbol)
        result["data_sources"] = [
            "stock_bid_ask_em",
            "stock_zh_a_hist_em",
        ]
        result["status"] = "ok"
        result["completed_at"] = datetime.now(SHANGHAI).isoformat(timespec="seconds")
        OUTPUT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[OK] AKShare connectivity test passed. Output: {OUTPUT}")
        print(json.dumps(result["spot"]["target"], ensure_ascii=False, indent=2))
        print(
            "Latest daily bar:",
            result["daily_history"]["latest_bar_date"],
            "close:",
            result["daily_history"]["latest_close"],
        )
        return 0
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1500]
        result["completed_at"] = datetime.now(SHANGHAI).isoformat(timespec="seconds")
        OUTPUT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[ERROR] AKShare connectivity test failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
