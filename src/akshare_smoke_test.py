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
            if attempt < attempts:
                time.sleep(5 * attempt)
    raise RuntimeError(
        f"{label} failed after {attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def fetch_spot_snapshot(symbol: str) -> dict[str, Any]:
    frame = fetch_with_retry("A-share spot snapshot", ak.stock_zh_a_spot_em)
    if frame is None or frame.empty:
        raise RuntimeError("AKShare returned an empty A-share spot snapshot.")

    code_col = find_column(frame, ("代码", "股票代码", "symbol"))
    name_col = find_column(frame, ("名称", "股票简称", "name"))
    price_col = find_column(frame, ("最新价", "最新", "price"))
    change_col = find_column(frame, ("涨跌幅", "涨幅", "change_pct"))
    amount_col = find_column(frame, ("成交额", "amount"))

    codes = frame[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    matches = frame.loc[codes == symbol]
    if matches.empty:
        raise RuntimeError(
            f"Stock {symbol} was not found in the current spot snapshot. "
            "It may be suspended, delisted, or the upstream response may be incomplete."
        )

    row = matches.iloc[0]
    amount_sorted = frame.copy()
    amount_sorted["_amount_numeric"] = pd.to_numeric(
        amount_sorted[amount_col], errors="coerce"
    )
    top_rows = amount_sorted.sort_values(
        "_amount_numeric", ascending=False
    ).head(10)

    top_turnover = []
    for _, item in top_rows.iterrows():
        top_turnover.append(
            {
                "code": str(item[code_col]).replace(".0", "").zfill(6),
                "name": json_value(item[name_col]),
                "latest_price": as_float(item[price_col]),
                "change_pct": as_float(item[change_col]),
                "amount_cny": as_float(item[amount_col]),
            }
        )

    return {
        "universe_rows": int(len(frame)),
        "columns": [str(col) for col in frame.columns],
        "target": {
            "code": symbol,
            "name": json_value(row[name_col]),
            "latest_price": as_float(row[price_col]),
            "change_pct": as_float(row[change_col]),
            "amount_cny": as_float(row[amount_col]),
        },
        "top_10_by_turnover": top_turnover,
        "source": "AKShare stock_zh_a_spot_em (Eastmoney source)",
        "source_quote_timestamp": None,
        "timestamp_note": (
            "AKShare spot response does not provide a reliable per-quote "
            "exchange timestamp in this output. retrieved_at is the collection "
            "time, not a guaranteed exchange quote timestamp."
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
            "Spot retrieved_at must not be represented as an exchange quote timestamp.",
            "Historical daily bars may not include a completed bar for the current session before market close.",
        ],
    }

    try:
        result["spot"] = fetch_spot_snapshot(symbol)
        result["daily_history"] = fetch_daily_history(symbol)
        result["data_sources"] = [
            "stock_zh_a_spot_em",
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
