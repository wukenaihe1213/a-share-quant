from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.tencent_quote import (
    fetch_baostock_universe,
    fetch_tencent_quotes,
    market_code_to_tencent,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "tencent_smoke_test.json"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Direct Tencent live quote + BaoStock universe smoke test."
    )
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="0 tests the entire BaoStock A-share universe.")
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=80)
    args = parser.parse_args()

    result = {
        "status": "error",
        "checked_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "data_source": "Tencent qt.gtimg.cn + BaoStock query_stock_basic",
    }
    try:
        universe = fetch_baostock_universe()
        if args.max_symbols > 0:
            universe = universe.head(args.max_symbols).copy()
        symbols = [
            market_code_to_tencent(code) for code in universe["code"].astype(str)
        ]
        quotes, warnings = fetch_tencent_quotes(
            symbols, batch_size=args.batch_size
        )
        unique = {str(row["code"]): row for row in quotes}
        quote_rows = list(unique.values())
        coverage = len(quote_rows) / max(len(universe), 1)
        timestamps = [
            str(row.get("quote_timestamp"))
            for row in quote_rows if row.get("quote_timestamp")
        ]
        timestamp_counts = Counter(timestamps)
        missing_timestamp = len(quote_rows) - len(timestamps)
        valid_prices = sum(
            1 for row in quote_rows
            if isinstance(row.get("price"), (int, float)) and row["price"] > 0
        )
        valid_amounts = sum(
            1 for row in quote_rows
            if isinstance(row.get("amount"), (int, float)) and row["amount"] >= 0
        )

        result.update({
            "status": (
                "ok"
                if len(quote_rows) >= 1000
                and coverage >= args.min_coverage
                and valid_prices == len(quote_rows)
                and len(timestamps) > 0
                else "partial"
            ),
            "universe_count": len(universe),
            "quote_count": len(quote_rows),
            "coverage_pct": round(coverage * 100, 2),
            "valid_price_count": valid_prices,
            "valid_amount_count": valid_amounts,
            "timestamp_count": len(timestamps),
            "missing_timestamp_count": missing_timestamp,
            "latest_quote_timestamp": max(timestamps) if timestamps else None,
            "most_common_quote_timestamps": [
                {"timestamp": key, "count": count}
                for key, count in timestamp_counts.most_common(5)
            ],
            "sample_quotes": quote_rows[:10],
            "warnings": warnings[:30],
        })
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("[SUMMARY] " + json.dumps({
            key: result[key] for key in (
                "status", "universe_count", "quote_count", "coverage_pct",
                "valid_price_count", "valid_amount_count",
                "timestamp_count", "missing_timestamp_count",
                "latest_quote_timestamp",
            )
        }, ensure_ascii=False))
        print(f"[OK] JSON: {OUTPUT}")
        return 0 if result["status"] == "ok" else 1
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
