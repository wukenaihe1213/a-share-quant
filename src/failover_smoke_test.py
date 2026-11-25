from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import src.market_scan as market_scan

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "failover_smoke_test.json"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def injected_outage(name: str):
    def fail(*args, **kwargs):
        raise RuntimeError(f"INJECTED_OUTAGE: {name}")
    return fail


def main() -> int:
    checked_at = datetime.now(SHANGHAI)
    result = {
        "status": "error",
        "checked_at": checked_at.isoformat(timespec="seconds"),
        "purpose": "Inject AKShare spot/history outages and verify Tencent/BaoStock takeover.",
    }
    try:
        # Inject simultaneous failures in the live-quote sources. The actual
        # Tencent + BaoStock path remains unmocked and makes real network calls.
        spot_sources = [
            "stock_zh_a_spot_em",
            "stock_zh_a_spot_sina",
            "stock_zh_a_spot",
        ]
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                market_scan, "interface_alternatives",
                return_value=["stock_zh_a_spot_sina"]
            ))
            for source in spot_sources:
                stack.enter_context(patch.object(
                    market_scan.ak, source,
                    side_effect=injected_outage(source), create=True
                ))
            spot, spot_source, spot_errors = market_scan.fetch_spot_universe()

        quote_times = (
            spot["quote_timestamp"].dropna().astype(str).tolist()
            if "quote_timestamp" in spot.columns else []
        )
        quote_count = int(spot["code"].nunique())
        universe_coverage = quote_count / max(len(spot), 1)
        valid_prices = int(
            ((spot["price"] > 0) & spot["price"].notna()).sum()
        )
        valid_amounts = int(
            (spot["amount"].notna() & (spot["amount"] >= 0)).sum()
        )
        if not spot_source.startswith("tencent_"):
            raise RuntimeError(f"Expected Tencent fallback, received {spot_source}")
        if quote_count < 1000 or valid_prices != len(spot):
            raise RuntimeError(
                f"Tencent fallback data invalid: rows={len(spot)}, "
                f"unique_codes={quote_count}, valid_prices={valid_prices}"
            )
        if len(spot_errors) < len(spot_sources):
            raise RuntimeError(
                f"Expected all injected spot failures to be recorded; got {spot_errors}"
            )

        # Inject failures in all AKShare historical providers. The actual
        # BaoStock history request remains unmocked and must return usable bars.
        history_sources = [
            "stock_zh_a_hist_tx",
            "stock_zh_a_hist_sina",
            "stock_zh_a_daily",
        ]
        start = (checked_at.date() - timedelta(days=180)).strftime("%Y%m%d")
        end = checked_at.date().strftime("%Y%m%d")
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                market_scan, "interface_alternatives", return_value=[]
            ))
            for source in history_sources:
                stack.enter_context(patch.object(
                    market_scan.ak, source,
                    side_effect=injected_outage(source), create=True
                ))
            history, history_source, eastmoney_limited = market_scan.fetch_history(
                "600000", start, end, skip_eastmoney=True
            )

        if history_source != "baostock_query_history_k_data_plus":
            raise RuntimeError(
                f"Expected BaoStock history fallback, received {history_source}"
            )
        if len(history) < 10:
            raise RuntimeError(
                f"BaoStock history too short for validation: {len(history)} rows"
            )

        latest_history_date = str(
            history["date"].max().date()
            if hasattr(history["date"].max(), "date")
            else history["date"].max()
        )
        result.update({
            "status": "ok",
            "spot_failover": {
                "injected_failures": spot_sources,
                "observed_errors": spot_errors,
                "selected_source": spot_source,
                "quote_rows": len(spot),
                "unique_codes": quote_count,
                "valid_price_rows": valid_prices,
                "valid_amount_rows": valid_amounts,
                "quote_timestamp_rows": len(quote_times),
                "latest_quote_timestamp": max(quote_times) if quote_times else None,
                "sample_quotes": spot.head(5).to_dict(orient="records"),
            },
            "history_failover": {
                "injected_failures": history_sources,
                "selected_source": history_source,
                "symbol": "600000",
                "start_date": start,
                "end_date": end,
                "history_rows": len(history),
                "latest_history_date": latest_history_date,
                "eastmoney_rate_limited": eastmoney_limited,
            },
        })
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print("[SUMMARY] " + json.dumps({
            "status": result["status"],
            "spot_source": spot_source,
            "spot_rows": len(spot),
            "spot_quote_timestamps": len(quote_times),
            "history_source": history_source,
            "history_rows": len(history),
            "latest_history_date": latest_history_date,
        }, ensure_ascii=False))
        print(f"[OK] JSON: {OUTPUT}")
        return 0
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:2000]
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
