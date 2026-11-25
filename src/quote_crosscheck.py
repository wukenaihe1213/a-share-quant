from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time as day_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

from src.market_scan import normalize_spot
from src.tencent_quote import (
    fetch_baostock_universe,
    fetch_tencent_quotes,
    market_code_to_tencent,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
JSON_OUTPUT = DATA_DIR / "quote_crosscheck.json"
DETAILS_OUTPUT = DATA_DIR / "quote_crosscheck_details.csv"
PAIR_SUMMARY_OUTPUT = DATA_DIR / "quote_crosscheck_pairs.csv"

PRICE_OK_PCT = 0.20
PRICE_REVIEW_PCT = 0.50


def now_cn() -> datetime:
    return datetime.now(SHANGHAI)


def parse_tencent_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=SHANGHAI)
        except ValueError:
            continue
    return None


def is_cn_continuous_trading_clock_window(value: datetime) -> bool:
    local = value.astimezone(SHANGHAI)
    if local.weekday() >= 5:
        return False
    clock = local.time()
    return (
        day_time(9, 30) <= clock <= day_time(11, 30)
        or day_time(13, 0) <= clock <= day_time(15, 0)
    )


def classify_price_difference(diff_pct: float | None) -> str:
    if diff_pct is None or pd.isna(diff_pct):
        return "unavailable"
    if diff_pct <= PRICE_OK_PCT:
        return "normal"
    if diff_pct <= PRICE_REVIEW_PCT:
        return "review"
    return "abnormal"


def fetch_akshare_snapshot(interface_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = now_cn()
    func = getattr(ak, interface_name, None)
    if not callable(func):
        raise RuntimeError(f"AKShare interface unavailable: {interface_name}")
    raw = func()
    frame = normalize_spot(raw)
    finished = now_cn()
    metadata = {
        "source": interface_name,
        "status": "ok",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "quote_timestamp_available": False,
        "note": "AKShare spot snapshot has no dependable per-security exchange timestamp; local retrieval time is recorded.",
    }
    if len(frame) < 1000:
        raise RuntimeError(
            f"{interface_name} returned only {len(frame)} normalized rows."
        )
    return frame, metadata


def choose_sample_codes(
    snapshots: list[pd.DataFrame], sample_size: int = 150
) -> list[str]:
    if sample_size < 30:
        raise ValueError("sample_size must be at least 30.")
    if not snapshots:
        raise ValueError("At least one AKShare snapshot is required to choose comparison symbols.")

    parts = []
    for frame in snapshots:
        if frame is None or frame.empty:
            continue
        part = frame[["code", "amount"]].copy()
        part["code"] = part["code"].astype(str).str.zfill(6)
        part["amount"] = pd.to_numeric(part["amount"], errors="coerce")
        parts.append(part)
    if not parts:
        raise ValueError("No usable stock codes in source snapshots.")

    combined = pd.concat(parts, ignore_index=True)
    liquidity = (
        combined.groupby("code", as_index=False)["amount"]
        .median()
        .sort_values("amount", ascending=False, na_position="last")
    )
    selected: list[str] = []

    def add_codes(values) -> None:
        for value in values:
            code = str(value).zfill(6)
            if code not in selected:
                selected.append(code)

    # Ensure the validation sample is not made up only of Shanghai mega-caps.
    quota = max(5, min(15, sample_size // 10))
    codes = liquidity["code"].astype(str)
    add_codes(liquidity.loc[codes.str.startswith(("0", "3")), "code"].head(quota))
    add_codes(liquidity.loc[codes.str.startswith(("6",)), "code"].head(quota))
    add_codes(liquidity.loc[codes.str.startswith(("4", "8", "92")), "code"].head(quota))
    add_codes(liquidity["code"].head(sample_size))
    return selected[:sample_size]


def compare_pair(
    source_a: str,
    frame_a: pd.DataFrame,
    source_b: str,
    frame_b: pd.DataFrame,
    time_gap_seconds: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = ["code", "name", "price", "change_pct", "amount"]
    left = frame_a[[column for column in columns if column in frame_a.columns]].copy()
    right = frame_b[[column for column in columns if column in frame_b.columns]].copy()
    left = left.rename(columns={
        "name": "name_a", "price": "price_a",
        "change_pct": "change_pct_a", "amount": "amount_a",
    })
    right = right.rename(columns={
        "name": "name_b", "price": "price_b",
        "change_pct": "change_pct_b", "amount": "amount_b",
    })
    merged = left.merge(right, on="code", how="inner")
    if merged.empty:
        return {
            "source_a": source_a, "source_b": source_b, "common_count": 0,
            "status": "insufficient_common_symbols",
            "collection_time_gap_seconds": time_gap_seconds,
        }, merged

    for col in ("price_a", "price_b", "change_pct_a", "change_pct_b", "amount_a", "amount_b"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    denominator = (merged["price_a"].abs() + merged["price_b"].abs()) / 2
    merged["price_diff_pct"] = (
        (merged["price_a"] - merged["price_b"]).abs()
        / denominator.where(denominator > 0)
        * 100
    )
    merged["price_diff_abs"] = (merged["price_a"] - merged["price_b"]).abs()
    merged["change_pct_diff_pp"] = (
        merged["change_pct_a"] - merged["change_pct_b"]
    ).abs()
    amount_denominator = merged[["amount_a", "amount_b"]].abs().max(axis=1)
    merged["amount_diff_pct"] = (
        (merged["amount_a"] - merged["amount_b"]).abs()
        / amount_denominator.where(amount_denominator > 0)
        * 100
    )
    merged["price_status"] = merged["price_diff_pct"].map(classify_price_difference)
    merged["source_a"] = source_a
    merged["source_b"] = source_b

    valid = merged["price_diff_pct"].dropna()
    if valid.empty:
        summary = {
            "source_a": source_a, "source_b": source_b,
            "common_count": int(len(merged)),
            "status": "no_comparable_prices",
            "collection_time_gap_seconds": time_gap_seconds,
        }
    else:
        normal = int((valid <= PRICE_OK_PCT).sum())
        review = int(((valid > PRICE_OK_PCT) & (valid <= PRICE_REVIEW_PCT)).sum())
        abnormal = int((valid > PRICE_REVIEW_PCT).sum())
        p95 = float(valid.quantile(0.95))
        summary = {
            "source_a": source_a,
            "source_b": source_b,
            "common_count": int(len(merged)),
            "comparable_price_count": int(valid.count()),
            "median_price_diff_pct": round(float(valid.median()), 4),
            "p95_price_diff_pct": round(p95, 4),
            "max_price_diff_pct": round(float(valid.max()), 4),
            "normal_within_0_20_pct": normal,
            "review_0_20_to_0_50_pct": review,
            "abnormal_over_0_50_pct": abnormal,
            "normal_share_pct": round(normal / len(valid) * 100, 2),
            "collection_time_gap_seconds": (
                round(float(time_gap_seconds), 1)
                if time_gap_seconds is not None else None
            ),
            "amount_diff_median_pct": (
                round(float(merged["amount_diff_pct"].median()), 2)
                if merged["amount_diff_pct"].notna().any() else None
            ),
            "change_pct_diff_median_pp": (
                round(float(merged["change_pct_diff_pp"].median()), 4)
                if merged["change_pct_diff_pp"].notna().any() else None
            ),
            "status": "pass" if len(valid) >= 30 and p95 <= PRICE_REVIEW_PCT else "review",
        }

    details = merged.sort_values("price_diff_pct", ascending=False).reset_index(drop=True)
    return summary, details



def choose_universe_sample_codes(universe: pd.DataFrame, sample_size: int = 150) -> list[str]:
    """Choose a deterministic, board-stratified fallback sample from BaoStock's code list."""
    if universe is None or universe.empty or "code" not in universe.columns:
        raise ValueError("BaoStock universe is empty or has no code column.")
    if sample_size < 30:
        raise ValueError("sample_size must be at least 30.")

    codes = (
        universe["code"].astype(str).str.extract(r"(\d{6})", expand=False)
        .dropna().drop_duplicates().tolist()
    )
    if len(codes) < 30:
        raise ValueError(f"BaoStock universe contains only {len(codes)} usable stock codes.")

    groups = (("0", "3"), ("6",), ("4", "8", "92"))
    selected: list[str] = []
    quota = max(1, sample_size // len(groups))
    for prefixes in groups:
        group_codes = [code for code in codes if code.startswith(prefixes)]
        for code in group_codes[:quota]:
            if code not in selected:
                selected.append(code)

    for code in codes:
        if code not in selected:
            selected.append(code)
        if len(selected) >= sample_size:
            break
    return selected[:sample_size]


def fetch_tencent_sample(sample_codes: list[str]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    started = now_cn()
    symbols = [market_code_to_tencent(code) for code in sample_codes]
    quotes, errors = fetch_tencent_quotes(symbols, batch_size=80)
    frame = pd.DataFrame(quotes)
    if frame.empty:
        raise RuntimeError("Tencent returned no comparable sample quotes.")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame = frame.drop_duplicates("code").reset_index(drop=True)
    finished = now_cn()
    timestamps = [
        parse_tencent_timestamp(value)
        for value in frame.get("quote_timestamp", pd.Series(dtype=str)).tolist()
    ]
    valid_timestamps = [value for value in timestamps if value is not None]
    current = finished
    ages = [(current - value).total_seconds() for value in valid_timestamps]
    session_open = is_cn_continuous_trading_clock_window(current)
    returned_symbols = int(frame["code"].nunique())
    coverage_pct = returned_symbols / max(len(sample_codes), 1) * 100
    metadata = {
        "source": "tencent_qt_gtimg",
        "status": (
            "ok"
            if returned_symbols == len(sample_codes) and not errors
            else "partial"
        ),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "requested_symbols": len(sample_codes),
        "returned_symbols": returned_symbols,
        "quote_coverage_pct": round(coverage_pct, 2),
        "batch_errors": errors[:20],
        "quote_timestamp_count": len(valid_timestamps),
        "missing_or_invalid_timestamp_count": int(len(frame) - len(valid_timestamps)),
        "latest_quote_timestamp": (
            max(valid_timestamps).isoformat(timespec="seconds")
            if valid_timestamps else None
        ),
        "median_quote_age_seconds": (
            round(float(pd.Series(ages).median()), 1) if ages else None
        ),
        "p95_quote_age_seconds": (
            round(float(pd.Series(ages).quantile(0.95)), 1) if ages else None
        ),
        "within_continuous_trading_clock_window": session_open,
        "trading_day_calendar_checked": False,
        "freshness_status": (
            "fresh"
            if session_open and ages and pd.Series(ages).quantile(0.95) <= 180
            else (
                "stale_or_unverifiable"
                if session_open else "not_evaluated_outside_continuous_trading"
            )
        ),
        "note": "Tencent provides per-security quote timestamps; compare them with local collection time. The clock-window check does not verify exchange holidays; outside the regular clock window freshness is not graded.",
    }
    return frame, metadata, errors


def run_crosscheck(sample_size: int = 150) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checked_at = now_cn()
    result: dict[str, Any] = {
        "project": "a-share-quant",
        "status": "error",
        "checked_at": checked_at.isoformat(timespec="seconds"),
        "sample_size_requested": sample_size,
        "price_tolerance": {
            "normal_max_pct": PRICE_OK_PCT,
            "review_max_pct": PRICE_REVIEW_PCT,
            "note": "Engineering triage thresholds, not exchange-mandated quote tolerances.",
        },
        "sources": {},
        "comparisons": [],
        "warnings": [],
    }
    snapshots: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}


    for source in (
        "stock_zh_a_spot_em",
        "stock_zh_a_spot_sina",
        "stock_zh_a_spot",
    ):
        try:
            frame, meta = fetch_akshare_snapshot(source)
            snapshots[source] = frame
            metadata[source] = meta
            print(f"[OK] {source}: {len(frame)} rows")
        except Exception as exc:
            metadata[source] = {
                "source": source,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
            result["warnings"].append(f"{source}: {type(exc).__name__}: {exc}"[:500])
            print(f"[WARN] {source}: {type(exc).__name__}: {exc}")

    if snapshots:
        sample_codes = choose_sample_codes(list(snapshots.values()), sample_size)
        result["sample_universe_source"] = "successful_akshare_snapshot"
    else:
        # Use BaoStock only to obtain stock codes if all live AKShare snapshots fail.
        try:
            universe = fetch_baostock_universe()
            sample_codes = choose_universe_sample_codes(universe, sample_size)
            metadata["baostock_stock_universe"] = {
                "source": "baostock_query_stock_basic",
                "status": "ok",
                "rows": int(len(universe)),
                "use": "code-universe fallback only; not treated as a live quote source",
            }
            result["sample_universe_source"] = "baostock_stock_basic"
            print(
                f"[OK] BaoStock fallback universe: {len(universe)} stocks; "
                f"selected {len(sample_codes)} codes for Tencent quote check"
            )
        except Exception as exc:
            metadata["baostock_stock_universe"] = {
                "source": "baostock_query_stock_basic",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
            result["warnings"].append(
                f"BaoStock universe fallback: {type(exc).__name__}: {exc}"[:500]
            )
            raise RuntimeError(
                "All AKShare live snapshots failed and BaoStock could not provide "
                "a fallback stock-code universe. " + str(exc)
            ) from exc

    if not sample_codes:
        raise RuntimeError("No stock codes are available for Tencent quote validation.")
    result["sample_codes"] = sample_codes

    tencent_frame = pd.DataFrame()
    try:
        tencent_frame, tencent_meta, tencent_errors = fetch_tencent_sample(sample_codes)
        snapshots["tencent_qt_gtimg"] = tencent_frame
        metadata["tencent_qt_gtimg"] = tencent_meta
        result["warnings"].extend(tencent_errors[:20])
        print(
            f"[OK] tencent_qt_gtimg: {tencent_meta['returned_symbols']}/"
            f"{tencent_meta['requested_symbols']} sample quotes "
            f"({tencent_meta['quote_coverage_pct']}%)"
        )
    except Exception as exc:
        metadata["tencent_qt_gtimg"] = {
            "source": "tencent_qt_gtimg",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
        result["warnings"].append(
            f"tencent_qt_gtimg: {type(exc).__name__}: {exc}"[:500]
        )
        print(f"[WARN] tencent_qt_gtimg: {type(exc).__name__}: {exc}")

    for source, meta in metadata.items():
        result["sources"][source] = meta

    pair_summaries: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    source_names = list(snapshots.keys())
    for i, source_a in enumerate(source_names):
        for source_b in source_names[i + 1:]:
            meta_a = metadata[source_a]
            meta_b = metadata[source_b]
            end_a = meta_a.get("finished_at")
            end_b = meta_b.get("finished_at")
            gap = None
            if end_a and end_b:
                gap = abs(
                    (datetime.fromisoformat(end_a) - datetime.fromisoformat(end_b)).total_seconds()
                )
            summary, details = compare_pair(
                source_a, snapshots[source_a], source_b, snapshots[source_b], gap
            )
            pair_summaries.append(summary)
            if not details.empty:
                detail_frames.append(details)

    usable_pairs = [
        item for item in pair_summaries
        if item.get("comparable_price_count", 0) >= 30
    ]
    if not usable_pairs:
        result["status"] = "insufficient_comparisons"
    elif all(item.get("status") == "pass" for item in usable_pairs):
        result["status"] = "pass"
    else:
        result["status"] = "review"

    result["comparisons"] = pair_summaries
    result["source_count_success"] = len(snapshots)
    result["sample_size_actual"] = len(sample_codes)
    result["tencent_returned_count"] = (
        int(tencent_frame["code"].nunique()) if not tencent_frame.empty else 0
    )
    result["comparison_count"] = len(usable_pairs)
    if not usable_pairs:
        result["warnings"].append(
            "Fewer than two usable live quote providers were available; "
            "the report is diagnostic only and does not establish cross-source agreement."
        )

    details_all = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames else pd.DataFrame()
    )
    if not details_all.empty:
        details_all.to_csv(DETAILS_OUTPUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_summaries).to_csv(
        PAIR_SUMMARY_OUTPUT, index=False, encoding="utf-8-sig"
    )
    JSON_OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("[SUMMARY] " + json.dumps({
        "status": result["status"],
        "sources": {
            key: value["status"] for key, value in metadata.items()
        },
        "sample_size": result["sample_size_actual"],
        "tencent_returned": result["tencent_returned_count"],
        "comparisons": pair_summaries,
    }, ensure_ascii=False, default=str))
    print(f"[OK] JSON: {JSON_OUTPUT}")
    print(f"[OK] Details CSV: {DETAILS_OUTPUT}")
    print(f"[OK] Pair summary CSV: {PAIR_SUMMARY_OUTPUT}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-check A-share live quotes from Eastmoney, Sina and Tencent."
    )
    parser.add_argument("--sample-size", type=int, default=150)
    args = parser.parse_args()
    if not 30 <= args.sample_size <= 500:
        parser.error("--sample-size must be between 30 and 500.")
    try:
        result = run_crosscheck(sample_size=args.sample_size)
        # "insufficient_comparisons" is a valid diagnostic outcome: a report was
        # produced, but this status is not a successful accuracy validation.
        return 0 if result["status"] in ("pass", "review", "insufficient_comparisons") else 1
    except Exception as exc:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "project": "a-share-quant",
            "status": "error",
            "checked_at": now_cn().isoformat(timespec="seconds"),
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
