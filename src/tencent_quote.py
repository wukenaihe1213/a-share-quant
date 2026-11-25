from __future__ import annotations

import re
import time
from typing import Any

import pandas as pd
import requests

TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q="
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Referer": "https://gu.qq.com/",
}


def market_code_to_tencent(code: str) -> str:
    """Convert BaoStock-style or plain A-share code to Tencent quote symbol."""
    value = str(code).strip().lower()
    if "." in value:
        market, number = value.split(".", 1)
    else:
        number = value.zfill(6)
        if number.startswith(("92", "4", "8")):
            market = "bj"
        elif number.startswith(("6", "5", "9")):
            market = "sh"
        else:
            market = "sz"
    return f"{market}{number.zfill(6)}"


def _number(fields: list[str], index: int) -> float | None:
    if index >= len(fields) or not fields[index].strip():
        return None
    try:
        value = float(fields[index].replace(",", "").strip())
        return value if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def parse_tencent_quote_text(text: str) -> list[dict[str, Any]]:
    """Parse Tencent's public GBK text quote response into normalized rows.

    Tencent field 37 is turnover in CNY 10,000; field 30 is quote timestamp.
    """
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r'v_([A-Za-z0-9_]+)\s*=\s*"([^"]*)"')
    for match in pattern.finditer(text):
        fields = match.group(2).split("~")
        if len(fields) < 39:
            continue
        code = fields[2].strip().zfill(6)
        name = fields[1].strip()
        price = _number(fields, 3)
        if not code.isdigit() or not name or price is None or price <= 0:
            continue
        amount_wan = _number(fields, 37)
        rows.append({
            "code": code,
            "name": name,
            "price": price,
            "change_pct": _number(fields, 32),
            "amount": amount_wan * 10_000 if amount_wan is not None else None,
            "volume": _number(fields, 36),
            "turnover_pct": _number(fields, 38),
            "volume_ratio": _number(fields, 47),
            "pct_60d": None,
            "industry": None,
            "quote_timestamp": fields[30].strip() or None,
            "quote_source": "tencent_qt_gtimg",
        })
    return rows


def fetch_tencent_quotes(
    symbols: list[str],
    batch_size: int = 80,
    timeout_seconds: int = 20,
    pause_seconds: float = 0.15,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch Tencent quotes in bounded batches; failed batches do not abort others."""
    if not symbols:
        raise ValueError("No Tencent quote symbols supplied.")
    if not 1 <= batch_size <= 150:
        raise ValueError("batch_size must be between 1 and 150")

    session = requests.Session()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for offset in range(0, len(symbols), batch_size):
            batch = symbols[offset:offset + batch_size]
            url = TENCENT_QUOTE_URL + ",".join(batch)
            batch_rows: list[dict[str, Any]] = []
            last_error: str | None = None
            for attempt in range(3):
                try:
                    response = session.get(
                        url, headers=REQUEST_HEADERS, timeout=timeout_seconds
                    )
                    response.raise_for_status()
                    body = response.content.decode("gb18030", errors="replace")
                    batch_rows = parse_tencent_quote_text(body)
                    if not batch_rows:
                        raise RuntimeError("Tencent returned no parseable quote rows.")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < 2:
                        time.sleep(0.5 * (attempt + 1))
            if last_error:
                errors.append(
                    f"Tencent batch {offset // batch_size + 1}: {last_error}"[:400]
                )
            else:
                rows.extend(batch_rows)
            if pause_seconds > 0 and offset + batch_size < len(symbols):
                time.sleep(pause_seconds)
    finally:
        session.close()

    if not rows:
        raise RuntimeError(
            "Tencent quote endpoint returned no usable rows. " + " | ".join(errors[-3:])
        )
    return rows, errors


def fetch_baostock_universe() -> pd.DataFrame:
    """Use BaoStock's independent stock-basic endpoint as a code-universe fallback."""
    import baostock as bs

    login = bs.login()
    if getattr(login, "error_code", None) != "0":
        raise RuntimeError(
            f"BaoStock login failed: {getattr(login, 'error_code', '?')} "
            f"{getattr(login, 'error_msg', '')}"
        )
    try:
        result = bs.query_stock_basic()
        if getattr(result, "error_code", None) != "0":
            raise RuntimeError(
                f"BaoStock query_stock_basic failed: {result.error_code} {result.error_msg}"
            )
        rows: list[list[str]] = []
        fields = list(result.fields)
        while result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=fields)
    finally:
        bs.logout()

    required = {"code", "code_name", "type", "status"}
    if frame.empty or not required.issubset(frame.columns):
        raise RuntimeError(
            f"BaoStock stock universe schema invalid: {list(frame.columns)}"
        )
    frame = frame.loc[
        (frame["type"].astype(str) == "1")
        & (frame["status"].astype(str) == "1")
    ].copy()
    frame["code"] = frame["code"].astype(str).str.split(".").str[-1].str.zfill(6)
    frame["name"] = frame["code_name"].astype(str).str.strip()
    frame = frame.loc[frame["code"].str.match(r"^\d{6}$")]
    # Keep A-share prefixes only; exclude B shares, ETFs, indices and convertibles.
    frame = frame.loc[
        frame["code"].str.startswith(("0", "3", "6", "4", "8", "92"))
    ]
    frame = frame.drop_duplicates("code").reset_index(drop=True)
    if len(frame) < 1000:
        raise RuntimeError(
            f"BaoStock returned only {len(frame)} active stock records; "
            "refusing to use an incomplete universe."
        )
    return frame[["code", "name"]]


def fetch_baostock_history(
    symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Last-resort daily K-line fallback; returns raw columns accepted by normalize_history."""
    import baostock as bs

    code = str(symbol).strip().zfill(6)
    if code.startswith(("92", "4", "8")):
        market = "bj"
    elif code.startswith(("6", "5", "9")):
        market = "sh"
    else:
        market = "sz"
    bs_code = f"{market}.{code}"
    start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    login = bs.login()
    if getattr(login, "error_code", None) != "0":
        raise RuntimeError(
            f"BaoStock login failed: {getattr(login, 'error_code', '?')} "
            f"{getattr(login, 'error_msg', '')}"
        )
    try:
        result = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",
        )
        if getattr(result, "error_code", None) != "0":
            raise RuntimeError(
                f"BaoStock history failed for {bs_code}: "
                f"{result.error_code} {result.error_msg}"
            )
        rows: list[list[str]] = []
        fields = list(result.fields)
        while result.next():
            rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=fields)
    finally:
        bs.logout()
    if frame.empty:
        raise RuntimeError(f"BaoStock returned no daily bars for {bs_code}.")
    return frame
