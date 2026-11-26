from __future__ import annotations

import inspect
import re
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEWS_LOOKBACK_HOURS = 48

GLOBAL_FEEDS = (
    ("eastmoney_global", "stock_info_global_em"),
    ("sina_global", "stock_info_global_sina"),
    ("cls_global", "stock_info_global_cls"),
    ("ths_global", "stock_info_global_ths"),
    ("futu_global", "stock_info_global_futu"),
    ("caixin_selected", "stock_news_main_cx"),
)

COLUMN_ALIASES = {
    "title": ("标题", "新闻标题", "资讯标题", "标题内容", "公告标题", "announcementTitle", "title", "headline"),
    "content": ("内容", "新闻内容", "摘要", "简要内容", "content", "summary", "digest"),
    "published_at": ("发布时间", "时间", "日期", "新闻时间", "公告时间", "announcementTime", "public_time", "datetime", "pub_time", "date"),
    "url": ("链接", "新闻链接", "公告链接", "url", "href", "link"),
    "code": ("代码", "股票代码", "证券代码", "code", "symbol"),
    "name": ("名称", "股票名称", "证券简称", "简称", "name"),
}

POSITIVE_KEYWORDS = (
    "业绩预增", "净利润预增", "业绩大增", "扭亏为盈", "超预期", "中标", "预中标",
    "重大合同", "新签订单", "订单增长", "回购", "增持", "获批", "核准", "投产",
    "扩产", "涨价", "供不应求", "政策支持", "补贴", "降息", "降准", "现金分红",
    "战略合作", "资产注入", "股权激励", "上调评级", "扭亏", "盈利增长",
)
NEGATIVE_KEYWORDS = (
    "业绩预亏", "预计亏损", "业绩大幅下降", "净利润下降", "业绩下滑", "预减",
    "大股东减持", "控股股东减持", "股东减持", "立案调查", "行政处罚", "监管处罚",
    "监管函", "问询函", "债务违约", "逾期债务", "重大诉讼", "重大仲裁", "财务造假",
    "商誉减值", "计提减值", "停产", "停业", "安全事故", "产品召回", "终止重组",
    "终止收购", "订单取消", "项目取消", "退市风险", "退市整理", "风险警示",
    "下修业绩", "价格大跌", "价格下跌", "暴跌",
)
HIGH_RISK_KEYWORDS = (
    "立案调查", "行政处罚", "监管处罚", "财务造假", "债务违约", "逾期债务",
    "退市风险", "退市整理", "重大安全事故", "重大诉讼", "重大仲裁",
    "业绩预亏", "预计亏损", "终止重组", "终止收购", "停产",
)
CATEGORY_KEYWORDS = {
    "earnings": ("业绩", "净利润", "营收", "预增", "预亏", "扭亏", "财报", "快报"),
    "contract_order": ("中标", "合同", "订单", "采购", "项目", "交付"),
    "capital_action": ("回购", "增持", "减持", "质押", "解禁", "股权激励", "分红"),
    "regulatory": ("监管", "立案", "处罚", "问询", "警示", "退市", "合规"),
    "litigation": ("诉讼", "仲裁", "违约", "债务", "担保"),
    "operations": ("停产", "投产", "扩产", "事故", "召回", "涨价", "降价"),
    "policy": ("政策", "补贴", "规划", "意见", "办法", "条例", "试点"),
    "macro_rates_fx": ("降息", "加息", "降准", "利率", "汇率", "央行", "美联储"),
    "commodities": ("原油", "黄金", "铜价", "铝价", "煤价", "天然气", "锂价", "碳酸锂"),
    "geopolitics": ("制裁", "关税", "贸易战", "冲突", "战争", "出口管制"),
}


def _column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _clean_text(value: Any, limit: int = 2000) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(SHANGHAI)
        else:
            parsed = parsed.tz_convert(SHANGHAI)
        return parsed.to_pydatetime()
    except Exception:
        return None


def classify_news(title: str, content: str = "") -> dict[str, Any]:
    text = f"{title} {content}".lower()
    positive = [word for word in POSITIVE_KEYWORDS if word.lower() in text]
    negative = [word for word in NEGATIVE_KEYWORDS if word.lower() in text]
    high_risk = [word for word in HIGH_RISK_KEYWORDS if word.lower() in text]
    categories = [
        category for category, words in CATEGORY_KEYWORDS.items()
        if any(word.lower() in text for word in words)
    ]
    raw_score = len(positive) - len(negative)
    if high_risk:
        raw_score = min(raw_score, -3)
    score = max(-5, min(5, raw_score))
    if positive and negative:
        label = "mixed"
    elif score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "unclassified"
    return {
        "sentiment_label": label,
        "sentiment_score": score,
        "positive_hits": positive,
        "negative_hits": negative,
        "risk_flags": high_risk,
        "event_categories": categories,
        "classification_method": "transparent_keyword_heuristic_v1",
    }


def normalize_news_frame(
    frame: pd.DataFrame | None,
    source: str,
    scope: str,
    now: datetime | None = None,
    default_code: str | None = None,
    default_name: str | None = None,
    max_items: int = 100,
) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    columns = {
        key: _column(frame, aliases)
        for key, aliases in COLUMN_ALIASES.items()
    }
    if columns["title"] is None:
        return []
    current = now or datetime.now(SHANGHAI)
    if current.tzinfo is None:
        current = current.replace(tzinfo=SHANGHAI)
    rows: list[dict[str, Any]] = []
    for _, row in frame.head(max_items).iterrows():
        title = _clean_text(row[columns["title"]], 500)
        if not title:
            continue
        content = _clean_text(row[columns["content"]], 1800) if columns["content"] else ""
        raw_time = _clean_text(row[columns["published_at"]], 100) if columns["published_at"] else ""
        parsed_time = _parse_time(raw_time)
        code = _clean_text(row[columns["code"]], 20) if columns["code"] else default_code or ""
        name = _clean_text(row[columns["name"]], 80) if columns["name"] else default_name or ""
        url = _clean_text(row[columns["url"]], 1000) if columns["url"] else ""
        classification = classify_news(title, content)
        age_hours = (
            round((current - parsed_time).total_seconds() / 3600, 2)
            if parsed_time else None
        )
        rows.append({
            "scope": scope,
            "code": code or None,
            "name": name or None,
            "source": source,
            "title": title,
            "content": content,
            "published_at": parsed_time.isoformat(timespec="seconds") if parsed_time else raw_time or None,
            "published_time_verified": parsed_time is not None,
            "age_hours": age_hours,
            "is_recent_48h": bool(age_hours is not None and -0.25 <= age_hours <= NEWS_LOOKBACK_HOURS),
            "url": url or None,
            **classification,
        })
    return rows


def _call_global_feed(interface_name: str) -> pd.DataFrame:
    function = getattr(ak, interface_name, None)
    if not callable(function):
        raise RuntimeError(f"{interface_name}: interface unavailable in installed AKShare")
    frame = function()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{interface_name}: expected DataFrame, got {type(frame).__name__}")
    return frame


def _call_stock_news(code: str) -> pd.DataFrame:
    function = getattr(ak, "stock_news_em", None)
    if not callable(function):
        raise RuntimeError("stock_news_em: interface unavailable in installed AKShare")
    try:
        params = inspect.signature(function).parameters
    except (TypeError, ValueError):
        params = {}
    if "stock" in params:
        frame = function(stock=code)
    elif "symbol" in params:
        frame = function(symbol=code)
    else:
        frame = function(code)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"stock_news_em: expected DataFrame, got {type(frame).__name__}")
    return frame


def _call_cninfo_disclosures(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    function = getattr(ak, "stock_zh_a_disclosure_report_cninfo", None)
    if not callable(function):
        raise RuntimeError("stock_zh_a_disclosure_report_cninfo: interface unavailable")
    frame = function(
        symbol=code,
        market="沪深京",
        keyword="",
        category="",
        start_date=start_date,
        end_date=end_date,
    )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("stock_zh_a_disclosure_report_cninfo: expected DataFrame")
    return frame


def _deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        title = re.sub(r"[\W_]+", "", str(item.get("title", "")).lower())
        published = str(item.get("published_at") or "")[:10]
        key = f"{title}|{published}"
        if not title:
            continue
        if key not in grouped:
            grouped[key] = dict(item)
            grouped[key]["sources"] = [item.get("source")] if item.get("source") else []
            continue
        current = grouped[key]
        if item.get("source") and item["source"] not in current["sources"]:
            current["sources"].append(item["source"])
        if len(item.get("content") or "") > len(current.get("content") or ""):
            current["content"] = item["content"]
        if not current.get("url") and item.get("url"):
            current["url"] = item["url"]
        if item.get("risk_flags"):
            current["risk_flags"] = sorted(set(current.get("risk_flags", []) + item["risk_flags"]))
        current["is_recent_48h"] = current.get("is_recent_48h", False) or item.get("is_recent_48h", False)
    return sorted(
        grouped.values(),
        key=lambda item: (item.get("published_at") or ""),
        reverse=True,
    )


def summarize_news(items: list[dict[str, Any]], scanned: bool = True) -> dict[str, Any]:
    recent = [item for item in items if item.get("is_recent_48h")]
    scores = [int(item.get("sentiment_score", 0)) for item in recent]
    positive_count = sum(1 for item in recent if item.get("sentiment_label") == "positive")
    negative_count = sum(1 for item in recent if item.get("sentiment_label") == "negative")
    mixed_count = sum(1 for item in recent if item.get("sentiment_label") == "mixed")
    risk_flags = sorted({
        flag for item in recent for flag in item.get("risk_flags", [])
    })
    avg_score = round(sum(scores) / len(scores), 3) if scores else None
    if not recent:
        label = "no_recent_news" if scanned else "not_scanned"
    elif positive_count and negative_count:
        label = "mixed"
    elif avg_score is not None and avg_score >= 0.5:
        label = "positive"
    elif avg_score is not None and avg_score <= -0.5:
        label = "negative"
    else:
        label = "neutral_or_unclear"
    sources = sorted({
        source for item in recent
        for source in (item.get("sources") or [item.get("source")])
        if source
    })
    latest = next((item.get("published_at") for item in recent if item.get("published_at")), None)
    return {
        "status": "ok" if recent else ("no_recent_news" if scanned else "not_scanned"),
        "sentiment_label": label,
        "average_keyword_score": avg_score,
        "recent_news_count_48h": len(recent),
        "positive_news_count": positive_count,
        "negative_news_count": negative_count,
        "mixed_news_count": mixed_count,
        "risk_alert": bool(risk_flags),
        "risk_flags": risk_flags,
        "latest_news_at": latest,
        "sources": sources,
        "headlines": [item.get("title") for item in recent[:5]],
        "classification_method": "transparent_keyword_heuristic_v1",
        "use_as_standalone_buy_signal": False,
    }


def collect_news(
    candidates: list[dict[str, Any]],
    max_stock_symbols: int = 20,
    delay_seconds: float = 0.25,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Collect multi-source market headlines, candidate news and CNINFO disclosures.

    This module is intentionally additive: news does not change the six technical
    strategy scores or ranking until enough out-of-sample evidence supports a change.
    """
    current = now or datetime.now(SHANGHAI)
    if current.tzinfo is None:
        current = current.replace(tzinfo=SHANGHAI)
    start_date = (current - timedelta(days=7)).strftime("%Y%m%d")
    end_date = current.strftime("%Y%m%d")
    errors: list[str] = []
    provider_status: dict[str, dict[str, Any]] = {}
    all_items: list[dict[str, Any]] = []

    for source, interface_name in GLOBAL_FEEDS:
        try:
            frame = _call_global_feed(interface_name)
            normalized = normalize_news_frame(
                frame, source=source, scope="market", now=current, max_items=120
            )
            all_items.extend(normalized)
            provider_status[source] = {
                "status": "ok" if normalized else "empty",
                "interface": interface_name,
                "items_received": len(normalized),
            }
            print(f"[NEWS] {interface_name}: {len(normalized)} rows")
        except Exception as exc:
            message = f"{interface_name}: {type(exc).__name__}: {exc}"
            errors.append(message[:500])
            provider_status[source] = {
                "status": "error",
                "interface": interface_name,
                "error": message[:500],
            }
            print(f"[NEWS][WARN] {message[:300]}")

    candidate_news: dict[str, dict[str, Any]] = {}
    candidate_items: list[dict[str, Any]] = []
    scanned_candidates = candidates[:max_stock_symbols]
    for index, candidate in enumerate(scanned_candidates):
        code = str(candidate.get("code") or "").zfill(6)
        name = str(candidate.get("name") or "")
        stock_items: list[dict[str, Any]] = []
        if index and delay_seconds > 0:
            time.sleep(delay_seconds)
        try:
            frame = _call_stock_news(code)
            stock_items.extend(normalize_news_frame(
                frame, source="eastmoney_stock_news", scope="stock",
                now=current, default_code=code, default_name=name, max_items=30,
            ))
        except Exception as exc:
            message = f"stock_news_em({code}): {type(exc).__name__}: {exc}"
            errors.append(message[:500])
            print(f"[NEWS][WARN] {message[:300]}")
        try:
            frame = _call_cninfo_disclosures(code, start_date, end_date)
            stock_items.extend(normalize_news_frame(
                frame, source="cninfo_disclosure", scope="disclosure",
                now=current, default_code=code, default_name=name, max_items=50,
            ))
        except Exception as exc:
            message = f"stock_zh_a_disclosure_report_cninfo({code}): {type(exc).__name__}: {exc}"
            errors.append(message[:500])
            print(f"[NEWS][WARN] {message[:300]}")
        deduped = _deduplicate(stock_items)
        candidate_items.extend(deduped)
        candidate_news[code] = summarize_news(deduped, scanned=True)
        candidate_news[code]["name"] = name
        candidate_news[code]["scanned"] = True

    for candidate in candidates[max_stock_symbols:]:
        code = str(candidate.get("code") or "").zfill(6)
        candidate_news[code] = summarize_news([], scanned=False)
        candidate_news[code]["name"] = str(candidate.get("name") or "")
        candidate_news[code]["scanned"] = False

    market_items = _deduplicate([item for item in all_items if item.get("is_recent_48h")])
    global_summary = summarize_news(market_items, scanned=bool(provider_status))
    global_risk = [
        item for item in market_items if item.get("risk_flags")
    ][:20]
    candidate_items = _deduplicate(candidate_items)
    candidate_success = sum(
        1 for summary in candidate_news.values()
        if summary.get("status") == "ok"
    )
    successful_global = sum(
        1 for item in provider_status.values() if item.get("status") == "ok"
    )
    status = "ok" if successful_global and candidate_success else (
        "partial" if successful_global or candidate_success else "unavailable"
    )
    return {
        "schema_version": 1,
        "status": status,
        "collected_at": current.isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "lookback_hours": NEWS_LOOKBACK_HOURS,
        "sentiment_method": "transparent_keyword_heuristic_v1",
        "sentiment_method_note": (
            "Keyword classification is an explainable triage aid, not a verified semantic "
            "model. It does not change technical scores or make buy/sell decisions."
        ),
        "provider_status": provider_status,
        "coverage": {
            "global_providers_configured": len(GLOBAL_FEEDS),
            "global_providers_successful": successful_global,
            "global_recent_news_count": len(market_items),
            "candidate_symbols_requested": len(scanned_candidates),
            "candidate_symbols_with_recent_news": candidate_success,
            "candidate_symbols_total": len(candidates),
            "candidate_news_items": len(candidate_items),
        },
        "market_sentiment": global_summary,
        "market_risk_headlines": global_risk,
        "market_news": market_items[:150],
        "candidate_news": candidate_news,
        "candidate_news_items": candidate_items[:500],
        "errors": errors[-40:],
        "limitations": [
            "News sources are public web interfaces accessed through AKShare; they may be delayed, rate-limited, changed or unavailable and have no guaranteed SLA.",
            "Only the highest-ranked candidate symbols are queried individually; unqueried candidates are explicitly marked not_scanned.",
            "Only news with a parseable publication time within the last 48 hours contributes to the recent-news summary; undated items are not treated as fresh.",
            "Keyword sentiment can miss context, negation, sarcasm and event magnitude; every material negative or positive claim should be checked against the linked original disclosure or issuer/exchange source.",
            "News sentiment is reported as an independent context/risk dimension and does not modify the existing technical scores or candidate ranking.",
        ],
    }
