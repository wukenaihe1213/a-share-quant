import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

from src.news_sentiment import (
    classify_news,
    normalize_news_frame,
    summarize_news,
    collect_news,
    GLOBAL_FEEDS,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


class NewsSentimentTests(unittest.TestCase):
    def test_classify_positive_contract_and_earnings_news(self):
        result = classify_news("公司中标重大合同，业绩预增", "新签订单金额显著增长")
        self.assertEqual(result["sentiment_label"], "positive")
        self.assertGreater(result["sentiment_score"], 0)
        self.assertIn("contract_order", result["event_categories"])
        self.assertIn("earnings", result["event_categories"])

    def test_classify_high_risk_negative_news(self):
        result = classify_news("公司因财务造假被立案调查")
        self.assertEqual(result["sentiment_label"], "negative")
        self.assertTrue(result["risk_flags"])
        self.assertLessEqual(result["sentiment_score"], -3)

    def test_mixed_news_is_not_forced_positive_or_negative(self):
        result = classify_news("公司业绩预增，但控股股东计划减持")
        self.assertEqual(result["sentiment_label"], "mixed")

    def test_normalize_news_preserves_source_time_and_url(self):
        now = datetime(2026, 11, 26, 14, 30, tzinfo=SHANGHAI)
        frame = pd.DataFrame([{
            "标题": "公司中标项目",
            "内容": "公司公告称中标重大项目",
            "发布时间": "2026-11-26 13:55:00",
            "链接": "https://example.com/news/1",
        }])
        rows = normalize_news_frame(
            frame, source="test_source", scope="market", now=now
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "test_source")
        self.assertTrue(rows[0]["is_recent_48h"])
        self.assertEqual(rows[0]["url"], "https://example.com/news/1")

    def test_parse_compact_and_epoch_news_timestamps(self):
        from src.news_sentiment import _parse_time
        compact = _parse_time("20261126143000")
        epoch_ms = _parse_time("1795674600000")
        self.assertIsNotNone(compact)
        self.assertEqual(compact.year, 2026)
        self.assertIsNotNone(epoch_ms)
        self.assertEqual(epoch_ms.year, 2026)

    def test_summary_marks_unscanned_candidates(self):
        summary = summarize_news([], scanned=False)
        self.assertEqual(summary["status"], "not_scanned")
        self.assertEqual(summary["sentiment_label"], "not_scanned")

    def test_summary_does_not_call_no_recent_news_positive_or_negative(self):
        summary = summarize_news([], scanned=True)
        self.assertEqual(summary["sentiment_label"], "no_recent_news")
        self.assertIsNone(summary["average_keyword_score"])


    def test_global_feed_names_match_current_akshare_interfaces(self):
        names = [interface for _, interface in GLOBAL_FEEDS]
        self.assertEqual(
            names,
            [
                "stock_news_global_em",
                "stock_news_global_sina",
                "stock_news_global_cls",
                "stock_news_global_ths",
                "stock_news_global_futu",
                "stock_news_main_cx",
            ],
        )


    def test_global_feed_functions_exist_in_installed_akshare(self):
        for _, interface_name in GLOBAL_FEEDS:
            self.assertTrue(
                callable(getattr(ak, interface_name, None)),
                f"AKShare interface missing: {interface_name}",
            )

    def test_collect_news_handles_all_providers_unavailable(self):
        # A network-free safety check: the function must still return a structured
        # unavailable result when the upstream functions are missing or raise.
        class Candidate(dict):
            pass
        with patch("src.news_sentiment._call_global_feed", side_effect=RuntimeError("offline")):
            result = collect_news(
                [{"code": "600000", "name": "测试股"}],
                max_stock_symbols=0,
                delay_seconds=0,
                now=datetime(2026, 11, 26, 14, 30, tzinfo=SHANGHAI),
            )
        self.assertIn(result["status"], {"unavailable", "partial", "ok"})
        self.assertIn("candidate_news", result)
        self.assertEqual(result["coverage"]["candidate_symbols_requested"], 0)
        self.assertEqual(result["candidate_news"]["600000"]["status"], "not_scanned")


if __name__ == "__main__":
    unittest.main()
