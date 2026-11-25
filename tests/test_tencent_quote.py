import unittest

from src.tencent_quote import market_code_to_tencent, parse_tencent_quote_text


class TencentQuoteTests(unittest.TestCase):
    def make_quote_text(self):
        fields = [""] * 52
        fields[0] = "1"
        fields[1] = "浦发银行"
        fields[2] = "600000"
        fields[3] = "10.12"
        fields[4] = "10.07"
        fields[5] = "10.05"
        fields[6] = "100"
        fields[30] = "20261125143000"
        fields[31] = "0.05"
        fields[32] = "0.50"
        fields[33] = "10.20"
        fields[34] = "10.01"
        fields[36] = "100"
        fields[37] = "104629"
        fields[38] = "0.33"
        fields[47] = "0.42"
        return 'v_sh600000="' + "~".join(fields) + '";'

    def test_parse_tencent_quote_and_convert_amount_from_wan_to_yuan(self):
        rows = parse_tencent_quote_text(self.make_quote_text())

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["code"], "600000")
        self.assertEqual(row["name"], "浦发银行")
        self.assertEqual(row["price"], 10.12)
        self.assertEqual(row["change_pct"], 0.50)
        self.assertEqual(row["amount"], 1_046_290_000)
        self.assertEqual(row["quote_timestamp"], "20261125143000")
        self.assertEqual(row["volume_ratio"], 0.42)

    def test_invalid_or_empty_tencent_rows_are_ignored(self):
        text = 'v_sh600000="1~浦发银行~600000~0~10~10";'
        self.assertEqual(parse_tencent_quote_text(text), [])

    def test_baostock_and_plain_codes_map_to_tencent_symbols(self):
        self.assertEqual(market_code_to_tencent("sh.600000"), "sh600000")
        self.assertEqual(market_code_to_tencent("sz.000001"), "sz000001")
        self.assertEqual(market_code_to_tencent("bj.920001"), "bj920001")
        self.assertEqual(market_code_to_tencent("600000"), "sh600000")
        self.assertEqual(market_code_to_tencent("000001"), "sz000001")
        self.assertEqual(market_code_to_tencent("920001"), "bj920001")


if __name__ == "__main__":
    unittest.main()
