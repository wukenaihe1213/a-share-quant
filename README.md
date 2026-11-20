# A-share Quant

个人 A 股量化研究与选股项目，使用 Python + AKShare 获取公开市场数据。

> 当前处于 **Phase 1：行情连通性验证**。先验证 GitHub Actions 云端运行环境能否访问 AKShare，再逐步增加历史行情指标、候选筛选、历史推荐复盘和公开 JSON 输出。

## 当前功能

- 获取指定股票实时盘口行情：`stock_bid_ask_em`（单股请求，避免测试时遍历全市场分页）
- 获取指定股票前复权日线：`stock_zh_a_hist_em`
- 对实时快照和历史日线执行基础字段检查
- 将测试结果保存为 GitHub Actions Artifact（不自动提交个人数据）

## 运行方式

1. 打开仓库的 **Actions** 页面。
2. 选择 **AKShare connectivity smoke test**。
3. 点击 **Run workflow**。
4. 股票代码默认是 `000001`（平安银行），也可以填写其他六位 A 股代码。
5. 运行结束后，在本次 workflow run 页面下载 `akshare-smoke-test` Artifact，查看 `smoke_test.json`。

本地运行需要 Python 3.11.5+：

```bash
python -m pip install -r requirements.txt
python -m src.akshare_smoke_test --symbol 000001
```

## 数据时间说明

- `collected_at` 是本程序采集数据的时间。
- 当前 AKShare 个股盘口输出不保证包含独立的交易所报价时间；程序将 `source_quote_timestamp` 标为 `null`，不会把采集时间冒充交易所报价时间。
- 日线数据可能在收盘前尚未形成完整当日 K 线。
- 数据接口依赖公开数据源，可能出现限流、字段变化或暂时不可用；失败时会保存错误状态，不会伪造数据。

## 安全与隐私

- 不要把 GitHub Token、密码、券商账号、个人持仓或交易记录提交到公开仓库。
- 后续公开发布的 JSON 只包含市场数据和技术指标，不包含个人身份或账户信息。

## 后续路线

- Phase 1：AKShare 连通性测试。
- Phase 2：历史日线、分钟线、指数与板块数据采集。
- Phase 3：趋势回调筛选、数据质量门槛与候选评分。
- Phase 4：推荐历史快照、10 个交易日复盘与表现统计。
- Phase 5：GitHub Pages/公开 JSON 发布，并验证 ChatGPT 读取链路。

## 数据来源与免责声明

本项目使用 [AKShare](https://github.com/akfamily/akshare) 获取公开财经数据。数据仅用于个人研究与技术验证，不构成投资建议。数据准确性、完整性和及时性受上游网站及网络条件影响。
