# nof2 
参考nof1设计的AI自主分析交易机器人
***注意，请设置币安合约为逐仓、单币保证金模式
## 项目简介

本项目是一个基于人工智能 (AI) 的自动化加密货币合约交易机器人。它利用大型语言模型 (LLM)，结合市场数据（价格、K 线）和技术指标（如 EMA、RSI、MACD），自主分析行情、做出交易决策（开仓、平仓、持仓），并通过 CCXT 库与交易所 API（目前主要针对币安 Binance）交互，执行实盘或模拟盘交易。

系统设计支持同时监控和操作多个 USDT 本位永续合约交易对。内置严格的风险管理机制，强制 AI 遵循基于可用现金百分比的仓位大小计算规则，并在下单前进行服务器端验证，防止单笔保证金超出预设风险限制。

项目核心功能包括：

* **AI 驱动决策**: 集成 AI 模型 (支持 OpenAI, Azure OpenAI, DeepSeek 等兼容 OpenAI API 的服务) 进行分析和下单。
* **多币种交易**: 可配置同时监控和交易多个交易对。
* **风险管理**: 强制仓位大小计算，服务器端验证保证金限制，自动缩减超额仓位。
* **实盘/模拟盘**: 通过配置文件 (`config.py`) 灵活切换真实交易与模拟回测。
* **状态持久化**: 实盘模式下，持仓状态 (`alpha_live_positions_v2.json`) 和交易历史 (`alpha_live_trades.json`) 会自动保存到本地文件，方便重启后恢复。
* **Web 仪表盘**: 提供一个简单的网页界面 (`alpha_web_server.py`)，用于实时监控账户净值、持仓、交易历史和运行日志。
* **实时通知**: 支持通过 Bark 发送开平仓、错误等关键事件的实时推送通知。
  
## 策略简述

这是一个**单一策略**模型，完全由 AI (LLM) 驱动进行**限价单 (Limit Order)** 交易提议，并由 Python 执行严格的风险管理和 Veto（否决）规则。

### 1. 核心入场策略 (AI 提议)

AI (LLM) 负责分析市场，并根据市场状态选择两种策略之一：

* **策略 A: 趋势回调 (ADX > 20)**
    * **规则:** 当 1h ADX > 20 时，市场处于趋势中。
    * **动作:** AI 会寻找“回调汇合区” (Pullback Confluence Zone) 来设置 `LIMIT_BUY` (牛市) 或 `LIMIT_SELL` (熊市)。
    * **过滤器:** 此策略**必须**顺从 `Rule 1.A` (4h EMA 宏观趋势)。

* **策略 B: 震荡均值回归 (ADX < 20)**
    * **规则:** 当 1h ADX < 20 时，市场处于震荡中。
    * **动作:** AI 会在 `BB_Lower` (布林带下轨) 提议 `LIMIT_BUY`，或在 `BB_Upper` (上轨) 提议 `LIMIT_SELL`。
    * **过滤器:** 此策略**豁免** 4h EMA 宏观趋势的限制。

* **策略 C: 加仓 (Pyramiding)**
    * **规则:** AI 被授权分析已盈利的仓位 (例如 > 1R)。
    * **动作:** 如果趋势仍然强劲且价格出现回调，AI 可以提议新的 `LIMIT_BUY` / `LIMIT_SELL` 作为加仓。

---

### 2. 关键入场过滤器

在 AI 提议后，Python代码 会进行最终审核：

1.  **数据确认 (Rule 3):** AI 提议必须得到 `OI_Regime_1h` (持仓量) 和 `Taker_Ratio_1h_Regime` (主动成交) 的数据支持。
2.  **情绪过滤 (Rule 5):** 结合 F&G 情绪指数和 4h EMA 趋势。例如，在牛市中 (Price > 4h EMA)，"极度恐惧" (F&G < 25) 反而是**高信心买入信号**。
3.  **R:R 检查 (Rule 4.5):** AI 必须确保提议满足 1.5 R:R。Python 会在 `_validate_ai_trade` 中再次计算并否决不合格的订单。
4.  **趋势 Veto:** Python 确认 4h EMA 过滤器**仅**在 `1h ADX > 20` (趋势市) 时才生效。
5.  **异常 Veto:** `Anomaly_Score` (ML 异常检测) 必须 > -0.1 (即 "Safe")。
6.  **过时 Veto:**
    * `Stale Plan Veto`: 否决 AI 因数据延迟而提议的、会**立即成交**的限价单。
    * `Stale Order Veto`: 自动取消已挂出但**价格偏离现价过远**的限价单。

---

### 3. 止盈逻辑

AI 不参与止盈，Python 通过高频 (2s) 和低频 (10s) 循环执行：

* **阶梯止盈:**
    * `high_frequency_risk_loop` (2s)。
    * 一旦盈利超过 1% (`current_peak_rate >= 0.01`)，立即启动。
    * 止损点会**紧跟**峰值利润的 60% 至 80%，高频锁定利润。

* **原始止盈:**
    * `high_frequency_risk_loop` (2s)。
    * AI 提议的 `take_profit_price` 会作为最终的硬止盈目标。

* **1R 盈亏平衡:**
    * `high_frequency_risk_loop` (2s)。
    * 当利润达到 1R (即浮盈 = 初始风险 `entry - initial_sl`) 时，止损**立即移动到成本价** (Breakeven)。

---

### 4. 止损逻辑

* **AI 初始止损:** AI 提议的 `stop_loss_price` 作为初始硬止损。
* **AI 更新止损:**
    * AI 被授权分析**亏损中**的仓位。
    * 如果 AI 认为市场结构已变坏 (例如 15m MACD 交叉)，它可以提议 `UPDATE_STOPLOSS` 来**收紧止损**，提前离场。
* **动态安全网:**
    * 有 `15 分钟` 的入场宽限期 (`SAFETY_NET_GRACE_PERIOD_MS`)。
    * **ADX 过滤:** 此安全网**仅**在趋势市 (`1h ADX > 20`) 中启动，在震荡市中 (ADX < 20) **禁用**。
    * **触发:** 如果价格 (多头) 跌破 `1h EMA 50 - (1.5 * 1h ATR)` 的缓冲带，立即平仓。
* **ATR 追踪止损 (盈利时):**
    * 同样有 `15 分钟` 宽限期。
    * 使用 `2.0 * 15min_atr_14` 作为追踪止损，跟随盈利。

## 使用方式

### 1. 下载与安装

```bash
git clone https://github.com/hzy11fk/nof2.git
cd nof2

# (推荐) 创建并激活 Python 虚拟环境
python -m venv .venv
Windows:
.venv\Scripts\activate
Linux / macOS:
source .venv/bin/activate

# 安装依赖库
pip install -r requirements.txt

# 在.env中填入对应信息
vi .env

# 运行程序
python run_alpha.py
```
## 文件介绍

* **`run_alpha.py`** 程序入口，初始化并启动 `AlphaTrader`。
* **`config.py`**: 定义所有配置参数，加载 `.env` 文件。
* **`.env`**: 存储 API 密钥等敏感信息。
* **`alpha_trader.py`**: 核心交易逻辑和 AI 交互控制器。
* **`alpha_portfolio.py`**: 账户状态管理器，实盘/模拟盘交易执行的桥梁。
* **`alpha_ai_analyzer.py`**: 与 AI 大模型 API 通信的模块。
* **`exchange_client.py`**: 封装 CCXT 交易所 API 调用，增加重试逻辑。
* **`alpha_position_manager.py`**: 管理和持久化**实盘**持仓状态 (`alpha_live_positions_v2.json`)。
* **`alpha_trade_logger.py`**: 记录和持久化**实盘**交易历史 (`alpha_live_trades.json`)。
* **`alpha_web_server.py`**: (可选) 提供 Web 监控界面。
* **`bark_notifier.py`** (可选) 发送 Bark 通知的工具函数。
* **`requirements.txt`**: 项目所需的 Python 依赖库列表。
* **`train_anomaly_detector.py`**: 趋势策略相关数据集训练脚本。
* **`train_rule8_model.py`**: 突破策略相关数据集训练脚本。
* **`fetch_binance_data.py`**: 训练集所需文件下载脚本。
* **`/models`**: 训练完成的数据。


## 免责声明
本项目仅供学习和研究目的，作者不对任何使用本项目代码造成的实际交易盈亏负责。请在充分了解风险的情况下使用。
