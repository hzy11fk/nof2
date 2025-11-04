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
  
## 策略介绍

### 核心架构：AI 提议，Python 执行

本系统是一个**单一策略**架构，完全专注于“AI 战略限价单” (Rule 6)。

* **AI (L3) - 策略师 (The Strategist)：**
    * `alpha_trader.py` 中的 AI (LLM) **只负责一件事**：分析市场数据，并**提议 (Propose)** 高质量的限价单交易计划（`entry_price`, `stop_loss_price`, `take_profit_price`）。
* **Python (L1) - 执行者 (The Executor)：**
    * Python 负责**所有**的计算、验证和风险管理。
    * AI **没有**平仓权限。
    * AI **没有**杠杆控制权。
    * AI **没有**最终决定权。如果 AI 的提议不符合 Python 的硬性 Veto 规则，该提议将被**静默否决**。

---

### 入场逻辑

AI (LLM) 必须在其 `SYSTEM_PROMPT_TEMPLATE` 内部**预先通过**以下所有检查，才能生成一个交易提议：

1.  **AI 状态检查 (Prompt)：**
    * **禁止对冲/重复：** 必须跳过任何已有持仓或已有挂单的币种。
2.  **AI 策略检查 (Prompt)：**
    * **市场状态 (Rule 2)：** 必须根据 `ADX` 选择正确的策略（趋势回调 或震荡均值回归）。
    * **信念确认 (Rule 3)：** 必须得到 `OI_Regime_1h` 和 `Taker_Ratio_1h_Regime` 的双重数据确认。
    * **R:R 预检查 (Rule 4.5)：** **AI 必须自己计算 R:R**。如果一个 1.5R 的止盈目标不切实际（例如被关键S/R阻挡），AI 必须**自我否决 (ABORT)** 这个计划。
3.  **AI 情绪检查 (Prompt)：**
    * **情绪指导 (Rule 5)：** AI 被**强烈不鼓励**在“极度贪婪”时买入，或在“极度恐惧”时卖出。

#### Python 的最终 Veto（否决）层

即使 AI 提议了一个交易，`alpha_trader.py` 的 Python 执行层也会在 `start()` 和 `_execute_decisions` 中进行**最终的、强制性**检查：

1.  **过时订单 Veto (Stale Limit Order Veto)：**
    * `start()` 循环（10 秒）会**自动取消**任何已挂起、但价格偏离现价过远（`AI_LIMIT_ORDER_DEVIATION_PERCENT`）的订单。
2.  **过时计划 Veto (Stale Plan Veto)：**
    * 在 `_execute_decisions` 中，在下单前**立即获取**最新市价（`fresh_price`）。
    * 如果 AI 的限价单会**立即成交**（即 AI 的数据已过时），订单将被**否决**。
3.  **技术 Veto (`_validate_ai_trade`)：**
    * **异常 Veto：** `Anomaly_Score` 必须为 "Safe"。
    * **趋势 Veto (可配置)：** 如果 `ENABLE_4H_EMA_FILTER` 为 `True`，交易必须顺从 4h EMA。
    * **OI 矩阵 Veto：** 必须通过 P↑O↓（轧空）或 P↓O↓（多杀多）检查。
    * **R:R Veto：** Python 会**再次**计算 R:R，如果低于 1.5，将执行最终否决。

### 风险与退出逻辑

**AI 不参与任何退出决策。** 退出 100% 由 `alpha_trader.py` 和 `alpha_portfolio.py` 中的 Python 规则自动执行。

#### A. 初始止损

* **AI 提议：** AI 必须在 `Rule 4` 中提议一个基于 ATR 的 `stop_loss_price`。

#### B. 针对【亏损中】仓位的退出规则 (安全网)

1.  **AI 初始止损 (硬止损)：**
    * `high_frequency_risk_loop`（2 秒） 检查价格是否触及 AI 设定的 `ai_suggested_stop_loss`。
2.  **V3 动态安全网 (结构止损)：**
    * `start()` 循环（10 秒）检查。如果价格跌破 `1hour_ema_50` 并减去一个 `0.5 * 1hour_atr_14` 的“缓冲带”，立即平仓。

#### C. 针对【盈利中】仓位的退出规则 (利润锁定)

1.  **1R 盈亏平衡：**
    * `high_frequency_risk_loop` 检查。一旦浮盈超过初始风险 (`entry - initial_sl`)，止损立即移至**开仓成本价**。
2.  **阶梯止盈 (主要方式)：**
    * `high_frequency_risk_loop` 检查。
    * 从 `1%` 利润开始，分三阶段（1-2.5%, 2.5-5%, >5%）逐步收紧追踪止损，锁定 60% 至 80% 的峰值利润。
3.  **15m ATR 追踪止损：**
    * `start()` 循环（10 秒） 检查。如果盈利，一个相对激进的 `2.0 * 15min_atr_14` 止损会跟在价格后面。

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
