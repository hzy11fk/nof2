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
# 策略库总结

本系统是一个“混合”交易架构，结合了两种截然不同的策略：一种是高频、机械性的 Python 突破策略；另一种是低频、高上下文的 AI (LLM) 战略策略。

---

## 策略一：高频突破策略 (Python 驱动)

* **策略目标：** 捕捉由“K线收盘价” 确认的早期趋势突破，旨在规避“毛刺”和“假突破”。
* **执行层级：** L1 (Python) + L2 (ML 模型)。
* **执行频率：** `_check_python_rule_8` 在主循环中（每 10 秒） 检查一次入场信号。
* **订单类型：** 市价单 (Market Order)。

## 入场逻辑

系统必须**依次通过**以下所有过滤器才能执行交易：

1.  **状态过滤器：**
    * **禁止对冲：** 该币种不得有任何**已存在**的持仓 (`open_positions`)。
    * **避免冲突：** AI 策略不得有**已挂起**的限价单 (`pending_limit_orders`)。

2.  **形态过滤器：**
    * **市场挤压 (Squeeze)：** `1hour_adx_14` (1小时 ADX) 必须小于 25，表明市场处于低波动、等待突破的状态。
    * **成交量确认：** `5min_volume_ratio` 或 `15min_volume_ratio` 必须激增（> 2.0 或 2.5），确认突破有“能量”支持。

3.  **信号触发器 (K线收盘价确认)：**
    * **做多：** 上一根 `15min` K线的**收盘价** (`close_prev`) 必须**高于**上一根 `15min` K线的**布林带上轨** (`bb_upper_prev`)。
    * **做空：** 上一根 `15min` K线的**收盘价** (`close_prev`) 必须**低于**上一根 `15min` K线的**布林带下轨** (`bb_lower_prev`)。

4.  **最终否决 (Veto) 过滤器：**
    * **趋势否决：** 必须顺着大趋势（`4hour_ema_50`） 交易。
    * **情绪否决：** 不得在“极度贪婪”（> 75） 时做多，也不得在“极度恐惧”（< 25） 时做空。
    * **背离否决：** `_check_divergence` 函数不得检测到与交易方向相反的 RSI 背离。
    * **ML 否决：** `RandomForest` (L2) 模型 必须对该方向有高置信度（例如 `ML_Proba_UP > 0.65`）。

### 加仓逻辑

* **不允许。** 此策略被硬编码为**仅开新仓**。

### 止盈止损 (全自动)

此策略的风险由 `high_frequency_risk_loop`（高频风控循环，2 秒周期） **自动**管理：

* **止损 (Stop Loss)：**
    1.  **初始止损：** 入场时，`_build_python_order` 会根据 `1h_atr_14` (1小时 ATR) 设置一个较宽的初始止损。
    2.  **追踪止损 (主要退出方式)：** 仓位建立后，`high_frequency_risk_loop` 会根据 `BREAKOUT_TRAIL_STOP_PERCENT`（例如 1.5%）高频更新止损位，**自动锁定利润**。
* **止盈 (Take Profit)：**
    * **没有固定止盈。** `take_profit` 被设为 `None`，以激活上述的追踪止损。

---

## 策略二：AI 战略限价单 (LLM 驱动)

* **策略目标：** 作为“投资组合经理”，在市场处于不同状态（趋势、震荡）时，通过 L2 专家顾问（ML 模型） 的辅助，执行高置信度的交易。
* **执行层级：** L3 (AI / LLM)。
* **执行频率：** 低频（例如每 1-5 分钟，或由高级事件触发）。
* **订单类型：** 限价单 (Limit Order)。

### 入场逻辑

AI 在寻找任何新机会前，必须**首先**通过**全局 VETO 检查**：

1.  **Anomaly Veto (Rule 1.5)：** `Anomaly_Score` (来自 `IsolationForest`) 不得为“High Risk”(< -0.1)。
2.  **No Hedging (禁止对冲)：** 不得为已有持仓的币种开反向单。
3.  **No Duplicate Orders (禁止重复)：** 不得为已有挂单的币种挂新单。
4.  **Signal Veto (动能否决)：** 15m 动量不得与 4h 趋势相反。

通过 VETO 后，AI (LLM) 将根据市场状态选择以下策略之一：

#### A. 趋势策略 (Rule 6.1: 回调)

* **适用指标：** `1h ADX` 或 `4h ADX` > 25（市场处于趋势中）。
* **入场逻辑 (多重确认)：**
    1.  **区域 (Zone)：** AI 必须找到一个由**至少两个**S/R指标（例如 `1h EMA 20` 和 `4h BB_Mid`）重叠而成的**“共振支撑/阻力区”**。
    2.  **时机 (Timing)：** `15m RSI` 或 `1h RSI` 必须回撤到“复位”水平（例如，在上涨趋势中回调至 `RSI < 40`）。
    3.  **ML 确认:** `RandomForest` 模型必须确认该方向（例如 `ML_Proba_UP > 0.60`）。
* **动作：** 在该“共振区”挂一个 `LIMIT_BUY` 或 `LIMIT_SELL` 单。

#### B. 震荡策略 (Rule 6.2: 均值回归)

* **适用指标：** `1h ADX` **和** `4h ADX` < 20（市场处于盘整中）。
* **入场逻辑 (多重确认)：**
    1.  **区域 (Zone)：** 价格必须接近 `BB_Upper` (做空) 或 `BB_Lower` (做多)。
    2.  **ML 确认 ：** AI **必须**检查正确的方向（例如，做空时检查 `ML_Proba_DOWN > 0.60`）。
* **动作：** 在布林带“上下轨”附近挂单。

#### C. 盘整策略 (Rule 6.3: 短线)

* **适用指标：** `ADX` 在 20-25 之间（市场方向不明）。
* **入场逻辑：** 同上（Rule 6.2），但 AI (LLM) 会切换到 `15m` 布林带，并且**必须**使用更低的风险（`risk_percent`）。

### 加仓逻辑 (Pyramiding)

* **允许**。AI (LLM) 可以在满足以下条件时为**已有的 `Rule 6` 仓位**加仓：
    1.  `UPL Percent > +2.5%`。
    2.  `ADX > 25` (原始趋势仍在继续)。
    3.  价格已回调到支撑位。
    4.  没有重复挂单。

### 止盈止损 (AI 与 Python 协作)

`Rule 6` 策略的风险由 L1 和 L3 共同管理。

* **止损 (Stop Loss)：**
    1.  **AI 设定：** AI (LLM) **必须**在开仓时（`Rule 5`） 设置一个基于 **ATR** 的 `stop_loss`。
    2.  **AI 设定：** `invalidation_condition` 被强制设为 `Stop Loss hit (ATR-based)`，以防止因脆弱的技术指标而过早平仓。
    3.  **Python 执行：** `high_frequency_risk_loop`（2 秒循环） 负责高频监控并执行这个 `ai_suggested_stop_loss`。
* **平仓 (Exit)：** `Rule 6` 仓位有多种退出方式：
    1.  **Python (L1) 自动平仓：**
        * 价格触及 AI 设定的 `ai_suggested_stop_loss`。
        * 价格触及 AI 设定的 `ai_suggested_take_profit`。
        * 价格触及 `Hard TP Stage 1/2`（多阶段硬止盈）。
    2.  **AI (L3) 主动平仓 (战略退出)：**
        * **风险否决：** AI (LLM) 在 `run_cycle` 中检测到 `Anomaly_Score < -0.1` 并发出 `CLOSE` 指令。
        * **利润保护：** AI (LLM) 在 `run_cycle` 中发现 `UPL > 1.0%` 且出现“反转迹象”（`Reversal & Profit Save Check`），并发出 `CLOSE` 指令以“确保利润”。
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
