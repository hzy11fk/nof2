# nof2 
参考nof1设计的AI自主分析交易机器人

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

## 免责声明
本项目仅供学习和研究目的，作者不对任何使用本项目代码造成的实际交易盈亏负责。请在充分了解风险的情况下使用。
