import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional, List

# 加载 .env 文件中的环境变量 (例如 API 密钥)
load_dotenv()

class Settings(BaseSettings):
    """
    AlphaTrader (AI + Python 混合模型) 专用配置
    """
    
    ### 1. 全局及账户设置 ###
    
    # 交易所 API 密钥 (从 .env 文件读取)
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    
    # 交易所测试网 API 密钥 (从 .env 文件读取)
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_SECRET_KEY: str = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")

    # 是否连接到币安测试网 (True: 测试网, False: 实盘)
    USE_TESTNET: bool = True

    # [!!! 核心安全开关 !!!] 
    # True: 实盘交易, False: 模拟盘。
    # 注意: Python Rule 8 和 AI Rule 6 都会遵守此开关。
    ALPHA_LIVE_TRADING: bool = False

    # 模拟盘 (Paper Trading) 的初始资金
    ALPHA_PAPER_CAPITAL: float = 10000.0

    # (可选) 实盘初始资本 (用于计算 PnL 百分比)
    ALPHA_LIVE_INITIAL_CAPITAL: float = 1000.0

    # 机器人将交易的合约列表
    # 原始配置: ["BNB/USDT:USDT", "ETH/USDT:USDT"]
    # alpha_trader.py 默认: ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT", "XRP/USDT:USDT"]
    FUTURES_SYMBOLS_LIST: List[str] = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", 
        "BNB/USDT:USDT", "DOGE/USDT:USDT", "XRP/USDT:USDT"
    ]

    ### 2. AI (Rule 6 - 限价单) 策略配置 ###
    
    # AI (Rule 6) 的分析周期 (秒)
    # AI 将每隔 900 秒 (15 分钟) 运行一次，寻找限价单机会
    ALPHA_ANALYSIS_INTERVAL_SECONDS: int = 900 #

    # AI 事件触发器 (如背离、RSI越界) 的冷却时间 (分钟)
    # 在事件触发 AI 后，AI 会 "冷静" 14 分钟，防止被连续触发
    AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES: int = 14 #

    # AI (Rule 6) 限价单的超时时间 (秒)
    # 如果 AI 的限价单挂出 900 秒 (15 分钟) 未成交，则自动取消
    AI_LIMIT_ORDER_TIMEOUT_SECONDS: int = 900 #

    ### 3. Python (Rule 8 - 突破) 策略配置 ###

    # [Rule 8 主开关] 是否启用 Python 高频突破策略
    # True: 启用, False: 禁用 (此时机器人将只执行 AI Rule 6)
    ENABLE_BREAKOUT_MODIFIER: bool = True #

    # [Rule 8 杠杆] Python 突破策略使用的杠杆
    FUTURES_LEVERAGE: int = 5 #

    # [Rule 8 风险] Python 突破策略的单笔风险 (占总权益的百分比)
    # 1.5 表示 1.5%
    FUTURES_RISK_PER_TRADE_PERCENT: float = 1.5 #

    # [Rule 8 止损] 是否使用 ATR (平均真实波幅) 来计算初始止损
    # True: 使用 ATR, False: 使用备用的 2% 静态止损
    USE_ATR_FOR_INITIAL_STOP: bool = True #

    # [Rule 8 止损] 如果使用 ATR，ATR 的乘数
    # (例如，止损位 = 价格 - (ATR * 2.6))
    INITIAL_STOP_ATR_MULTIPLIER: float = 2.6 #

    # [Rule 8 止盈] Python 突破策略的动态追踪止损百分比
    # 0.003 表示 0.3%。
    # 这是 Rule 8 快速反应的核心：价格每创新高，止损就上移到 (新高 * (1 - 0.003))
    BREAKOUT_TRAIL_STOP_PERCENT: float = 0.003 #

    ### 4. 全局风控配置 ###
    
    # [全局风控] AI (Rule 6) 仓位的最大亏损切断百分比
    # 20.0 表示 20%。当单笔 AI 仓位的保证金亏损达到 20% 时，立即平仓
    # (注意: 此规则不适用于 Rule 8，因其有自己的追踪止损)
    MAX_LOSS_CUTOFF_PERCENT: float = 20.0 #

    # [全局风控] 最小开仓保证金 (USDT)
    # (在 alpha_trader.py 中硬编码为 6.0，使用此配置更佳)
    MIN_NOMINAL_VALUE_USDT: float = 6.0 #

    # [全局风控] Python (Rule 8) 仓位允许的最大保证金占用率
    # 0.15 表示 Rule 8 仓位的保证金不能超过总权益的 15%
    # (这是一个有用的硬顶，防止 Rule 8 开出过大仓位)
    MAX_MARGIN_PER_TRADE_RATIO: float = 0.15  #

    ### 5. AI 服务及通知配置 ###

    # AI (Gemini / OpenAI) 供应商 (从 .env 读取)
    # 必须设置为 "AZURE", "OPENAI" 或 "DEEPSEEK" 之一
    AI_PROVIDER: str = os.getenv("AI_PROVIDER")

    # Azure OpenAI API 配置 (从 .env 文件读取)
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "") #
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "") #
    AZURE_OPENAI_MODEL_NAME: str = os.getenv("AZURE_OPENAI_MODEL_NAME", "") #

    # 兼容 OpenAI/DeepSeek 的配置 (从 .env 读取)
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE") #
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") #
    OPENAI_MODEL_NAME: str  = os.getenv("OPENAI_MODEL_NAME") #

    # Bark App 通知服务 (可选)
    BARK_ENABLED: bool = os.getenv('BARK_ENABLED', 'false').lower() == 'true' #
    BARK_KEY: Optional[str] = os.getenv('BARK_KEY') #

    # Pydantic 模型配置
    model_config = ConfigDict(env_file=".env", env_file_encoding='utf-8', case_sensitive=True, extra='ignore')


# 实例化配置对象，供其他文件导入
settings = Settings()

# 打印关键配置以便调试
print(f"--- Config Check (End of config.py) ---")
print(f"ALPHA_LIVE_TRADING: {settings.ALPHA_LIVE_TRADING}")
print(f"ENABLE_BREAKOUT_MODIFIER (Rule 8): {settings.ENABLE_BREAKOUT_MODIFIER}")
print(f"AI_PROVIDER: {settings.AI_PROVIDER}")
print(f"-----------------------------------")


# (原 FuturesSettings 类已被合并和移除)
