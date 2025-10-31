import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    """全局应用程序设置"""
    USE_TESTNET: bool = False
    FUTURES_SYMBOLS_LIST: list = ["BNB/USDT:USDT", "ETH/USDT:USDT"]
    FUTURES_INITIAL_PRINCIPAL: float = 289
    ENABLE_TRENDLINE_FILTER: bool = True
    TRENDLINE_LOOKBACK_PERIOD: int = 180
    TRENDLINE_PIVOT_WINDOW: int = 5
    TRENDLINE_NEARNESS_ATR_MULTIPLIER: float = 0.5
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_SECRET_KEY: str = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")
    BARK_URL_KEY: str = os.getenv("BARK_URL_KEY", "")
    BREAKOUT_TRAIL_STOP_PERCENT: float = 0.003
    TREND_SIGNAL_TIMEFRAME: str = '5m'
    TREND_FILTER_TIMEFRAME: str = '15m'
    TREND_SHORT_MA_PERIOD: int = 7
    TREND_LONG_MA_PERIOD: int = 21
    TREND_FILTER_MA_PERIOD: int = 30
    TREND_ADX_THRESHOLD_STRONG: int = 25
    TREND_ADX_THRESHOLD_WEAK: int = 20
    
    TREND_ATR_MULTIPLIER_STRONG: float = 1.0
    TREND_ATR_MULTIPLIER_WEAK: float = 0.3

    ENABLE_VOLUME_CONFIRMATION: bool = True
    ENABLE_RSI_CONFIRMATION: bool = True

    ENABLE_TREND_MEMORY: bool = True
    TREND_CONFIRMATION_GRACE_PERIOD: int = 3
    TREND_VOLUME_CONFIRM_PERIOD: int = 20
    TREND_RSI_CONFIRM_PERIOD: int = 14
    TREND_RSI_UPPER_BOUND: int = 60
    TREND_RSI_LOWER_BOUND: int = 40

    DYNAMIC_VOLUME_ENABLED: bool = True
    DYNAMIC_VOLUME_BASE_MULTIPLIER: float = 1.5
    DYNAMIC_VOLUME_ATR_PERIOD_SHORT: int = 10
    DYNAMIC_VOLUME_ATR_PERIOD_LONG: int = 50
    DYNAMIC_VOLUME_ADJUST_FACTOR: float = 0.5

    ENABLE_BREAKOUT_MODIFIER: bool = True
    REQUIRE_FILTER_FOR_AGGRESSIVE: bool = True
    BREAKOUT_NOMINAL_VALUE_USDT: float = 50.0
    BREAKOUT_TIMEFRAME: str = '3m'
    BREAKOUT_BBANDS_PERIOD: int = 20
    BREAKOUT_BBANDS_STD_DEV: float = 2.0
    # --- [核心优化] 新增布林带挤压过滤器配置 ---
    ENABLE_BBAND_SQUEEZE_FILTER: bool = True       # 是否启用布林带挤压过滤器
    BBAND_SQUEEZE_LOOKBACK_PERIOD: int = 120        # 判断挤压状态的回看周期 (多少根K线)
    BBAND_SQUEEZE_THRESHOLD_PERCENTILE: float = 0.25 # 带宽小于过去N周期中25%的时间，则视为挤压

    BREAKOUT_GRACE_PERIOD_SECONDS: int = 300
    BREAKOUT_GRACE_PERIOD_SECONDS: int = 300
    AGGRESSIVE_PULLBACK_ZONE_MULTIPLIER: float = 2.0
    AGGRESSIVE_RELAXED_VOLUME_MULTIPLIER: float = 0.8
    ENABLE_FUNDING_FEE_SYNC: bool = True
    ENABLE_SPIKE_MODIFIER: bool = True
    SPIKE_TIMEFRAME: str = '5m'
    SPIKE_BODY_ATR_MULTIPLIER: float = 2.0
    SPIKE_VOLUME_MULTIPLIER: float = 2.5
    SPIKE_GRACE_PERIOD_SECONDS: int = 600
    SPIKE_ENTRY_CONFIRMATION_BARS: int = 3
    SUPER_AGGRESSIVE_PULLBACK_ZONE_MULTIPLIER: float = 3.0
    SUPER_AGGRESSIVE_RELAXED_VOLUME_MULTIPLIER: float = 0.5
    SPIKE_ENTRY_GRACE_PERIOD_MINUTES: int = 10

    ENABLE_RANGING_STRATEGY: bool = True
    RANGING_TIMEFRAME: str = '15m'
    RANGING_NOMINAL_VALUE_USDT: float = 30.0
    RANGING_ADX_THRESHOLD: int = 18
    RANGING_BBANDS_PERIOD: int = 20
    RANGING_BBANDS_STD_DEV: float = 2.0
    RANGING_TAKE_PROFIT_TARGET: str = 'middle'
    RANGING_STOP_LOSS_ATR_MULTIPLIER: float = 1.5

    ENABLE_PULLBACK_QUALITY_FILTER: bool = True
    PULLBACK_MAX_VOLUME_RATIO: float = 0.8

    ENABLE_ENTRY_MOMENTUM_CONFIRMATION: bool = True
    ENTRY_RSI_PERIOD: int = 7
    ENTRY_RSI_CONFIRMATION_BARS: int = 3

    ENABLE_PERFORMANCE_FEEDBACK: bool = False
    FUNDING_FEE_SYNC_INTERVAL_HOURS: int = 1
    PERFORMANCE_CHECK_INTERVAL_HOURS: int = 4
    MIN_TRADES_FOR_EVALUATION: int = 5
    PERF_WEIGHT_WIN_RATE: float = 0.40
    PERF_WEIGHT_PAYOFF_RATIO: float = 0.25
    PERF_WEIGHT_DRAWDOWN: float = 0.35

    BREAKOUT_VOLUME_CONFIRMATION: bool = True
    BREAKOUT_VOLUME_PERIOD: int = 20
    BREAKOUT_VOLUME_MULTIPLIER: float = 1.5

    BREAKOUT_RSI_CONFIRMATION: bool = True
    BREAKOUT_RSI_PERIOD: int = 14
    BREAKOUT_RSI_THRESHOLD: int = 50
    AGGRESSIVE_PARAMS: dict = {"PULLBACK_ZONE_PERCENT": 0.2, "ATR_MULTIPLIER": 2.0, "PYRAMIDING_TRIGGER_PROFIT_MULTIPLE": 0.8}
    DEFENSIVE_PARAMS: dict = {"PULLBACK_ZONE_PERCENT": 0.6, "ATR_MULTIPLIER": 3.5, "PYRAMIDING_TRIGGER_PROFIT_MULTIPLE": 1.5}

    model_config = ConfigDict(env_file=".env", env_file_encoding='utf-8', case_sensitive=True, extra='ignore')

# --- [AI 决策引擎配置] ---
    ENABLE_AI_MODE: bool = True                     # 是否启用 AI 判断模式
    AI_ANALYSIS_INTERVAL_MINUTES: int = 5          # AI 分析的周期（分钟）
    AI_CONFIDENCE_THRESHOLD: int = 80               # AI 判断可用于真实交易的置信度分数阈值 (0-100)
    AI_ENABLE_LIVE_TRADING: bool = False            # !!! 安全开关 !!! AI达到阈值后是否允许真实下单
    AI_STATE_DIR: str = 'data'                      # 存储 AI 表现和状态文件的目录
    AI_PERFORMANCE_LOOKBACK_TRADES: int = 50        # 用于计算表现分数的最近交易笔数
    AI_MIN_RISK_REWARD_RATIO: float = 1.2
    AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES: int = 14
    # [新增] AI 下单类型配置
    AI_ORDER_TYPE: str = os.getenv("AI_ORDER_TYPE", "limit") # 可选 "market" (市价) 或 "limit" (限价)
    AI_LIMIT_ORDER_CANCEL_THRESHOLD_PERCENT: float = 0.5    # 价格偏离挂单价 0.5% 时，自动取消挂单
    # Azure OpenAI API 配置 (从 .env 文件读取)
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_MODEL_NAME: str = os.getenv("AZURE_OPENAI_MODEL_NAME", "") # 例如 "gpt-4-turbo"
# 确保 AI_PROVIDER 被读取
    AI_PROVIDER: str = os.getenv("AI_PROVIDER")

# 兼容 OpenAI/DeepSeek 的配置
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str  = os.getenv("OPENAI_MODEL_NAME")


# --- AI Alpha Trader 配置 ---
    ALPHA_MODE_ENABLED: bool = True
    ALPHA_LIVE_TRADING: bool = False
    ALPHA_PAPER_CAPITAL: float = 10000.0
    ALPHA_LIVE_INITIAL_CAPITAL: float = 200.0 #
    ALPHA_ANALYSIS_INTERVAL_SECONDS: int = 900
# --- [新增] alpha_trader 的事件触发器配置 ---
    AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES: int = 14 # 事件触发后的冷却时间（分钟）
    AI_VOLATILITY_TRIGGER_PERCENT: float = 1.0      # 1小时K线价格变动超过3%，触发分析
    BARK_ENABLED: bool = os.getenv('BARK_ENABLED', 'false').lower() == 'true'
    BARK_KEY: Optional[str] = os.getenv('BARK_KEY')
    MAX_LOSS_CUTOFF_PERCENT: float = 20.0
    AI_LIMIT_ORDER_TIMEOUT_SECONDS: int = 900

class FuturesSettings:
    FUTURES_LEVERAGE: int = 5
    FUTURES_MARGIN_MODE: str = 'isolated'
    FUTURES_RISK_PER_TRADE_PERCENT: float = 1.5
    # --- [核心优化] 新增单笔交易最大保证金占用率 (占总权益) ---
    MAX_MARGIN_PER_TRADE_RATIO: float = 0.15  # 值为 0.20 代表 20%
    MIN_NOMINAL_VALUE_USDT: float = 50.0
    USE_ATR_FOR_INITIAL_STOP: bool = True
    INITIAL_STOP_ATR_MULTIPLIER: float = 2.6
    
    FUTURES_ENTRY_PULLBACK_EMA_PERIOD: int = 10
    FUTURES_STATE_DIR: str = 'data'
    TREND_EXIT_ADJUST_SL_ENABLED: bool = True
    TREND_EXIT_CONFIRMATION_COUNT: int = 3
    TREND_EXIT_ATR_MULTIPLIER: float = 1.8
    
    PYRAMIDING_ENABLED: bool = True
    PYRAMIDING_PULLBACK_CHECK_ENABLED: bool = False
    PYRAMIDING_MAX_ADD_COUNT: int = 2
    PYRAMIDING_ADD_SIZE_RATIO: float = 0.75
    
    CHANDELIER_EXIT_ENABLED: bool = True
    CHANDELIER_ACTIVATION_PROFIT_MULTIPLE: float = 2.0
    CHANDELIER_PERIOD: int = 16
    CHANDELIER_ATR_MULTIPLIER: float = 2.5
    
    TRAILING_STOP_MIN_UPDATE_SECONDS: int = 60
    ADAPTIVE_TRAILING_STOP_ENABLED: bool = True
    TRAILING_STOP_ATR_SHORT_PERIOD: int = 10
    TRAILING_STOP_ATR_LONG_PERIOD: int = 50
    TRAILING_STOP_VOLATILITY_PAUSE_THRESHOLD: float = 0.0005

    ENABLE_EXHAUSTION_ALERT: bool = True
    EXHAUSTION_ADX_PERIOD: int = 14
    EXHAUSTION_ADX_THRESHOLD: float = 25.0
    EXHAUSTION_ADX_FALLING_BARS: int = 3
    
    # --- [核心修复] 将风险预警设置移动到这里 ---
    ENABLE_REVERSAL_SIGNAL_ALERT: bool = True       # 是否启用反转型K线风险预警
    REVERSAL_ALERT_BODY_ATR_MULTIPLIER: float = 1.5   # K线实体必须超过ATR的倍数
    REVERSAL_ALERT_VOLUME_MULTIPLE: float = 2.0     # K线成交量必须超过均量的倍数
# --- [新增] 强制止盈配置 ---
    ENABLE_FORCED_TAKE_PROFIT: bool = True     # 是否启用强制止盈功能
    FORCED_TAKE_PROFIT_PERCENT: float = 20.0   # 收益率达到此百分比 (%) 时强制止盈 (基于近似初始保证金)
settings = Settings()
futures_settings = FuturesSettings()
print(f"--- Config Check (End of config.py): ALPHA_LIVE_TRADING = {settings.ALPHA_LIVE_TRADING} ---")
