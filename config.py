# 文件: config.py (V2.0 - 针对 AlphaTrader V45+ AI 策略的精简版)

import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

# 加载 .env 文件中的环境变量
load_dotenv()

class Settings(BaseSettings):
    """
    全局应用程序设置 (Pydantic模型)
    这些设置主要由 alpha_trader.py, alpha_ai_analyzer.py, alpha_web_server.py 和 helpers.py 使用。
    """
    
    # --- [1. 交易所与API密钥] ---
    # 是否使用币安测试网 (True: 测试网, False: 实盘)
    USE_TESTNET: bool = True 
    
    # 币安实盘 API 密钥 (从 .env 文件读取)
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    
    # 币安测试网 API 密钥 (从 .env 文件读取)
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_SECRET_KEY: str = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")

    
    # --- [2. AI Alpha Trader 核心配置] ---
    # AlphaTrader 是否启动 (用于 main.py 的总开关)
    ALPHA_MODE_ENABLED: bool = True
    
    # !!! 核心安全开关: AI 是否允许实盘交易 !!!
    # (True: 实盘下单, False: 模拟盘下单)
    # **注意: alpha_web_server.py 也会读取此项来显示模式**
    ALPHA_LIVE_TRADING: bool = False
    
    # 模拟盘的初始资本
    ALPHA_PAPER_CAPITAL: float = 10000.0
    
    # 实盘的初始资本 (用于计算收益率)
    ALPHA_LIVE_INITIAL_CAPITAL: float = 100.0
    
    # AI 决策的定时周期（秒）
    # (例如: 300 = 每 5 分钟调用一次 AI)
    ALPHA_ANALYSIS_INTERVAL_SECONDS: int = 300
    
    # 当指标触发（例如 MACD 交叉）AI 分析后，进入冷却的时间（分钟）
    # (防止因市场频繁小幅波动而过于频繁地触发 AI)
    AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES: int = 14
    
    # 1小时K线的价格波动百分比（例如 3.0 = 3%）
    # (如果1小时内价格剧烈波动超过此阈值，将立即触发 AI 分析)
    AI_VOLATILITY_TRIGGER_PERCENT: float = 3.0


    # --- [3. AI (LLM) 提供商配置] ---
    # AI 服务提供商 (例如 'azure', 'openai', 'deepseek')
    # (由 alpha_ai_analyzer.py 读取)
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "azure")

    # Azure OpenAI 配置 (如果 AI_PROVIDER = 'azure')
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_MODEL_NAME: str = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4-turbo")
    # Azure API 版本 (alpha_ai_analyzer.py 会使用默认值 '2024-02-01' 如果未提供)
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2024-02-01")

    # OpenAI / DeepSeek / 兼容 API 配置 (如果 AI_PROVIDER = 'openai' 或 'deepseek')
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str  = os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo")


    # --- [4. 通知配置] ---
    # Bark App 的通知 URL (包含 key)
    # (由 helpers.py 读取)
    BARK_URL_KEY: str = os.getenv("BARK_URL_KEY", "")


    # Pydantic 模型配置
    model_config = ConfigDict(env_file=".env", env_file_encoding='utf-8', case_sensitive=True, extra='ignore')


class FuturesSettings:
    """
    合约交易的特定设置
    这些设置主要由 alpha_portfolio.py 和 alpha_position_manager.py 使用。
    """
    
    # 默认杠杆倍数 (AI 可以在 Prompt 中覆盖此值)
    FUTURES_LEVERAGE: int = 10
    
    # 保证金模式: 'isolated' (逐仓) 或 'crossed' (全仓)
    FUTURES_MARGIN_MODE: str = 'isolated'
    
    # 单笔交易的最大保证金占用率 (基于*可用现金*)
    # (例如: 0.10 = AI 开仓或加仓时，单笔订单占用的保证金不能超过可用现金的 10%)
    # **注意: AI Prompt 中的规则 (6 USDT 最小保证金) 优先级更高**
    MAX_MARGIN_PER_TRADE_RATIO: float = 0.10
    
    # 状态文件存储目录
    # (alpha_position_manager.py 和 alpha_trade_logger.py 会在此目录下创建 .json 文件)
    FUTURES_STATE_DIR: str = 'data'
    
    # --- [强制止盈 (风控)] ---
    # 是否启用硬编码的强制止盈 (由 alpha_trader.py 在循环开始时检查)
    ENABLE_FORCED_TAKE_PROFIT: bool = True
    
    # 强制止盈的百分比阈值 (基于 *初始保证金* 计算)
    # (例如: 20.0 = 收益率达到 20% 时，强制平仓)
    FORCED_TAKE_PROFIT_PERCENT: float = 20.0


# --- [实例化配置] ---
# 创建全局可用的 settings 和 futures_settings 实例
settings = Settings()
futures_settings = FuturesSettings()

