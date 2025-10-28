# 文件: config.py (精简版 - 仅包含 AlphaTrader 所需参数 - 简体中文注释)

import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

# 加载 .env 文件 (如果存在的话)
load_dotenv()

class Settings(BaseSettings):
    """全局应用程序设置 (AlphaTrader 精简版)"""
    # --- 交易所连接 ---
    USE_TESTNET: bool = False # True 使用测试网 API 密钥, False 使用主网
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_SECRET_KEY: str = os.getenv("BINANCE_TESTNET_SECRET_KEY", "")

    # --- AI 分析器 (alpha_ai_analyzer.py) 配置 ---
    # 根据您的 AI 分析器实现选择和配置以下选项
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "openai") # 'openai', 'azure', 'deepseek' 等
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_MODEL_NAME: str = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4-turbo") # 部署名
    # OpenAI / DeepSeek (或其他兼容 OpenAI API 的服务)
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE") # 例如 DeepSeek 的 base url
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_NAME: str  = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat") # 模型名

    # --- AI Alpha Trader (alpha_trader.py) 核心配置 ---
    ALPHA_LIVE_TRADING: bool = True               # !!! 核心开关: True=实盘, False=模拟盘 !!!
    ALPHA_PAPER_CAPITAL: float = 10000.0          # 模拟盘初始资金 (USDT)
    ALPHA_LIVE_INITIAL_CAPITAL: float = 100.0     # !!! 请设置为您的实盘初始本金 (USDT) !!!
    ALPHA_ANALYSIS_INTERVAL_SECONDS: int = 180    # AI 定期分析的间隔 (秒)

    # --- AI Alpha Trader 事件触发器配置 ---
    AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES: int = 14 # 事件触发后的冷却时间 (分钟)
    AI_VOLATILITY_TRIGGER_PERCENT: float = 3.0      # 1小时K线价格变动超过此百分比 (%)，触发分析

    # --- Bark 通知 (bark_notifier.py) ---
    BARK_ENABLED: bool = os.getenv('BARK_ENABLED', 'false').lower() == 'true'
    BARK_KEY: Optional[str] = os.getenv('BARK_KEY') # 你的 Bark 设备 Key
    BARK_URL_KEY: Optional[str] = os.getenv("BARK_URL_KEY") # 备用，如果 BARK_KEY 不直接可用

    # Pydantic 配置
    model_config = ConfigDict(env_file=".env", env_file_encoding='utf-8', case_sensitive=True, extra='ignore')

class FuturesSettings:
    """合约交易特定设置 (AlphaTrader 精简版)"""
    # --- 交易执行 (alpha_portfolio.py) ---
    FUTURES_MARGIN_MODE: str = 'isolated'     # 实盘保证金模式 ('isolated' 或 'cross')
    # 单笔订单最大保证金占用率 (相对于 *可用现金*)，用于服务器端验证
    MAX_MARGIN_PER_TRADE_RATIO: float = 0.10  # 0.10 代表最多使用 10% 的可用现金作为单次下单的保证金

    # --- 状态文件存储目录 (多个模块使用) ---
    FUTURES_STATE_DIR: str = 'data'           # 保存交易记录、持仓状态等的目录

# 创建配置实例
settings = Settings()
futures_settings = FuturesSettings()

