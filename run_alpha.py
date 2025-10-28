# 文件: run_alpha.py (V2 - 功能完整版)

# 步骤 1: 确保 .env 文件在所有自定义模块导入前被加载
from dotenv import load_dotenv
load_dotenv()

import asyncio
import ccxt.pro as ccxtpro
import logging
from config import settings
from exchange_client import ExchangeClient
from alpha_trader import AlphaTrader
# 步骤 2: 正确导入 Web 服务器启动函数
from alpha_web_server import start_alpha_web_server 
from helpers import setup_logging

async def main():
    setup_logging()
    logger = logging.getLogger("AlphaRun")

    if not settings.ALPHA_MODE_ENABLED:
        logger.warning("AI Alpha Trader 模块未在配置文件中启用 (ALPHA_MODE_ENABLED=false)，程序退出。")
        return

    # 创建交易所连接
    exchange_config = {
        'apiKey': settings.BINANCE_API_KEY, 'secret': settings.BINANCE_SECRET_KEY,
        'options': {'defaultType': 'swap'}
    }
    if settings.USE_TESTNET:
        exchange_config.update({
            'apiKey': settings.BINANCE_TESTNET_API_KEY,
            'secret': settings.BINANCE_TESTNET_SECRET_KEY,
        })
        exchange = ccxtpro.binance(exchange_config)
        exchange.set_sandbox_mode(True)
        logger.info("正在连接到币安测试网...")
    else:
        exchange = ccxtpro.binance(exchange_config)
        logger.info("正在连接到币安实盘...")
    
    exchange_client = ExchangeClient(exchange)

    # 创建 AI Alpha Trader 实例
    alpha_trader = AlphaTrader(exchange=exchange_client)
    
    # 步骤 3: 调用函数，尝试启动 Web 服务器
    web_server_site = await start_alpha_web_server(alpha_trader)
    
    # 步骤 4: 检查启动结果，如果失败则打印日志，但程序不中断
    if not web_server_site:
        logger.critical("Web服务器启动失败，程序将不带UI界面继续运行。请检查端口是否被占用。")
    
    try:
        # 启动交易主循环
        await alpha_trader.start()
    except KeyboardInterrupt:
        logger.warning("接收到关闭信号，正在优雅地关闭程序...")
    finally:
        # 步骤 5: 在程序退出前，安全地停止 Web 服务器和交易所连接
        if web_server_site:
            await web_server_site.stop()
            logger.info("Web 仪表盘已关闭。")
        await exchange.close()
        logger.info("交易所连接已关闭。程序退出。")

if __name__ == "__main__":
    asyncio.run(main())
