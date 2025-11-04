# 文件: exchange_client.py (V9 - 移除了 Taker Ratio 函数)

import logging
import asyncio
from ccxt.base.errors import RequestTimeout, NetworkError, ExchangeNotAvailable, DDoSProtection

class ExchangeClient:
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(self.__class__.__name__)

    # --- [ V-FIX 3: 添加 .market() 方法 ] ---
    def market(self, symbol):
        """将 .market() 方法从内部 ccxt 对象传递出去。"""
        if hasattr(self.exchange, 'market'):
            return self.exchange.market(symbol)
        self.logger.error(f"内部交易所对象缺少 'market' 方法。")
        # 返回一个带 'id' 的回退字典，以防止 ['id'] 访问崩溃
        return {'id': symbol} 
    # --- [ V-FIX 3 结束 ] ---

    # --- [ V-FIX 1: 添加 .has 属性 ] ---
    @property
    def has(self):
        """将 .has 属性从内部的 ccxt 交易所对象传递出去。"""
        if hasattr(self.exchange, 'has'):
            return self.exchange.has
        # 提供一个回退，以防内部对象也没有 .has
        self.logger.warning("内部交易所对象缺少 'has' 属性。")
        return {} 
    # --- [ V-FIX 1 结束 ] ---

    async def _retry_async_method(self, method, *args, **kwargs):
        """
        一个健壮的异步方法重试包装器。
        - max_retries: 最大重试次数
        - delay: 每次重试前的等待时间（秒）
        """
        max_retries = 3
        delay = 5  # 5秒
        for attempt in range(max_retries):
            try:
                # 尝试调用原始方法
                return await method(*args, **kwargs)
            except (RequestTimeout, NetworkError, ExchangeNotAvailable, DDoSProtection) as e:
                # 只对可恢复的网络或超时错误进行重试
                if attempt < max_retries - 1:
                    self.logger.warning(f"调用 {method.__name__} 时发生可重试错误: {e}。将在 {delay} 秒后进行第 {attempt + 2} 次尝试...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"调用 {method.__name__} 失败，已达到最大重试次数 ({max_retries})。")
                    raise  # 重试次数用尽后，重新抛出最后的异常
            except Exception as e:
                # 对于其他所有错误（如API密钥错误、参数错误），不进行重试，立即抛出
                self.logger.error(f"调用 {method.__name__} 时发生不可重试的严重错误: {e}")
                raise

    async def load_markets(self):
        """加载市场信息，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.load_markets)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
        """获取K线数据，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)

    async def fetch_tickers(self, symbols: list):
        """[新增整合] 获取多个交易对的 Tickers，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_tickers, symbols)

    async def fetch_ticker(self, symbol: str):
        """获取最新价格，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_ticker, symbol)

    async def fetch_funding_rate(self, symbol: str):
        """[新增整合] 获取资金费率，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_funding_rate, symbol)

    async def fetch_open_interest(self, symbol: str):
        """[新增整合] 获取未平仓合约量，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_open_interest, symbol)

    # --- [ V-FIX 2: 添加缺失的方法 ] ---
    async def fetch_open_interest_history(self, symbol: str, timeframe: str = '5m', limit: int = 2, params={}):
        """[新增] 获取OI历史数据，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_open_interest_history, symbol, timeframe=timeframe, limit=limit, params=params)
    # --- [ V-FIX 2 结束 ] ---

    # --- [ V-FIX 4: 导致崩溃的 Taker Ratio 函数已被移除 ] ---
# [V-Ultimate 优化] 新增 Taker Ratio 方法
    
# [V-Ultimate 优化] 新增 Taker Ratio 方法
    async def fetch_taker_long_short_ratio(self, symbol: str, period: str = '1h', limit: int = 21, params={}):
        """
        [V-Ultimate 新增] 获取币安合约 Taker 多空比 (非 CCXT 标准方法)。
        这会直接调用 ccxt.binance.request() 方法。
        """
        if not hasattr(self.exchange, 'request'):
            self.logger.error("Internal exchange object has no 'request' method. Cannot fetch Taker Ratio.")
            raise NotImplementedError("Exchange object does not support raw 'request' calls.")

        # [V3 修复] path 只是端点名称。
        path = 'takerlongshortRatio'
        
        market_id = self.exchange.market_id(symbol)
        
        request_params = {
            'symbol': market_id,
            'period': period,
            'limit': limit,
            **params
        }
        
        try:
            # [V3 修复] API 组必须是 'futuresData' 而不是 'fapiPublic'
            response = await self._retry_async_method(
                self.exchange.request,
                path,
                'futuresData',  # <-- V3 关键修复
                'GET',
                request_params
            )
            return response
        except Exception as e:
            self.logger.error(f"Failed to fetch taker ratio for {symbol} via self.exchange.request: {e}")
            raise

    async def fetch_balance(self, params={}):
        """获取余额，并应用重试逻辑。"""
        try:
            balance = await self._retry_async_method(self.exchange.fetch_balance, params=params)
            return balance
        except Exception as e:
            self.logger.error(f"获取余额最终失败: {e}", exc_info=True)
            raise

    async def fetch_positions(self, symbols: list = None):
        """[新增整合] 获取持仓信息，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_positions, symbols)
        
    async def fetch_my_trades(self, symbol: str, limit: int = 1000):
        """获取历史成交，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_my_trades, symbol, limit=limit)

    async def fetch_order(self, order_id: str, symbol: str):
        """获取订单信息，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_order, order_id, symbol=symbol)

    async def fetch_open_orders(self, symbol: str = None, since: int = None, limit: int = None, params={}):
        """获取所有未结订单，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.fetch_open_orders, symbol=symbol, since=since, limit=limit, params=params)

    async def cancel_order(self, order_id: str, symbol: str):
        """取消订单，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.cancel_order, order_id, symbol=symbol)

    async def set_leverage(self, leverage, symbol):
        """设置杠杆，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.set_leverage, leverage, symbol=symbol)

    async def set_margin_mode(self, margin_mode, symbol):
        """设置保证金模式，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.set_margin_mode, margin_mode, symbol=symbol)
        
    async def create_market_order(self, symbol: str, side: str, amount: float, params={}):
        """创建市价单，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.create_market_order, symbol, side, amount, params=params)

    async def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params={}):
        """创建限价单，并应用重试逻辑。"""
        return await self._retry_async_method(self.exchange.create_limit_order, symbol, side, amount, price, params=params)
