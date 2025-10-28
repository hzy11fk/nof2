# 文件: alpha_trader.py (V43 - 支持加仓和部分平仓指令)

import logging
import asyncio
import time
import json
import pandas as pd
import pandas_ta as ta # 确保导入 pandas_ta
import re
from collections import deque
from config import settings
from alpha_ai_analyzer import AlphaAIAnalyzer
# [修改] 导入更新后的 Portfolio V22
from alpha_portfolio import AlphaPortfolio # 假设这是 V22
from datetime import datetime
from typing import Tuple, Dict, Any # 增加类型提示

class AlphaTrader:
    def __init__(self, exchange):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exchange = exchange
        self.symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT", "XRP/USDT:USDT"]
        # [修改] 使用更新后的 Portfolio V22
        self.portfolio = AlphaPortfolio(exchange, self.symbols)
        self.ai_analyzer = AlphaAIAnalyzer(exchange, "ALPHA_TRADER")
        self.is_live_trading = settings.ALPHA_LIVE_TRADING
        self.start_time = time.time(); self.invocation_count = 0; self.last_run_time = 0; self.last_event_trigger_ai_time = 0
        self.log_deque = deque(maxlen=50); self._setup_log_handler()
        self.last_strategy_summary = "Initializing, waiting for the first decision cycle..."
        self.initial_capital = settings.ALPHA_LIVE_INITIAL_CAPITAL if self.is_live_trading else settings.ALPHA_PAPER_CAPITAL
        self.logger.info(f"Initialized with Initial Capital: {self.initial_capital:.2f} USDT")

    # --- _setup_log_handler 到 _get_ai_decision 保持不变 ---
    def _setup_log_handler(self):
        class DequeLogHandler(logging.Handler):
            def __init__(self, deque_instance): super().__init__(); self.deque_instance = deque_instance
            def emit(self, record): self.deque_instance.append(self.format(record))
        handler = DequeLogHandler(self.log_deque); handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'); handler.setFormatter(formatter)
        # 确保只添加一次 handler
        if not any(isinstance(h, DequeLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(handler)
            self.logger.propagate = False # 可选：防止日志重复输出到根 logger

    async def _log_portfolio_status(self):
        self.logger.info("--- [Status Update] Portfolio ---")
        self.logger.info(f"Total Equity: {self.portfolio.equity:.2f} USDT, Cash: {self.portfolio.cash:.2f} USDT")
        initial_capital_for_calc = self.initial_capital
        if initial_capital_for_calc <= 0:
             performance_percent = 0.0
             self.logger.warning("Initial capital is zero or negative, cannot calculate performance percentage.")
        else:
             performance_percent = (self.portfolio.equity / initial_capital_for_calc - 1) * 100
        self.logger.info(f"Overall Performance: {performance_percent:.2f}% (based on initial capital: {initial_capital_for_calc:.2f})")

    # --- _gather_all_market_data V44 版本 (计算指标) ---
    async def _gather_all_market_data(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        self.logger.info("Gathering full market data including indicators...")
        market_indicators_data: Dict[str, Dict[str, Any]] = {}
        fetched_tickers: Dict[str, Any] = {}
        try:
            tasks = []
            for symbol in self.symbols:
                tasks.append(self.exchange.fetch_ohlcv(symbol, '3m', limit=100))
                tasks.append(self.exchange.fetch_ohlcv(symbol, '4h', limit=100))
                tasks.append(self.exchange.fetch_ticker(symbol))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 先收集所有 tickers
            fetched_tickers = {symbol: results[i*3+2] for i, symbol in enumerate(self.symbols) if not isinstance(results[i*3+2], Exception) and results[i*3+2]}

            chunk_size = 3
            for i, symbol in enumerate(self.symbols):
                start_index = i * chunk_size
                ohlcv_3m, ohlcv_4h, ticker = results[start_index : start_index + chunk_size]

                if any(isinstance(res, Exception) or not res for res in [ohlcv_3m, ohlcv_4h, ticker]):
                    self.logger.warning(f"Failed to fetch complete data for {symbol}. Skipping indicator calculation.")
                    continue

                market_indicators_data[symbol] = {'current_price': ticker.get('last')}

                try:
                    df_3m = pd.DataFrame(ohlcv_3m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'], unit='ms') # 转换时间戳
                    df_3m.set_index('timestamp', inplace=True) # 设置时间索引
                    if not df_3m.empty and len(df_3m) >= 14: # 确保数据足够计算 RSI
                        rsi_series = ta.rsi(df_3m['close'], length=14)
                        if rsi_series is not None and not rsi_series.empty: market_indicators_data[symbol]['rsi_14_3m'] = rsi_series.iloc[-1]
                    if not df_3m.empty and len(df_3m) >= 26: # 确保数据足够计算 MACD
                        macd_df = ta.macd(df_3m['close'], fast=12, slow=26, signal=9)
                        if macd_df is not None and not macd_df.empty:
                            if 'MACD_12_26_9' in macd_df.columns: market_indicators_data[symbol]['macd_3m'] = macd_df['MACD_12_26_9'].iloc[-1]
                            if 'MACDs_12_26_9' in macd_df.columns: market_indicators_data[symbol]['macd_signal_3m'] = macd_df['MACDs_12_26_9'].iloc[-1]
                            if 'MACDh_12_26_9' in macd_df.columns: market_indicators_data[symbol]['macd_hist_3m'] = macd_df['MACDh_12_26_9'].iloc[-1]
                except Exception as e_3m: self.logger.error(f"Error calculating 3m indicators for {symbol}: {e_3m}", exc_info=False)

                try:
                    df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
                    df_4h.set_index('timestamp', inplace=True)
                    if not df_4h.empty and len(df_4h) >= 50: # 确保数据足够计算 EMA
                        ema20_series = ta.ema(df_4h['close'], length=20)
                        ema50_series = ta.ema(df_4h['close'], length=50)
                        if ema20_series is not None and not ema20_series.empty: market_indicators_data[symbol]['ema_20_4h'] = ema20_series.iloc[-1]
                        if ema50_series is not None and not ema50_series.empty: market_indicators_data[symbol]['ema_50_4h'] = ema50_series.iloc[-1]
                    if not df_4h.empty and len(df_4h) >= 26: # 确保数据足够计算 MACD
                        macd_df_4h = ta.macd(df_4h['close'], fast=12, slow=26, signal=9)
                        if macd_df_4h is not None and not macd_df_4h.empty:
                            if 'MACD_12_26_9' in macd_df_4h.columns: market_indicators_data[symbol]['macd_4h'] = macd_df_4h['MACD_12_26_9'].iloc[-1]
                            if 'MACDs_12_26_9' in macd_df_4h.columns: market_indicators_data[symbol]['macd_signal_4h'] = macd_df_4h['MACDs_12_26_9'].iloc[-1]
                            if 'MACDh_12_26_9' in macd_df_4h.columns: market_indicators_data[symbol]['macd_hist_4h'] = macd_df_4h['MACDh_12_26_9'].iloc[-1]
                except Exception as e_4h: self.logger.error(f"Error calculating 4h indicators for {symbol}: {e_4h}", exc_info=False)

        except Exception as e: self.logger.error(f"Error gathering market data: {e}", exc_info=True)
        return market_indicators_data, fetched_tickers

    # --- _build_prompt V44 版本 (处理指标和实盘 UPL) ---
    def _build_prompt(self, market_data: Dict[str, Dict[str, Any]], portfolio_state: Dict, tickers: Dict) -> str:
        prompt = f"It has been {(time.time() - self.start_time)/60:.1f} minutes since you started trading...\n"
        prompt += "\n--- Market Data Overview ---\n"
        # 安全格式化函数
        def safe_format(value, precision):
            if isinstance(value, (int, float)) and not pd.isna(value):
                return f"{value:.{precision}f}"
            return "N/A"

        for symbol, d in market_data.items():
            if not d: continue
            symbol_short = symbol.split('/')[0]
            prompt += f"\n# {symbol_short} Data\n"
            prompt += f"current_price: {safe_format(d.get('current_price'), 4)}\n"
            prompt += f"rsi_14_3m: {safe_format(d.get('rsi_14_3m'), 2)}\n"
            prompt += f"macd_3m: {safe_format(d.get('macd_3m'), 4)} (Signal: {safe_format(d.get('macd_signal_3m'), 4)}, Hist: {safe_format(d.get('macd_hist_3m'), 4)})\n"
            prompt += f"ema_20_4h: {safe_format(d.get('ema_20_4h'), 4)}\n"
            prompt += f"ema_50_4h: {safe_format(d.get('ema_50_4h'), 4)}\n"
            prompt += f"macd_4h: {safe_format(d.get('macd_4h'), 4)} (Signal: {safe_format(d.get('macd_signal_4h'), 4)}, Hist: {safe_format(d.get('macd_hist_4h'), 4)})\n"

        prompt += "\n--- Account Information ---\n"
        initial_capital_for_calc = self.initial_capital
        if initial_capital_for_calc <= 0: perf_perc_str = "N/A (Invalid Initial Capital)"
        else: perf_perc = (self.portfolio.equity / initial_capital_for_calc - 1) * 100; perf_perc_str = f"{perf_perc:.2f}%"
        prompt += f"Total Return %: {perf_perc_str}\n"
        prompt += f"Available Cash: {portfolio_state.get('cash_usd', '0.00')}\n"
        prompt += f"Current Account Value: {portfolio_state.get('account_value_usd', '0.00')}\n"
        prompt += "Current Positions:\n"

        open_positions_str_list = []
        if self.is_live_trading:
            # 实盘模式: 从 position_manager 获取, 手动计算 UPL
            open_positions_live = self.portfolio.position_manager.get_all_open_positions() # V2 方法
            if not open_positions_live: open_positions_str_list.append("  You have no open positions.")
            else:
                for symbol, state in open_positions_live.items():
                    current_price = tickers.get(symbol, {}).get('last')
                    upl_str = "N/A"
                    # V2 state 包含 avg_entry_price 和 total_size
                    if current_price and state.get('avg_entry_price') and state.get('total_size'):
                        entry_price = state['avg_entry_price']
                        size = state['total_size']
                        side = state['side']
                        if side == 'long': upl = (current_price - entry_price) * size
                        elif side == 'short': upl = (entry_price - current_price) * size
                        else: upl = 0.0
                        upl_str = f"{upl:.2f}"

                    pos_str = (
                        f"  - {symbol.split(':')[0]}: Side={state['side'].upper()}, Size={state['total_size']:.4f}, Entry={state['avg_entry_price']:.4f}, " # 使用 V2 字段
                        f"UPL={upl_str}, TP={state.get('ai_suggested_take_profit', 'N/A')}, SL={state.get('ai_suggested_stop_loss', 'N/A')}, "
                        f"Invalidation_Condition='{state.get('invalidation_condition', 'N/A')}'"
                    )
                    open_positions_str_list.append(pos_str)
        else:
            # 模拟盘模式: 直接使用 portfolio_state 提供的字符串
            open_positions_str_list.append(f"  {portfolio_state.get('open_positions', 'You have no open positions.')}")
        prompt += "\n".join(open_positions_str_list)
        return prompt

    async def _get_ai_decision(self, system_prompt: str, user_prompt: str) -> dict:
        if not self.ai_analyzer: return {}
        return await self.ai_analyzer.get_ai_response(system_prompt, user_prompt)


    # --- [核心修改] _execute_decisions 支持新指令 ---
    async def _execute_decisions(self, decisions: list):
        tickers = {}
        if not self.is_live_trading:
             try: tickers = await self.exchange.fetch_tickers(self.symbols) # 模拟盘直接用 self.exchange
             except Exception as e: self.logger.error(f"获取 Tickers 失败 (模拟盘): {e}", exc_info=True); return

        for order in decisions:
            try:
                action = order.get('action')
                symbol = order.get('symbol')

                if not action or not symbol or symbol not in self.symbols:
                    self.logger.warning(f"跳过无效指令: {order}")
                    continue

                reason = order.get('reasoning', 'No reason provided.')

                # --- 处理全平指令 ---
                if action == "CLOSE":
                    if self.is_live_trading:
                        await self.portfolio.live_close(symbol, reason=reason)
                    else:
                        # 模拟盘需要检查是否存在
                        if self.portfolio.paper_positions.get(symbol):
                            current_price = tickers.get(symbol, {}).get('last')
                            if current_price: await self.portfolio.paper_close(symbol, current_price, reason=reason)
                            else: self.logger.error(f"模拟平仓失败：无法获取 {symbol} 的价格。")


                # --- 处理开仓/加仓指令 ---
                elif action in ["BUY", "SELL"]:
                    side = 'long' if action == 'BUY' else 'short'
                    try:
                        size = float(order.get('size'))
                        leverage = int(order.get('leverage'))
                        stop_loss = float(order.get('stop_loss'))
                        take_profit = float(order.get('take_profit'))
                        # 检查数值有效性
                        if size <= 0 or leverage <= 0: raise ValueError("Size 或 Leverage 无效")
                    except (ValueError, TypeError, KeyError) as e:
                        self.logger.error(f"跳过无效的 BUY/SELL 指令 (参数错误): {order}. 错误: {e}")
                        continue

                    invalidation_condition = order.get('invalidation_condition')

                    if self.is_live_trading:
                        # 实盘 live_open 函数现在能自动处理开新仓或加仓
                        await self.portfolio.live_open(
                            symbol, side, size, leverage,
                            reason=reason, stop_loss=stop_loss, take_profit=take_profit,
                            invalidation_condition=invalidation_condition
                        )
                    else:
                        # 模拟盘 paper_open 函数也能处理开新仓或加仓
                        current_price = tickers.get(symbol, {}).get('last') # 模拟盘需要价格
                        if current_price:
                             await self.portfolio.paper_open(
                                 symbol, side, size,
                                 price=current_price,
                                 leverage=leverage, reason=reason, stop_loss=stop_loss,
                                 take_profit=take_profit, invalidation_condition=invalidation_condition
                             )
                        else:
                             self.logger.error(f"模拟开仓/加仓失败：无法获取 {symbol} 的价格。")


                # --- [新增] 处理部分平仓指令 ---
                elif action == "PARTIAL_CLOSE":
                    try:
                        # 优先使用 size_percent，如果不存在则尝试 size_absolute
                        size_percent = order.get('size_percent')
                        size_absolute = order.get('size_absolute')
                        if size_percent is not None:
                             size_to_close_percent = float(size_percent)
                             if not (0 < size_to_close_percent < 1): raise ValueError("size_percent 必须在 0 和 1 之间")
                             size_to_close_absolute = None # 标记使用百分比
                        elif size_absolute is not None:
                             size_to_close_absolute = float(size_absolute)
                             if size_to_close_absolute <= 0: raise ValueError("size_absolute 必须大于 0")
                             size_to_close_percent = None # 标记使用绝对值
                        else:
                             raise ValueError("必须提供 size_percent 或 size_absolute")

                    except (ValueError, TypeError, KeyError) as e:
                        self.logger.error(f"跳过无效的 PARTIAL_CLOSE 指令 (参数错误): {order}. 错误: {e}")
                        continue

                    if self.is_live_trading:
                        await self.portfolio.live_partial_close(
                            symbol,
                            size_percent=size_to_close_percent,
                            size_absolute=size_to_close_absolute,
                            reason=reason
                        )
                    else:
                        # 模拟盘也需要实现部分平仓逻辑
                        current_price = tickers.get(symbol, {}).get('last')
                        if current_price:
                            await self.portfolio.paper_partial_close(
                                symbol,
                                current_price, # 模拟盘需要价格
                                size_percent=size_to_close_percent,
                                size_absolute=size_to_close_absolute,
                                reason=reason
                            )
                        else:
                             self.logger.error(f"模拟部分平仓失败：无法获取 {symbol} 的当前价格。")

                # --- 其他未知指令 ---
                else:
                    self.logger.warning(f"收到未知的 AI 指令 action: {action} in {order}")

            except Exception as e:
                # 捕获循环内的其他错误，防止中断整个执行流程
                self.logger.error(f"处理 AI 指令时发生意外错误: {order}. 错误: {e}", exc_info=True)


    # --- _check_significant_indicator_change 到 _check_and_execute_hard_stops 保持不变 ---
    async def _check_significant_indicator_change(self, ohlcv_15m: list) -> (bool, str):
        try:
            if len(ohlcv_15m) < 30: return False, "" # MACD 需要足够数据
            df = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            macd_df = df.ta.macd(fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty and len(macd_df) >= 2:
                macd_line = macd_df['MACD_12_26_9']
                signal_line = macd_df['MACDs_12_26_9']
                # 检查金叉
                if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
                    return True, "15m MACD Golden Cross"
                # 检查死叉
                if macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
                    return True, "15m MACD Dead Cross"
            return False, ""
        except Exception as e:
            self.logger.error(f"Error checking indicator change: {e}", exc_info=False); return False, "" # 简化日志

    async def _check_market_volatility_spike(self, ohlcv_1h: list) -> (bool, str):
        try:
            if len(ohlcv_1h) < 2: return False, ""
            # 使用倒数第二根K线计算波动
            prev_candle = ohlcv_1h[-2]
            open_price, close_price = prev_candle[1], prev_candle[4]
            if open_price > 0:
                price_change_percent = abs(close_price - open_price) / open_price
                if price_change_percent >= (settings.AI_VOLATILITY_TRIGGER_PERCENT / 100.0):
                    direction = 'up' if close_price > open_price else 'down'
                    return True, f"1-hour price spike of {price_change_percent:.2%} ({direction})"
            return False, ""
        except Exception as e:
            self.logger.error(f"Error checking market volatility: {e}", exc_info=False); return False, "" # 简化日志

    async def _check_and_execute_hard_stops(self):
        if self.is_live_trading: return False
        self.logger.info("Checking for hard Take Profit / Stop Loss triggers...")
        positions_to_close = []
        try: tickers = await self.exchange.fetch_tickers(self.symbols) # 模拟盘直接用 self.exchange
        except Exception as e: self.logger.error(f"硬止损检查失败：无法获取 Tickers: {e}"); return False

        for symbol, pos in list(self.portfolio.paper_positions.items()):
            if not pos: continue
            current_price = tickers.get(symbol, {}).get('last')
            if not current_price: self.logger.warning(f"硬止损检查跳过：无法获取 {symbol} 的价格。"); continue
            side = pos['side']; sl = pos.get('stop_loss'); tp = pos.get('take_profit')
            if side == 'long':
                if sl and current_price <= sl: positions_to_close.append((symbol, current_price, f"Hard SL @ {sl}"))
                elif tp and current_price >= tp: positions_to_close.append((symbol, current_price, f"Hard TP @ {tp}"))
            elif side == 'short':
                if sl and current_price >= sl: positions_to_close.append((symbol, current_price, f"Hard SL @ {sl}"))
                elif tp and current_price <= tp: positions_to_close.append((symbol, current_price, f"Hard TP @ {tp}"))
        for symbol, price, reason in positions_to_close:
            self.logger.warning(f"AUTO-CLOSING (Paper): {symbol} | Reason: {reason}")
            await self.portfolio.paper_close(symbol, price, reason)
        return len(positions_to_close) > 0


    # --- [核心修改] run_cycle 更新 system_prompt ---
    async def run_cycle(self):
        self.logger.info("="*20 + " Starting new AI decision cycle " + "="*20)
        self.invocation_count += 1

        if not self.is_live_trading:
            await self._check_and_execute_hard_stops()

        # 调用修改后的 gather 函数
        market_data, tickers = await self._gather_all_market_data()
        portfolio_state = self.portfolio.get_state_for_prompt() # 这个返回的结构保持不变
        # 调用修改后的 build 函数, 传入 tickers
        user_prompt_string = self._build_prompt(market_data, portfolio_state, tickers)


        # <-- [核心修改] 更新 system_prompt 以支持新指令和分析细节 -->
        system_prompt = f"""
        You are an expert quantitative trading AI with a strong focus on risk management. Your task is to analyze the market, manage positions based on pre-defined rules, and generate new, well-defined trading orders.

        **Core Mandates & Rules:**
        1.  **Rule-Based Position Management:** For every open position, you MUST check its `Invalidation_Condition`. If this condition is met by the current market data, you MUST issue a `CLOSE` (full close) order. This is your top priority.

        2.  **Risk Management First (CRITICAL):** Your primary goal is capital preservation. You MUST NOT use all your available cash.
            -   **Single Position Sizing (Open/Add):** When opening a new position OR adding to an existing one, you MUST calculate the `size` so the required margin for THIS ORDER is a small fraction of your **Available Cash**.
            -   **CALCULATION FORMULA (MANDATORY):** You MUST follow this formula for EACH `BUY`/`SELL` order:
                1.  Choose a `risk_percent` (e.g., 0.05 for 5%, 0.1 for 10%). This MUST be less than or equal to 0.1 (10%).
                2.  `desired_margin_usd = Available Cash * risk_percent`.
                3.  `size = (desired_margin_usd * leverage) / current_price`.
            -   **Example:** If Available Cash is $10,000, leverage is 10x, and BTC price is $60,000, choosing 10% risk means `size = (10000 * 0.1) * 10 / 60000 = 0.1667`.
            -   **Total Exposure:** The sum of all margins for all open positions should generally not exceed 50-60% of your total equity. Consider this when deciding whether to open/add.

        3.  **Complete Trade Plans (Open/Add):** Every new `BUY` or `SELL` order is a complete plan. You MUST provide: `take_profit`, `stop_loss`, `invalidation_condition`. These will apply to the *entire* position after the order executes (overwriting previous settings if adding).

        **MANDATORY OUTPUT FORMAT:**
        Your entire response must be a single JSON object with two keys: "chain_of_thought" and "orders".

        1.  `"chain_of_thought"` (string): A multi-line string containing your detailed analysis in English. It MUST follow this template precisely:
            ```
            My Current Assessment & Actions

            Okay, here's what I'm thinking. [Your general market overview, current P/L, and a summary of your open positions goes here. Briefly mention current total margin usage vs equity if relevant.]

            Let's break down each position:
            1. [SYMBOL] ([SIDE]): UPL: [Current Unrealized PNL (e.g., +123.45 or N/A if live)]. Price vs 4H EMA20/50: [e.g., Above both, crossing down]. Key Indicators: [e.g., 3m RSI=XX, 4h MACD=YY]. Invalidation Check: [Check condition vs current data, e.g., Condition 'Price closes below 68000 on 3m candle' vs Current Price 69000 -> Not Met]. Decision: [Hold/Close/Partial Close/Add + Brief Reason].
            2. [SYMBOL] ([SIDE]): UPL: [...]. Price vs 4H EMA20/50: [...]. Key Indicators: [...]. Invalidation Check: [...]. Decision: [Hold/Close/Partial Close/Add + Brief Reason].
            ... [Add a line for each open position] ...

            New Trade Opportunities Analysis:
            Available Margin for New Trades: [Calculate based on Available Cash and risk rules, e.g., approx. $1000 available for 10% risk].
            [Analyze potential entry signals for relevant symbols (BTC, ETH, etc.) based on indicators like EMA crosses, RSI, MACD provided in the Market Data Overview. State clearly if a strong signal is found or not for each considered symbol.]
            Example: BTC shows 4H EMAs bullish but 3m RSI (currently 75.2) is overbought, MACD_3m is declining. Signal is mixed, wait. ETH 4H EMAs bearish, 3m MACD crossed below signal. Potential short signal forming, but wait for confirmation. SOL is ranging. No strong entry signals found this cycle.

            In summary, [Your final, concise decision overview for this cycle goes here. This text will be shown on the UI. Example: Holding existing positions as invalidation conditions are not met. Monitoring ETH for a potential short entry if price breaks below support.]
            ```

        2.  `"orders"` (list): A list of JSON objects for trades. Empty list `[]` if holding all.

        **Order Object Rules:**
        -   **To Open or Add:** `{{"action": "BUY", "symbol": "...", "size": [CALCULATED_SIZE], "leverage": 10, "take_profit": ..., "stop_loss": ..., "invalidation_condition": "...", "reasoning": "Calculation: size = (Cash * risk%) * lev / price = ..."}}`
            *(System automatically handles if it's opening or adding)*
        -   **To Close Fully:** `{{"action": "CLOSE", "symbol": "...", "reasoning": "Invalidation met / SL hit / TP hit / Manual decision..."}}`
        -   **To Close Partially (Take Profit):** `{{"action": "PARTIAL_CLOSE", "symbol": "...", "size_percent": 0.5, "reasoning": "Taking 50% profit near resistance..."}}`
            * Use `"size_percent"` (0.0 to 1.0, e.g., 0.5 for 50%) to specify the percentage of the current position size to close.
            * Alternatively, use `"size_absolute"` (e.g., 0.1) to specify an absolute amount to close. Provide ONLY ONE of `size_percent` or `size_absolute`.
        -   **To Hold:** Do NOT include in `orders`. Reasoning must be in `chain_of_thought`.
        -   **Symbol Validity:** `symbol` MUST be one of {self.symbols}.
        """
        # <-- [核心修改结束] -->

        self.logger.info("Getting AI trading decision with new rule-based prompt...")
        ai_decision = await self._get_ai_decision(system_prompt, user_prompt_string)

        original_chain_of_thought = ai_decision.get("chain_of_thought", "AI did not provide a chain of thought.")
        orders = ai_decision.get("orders", [])
        self.logger.warning("--- AI Chain of Thought (Original English) ---")
        self.logger.warning(original_chain_of_thought)

        # --- 摘要提取邏輯 (保持不變) ---
        summary_for_ui = "Waiting for AI summary..."
        summary_keyword_pattern = re.compile(r"In summary,", re.IGNORECASE)
        parts = summary_keyword_pattern.split(original_chain_of_thought, maxsplit=1)
        if len(parts) > 1:
            english_summary = parts[1].strip().lstrip(' :').rstrip('`')
            if english_summary:
                chinese_summary = await self.ai_analyzer.get_translation_response(english_summary)
                summary_for_ui = chinese_summary
            else:
                summary_for_ui = "AI 提供了摘要，但內容為空。"
        else:
            self.logger.warning("AI 'chain_of_thought' 中未找到 'In summary,' 關鍵字。")
            summary_for_ui = "AI 本輪未提供策略摘要。正在檢查日誌..."
        self.last_strategy_summary = summary_for_ui

        if orders:
            self.logger.info(f"AI has proposed {len(orders)} order(s), executing...")
            # --- 調用更新後的 _execute_decisions ---
            await self._execute_decisions(orders)
        else:
            self.logger.info("AI did not propose any orders this cycle.")

        self.logger.info("="*20 + " AI decision cycle finished " + "="*20 + "\n")

    # --- start 方法保持不變 ---
    async def start(self):
        print(f"--- AlphaTrader Start Check: self.is_live_trading = {self.is_live_trading} ---") # Debug print
        self.logger.warning(f"🚀 AI Alpha Trader starting! Mode: {'LIVE' if self.is_live_trading else 'PAPER'}")

        if self.is_live_trading:
            self.logger.warning("!!! 運行在實盤模式 !!! 啟動時同步狀態...")
            await self.portfolio.sync_state()
            self.logger.warning("!!! 實盤狀態同步完成 !!!")

        while True:
            try:
                await self.portfolio.sync_state()
                await self._log_portfolio_status()

                trigger_ai_analysis, reason_for_trigger, current_time = False, "", time.time()
                analysis_interval = settings.ALPHA_ANALYSIS_INTERVAL_SECONDS

                if current_time - self.last_run_time >= analysis_interval:
                    trigger_ai_analysis, reason_for_trigger = True, "Scheduled Analysis"

                if not trigger_ai_analysis:
                    representative_symbol = self.symbols[0]
                    # 获取 K 线数据用于事件触发器检查
                    try:
                        ohlcv_15m, ohlcv_1h = await asyncio.gather(
                            self.exchange.fetch_ohlcv(representative_symbol, '15m', limit=150), # 确保足够数据
                            self.exchange.fetch_ohlcv(representative_symbol, '1h', limit=20)
                        )
                    except Exception as e_fetch:
                        self.logger.error(f"事件触发器检查：获取 K 线数据失败: {e_fetch}")
                        ohlcv_15m, ohlcv_1h = [], [] # 出错时给空列表

                    cooldown = settings.AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES * 60
                    if current_time - self.last_event_trigger_ai_time > cooldown:
                        event, reason = await self._check_significant_indicator_change(ohlcv_15m)
                        if not event:
                            event, reason = await self._check_market_volatility_spike(ohlcv_1h)
                        if event:
                            trigger_ai_analysis, reason_for_trigger = True, reason

                if trigger_ai_analysis:
                    self.logger.warning(f"🔥 AI analysis triggered! Reason: {reason_for_trigger}")
                    if reason_for_trigger != "Scheduled Analysis": self.last_event_trigger_ai_time = current_time
                    await self.run_cycle()
                    self.last_run_time = current_time

                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.logger.warning("Task was cancelled, shutting down..."); break
            except Exception as e:
                self.logger.critical(f"A fatal error occurred in the main loop: {e}", exc_info=True); await asyncio.sleep(60)
