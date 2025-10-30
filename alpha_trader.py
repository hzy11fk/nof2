# 文件: alpha_trader.py (V45.16 - Token 优化: 原生中文摘要)

import logging
import asyncio
import time
import json
import pandas as pd
import pandas_ta as ta
import re
from collections import deque
# import Levenshtein # [V45.16 移除]
from config import settings, futures_settings
from alpha_ai_analyzer import AlphaAIAnalyzer
from alpha_portfolio import AlphaPortfolio # 假设 V23.4 或更高
from datetime import datetime
from typing import Tuple, Dict, Any, Set, Optional # [V45.16 移除] Set

try:
    import pandas as pd
except ImportError:
    pd = None

class AlphaTrader:
    # --- [V45.16 核心修改] System Prompt 模板 (要求原生中文摘要) ---
    SYSTEM_PROMPT_TEMPLATE = """
    You are a **profit-driven, analytical, and disciplined** quantitative trading AI. Your primary goal is to **generate and secure realized profit**. You are not a gambler; you are a calculating strategist.

    **You ONLY execute a trade (BUY, SELL, PARTIAL_CLOSE) if you have a high-confidence assessment that the action will lead to profit.** A medium or low-confidence signal means you WAIT.

    Your discipline is demonstrated by strict adherence to the risk management rules below, which are your foundation for sustained profitability.

    **Core Mandates & Rules:**
    1.  **Rule-Based Position Management:** For every open position, you MUST check its `Invalidation_Condition`. If this condition is met by the current market data, you MUST issue a `CLOSE` (full close) order. This is your top priority for existing positions.

    2.  **Risk Management Foundation (CRITICAL):** Profit is the goal, but capital preservation is the foundation. **You will fail your objective if you are reckless.** Therefore, you MUST strictly follow these rules to manage your capital and stay in the game:
        -   **Single Position Sizing (Open/Add):** When opening a new position OR adding to an existing one, you MUST calculate the size so the required margin for THIS ORDER is a small fraction of your Available Cash.
        -   **CRITICAL MINIMUM MARGIN RULE (MANDATORY):** The **margin required** for any new `BUY` or `SELL` order (`final_desired_margin` after check) MUST be at least **6.0 USDT**.
        -   **CALCULATION FORMULA (MANDATORY):** You MUST follow this formula for EACH `BUY`/`SELL` order:
            1.  Choose a `risk_percent` (e.g., 0.05 for 5%, 0.1 for 10%). This MUST be less than or equal to 0.1 (10%).
            2.  `calculated_desired_margin = Available Cash * risk_percent`.
            3.  **Check Minimum Margin:**
                -   IF `calculated_desired_margin` >= 6.0: `final_desired_margin = calculated_desired_margin`.
                -   IF `calculated_desired_margin` < 6.0: You have two choices:
                    a) (If Confidence is HIGH): Set `final_desired_margin = 6.0` (to meet the minimum margin).
                    b) (If Confidence is Medium/Low): **Abort the trade.** Do not create an order.
            4.  `size = (final_desired_margin * leverage) / current_price`.
        -   **Example (Good):** Available Cash is $100, leverage 10x, risk 10%. `calculated_margin = 100 * 0.1 = 10.0`. `10.0` >= 6.0, so `final_margin = 10.0`. `size = (10.0 * 10) / price`.
        -   **Example (Bad, but fixed):** Available Cash is $50, leverage 10x, risk 10%. `calculated_margin = 50 * 0.1 = 5.0`. `5.0` < 6.0. If confidence is HIGH, set `final_margin = 6.0`. `size = (6.0 * 10) / price`. If confidence is LOW, abort.
        -   **Total Exposure:** The sum of all margins for all open positions should generally not exceed 50-60% of your total equity. Consider this when deciding whether to open/add.
        -   **Correlation Control:** Avoid holding highly correlated assets in the same direction.

    3.  **Complete Trade Plans (Open/Add):** Every new `BUY` or `SELL` order is a complete plan. You MUST provide: `take_profit`, `stop_loss`, `invalidation_condition`. These will apply to the *entire* position after the order executes.
        -   **Profit-Taking Strategy:** Consider using multiple take-profit levels.
        -   **Trailing Stop Loss:** Consider implementing a trailing stop loss when in significant profit (+3% or more).

    4.  **Market State Recognition:** Continuously assess the overall market regime (Trending Bullish/Bearish, Ranging, High/Low Volatility).

    **Multi-Timeframe Confirmation Requirement (CRITICAL):**
    - You MUST analyze and confirm signals across available timeframes: **5min, 15min, 1hour, and 4hour**. (1min removed)
    - **High-Confidence Signal Definition:** A signal is only high-confidence when it shows alignment across **at least 3** of the available timeframes with consistent direction and momentum.
    - **Timeframe Hierarchy:** Give more weight to longer timeframes (**4h > 1h > 15min > 5min**) for trend direction, but use shorter timeframes for precise entry timing. (1min removed)
    - **Volume Confirmation:** Significant price moves MUST be confirmed by above-average volume (volume_ratio > 1.2) on the corresponding timeframe.

    **Psychological Safeguards:**
    - Confirmation Bias Protection: Seek counter-evidence.
    - Overtrading Protection: No high-confidence signal = WAIT.
    - Loss Aversion Protection: Stick to stop losses.

    **MANDATORY OUTPUT FORMAT:**
    Your entire response must be a single JSON object with two keys: "chain_of_thought" and "orders".

    1.  `"chain_of_thought"` (string): A multi-line string containing your detailed analysis in English. It MUST follow this template precisely:
        ```
        My Current Assessment & Actions

        Market State Analysis:
        - Regime: [Trending Bullish/Trending Bearish/Ranging/High Volatility/Low Volatility]
        - Key Support/Resistance Levels: [Identify major technical levels for relevant symbols]
        - Volume Analysis: [Assess volume confirmation for price movements across timeframes]
        - Market Sentiment: [If available, mention fear/greed index or similar indicators]

        Portfolio Overview:
        Total Equity: $X, Available Cash: $Y, Current Margin Usage: Z%
        Current Market Correlation Assessment: [Evaluate if positions are overly correlated]

        Let's break down each position:
        1. [SYMBOL] ([SIDE]):
           UPL: [Current Unrealized PNL]
           Multi-Timeframe Analysis: [Brief assessment across 5min, 15min, 1h, 4h]
           Volume Confirmation: [Volume analysis for current move across timeframes]
           Key Indicators: [RSI, MACD, plus volume and sentiment if available]
           Invalidation Check: [Check condition vs current data]
           Profit Management: [Assess if partial close is warranted for profit locking]
           Trailing Stop Assessment: [Evaluate if trailing stop should be implemented]
           Decision: [Hold/Close/Partial Close/Add + Reason]

        ... [Repeat for each open position] ...

        New Trade Opportunities Analysis:
        Available Margin for New Trades: [Calculate based on Available Cash and risk rules]
        Correlation Check: [Ensure new trades don't over-concentrate in correlated assets]

        Multi-Timeframe Signal Requirements (Must meet 3+ factors on 5m, 15m, 1h, 4h):
        - Trend alignment across multiple timeframes (at least 3)
        - Volume confirmation of price movement (volume_ratio > 1.2)
        - Technical breakout/breakdown with conviction
        - Absence of strong counter-evidence across timeframes

        Specific Multi-Timeframe Opportunity Analysis:
        [For each symbol, analyze BOTH long and short scenarios with timeframe-by-timeframe confidence ratings based on 5m, 15m, 1h, 4h]
        [EXAMPLES:]
        BTC Multi-Timeframe Assessment:
        - 4h Trend: Bullish | 1h Momentum: Strong | 15min Setup: Pullback | 5min Trigger: Confirmed
        Signal Confluence Score: 4/4 | Final Confidence: High - CONSIDER LONG

        ETH Multi-Timeframe Assessment:
        - 4h Trend: Bearish | 1h Momentum: Weak | 15min Setup: Consolidating | 5min Trigger: Weak
        Signal Confluence Score: 1/4 | Final Confidence: Low - NO TRADE

        In summary, [**Key Instruction: Please provide your final concise decision overview directly here, in Chinese.**Final concise decision overview. Example: Implementing 25% partial close on profitable BTC position based on multi-timeframe analysis. Opening small ETH long with tight stop after multi-timeframe confirmation. Monitoring correlation exposure.]
        ```

    2.  `"orders"` (list): A list of JSON objects for trades. Empty list `[]` if holding all.

    **Order Object Rules:**
    -   **To Open or Add:** `{{"action": "BUY", "symbol": "...", "size": [CALCULATED_SIZE], "leverage": 10, "take_profit": ..., "stop_loss": ..., "invalidation_condition": "...", "reasoning": "Calculation: final_margin={{final_margin_usd:.2f}} (must be >= 6.0). size=(Final Margin)*lev/price=... Multi-TF confirm: [...]. Corr check: [...]"}}`
    -   **To Close Fully:** `{{"action": "CLOSE", "symbol": "...", "reasoning": "Invalidation met / SL hit / TP hit / Manual decision..."}}`
    -   **To Close Partially (Take Profit):** `{{"action": "PARTIAL_CLOSE", "symbol": "...", "size_percent": 0.5, "reasoning": "Taking 50% profit near resistance..."}}` (or `size_absolute`)
    -   **To Update Stop Loss (Trailing):** `{{"action": "UPDATE_STOPLOSS", "symbol": "...", "new_stop_loss": ..., "reasoning": "Moving stop loss to protect profits..."}}`
    -   **To Hold:** Do NOT include in `orders`. Reasoning must be in `chain_of_thought`.
    -   **Symbol Validity:** `symbol` MUST be one of {symbol_list}.

    **Remember:** Quality over quantity.
    """
    # --- [模板结束] ---

    # --- [V45.16 移除] 不再需要翻译相关常量 ---
    # TRANSLATION_LENGTH_CHANGE_THRESHOLD = 0.3
    # TRANSLATION_SIMILARITY_THRESHOLD = 0.7
    # TRANSLATION_KEYWORDS = { ... }
    # --- [移除结束] ---

    def __init__(self, exchange):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exchange = exchange
        self.symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT", "XRP/USDT:USDT"]
        self.portfolio = AlphaPortfolio(exchange, self.symbols) # 假设 V23.4+
        self.ai_analyzer = AlphaAIAnalyzer(exchange, "ALPHA_TRADER")
        self.is_live_trading = settings.ALPHA_LIVE_TRADING
        self.start_time = time.time(); self.invocation_count = 0; self.last_run_time = 0; self.last_event_trigger_ai_time = 0
        self.log_deque = deque(maxlen=50); self._setup_log_handler()
        self.last_strategy_summary = "Initializing..."
        self.initial_capital = settings.ALPHA_LIVE_INITIAL_CAPITAL if self.is_live_trading else settings.ALPHA_PAPER_CAPITAL
        self.logger.info(f"Initialized with Initial Capital: {self.initial_capital:.2f} USDT")
        self.formatted_symbols = ", ".join(f'"{s}"' for s in self.symbols)
        # --- [V45.16 移除] 不再需要 last_translated_english_summary ---
        # self.last_translated_english_summary: Optional[str] = None
        # --- [移除结束] ---
        if hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
        else:
            if hasattr(self.portfolio, 'exchange') and isinstance(self.portfolio.exchange, object): self.client = self.portfolio.exchange; self.logger.warning("Portfolio missing 'client', falling back.")
            else: self.client = self.exchange; self.logger.warning("Portfolio missing 'client', using exchange directly.")

    def _setup_log_handler(self):
        """配置内存日志记录器"""
        class DequeLogHandler(logging.Handler):
            def __init__(self, deque_instance): super().__init__(); self.deque_instance = deque_instance
            def emit(self, record): self.deque_instance.append(self.format(record))
        handler = DequeLogHandler(self.log_deque); handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'); handler.setFormatter(formatter)
        if not any(isinstance(h, DequeLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(handler); self.logger.propagate = False

    async def _log_portfolio_status(self):
        """记录当前的投资组合状态"""
        self.logger.info("--- [Status Update] Portfolio ---")
        equity_val = float(self.portfolio.equity) if self.portfolio.equity is not None else 0.0
        cash_val = float(self.portfolio.cash) if self.portfolio.cash is not None else 0.0
        self.logger.info(f"Total Equity: {equity_val:.2f} USDT, Cash: {cash_val:.2f} USDT")
        initial_capital_for_calc = self.initial_capital; performance_percent = 0.0
        if initial_capital_for_calc > 0: performance_percent = (equity_val / initial_capital_for_calc - 1) * 100
        else: self.logger.warning("Initial capital <= 0, cannot calc performance %.")
        self.logger.info(f"Overall Performance: {performance_percent:.2f}% (Initial: {initial_capital_for_calc:.2f})")

    async def _gather_all_market_data(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """[V45.14] 获取市场数据 (5m, 15m, 1h, 4h)"""
        self.logger.info("Gathering multi-TF market data (5m, 15m, 1h, 4h)...")
        market_indicators_data: Dict[str, Dict[str, Any]] = {}
        fetched_tickers: Dict[str, Any] = {}
        try:
            timeframes = ['5m', '15m', '1h', '4h'] # 1m 移除
            tasks = []
            for symbol in self.symbols:
                for timeframe in timeframes: tasks.append(self.exchange.fetch_ohlcv(symbol, timeframe, limit=100))
                tasks.append(self.exchange.fetch_ticker(symbol))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_timeframes = len(timeframes); tasks_per_symbol = total_timeframes + 1
            for i, symbol in enumerate(self.symbols):
                start_index = i * tasks_per_symbol; symbol_ohlcv_results = results[start_index:start_index + total_timeframes]
                ticker_result = results[start_index + total_timeframes]
                if not isinstance(ticker_result, Exception) and ticker_result and ticker_result.get('last') is not None:
                    fetched_tickers[symbol] = ticker_result; market_indicators_data[symbol] = {'current_price': ticker_result.get('last')}
                else: market_indicators_data[symbol] = {'current_price': None}; self.logger.warning(f"Failed fetch ticker/price for {symbol}")
                for j, timeframe in enumerate(timeframes):
                    ohlcv_data = symbol_ohlcv_results[j]
                    if isinstance(ohlcv_data, Exception) or not ohlcv_data: self.logger.warning(f"Failed fetch {timeframe} for {symbol}"); continue
                    try:
                        df = pd.DataFrame(ohlcv_data, columns=['ts', 'o', 'h', 'l', 'c', 'v']); df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore'); df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
                        if df.empty: continue
                        prefix = f"{timeframe.replace('m', 'min').replace('h', 'hour')}_"
                        if len(df)>=14: rsi=ta.rsi(df['c'], 14); market_indicators_data[symbol][f'{prefix}rsi_14']=rsi.iloc[-1] if rsi is not None and not rsi.empty else None
                        if len(df)>=26: macd=ta.macd(df['c'], 12, 26, 9);
                        if macd is not None and not macd.empty:
                            if 'MACD_12_26_9' in macd.columns: market_indicators_data[symbol][f'{prefix}macd']=macd['MACD_12_26_9'].iloc[-1]
                            if 'MACDs_12_26_9' in macd.columns: market_indicators_data[symbol][f'{prefix}macd_signal']=macd['MACDs_12_26_9'].iloc[-1]
                            if 'MACDh_12_26_9' in macd.columns: market_indicators_data[symbol][f'{prefix}macd_hist']=macd['MACDh_12_26_9'].iloc[-1]
                        if len(df)>=50: ema20=ta.ema(df['c'], 20); ema50=ta.ema(df['c'], 50);
                        if ema20 is not None and not ema20.empty: market_indicators_data[symbol][f'{prefix}ema_20']=ema20.iloc[-1]
                        if ema50 is not None and not ema50.empty: market_indicators_data[symbol][f'{prefix}ema_50']=ema50.iloc[-1]
                        if len(df)>=2: cur_v=df['v'].iloc[-1]; avg_v=df['v'].tail(20).mean() if len(df)>=20 else df['v'].mean();
                        if avg_v>0: market_indicators_data[symbol][f'{prefix}volume_ratio']=cur_v/avg_v
                        prev_close = df['c'].iloc[-2];
                        if prev_close > 0: chg=(df['c'].iloc[-1]-prev_close)/prev_close*100; market_indicators_data[symbol][f'{prefix}price_change_pct']=chg
                        else: market_indicators_data[symbol][f'{prefix}price_change_pct'] = 0.0
                        if len(df)>=20: market_indicators_data[symbol][f'{prefix}recent_high']=df['h'].tail(20).max(); market_indicators_data[symbol][f'{prefix}recent_low']=df['l'].tail(20).min()
                    except Exception as e: self.logger.error(f"Error calc {timeframe} indicators for {symbol}: {e}", exc_info=False)
        except Exception as e: self.logger.error(f"Error gathering market data: {e}", exc_info=True)
        return market_indicators_data, fetched_tickers

    def _build_prompt(self, market_data: Dict[str, Dict[str, Any]], portfolio_state: Dict, tickers: Dict) -> str:
        """[V45.14] 构建 User Prompt, 移除 1m, 优化精度"""
        prompt = f"It has been {(time.time() - self.start_time)/60:.0f} minutes since start.\n"
        prompt += "\n--- Multi-Timeframe Market Data Overview (5m, 15m, 1h, 4h) ---\n"
        def safe_format(value, precision, is_rsi=False):
            is_na = pd.isna(value) if pd else value is None
            if isinstance(value, (int, float)) and not is_na:
                if is_rsi: return f"{round(value):d}"
                try: return f"{value:.{precision}f}"
                except (ValueError, TypeError): return str(value) if value is not None else "N/A"
            return "N/A"
        for symbol, d in market_data.items():
            if not d: continue
            symbol_short = symbol.split('/')[0]
            prompt += f"\n# {symbol_short} Multi-TF Analysis\n"
            prompt += f"Price: {safe_format(d.get('current_price'), 2)}\n"
            timeframes = ['5min', '15min', '1hour', '4hour'] # 1m 移除
            for tf in timeframes:
                prompt += f"\n[{tf.upper()}]\n"
                prompt += f" RSI:{safe_format(d.get(f'{tf}_rsi_14'), 0, is_rsi=True)}|"
                prompt += f"MACD:{safe_format(d.get(f'{tf}_macd'), 4)}|"
                prompt += f"Sig:{safe_format(d.get(f'{tf}_macd_signal'), 4)}|"
                prompt += f"Hist:{safe_format(d.get(f'{tf}_macd_hist'), 4)}\n"
                prompt += f" EMA20:{safe_format(d.get(f'{tf}_ema_20'), 3)}|"
                prompt += f"EMA50:{safe_format(d.get(f'{tf}_ema_50'), 3)}|"
                prompt += f"VolR:{safe_format(d.get(f'{tf}_volume_ratio'), 1)}x\n"
                prompt += f" Hi:{safe_format(d.get(f'{tf}_recent_high'), 2)}|"
                prompt += f"Lo:{safe_format(d.get(f'{tf}_recent_low'), 2)}|"
                prompt += f"Chg:{safe_format(d.get(f'{tf}_price_change_pct'), 1)}%\n"
            prompt += "-----\n"
        prompt += "\n--- Account Info ---\n"
        prompt += f"Return%: {portfolio_state.get('performance_percent', 'N/A')}\n"
        prompt += f"Cash: {portfolio_state.get('cash_usd', 'N/A')}\n"
        prompt += f"Value: {portfolio_state.get('account_value_usd', 'N/A')}\n"
        prompt += "Positions:\n"
        prompt += portfolio_state.get('open_positions', "No open positions.")
        return prompt

    async def _get_ai_decision(self, system_prompt: str, user_prompt: str) -> dict:
        """调用 AI 分析器"""
        if not self.ai_analyzer: return {}
        return await self.ai_analyzer.get_ai_response(system_prompt, user_prompt)

    async def _execute_decisions(self, decisions: list, market_data: Dict[str, Dict[str, Any]]):
        """[V45.14] 执行 AI 决策, 包含最低保证金 6U 修正"""
        MIN_MARGIN_USDT = 6.0
        for order in decisions:
            try:
                action = order.get('action'); symbol = order.get('symbol')
                if not action or not symbol or symbol not in self.symbols: self.logger.warning(f"跳过无效指令: {order}"); continue
                reason = order.get('reasoning', 'N/A'); current_price = market_data.get(symbol, {}).get('current_price')
                if not current_price or current_price <= 0: self.logger.error(f"无价格 {symbol}，跳过: {order}"); continue
                if action == "CLOSE":
                    if self.is_live_trading: await self.portfolio.live_close(symbol, reason=reason)
                    else: await self.portfolio.paper_close(symbol, current_price, reason=reason)
                elif action in ["BUY", "SELL"]:
                    side = 'long' if action == 'BUY' else 'short'; final_size = 0.0
                    try:
                        original_size = float(order.get('size')); leverage = int(order.get('leverage'))
                        stop_loss = float(order.get('stop_loss')); take_profit = float(order.get('take_profit'))
                        if original_size <= 0 or leverage <= 0: raise ValueError("Size/Lev无效")
                        intended_margin = (original_size * current_price) / leverage if leverage > 0 else 0.0
                        final_size = original_size; final_margin = intended_margin
                        if intended_margin < MIN_MARGIN_USDT:
                            self.logger.warning(f"!!! 硬控触发 (保证金) !!! AI订单 {symbol} 保证金 {intended_margin:.4f} < {MIN_MARGIN_USDT} USDT.")
                            final_margin = MIN_MARGIN_USDT
                            if leverage > 0 and current_price > 0:
                                final_size = (final_margin * leverage) / current_price
                                self.logger.warning(f"已修正保证金为 {MIN_MARGIN_USDT} USDT。新Size: {final_size:.8f}")
                            else: raise ValueError("无法重新计算 size (杠杆/价格无效)")
                        if final_size <= 0: raise ValueError("最终 size <= 0")
                    except (ValueError, TypeError, KeyError) as e: self.logger.error(f"跳过BUY/SELL参数/计算错误: {order}. Err: {e}"); continue
                    invalidation_condition = order.get('invalidation_condition')
                    if self.is_live_trading: await self.portfolio.live_open(symbol, side, final_size, leverage, reason=reason, stop_loss=stop_loss, take_profit=take_profit, invalidation_condition=invalidation_condition)
                    else: await self.portfolio.paper_open(symbol, side, final_size, price=current_price, leverage=leverage, reason=reason, stop_loss=stop_loss, take_profit=take_profit, invalidation_condition=invalidation_condition)
                elif action == "PARTIAL_CLOSE":
                    size_to_close_percent=None; size_to_close_absolute=None
                    try:
                        sp=order.get('size_percent'); sa=order.get('size_absolute')
                        if sp is not None: size_to_close_percent=float(sp)
                        elif sa is not None: size_to_close_absolute=float(sa)
                        else: raise ValueError("需提供 size_% 或 size_abs")
                        if size_to_close_percent is not None and not (0<size_to_close_percent<1): raise ValueError("size_% 需在 0-1")
                        if size_to_close_absolute is not None and size_to_close_absolute<=0: raise ValueError("size_abs 需 > 0")
                    except (ValueError,TypeError,KeyError) as e: self.logger.error(f"跳过PARTIAL_CLOSE参数错误: {order}. Err: {e}"); continue
                    if self.is_live_trading: await self.portfolio.live_partial_close(symbol, size_percent=size_to_close_percent, size_absolute=size_to_close_absolute, reason=reason)
                    else: await self.portfolio.paper_partial_close(symbol, current_price, size_percent=size_to_close_percent, size_absolute=size_to_close_absolute, reason=reason)
                elif action == "UPDATE_STOPLOSS":
                    new_stop_loss = 0.0
                    try:
                        nsl=order.get('new_stop_loss');
                        if nsl is None: raise ValueError("缺少 new_stop_loss")
                        new_stop_loss=float(nsl)
                        if new_stop_loss<=0: raise ValueError("无效止损价")
                    except (ValueError,TypeError,KeyError) as e: self.logger.error(f"跳过UPDATE_STOPLOSS参数错误: {order}. Err: {e}"); continue
                    if hasattr(self.portfolio, 'update_position_rules'): await self.portfolio.update_position_rules(symbol, stop_loss=new_stop_loss, reason=reason)
                    else: self.logger.warning(f"AI 尝试 UPDATE_STOPLOSS 但 portfolio 无 update_position_rules 方法。")
                else: self.logger.warning(f"收到未知 AI 指令 action: {action} in {order}")
            except Exception as e: self.logger.error(f"处理 AI 指令时意外错误: {order}. Err: {e}", exc_info=True)


    async def _check_significant_indicator_change(self, ohlcv_15m: list) -> Tuple[bool, str]:
        """检查 15m MACD 交叉"""
        try:
            if len(ohlcv_15m) < 30: return False, ""
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v']); df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore');
            macd_df = df.ta.macd(close=df['c'], fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty and len(macd_df) >= 2:
                macd=macd_df['MACD_12_26_9']; sig=macd_df['MACDs_12_26_9']
                if macd.iloc[-2]<sig.iloc[-2] and macd.iloc[-1]>sig.iloc[-1]: return True, "15m MACD Golden Cross"
                if macd.iloc[-2]>sig.iloc[-2] and macd.iloc[-1]<sig.iloc[-1]: return True, "15m MACD Dead Cross"
            return False, ""
        except Exception as e: self.logger.error(f"Err check indicator change: {e}", exc_info=False); return False, ""

    async def _check_market_volatility_spike(self, ohlcv_1h: list) -> Tuple[bool, str]:
        """检查 1h 价格大幅波动"""
        try:
            if len(ohlcv_1h) < 2: return False, ""
            o, c = ohlcv_1h[-2][1], ohlcv_1h[-2][4]
            if o > 0:
                chg = abs(c - o) / o
                if chg >= (settings.AI_VOLATILITY_TRIGGER_PERCENT / 100.0):
                    direction = 'up' if c > o else 'down'; return True, f"1h price spike {chg:.1%} ({direction})"
            return False, ""
        except Exception as e: self.logger.error(f"Err check market volatility: {e}", exc_info=False); return False, ""

    async def _check_and_execute_hard_stops(self):
        """[仅模拟盘] 检查并执行硬止损/止盈"""
        if self.is_live_trading: return False
        self.logger.info("Checking hard TP/SL (Paper)..."); to_close = []; tickers = {}
        try: tickers = await self.exchange.fetch_tickers(self.symbols)
        except Exception as e: self.logger.error(f"Hard stop failed: Fetch Tickers err: {e}"); return False
        for symbol, pos in list(self.portfolio.paper_positions.items()):
            if not pos or not isinstance(pos, dict) or pos.get('size', 0) <= 0: continue
            price = tickers.get(symbol, {}).get('last');
            if not price: self.logger.warning(f"Hard stop skipped: No price {symbol}."); continue
            side=pos.get('side'); sl=pos.get('stop_loss'); tp=pos.get('take_profit'); sl_valid = sl is not None and isinstance(sl, (int, float)) and sl > 0; tp_valid = tp is not None and isinstance(tp, (int, float)) and tp > 0
            if side=='long':
                if sl_valid and price <= sl: to_close.append((symbol, price, f"Hard SL @ {sl}"))
                elif tp_valid and price >= tp: to_close.append((symbol, price, f"Hard TP @ {tp}"))
            elif side=='short':
                if sl_valid and price >= sl: to_close.append((symbol, price, f"Hard SL @ {sl}"))
                elif tp_valid and price <= tp: to_close.append((symbol, price, f"Hard TP @ {tp}"))
        for symbol, price, reason in to_close:
            self.logger.warning(f"AUTO-CLOSING (Paper): {symbol} | Reason: {reason}"); await self.portfolio.paper_close(symbol, price, reason)
        return len(to_close) > 0

    # --- [V45.16 移除] 不再需要 _should_translate 和 _extract_keywords ---
    # def _extract_keywords(self, text: str) -> Set[str]:
    #     ...
    #
    # def _should_translate(self, new_summary: Optional[str], last_summary: Optional[str]) -> bool:
    #     ...
    # --- [移除结束] ---


    async def run_cycle(self):
        """[V45.16] AI 决策主循环。移除 1m, 最低保证金 6U, 原生中文摘要。"""
        self.logger.info("="*20 + " Starting AI Cycle " + "="*20)
        self.invocation_count += 1
        if not self.is_live_trading: await self._check_and_execute_hard_stops()

        # 1. 获取数据 & 构建 Prompt (已更新)
        market_data, tickers = await self._gather_all_market_data()
        portfolio_state = self.portfolio.get_state_for_prompt(tickers)
        user_prompt_string = self._build_prompt(market_data, portfolio_state, tickers)

        # 2. 格式化 System Prompt (已更新)
        try:
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                symbol_list=self.formatted_symbols,
                specific_rules_or_notes=""
            )
        except KeyError as e: self.logger.error(f"Format System Prompt failed: {e}"); return

        self.logger.info("Getting AI decision...")
        # 3. 获取 AI 决策
        ai_decision = await self._get_ai_decision(system_prompt, user_prompt_string)

        original_chain_of_thought = ai_decision.get("chain_of_thought", "AI No CoT.")
        orders = ai_decision.get("orders", [])
        self.logger.warning("--- AI CoT ---"); self.logger.warning(original_chain_of_thought)

        # --- [V45.16 核心修改] 简化的摘要提取 (无需翻译) ---
        summary_for_ui = "AI 未提供摘要。" # 默认值
        summary_keyword_pattern = re.compile(r"In summary,", re.IGNORECASE)
        parts = summary_keyword_pattern.split(original_chain_of_thought, maxsplit=1)

        if len(parts) > 1:
            extracted_summary = parts[1].strip().lstrip(' :').rstrip('`')
            if extracted_summary:
                summary_for_ui = extracted_summary # 提取到的摘要直接就是中文
                self.logger.info(f"Extracted Chinese summary: '{summary_for_ui[:50]}...'")
            else:
                summary_for_ui = "AI 摘要为空。"
                self.logger.warning("AI CoT 'In summary,' found but content is empty.")
        else:
            self.logger.warning("AI CoT 未找到 'In summary,' 关键字。")
        
        # 无论如何都更新
        self.last_strategy_summary = summary_for_ui
        # --- [修改结束] ---

        # 5. 执行决策 (包含 V45.14 最低保证金 6U 修正)
        if orders:
            self.logger.info(f"AI proposed {len(orders)} order(s), executing...")
            await self._execute_decisions(orders, market_data)
        else:
            self.logger.info("AI proposed no orders.")

        self.logger.info("="*20 + " AI Cycle Finished " + "="*20 + "\n")


    async def start(self):
        """启动 AlphaTrader 主循环"""
        self.logger.warning(f"🚀 AlphaTrader starting! Mode: {'LIVE' if self.is_live_trading else 'PAPER'}")
        if self.is_live_trading:
            self.logger.warning("!!! LIVE MODE !!! Syncing state on startup...")
            if not hasattr(self, 'client') and hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
            try: await self.portfolio.sync_state(); self.logger.warning("!!! LIVE State Sync Complete !!!")
            except Exception as e_sync: self.logger.critical(f"Initial LIVE state sync failed: {e_sync}", exc_info=True)
        while True:
            try:
                await self.portfolio.sync_state(); await self._log_portfolio_status()
                if self.is_live_trading and futures_settings.ENABLE_FORCED_TAKE_PROFIT:
                    try:
                        if not hasattr(self, 'client'): self.logger.error("Forced TP check failed: No client.")
                        else:
                            latest_tickers = await self.client.fetch_tickers(self.symbols); open_positions = self.portfolio.position_manager.get_all_open_positions(); positions_to_force_close = []
                            for symbol, state in open_positions.items():
                                price=latest_tickers.get(symbol,{}).get('last');
                                if not price or price<=0: continue
                                entry=state.get('avg_entry_price'); size=state.get('total_size'); side=state.get('side'); lev=state.get('leverage');
                                if not all([entry, size, side, lev]) or lev<=0: continue
                                upl = (price - entry) * size if side == 'long' else (entry - price) * size; margin = (entry * size) / lev if lev > 0 else 0.0
                                if margin > 0:
                                    rate = upl / margin; threshold = futures_settings.FORCED_TAKE_PROFIT_PERCENT / 100.0;
                                    self.logger.debug(f"Forced TP {symbol}: R={rate:.4f}, T={threshold:.4f}")
                                    if rate >= threshold: self.logger.warning(f"!!! Forced TP !!! {symbol} | R {rate:.2%} >= {threshold:.2%}"); positions_to_force_close.append(symbol)
                            if positions_to_force_close:
                                 tasks=[self.portfolio.live_close(s, f"Forced TP (R>={futures_settings.FORCED_TAKE_PROFIT_PERCENT}%)") for s in positions_to_force_close]; await asyncio.gather(*tasks);
                                 self.logger.info("Forced TP done, re-syncing..."); await self.portfolio.sync_state(); await self._log_portfolio_status()
                    except Exception as e_ftp: self.logger.error(f"Forced TP error: {e_ftp}", exc_info=True)
                trigger_ai, reason, now = False, "", time.time(); interval = settings.ALPHA_ANALYSIS_INTERVAL_SECONDS;
                if now - self.last_run_time >= interval: trigger_ai, reason = True, "Scheduled"
                if not trigger_ai:
                    sym=self.symbols[0]; ohlcv_15m, ohlcv_1h = [], []
                    try: 
                        # [V45.14 修改] 确保获取 15m (因为 1m 移除了)
                        ohlcv_15m, ohlcv_1h = await asyncio.gather(self.exchange.fetch_ohlcv(sym, '15m', limit=150), self.exchange.fetch_ohlcv(sym, '1h', limit=20))
                    except Exception as e_fetch: self.logger.error(f"Event check: Fetch OHLCV fail: {e_fetch}")
                    cooldown = settings.AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES * 60
                    if now - self.last_event_trigger_ai_time > cooldown:
                        event, ev_reason = await self._check_significant_indicator_change(ohlcv_15m) # 15m MACD 检查
                        if not event: event, ev_reason = await self._check_market_volatility_spike(ohlcv_1h) # 1h 波动检查
                        if event: trigger_ai, reason = True, ev_reason
                if trigger_ai:
                    self.logger.warning(f"🔥 AI triggered! Reason: {reason}")
                    if reason != "Scheduled": self.last_event_trigger_ai_time = now
                    await self.run_cycle(); self.last_run_time = now
                await asyncio.sleep(10)
            except asyncio.CancelledError: self.logger.warning("Task cancelled, shutting down..."); break
            except Exception as e: self.logger.critical(f"Main loop fatal error: {e}", exc_info=True); await asyncio.sleep(60)
