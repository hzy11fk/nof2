# 文件: alpha_trader.py (完整优化版 - 混合模型 + 完整通知)
# 1. 策略:
#    - [AI - Rule 6] AI 负责低频、高上下文的限价单 (LIMIT) 策略。
#    - [Python - Rule 8] Python 负责高频、低延迟的市价单 (MARKET) 突破策略。
# 2. 通知:
#    - 所有通知 (开仓/挂单/平仓/成交) 均由 alpha_portfolio.py (Source 2) 正确处理。
# 3. Bug 修复:
#    - [修复] 修复了 start() 循环中 'ohlcv_1sSsm' 的 NameError。
#    - [修复] 修复了 'futures_settings' 的 AttributeError。
#    - [修复] 修复了 'NoneType' is not iterable 的 TypeError。
#    - [修复] [V45.38] start() 循环现在调用 portfolio.remove_pending_limit_order() 以确保持久化。

import logging
import asyncio
import time
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
import re
import httpx # 用于 F&G Index
from collections import deque
# [修复] 导入 settings 和 futures_settings (适配您原始的 config.py)
from config import settings, futures_settings
from alpha_ai_analyzer import AlphaAIAnalyzer
from alpha_portfolio import AlphaPortfolio # 假设 V23.4 或更高
from datetime import datetime
from typing import Tuple, Dict, Any, Set, Optional
# [修复] 移除冗余的 bark_notifier 导入， portfolio 会处理
# from bark_notifier import send_bark_notification 

try:
    import pandas as pd
except ImportError:
    pd = None

class AlphaTrader:
    
    # --- [AI (Rule 6) 专用 PROMPT] ---
    # (此部分无变化)
    SYSTEM_PROMPT_TEMPLATE = """
    You are a **profit-driven, analytical, and disciplined** quantitative trading AI. Your primary goal is to **generate and secure realized profit**. You are not a gambler; you are a calculating strategist.

    **You ONLY execute a trade if you have a high-confidence assessment that the action will lead to profit.** A medium or low-confidence signal means you WAIT.

    Your discipline is demonstrated by strict adherence to the risk management rules below, which are your foundation for sustained profitability.

    **Core Mandates & Rules:**
    1.  **Strategy: Limit Orders Only (CRITICAL):**
        -   To prevent "chasing price," your **only** strategy is to be patient and trade pullbacks (Rule 6.1), mean-reversion (Rule 6.2), or chop-zones (Rule 6.3).
        -   You MUST and ONLY use `LIMIT_BUY` or `LIMIT_SELL`.
        -   All fast market-order trades (Rule 8) are handled by a separate, high-frequency Python algorithm.

    2.  **Rule-Based Position Management:**
        For every open position, you MUST check its `InvalidATION_CONDITION`. If this condition is met, you MUST issue a `CLOSE` order. This is your top priority for existing positions.

    3.  **Active SL/TP Management (CRITICAL):**
        Your main task for existing positions is to actively manage risk.
        -   For *every* open position on *every* analysis cycle, you MUST assess if the existing `ai_suggested_stop_loss` or `ai_suggested_take_profit` targets are still optimal.
        -   The system executes these targets automatically, but your job is to UPDATE them.
        -   If the market structure changes (e.g., a new S/R level appears, volatility drops), you MUST issue `UPDATE_STOPLOSS` or `UPDATE_TAKEPROFIT` orders with new, improved targets and your reasoning.

    4.  **Risk Management Foundation (CRITICAL):**
        Profit is the goal, but capital preservation is the foundation.
        -   **AI's Task (Strategy):** Your job is to select the *risk parameters* based on your confidence, not to perform the final math. The Python system will perform all final calculations and safety checks.
        -   **AI Must Provide (for LIMIT_BUY, LIMIT_SELL):**
            1.  `"leverage": [e.g., 8]` (Choose appropriate leverage based on asset volatility, e.g., 5x-8x for SOL/DOGE, 10x-15x for BTC/ETH).
            2.  `"risk_percent": [e.g., 0.025]` (Dynamically choose a risk percentage of Total Equity. Example: Medium Confidence = 0.025 (2.5% Equity), High Confidence = 0.05 (5.0% Equity)).
        -   **System's Task (Calculation):** The Python system will automatically use your `risk_percent` and `Total Equity` to calculate the `final_desired_margin`, check it against `Available Cash`, and perform all hard checks (Min Margin 6 USDT, Min BTC Size 0.001). The order will be aborted by the system if any check fails.
        -   **Total Exposure:** The sum of all margins for all open positions should generally not exceed 50-60% of your total equity.
        -   **Correlation Control (Hard Cap):** You MUST limit total risk exposure to highly correlated assets.
            -   Define 'Core Crypto Group' as [BTC, ETH]. Total margin for this group MUST NOT exceed 30% of Total Equity.
            -   Define 'Altcoin Group' as [SOL, BNB, DOGE, XRP]. Total margin for this group MUST NOT exceed 40% of Total Equity.
            -   If opening a new position (e.g., SOL) would breach its group cap, you MUST ABORT the trade.

    5.  **Complete Trade Plans (Open/Add):**
        Every new order (LIMIT) is a complete plan. You MUST provide: `take_profit`, `stop_loss`, `invalidation_condition`.
        -   **Smarter Invalidation:** Your `invalidation_condition` MUST be based on a clear technical breakdown of the *original trade thesis*.
            -   *Trend Trade Example:* `Invalidation='1h Close below the EMA 50'`
            -   *Ranging Trade Example:* `InvalidATION='15m RSI breaks above 60'`
        -   **Profit-Taking Strategy:** You SHOULD consider using multiple take-profit levels (by using `PARTIAL_CLOSE` later) rather than a single `take_profit`.

    6.  **Market State Recognition (Default Strategy):**
        You MUST continuously assess the market regime using the **1hour** and **4hour** timeframes. This is your **Default Strategy**.
        -   **1. Strong Trend (Trending Bullish/Bearish):**
            -   **Condition:** 1h or 4h **ADX_14 > 25**.
            -   **Strategy (LIMIT ONLY):** Your **only** strategy is to trade **pullbacks**. Identify key S/R levels (e.g., 1h EMA 20, 4h BB_Mid). Your job is to place a **`LIMIT_BUY` (in uptrend) or `LIMIT_SELL` (in downtrend)** at that calculated level.
        -   **2. Ranging (No Trend):**
            -   **Condition:** 1h and 4h **ADX_14 < 20**.
            -   **Strategy (LIMIT ONLY):** In this regime, your **only** strategy is **mean-reversion**. Your task is to identify the `BB_Upper` and `BB_Lower` levels. You MUST issue **`LIMIT_SELL` at (or near) the upper band** or **`LIMIT_BUY` at (or near) the lower band**.
        -   **3. Chop (Short-Term Ranging):**
            -   **Condition:** 1h or 4h **ADX_14 is between 20 and 25**.
            -   **Strategy (LIMIT ONLY):** This is a low-conviction market. Shift focus to the **15min timeframe**.
            -   Identify the `15min_bb_upper` and `15min_bb_lower` levels.
            -   You MAY issue **`LIMIT_SELL` at the 15m upper band** or **`LIMIT_BUY` at the 15m lower band**.
            -   **Risk:** This is a lower-confidence trade. You MUST use a reduced `risk_percent` (e.g., 0.01 or 0.015) and tighter stops.

    7.  **Market Sentiment Filter (Fear & Greed Index):**
        (This rule remains unchanged)
        You MUST use the provided `Fear & Greed Index` (from the User Prompt) as a macro filter for your decisions.
        -   **Extreme Fear (Index < 25):** ...
        -   **Fear (Index 25-45):** ...
        -   **Neutral (Index 45-55):** ...
        -   **Greed (Index 55-75):** ...
        -   **Extreme Greed (Index > 75):** ...

    (Rule 8 Removed - Handled by Python)

    **Multi-Timeframe Confirmation Requirement (CRITICAL):**
    - You MUST analyze and confirm signals across available timeframes: **5min, 15min, 1hour, and 4hour**.
    - **High-Confidence Signal Definition:** A signal is only high-confidence when it aligns with the **Market State** (see Rule 6) and shows alignment across **at least 3** timeframes.
    - **Timeframe Hierarchy:** Use longer timeframes (**4h, 1h**) to determine the **Market State** and **Overall Trend**. Use shorter timeframes (**15min, 5min**) for precise entry timing.
    -   **Signal Veto Rule (CRITICAL):**
        -   Even if 4h/1h trend signals (e.g., ADX > 25) are strong, if the 15min timeframe shows a **strong opposing signal** (e.g., a bearish RSI divergence, a 15m MACD Dead Cross, or 15m EMA 20 has crossed below 50), you **MUST ABORT** the trade (this applies to Rule 6).
        -   **Never** trade against 15m momentum when trying to enter on a pullback.

    **Psychological Safeguards:**
    - Confirmation Bias Protection: Seek counter-evidence.
    - Overtrading Protection: No high-confidence signal = WAIT.
    - Loss Aversion Protection: Stick to stop losses.

    **MANDATORY OUTPUT FORMAT:**
    Your entire response must be a single JSON object with two keys: "chain_of_thought" and "orders".

    1.  `"chain_of_thought"` (string): A multi-line string containing your detailed analysis in English. It MUST follow this template precisely:
        ```
        My Current Assessment & Actions (Rule 6 - Limit Orders)

        Market State Analysis:
        - 1h ADX: [Value] | 4h ADX: [Value]
        - Regime: [Applying Rule 6: Trending Pullback (ADX>25) / Applying Rule 6: Ranging (ADX<20) / Applying Rule 6: Chop Zone (ADX 20-25)]
        - Key Support/Resistance Levels: [Identify major S/R levels, including BB_Upper/Lower and recent_high/low for relevant symbols]
        - Market Sentiment: [MUST state the F&G Index value and its implication, e.g., "Extreme Greed (80)"]

        Portfolio Overview:
        Total Equity: $X, Available Cash: $Y, Current Margin Usage: Z%
        Current Market Correlation Assessment: [Assess if positions are overly correlated based on Correlation Control (Rule 4) hard caps]

        Let's break down each position:
        1. [SYMBOL] ([SIDE]):
           UPL: [Current Unrealized PNL and Percent (e.g., +$50.00 (+5.5%))]
           Multi-Timeframe Analysis: [Brief assessment across 5m, 15m, 1h, 4h, mentioning ADX/BBands]
           
           Invalidation Check: [Check condition vs current data. If met, MUST issue CLOSE.]
           (Note: Python-based Rule 8 trades use their own trailing stop logic and are not managed by you)
           
           Reversal & Profit Save Check:
           - [Is this profitable position (UPL > +1.0%) showing strong signs of reversal (e.g., 15m bearish divergence, 1h ADX weakening, F&G Extreme Greed)?]
           - [IF YES: The risk of giving back profits is high. Prioritize capital preservation. Decision: MUST issue a CLOSE order to secure profits.]

           Pyramiding Check (Adding to a Winner):
           - [Is UPL Percent > +2.5% AND is the original trend (ADX > 25) still strong?]
           - [AND has price pulled back to a key support (for Long) / resistance (for Short) (e.g., 1h EMA 20)?]
           - [IF YES: Consider a `LIMIT_BUY`/`LIMIT_SELL` (Limit) order. This new entry is treated as a separate trade and MUST follow the full Rule 4 (Sizing) / Rule 5 (SL/TP) logic.]
           - [CRITICAL: You MUST NEVER add to a losing position (UPL < 0). Averaging down is forbidden.]
           
           SL/TP Target Update Check:
           - [Are the current ai_suggested_stop_loss/take_profit targets still optimal based on new data (e.g., move SL up to new 15m BB_Mid, lower TP from 4h BB_Upper)?]
           - [IF NOT OPTIMAL: Issue UPDATE_STOPLOSS / UPDATE_TAKEPROFIT with new targets and reasoning.]
           
           Decision: [Hold/Close/Partial Close/Add/Update StopLoss/Update TakeProfit + Reason. NOTE: Invalidation and Reversal checks override all "Hold" decisions.]

        ... [Repeat for each open position] ...

        New Trade Opportunities Analysis (Rule 6 - Limit Orders Only):
        Available Margin for New Trades: [Calculate based on Total Equity and risk rules]
        Correlation Check: [Ensure new trades don't breach Rule 4 Correlation Control hard caps]

        [Analyze opportunities based ONLY on Rule 6]
        
        [EXAMPLE - RULE 6 (TRENDING PULLBACK):]
        BTC Multi-Timeframe Assessment (Market State: Trending Bullish, 4h ADX=28):
        - 4h Trend: Bullish (EMA 20 > 50) | 1h Momentum: Strong (MACD > 0) | 15min Setup: Price is *approaching* pullback support (1h EMA 20 @ 65000.0).
        - Signal Confluence Score: 4/4 | Final Confidence: High - **PREPARE LIMIT_BUY at 65000.0 (Rule 6)**

        [EXAMPLE - RULE 6 (RANGING):]
        ETH Multi-Timeframe Assessment (Market State: Ranging, 1h ADX=18):
        - 4h Trend: N/A (ADX < 20) | 1h Setup: Price is *approaching* 1h BB_Upper (@ 3900.0) | 15min RSI: 68.5 (Approaching Overbought)
        - Signal Confluence Score: 3/4 (RSI/BBands align) | Final Confidence: High (for Ranging) - **PREPARE LIMIT_SELL at 3900.0 (Rule 6)**
        
        [EXAMPLE - RULE 6.3 (CHOP ZONE):]
        BNB Multi-Timeframe Assessment (Market State: Chop, 1h ADX=23):
        - 4h/1h Trend: N/A (Chop). Shifting to 15m timeframe.
        - 15m Setup: Price is *approaching* 15m BB_Upper (@ 305.0). 15m RSI: 65.
        - Signal Confluence Score: 3/4 (15m RSI/BBands align) | Final Confidence: Low/Medium (Reduced Risk 1.5%) - **PREPARE LIMIT_SELL at 305.0 (Rule 6.3 Chop)**

        In summary, [**Key Instruction: Please provide your final concise decision overview directly here, in Chinese.**Final concise decision overview.]
        ```

    2.  `"orders"` (list): A list of JSON objects for trades. Empty list `[]` if holding all.

    **Order Object Rules:**
    (Market Order Templates Removed)
    -   **To Open Limit (LONG - Rule 6):** `{{"action": "LIMIT_BUY", "symbol": "...", "leverage": [CHOSEN_LEVERAGE], "risk_percent": [CHOSEN_RISK_PERCENT], "limit_price": [CALCULATED_PRICE], "take_profit": ..., "stop_loss": ..., "invalidation_condition": "...", "reasoning": "Limit Order (Rule 6). Leverage: [...]. Risk: [...]. Market State: [Trending Pullback or Ranging Support]"}}`
    -   **To Open Limit (SHORT - Rule 6):** `{{"action": "LIMIT_SELL", "symbol": "...", "leverage": [CHOSEN_LEVERAGE], "risk_percent": [CHOSEN_RISK_PERCENT], "limit_price": [CALCULATED_PRICE], "take_profit": ..., "stop_loss": ..., "invalidation_condition": "...", "reasoning": "Limit Order (Rule 6). Leverage: [...]. Risk: [...]. Market State: [Trending Pullback or Ranging Resistance]"}}`
    -   **To Close Fully:** `{{"action": "CLOSE", "symbol": "...", "reasoning": "Invalidation met / SL hit / TP hit / Max Loss Cutoff / Manual decision..."}}`
    -   **To Close Partially (Take Profit):** `{{"action": "PARTIAL_CLOSE", "symbol": "...", "size_percent": 0.5, "reasoning": "Taking 50% profit near resistance per Rule 5..."}}` (or `size_absolute`)
    -   **To Update Stop Loss:** `{{"action": "UPDATE_STOPLOSS", "symbol": "...", "new_stop_loss": ..., "reasoning": "Actively moving SL to new 15m support level..."}}`
    -   **To Update Take Profit:** `{{"action": "UPDATE_TAKEPROFIT", "symbol": "...", "new_take_profit": ..., "reasoning": "Actively moving TP to new 4h resistance level..."}}`
    -   **To Hold:** Do NOT include in `orders`.
    -   **Symbol Validity:** `symbol` MUST be one of {symbol_list}.

    **Remember:** Quality over quantity.
    """

    def __init__(self, exchange):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.exchange = exchange
        # [修复] 从 settings 中读取 symbols
        self.symbols = settings.FUTURES_SYMBOLS_LIST
        self.portfolio = AlphaPortfolio(exchange, self.symbols) # 假设 V23.4+
        self.ai_analyzer = AlphaAIAnalyzer(exchange, "ALPHA_TRADER")
        self.is_live_trading = settings.ALPHA_LIVE_TRADING
        self.start_time = time.time(); self.invocation_count = 0; self.last_run_time = 0; self.last_event_trigger_ai_time = 0
        self.log_deque = deque(maxlen=50); self._setup_log_handler()
        self.last_strategy_summary = "Initializing..."
        self.initial_capital = settings.ALPHA_LIVE_INITIAL_CAPITAL if self.is_live_trading else settings.ALPHA_PAPER_CAPITAL
        self.logger.info(f"Initialized with Initial Capital: {self.initial_capital:.2f} USDT")
        self.formatted_symbols = ", ".join(f'"{s}"' for s in self.symbols)
        
        if hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
        else:
            if hasattr(self.portfolio, 'exchange') and isinstance(self.portfolio.exchange, object): self.client = self.portfolio.exchange; self.logger.warning("Portfolio missing 'client', falling back.")
            else: self.client = self.exchange; self.logger.warning("Portfolio missing 'client', using exchange directly.")
        
        self.fng_data: Dict[str, Any] = {"value": 50, "value_classification": "Neutral"}
        self.last_fng_fetch_time: float = 0.0
        self.FNG_CACHE_DURATION_SECONDS = 3600 # 1小时缓存 (3600秒)
        
        # 止盈计数器，用于多阶段止盈。
        self.tp_counters: Dict[str, Dict[str, int]] = {}


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
        """
        引入 asyncio.Semaphore 来限制并发请求，防止因 API 频率限制导致的数据缺失。
        确保 '1h' 变为 '1hour'，'5m' 变为 '5min'，与 _build_prompt 严格一致。
        [优化] 新增 ATR 计算，供 Rule 8 止损使用。
        """
        # [修复] 日志级别降低
        self.logger.debug("Gathering multi-TF market data (5m, 15m, 1h, 4h) + Indicators (Manual Calc)...")
        market_indicators_data: Dict[str, Dict[str, Any]] = {}
        fetched_tickers: Dict[str, Any] = {}
        
        # 限制并发请求为 10 个，防止触发交易所的 DDos/Rate Limit
        CONCURRENT_REQUEST_LIMIT = 10
        semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

        async def _safe_fetch_ohlcv(symbol, timeframe, limit):
            async with semaphore:
                try:
                    return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                except Exception as e:
                    self.logger.error(f"Safe Fetch OHLCV Error ({symbol} {timeframe}): {e}", exc_info=False)
                    return e # 返回异常而不是抛出

        async def _safe_fetch_ticker(symbol):
            async with semaphore:
                try:
                    return await self.exchange.fetch_ticker(symbol)
                except Exception as e:
                    self.logger.error(f"Safe Fetch Ticker Error ({symbol}): {e}", exc_info=False)
                    return e # 返回异常而不是抛出
        
        try:
            # 内部使用 '1h', '4h'
            timeframes = ['5m', '15m', '1h', '4h']
            tasks = []
            for symbol in self.symbols:
                for timeframe in timeframes: 
                    # 使用安全包装器
                    tasks.append(_safe_fetch_ohlcv(symbol, timeframe, limit=100))
                # 使用安全包装器
                tasks.append(_safe_fetch_ticker(symbol))
                
            results = await asyncio.gather(*tasks) # 不再需要 return_exceptions=True
            
            total_timeframes = len(timeframes); tasks_per_symbol = total_timeframes + 1
            
            for i, symbol in enumerate(self.symbols):
                start_index = i * tasks_per_symbol; symbol_ohlcv_results = results[start_index:start_index + total_timeframes]
                ticker_result = results[start_index + total_timeframes]
                
                # 检查返回的是否是异常
                if not isinstance(ticker_result, Exception) and ticker_result and ticker_result.get('last') is not None:
                    fetched_tickers[symbol] = ticker_result; market_indicators_data[symbol] = {'current_price': ticker_result.get('last')}
                else: 
                    market_indicators_data[symbol] = {'current_price': None}
                    self.logger.warning(f"Failed fetch ticker/price for {symbol} (Result: {ticker_result})")
                
                for j, timeframe in enumerate(timeframes):
                    ohlcv_data = symbol_ohlcv_results[j]
                    
                    # 检查返回的是否是异常
                    if isinstance(ohlcv_data, Exception) or not ohlcv_data: 
                        self.logger.warning(f"Failed fetch {timeframe} for {symbol} (Result: {ohlcv_data})")
                        continue
                    
                    try:
                        df = pd.DataFrame(ohlcv_data, columns=['ts', 'o', 'h', 'l', 'c', 'v']); df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore'); 
                        
                        cols_to_numeric = ['o', 'h', 'l', 'c', 'v']
                        for col in cols_to_numeric:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df.ffill(inplace=True)
                        df.dropna(inplace=True)
                        
                        df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
                        
                        if df.empty: 
                            self.logger.warning(f"DataFrame empty for {symbol} {timeframe} after cleaning NaNs.")
                            continue
                            
                        # 关键修复：确保 '1h' -> '1hour', '5m' -> '5min'
                        prefix = f"{timeframe.replace('m', 'min').replace('h', 'hour')}_"

                        # 1. 手动计算 ADX
                        if len(df) >= 28: # (14*2)
                            try:
                                high = df['h']; low = df['l']; close = df['c']; period = 14
                                move_up = high.diff(); move_down = low.diff().mul(-1)
                                plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0), index=df.index)
                                minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0), index=df.index)
                                tr1 = pd.DataFrame(high - low)
                                tr2 = pd.DataFrame(abs(high - close.shift(1)))
                                tr3 = pd.DataFrame(abs(low - close.shift(1)))
                                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                                atr = tr.ewm(alpha=1/period, adjust=False).mean()
                                
                                # [优化] 存储 ATR
                                if not atr.empty:
                                    market_indicators_data[symbol][f'{prefix}atr_14'] = atr.iloc[-1]
                                
                                plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9))
                                minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9))
                                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
                                adx = dx.ewm(alpha=1/period, adjust=False).mean()
                                if not adx.empty:
                                    market_indicators_data[symbol][f'{prefix}adx_14'] = adx.iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"Manual ADX/ATR calc failed for {symbol} {timeframe}: {e}", exc_info=False)

                        # 2. 手动计算 BBands
                        if len(df) >= 20:
                            try:
                                period = 20; std_dev = 2.0; closes = df['c']
                                middle_band = closes.rolling(window=period).mean()
                                rolling_std = closes.rolling(window=period).std()
                                upper_band = middle_band + (rolling_std * std_dev)
                                lower_band = middle_band - (rolling_std * std_dev)
                                if not upper_band.empty:
                                    market_indicators_data[symbol][f'{prefix}bb_upper'] = upper_band.iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}bb_middle'] = middle_band.iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}bb_lower'] = lower_band.iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"Manual BBands calc failed for {symbol} {timeframe}: {e}", exc_info=False)

                        # 3. 手动计算 RSI
                        if len(df) >= 15: # (14+1)
                            try:
                                period = 14; delta = df['c'].diff()
                                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
                                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
                                rs = gain / loss.replace(0, 1e-9)
                                rsi = 100 - (100 / (1 + rs))
                                if not rsi.empty:
                                    market_indicators_data[symbol][f'{prefix}rsi_14'] = rsi.iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"Manual RSI calc failed for {symbol} {timeframe}: {e}", exc_info=False)

                        # 4. ta.macd
                        if len(df) >= 26:
                            try:
                                macd = ta.macd(df['c'], 12, 26, 9)
                                if macd is not None and not macd.empty and 'MACD_12_26_9' in macd.columns:
                                    market_indicators_data[symbol][f'{prefix}macd'] = macd['MACD_12_26_9'].iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}macd_hist'] = macd['MACDh_12_26_9'].iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"ta.macd calc failed for {symbol} {timeframe}: {e}", exc_info=False)
                                
                        # 5. ta.ema
                        if len(df) >= 50:
                            try:
                                ema20 = ta.ema(df['c'], 20)
                                if ema20 is not None and not ema20.empty:
                                    market_indicators_data[symbol][f'{prefix}ema_20'] = ema20.iloc[-1]
                                ema50 = ta.ema(df['c'], 50)
                                if ema50 is not None and not ema50.empty:
                                    market_indicators_data[symbol][f'{prefix}ema_50'] = ema50.iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"ta.ema calc failed for {symbol} {timeframe}: {e}", exc_info=False)

                        # 6. 其他指标
                        if len(df)>=2: 
                            cur_v=df['v'].iloc[-1]; avg_v=df['v'].tail(20).mean() if len(df)>=20 else df['v'].mean();
                            if avg_v>0: market_indicators_data[symbol][f'{prefix}volume_ratio']=cur_v/avg_v
                        
                        if len(df) >= 2:
                            prev_close = df['c'].iloc[-2];
                            if prev_close > 0: 
                                chg=(df['c'].iloc[-1]-prev_close)/prev_close*100; market_indicators_data[symbol][f'{prefix}price_change_pct']=chg
                            else: 
                                market_indicators_data[symbol][f'{prefix}price_change_pct'] = 0.0
                        
                        if len(df)>=20: 
                            market_indicators_data[symbol][f'{prefix}recent_high']=df['h'].tail(20).max(); 
                            market_indicators_data[symbol][f'{prefix}recent_low']=df['l'].tail(20).min()
                            
                    except Exception as e: self.logger.error(f"Error during indicator calc loop for {symbol} {timeframe}: {e}", exc_info=False)
        except Exception as e: self.logger.error(f"Error gathering market data: {e}", exc_info=True)
        return market_indicators_data, fetched_tickers


    def _build_prompt(self, market_data: Dict[str, Dict[str, Any]], portfolio_state: Dict, tickers: Dict) -> str:
        """构建 User Prompt"""
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
            
            # [修复] 确保列表与 '...min' 和 '...hour' 键名匹配
            timeframes = ['5min', '15min', '1hour', '4hour']
            
            for tf in timeframes:
                prompt += f"\n[{tf.upper()}]\n"
                prompt += f" RSI:{safe_format(d.get(f'{tf}_rsi_14'), 0, is_rsi=True)}|"
                prompt += f" ADX:{safe_format(d.get(f'{tf}_adx_14'), 0, is_rsi=True)}|" 
                prompt += f" ATR:{safe_format(d.get(f'{tf}_atr_14'), 4)}|"
                prompt += f"MACD:{safe_format(d.get(f'{tf}_macd'), 4)}|"
                prompt += f"Sig:{safe_format(d.get(f'{tf}_macd_signal'), 4)}|"
                prompt += f"Hist:{safe_format(d.get(f'{tf}_macd_hist'), 4)}\n"
                prompt += f" EMA20:{safe_format(d.get(f'{tf}_ema_20'), 3)}|"
                prompt += f"EMA50:{safe_format(d.get(f'{tf}_ema_50'), 3)}|"
                prompt += f"VolR:{safe_format(d.get(f'{tf}_volume_ratio'), 1)}x\n"
                prompt += f" BB_Up:{safe_format(d.get(f'{tf}_bb_upper'), 3)}|"   
                prompt += f"BB_Mid:{safe_format(d.get(f'{tf}_bb_middle'), 3)}|" 
                prompt += f"BB_Low:{safe_format(d.get(f'{tf}_bb_lower'), 3)}\n" 
                prompt += f" Hi:{safe_format(d.get(f'{tf}_recent_high'), 2)}|"
                prompt += f"Lo:{safe_format(d.get(f'{tf}_recent_low'), 2)}|"
                prompt += f"Chg:{safe_format(d.get(f'{tf}_price_change_pct'), 1)}%\n"
            prompt += "-----\n"
        
        prompt += "\n--- Market Context ---\n"
        fng_val = self.fng_data.get('value', 50)
        fng_class = self.fng_data.get('value_classification', 'Neutral').title() # e.g. "Extreme Fear"
        prompt += f"Fear & Greed Index: {fng_val} ({fng_class})\n"
        
        prompt += "\n--- Account Info ---\n"
        prompt += f"Return%: {portfolio_state.get('performance_percent', 'N/A')}\n"
        prompt += f"Total Equity: {portfolio_state.get('account_value_usd', 'N/A')}\n" 
        prompt += f"Available Cash: {portfolio_state.get('cash_usd', 'N/A')}\n" 
        prompt += "Positions:\n"
        prompt += portfolio_state.get('open_positions', "No open positions.")
        return prompt

    async def _get_ai_decision(self, system_prompt: str, user_prompt: str) -> dict:
        """调用 AI 分析器"""
        if not self.ai_analyzer: return {}
        return await self.ai_analyzer.get_ai_response(system_prompt, user_prompt)

    async def _execute_decisions(self, decisions: list, market_data: Dict[str, Dict[str, Any]]):
        """
        混合策略执行:
        - "BUY"/"SELL" (市价开仓) 由 Python Rule 8 调用。
        - "LIMIT_BUY"/"LIMIT_SELL" 由 AI Rule 6 调用。
        - 将 Python 端的安全计算 (risk_percent -> size) 应用于所有开仓类型。
        - [修复] 移除所有通知调用， portfolio.py (Source 2) 会处理。
        """
        
        # [修复] 从 futures_settings (Source 1) 中读取
        MIN_MARGIN_USDT = futures_settings.MIN_NOMINAL_VALUE_USDT
        MIN_SIZE_BTC = 0.001 # (这个目前是硬编码，可以移到 config)

        for order in decisions:
            try:
                action = order.get('action'); symbol = order.get('symbol')
                if not action or not symbol or symbol not in self.symbols: 
                    self.logger.warning(f"跳过无效指令: {order}"); continue
                
                reason = order.get('reasoning', 'N/A')
                
                current_price = market_data.get(symbol, {}).get('current_price')
                limit_price_from_ai = order.get('limit_price') # 仅用于限价单
                
                # --- 平仓 (CLOSE) ---
                if action == "CLOSE":
                    if (not current_price or current_price <= 0) and not self.is_live_trading:
                        self.logger.error(f"模拟盘平仓失败: 无当前价格 {symbol}"); continue
                    # [通知] portfolio.live_close / paper_close 会处理通知
                    if self.is_live_trading: await self.portfolio.live_close(symbol, reason=reason)
                    else: await self.portfolio.paper_close(symbol, current_price, reason=reason)
                
                # --- 市价开仓 (BUY / SELL) - 仅由 Python Rule 8 使用 ---
                elif action in ["BUY", "SELL"]:
                    if not self.is_live_trading:
                        self.logger.warning(f"模拟盘：跳过 {action} 市价单 (模拟盘仅支持限价单转换)。"); continue
                    
                    if (not current_price or current_price <= 0):
                        self.logger.error(f"无当前价格 {symbol}，跳过 Market Action: {order}"); continue
                        
                    side = 'long' if action == 'BUY' else 'short'; final_size = 0.0
                    price_to_calc = current_price # 市价单使用当前价格计算

                    try:
                        # 1. 从 AI/Python 获取参数
                        leverage = int(order.get('leverage'))
                        risk_percent = float(order.get('risk_percent'))
                        stop_loss = float(order.get('stop_loss'))
                        take_profit = order.get('take_profit') # 可以为 None (用于追踪止损)
                        
                        if risk_percent <= 0 or risk_percent > 0.5: raise ValueError(f"无效的 risk_percent: {risk_percent}")
                        if leverage <= 0 or leverage > 100: raise ValueError(f"无效的 leverage: {leverage}")
                        
                        # 2. 从 Portfolio 获取账户状态
                        total_equity = float(self.portfolio.equity)
                        available_cash = float(self.portfolio.cash)
                        if total_equity <= 0: raise ValueError(f"无效账户状态 (Equity <= 0)")
                        
                        # 3. 执行 Prompt Rule 4 的计算
                        calculated_desired_margin = total_equity * risk_percent
                        
                        # [修复] 检查最大保证金占用率
                        max_margin_cap = total_equity * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
                        if calculated_desired_margin > max_margin_cap:
                            self.logger.warning(f"!!! {action} Margin Capped !!! AI 期望保证金 {calculated_desired_margin:.2f} > 最大 {max_margin_cap:.2f} ({futures_settings.MAX_MARGIN_PER_TRADE_RATIO*100}%)")
                            calculated_desired_margin = max_margin_cap
                        
                        if calculated_desired_margin > available_cash:
                            self.logger.error(f"!!! {action} Aborted (Cash Insufficient) !!! AI 期望保证金 {calculated_desired_margin:.2f} > 可用 {available_cash:.2f}")
                            continue
                        
                        if calculated_desired_margin < MIN_MARGIN_USDT:
                            self.logger.warning(f"!!! {action} Margin Adjusted !!! AI 期望保证金 {calculated_desired_margin:.2f} < 最小 {MIN_MARGIN_USDT} USDT. 正在上调。")
                            final_desired_margin = MIN_MARGIN_USDT
                        else:
                            final_desired_margin = calculated_desired_margin
                        
                        final_size = (final_desired_margin * leverage) / price_to_calc

                        if symbol == "BTC/USDT:USDT":
                            if final_size < MIN_SIZE_BTC:
                                self.logger.warning(f"!!! {action} BTC Size Adjusted !!! 计算后 size {final_size} < 最小 {MIN_SIZE_BTC}. 正在上调。")
                                final_size = MIN_SIZE_BTC
                                recalculated_margin = (final_size * price_to_calc) / leverage
                                if recalculated_margin > available_cash:
                                    self.logger.error(f"!!! {action} Aborted (Cash Insufficient for Min BTC Size) !!! 最小 BTC size 需要 {recalculated_margin:.2f} 保证金 > 可用 {available_cash:.2f}")
                                    continue
                        
                        if final_size <= 0: raise ValueError("最终计算 size 为 0")

                        # 执行实盘市价单
                        invalidation_condition = order.get('invalidation_condition')
                        # [通知] portfolio.live_open 会自动发送通知
                        await self.portfolio.live_open(
                            symbol, 
                            side, 
                            final_size, # <-- 经 Python 计算后的
                            leverage, 
                            reason=reason, 
                            stop_loss=stop_loss, 
                            take_profit=take_profit, # 传入 None 或具体数值
                            invalidation_condition=invalidation_condition
                        )

                        # [修复] 移除了此处的冗余通知调用

                    except (ValueError, TypeError, KeyError) as e: 
                        self.logger.error(f"跳过 {action} (Python 计算/参数错误): {order}. Err: {e}"); continue

                # --- 限价开仓 (LIMIT_BUY / LIMIT_SELL) - 仅由 AI Rule 6 使用 ---
                elif action in ["LIMIT_BUY", "LIMIT_SELL"]:
                    
                    if not limit_price_from_ai or float(limit_price_from_ai) <= 0:
                        self.logger.error(f"无效限价 {limit_price_from_ai} {symbol}，跳过 Limit Action: {order}"); continue
                    
                    if not self.is_live_trading:
                        # 模拟盘不支持限价单，转为市价单 (如果价格有利)
                        self.logger.warning(f"模拟盘不支持限价单，正在检查是否转为市价单...")
                        try:
                            price_to_calc = float(limit_price_from_ai)
                            if (action == 'LIMIT_BUY' and current_price <= price_to_calc) or (action == 'LIMIT_SELL' and current_price >= price_to_calc):
                                self.logger.warning(f"模拟盘：价格 {current_price} 有利，转为市价单。")
                                leverage = int(order.get('leverage'))
                                risk_percent = float(order.get('risk_percent'))
                                stop_loss = float(order.get('stop_loss'))
                                take_profit = float(order.get('take_profit'))
                                invalidation_condition = order.get('invalidation_condition')
                                
                                # 模拟盘的计算 (使用当前价格)
                                total_equity = float(self.portfolio.equity)
                                calculated_desired_margin = total_equity * risk_percent
                                final_margin = max(calculated_desired_margin, MIN_MARGIN_USDT)
                                final_size = (final_margin * leverage) / current_price
                                if symbol == "BTC/USDT:USDT" and final_size < MIN_SIZE_BTC:
                                    final_size = MIN_SIZE_BTC

                                # [通知] portfolio.paper_open 会自动发送通知
                                await self.portfolio.paper_open(symbol, 'long' if action == 'LIMIT_BUY' else 'short', final_size, price=current_price, leverage=leverage, reason=reason, stop_loss=stop_loss, take_profit=take_profit, invalidation_condition=invalidation_condition)
                            else:
                                self.logger.info(f"模拟盘：价格 {current_price} 不利 (vs {price_to_calc})，跳过限价单。")
                        except Exception as e_paper:
                             self.logger.error(f"模拟盘限价单转换失败: {e_paper}, Order: {order}"); continue
                        continue # 模拟盘逻辑结束

                    # --- 实盘的 Python 计算逻辑 ---
                    if self.is_live_trading:
                        side = 'long' if action == 'LIMIT_BUY' else 'short'; final_size = 0.0
                        
                        try:
                            # 1. 从 AI 获取参数
                            leverage = int(order.get('leverage'))
                            risk_percent = float(order.get('risk_percent'))
                            limit_price = float(limit_price_from_ai)
                            stop_loss = float(order.get('stop_loss'))
                            take_profit = float(order.get('take_profit'))
                            
                            if risk_percent <= 0 or risk_percent > 0.5: raise ValueError(f"无效的 risk_percent: {risk_percent}")
                            if leverage <= 0 or leverage > 100: raise ValueError(f"无效的 leverage: {leverage}")
                            if limit_price <= 0: raise ValueError(f"无效的 limit_price: {limit_price}")

                            # 2. 从 Portfolio 获取账户状态
                            total_equity = float(self.portfolio.equity)
                            available_cash = float(self.portfolio.cash)
                            if total_equity <= 0: raise ValueError(f"无效账户状态 (Equity <= 0)")
                            
                            # 3. 执行 Prompt Rule 4 的计算
                            calculated_desired_margin = total_equity * risk_percent
                            
                            # [修复] 检查最大保证金占用率
                            max_margin_cap = total_equity * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
                            if calculated_desired_margin > max_margin_cap:
                                self.logger.warning(f"!!! {action} Margin Capped !!! AI 期望保证金 {calculated_desired_margin:.2f} > 最大 {max_margin_cap:.2f} ({futures_settings.MAX_MARGIN_PER_TRADE_RATIO*100}%)")
                                calculated_desired_margin = max_margin_cap

                            if calculated_desired_margin > available_cash:
                                self.logger.error(f"!!! {action} Aborted (Cash Insufficient) !!! AI 期望保证金 {calculated_desired_margin:.2f} > 可用 {available_cash:.2f}")
                                continue
                            
                            if calculated_desired_margin < MIN_MARGIN_USDT:
                                self.logger.warning(f"!!! {action} Margin Adjusted !!! AI 期望保证金 {calculated_desired_margin:.2f} < 最小 {MIN_MARGIN_USDT} USDT. 正在上调。")
                                final_desired_margin = MIN_MARGIN_USDT
                            else:
                                final_desired_margin = calculated_desired_margin
                            
                            final_size = (final_desired_margin * leverage) / limit_price

                            if symbol == "BTC/USDT:USDT":
                                if final_size < MIN_SIZE_BTC:
                                    self.logger.warning(f"!!! {action} BTC Size Adjusted !!! 计算后 size {final_size} < 最小 {MIN_SIZE_BTC}. 正在上调。")
                                    final_size = MIN_SIZE_BTC
                                    recalculated_margin = (final_size * limit_price) / leverage
                                    if recalculated_margin > available_cash:
                                        self.logger.error(f"!!! {action} Aborted (Cash Insufficient for Min BTC Size) !!! 最小 BTC size 需要 {recalculated_margin:.2f} 保证金 > 可用 {available_cash:.2f}")
                                        continue
                            
                            if final_size <= 0: raise ValueError("最终计算 size 为 0")

                        except (ValueError, TypeError, KeyError) as e: 
                            self.logger.error(f"跳过 {action} (Python 计算/参数错误): {order}. Err: {e}"); continue
                        
                        # 执行实盘限价单
                        invalidation_condition = order.get('invalidation_condition')
                        if hasattr(self.portfolio, 'live_open_limit'): 
                            # [通知] portfolio.live_open_limit 会自动发送通知
                            await self.portfolio.live_open_limit(
                                symbol, 
                                side, 
                                final_size, # <-- 经 Python 计算后的
                                leverage, 
                                limit_price,
                                reason=reason, 
                                stop_loss=stop_loss, 
                                take_profit=take_profit, 
                                invalidation_condition=invalidation_condition
                            )
                        else:
                             self.logger.error(f"AI 请求 LIMIT_BUY/SELL 但 portfolio 不支持 live_open_limit！")
                
                # --- 其他指令 ---
                elif action == "PARTIAL_CLOSE":
                    if (not current_price or current_price <= 0) and not self.is_live_trading:
                        self.logger.error(f"模拟盘部分平仓失败: 无当前价格 {symbol}"); continue
                    size_to_close_percent=None; size_to_close_absolute=None
                    try:
                        sp=order.get('size_percent'); sa=order.get('size_absolute')
                        if sp is not None: size_to_close_percent=float(sp)
                        elif sa is not None: size_to_close_absolute=float(sa)
                        else: raise ValueError("需提供 size_% 或 size_abs")
                        if size_to_close_percent is not None and not (0<size_to_close_percent<1.01): # 允许 1.0 (100%)
                            if abs(size_to_close_percent) < 1e-9: raise ValueError("size_% 必须 > 0")
                        if size_to_close_absolute is not None and size_to_close_absolute<=0: raise ValueError("size_abs 需 > 0")
                    except (ValueError,TypeError,KeyError) as e: self.logger.error(f"跳过PARTIAL_CLOSE参数错误: {order}. Err: {e}"); continue
                    
                    # [通知] portfolio.live_partial_close / paper_partial_close 会自动发送通知
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
                    
                    self.logger.warning(f"AI 请求更新止损 {symbol}: {new_stop_loss:.4f}. 原因: {reason}")
                    if hasattr(self.portfolio, 'update_position_rules'): 
                        await self.portfolio.update_position_rules(symbol, stop_loss=new_stop_loss, reason=reason)
                    else: 
                        self.logger.error(f"AI 尝试 UPDATE_STOPLOSS 但 portfolio 无 update_position_rules 方法。")
                
                elif action == "UPDATE_TAKEPROFIT":
                    new_take_profit = 0.0
                    try:
                        ntp=order.get('new_take_profit');
                        if ntp is None: raise ValueError("缺少 new_take_profit")
                        new_take_profit=float(ntp)
                        if new_take_profit<=0: raise ValueError("无效止盈价")
                    except (ValueError,TypeError,KeyError) as e: self.logger.error(f"跳过UPDATE_TAKEPROFIT参数错误: {order}. Err: {e}"); continue
                    
                    self.logger.warning(f"AI 请求更新止盈 {symbol}: {new_take_profit:.4f}. 原因: {reason}")
                    if hasattr(self.portfolio, 'update_position_rules'): 
                        await self.portfolio.update_position_rules(symbol, take_profit=new_take_profit, reason=reason)
                    else: 
                        self.logger.error(f"AI 尝试 UPDATE_TAKEPROFIT 但 portfolio 无 update_position_rules 方法。")

                else: 
                    self.logger.warning(f"收到未知 AI 指令 action: {action} in {order}")
            except Exception as e: 
                self.logger.error(f"处理 AI 指令时意外错误: {order}. Err: {e}", exc_info=True)


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
            self.logger.warning(f"AUTO-CLOSING (Paper): {symbol} | Reason: {reason}")
            # [通知] portfolio.paper_close 会自动发送通知
            await self.portfolio.paper_close(symbol, price, reason)
        return len(to_close) > 0

    
    # [新增] Python Rule 8 检查器
    async def _check_python_rule_8(self, market_data: Dict[str, Dict[str, Any]]) -> list:
        """
        在 Python 中高频执行 Rule 8 检查 (每 10 秒)。
        这将绕过 AI 的 1 分钟延迟。
        使用 config.py 中的设置。
        """
        # [配置] 检查 Rule 8 总开关
        if not settings.ENABLE_BREAKOUT_MODIFIER:
            return []
            
        orders_to_execute = []
        
        # 从 portfolio 获取当前仓位，避免 HFT 和 AI 冲突
        open_positions = self.portfolio.position_manager.get_all_open_positions()
        pending_limits = self.portfolio.pending_limit_orders

        # 定义 Rule 8 的定量规则 (来自原 Prompt)
        R8_ADX_MAX = 25.0 # (原 Prompt: 1h ADX < 25)
        R8_VOL_RATIO_5M = 2.5 # (原 Prompt: 5min_volume_ratio > 2.5)
        R8_VOL_RATIO_15M = 2.0 # (原 Prompt: 15min_volume_ratio > 2.0)
        
        # [修复] 使用 futures_settings
        leverage = futures_settings.FUTURES_LEVERAGE
        risk_percent = futures_settings.FUTURES_RISK_PER_TRADE_PERCENT / 100.0

        for symbol in self.symbols:
            # 状态管理：如果已有仓位，Rule 8 (Python) 不应干预
            if symbol in open_positions:
                continue
            
            # 状态管理：如果已有挂单，Rule 8 也不应干预 (防止与 Rule 6 冲突)
            if symbol in pending_limits:
                continue

            data = market_data.get(symbol)
            if not data: continue

            try:
                # 提取 Rule 8 所需的所有数据
                price = data.get('current_price')
                adx_1h = data.get('1hour_adx_14')
                bb_upper_15m = data.get('15min_bb_upper')
                bb_lower_15m = data.get('15min_bb_lower')
                vol_ratio_5m = data.get('5min_volume_ratio')
                vol_ratio_15m = data.get('15min_volume_ratio')
                ema_50_4h = data.get('4hour_ema_50')
                
                # 检查数据是否完整
                if not all([price, adx_1h, bb_upper_15m, bb_lower_15m, vol_ratio_5m, vol_ratio_15m, ema_50_4h]):
                    # self.logger.debug(f"Rule 8 Skipping {symbol}: Missing key data.") # 日志过于频繁
                    continue

                # --- 执行 Rule 8 逻辑 (a, b, c, d) ---
                
                # a. PRE-CONDITION (Squeeze)
                if not (adx_1h < R8_ADX_MAX):
                    continue # 不处于 Squeeze 状态

                # c. CONFIRMATION (Volume)
                volume_confirmed = (vol_ratio_5m > R8_VOL_RATIO_5M) or (vol_ratio_15m > R8_VOL_RATIO_15M)
                if not volume_confirmed:
                    continue # 成交量未确认

                # d. VETO (Macro)
                fng_value = self.fng_data.get('value', 50)
                
                # b. SIGNAL (Break) - 检查做多和做空
                
                action = None
                
                # --- 检查做多 (BUY) ---
                if price > bb_upper_15m:
                    # d. VETO (Trend & Macro)
                    if price < ema_50_4h: # 趋势否决 (价格低于 4h EMA 50)
                        # self.logger.debug(f"Rule 8 (Py) BUY Veto (Trend): {symbol}")
                        continue
                    if fng_value > 75: # 宏观否决 (极度贪婪)
                        self.logger.info(f"Rule 8 (Py) BUY Veto (F&G): {symbol}")
                        continue
                    action = "BUY"

                # --- 检查做空 (SELL) ---
                elif price < bb_lower_15m:
                    # d. VETO (Trend & Macro)
                    if price > ema_50_4h: # 趋势否决 (价格高于 4h EMA 50)
                        # self.logger.debug(f"Rule 8 (Py) SELL Veto (Trend): {symbol}")
                        continue
                    if fng_value < 25: # 宏观否决 (极度恐惧)
                        self.logger.info(f"Rule 8 (Py) SELL Veto (F&G): {symbol}")
                        continue
                    action = "SELL"

                # --- 构建订单 ---
                if action:
                    self.logger.warning(f"RULE 8 (Python): {action} Signal for {symbol}")
                    order = self._build_python_order(
                        symbol, 
                        action, 
                        risk_percent, 
                        leverage, 
                        price, 
                        market_data.get(symbol, {})
                    )
                    if order:
                        orders_to_execute.append(order)

            except Exception as e:
                self.logger.error(f"Rule 8 (Python) check failed for {symbol}: {e}")
                
        return orders_to_execute

    # [新增] Python Rule 8 订单构建器
    def _build_python_order(self, symbol: str, action: str, risk: float, lev: int, price: float, symbol_data: Dict) -> Optional[Dict]:
        """
        为 Python Rule 8 构建一个标准的订单对象。
        使用 config.py 中的 ATR 设置 来计算止损。
        止盈(TP) 为 None，以激活追踪止损逻辑。
        """
        
        stop_loss = 0.0
        
        # [修复] 使用 futures_settings
        if futures_settings.USE_ATR_FOR_INITIAL_STOP:
            # 我们使用 1h ATR 作为默认的止损基准
            atr_1h = symbol_data.get('1hour_atr_14')
            if not atr_1h or atr_1h <= 0:
                self.logger.warning(f"Rule 8: Missing 1h ATR for {symbol}. Using 2% static SL.")
                sl_pct = 0.02 # 备用静态止损
                stop_loss = price * (1 - sl_pct) if action == "BUY" else price * (1 + sl_pct)
            else:
                multiplier = futures_settings.INITIAL_STOP_ATR_MULTIPLIER
                stop_loss = price - (atr_1h * multiplier) if action == "BUY" else price + (atr_1h * multiplier)
        else:
            # 备用静态止损
            sl_pct = 0.02 
            stop_loss = price * (1 - sl_pct) if action == "BUY" else price * (1 + sl_pct)

        if stop_loss <= 0:
            self.logger.error(f"Rule 8: Invalid SL calculated for {symbol} (SL={stop_loss}). Aborting order.")
            return None

        return {
            "action": action,
            "symbol": symbol,
            "leverage": lev,
            "risk_percent": risk,
            "take_profit": None, # 关键: 设置为 None, 激活追踪止损
            "stop_loss": stop_loss,
            "invalidation_condition": "Python Rule 8", # "胶水" - 用于在风控循环中识别
            "reasoning": f"Python Rule 8 ({action}). ATR SL @ {stop_loss:.4f}"
        }


    # [新增] RSI 背离
    async def _check_divergence(self, ohlcv_15m: list) -> Tuple[bool, str]:
        """
        检查 15m K线上的 RSI 背离 (预测性触发)。
        - 看涨背离: 价格创新低, RSI 形成更高低点。
        - 看跌背离: 价格创新高, RSI 形成更低高点。
        """
        try:
            if len(ohlcv_15m) < 40: # 需要足够的回看周期 (e.g., 30 + 5)
                return False, ""
                
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v'])
            df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore')
            
            # 转换数据类型
            cols_to_numeric = ['h', 'l', 'c']
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['rsi'] = ta.rsi(df['c'], 14)
            df.dropna(inplace=True)
            
            if len(df) < 40: # 确保RSI计算后仍有足够数据
                return False, ""

            # 定义回看窗口
            recent_window = df.iloc[-5:] # 最近 5 根 K 线
            older_window = df.iloc[-30:-5] # 之前的 25 根 K 线 (-30 到 -5)

            # --- 1. 检查看跌背离 (Bearish Divergence) ---
            recent_high = recent_window['h'].max()
            recent_high_rsi = recent_window.loc[recent_window['h'].idxmax()]['rsi']
            older_high = older_window['h'].max()
            older_high_rsi = older_window.loc[older_window['h'].idxmax()]['rsi']
            if recent_high > older_high and recent_high_rsi < older_high_rsi:
                if df['h'].iloc[-1] == recent_high or df['h'].iloc[-2] == recent_high:
                    return True, "Event: 15m Bearish RSI Divergence"

            # --- 2. 检查看涨背离 (Bullish Divergence) ---
            recent_low = recent_window['l'].min()
            recent_low_rsi = recent_window.loc[recent_window['l'].idxmin()]['rsi']
            older_low = older_window['l'].min()
            older_low_rsi = older_window.loc[older_window['l'].idxmin()]['rsi']
            if recent_low < older_low and recent_low_rsi > older_low_rsi:
                if df['l'].iloc[-1] == recent_low or df['l'].iloc[-2] == recent_low:
                    return True, "Event: 15m Bullish RSI Divergence"
                
            return False, ""
        except Exception as e:
            self.logger.error(f"Err check divergence: {e}", exc_info=False)
            return False, ""

    # [新增] EMA 挤压
    async def _check_ema_squeeze(self, ohlcv_15m: list) -> Tuple[bool, str]:
        """
        检查 15m EMA(20) 和 EMA(50) 是否进入 "挤压" 状态 (预测性触发)。
        """
        try:
            if len(ohlcv_15m) < 60: 
                return False, ""
                
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v'])
            df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore')
            
            df['c'] = pd.to_numeric(df['c'], errors='coerce')
                
            df['ema20'] = ta.ema(df['c'], 20)
            df['ema50'] = ta.ema(df['c'], 50)
            df.dropna(inplace=True)
            
            if len(df) < 5: 
                return False, ""
            
            df['ema_diff_pct'] = abs(df['ema20'] - df['ema50']) / df['ema50']
            SQUEEZE_THRESHOLD = 0.005 # 0.5%
            diff_curr = df['ema_diff_pct'].iloc[-1]
            diff_prev = df['ema_diff_pct'].iloc[-2]
            is_squeezing_now = (diff_curr < SQUEEZE_THRESHOLD)
            was_squeezing_before = (diff_prev < SQUEEZE_THRESHOLD)
            
            if is_squeezing_now and not was_squeezing_before:
                self.logger.info(f"EMA Squeeze Trigger Check: Entered squeeze zone. Diff {diff_curr:.4%}")
                return True, f"Event: 15m EMA (20/50) Squeeze Entered (< {SQUEEZE_THRESHOLD*100:.1f}%)"

            return False, ""
        except Exception as e:
            self.logger.error(f"Err check EMA squeeze: {e}", exc_info=False)
            return False, ""

    # [新增] RSI 阈值
    async def _check_rsi_threshold_breach(self, ohlcv_15m: list) -> Tuple[bool, str]:
        """
        检查 15m RSI 是否 *接近* 超买/超卖 (预测性触发)。
        """
        try:
            if len(ohlcv_15m) < 16: 
                return False, ""
            
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v'])
            df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore')
            
            df['c'] = pd.to_numeric(df['c'], errors='coerce')
            
            rsi_df = df.ta.rsi(close=df['c'], length=14)
            if rsi_df is None or rsi_df.empty or len(rsi_df) < 2:
                return False, ""
            
            rsi_prev = rsi_df.iloc[-2] 
            rsi_curr = rsi_df.iloc[-1] 
            
            if rsi_prev < 68 and rsi_curr >= 68:
                return True, "Event: 15m RSI *Approaching* Overbought (68)"
                
            if rsi_prev > 32 and rsi_curr <= 32:
                return True, "Event: 15m RSI *Approaching* Oversold (32)"
                
            return False, ""
        except Exception as e:
            self.logger.error(f"Err check RSI threshold: {e}", exc_info=False)
            return False, ""

    # [新增] BBands 阈值
    async def _check_bollinger_band_breach(self, ohlcv_15m: list) -> Tuple[bool, str]:
        """
        检查 15m K线是否 *接近* 布林带 (预测性触发)。
        """
        try:
            if len(ohlcv_15m) < 22: 
                return False, ""
            
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v'])
            df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore')
            
            df['c'] = pd.to_numeric(df['c'], errors='coerce')
            
            period = 20; std_dev = 2.0; closes = df['c']
            if len(closes) < period: return False, ""

            middle_band = closes.rolling(window=period).mean()
            rolling_std = closes.rolling(window=period).std()
            upper_band = middle_band + (rolling_std * std_dev)
            lower_band = middle_band - (rolling_std * std_dev)

            if upper_band.isnull().all() or lower_band.isnull().all():
                return False, ""
            
            if pd.isna(upper_band.iloc[-1]) or pd.isna(lower_band.iloc[-1]):
                return False, ""

            close_curr = closes.iloc[-1]
            upper_curr = upper_band.iloc[-1]
            lower_curr = lower_band.iloc[-1]
            APPROACH_PERCENT = 0.002 
            
            if (upper_curr * (1.0 - APPROACH_PERCENT)) < close_curr < (upper_curr * (1.0 + APPROACH_PERCENT)):
                return True, "Event: 15m Price *Approaching* BB_Upper"
                
            if (lower_curr * (1.0 - APPROACH_PERCENT)) < close_curr < (lower_curr * (1.0 + APPROACH_PERCENT)):
                return True, "Event: 15m Price *Approaching* BB_Lower"
                
            return False, ""
        except Exception as e:
            self.logger.error(f"Err check BBand breach: {e}", exc_info=True) 
            return False, ""

    async def _update_fear_and_greed_index(self):
        """
        异步获取并缓存 Fear & Greed Index。
        """
        now = time.time()
        if now - self.last_fng_fetch_time < self.FNG_CACHE_DURATION_SECONDS:
            # self.logger.info(f"Using cached F&G Index: {self.fng_data['value_classification']} ({self.fng_data['value']})") # 日志过于频繁
            return

        self.logger.info("Fetching new Fear & Greed Index data from alternative.me...")
        url = "https://api.alternative.me/fng/?limit=1"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and len(data['data']) > 0:
                    fng_info = data['data'][0]
                    self.fng_data = {
                        "value": int(fng_info.get('value', 50)),
                        "value_classification": fng_info.get('value_classification', "Neutral")
                    }
                    self.last_fng_fetch_time = now
                    self.logger.info(f"Fetched new F&G Index: {self.fng_data['value_classification']} ({self.fng_data['value']})")
            else:
                self.logger.error(f"Failed to fetch F&G Index. Status code: {response.status_code}")
        
        except httpx.RequestError as e:
            self.logger.error(f"Error fetching F&G Index (httpx): {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error updating F&G Index: {e}", exc_info=True)


    async def run_cycle(self, market_data: Dict[str, Dict[str, Any]], tickers: Dict):
        """[AI 决策主循环 (Rule 6 - Limit Orders Only)]"""
        self.logger.info("="*20 + " Starting AI Cycle (Rule 6 - Limit Orders) " + "="*20)
        self.invocation_count += 1
        if not self.is_live_trading: await self._check_and_execute_hard_stops()

        # 1. 获取数据 & 构建 Prompt
        # [修改] 数据已从 start() 循环传入
        portfolio_state = self.portfolio.get_state_for_prompt(tickers)
        
        # [修改] F&G 已在 start() 循环中更新
        user_prompt_string = self._build_prompt(market_data, portfolio_state, tickers)

        # 2. 格式化 System Prompt (已移除 Rule 8)
        try:
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                symbol_list=self.formatted_symbols,
                specific_rules_or_notes=""
            )
        except KeyError as e: self.logger.error(f"Format System Prompt failed: {e}"); return

        self.logger.info("Getting AI decision (Rule 6)...")
        # 3. 获取 AI 决策
        ai_decision = await self._get_ai_decision(system_prompt, user_prompt_string)

        original_chain_of_thought = ai_decision.get("chain_of_thought", "AI No CoT.")
        orders = ai_decision.get("orders", [])
        self.logger.warning("--- AI CoT ---"); self.logger.warning(original_chain_of_thought)

        # 4. 提取摘要
        summary_for_ui = "AI 未提供摘要。" 
        summary_keyword_pattern = re.compile(r"In summary,", re.IGNORECASE)
        parts = summary_keyword_pattern.split(original_chain_of_thought, maxsplit=1)

        if len(parts) > 1:
            extracted_summary = parts[1].strip().lstrip(' :').rstrip('`')
            if extracted_summary:
                summary_for_ui = extracted_summary 
                self.logger.info(f"Extracted Chinese summary: '{summary_for_ui[:50]}...'")
            else:
                summary_for_ui = "AI 摘要为空。"
        else:
            self.logger.warning("AI CoT 未找到 'In summary,' 关键字。")
        
        self.last_strategy_summary = summary_for_ui

        # 5. 执行决策 (仅 Rule 6 LIMIT 订单)
        if orders:
            self.logger.info(f"AI proposed {len(orders)} order(s), executing...")
            await self._execute_decisions(orders, market_data)
        else:
            self.logger.info("AI proposed no orders.")

        self.logger.info("="*20 + " AI Cycle Finished " + "="*20 + "\n")

    async def start(self):
        """启动 AlphaTrader 主循环 (混合模型)。
        1. [高频] Python Rule 8 (突破) 在 10 秒循环中运行。
        2. [高频] 风控 (SL/TP/追踪) 在 10 秒循环中运行。
        3. [低频] AI Rule 6 (限价) 在定时器或事件触发时运行。
        """
        self.logger.warning(f"🚀 AlphaTrader starting! Mode: {'LIVE' if self.is_live_trading else 'PAPER'}")
        if self.is_live_trading:
            self.logger.warning("!!! LIVE MODE !!! Syncing state on startup...")
            if not hasattr(self, 'client') and hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
            try: 
                await self.portfolio.sync_state(); self.logger.warning("!!! LIVE State Sync Complete !!!")
            except Exception as e_sync: self.logger.critical(f"Initial LIVE state sync failed: {e_sync}", exc_info=True)
        
        # --- 定义风控阈值 (从 config 加载) ---
        MULTI_TP_STAGE_1_PERCENT = 0.04  # 4.0% (硬编码, 可移入 config)
        MULTI_TP_STAGE_2_PERCENT = 0.10  # 10.0% (硬编码, 可移入 config)
        MAX_LOSS_PERCENT = settings.MAX_LOSS_CUTOFF_PERCENT / 100.0
        DUST_MARGIN_USDT = 1.0 # (硬编码, 可移入 config)
        LIMIT_ORDER_TIMEOUT_MS = settings.AI_LIMIT_ORDER_TIMEOUT_SECONDS * 1000
        
        while True:
            try:
                # 步骤 1: 状态同步 (必须成功)
                try:
                    await self.portfolio.sync_state()
                    # self.logger.info("Portfolio state sync successful.") # 日志过于频繁
                except Exception as e_sync:
                    self.logger.critical(f"Main loop sync_state failed: {e_sync}. Skipping AI cycle, will retry...", exc_info=True)
                    await asyncio.sleep(30) 
                    continue
                
                # 步骤 2: 限价单超时清理 (AI Rule 6)
                if self.is_live_trading and self.portfolio.pending_limit_orders:
                    now_ms = time.time() * 1000
                    orders_to_cancel = []
                    try:
                        # [修复 V45.38] 迭代副本以安全删除
                        for symbol, plan in list(self.portfolio.pending_limit_orders.items()):
                            order_id = plan.get('order_id')
                            timestamp = plan.get('timestamp')
                            if not order_id:
                                self.logger.warning(f"Pending order {symbol} 缺少 order_id，正在从本地清理...")
                                # [修复 V45.38] 使用 "安全" 移除函数
                                await self.portfolio.remove_pending_limit_order(symbol)
                                continue
                            if not timestamp:
                                self.logger.warning(f"!!! ORPHAN TIMEOUT !!! {symbol} (ID: {order_id}) 缺少 timestamp。立即取消...")
                                orders_to_cancel.append((order_id, symbol))
                                # [修复 V45.38] 使用 "安全" 移除函数
                                await self.portfolio.remove_pending_limit_order(symbol)
                                continue
                            if (now_ms - timestamp) > LIMIT_ORDER_TIMEOUT_MS:
                                self.logger.warning(f"!!! LIMIT ORDER TIMEOUT !!! {symbol} (ID: {order_id}) 已超时 {LIMIT_ORDER_TIMEOUT_MS / 1000}s。正在取消...")
                                orders_to_cancel.append((order_id, symbol))
                                # [修复 V45.38] 使用 "安全" 移除函数
                                await self.portfolio.remove_pending_limit_order(symbol)
                        
                        if orders_to_cancel:
                            cancel_tasks = [self.client.cancel_order(oid, sym) for oid, sym in orders_to_cancel]
                            await asyncio.gather(*cancel_tasks, return_exceptions=True)
                            self.logger.info(f"成功取消 {len(orders_to_cancel)} 个超时/孤儿订单。")
                    except Exception as e_timeout:
                        self.logger.error(f"限价单超时清理时发生错误: {e_timeout}", exc_info=True)

                # 步骤 3: [高频] 获取数据 (为 Rule 8 和风控)
                await self._update_fear_and_greed_index()
                market_data, tickers = await self._gather_all_market_data()

                # 步骤 4: [高频] Python Rule 8 执行 (如果开启)
                # [配置] 检查 Rule 8 总开关
                if settings.ENABLE_BREAKOUT_MODIFIER:
                    if self.is_live_trading:
                        rule8_orders = await self._check_python_rule_8(market_data)
                        if rule8_orders:
                            self.logger.warning(f"🔥 Python Rule 8 TRIGGERED! Executing {len(rule8_orders)} orders.")
                            await self._execute_decisions(rule8_orders, market_data)
                    # else:
                        # self.logger.info("Python Rule 8 is enabled but in PAPER mode, skipping execution.") # 日志过于频繁
                
                # 步骤 5: [高频] 硬性风控检查 (每10秒)
                if self.is_live_trading and tickers:
                    positions_to_close = {} 
                    positions_to_partial_close = []
                    sl_update_tasks = [] # [新增] 用于 Rule 8 追踪止损

                    open_symbols = set(self.portfolio.position_manager.get_all_open_positions().keys())
                    for symbol in list(self.tp_counters.keys()):
                        if symbol not in open_symbols:
                            self.logger.info(f"Removing TP counter for closed position: {symbol}")
                            del self.tp_counters[symbol]

                    try:
                        open_positions = self.portfolio.position_manager.get_all_open_positions()
                        
                        for symbol, state in open_positions.items():
                            price = tickers.get(symbol, {}).get('last')
                            if not price or price <= 0: continue
                            
                            entry = state.get('avg_entry_price')
                            size = state.get('total_size')
                            side = state.get('side')
                            lev = state.get('leverage')
                            margin = state.get('margin')
                            
                            if not all([entry, size, side, lev, margin]) or lev <= 0 or entry <= 0 or margin <= 0:
                                self.logger.warning(f"Risk Check: Skipping {symbol}, invalid state data.")
                                continue

                            upl = (price - entry) * size if side == 'long' else (entry - price) * size
                            rate = upl / margin 

                            # --- [核心] 识别仓位类型 ---
                            # [!!! Bug 修复 (NoneType) !!!] 确保 inval_cond 始终为字符串，防止 'NoneType' is not iterable
                            inval_cond = state.get('invalidation_condition') or '' 
                            is_rule_8_trade = "Python Rule 8" in inval_cond

                            # --- 规则 A: Python Rule 8 动态追踪止损 ---
                            if is_rule_8_trade:
                                # [配置] Rule 8 仓位使用追踪止损
                                trail_percent = settings.BREAKOUT_TRAIL_STOP_PERCENT
                                current_sl = state.get('ai_suggested_stop_loss', 0.0)
                                new_sl = 0.0
                                
                                if side == 'long': new_sl = price * (1 - trail_percent)
                                else: new_sl = price * (1 + trail_percent)
                                
                                # 止损只能前进，不能后退
                                if (side == 'long' and new_sl > current_sl) or \
                                   (side == 'short' and new_sl < current_sl and new_sl > 0 and price < current_sl): # 确保空头新 SL 低于当前价格
                                    
                                    # self.logger.debug(f"!!! RULE 8 TRAIL STOP !!! {symbol} | Moving SL to {new_sl:.4f}") # 日志过于频繁
                                    sl_update_tasks.append(
                                        self.portfolio.update_position_rules(symbol, stop_loss=new_sl, reason="Rule 8 Trail Stop")
                                    )
                                continue # Rule 8 仓位不参与下面的 AI 规则

                            # --- 规则 B: AI (Rule 6) 仓位管理 ---

                            # 规则 B.1: 最大亏损 (Max Loss)
                            if rate <= -MAX_LOSS_PERCENT:
                                reason = f"Hard Max Loss ({-MAX_LOSS_PERCENT:.0%})"
                                if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                continue 

                            # 规则 B.2: 粉尘仓位 (Dust)
                            if margin < DUST_MARGIN_USDT:
                                reason = f"Dust Close (<{DUST_MARGIN_USDT:.1f}U)"
                                if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                continue 

                            # 规则 B.3: AI 设置的止损 (AI SL)
                            ai_sl = state.get('ai_suggested_stop_loss')
                            if ai_sl and ai_sl > 0:
                                if (side == 'long' and price <= ai_sl) or (side == 'short' and price >= ai_sl):
                                    reason = f"AI SL Hit ({ai_sl:.4f})"
                                    if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                    continue 

                            # 规则 B.4: AI 设置的止盈 (AI TP)
                            ai_tp = state.get('ai_suggested_take_profit')
                            if ai_tp and ai_tp > 0:
                                if (side == 'long' and price >= ai_tp) or (side == 'short' and price <= ai_tp):
                                    reason = f"AI TP Hit ({ai_tp:.4f})"
                                    if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                    continue 

                            # 规则 B.5: 多阶段止盈 (Multi-Stage TP)
                            self.tp_counters.setdefault(symbol, {'stage1': 0, 'stage2': 0})
                            counters = self.tp_counters[symbol]
                            if rate < 0:
                                if counters['stage1'] == 1 or counters['stage2'] == 1:
                                    counters['stage1'] = 0; counters['stage2'] = 0
                            elif rate >= MULTI_TP_STAGE_2_PERCENT and counters['stage2'] == 0:
                                positions_to_partial_close.append((symbol, 0.5, f"Hard TP Stage 2 (>{MULTI_TP_STAGE_2_PERCENT:.0%})"))
                                counters['stage2'] = 1; counters['stage1'] = 1 
                            elif rate >= MULTI_TP_STAGE_1_PERCENT and counters['stage1'] == 0:
                                positions_to_partial_close.append((symbol, 0.5, f"Hard TP Stage 1 (>{MULTI_TP_STAGE_1_PERCENT:.0%})"))
                                counters['stage1'] = 1 

                    except Exception as e_risk:
                        self.logger.error(f"High-frequency risk check error: {e_risk}", exc_info=True)

                    # 3. 执行风控动作
                    # [通知] portfolio.py (Source 2) 会自动处理这些动作的通知
                    if sl_update_tasks:
                        await asyncio.gather(*sl_update_tasks, return_exceptions=True) # 忽略错误 (例如仓位已不存在)
                    if positions_to_close:
                         tasks_close = [self.portfolio.live_close(symbol, reason=reason) for symbol, reason in positions_to_close.items()]
                         await asyncio.gather(*tasks_close, return_exceptions=True)
                         self.logger.info(f"Hard Close actions executed for: {list(positions_to_close.keys())}")
                    if positions_to_partial_close:
                        final_partial_tasks = []
                        for symbol, size_pct, reason in positions_to_partial_close:
                            if symbol not in positions_to_close:
                                final_partial_tasks.append(self.portfolio.live_partial_close(symbol, size_percent=size_pct, reason=reason))
                        if final_partial_tasks:
                            await asyncio.gather(*final_partial_tasks, return_exceptions=True)
                
                # --- 风控结束 ---

                # 步骤 6: [低频] 决定是否触发 AI (Rule 6)
                trigger_ai, reason, now = False, "", time.time()
                interval = settings.ALPHA_ANALYSIS_INTERVAL_SECONDS
                if now - self.last_run_time >= interval: trigger_ai, reason = True, "Scheduled"
                
                if not trigger_ai:
                    cooldown = settings.AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES
                    
                    if now - self.last_event_trigger_ai_time > (cooldown * 60):
                        # self.logger.info("Checking for advanced triggers (Divergence, Squeeze, etc.) across all symbols...") # 日志过于频繁
                        for sym in self.symbols:
                            ohlcv_15m = []
                            try: 
                                # [优化] 我们需要从 market_data 中获取 ohlcv 列表
                                # (假设 _gather_all_market_data 未来会返回这个)
                                # [B计划] 现在，我们还是手动获取
                                ohlcv_15m = await self.exchange.fetch_ohlcv(sym, '15m', limit=150)
                                if not ohlcv_15m: continue

                                event, ev_reason = await self._check_divergence(ohlcv_15m)
                                if not event: event, ev_reason = await self._check_ema_squeeze(ohlcv_15m)
                                
                                # [!!! Bug 修复 (NameError) !!!]
                                # 将 'ohlcv_1sSsm' 修正为 'ohlcv_15m'
                                if not event: event, ev_reason = await self._check_rsi_threshold_breach(ohlcv_15m)
                                
                                if not event: event, ev_reason = await self._check_bollinger_band_breach(ohlcv_15m)
                                
                                if event: 
                                    trigger_ai, reason = True, f"{sym}: {ev_reason}"
                                    self.logger.info(f"Advanced trigger found for {sym}: {ev_reason}")
                                    break
                            except Exception as e_fetch:
                                self.logger.error(f"Event check for {sym} fail: {e_fetch}", exc_info=False) # 减少日志噪音
                        # if not trigger_ai:
                             # self.logger.info("No advanced triggers found.") # 日志过于频繁
                
                # 步骤 7: (安全地) 运行 AI 循环
                if trigger_ai:
                    self.logger.warning(f"🔥 AI triggered! Reason: {reason} (Sync was successful)")
                    if reason != "Scheduled": self.last_event_trigger_ai_time = now
                    # [修改] 传入已获取的数据
                    await self.run_cycle(market_data, tickers); 
                    self.last_run_time = now
                
                await asyncio.sleep(10) # 10秒主循环
            except asyncio.CancelledError: self.logger.warning("Task cancelled, shutting down..."); break
            except Exception as e: 
                self.logger.critical(f"Main loop fatal error (outside sync/AI): {e}", exc_info=True); 
                await asyncio.sleep(60)
