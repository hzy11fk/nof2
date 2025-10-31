# 文件: alpha_trader.py (V45.34 - 硬风控重构版)
# 1. [硬风控] 将 多阶段止盈(+4%/+10%) 移至 start() 循环。
# 2. [硬风控] 将 AI设置的SL/TP执行 移至 start() 循环。
# 3. [硬风控] 增加 粉尘仓位(<1U) 自动平仓。
# 4. [硬风控] 将 最大亏损(-20%) 移至 start() 循环。
# 5. [AI Prompt] AI 现在专注于“更新SL/TP”和“识别反转行情”。

import logging
import asyncio
import time
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
import re
import httpx
from collections import deque
from config import settings, futures_settings
from alpha_ai_analyzer import AlphaAIAnalyzer
from alpha_portfolio import AlphaPortfolio 
from datetime import datetime
from typing import Tuple, Dict, Any, Set, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

class AlphaTrader:
    
    # --- [ V45.34 PROMPT 优化 ] ---
    # 1. [移除] 删除了旧的 "Rule 2 (Profit Protection Mandate)"。
    # 2. [移除] 删除了 CoT 模板中的 "Profit Management", "Max Loss Cutoff", "Trailing Stop Assessment"。
    # 3. [新增] "Rule 2: Active SL/TP Management" -> AI 必须主动更新 SL/TP。
    # 4. [新增] CoT 模板中增加 "Reversal & Profit Save Check" -> AI 主动识别反转以平仓。
    # 5. [新增] CoT 模板中增加 "SL/TP Target Update Check" -> AI 执行 Rule 2。
    SYSTEM_PROMPT_TEMPLATE = """
    You are a **profit-driven, analytical, and disciplined** quantitative trading AI. Your primary goal is to **generate and secure realized profit**. You are not a gambler; you are a calculating strategist.

    **You ONLY execute a trade (BUY, SELL, PARTIAL_CLOSE) if you have a high-confidence assessment that the action will lead to profit.** A medium or low-confidence signal means you WAIT.

    Your discipline is demonstrated by strict adherence to the risk management rules below, which are your foundation for sustained profitability.

    **Core Mandates & Rules:**
    1.  **Rule-Based Position Management:** For every open position, you MUST check its `InvalidATION_CONDITION`. If this condition is met, you MUST issue a `CLOSE` order. This is your top priority for existing positions.

    2.  **Active SL/TP Management (CRITICAL):** Your main task for existing positions is to actively manage risk.
        -   For *every* open position on *every* analysis cycle, you MUST assess if the existing `ai_suggested_stop_loss` or `ai_suggested_take_profit` targets are still optimal.
        -   The system executes these targets automatically, but your job is to UPDATE them.
        -   If the market structure changes (e.g., a new S/R level appears, volatility drops), you MUST issue `UPDATE_STOPLOSS` or `UPDATE_TAKEPROFIT` orders with new, improved targets and your reasoning.

    3.  **Risk Management Foundation (CRITICAL):** Profit is the goal, but capital preservation is the foundation. You MUST strictly follow these rules:
        -   **Leverage Selection:**You should use *lower* leverage (e.g., 5x-8x) for higher volatility assets (e.g., SOL, DOGE) and *moderate* leverage (e.g., 10x-15x) for lower volatility assets (e.g., BTC, ETH). Your chosen leverage (e.g., `leverage: 8`) MUST be stated in the reasoning.
        -   **Single Position Sizing (Open/Add):** When opening a new position OR adding to an existing one, you MUST calculate the size based on **Total Equity**, not Available Cash.
        -   **CALCULATION FORMULA (MANDATORY):** You MUST follow this formula for EACH `BUY`/`SELL` order:
            1.  **Choose a `risk_percent` (DYNAMICALLY):** Your chosen `risk_percent` MUST be based on the **trade confidence** (derived from your 'Signal Confluence Score'):
                * **High Confidence (Score: 4/4, all signals align, F&G confirms, Volume confirms):** Use a higher risk, e.g., `risk_percent = 0.05` (5% Equity).
                * **Medium Confidence (Score: 3/4, minor conflicts, F&G neutral):** Use a lower risk, e.g., `risk_percent = 0.025` (2.5% Equity).
                * **Low Confidence (Score: 1-2/4):** ABORT. Do not trade.
            2.  `calculated_desired_margin = Total Equity * risk_percent`.
            3.  **Check Cash:** Is `calculated_desired_margin` <= `Available Cash`?
                -   IF NO: **Abort the trade.** (Cash is insufficient for this risk).
                -   IF YES: Proceed to next step.
            4.  **Check Minimum Margin (CRITICAL):**
                -   IF `calculated_desired_margin` < 6.0: **Abort the trade.** Your risk calculation (${{calculated_desired_margin:.2f}}) is below the 6.0 USDT minimum margin. The trade is too small to be valid.
                -   IF `calculated_desired_margin` >= 6.0: `final_desired_margin = calculated_desired_margin`. (Proceed)
            5.  `size = (final_desired_margin * leverage) / current_price`.
            6.  **Check BTC Minimum Size (CRITICAL):**
                -   IF `symbol` is "BTC/USDT:USDT":
                    -   IF `size` >= 0.001: **Proceed.** (Size is valid).
                    -   IF `size` < 0.001:
                        -   **Action:** Recalculate based on the minimum size.
                        -   `new_size = 0.001`
                        -   `recalculated_margin = (0.001 * current_price) / leverage`
                        -   **Check Cash Again:** Is `recalculated_margin` <= `Available Cash`?
                            -   IF NO: **Abort the trade.** (Cash is insufficient for the minimum BTC size: ${{recalculated_margin:.2f}} > ${{Available Cash:.2f}}).
                            -   IF YES: **Proceed.** (Use the adjusted values: `final_desired_margin = recalculated_margin`, `size = new_size`).
        -   **Total Exposure:** The sum of all margins for all open positions should generally not exceed 50-60% of your total equity.
        -   **Correlation Control (Hard Cap):** You MUST limit total risk exposure to highly correlated assets.
            -   Define 'Core Crypto Group' as [BTC, ETH]. Total margin for this group MUST NOT exceed 30% of Total Equity.
            -   Define 'Altcoin Group' as [SOL, BNB, DOGE, XRP]. Total margin for this group MUST NOT exceed 40% of Total Equity.
            -   If opening a new position (e.g., SOL) would breach its group cap, you MUST ABORT the trade.

    4.  **Complete Trade Plans (Open/Add):** Every new `BUY` or `SELL` order is a complete plan. You MUST provide: `take_profit`, `stop_loss`, `invalidation_condition`.
        -   **Smarter Invalidation:** Your `invalidation_condition` MUST be based on a clear technical breakdown of the *original trade thesis*.
            -   *Trend Trade Example:* `Invalidation='1h Close below the EMA 50'` (if thesis was a 1h uptrend).
            -   *Trend Trade Example:* `Invalidation='4h ADX drops below 20'` (if thesis was a 4h trend).
            -   *Ranging Trade Example:* `InvalidATION='15m RSI breaks above 60'` (if thesis was a 15m overbought short).
        -   **Profit-Taking Strategy:** You SHOULD consider using multiple take-profit levels (by using `PARTIAL_CLOSE` later) rather than a single `take_profit`.

    5.  **Market State Recognition (Using ADX & BBands):**
        You MUST continuously assess the market regime using the **1hour** and **4hour** timeframes.
        -   **1. Strong Trend (Trending Bullish/Bearish):**
            -   **Condition:** 1h or 4h **ADX_14 > 25**.
            -   **Strategy:** In this regime, **EMA crossovers** and **MACD** signals are your primary tools. You MUST trade WITH the trend.
            -   **RSI:** In strong trends, RSI can stay "overbought"/"oversold" for a long time. DO NOT use RSI for counter-trend entries.
        -   **2. Ranging (No Trend):**
            -   **Condition:** 1h and 4h **ADX_14 < 20**.
            -   **Strategy:** In this regime, **RSI** (overbought/oversold) and **Bollinger Bands (BB_Upper/BB_Lower)** are your primary tools. You should look for mean-reversion trades.
            -   **INVALIDATION:** In a ranging market, you MUST **IGNORE** all EMA and MACD crossover signals, as they will be false.
        -   **3. Chop (Uncertain):**
            -   **Condition:** 1h or 4h **ADX_14 is between 20 and 25**.
            -   **Strategy:** This is an uncertain market. **WAIT** for a clear signal (ADX > 25 or ADX < 20).

    6.  **Market Sentiment Filter (Fear & Greed Index):**
        You MUST use the provided `Fear & Greed Index` (from the User Prompt) as a macro filter for your decisions.
        -   **Extreme Fear (Index < 25):** The market is panicking.
            -   **Action:** Be EXTREMELY cautious with new LONG signals (high risk of failure). Prioritize capital preservation. SHORT signals (breakdowns) are higher confidence.
        -   **Fear (Index 25-45):** Market is fearful.
            -   **Action:** Be cautious with LONGs. Seek strong (4/4) confluence.
        -   **Neutral (Index 45-55):** No strong sentiment bias.
            -   **Action:** Rely primarily on technical (ADX/BBands/RSI) analysis.
        -   **Greed (Index 55-75):** Market is optimistic.
            -   **Action:** LONG signals (pullbacks) are higher confidence. Be cautious with new SHORTs.
        -   **Extreme Greed (Index > 75):** The market is euphoric (high risk of reversal).
            -   **Action:** Be EXTREMELY cautious opening new LONGs (risk of "buying the top"). Actively look for `PARTIAL_CLOSE` opportunities on existing LONG positions.

    **Multi-Timeframe Confirmation Requirement (CRITICAL):**
    - You MUST analyze and confirm signals across available timeframes: **5min, 15min, 1hour, and 4hour**.
    - **High-Confidence Signal Definition:** A signal is only high-confidence when it aligns with the **Market State** (see Rule 5) and shows alignment across **at least 3** timeframes.
    - **Timeframe Hierarchy:** Use longer timeframes (**4h, 1h**) to determine the **Market State** and **Overall Trend**. Use shorter timeframes (**15min, 5min**) for precise entry timing.
    - **Volume Confirmation:** Significant price moves MUST be confirmed by above-average volume (volume_ratio > 1.2).

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
        - 1h ADX: [Value] | 4h ADX: [Value]
        - Regime: [Trending Bullish (ADX>25) / Trending Bearish (ADX>25) / Ranging (ADX<20) / Chop (ADX 20-25)]
        - Key Support/Resistance Levels: [Identify major S/R levels, including BB_Upper/Lower and recent_high/low for relevant symbols]
        - Volume Analysis: [Assess volume confirmation]
        - Market Sentiment: [MUST state the F&G Index value and its implication, e.g., "Extreme Greed (80)"]

        Portfolio Overview:
        Total Equity: $X, Available Cash: $Y, Current Margin Usage: Z%
        Current Market Correlation Assessment: [Assess if positions are overly correlated based on Correlation Control (Rule 3) hard caps]

        Let's break down each position:
        1. [SYMBOL] ([SIDE]):
           UPL: [Current Unrealized PNL and Percent (e.g., +$50.00 (+5.5%))]
           Multi-Timeframe Analysis: [Brief assessment across 5m, 15m, 1h, 4h, mentioning ADX/BBands]
           
           Invalidation Check: [Check condition vs current data. If met, MUST issue CLOSE.]
           
           Reversal & Profit Save Check (NEW):
           - [Is this profitable position (UPL > +1.0%) showing strong signs of reversal (e.g., 15m bearish divergence, 1h ADX weakening, F&G Extreme Greed)?]
           - [IF YES: The risk of giving back profits is high. Prioritize capital preservation. Decision: MUST issue a CLOSE order to secure profits.]

           Pyramiding Check (Adding to a Winner):
           - [Is UPL Percent > +2.5% AND is the original trend (ADX > 25) still strong?]
           - [AND has price pulled back to a key support (for Long) / resistance (for Short) (e.g., 1h EMA 20)?]
           - [IF YES: Consider an `ADD` order. This new entry is treated as a separate trade and MUST follow the full Rule 3 (Sizing) / Rule 4 (SL/TP) logic.]
           - [CRITICAL: You MUST NEVER add to a losing position (UPL < 0). Averaging down is forbidden.]
           
           SL/TP Target Update Check (NEW):
           - [Are the current ai_suggested_stop_loss/take_profit targets still optimal based on new data (e.g., move SL up to new 15m BB_Mid, lower TP from 4h BB_Upper)?]
           - [IF NOT OPTIMAL: Issue UPDATE_STOPLOSS / UPDATE_TAKEPROFIT with new targets and reasoning.]
           
           Decision: [Hold/Close/Partial Close/Add/Update StopLoss/Update TakeProfit + Reason. NOTE: Invalidation and Reversal checks override all "Hold" decisions.]

        ... [Repeat for each open position] ...

        New Trade Opportunities Analysis:
        Available Margin for New Trades: [Calculate based on Total Equity and risk rules]
        Correlation Check: [Ensure new trades don't breach Rule 3 Correlation Control hard caps]

        Multi-Timeframe Signal Requirements (Must meet 3+ factors on 5m, 15m, 1h, 4h):
        - Trend alignment (or Ranging setup) confirmed by Market State (ADX/BBands)
        - Signal confirmed by appropriate indicators (MACD/EMA for Trend, RSI/BBands for Range)
        - Volume confirmation (volume_ratio > 1.2)
        - Absence of strong counter-evidence across timeframes

        Specific Multi-Timeframe Opportunity Analysis:
        [For each symbol, analyze BOTH long and short scenarios based on the detected Market State (Trend vs Range)]
        [EXAMPLE - TRENDING MARKET:]
        BTC Multi-Timeframe Assessment (Market State: Trending Bullish, 4h ADX=28):
        - 4h Trend: Bullish (EMA 20 > 50) | 1h Momentum: Strong (MACD > 0) | 15min Setup: Pullback to EMA 20 | 5min Trigger: Confirmed
        - Signal Confluence Score: 4/4 | Final Confidence: High - CONSIDER LONG

        [EXAMPLE - RANGING MARKET:]
        ETH Multi-Timeframe Assessment (Market State: Ranging, 1h ADX=18):
        - 4h Trend: N/A (ADX < 20) | 1h Setup: Price approaching BB_Upper | 15min RSI: 72 (Overbought) | 5min Trigger: Weak bearish div
        - Signal Confluence Score: 3/4 (RSI/BBands align) | Final Confidence: Medium - CONSIDER SHORT (Mean Reversion)

        In summary, [**Key Instruction: Please provide your final concise decision overview directly here, in Chinese.**Final concise decision overview.]
        ```

    2.  `"orders"` (list): A list of JSON objects for trades. Empty list `[]` if holding all.

    **Order Object Rules:**
    -   **To Open or Add (LONG):**`{{"action": "BUY", "symbol": "...", "size": [CALCULATED_SIZE], "leverage": [CHOSEN_LEVERAGE], "take_profit": ..., "stop_loss": ..., "invalidation_condition": "...", "reasoning": "Leverage chosen: [Why 10x? or Why 5x?]. Calculation: Based on Total Equity. final_margin={{final_margin_usd:.2f}} (must be >= 6.0). size=(Final Margin)*lev/price=... Multi-TF confirm: [...]. Market State: [...]"}}`
    -   **To Open or Add (SHORT):**`{{"action": "SELL", "symbol": "...", "size": [CALCULATED_SIZE], "leverage": [CHOSEN_LEVERAGE], "take_profit": ..., "stop_loss": ..., "invalidation_condition": "...", "reasoning": "Leverage chosen: [Why 10x? or Why 5x?]. Calculation: Based on Total Equity. final_margin={{final_margin_usd:.2f}} (must be >= 6.0). size=(Final Margin)*lev/price=... Multi-TF confirm: [...]. Market State: [...]"}}`
    -   **To Close Fully:** `{{"action": "CLOSE", "symbol": "...", "reasoning": "Invalidation met / SL hit / TP hit / Max Loss Cutoff / Manual decision..."}}`
    -   **To Close Partially (Take Profit):** `{{"action": "PARTIAL_CLOSE", "symbol": "...", "size_percent": 0.5, "reasoning": "Taking 50% profit near resistance per Rule 4..."}}` (or `size_absolute`)
    -   **To Update Stop Loss:** `{{"action": "UPDATE_STOPLOSS", "symbol": "...", "new_stop_loss": ..., "reasoning": "Actively moving SL to new 15m support level..."}}`
    -   **To Update Take Profit (NEW):** `{{"action": "UPDATE_TAKEPROFIT", "symbol": "...", "new_take_profit": ..., "reasoning": "Actively moving TP to new 4h resistance level..."}}`
    -   **To Hold:** Do NOT include in `orders`.
    -   **Symbol Validity:** `symbol` MUST be one of {symbol_list}.

    **Remember:** Quality over quantity.
    """

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
        
        if hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
        else:
            if hasattr(self.portfolio, 'exchange') and isinstance(self.portfolio.exchange, object): self.client = self.portfolio.exchange; self.logger.warning("Portfolio missing 'client', falling back.")
            else: self.client = self.exchange; self.logger.warning("Portfolio missing 'client', using exchange directly.")
        
        # V45.26 新增
        self.fng_data: Dict[str, Any] = {"value": 50, "value_classification": "Neutral"}
        self.last_fng_fetch_time: float = 0.0
        self.FNG_CACHE_DURATION_SECONDS = 3600 # 1小时缓存 (3600秒)
        
        # --- [V45.34 新增] ---
        # 止盈计数器，用于多阶段止盈。
        # 结构: {'BTC/USDT:USDT': {'stage1': 0, 'stage2': 0}}
        # 0 = 未触发, 1 = 已触发
        # 注意：这将在机器人重启时重置。更健壮的方案是将其存储在 AlphaPositionManager 中。
        self.tp_counters: Dict[str, Dict[str, int]] = {}
        # --- [新增结束] ---


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
        """[V45.24] 修复：将 df.fillna(method='ffill') 更新为 df.ffill()，消除 FutureWarning。"""
        self.logger.info("Gathering multi-TF market data (5m, 15m, 1h, 4h) + ADX/BBands (Manual Calc)...")
        market_indicators_data: Dict[str, Dict[str, Any]] = {}
        fetched_tickers: Dict[str, Any] = {}
        
        try:
            timeframes = ['5m', '15m', '1h', '4h']
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
                        df = pd.DataFrame(ohlcv_data, columns=['ts', 'o', 'h', 'l', 'c', 'v']); df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore'); 
                        
                        cols_to_numeric = ['o', 'h', 'l', 'c', 'v']
                        for col in cols_to_numeric:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # --- [V45.24 核心修复] ---
                        # df.fillna(method='ffill', inplace=True) # 旧的、产生警告的代码
                        df.ffill(inplace=True) # 新的、建议的代码
                        # --- [修复结束] ---
                        
                        df.dropna(inplace=True)
                        
                        df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
                        
                        if df.empty: 
                            self.logger.warning(f"DataFrame empty for {symbol} {timeframe} after cleaning NaNs.")
                            continue
                            
                        prefix = f"{timeframe.replace('m', 'min').replace('h', 'hour')}_"

                        # --- [V45.23 核心修复：移植 futures_trader.py 的手动计算逻辑] ---
                        
                        # 1. 手动计算 ADX (移植自 get_adx_data)
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
                                plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9))
                                minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, 1e-9))
                                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
                                adx = dx.ewm(alpha=1/period, adjust=False).mean()
                                if not adx.empty:
                                    market_indicators_data[symbol][f'{prefix}adx_14'] = adx.iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"Manual ADX calc failed for {symbol} {timeframe}: {e}", exc_info=False)

                        # 2. 手动计算 BBands (移植自 get_bollinger_bands_data)
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

                        # 3. 手动计算 RSI (移植自 get_rsi_data)
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

                        # 4. 保留 ta.macd (之前未报错, 包裹 try-except)
                        if len(df) >= 26:
                            try:
                                macd = ta.macd(df['c'], 12, 26, 9)
                                if macd is not None and not macd.empty and 'MACD_12_26_9' in macd.columns:
                                    market_indicators_data[symbol][f'{prefix}macd'] = macd['MACD_12_26_9'].iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}macd_hist'] = macd['MACDh_12_26_9'].iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"ta.macd calc failed for {symbol} {timeframe}: {e}", exc_info=False)
                                
                        # 5. 保留 ta.ema (之前未报错, 包裹 try-except)
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
                        # --- [V45.23 修复结束] ---

                        # (非 ta-strategy 计算的指标)
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
        """[V45.17] 构建 User Prompt, 增加 ADX/BBands"""
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
                prompt += f" ADX:{safe_format(d.get(f'{tf}_adx_14'), 0, is_rsi=True)}|" # [V45.17 新增]
                prompt += f"MACD:{safe_format(d.get(f'{tf}_macd'), 4)}|"
                prompt += f"Sig:{safe_format(d.get(f'{tf}_macd_signal'), 4)}|"
                prompt += f"Hist:{safe_format(d.get(f'{tf}_macd_hist'), 4)}\n"
                prompt += f" EMA20:{safe_format(d.get(f'{tf}_ema_20'), 3)}|"
                prompt += f"EMA50:{safe_format(d.get(f'{tf}_ema_50'), 3)}|"
                prompt += f"VolR:{safe_format(d.get(f'{tf}_volume_ratio'), 1)}x\n"
                prompt += f" BB_Up:{safe_format(d.get(f'{tf}_bb_upper'), 3)}|"   # [V45.17 新增]
                prompt += f"BB_Mid:{safe_format(d.get(f'{tf}_bb_middle'), 3)}|" # [V45.17 新增]
                prompt += f"BB_Low:{safe_format(d.get(f'{tf}_bb_lower'), 3)}\n" # [V45.17 新增]
                prompt += f" Hi:{safe_format(d.get(f'{tf}_recent_high'), 2)}|"
                prompt += f"Lo:{safe_format(d.get(f'{tf}_recent_low'), 2)}|"
                prompt += f"Chg:{safe_format(d.get(f'{tf}_price_change_pct'), 1)}%\n"
            prompt += "-----\n"
        # --- [修改] 在账户信息之前插入市场情绪上下文 ---
        prompt += "\n--- Market Context ---\n"
        fng_val = self.fng_data.get('value', 50)
        fng_class = self.fng_data.get('value_classification', 'Neutral').title() # e.g. "Extreme Fear"
        prompt += f"Fear & Greed Index: {fng_val} ({fng_class})\n"
        # --- [修改结束] ---    
        prompt += "\n--- Account Info ---\n"
        prompt += f"Return%: {portfolio_state.get('performance_percent', 'N/A')}\n"
        prompt += f"Total Equity: {portfolio_state.get('account_value_usd', 'N/A')}\n" # [V4S.17 核心] AI 现在需要 Equity
        prompt += f"Available Cash: {portfolio_state.get('cash_usd', 'N/A')}\n" # [V45.17 核心] AI 同时需要 Cash
        prompt += "Positions:\n"
        prompt += portfolio_state.get('open_positions', "No open positions.")
        return prompt

    async def _get_ai_decision(self, system_prompt: str, user_prompt: str) -> dict:
        """调用 AI 分析器"""
        if not self.ai_analyzer: return {}
        return await self.ai_analyzer.get_ai_response(system_prompt, user_prompt)

    async def _execute_decisions(self, decisions: list, market_data: Dict[str, Dict[str, Any]]):
        """[V4R.34 优化] 增加 UPDATE_TAKEPROFIT 逻辑"""
        
        MIN_MARGIN_USDT = 6.0
        MIN_SIZE_BTC = 0.001 

        for order in decisions:
            try:
                action = order.get('action'); symbol = order.get('symbol')
                if not action or not symbol or symbol not in self.symbols: self.logger.warning(f"跳过无效指令: {order}"); continue
                reason = order.get('reasoning', 'N/A'); current_price = market_data.get(symbol, {}).get('current_price')
                
                # [V45.17 修复] 允许 UPDATE_STOPLOSS/TAKEPROFIT 在没有价格时也能执行
                if action not in ["UPDATE_STOPLOSS", "UPDATE_TAKEPROFIT"] and (not current_price or current_price <= 0):
                    self.logger.error(f"无价格 {symbol}，跳过: {order}"); continue
                
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
                        
                        if symbol == "BTC/USDT:USDT" and final_size > 0 and final_size < MIN_SIZE_BTC:
                            self.logger.warning(f"!!! 硬控触发 (BTC最小数量) !!! 计算出的 Size {final_size:.8f} < {MIN_SIZE_BTC}.")
                            final_size = MIN_SIZE_BTC
                            final_margin = (final_size * current_price) / leverage if leverage > 0 else 0.0
                            self.logger.warning(f"已修正 Size 为 {MIN_SIZE_BTC}。实际保证金变为: {final_margin:.4f} USDT。")

                        if final_size <= 0: raise ValueError("最终 size <= 0")
                        
                    except (ValueError, TypeError, KeyError) as e: 
                        self.logger.error(f"跳过BUY/SELL参数/计算错误: {order}. Err: {e}"); continue
                    
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
                    
                    self.logger.warning(f"AI 请求更新止损 {symbol}: {new_stop_loss:.4f}. 原因: {reason}")
                    if hasattr(self.portfolio, 'update_position_rules'): 
                        await self.portfolio.update_position_rules(symbol, stop_loss=new_stop_loss, reason=reason)
                    else: 
                        self.logger.error(f"AI 尝试 UPDATE_STOPLOSS 但 portfolio 无 update_position_rules 方法。")
                
                # --- [V45.34 新增] ---
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
                # --- [新增结束] ---

                else: 
                    self.logger.warning(f"收到未知 AI 指令 action: {action} in {order}")
            except Exception as e: 
                self.logger.error(f"处理 AI 指令时意外错误: {order}. Err: {e}", exc_info=True)

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
        """
        [V45.29 修复] 检查 1h 价格大幅波动 (边缘触发)。
        仅在 "上一根K线" 出现波动，而 "上上一根K线" 没有波动时，才触发一次。
        """
        try:
            # 我们现在需要比较 [-2] 和 [-3]，所以至少需要 3 根K线
            if len(ohlcv_1h) < 3: 
                return False, ""
            
            # --- V45.29 修复逻辑 ---
            
            # 1. 定义波动阈值
            threshold_pct = settings.AI_VOLATILITY_TRIGGER_PERCENT / 100.0
            
            # 2. 检查 "上一根K线" ([-2])
            o_curr, c_curr = ohlcv_1h[-2][1], ohlcv_1h[-2][4]
            chg_curr = abs(c_curr - o_curr) / o_curr if o_curr > 0 else 0.0
            is_spiked_curr = chg_curr >= threshold_pct

            # 3. 检查 "上上一根K线" ([-3])
            o_prev, c_prev = ohlcv_1h[-3][1], ohlcv_1h[-3][4]
            chg_prev = abs(c_prev - o_prev) / o_prev if o_prev > 0 else 0.0
            is_spiked_prev = chg_prev >= threshold_pct
            
            # 4. 边缘触发：仅在当前K线有波动，且前一根K线没有波动时触发
            if is_spiked_curr and not is_spiked_prev:
                direction = 'up' if c_curr > o_curr else 'down'
                return True, f"Event: 1h price spike {chg_curr:.1%} ({direction})"
            
            # --- 修复结束 ---
                
            return False, ""
        except Exception as e: 
            self.logger.error(f"Err check market volatility: {e}", exc_info=False)
            return False, ""

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


    async def _check_rsi_threshold_breach(self, ohlcv_15m: list) -> Tuple[bool, str]:
        """
        [新增 V45.29] 检查 15m RSI 是否穿越了超买/超卖阈值。
        这对于 AI 的 'Ranging' 策略至关重要。
        """
        try:
            if len(ohlcv_15m) < 16: # 需要 14 + 2 根 K线
                return False, ""
            
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v'])
            df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore')
            
            # 计算 RSI
            rsi_df = df.ta.rsi(close=df['c'], length=14)
            if rsi_df is None or rsi_df.empty or len(rsi_df) < 2:
                return False, ""
            
            rsi_prev = rsi_df.iloc[-2] # 上一根 K 线的 RSI
            rsi_curr = rsi_df.iloc[-1] # 当前 K 线的 RSI
            
            # 检查超买 (从下往上穿过 70)
            if rsi_prev < 70 and rsi_curr >= 70:
                return True, "Event: 15m RSI Breach Overbought (70)"
                
            # 检查超卖 (从上往下穿过 30)
            if rsi_prev > 30 and rsi_curr <= 30:
                return True, "Event: 15m RSI Breach Oversold (30)"
                
            return False, ""
        except Exception as e:
            self.logger.error(f"Err check RSI threshold: {e}", exc_info=False)
            return False, ""

    async def _check_bollinger_band_breach(self, ohlcv_15m: list) -> Tuple[bool, str]:
        """
        [新增 V45.29] 检查 15m K线是否穿越了布林带上轨或下轨。
        [V45.30 修复] 改为使用与 _gather_all_market_data 一致的手动计算逻辑，不再依赖 ta.bbands。
        """
        try:
            # 需要 20 (period) + 2 (for compare) = 22 根 K线
            if len(ohlcv_15m) < 22: 
                return False, ""
            
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v'])
            df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore')
            
            # --- [V45.30 修复] ---
            # 1. 使用与 _gather_all_market_data 相同的手动计算逻辑
            period = 20
            std_dev = 2.0
            closes = df['c'] # 这是 pd.Series
            
            if len(closes) < period: # 再次检查
                return False, ""

            middle_band = closes.rolling(window=period).mean()
            rolling_std = closes.rolling(window=period).std()
            upper_band = middle_band + (rolling_std * std_dev) # 这是 pd.Series
            lower_band = middle_band - (rolling_std * std_dev) # 这是 pd.Series
            # --- [修复结束] ---

            # 2. 检查 Series 是否有效 (确保有足够的数据计算)
            if upper_band.isnull().all() or lower_band.isnull().all():
                self.logger.warning("BBand check: Calculated bands contain only NaNs.")
                return False, ""
            
            # 3. 获取上一根 K 线和当前 K 线的数据
            # 确保索引 -2 和 -1 存在且非 NaN (iloc[-1] 是最新数据, iloc[-2] 是上一根)
            if pd.isna(upper_band.iloc[-2]) or pd.isna(upper_band.iloc[-1]) or \
               pd.isna(lower_band.iloc[-2]) or pd.isna(lower_band.iloc[-1]):
                self.logger.debug("BBand check: Skipping, not enough data for prev/curr comparison (NaN).")
                return False, ""

            close_prev = closes.iloc[-2]
            close_curr = closes.iloc[-1]
            upper_prev = upper_band.iloc[-2]
            upper_curr = upper_band.iloc[-1]
            lower_prev = lower_band.iloc[-2]
            lower_curr = lower_band.iloc[-1]

            # 4. 检查是否上穿上轨 (之前在带内，现在在带外)
            if close_prev <= upper_prev and close_curr > upper_curr:
                return True, "Event: 15m Price Crossed BB_Upper"
                
            # 5. 检查是否下穿下轨 (之前在带内，现在在带外)
            if close_prev >= lower_prev and close_curr < lower_curr:
                return True, "Event: 15m Price Crossed BB_Lower"
                
            return False, ""
        except Exception as e:
            # 增加 exc_info=True 来记录完整的错误堆栈
            self.logger.error(f"Err check BBand breach: {e}", exc_info=True) 
            return False, ""

    async def _update_fear_and_greed_index(self):
        """
        [新增] 异步获取并缓存 Fear & Greed Index。
        使用1小时缓存避免频繁调用 API。
        """
        now = time.time()
        # 1. 检查缓存是否有效
        if now - self.last_fng_fetch_time < self.FNG_CACHE_DURATION_SECONDS:
            self.logger.info(f"Using cached F&G Index: {self.fng_data['value_classification']} ({self.fng_data['value']})")
            return

        self.logger.info("Fetching new Fear & Greed Index data from alternative.me...")
        url = "https://api.alternative.me/fng/?limit=1"
        
        try:
            # 2. 使用 httpx 进行异步请求
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                # 3. 解析和更新数据
                if data and 'data' in data and len(data['data']) > 0:
                    fng_info = data['data'][0]
                    self.fng_data = {
                        "value": int(fng_info.get('value', 50)),
                        "value_classification": fng_info.get('value_classification', "Neutral")
                    }
                    self.last_fng_fetch_time = now
                    self.logger.info(f"Fetched new F&G Index: {self.fng_data['value_classification']} ({self.fng_data['value']})")
                else:
                    self.logger.warning("F&G API response was successful but empty or malformed.")
            else:
                self.logger.error(f"Failed to fetch F&G Index. Status code: {response.status_code}")
        
        except httpx.RequestError as e:
            self.logger.error(f"Error fetching F&G Index (httpx): {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error updating F&G Index: {e}", exc_info=True)
        # 即使失败，也会保留旧的(或默认的)缓存数据
    # --- [V45.16 移除] 不再需要 _should_translate 和 _extract_keywords ---
    # ...
    # --- [移除结束] ---


    async def run_cycle(self):
        """[V45.17] AI 决策主循环。集成新规则。"""
        self.logger.info("="*20 + " Starting AI Cycle " + "="*20)
        self.invocation_count += 1
        if not self.is_live_trading: await self._check_and_execute_hard_stops()

        # 1. 获取数据 & 构建 Prompt (已更新 V45.17, 包含 ADX/BBands)
        market_data, tickers = await self._gather_all_market_data()
        
        # [V45.17 修复] 传入 tickers 以便 portfolio 计算实盘 UPL
        # (这是我们在上一个回复中修复
        portfolio_state = self.portfolio.get_state_for_prompt(tickers)
        
        user_prompt_string = self._build_prompt(market_data, portfolio_state, tickers)

        # 2. 格式化 System Prompt (已更新 V45.17)
        try:
            # [V45.34] 提示词已更新，不再包含 Rule 2
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

        # 5. 执行决策 (已更新 V45.34, 增加 UPDATE_TAKEPROFIT)
        if orders:
            self.logger.info(f"AI proposed {len(orders)} order(s), executing...")
            await self._execute_decisions(orders, market_data)
        else:
            self.logger.info("AI proposed no orders.")

        self.logger.info("="*20 + " AI Cycle Finished " + "="*20 + "\n")



    async def start(self):
        """[V45.34 重构] 启动 AlphaTrader 主循环。
        将所有硬性风控逻辑 (SL, TP, MaxLoss, Dust, Multi-Stage TP) 移至此循环，
        使其每10秒执行一次，独立于AI。
        """
        self.logger.warning(f"🚀 AlphaTrader starting! Mode: {'LIVE' if self.is_live_trading else 'PAPER'}")
        if self.is_live_trading:
            self.logger.warning("!!! LIVE MODE !!! Syncing state on startup...")
            if not hasattr(self, 'client') and hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
            try: 
                await self.portfolio.sync_state(); self.logger.warning("!!! LIVE State Sync Complete !!!")
            except Exception as e_sync: self.logger.critical(f"Initial LIVE state sync failed: {e_sync}", exc_info=True)
        
        # --- [V45.34 定义风控阈值] ---
        # 多阶段止盈阈值
        MULTI_TP_STAGE_1_PERCENT = 0.04  # 4.0%
        MULTI_TP_STAGE_2_PERCENT = 0.10  # 10.0%
        # 最大亏损 (从 config 读取, 默认 -20%)
        MAX_LOSS_PERCENT = getattr(futures_settings, 'MAX_LOSS_CUTOFF_PERCENT', 20.0) / 100.0
        # 粉尘仓位阈值
        DUST_MARGIN_USDT = 1.0
        # --- [定义结束] ---
        
        while True:
            try:
                # 步骤 1: 状态同步 (必须成功)
                try:
                    await self.portfolio.sync_state()
                    self.logger.info("Portfolio state sync successful.")
                except Exception as e_sync:
                    self.logger.critical(f"Main loop sync_state failed: {e_sync}. Skipping AI cycle, will retry...", exc_info=True)
                    await asyncio.sleep(30) # 等待 30 秒后重试同步
                    continue # 跳过本轮循环，直接进入下一次循环尝试 sync_state
                
                # 步骤 2: 获取 Tickers (用于高频风控检查)
                tickers = {}
                try:
                    if not hasattr(self, 'client'): self.client = self.portfolio.client
                    tickers = await self.client.fetch_tickers(self.symbols)
                except Exception as e_tick:
                    self.logger.error(f"Main loop fetch_tickers (for risk check) failed: {e_tick}", exc_info=False)
                
                # 步骤 3: 记录状态 (可选, 可注释掉以减少日志)
                # await self._log_portfolio_status()
                
                # --- [ V45.34 核心: 高频硬性风控检查 (每10秒) ] ---
                if self.is_live_trading and tickers:
                    
                    # 待执行的动作
                    positions_to_close = {} # 使用字典防止重复: {symbol: reason}
                    positions_to_partial_close = [] # [(symbol, size_pct, reason)]
                    
                    # 1. 清理已平仓的计数器
                    open_symbols = set(self.portfolio.position_manager.get_all_open_positions().keys())
                    for symbol in list(self.tp_counters.keys()):
                        if symbol not in open_symbols:
                            self.logger.info(f"Removing TP counter for closed position: {symbol}")
                            del self.tp_counters[symbol]

                    # 2. 迭代所有持仓，应用风控规则
                    try:
                        open_positions = self.portfolio.position_manager.get_all_open_positions()
                        
                        for symbol, state in open_positions.items():
                            price = tickers.get(symbol, {}).get('last')
                            if not price or price <= 0: continue
                            
                            entry = state.get('avg_entry_price')
                            size = state.get('total_size')
                            side = state.get('side')
                            lev = state.get('leverage')
                            margin = state.get('margin') # 从 position_manager 获取
                            
                            if not all([entry, size, side, lev, margin]) or lev <= 0 or entry <= 0 or margin <= 0:
                                self.logger.warning(f"Risk Check: Skipping {symbol}, invalid state data.")
                                continue

                            # 计算利润率 (UPL / Margin)
                            upl = (price - entry) * size if side == 'long' else (entry - price) * size
                            rate = upl / margin # 使用 PM 管理器计算的保证金

                            # --- 风控规则应用 (优先级顺序) ---
                            
                            # 规则 1: 最大亏损 (Max Loss)
                            if rate <= -MAX_LOSS_PERCENT:
                                reason = f"Hard Max Loss ({-MAX_LOSS_PERCENT:.0%})"
                                if symbol not in positions_to_close:
                                    self.logger.warning(f"!!! HARD STOP: MAX LOSS !!! {symbol} | R {rate:.2%} <= {-MAX_LOSS_PERCENT:.0%}")
                                    positions_to_close[symbol] = reason
                                continue # 强制平仓，跳过后续检查

                            # 规则 2: 粉尘仓位 (Dust)
                            if margin < DUST_MARGIN_USDT:
                                reason = f"Dust Close (<{DUST_MARGIN_USDT:.1f}U)"
                                if symbol not in positions_to_close:
                                    self.logger.warning(f"!!! HARD STOP: DUST !!! {symbol} | Margin {margin:.2f} < {DUST_MARGIN_USDT:.1f}U")
                                    positions_to_close[symbol] = reason
                                continue # 强制平仓，跳过后续检查

                            # 规则 3: AI 设置的止损 (AI SL)
                            ai_sl = state.get('ai_suggested_stop_loss')
                            if ai_sl and ai_sl > 0:
                                if (side == 'long' and price <= ai_sl) or (side == 'short' and price >= ai_sl):
                                    reason = f"AI SL Hit ({ai_sl:.4f})"
                                    if symbol not in positions_to_close:
                                        self.logger.warning(f"!!! HARD STOP: AI SL !!! {symbol} | Price {price:.4f} hit SL {ai_sl:.4f}")
                                        positions_to_close[symbol] = reason
                                    continue # 强制平仓，跳过后续检查

                            # 规则 4: AI 设置的止盈 (AI TP)
                            ai_tp = state.get('ai_suggested_take_profit')
                            if ai_tp and ai_tp > 0:
                                if (side == 'long' and price >= ai_tp) or (side == 'short' and price <= ai_tp):
                                    reason = f"AI TP Hit ({ai_tp:.4f})"
                                    if symbol not in positions_to_close:
                                        self.logger.warning(f"!!! HARD STOP: AI TP !!! {symbol} | Price {price:.4f} hit TP {ai_tp:.4f}")
                                        positions_to_close[symbol] = reason
                                    continue # 强制平仓，跳过后续检查

                            # 规则 5: 多阶段止盈 (Multi-Stage TP)
                            # (确保计数器存在)
                            self.tp_counters.setdefault(symbol, {'stage1': 0, 'stage2': 0})
                            counters = self.tp_counters[symbol]

                            if rate < 0:
                                # 收益率为负，重置计数器
                                if counters['stage1'] == 1 or counters['stage2'] == 1:
                                    self.logger.info(f"Resetting TP counters for {symbol} (UPL negative).")
                                    counters['stage1'] = 0
                                    counters['stage2'] = 0
                            
                            elif rate >= MULTI_TP_STAGE_2_PERCENT and counters['stage2'] == 0:
                                # 触发阶段 2 (+10%)
                                self.logger.warning(f"!!! HARD TP (Stage 2) !!! {symbol} | R {rate:.2%} >= {MULTI_TP_STAGE_2_PERCENT:.0%}")
                                positions_to_partial_close.append((symbol, 0.5, f"Hard TP Stage 2 (>{MULTI_TP_STAGE_2_PERCENT:.0%})"))
                                counters['stage2'] = 1 # 标记为已触发
                                # (我们假设阶段1也自动触发)
                                counters['stage1'] = 1 
                            
                            elif rate >= MULTI_TP_STAGE_1_PERCENT and counters['stage1'] == 0:
                                # 触发阶段 1 (+4%)
                                self.logger.warning(f"!!! HARD TP (Stage 1) !!! {symbol} | R {rate:.2%} >= {MULTI_TP_STAGE_1_PERCENT:.0%}")
                                positions_to_partial_close.append((symbol, 0.5, f"Hard TP Stage 1 (>{MULTI_TP_STAGE_1_PERCENT:.0%})"))
                                counters['stage1'] = 1 # 标记为已触发

                    except Exception as e_risk:
                        self.logger.error(f"High-frequency risk check error: {e_risk}", exc_info=True)

                    # 3. 执行风控动作
                    # (我们不需要在这里 re-sync，因为下一次循环会在开头 sync)
                    
                    if positions_to_close:
                         tasks_close = [self.portfolio.live_close(symbol, reason=reason) for symbol, reason in positions_to_close.items()]
                         await asyncio.gather(*tasks_close)
                         self.logger.info(f"Hard Close actions executed for: {list(positions_to_close.keys())}")
                    
                    if positions_to_partial_close:
                        # 确保我们不会部分平仓一个刚刚被全平的仓位
                        final_partial_tasks = []
                        for symbol, size_pct, reason in positions_to_partial_close:
                            if symbol not in positions_to_close:
                                final_partial_tasks.append(self.portfolio.live_partial_close(symbol, size_percent=size_pct, reason=reason))
                        
                        if final_partial_tasks:
                            await asyncio.gather(*final_partial_tasks)
                            self.logger.info("Hard Partial TP actions executed.")
                
                # --- [ V45.34 风控结束 ] ---


                # 步骤 5: 决定是否触发 AI (低频)
                trigger_ai, reason, now = False, "", time.time(); interval = settings.ALPHA_ANALYSIS_INTERVAL_SECONDS;
                if now - self.last_run_time >= interval: trigger_ai, reason = True, "Scheduled"
                
                if not trigger_ai:
                    sym=self.symbols[0]; ohlcv_15m, ohlcv_1h = [], []
                    try: 
                        ohlcv_15m, ohlcv_1h = await asyncio.gather(
                            self.exchange.fetch_ohlcv(sym, '15m', limit=150), 
                            self.exchange.fetch_ohlcv(sym, '1h', limit=20)
                        )
                    except Exception as e_fetch: self.logger.error(f"Event check: Fetch OHLCV fail: {e_fetch}")
                    
                    cooldown = settings.AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES * 60
                    
                    if now - self.last_event_trigger_ai_time > cooldown:
                        event, ev_reason = await self._check_significant_indicator_change(ohlcv_15m) # 1. 15m MACD
                        if not event: event, ev_reason = await self._check_rsi_threshold_breach(ohlcv_15m) # 2. 15m RSI
                        if not event: event, ev_reason = await self._check_bollinger_band_breach(ohlcv_15m) # 3. 15m BBands
                        if not event: event, ev_reason = await self._check_market_volatility_spike(ohlcv_1h) # 4. 1h Volatility
                        if event: trigger_ai, reason = True, ev_reason
                
                # 步骤 6: (安全地) 运行 AI 循环
                if trigger_ai:
                    self.logger.warning(f"🔥 AI triggered! Reason: {reason} (Sync was successful)")
                    if reason != "Scheduled": self.last_event_trigger_ai_time = now
                    await self.run_cycle(); self.last_run_time = now
                
                await asyncio.sleep(10) # 10秒主循环
            except asyncio.CancelledError: self.logger.warning("Task cancelled, shutting down..."); break
            except Exception as e: 
                self.logger.critical(f"Main loop fatal error (outside sync/AI): {e}", exc_info=True); 
                await asyncio.sleep(60)
