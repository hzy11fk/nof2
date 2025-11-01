# 文件: alpha_trader.py (完整优化版 - L2/L3 混合模型 + 高级回调)
# 描述: 
# L3 (LLM): Rule 6 策略, 由 L2 辅助
# L2 (ML): Rule 8 RF模型 + Anomaly模型, 作为“专家顾问”
# L1 (Python): Rule 8 执行, 硬风控

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
import joblib
import os

try:
    import pandas as pd
except ImportError:
    pd = None

class AlphaTrader:
    
    # --- [AI (Rule 6) 专用 PROMPT] ---
    # (已修改 - 整合 Anomaly, ML Proba, 高级回调逻辑)
    SYSTEM_PROMPT_TEMPLATE = """
    You are a **profit-driven, analytical, and disciplined** quantitative trading AI. Your primary goal is to **generate and secure realized profit**. You are not a gambler; you are a calculating strategist.

    **You ONLY execute a trade if you have a high-confidence assessment that the action will lead to profit.** A medium or low-confidence signal means you WAIT.

    Your discipline is demonstrated by strict adherence to the risk management rules below, which are your foundation for sustained profitability.

    **Core Mandates & Rules:**
    1.  **Strategy: Limit Orders Only (CRITICAL):**
        -   To prevent "chasing price," your **only** strategy is to be patient and trade pullbacks (Rule 6.1), mean-reversion (Rule 6.2), or chop-zones (Rule 6.3).
        -   You MUST and ONLY use `LIMIT_BUY` or `LIMIT_SELL`.
        -   All fast market-order trades (Rule 8) are handled by a separate, high-frequency Python algorithm.

    1.5. **Anomaly Veto Rule (CRITICAL):**
        -   Before ANY action, you MUST check the `Anomaly_Score` for the symbol. A low score (e.g., < -0.1) indicates the market is behaving erratically.
        -   **New Trades:** If the score is in the 'High Risk' range (e.g., < -0.1), you MUST ABORT all new `LIMIT_BUY` or `LIMIT_SELL` plans.
        -   **Open Positions:** If the score is 'High Risk', your ONLY priority is to check `Invalidation_Condition` or consider using `CLOSE` to secure profits, as a crash or spike is likely.

    2.  **Rule-Based Position Management:**
        For every open position, you MUST check its `InvalidATION_CONDITION`. If this condition is met, you MUST issue a `CLOSE` order. This is your top priority for existing positions.

    3.  **Active SL/TP Management (CRITICAL):**
        Your main task for existing positions is to actively manage risk.
        -   For *every* open position on *every* analysis cycle, you MUST assess if the existing `ai_suggested_stop_loss` or `ai_suggested_take_profit` targets are still optimal.
        -   The system executes these targets automatically, but your job is to UPDATE them.
        -   If the market structure changes, you MUST issue `UPDATE_STOPLOSS` or `UPDATE_TAKEPROFIT` orders.

    4.  **Risk Management Foundation (CRITICAL):**
        Profit is the goal, but capital preservation is the foundation.
        -   **AI's Task (Strategy):** Your job is to select the *risk parameters* based on your confidence, not to perform the final math. The Python system will perform all final calculations and safety checks.
        -   **AI Must Provide (for LIMIT_BUY, LIMIT_SELL):**
            1.  `"leverage": [e.g., 8]`
            2.  `"risk_percent": [e.g., 0.025]`
        -   **System's Task (Calculation):** The Python system will automatically use your `risk_percent` and `Total Equity` to calculate the `final_desired_margin`, check it against `Available Cash`, and perform all hard checks.
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
        -   **Smarter Stop-Loss (Pullbacks):** For Rule 6.1 (Pullback) trades, your `stop_loss` MUST be placed relative to volatility using the **ATR**.
            -   *Example (Long):* Place `stop_loss` at `[Confluence_Zone_Low] - (1.5 * 1h_atr_14)`.
            -   *Example (Short):* Place `stop_loss` at `[Confluence_Zone_High] + (1.5 * 1h_atr_14)`.
        -   **Profit-Taking Strategy:** You SHOULD consider using multiple take-profit levels (by using `PARTIAL_CLOSE` later).

    6.  **Market State Recognition (Default Strategy):**
        You MUST continuously assess the market regime using the **1hour** and **4hour** timeframes. This is your **Default Strategy**.
        -   **1. Strong Trend (Trending Bullish/Bearish) [OPTIMIZED]:**
            -   **Condition:** 1h or 4h **ADX_14 > 25**.
            -   **Strategy (LIMIT ONLY):** Identify a **'Pullback Confluence Zone'**. This is a small price area where **at least two** key S/R levels overlap, creating a much stronger barrier.
                -   *Zone Example (Long):* The `1h EMA 20` is at $65,100 **AND** the `4h BB_Mid` is at $65,050. The zone is 65,050-65,100.
                -   *Zone Example (Long):* The `15min recent_low` is at $3,500 **AND** the `1h BB_Lower` is at $3,510.
            -   **Timing:** Place the `LIMIT_BUY` (in uptrend) or `LIMIT_SELL` (in downtrend) **only if** the pullback appears exhausted.
                -   *Timing Signal (Long):* Place the `LIMIT_BUY` only if price is in the 'Confluence Zone' **AND** the `15min RSI` or `1h RSI` has dropped to a 'reset' level (e.g., < 40).
                -   *Timing Signal (Short):* Place the `LIMIT_SELL` only if price is in the 'Confluence Zone' **AND** the `15min RSI` or `1h RSI` has risen to a 'reset' level (e.g., > 60).
            -   **[ML Confirmation]:** This is a High-Confidence trade *only if* the `ML_Proba_UP` (for Long) or `ML_Proba_DOWN` (for Short) is also confirming your thesis (e.g., > 0.60).
        -   **2. Ranging (No Trend):**
            -   **Condition:** 1h and 4h **ADX_14 < 20**.
            -   **Strategy (LIMIT ONLY):** In this regime, your **only** strategy is **mean-reversion**. Identify the `BB_Upper` and `BB_Lower` levels. You MUST issue **`LIMIT_SELL` at (or near) the upper band** or **`LIMIT_BUY` at (or near) the lower band**.
            -   **[ML Confirmation]:** This is a High-Confidence trade *only if* the `ML_Proba_DOWN` (for Sell) or `ML_Proba_UP` (for Buy) confirms the reversal (e.g., > 0.60).
        -   **3. Chop (Short-Term Ranging):**
            -   **Condition:** 1h or 4h **ADX_14 is between 20 and 25**.
            -   **Strategy (LIMIT ONLY):** This is a low-conviction market. Shift focus to the **15min timeframe**.
            -   Identify the `15min_bb_upper` and `15min_bb_lower` levels.
            -   You MAY issue **`LIMIT_SELL` at the 15m upper band** or **`LIMIT_BUY` at the 15m lower band**.
            -   **[ML Confirmation]:** This is a low-confidence trade. You MUST use a reduced `risk_percent` (e.g., 0.01 or 0.015) and tighter stops, AND the `ML_Proba` should confirm (e.g., > 0.55).

    7.  **Market Sentiment Filter (Fear & Greed Index):**
        You MUST use the provided `Fear & Greed Index` (from the User Prompt) as a macro filter.
        -   **Extreme Fear (Index < 25):** ...
        -   **Fear (Index 25-45):** ...
        -   **Neutral (Index 45-55):** ...
        -   **Greed (Index 55-75):** ...
        -   **Extreme Greed (Index > 75):** ...

    **Multi-Timeframe Confirmation Requirement (CRITICAL):**
    - You MUST analyze and confirm signals across available timeframes: **5min, 15min, 1hour, and 4hour**.
    - **High-Confidence Signal Definition:** A signal is only high-confidence when it aligns with the **Market State** (Rule 6), shows alignment across **at least 3** timeframes, and is confirmed by the **[ML Confirmation]** check.
    - **Timeframe Hierarchy:** Use longer timeframes (**4h, 1h**) to determine the **Market State** and **Overall Trend**. Use shorter timeframes (**15min, 5min**) for precise entry timing.
    -   **Signal Veto Rule (CRITICAL):**
        -   Even if 4h/1h trend signals (e.g., ADX > 25) are strong, if the 15min timeframe shows a **strong opposing signal** (e.g., a bearish RSI divergence, a 15m MACD Dead Cross, or 15m EMA 20 has crossed below 50), you **MUST ABORT** the trade.
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
           
           Anomaly Check: [MUST check Anomaly_Score. If < -0.1, prioritize CLOSE.]
           
           Invalidation Check: [Check condition vs current data. If met, MUST issue CLOSE.]
           (Note: Python-based Rule 8 trades use their own trailing stop logic and are not managed by you)
           
           Reversal & Profit Save Check:
           - [Is this profitable position (UPL > +1.0%) showing strong signs of reversal?]
           - [IF YES: The risk of giving back profits is high. Decision: MUST issue a CLOSE order to secure profits.]

           Pyramiding Check (Adding to a Winner):
           - [Is UPL Percent > +2.5% AND is the original trend (ADX > 25) still strong?]
           - [AND has price pulled back to a key support (for Long) / resistance (for Short)?]
           
           - [CRITICAL STATE CHECK: You MUST check the 'Pending Limit Orders' list. If a pending order for this [SYMBOL] already exists, you MUST ABORT this check and NOT issue a new order.]
           
           - [IF YES (and no pending order): Consider a `LIMIT_BUY`/`LIMIT_SELL` (Limit) order. This new entry is treated as a separate trade and MUST follow the full Rule 4 (Sizing) / Rule 5 (SL/TP) logic.]
           - [CRITICAL: You MUST NEVER add to a losing position (UPL < 0). Averaging down is forbidden.]
           
           SL/TP Target Update Check:
           - [Are the current ai_suggested_stop_loss/take_profit targets still optimal based on new data?]
           - [IF NOT OPTIMAL: Issue UPDATE_STOPLOSS / UPDATE_TAKEPROFIT.]
           
           Decision: [Hold/Close/Partial Close/Add/Update StopLoss/Update TakeProfit + Reason. NOTE: Anomaly, Invalidation and Reversal checks override all "Hold" decisions.]

        ... [Repeat for each open position] ...

        New Trade Opportunities Analysis (Rule 6 - Limit Orders Only):
        Available Margin for New Trades: [Calculate based on Total Equity and risk rules]
        Correlation Check: [Ensure new trades don't breach Rule 4 Correlation Control hard caps]

        [CRITICAL STATE CHECK: Before analyzing any new symbol, you MUST check the 'Pending Limit Orders' list. If a pending order for a symbol already exists, you MUST SKIP analysis for that symbol to prevent duplicate orders.]
        
        [CRITICAL STATE CHECK (NO HEDGING): Before analyzing any new symbol, you MUST check the 'Open Positions (Live / Filled)' list. If a position for that [SYMBOL] already exists, you MUST SKIP analysis for that symbol. AI's role is to MANAGE existing positions (in the section above) or open new ones, NEVER to hold LONG and SHORT on the same symbol (No Hedging).]

        [Analyze opportunities based ONLY on Rule 6, ensuring they pass Rule 1.5 (Anomaly) and Rule 6 (ML Confirmation)]
        
        [EXAMPLE - RULE 6.1 (OPTIMIZED PULLBACK):]
        BTC Multi-Timeframe Assessment (Market State: Trending Bullish, 4h ADX=28):
        - Anomaly Score: -0.05 (Safe)
        - 4h Trend: Bullish | 1h Momentum: Strong | 15min Setup: Price pulling back.
        - Confluence Zone: Found at 65,050 (4h BB_Mid) - 65,100 (1h EMA 20).
        - Timing: 15m RSI is 38 (Near reset level < 40).
        - ML Confirmation: ML_Proba_UP = 0.68 (Confidence: High)
        - Signal Confluence Score: 5/5 | Final Confidence: High - **PREPARE LIMIT_BUY at 65,100 (Rule 6.1)**

        [EXAMPLE - RULE 6.2 (RANGING VETO):]
        ETH Multi-Timeframe Assessment (Market State: Ranging, 1h ADX=18):
        - Anomaly Score: -0.08 (Safe)
        - 4h Trend: N/A | 1h Setup: Price is *approaching* 1h BB_Upper (@ 3900.0) | 15min RSI: 68.5 (Approaching Overbought)
        - ML Confirmation: ML_Proba_DOWN = 0.45 (Confidence: Low)
        - Signal Confluence Score: 2/4 (ML Confirmation FAILED) | Final Confidence: Low - **ABORT LIMIT_SELL**

        [EXAMPLE - RULE 1.5 (ANOMALY VETO):]
        SOL Multi-Timeframe Assessment (Market State: Trending Bullish, 4h ADX=30):
        - Anomaly Score: -0.21 (High Risk - VETO)
        - 4h Trend: Bullish | 1h Momentum: Strong.
        - ML Confirmation: ML_Proba_UP = 0.70 (Confidence: High)
        - Signal Confluence Score: N/A (Rule 1.5 VETO) | Final Confidence: VETO - **ABORT ALL NEW TRADES due to Anomaly**

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
        self.symbols = settings.FUTURES_SYMBOLS_LIST
        self.portfolio = AlphaPortfolio(exchange, self.symbols)
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
        self.FNG_CACHE_DURATION_SECONDS = 3600

        # --- 加载 Rule 8 ML 模型 (字典) ---
        self.ml_models_rule8 = {}
        self.ml_scalers_rule8 = {}
        self.ml_feature_names_rule8 = [
            '5min_rsi_14', '5min_adx_14', '5min_volume_ratio', '5min_price_change_pct',
            '15min_rsi_14', '15min_adx_14', '15min_volume_ratio', '15min_price_change_pct',
            '1hour_adx_14', '1hour_price_change_pct'
        ] 
        self.logger.info(f"Rule 8 ML 特征列表已定义 (数量: {len(self.ml_feature_names_rule8)})")

        for symbol in self.symbols:
            symbol_safe = symbol.split(':')[0].replace('/', '') 
            model_path = os.path.join('models', f'rf_classifier_rule8_{symbol_safe}.pkl')
            scaler_path = os.path.join('models', f'scaler_rule8_{symbol_safe}.pkl')
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.ml_models_rule8[symbol] = joblib.load(model_path)
                    self.ml_scalers_rule8[symbol] = joblib.load(scaler_path)
                    self.logger.info(f"成功加载 {symbol} 的 Rule 8 ML 模型和 Scaler。")
            except Exception as e:
                self.logger.error(f"加载 {symbol} 的 Rule 8 ML 模型时出错: {e}", exc_info=True)
        self.logger.info(f"--- 总共加载了 {len(self.ml_models_rule8)} 个 Rule 8 ML 模型 ---")

        # --- 加载 Anomaly Detection ML 模型 (字典) ---
        self.ml_anomaly_detectors = {}
        self.ml_anomaly_scalers = {}
        self.ml_feature_names_anomaly = [
            'volatility_5_20_ratio',
            'volume_ratio',
            'price_deviation'
        ]
        self.logger.info(f"Anomaly ML 特征列表已定义 (数量: {len(self.ml_feature_names_anomaly)})")

        for symbol in self.symbols:
            symbol_safe = symbol.split(':')[0].replace('/', '') 
            model_path = os.path.join('models', f'anomaly_detector_{symbol_safe}.pkl')
            scaler_path = os.path.join('models', f'anomaly_scaler_{symbol_safe}.pkl')
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.ml_anomaly_detectors[symbol] = joblib.load(model_path)
                    self.ml_anomaly_scalers[symbol] = joblib.load(scaler_path)
                    self.logger.info(f"成功加载 {symbol} 的 Anomaly Detector 模型和 Scaler。")
            except Exception as e:
                self.logger.error(f"加载 {symbol} 的 Anomaly Detector 模型时出错: {e}", exc_info=True)
        self.logger.info(f"--- 总共加载了 {len(self.ml_anomaly_detectors)} 个 Anomaly ML 模型 ---")
        
        self.tp_counters: Dict[str, Dict[str, int]] = {}


    def _setup_log_handler(self):
        class DequeLogHandler(logging.Handler):
            def __init__(self, deque_instance): super().__init__(); self.deque_instance = deque_instance
            def emit(self, record): self.deque_instance.append(self.format(record))
        handler = DequeLogHandler(self.log_deque); handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'); handler.setFormatter(formatter)
        if not any(isinstance(h, DequeLogHandler) for h in self.logger.handlers):
            self.logger.addHandler(handler); self.logger.propagate = False

    async def _log_portfolio_status(self):
        self.logger.info("--- [Status Update] Portfolio ---")
        equity_val = float(self.portfolio.equity) if self.portfolio.equity is not None else 0.0
        cash_val = float(self.portfolio.cash) if self.portfolio.cash is not None else 0.0
        self.logger.info(f"Total Equity: {equity_val:.2f} USDT, Cash: {cash_val:.2f} USDT")
        initial_capital_for_calc = self.initial_capital; performance_percent = 0.0
        if initial_capital_for_calc > 0: performance_percent = (equity_val / initial_capital_for_calc - 1) * 100
        else: self.logger.warning("Initial capital <= 0, cannot calc performance %.")
        self.logger.info(f"Overall Performance: {performance_percent:.2f}% (Initial: {initial_capital_for_calc:.2f})")

    
    async def _gather_all_market_data(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        self.logger.debug("Gathering multi-TF market data (5m, 15m, 1h, 4h) + Indicators (Manual Calc)...")
        market_indicators_data: Dict[str, Dict[str, Any]] = {}
        fetched_tickers: Dict[str, Any] = {}
        
        CONCURRENT_REQUEST_LIMIT = 10
        semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)

        async def _safe_fetch_ohlcv(symbol, timeframe, limit):
            async with semaphore:
                try:
                    return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                except Exception as e:
                    self.logger.error(f"Safe Fetch OHLCV Error ({symbol} {timeframe}): {e}", exc_info=False)
                    return e

        async def _safe_fetch_ticker(symbol):
            async with semaphore:
                try:
                    return await self.exchange.fetch_ticker(symbol)
                except Exception as e:
                    self.logger.error(f"Safe Fetch Ticker Error ({symbol}): {e}", exc_info=False)
                    return e
        
        try:
            timeframes = ['5m', '15m', '1h', '4h']
            tasks = []
            for symbol in self.symbols:
                for timeframe in timeframes: 
                    tasks.append(_safe_fetch_ohlcv(symbol, timeframe, limit=100))
                tasks.append(_safe_fetch_ticker(symbol))
                
            results = await asyncio.gather(*tasks)
            
            total_timeframes = len(timeframes); tasks_per_symbol = total_timeframes + 1
            
            for i, symbol in enumerate(self.symbols):
                start_index = i * tasks_per_symbol; symbol_ohlcv_results = results[start_index:start_index + total_timeframes]
                ticker_result = results[start_index + total_timeframes]
                
                if not isinstance(ticker_result, Exception) and ticker_result and ticker_result.get('last') is not None:
                    fetched_tickers[symbol] = ticker_result; market_indicators_data[symbol] = {'current_price': ticker_result.get('last')}
                else: 
                    market_indicators_data[symbol] = {'current_price': None}
                    self.logger.warning(f"Failed fetch ticker/price for {symbol} (Result: {ticker_result})")
                
                for j, timeframe in enumerate(timeframes):
                    ohlcv_data = symbol_ohlcv_results[j]
                    
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
                            
                        prefix = f"{timeframe.replace('m', 'min').replace('h', 'hour')}_"

                        # --- 1. 计算 Rule 8 (RF 模型) 特征 ---
                        if len(df) >= 28:
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

                        if len(df) >= 15:
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

                        if len(df) >= 26:
                            try:
                                macd = ta.macd(df['c'], 12, 26, 9)
                                if macd is not None and not macd.empty and 'MACD_12_26_9' in macd.columns:
                                    market_indicators_data[symbol][f'{prefix}macd'] = macd['MACD_12_26_9'].iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
                                    market_indicators_data[symbol][f'{prefix}macd_hist'] = macd['MACDh_12_26_9'].iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"ta.macd calc failed for {symbol} {timeframe}: {e}", exc_info=False)
                                
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
                        
                        
                        # --- 2. 计算 Anomaly (IsolationForest) 特征 ---
                        if timeframe == '15m' and len(df) >= 21: # (20 滚动 + 1 pct_change)
                            try:
                                df['returns'] = df['c'].pct_change()
                                
                                rolling_vol_5 = df['returns'].rolling(5).std()
                                rolling_vol_20 = df['returns'].rolling(20).std()
                                if not rolling_vol_5.empty and not rolling_vol_20.empty and rolling_vol_20.iloc[-1] > 1e-9:
                                    market_indicators_data[symbol]['volatility_5_20_ratio'] = rolling_vol_5.iloc[-1] / rolling_vol_20.iloc[-1]
                                else:
                                    market_indicators_data[symbol]['volatility_5_20_ratio'] = 1.0
                                
                                rolling_vol_mean_20 = df['v'].rolling(20).mean()
                                if not rolling_vol_mean_20.empty and rolling_vol_mean_20.iloc[-1] > 1e-9:
                                    market_indicators_data[symbol]['volume_ratio'] = df['v'].iloc[-1] / rolling_vol_mean_20.iloc[-1]
                                else:
                                    market_indicators_data[symbol]['volume_ratio'] = 1.0
                                    
                                rolling_mean_20 = df['c'].rolling(20).mean()
                                rolling_std_20 = df['c'].rolling(20).std()
                                if not rolling_mean_20.empty and not rolling_std_20.empty and rolling_std_20.iloc[-1] > 1e-9:
                                    market_indicators_data[symbol]['price_deviation'] = (df['c'].iloc[-1] - rolling_mean_20.iloc[-1]) / rolling_std_20.iloc[-1]
                                else:
                                    market_indicators_data[symbol]['price_deviation'] = 0.0

                            except Exception as e:
                                self.logger.warning(f"Anomaly feature calc failed for {symbol} 15m: {e}", exc_info=False)

                    except Exception as e: self.logger.error(f"Error during indicator calc loop for {symbol} {timeframe}: {e}", exc_info=False)
        except Exception as e: self.logger.error(f"Error gathering market data: {e}", exc_info=True)
        return market_indicators_data, fetched_tickers


    def _build_prompt(self, market_data: Dict[str, Dict[str, Any]], portfolio_state: Dict, tickers: Dict) -> str:
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
            
            anomaly_score = d.get('anomaly_score', 0.0)
            anomaly_status = "High Risk" if anomaly_score < -0.1 else "Safe"
            prompt += f"Anomaly_Score: {anomaly_score:.3f} ({anomaly_status})\n"
            prompt += f"ML_Proba_UP (15m): {safe_format(d.get('ml_proba_up'), 2)}\n"
            prompt += f"ML_Proba_DOWN (15m): {safe_format(d.get('ml_proba_down'), 2)}\n"
            
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
        fng_class = self.fng_data.get('value_classification', 'Neutral').title()
        prompt += f"Fear & Greed Index: {fng_val} ({fng_class})\n"
        
        prompt += "\n--- Account Info ---\n"
        prompt += f"Return%: {portfolio_state.get('performance_percent', 'N/A')}\n"
        prompt += f"Total Equity: {portfolio_state.get('account_value_usd', 'N/A')}\n" 
        prompt += f"Available Cash: {portfolio_state.get('cash_usd', 'N/A')}\n" 

        prompt += "Open Positions (Live / Filled):\n"
        prompt += portfolio_state.get('open_positions', "No open positions.")
        prompt += "\n\nPending Limit Orders (AI Rule 6 - Waiting / Unfilled):\n"
        prompt += portfolio_state.get('pending_limit_orders', "No pending limit orders.")
        
        return prompt

    async def _get_ai_decision(self, system_prompt: str, user_prompt: str) -> dict:
        if not self.ai_analyzer: return {}
        return await self.ai_analyzer.get_ai_response(system_prompt, user_prompt)

    async def _execute_decisions(self, decisions: list, market_data: Dict[str, Dict[str, Any]]):
        MIN_MARGIN_USDT = futures_settings.MIN_NOMINAL_VALUE_USDT
        MIN_SIZE_BTC = 0.001 

        for order in decisions:
            try:
                action = order.get('action'); symbol = order.get('symbol')
                if not action or not symbol or symbol not in self.symbols: 
                    self.logger.warning(f"跳过无效指令: {order}"); continue
                
                reason = order.get('reasoning', 'N/A')
                
                current_price = market_data.get(symbol, {}).get('current_price')
                limit_price_from_ai = order.get('limit_price')
                
                if action == "CLOSE":
                    if (not current_price or current_price <= 0) and not self.is_live_trading:
                        self.logger.error(f"模拟盘平仓失败: 无当前价格 {symbol}"); continue
                    if self.is_live_trading: await self.portfolio.live_close(symbol, reason=reason)
                    else: await self.portfolio.paper_close(symbol, current_price, reason=reason)
                
                elif action in ["BUY", "SELL"]:
                    if not self.is_live_trading:
                        self.logger.warning(f"模拟盘：跳过 {action} 市价单 (模拟盘仅支持限价单转换)。"); continue
                    
                    if (not current_price or current_price <= 0):
                        self.logger.error(f"无当前价格 {symbol}，跳过 Market Action: {order}"); continue
                        
                    side = 'long' if action == 'BUY' else 'short'; final_size = 0.0
                    price_to_calc = current_price

                    try:
                        leverage = int(order.get('leverage'))
                        risk_percent = float(order.get('risk_percent'))
                        stop_loss = float(order.get('stop_loss'))
                        take_profit = order.get('take_profit')
                        
                        if risk_percent <= 0 or risk_percent > 0.5: raise ValueError(f"无效的 risk_percent: {risk_percent}")
                        if leverage <= 0 or leverage > 100: raise ValueError(f"无效的 leverage: {leverage}")
                        
                        total_equity = float(self.portfolio.equity)
                        available_cash = float(self.portfolio.cash)
                        if total_equity <= 0: raise ValueError(f"无效账户状态 (Equity <= 0)")
                        
                        calculated_desired_margin = total_equity * risk_percent
                        
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

                        invalidation_condition = order.get('invalidation_condition')
                        await self.portfolio.live_open(
                            symbol, 
                            side, 
                            final_size, 
                            leverage, 
                            reason=reason, 
                            stop_loss=stop_loss, 
                            take_profit=take_profit, 
                            invalidation_condition=invalidation_condition
                        )
                    except (ValueError, TypeError, KeyError) as e: 
                        self.logger.error(f"跳过 {action} (Python 计算/参数错误): {order}. Err: {e}"); continue

                elif action in ["LIMIT_BUY", "LIMIT_SELL"]:
                    if not limit_price_from_ai or float(limit_price_from_ai) <= 0:
                        self.logger.error(f"无效限价 {limit_price_from_ai} {symbol}，跳过 Limit Action: {order}"); continue
                    
                    if not self.is_live_trading:
                        try:
                            price_to_calc = float(limit_price_from_ai)
                            if (action == 'LIMIT_BUY' and current_price <= price_to_calc) or (action == 'LIMIT_SELL' and current_price >= price_to_calc):
                                self.logger.warning(f"模拟盘：价格 {current_price} 有利，转为市价单。")
                                leverage = int(order.get('leverage'))
                                risk_percent = float(order.get('risk_percent'))
                                stop_loss = float(order.get('stop_loss'))
                                take_profit = float(order.get('take_profit'))
                                invalidation_condition = order.get('invalidation_condition')
                                
                                total_equity = float(self.portfolio.equity)
                                calculated_desired_margin = total_equity * risk_percent
                                final_margin = max(calculated_desired_margin, MIN_MARGIN_USDT)
                                final_size = (final_margin * leverage) / current_price
                                if symbol == "BTC/USDT:USDT" and final_size < MIN_SIZE_BTC:
                                    final_size = MIN_SIZE_BTC

                                await self.portfolio.paper_open(symbol, 'long' if action == 'LIMIT_BUY' else 'short', final_size, price=current_price, leverage=leverage, reason=reason, stop_loss=stop_loss, take_profit=take_profit, invalidation_condition=invalidation_condition)
                            else:
                                self.logger.info(f"模拟盘：价格 {current_price} 不利 (vs {price_to_calc})，跳过限价单。")
                        except Exception as e_paper:
                             self.logger.error(f"模拟盘限价单转换失败: {e_paper}, Order: {order}"); continue
                        continue 

                    if self.is_live_trading:
                        side = 'long' if action == 'LIMIT_BUY' else 'short'; final_size = 0.0
                        
                        try:
                            leverage = int(order.get('leverage'))
                            risk_percent = float(order.get('risk_percent'))
                            limit_price = float(limit_price_from_ai)
                            stop_loss = float(order.get('stop_loss'))
                            take_profit = float(order.get('take_profit'))
                            
                            if risk_percent <= 0 or risk_percent > 0.5: raise ValueError(f"无效的 risk_percent: {risk_percent}")
                            if leverage <= 0 or leverage > 100: raise ValueError(f"无效的 leverage: {leverage}")
                            if limit_price <= 0: raise ValueError(f"无效的 limit_price: {limit_price}")

                            total_equity = float(self.portfolio.equity)
                            available_cash = float(self.portfolio.cash)
                            if total_equity <= 0: raise ValueError(f"无效账户状态 (Equity <= 0)")
                            
                            calculated_desired_margin = total_equity * risk_percent
                            
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
                        
                        invalidation_condition = order.get('invalidation_condition')
                        if hasattr(self.portfolio, 'live_open_limit'): 
                            await self.portfolio.live_open_limit(
                                symbol, 
                                side, 
                                final_size, 
                                leverage, 
                                limit_price,
                                reason=reason, 
                                stop_loss=stop_loss, 
                                take_profit=take_profit, 
                                invalidation_condition=invalidation_condition
                            )
                        else:
                             self.logger.error(f"AI 请求 LIMIT_BUY/SELL 但 portfolio 不支持 live_open_limit！")
                
                elif action == "PARTIAL_CLOSE":
                    if (not current_price or current_price <= 0) and not self.is_live_trading:
                        self.logger.error(f"模拟盘部分平仓失败: 无当前价格 {symbol}"); continue
                    size_to_close_percent=None; size_to_close_absolute=None
                    try:
                        sp=order.get('size_percent'); sa=order.get('size_absolute')
                        if sp is not None: size_to_close_percent=float(sp)
                        elif sa is not None: size_to_close_absolute=float(sa)
                        else: raise ValueError("需提供 size_% 或 size_abs")
                        if size_to_close_percent is not None and not (0<size_to_close_percent<1.01):
                            if abs(size_to_close_percent) < 1e-9: raise ValueError("size_% 必须 > 0")
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
            await self.portfolio.paper_close(symbol, price, reason)
        return len(to_close) > 0

    
    async def _check_python_rule_8(self, market_data: Dict[str, Dict[str, Any]]) -> list:
        if not settings.ENABLE_BREAKOUT_MODIFIER:
            return []
            
        orders_to_execute = []
        
        open_positions = self.portfolio.position_manager.get_all_open_positions()
        pending_limits = self.portfolio.pending_limit_orders

        R8_ADX_MAX = 25.0
        R8_VOL_RATIO_5M = 2.5
        R8_VOL_RATIO_15M = 2.0
        ML_CONFIDENCE_THRESHOLD = 0.65
        
        leverage = futures_settings.FUTURES_LEVERAGE
        risk_percent = futures_settings.FUTURES_RISK_PER_TRADE_PERCENT / 100.0

        for symbol in self.symbols:
            if symbol in open_positions:
                continue
            
            if symbol in pending_limits:
                continue

            data = market_data.get(symbol)
            if not data: continue

            try:
                price = data.get('current_price')
                adx_1h = data.get('1hour_adx_14')
                bb_upper_15m = data.get('15min_bb_upper')
                bb_lower_15m = data.get('15min_bb_lower')
                vol_ratio_5m = data.get('5min_volume_ratio')
                vol_ratio_15m = data.get('15min_volume_ratio')
                ema_50_4h = data.get('4hour_ema_50')
                
                if not all([price, adx_1h, bb_upper_15m, bb_lower_15m, vol_ratio_5m, vol_ratio_15m, ema_50_4h]):
                    continue

                ml_proba_up = 0.5 
                ml_proba_down = 0.5
                ml_called = False

                if not (adx_1h < R8_ADX_MAX):
                    continue 

                volume_confirmed = (vol_ratio_5m > R8_VOL_RATIO_5M) or (vol_ratio_15m > R8_VOL_RATIO_15M)
                if not volume_confirmed:
                    continue 

                fng_value = self.fng_data.get('value', 50)
                
                action = None
                
                if price > bb_upper_15m:
                    if price < ema_50_4h: continue
                    if fng_value > 75: continue
                    
                    if not ml_called:
                        ml_pred = await self._get_ml_prediction_rule8(symbol, market_data)
                        ml_proba_up = ml_pred['proba_up']
                        ml_proba_down = ml_pred['proba_down']
                        ml_called = True
                    
                    if ml_proba_up < ML_CONFIDENCE_THRESHOLD:
                        self.logger.debug(f"Rule 8 (Py) BUY Veto (ML Confidence): {symbol} Prob={ml_proba_up:.2f}")
                        continue
                        
                    action = "BUY"

                elif price < bb_lower_15m:
                    if price > ema_50_4h: continue
                    if fng_value < 25: continue
                    
                    if not ml_called:
                        ml_pred = await self._get_ml_prediction_rule8(symbol, market_data)
                        ml_proba_up = ml_pred['proba_up']
                        ml_proba_down = ml_pred['proba_down']
                        ml_called = True

                    if ml_proba_down < ML_CONFIDENCE_THRESHOLD:
                        self.logger.debug(f"Rule 8 (Py) SELL Veto (ML Confidence): {symbol} Prob={ml_proba_down:.2f}")
                        continue

                    action = "SELL"

                if action:
                    self.logger.warning(f"🔥 RULE 8 (Python + ML): {action} Signal for {symbol} (Prob Up: {ml_proba_up:.2f}, Down: {ml_proba_down:.2f})")
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

    def _build_python_order(self, symbol: str, action: str, risk: float, lev: int, price: float, symbol_data: Dict) -> Optional[Dict]:
        stop_loss = 0.0
        
        if futures_settings.USE_ATR_FOR_INITIAL_STOP:
            atr_1h = symbol_data.get('1hour_atr_14')
            if not atr_1h or atr_1h <= 0:
                self.logger.warning(f"Rule 8: Missing 1h ATR for {symbol}. Using 2% static SL.")
                sl_pct = 0.02
                stop_loss = price * (1 - sl_pct) if action == "BUY" else price * (1 + sl_pct)
            else:
                multiplier = futures_settings.INITIAL_STOP_ATR_MULTIPLIER
                stop_loss = price - (atr_1h * multiplier) if action == "BUY" else price + (atr_1h * multiplier)
        else:
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
            "take_profit": None, 
            "stop_loss": stop_loss,
            "invalidation_condition": "Python Rule 8", 
            "reasoning": f"Python Rule 8 ({action}). ATR SL @ {stop_loss:.4f}"
        }


    async def _get_ml_prediction_rule8(self, symbol: str, market_data: Dict) -> Dict:
        model = self.ml_models_rule8.get(symbol)
        scaler = self.ml_scalers_rule8.get(symbol)
        
        if not model or not scaler:
            return {'proba_up': 0.5, 'proba_down': 0.5} 

        symbol_data = market_data.get(symbol)
        if not symbol_data:
            self.logger.warning(f"Rule 8 ML: 无法获取 {symbol} 的 market_data")
            return {'proba_up': 0.5, 'proba_down': 0.5}

        try:
            features_live = {}
            for f in self.ml_feature_names_rule8:
                val = symbol_data.get(f)
                if val is None or pd.isna(val):
                    val = 0.0
                features_live[f] = val

            df_live = pd.DataFrame([features_live], columns=self.ml_feature_names_rule8)
            
            X_scaled_live = scaler.transform(df_live)
            probabilities = model.predict_proba(X_scaled_live)[0]
            class_map = {cls: prob for cls, prob in zip(model.classes_, probabilities)}
            
            return {
                'proba_up': class_map.get(1, 0.0),    
                'proba_down': class_map.get(-1, 0.0) 
            }

        except Exception as e:
            self.logger.error(f"Rule 8 ML 预测失败 {symbol}: {e}", exc_info=False)
            return {'proba_up': 0.5, 'proba_down': 0.5}

    
    async def _get_ml_anomaly_score(self, symbol: str, market_data: Dict) -> float:
        model = self.ml_anomaly_detectors.get(symbol)
        scaler = self.ml_anomaly_scalers.get(symbol)
        
        if not model or not scaler:
            return 0.0 

        symbol_data = market_data.get(symbol)
        if not symbol_data:
            self.logger.warning(f"Anomaly ML: 无法获取 {symbol} 的 market_data")
            return 0.0

        try:
            features_live = {}
            for f in self.ml_feature_names_anomaly:
                val = symbol_data.get(f)
                if val is None or pd.isna(val):
                    val = 0.0
                features_live[f] = val

            df_live = pd.DataFrame([features_live], columns=self.ml_feature_names_anomaly)
            
            X_scaled_live = scaler.transform(df_live)
            
            score = model.decision_function(X_scaled_live)[0]
            
            return float(score)

        except Exception as e:
            self.logger.error(f"Anomaly ML 预测失败 {symbol}: {e}", exc_info=False)
            return 0.0


    async def _check_divergence(self, ohlcv_15m: list) -> Tuple[bool, str]:
        try:
            if len(ohlcv_15m) < 40:
                return False, ""
                
            df = pd.DataFrame(ohlcv_15m, columns=['ts','o','h','l','c','v'])
            df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore')
            
            cols_to_numeric = ['h', 'l', 'c']
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['rsi'] = ta.rsi(df['c'], 14)
            df.dropna(inplace=True)
            
            if len(df) < 40:
                return False, ""

            recent_window = df.iloc[-5:]
            older_window = df.iloc[-30:-5]

            recent_high = recent_window['h'].max()
            recent_high_rsi = recent_window.loc[recent_window['h'].idxmax()]['rsi']
            older_high = older_window['h'].max()
            older_high_rsi = older_window.loc[older_window['h'].idxmax()]['rsi']
            if recent_high > older_high and recent_high_rsi < older_high_rsi:
                if df['h'].iloc[-1] == recent_high or df['h'].iloc[-2] == recent_high:
                    return True, "Event: 15m Bearish RSI Divergence"

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

    async def _check_ema_squeeze(self, ohlcv_15m: list) -> Tuple[bool, str]:
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

    async def _check_rsi_threshold_breach(self, ohlcv_15m: list) -> Tuple[bool, str]:
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

    async def _check_bollinger_band_breach(self, ohlcv_15m: list) -> Tuple[bool, str]:
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
        now = time.time()
        if now - self.last_fng_fetch_time < self.FNG_CACHE_DURATION_SECONDS:
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
        self.logger.info("="*20 + " Starting AI Cycle (Rule 6 - Limit Orders) " + "="*20)
        self.invocation_count += 1
        if not self.is_live_trading: await self._check_and_execute_hard_stops()

        portfolio_state = self.portfolio.get_state_for_prompt(tickers)
        
        user_prompt_string = self._build_prompt(market_data, portfolio_state, tickers)

        try:
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                symbol_list=self.formatted_symbols,
                specific_rules_or_notes=""
            )
        except KeyError as e: self.logger.error(f"Format System Prompt failed: {e}"); return

        self.logger.info("Getting AI decision (Rule 6)...")
        ai_decision = await self._get_ai_decision(system_prompt, user_prompt_string)

        original_chain_of_thought = ai_decision.get("chain_of_thought", "AI No CoT.")
        orders = ai_decision.get("orders", [])
        self.logger.warning("--- AI CoT ---"); self.logger.warning(original_chain_of_thought)

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

        if orders:
            self.logger.info(f"AI proposed {len(orders)} order(s), executing...")
            await self._execute_decisions(orders, market_data)
        else:
            self.logger.info("AI proposed no orders.")

        self.logger.info("="*20 + " AI Cycle Finished " + "="*20 + "\n")

    async def start(self):
        self.logger.warning(f"🚀 AlphaTrader starting! Mode: {'LIVE' if self.is_live_trading else 'PAPER'}")
        if self.is_live_trading:
            self.logger.warning("!!! LIVE MODE !!! Syncing state on startup...")
            if not hasattr(self, 'client') and hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
            try: 
                await self.portfolio.sync_state(); self.logger.warning("!!! LIVE State Sync Complete !!!")
            except Exception as e_sync: self.logger.critical(f"Initial LIVE state sync failed: {e_sync}", exc_info=True)
        
        MULTI_TP_STAGE_1_PERCENT = 0.04
        MULTI_TP_STAGE_2_PERCENT = 0.10
        MAX_LOSS_PERCENT = settings.MAX_LOSS_CUTOFF_PERCENT / 100.0
        DUST_MARGIN_USDT = 1.0
        LIMIT_ORDER_TIMEOUT_MS = settings.AI_LIMIT_ORDER_TIMEOUT_SECONDS * 1000
        
        while True:
            try:
                # 步骤 1: 状态同步
                try:
                    await self.portfolio.sync_state()
                except Exception as e_sync:
                    self.logger.critical(f"Main loop sync_state failed: {e_sync}. Skipping AI cycle, will retry...", exc_info=True)
                    await asyncio.sleep(30) 
                    continue
                
                # 步骤 2: 限价单超时清理
                if self.is_live_trading and self.portfolio.pending_limit_orders:
                    now_ms = time.time() * 1000
                    orders_to_cancel = []
                    try:
                        for symbol, plan in list(self.portfolio.pending_limit_orders.items()):
                            order_id = plan.get('order_id')
                            timestamp = plan.get('timestamp')
                            if not order_id:
                                self.logger.warning(f"Pending order {symbol} 缺少 order_id，正在从本地清理...")
                                await self.portfolio.remove_pending_limit_order(symbol)
                                continue
                            if not timestamp:
                                self.logger.warning(f"!!! ORPHAN TIMEOUT !!! {symbol} (ID: {order_id}) 缺少 timestamp。立即取消...")
                                orders_to_cancel.append((order_id, symbol))
                                await self.portfolio.remove_pending_limit_order(symbol)
                                continue
                            if (now_ms - timestamp) > LIMIT_ORDER_TIMEOUT_MS:
                                self.logger.warning(f"!!! LIMIT ORDER TIMEOUT !!! {symbol} (ID: {order_id}) 已超时 {LIMIT_ORDER_TIMEOUT_MS / 1000}s。正在取消...")
                                orders_to_cancel.append((order_id, symbol))
                                await self.portfolio.remove_pending_limit_order(symbol)
                        
                        if orders_to_cancel:
                            cancel_tasks = [self.client.cancel_order(oid, sym) for oid, sym in orders_to_cancel]
                            await asyncio.gather(*cancel_tasks, return_exceptions=True)
                            self.logger.info(f"成功取消 {len(orders_to_cancel)} 个超时/孤儿订单。")
                    except Exception as e_timeout:
                        self.logger.error(f"限价单超时清理时发生错误: {e_timeout}", exc_info=True)

                # 步骤 3: [高频] 获取所有特征数据
                await self._update_fear_and_greed_index()
                market_data, tickers = await self._gather_all_market_data()

                # 步骤 4: [高频] Python Rule 8 执行 (已集成 ML)
                if settings.ENABLE_BREAKOUT_MODIFIER:
                    if self.is_live_trading:
                        rule8_orders = await self._check_python_rule_8(market_data)
                        if rule8_orders:
                            self.logger.warning(f"🔥 Python Rule 8 (ML) TRIGGERED! Executing {len(rule8_orders)} orders.")
                            await self._execute_decisions(rule8_orders, market_data)
                
                # 步骤 5: [高频] 硬性风控检查
                if self.is_live_trading and tickers:
                    positions_to_close = {} 
                    positions_to_partial_close = []
                    sl_update_tasks = [] 

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

                            inval_cond = state.get('invalidation_condition') or '' 
                            is_rule_8_trade = "Python Rule 8" in inval_cond

                            if is_rule_8_trade:
                                trail_percent = settings.BREAKOUT_TRAIL_STOP_PERCENT
                                current_sl = state.get('ai_suggested_stop_loss', 0.0)
                                new_sl = 0.0
                                
                                if side == 'long': new_sl = price * (1 - trail_percent)
                                else: new_sl = price * (1 + trail_percent)
                                
                                if (side == 'long' and new_sl > current_sl) or \
                                   (side == 'short' and new_sl < current_sl and new_sl > 0 and price < current_sl):
                                    
                                    sl_update_tasks.append(
                                        self.portfolio.update_position_rules(symbol, stop_loss=new_sl, reason="Rule 8 Trail Stop")
                                    )
                                continue 

                            if rate <= -MAX_LOSS_PERCENT:
                                reason = f"Hard Max Loss ({-MAX_LOSS_PERCENT:.0%})"
                                if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                continue 

                            if margin < DUST_MARGIN_USDT:
                                reason = f"Dust Close (<{DUST_MARGIN_USDT:.1f}U)"
                                if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                continue 

                            ai_sl = state.get('ai_suggested_stop_loss')
                            if ai_sl and ai_sl > 0:
                                if (side == 'long' and price <= ai_sl) or (side == 'short' and price >= ai_sl):
                                    reason = f"AI SL Hit ({ai_sl:.4f})"
                                    if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                    continue 

                            ai_tp = state.get('ai_suggested_take_profit')
                            if ai_tp and ai_tp > 0:
                                if (side == 'long' and price >= ai_tp) or (side == 'short' and price <= ai_tp):
                                    reason = f"AI TP Hit ({ai_tp:.4f})"
                                    if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                    continue 

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

                    if sl_update_tasks:
                        await asyncio.gather(*sl_update_tasks, return_exceptions=True)
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
                
                # 步骤 6: [低频] 决定是否触发 AI (Rule 6)
                trigger_ai, reason, now = False, "", time.time()
                interval = settings.ALPHA_ANALYSIS_INTERVAL_SECONDS
                if now - self.last_run_time >= interval: trigger_ai, reason = True, "Scheduled"
                
                if not trigger_ai:
                    cooldown = settings.AI_INDICATOR_TRIGGER_COOLDOWN_MINUTES
                    
                    if now - self.last_event_trigger_ai_time > (cooldown * 60):
                        for sym in self.symbols:
                            ohlcv_15m = []
                            try: 
                                ohlcv_15m = await self.exchange.fetch_ohlcv(sym, '15m', limit=150)
                                if not ohlcv_15m: continue

                                event, ev_reason = await self._check_divergence(ohlcv_15m)
                                if not event: event, ev_reason = await self._check_ema_squeeze(ohlcv_15m)
                                if not event: event, ev_reason = await self._check_rsi_threshold_breach(ohlcv_15m)
                                if not event: event, ev_reason = await self._check_bollinger_band_breach(ohlcv_15m)
                                
                                if event: 
                                    trigger_ai, reason = True, f"{sym}: {ev_reason}"
                                    self.logger.info(f"Advanced trigger found for {sym}: {ev_reason}")
                                    break
                            except Exception as e_fetch:
                                self.logger.error(f"Event check for {sym} fail: {e_fetch}", exc_info=False)
                
                # 步骤 7: (安全地) 运行 AI 循环
                if trigger_ai:
                    self.logger.warning(f"🔥 AI triggered! Reason: {reason} (Sync was successful)")
                    if reason != "Scheduled": self.last_event_trigger_ai_time = now
                    
                    self.logger.debug("Pre-processing ML scores for AI (L3)...")
                    for symbol in self.symbols:
                        if symbol in market_data:
                            # 1. 获取 Anomaly 得分
                            anomaly_score = await self._get_ml_anomaly_score(symbol, market_data)
                            market_data[symbol]['anomaly_score'] = anomaly_score
                            
                            # 2. 获取 Rule 8 (RF) 概率
                            ml_pred = await self._get_ml_prediction_rule8(symbol, market_data)
                            market_data[symbol]['ml_proba_up'] = ml_pred['proba_up']
                            market_data[symbol]['ml_proba_down'] = ml_pred['proba_down']
                    self.logger.debug("ML scores pre-processed.")

                    await self.run_cycle(market_data, tickers); 
                    self.last_run_time = now
                
                await asyncio.sleep(10) # 10秒主循环
            except asyncio.CancelledError: self.logger.warning("Task cancelled, shutting down..."); break
            except Exception as e: 
                self.logger.critical(f"Main loop fatal error (outside sync/AI): {e}", exc_info=True); 
                await asyncio.sleep(60)
