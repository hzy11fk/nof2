# 文件: alpha_trader.py (V-Final-Manager)
# 描述: 
# 1. (已移除) Rule 8 策略。
# 2. (已优化) _gather_all_market_data 获取 1h OI Regime 和 1h Taker Ratio Regime (通过 httpx Bypass)。
# 3. (已优化) SYSTEM_PROMPT_TEMPLATE (V-Manager):
#    - AI 角色为 "Strategist" 和 "Position Manager"。
#    - (修复 1) Rule 1.A (4h EMA 偏好) 和 Rule 2 (ADX) 被重构，4h EMA 仅在趋势市生效。
#    - (修复 2) AI 现在分析亏损仓位并被授权使用 "UPDATE_STOPLOSS"。
#    - (修复 3) AI 现在分析盈利仓位并被授权发出新的 "LIMIT_BUY"/"LIMIT_SELL" 作为加仓。
# 4. (已优化) _validate_ai_trade (修复 1):
#    - 4h EMA Veto 仅在 1h ADX > 20 (趋势市) 时触发。
# 5. (已优化) _execute_decisions (修复 2):
#    - 新增了处理 "UPDATE_STOPLOSS" 指令的逻辑。
# 6. (保留) 所有 V-Ultimate 风控:
#    - 动态风险 (F&G 惩罚 + ATR 调整)。
#    - Stale Plan Veto (最终价格验证)。
#    - V3 阶梯止盈 (1% 启动)。
#    - V4 动态安全网 (ADX 过滤 + 1.0*ATR 缓冲 + 宽限期)。
#    - 过时限价单 (Stale Order) 自动取消逻辑。
#    - 模拟盘 (PaperFix) 兼容。

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
    
    SYSTEM_PROMPT_TEMPLATE = """
    You are an **Expert Market Analyst and Trading Strategist**. Your primary goal is to **identify high-probability trading theses** and **actively manage** the portfolio's risk/reward profile.

    You are the 'Strategist' (the brain), and a separate, 100% reliable Python system is the 'Executor' (the hands).

    **Your Task (The 'Why'):**
    1.  Analyze all open positions for risk control (Losing Position Check) or profit optimization (Pyramiding Check).
    2.  Analyze the overall market to identify new, high-confidence trade setups.
    3.  Provide the **key strategic parameters** for any new orders (Open, Add, or Update SL).
    4.  Explain your reasoning in the `chain_of_thought`.

    **Python's Task (Do NOT do this yourself):**
    * Python will perform **all** final checks and calculations.
    * Python will **VETO** your trade if it fails hard rules (e.g., R:R, Stale Price).
    * Python manages the **Macro Trend Veto (Rule 1.A)** and its **Exceptions (Rule 1.A.E)**.
    * Python will **calculate** all final sizing, leverage, and `risk_percent` based on your `confidence_level` AND asset volatility (ATR).
    * Python will **automatically penalize (reduce risk)** for trades against extreme market sentiment (F&G).
    * Python manages all automated exits (SL/TP, Trailing Stops, Safety Nets) via a high-frequency loop.

    **Focus 100% on the quality of the thesis, not the math.**

    **Core Strategy Mandates (Your Guide):**

    1.  **Strategy: Limit Orders Only (CRITICAL):**
        * Your **only** strategy for opening new positions or adding to winners is to be patient.
        * You MUST and ONLY use `LIMIT_BUY` or `LIMIT_SELL` for new entries.

    1.B. **High-Confidence Mandate (CRITICAL):**
        * You **MUST NOT** submit any new `LIMIT_BUY` or `LIMIT_SELL` order unless your final assessment is **High Confidence**.
        * If your analysis results in 'Medium' or 'Low' confidence (due to weak ML, data conflicts [OI/Taker], or sentiment conflict), you MUST **ABORT** the plan.

    1.A. **Macro Market Bias (CRITICAL)**
        * This rule **ONLY applies to Trend Pullback strategies (Rule 2.1)** when ADX > 20.
        * It MUST NOT be used to filter Ranging/Mean-Reversion strategies (Rule 2.2).
        * **Bull Market Bias:** IF `current_price` is ABOVE `4hour_ema_50`:
            * Your **primary goal** is to find `LIMIT_BUY` opportunities (Rule 2.1).
            * Python will **strongly VETO** most `LIMIT_SELL` plans.
        * **Bear Market Bias:** IF `current_price` is BELOW `4hour_ema_50`:
            * Your **primary goal** is to find `LIMIT_SELL` opportunities (Rule 2.1).
            * Python will **strongly VETO** most `LIMIT_BUY` plans.

    1.A.E. **[CRITICAL EXCEPTION (Reversals)]:**
        * Python's Veto (Rule 1.A) has an exception. You **ARE AUTHORIZED** to propose a counter-trend trade (e.g., a `LIMIT_BUY` in a Bear Market Bias) **ONLY IF** you observe strong, data-driven evidence of a *reversal*.
        * **Reversal Signal Example:** `1h_ema` has crossed above `4h_ema` AND this cross is supported by `1h_volume_ratio` (e.g., > 1.5x).
        * **Your Action:** If you identify this signal, you may propose the trade. Python's Executor will then verify your data (EMA cross, Volume) and make the final decision to Override the Veto or not.

    2.  **Market State Recognition (Default Strategy)**
        You MUST continuously assess the market regime using the **1hour** and **4hour** timeframes.
        * **2.1. Strong Trend (ADX > 20):**
            * **Strategy (LIMIT ONLY):** Identify a **'Pullback Confluence Zone'**.
            * **[Trend Filter]:** You MUST obey `Rule 1.A` (Macro Market Bias) or identify an exception (Rule 1.A.E).
            * **Timing:** Place the `LIMIT_BUY` (in uptrend) or `LIMIT_SELL` (in downtrend) **only if** the pullback appears exhausted (e.g., `15min RSI` < 40 for Long).
            * **[Confirmation]:** This is High-Confidence *only if* `ML_Proba` confirms (e.g., > 0.60) AND `Rule 3` Data Confirmation is met.
        * **2.2. Ranging (ADX < 20):**
            * **Strategy (LIMIT ONLY):** **Mean-reversion**.
            * **[Trend Filter]:** This strategy is **EXEMPT** from `Rule 1.A`.
            * **Action:** Issue `LIMIT_SELL` at `BB_Upper` or `LIMIT_BUY` at `BB_Lower`.
            * **[Confirmation]:** You MUST check the correct `ML_Proba` (e.g., `ML_Proba_DOWN > 0.60` for `LIMIT_SELL`).

    3.  **Data Confirmation (CRITICAL):**
        * You MUST check `OI_Regime_1h` and `Taker_Ratio_1h_Regime`. They are your "Conviction" filter.
        * **Strong Confirmation (High Confidence):**
            * `LIMIT_BUY`: `OI_Regime_1h` is "Rising" **AND** `Taker_Ratio_1h_Regime` is "Buying". (P↑ O↑ T↑)
            * `LIMIT_SELL`: `OI_Regime_1h` is "Rising" **AND** `Taker_Ratio_1h_Regime` is "Selling". (P↓ O↑ T↓)
        * **Major Divergence VETO:**
            * **VETO LONG:** `LIMIT_BUY` is VETOED if `OI_Regime_1h` is "Falling" (P↑ O↓).
            * **VETO SHORT:** `LIMIT_SELL` is VETOED if `OI_Regime_1h` is "Falling" (P↓ O↓).

    4.  **Smarter Stop-Loss:**
        * Your `stop_loss_price` MUST be placed relative to volatility using the **ATR**.
        * *Example (Long):* Place `stop_loss_price` at `[Confluence_Zone_Low] - (2.0 * 1h_atr_14)`.
        * *Example (Short):* Place `stop_loss_price` at `[Confluence_Zone_High] + (2.0 * 1h_atr_14)`.
        
    4.5. **R:R Driven Take Profit (CRITICAL):**
        * Python WILL VETO any new trade with R:R < 1.5. You MUST respect this.
        * **Your Planning Process (for New Trades/Adds):**
            1.  First, determine your `entry_price` and `stop_loss_price` (using Rule 4).
            2.  **For LONG:**
                - `Risk_Distance` = `entry_price - stop_loss_price`
                - `Minimum_Reward_Distance` = `Risk_Distance * 1.5`
                - Set ideal `take_profit_price` = `entry_price + Minimum_Reward_Distance`
            3.  **For SHORT:**
                - `Risk_Distance` = `stop_loss_price - entry_price`
                - `Minimum_Reward_Distance` = `Risk_Distance * 1.5`
                - Set ideal `take_profit_price` = `entry_price - Minimum_Reward_Distance`
        * **CRITICAL VETO CHECK (by AI):**
            * You MUST now check if this calculated `take_profit_price` is **realistic**.
            * **VETO (Long):** If your calculated TP is $4100, but there is a major 4h Resistance level at $4000, your trade is INVALID. You MUST ABORT.
            * **VETO (Short):** If your calculated TP is $3900, but there is a major 4h Support level at $4000, your trade is INVALID. You MUST ABORT.
        * **Conclusion:** Only submit a trade if its 1.5R target is *clear* of any major opposing S/R levels.
        
    5.  **Market Sentiment Filter (Fear & Greed Index):**
        * You must use this filter, but it **must be subservient** to the Macro Bias (Rule 1.A).
        * **Rule 1.A (4h EMA trend) always comes first.**
        
        -   **Extreme Fear (Index < 25):** Market is capitulating.
            -   **IF Macro Bias is Bullish (Rule 1.A):** This is a **HIGH-CONFIDENCE `LIMIT_BUY` signal** (a bull market fear-flush). Aggressively seek entry points.
            -   **IF Macro Bias is Bearish (Rule 1.A):** This is a **BEARISH TREND CONFIRMATION**. This **STRONGLY DISCOURAGES** all `LIMIT_BUY` plans. **DO NOT** propose a `LIMIT_BUY` *unless* you have identified a **Rule 1.A.E. Reversal Exception**.
            -   **Python Penalty:** Python will still penalize (reduce risk) for `LIMIT_SELL` orders opened in this state.

        -   **Fear (Index 25-45):** Market is pessimistic.
            -   **Action:** Favorable for `LIMIT_BUY` plans *only if* Macro Bias is Bullish.

        -   **Neutral (Index 45-55):** No bias.
            -   **Action:** Rely 100% on Rule 1.A (Macro Bias) and Rule 2 (ADX State).

        -   **Greed (Index 55-75):** Market is optimistic.
            -   **Action:** Favorable for `LIMIT_SELL` plans *only if* Macro Bias is Bearish.

        -   **Extreme Greed (Index > 75):** Market is euphoric.
            -   **IF Macro Bias is Bearish (Rule 1.A):** This is a **HIGH-CONFIDENCE `LIMIT_SELL` signal** (a bear market relief-rally). Aggressively seek exit/short points.
            -   **IF Macro Bias is Bullish (Rule 1.A):** This is a **BULLISH TREND CONFIRMATION**. This **STRONGLY DISCOURAGES** all `LIMIT_SELL` plans. **DO NOT** propose a `LIMIT_SELL` *unless* you have identified a **Rule 1.A.E. Reversal Exception**.
            -   **Python Penalty:** Python will still penalize (reduce risk) for `LIMIT_BUY` orders opened in this state.

    **MANDATORY OUTPUT FORMAT:**
    Your entire response must be a single JSON object with two keys: "chain_of_thought" and "orders".

    1.  `"chain_of_thought"` (string): A multi-line string containing your detailed analysis in English. It MUST follow this template precisely:
        ```
        My Current Assessment & Actions (Rule 6 - Strategist)

        Market State Analysis:
        - 1h ADX: [Value] | 4h ADX: [Value]
        - Macro Bias (Rule 1.A): [Bullish (Price > 4h EMA) / Bearish (Price < 4h EMA)]
        - Regime: [Applying Rule 2.1 (Trending) / Applying Rule 2.2 (Ranging)]
        - Market Sentiment: [F&G Index value and its implication, e.g., "Extreme Fear (20). Macro Bias is Bearish, so this is a Bear Trend Confirmation. Strongly discouraging Longs."]

        Portfolio Overview & Active Management:
        (Python handles all automated exits. Your task is to analyze open positions for strategic changes.)

        [Analyze each open position one by one:]

        Analyzing Open Position: LONG BTC/USDT:USDT
        - UPL: [Current Unrealized PNL and Percent (e.g., -$25.00 (-1.2%))]
        - Peak_Profit_Achieved: [Historical high profit percent (e.g., +0.5%)]
        - Macro Bias (Rule 1.A): [Bullish/Bearish]
        - Market State (Rule 2): [Trending/Ranging]
        
        1. Losing Position Check (AI Risk Control):
           - [Is this position losing money significantly (e.g., UPL Percent < -1.0%)?]
           - [AND is the market structure *now* strongly opposing the original thesis? (e.g., for a LONG, has the 15m MACD crossed bearishly?)]
           - [IF YES: The original thesis is likely failing. Decision: Issue an **UPDATE_STOPLOSS** order to a new, tighter technical level (e.g., just below the 15m EMA 50) to cut losses early.]
           
        2. Pyramiding Check (AI Add to Winner):
           - [Is this position profitable AND has profit exceeded 1R (e.g., UPL > (Entry - Original SL))?]
           - [AND is the Macro Bias (Rule 1.A) and Market State (Rule 2) still strongly in favor?]
           - [AND has price pulled back to a new, high-confidence support (for Long) / resistance (for Short)?]
           - [AND is there NO existing pending order for this symbol?]
           - [IF YES: This is a high-confidence opportunity to add. Decision: Prepare a new **LIMIT_BUY** / **LIMIT_SELL** order with its own full SL/TP plan (following Rule 4, 4.5, etc.).]
        
        Decision for BTC: [Hold / Preparing UPDATE_STOPLOSS / Preparing new LIMIT_BUY (Add)]

        ... [Repeat for all open positions] ...

        New Trade Opportunities Analysis (Rule 6 - Limit Orders Only):
        
        [CRITICAL STATE CHECK (NO HEDGING): Before analyzing any new symbol, you MUST check the 'Open Positions' and 'Pending Limit Orders' lists. If a position or pending order for that [SYMBOL] already exists (and you are not Pyramiding), you MUST SKIP analysis for that symbol.]
        
        [EXAMPLE - RULE 2.1 (Trending Pullback):]
        ETH Multi-Timeframe Assessment (Market State: Trending Bullish, 4h ADX=28):
        - Macro Bias (Rule 1.A): Bullish.
        - Thesis: Price is pulling back to a confluence support zone (4h BB_Mid + 1h EMA 20). 15m RSI is low (38).
        - Data Check: `OI_Regime_1h` is "Rising" AND `Taker_Ratio_1h__Regime` is "Buying". (P↑ O↑ T↑).
        - ML Confirmation: ML_Proba_UP = 0.68.
        - Sentiment: F&G is 60 (Greed), which is slightly cautionary, but technicals are strong.
        - SL/TP Plan: Entry=3550, SL=3510 (Risk=40). R:R Check: Min TP=3610. Realism: 4h Res is at 3700 (Clear). Set TP=3690.
        - Final Confidence: High.
        - Plan: PREPARE LIMIT_BUY.

        [EXAMPLE - RULE 1.A.E (Reversal Exception):]
        BTC Multi-Timeframe Assessment (Market State: Trending, 1h ADX=25):
        - Macro Bias (Rule 1.A): Bearish (Price < 4h EMA).
        - Thesis: Market is showing a *potential bullish reversal*. 1h EMA has just crossed above 4h EMA, and 1h Volume is high (1.8x ratio). Price is now pulling back to 15m EMA 50 support.
        - Data Check: `OI_Regime_1h` is "Rising" AND `Taker_Ratio_1h_Regime` is "Buying". (P↑ O↑ T↑).
        - ML Confirmation: ML_Proba_UP = 0.75.
        - Sentiment: F&G is 22 (Extreme Fear), which is a Bearish Confirmation (but this reversal signal overrides it).
        - Final Confidence: High. (This meets the criteria for the Rule 1.A.E. Reversal Exception, which Python will validate).
        - Plan: PREPARE LIMIT_BUY.
        
        [EXAMPLE - RULE 1.B (Medium Confidence Veto):]
        XRP Multi-Timeframe Assessment (Market State: Trending Bearish, 1h ADX=49):
        - Macro Bias (Rule 1.A): Bearish.
        - Thesis: Price below 4h EMA_50, downtrend. 15m RSI=43 (low), potential pullback.
        - Data Check: OI_Regime_1h is "Rising" AND Taker_Ratio_1h_Regime is "Neutral". (Partial Confirmation).
        - ML Confirmation: ML_Proba_DOWN=0.45 (Weak).
        - Final Confidence: Medium. (OI confirms, but Taker/ML is weak).
        - Plan: ABORT. (Per Rule 1.B, Medium Confidence is insufficient to trade).

        In summary, [**Key Instruction: Please provide your final concise decision overview directly here, in Chinese.**Final concise decision overview.] 
        ```

    2.  `"orders"` (list): A list of JSON objects for trades. Empty list `[]` if holding.

    **Order Object Rules (NEW SIMPLIFIED FORMAT):**
    -   **To Open Limit (LONG - Rule 6):** `{{"action": "LIMIT_BUY", "symbol": "...", "thesis": "Rule 2.1 Pullback. Confidence: High. OI/Taker Confirmed.", "entry_price": [CALCULATED_PRICE], "take_profit_price": ..., "stop_loss_price": ..., "confidence_level": "High"}}`
    -   **To Open Limit (SHORT - Rule 6):** `{{"action": "LIMIT_SELL", "symbol": "...", "thesis": "Rule 1.A.E Reversal. Confidence: High. ML Confirmed.", "entry_price": [CALCULATED_PRICE], "take_profit_price": ..., "stop_loss_price": ..., "confidence_level": "High"}}`
    -   **To Update Stop Loss (Risk Control):** `{{"action": "UPDATE_STOPLOSS", "symbol": "...", "new_stop_loss": [CALCULATED_PRICE], "reasoning": "Losing Position Check: Tightening SL."}}`
    -   **Symbol Validity:** `symbol` MUST be one of {symbol_list}.
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
        
        self.httpx_client = httpx.AsyncClient()
        self.logger.info("专用的 HTTPA(Httpx) 客户端已初始化 (用于 F&G 和 Taker Ratio Bypass)。")
        
        self.fng_data: Dict[str, Any] = {"value": 50, "value_classification": "Neutral"}
        self.last_fng_fetch_time: float = 0.0
        self.FNG_CACHE_DURATION_SECONDS = 3600
        
        self.has_oi_history = self.client.has.get('fetchOpenInterestHistory', False)
        if not self.has_oi_history:
             self.logger.warning("Exchange does not support 'fetchOpenInterestHistory'. OI data will be unavailable.")

        self.has_taker_ratio = True 
        self.logger.info("Taker Ratio (TR-BYPASS) 已启用 (使用 httpx)。")

        self.ml_feature_names = [
            '5min_rsi_14', '5min_adx_14', '5min_volume_ratio', '5min_price_change_pct',
            '15min_rsi_14', '15min_adx_14', '15min_volume_ratio', '15min_price_change_pct',
            '1hour_adx_14', '1hour_price_change_pct'
        ] 
        self.logger.info(f"ML 特征列表已定义 (数量: {len(self.ml_feature_names)})")

        self.ml_models_strategic = {}
        self.ml_scalers_strategic = {}
        for symbol in self.symbols:
            symbol_safe = symbol.split(':')[0].replace('/', '') 
            model_path = os.path.join('models', f'rf_classifier_rule6_strategic_{symbol_safe}.pkl')
            scaler_path = os.path.join('models', f'scaler_rule6_strategic_{symbol_safe}.pkl')
            try:
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.ml_models_strategic[symbol] = joblib.load(model_path)
                    self.ml_scalers_strategic[symbol] = joblib.load(scaler_path)
                    self.logger.info(f"成功加载 {symbol} 的 (Strategic) ML 模型。")
            except Exception as e:
                self.logger.error(f"加载 {symbol} 的 (Strategic) ML 模型时出错: {e}", exc_info=True)
        self.logger.info(f"--- 总共加载了 {len(self.ml_models_strategic)} 个 (Strategic) ML 模型 ---")

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
        
        self.peak_profit_tracker: Dict[str, float] = {}
        # --- [新代码] 为“分批止盈”策略添加状态 ---
        # 追踪仓位当前所处的盈利阶段 (0=初始, 1=1-2.5%, 2=2.5-5%, 3=5%+)
        self.partial_tp_stage: Dict[str, int] = {}
        # 追踪您的计数器 (1=可用, 0=已用)
        self.partial_tp_counter: Dict[str, int] = {}
        # --- [新代码结束] ---
# --- [吊灯止损 升级] ---
        # 允许 10 秒循环将 ATR 和 High/Low 数据传递给毫秒级循环
        self.current_indicator_data: Dict[str, Dict[str, float]] = {}
        # --- [升级结束] ---
        self.SAFETY_NET_GRACE_PERIOD_MS = 15 * 60 * 1000 # 15 minutes


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
        self.logger.debug("Gathering multi-TF market data (5m, 15m, 1h, 4h) + Indicators + OI/TR Regimes...")
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
        
        async def _safe_fetch_oi_history(symbol):
            if not self.has_oi_history: return None
            async with semaphore:
                try:
                    return await self.exchange.fetch_open_interest_history(symbol, timeframe='1h', limit=50)
                except Exception as e:
                    self.logger.error(f"Safe Fetch OI History Error ({symbol} {timeframe}): {e}", exc_info=False)
                    return e

        async def _safe_fetch_taker_ratio_httpx(symbol):
            if not self.has_taker_ratio: return None
            
            market_id = "BTCUSDT" 
            try:
                market_id = self.client.market(symbol)['id']
            except Exception as e_market:
                self.logger.info(f"TR-Bypass: 无法从 client.market() 获取 {symbol} 的 market_id: {e_market}")
                return None 
            
            url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
            params = {
                "symbol": market_id,
                "period": "1h",
                "limit": 21
            }
            
            async with semaphore:
                try:
                    resp = await self.httpx_client.get(url, params=params, timeout=10.0)
                    resp.raise_for_status() 
                    return resp.json() 
                except httpx.HTTPStatusError as e_http:
                    self.logger.error(f"Safe Fetch Taker Ratio Error (HTTPX Status {e_http.response.status_code}) for {symbol}: {e_http}")
                except Exception as e:
                    self.logger.error(f"Safe Fetch Taker Ratio Error (HTTPX) for {symbol}: {e}", exc_info=False)
                return None 

        try:
            timeframes = ['5m', '15m', '1h', '4h']
            tasks = []
            for symbol in self.symbols:
                for timeframe in timeframes: 
                    limit = 150 if timeframe == '15m' else 100
                    tasks.append(_safe_fetch_ohlcv(symbol, timeframe, limit=limit))
                tasks.append(_safe_fetch_ticker(symbol))
                tasks.append(_safe_fetch_oi_history(symbol))
                tasks.append(_safe_fetch_taker_ratio_httpx(symbol)) 
                
            results = await asyncio.gather(*tasks)
            
            total_timeframes = len(timeframes)
            tasks_per_symbol = total_timeframes + 1 + 1 + 1 
            
            for i, symbol in enumerate(self.symbols):
                start_index = i * tasks_per_symbol
                symbol_ohlcv_results = results[start_index : start_index + total_timeframes]
                ticker_result = results[start_index + total_timeframes]
                oi_history_result = results[start_index + total_timeframes + 1]
                taker_ratio_result = results[start_index + total_timeframes + 2] 

                if not isinstance(ticker_result, Exception) and ticker_result and ticker_result.get('last') is not None:
                    fetched_tickers[symbol] = ticker_result; market_indicators_data[symbol] = {'current_price': ticker_result.get('last')}
                else: 
                    market_indicators_data[symbol] = {'current_price': None}
                    self.logger.warning(f"Failed fetch ticker/price for {symbol} (Result: {ticker_result})")
                
                market_indicators_data[symbol]['oi_regime_1h'] = "Neutral" 
                if isinstance(oi_history_result, list) and len(oi_history_result) >= 21: 
                    try:
                        oi_values = [float(item['openInterestValue']) for item in oi_history_result]
                        oi_df = pd.DataFrame({'oi': oi_values})
                        oi_ema_20 = ta.ema(oi_df['oi'], 20)
                        
                        if oi_ema_20 is not None and not oi_ema_20.empty:
                            current_oi = oi_df['oi'].iloc[-1]
                            current_oi_ema = oi_ema_20.iloc[-1]
                            
                            if current_oi > current_oi_ema:
                                market_indicators_data[symbol]['oi_regime_1h'] = "Rising"
                            elif current_oi < current_oi_ema:
                                market_indicators_data[symbol]['oi_regime_1h'] = "Falling"
                    except Exception as e_oi:
                        self.logger.warning(f"Error processing 1h OI Regime for {symbol}: {e_oi}")
                
                market_indicators_data[symbol]['taker_ratio_1h_regime'] = "Neutral" 
                if isinstance(taker_ratio_result, list) and len(taker_ratio_result) >= 21: 
                    try:
                        ratio_values = [float(item['buySellRatio']) for item in taker_ratio_result]
                        ratio_df = pd.DataFrame({'ratio': ratio_values})
                        ratio_ema_20 = ta.ema(ratio_df['ratio'], 20)
                        
                        if ratio_ema_20 is not None and not ratio_ema_20.empty:
                            current_ratio = ratio_df['ratio'].iloc[-1]
                            current_ratio_ema = ratio_ema_20.iloc[-1]
                            
                            if current_ratio > current_ratio_ema and current_ratio > 0.52:
                                market_indicators_data[symbol]['taker_ratio_1h_regime'] = "Buying"
                            elif current_ratio < current_ratio_ema and current_ratio < 0.48:
                                market_indicators_data[symbol]['taker_ratio_1h_regime'] = "Selling"
                    except Exception as e_tr:
                        self.logger.warning(f"Error processing 1h Taker Ratio Regime for {symbol}: {e_tr}")


                for j, timeframe in enumerate(timeframes):
                    ohlcv_data = symbol_ohlcv_results[j]
                    
                    if isinstance(ohlcv_data, Exception) or not ohlcv_data: 
                        self.logger.warning(f"Failed fetch {timeframe} for {symbol} (Result: {ohlcv_data})")
                        continue
                    
                    if timeframe == '15m':
                        market_indicators_data[symbol]['ohlcv_15m'] = ohlcv_data

                    try:
                        df = pd.DataFrame(ohlcv_data, columns=['ts', 'o', 'h', 'l', 'c', 'v']); df.rename(columns={'timestamp':'ts', 'open':'o', 'high':'h', 'low':'l', 'close':'c', 'volume':'v'}, inplace=True, errors='ignore'); 
                        
                        cols_to_numeric = ['o', 'h', 'l', 'c', 'v']
                        for col in cols_to_numeric:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df.ffill(inplace=True)
                        df.dropna(inplace=True)
                        
                        df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
                        
                        if df.empty or len(df) < 2:
                            self.logger.warning(f"DataFrame empty or too short for {symbol} {timeframe}.")
                            continue
                            
                        prefix = f"{timeframe.replace('m', 'min').replace('h', 'hour')}_"

                        if len(df) >= 28:
                            try:
                                high = df['h']; low = df['l']; close = df['c']; period = 14
                                adx_data = ta.adx(high, low, close, length=period)
                                atr_data = ta.atr(high, low, close, length=period)
                                if adx_data is not None and not adx_data.empty:
                                    market_indicators_data[symbol][f'{prefix}adx_14'] = adx_data[f'ADX_{period}'].iloc[-1]
                                    if timeframe == '15m':
                                        market_indicators_data[symbol][f'{prefix}adx_14_prev'] = adx_data[f'ADX_{period}'].iloc[-2]
                                if atr_data is not None and not atr_data.empty:
                                    market_indicators_data[symbol][f'{prefix}atr_14'] = atr_data.iloc[-1]
                            except Exception as e:
                                self.logger.warning(f"ta ADX/ATR calc failed for {symbol} {timeframe}: {e}", exc_info=False)

                        if len(df) >= 20:
                            try:
                                period = 20; std_dev = 2.0; closes = df['c']
                                bbands = ta.bbands(closes, length=period, std=std_dev)
                                if bbands is not None and not bbands.empty:
                                    market_indicators_data[symbol][f'{prefix}bb_upper'] = bbands.iloc[:, 2].iloc[-1] 
                                    market_indicators_data[symbol][f'{prefix}bb_middle'] = bbands.iloc[:, 1].iloc[-1] 
                                    market_indicators_data[symbol][f'{prefix}bb_lower'] = bbands.iloc[:, 0].iloc[-1] 
                                    
                                    if timeframe == '15m':
                                        market_indicators_data[symbol][f'{prefix}bb_upper_prev'] = bbands.iloc[:, 2].iloc[-2]
                                        market_indicators_data[symbol][f'{prefix}bb_lower_prev'] = bbands.iloc[:, 0].iloc[-2]
                                        market_indicators_data[symbol][f'{prefix}close_prev'] = closes.iloc[-2]

                            except Exception as e:
                                self.logger.warning(f"ta BBands calc failed for {symbol} {timeframe}: {e}", exc_info=True)

                        if len(df) >= 15:
                            try:
                                rsi = ta.rsi(df['c'], 14)
                                if rsi is not None and not rsi.empty:
                                    market_indicators_data[symbol][f'{prefix}rsi_14'] = rsi.iloc[-1]
                                    if timeframe == '15m':
                                        market_indicators_data[symbol][f'{prefix}rsi_14_prev'] = rsi.iloc[-2]
                            except Exception as e:
                                self.logger.warning(f"ta RSI calc failed for {symbol} {timeframe}: {e}", exc_info=False)

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
                        
                        
                        if timeframe == '15m' and len(df) >= 21: 
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
        def safe_format(value, precision, is_rsi=False, is_pct=False):
            is_na = pd.isna(value) if pd else value is None
            if isinstance(value, (int, float)) and not is_na:
                if is_rsi: return f"{round(value):d}"
                if is_pct: return f"{value:+.2%}"
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
            
            prompt += f"OI_Regime_1h: {d.get('oi_regime_1h', 'N/A')}\n"
            prompt += f"Taker_Ratio_1h_Regime: {d.get('taker_ratio_1h_regime', 'N/A')}\n"

            prompt += f"Peak_Profit_Achieved: {safe_format(d.get('peak_profit_achieved_percent'), 1)}%\n"
            
            timeframes = ['5min', '15min', '1hour', '4hour']
            
            for tf in timeframes:
                prompt += f"\n[{tf.upper()}]\n"
                prompt += f" RSI:{safe_format(d.get(f'{tf}_rsi_14'), 0, is_rsi=True)}|"
                prompt += f" ADX:{safe_format(d.get(f'{tf}_adx_14'), 0, is_rsi=True)}|\n"
                prompt += f" EMA20:{safe_format(d.get(f'{tf}_ema_20'), 3)}|"
                prompt += f"EMA50:{safe_format(d.get(f'{tf}_ema_50'), 3)}|\n"
                prompt += f" BB_Up:{safe_format(d.get(f'{tf}_bb_upper'), 3)}|"   
                prompt += f"BB_Mid:{safe_format(d.get(f'{tf}_bb_middle'), 3)}|" 
                prompt += f"BB_Low:{safe_format(d.get(f'{tf}_bb_lower'), 3)}\n" 
                prompt += f" Hi:{safe_format(d.get(f'{tf}_recent_high'), 2)}|"
                prompt += f"Lo:{safe_format(d.get(f'{tf}_recent_low'), 2)}|\n"
            prompt += "-----\n"
        
        prompt += "\n--- Market Context ---\n"
        fng_val = self.fng_data.get('value', 50)
        fng_class = self.fng_data.get('value_classification', 'Neutral').title()
        prompt += f"Fear & Greed Index: {fng_val} ({fng_class})\n"
        
        prompt += "\n--- Account Info ---\n"
        prompt += f"Return%: {portfolio_state.get('performance_percent', 'N/A')}\n"
        prompt += f"Total Equity: {portfolio_state.get('account_value_usd', 'N/A')}\n" 
        prompt += f"Available Cash: {portfolio_state.get('cash_usd', 'N/A')}\n" 

        prompt += "Open Positions (Live / Filled - Rule 6 Only):\n"
        prompt += portfolio_state.get('open_positions_rule6', "No open positions.")
        prompt += "\n\nPending Limit Orders (AI Rule 6 - Waiting / Unfilled):\n"
        prompt += portfolio_state.get('pending_limit_orders', "No open positions.")
        
        return prompt

    async def _get_ai_decision(self, system_prompt: str, user_prompt: str) -> dict:
        if not self.ai_analyzer: return {}
        return await self.ai_analyzer.get_ai_response(system_prompt, user_prompt)

    def _validate_ai_trade(self, order: Dict, market_data: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Python 执行者：验证 AI 策略的硬规则。
        [V-Ultimate V7 优化 - 整合了 EMA Cross + Volume Confirmation]
        """
        symbol = order.get('symbol')
        action = order.get('action')
        limit_price = order.get('entry_price')
        stop_loss_price = order.get('stop_loss_price')
        take_profit_price = order.get('take_profit_price')
        
        data = market_data.get(symbol)
        if not data:
            return False, "Market data unavailable"

        current_price = data.get('current_price')
        anomaly_score = data.get('anomaly_score', 0.0)
        ema_50_4h = data.get('4hour_ema_50')
        oi_regime = data.get('oi_regime_1h')
        taker_regime = data.get('taker_ratio_1h_regime') 
        
        # --- [V7 优化] ---
        # 1. 定义例外规则的参数
        ema_50_1h = data.get('1hour_ema_50')
        adx_1h = data.get('1hour_adx_14')
        volume_ratio_1h = data.get('1hour_volume_ratio') # 使用 1h 成交量比率作为确认
        
        # 2. 定义确认阈值 (例如：成交量必须是均值的 1.5 倍)
        VOL_CONFIRM_THRESHOLD = 1.5 
        # --- [V7 优化结束] ---

        if not all([current_price, ema_50_4h, oi_regime, taker_regime, limit_price, stop_loss_price, take_profit_price]):
            return False, f"Missing critical data (Price, EMA, OI, Taker, or Order Prices)"

        # 1. [VETO] Anomaly Check (Rule 1.5)
        if anomaly_score < -0.1:
            return False, f"Anomaly Veto (Score: {anomaly_score:.3f})"

        # 2. [VETO] Trend Filter Check (Rule 1.B) - [V7 优化]
        if settings.ENABLE_4H_EMA_FILTER:
            # 仅在趋势市场 (ADX > 20) 时才应用 4h EMA 过滤器
            if adx_1h and adx_1h > 20:
                
                # 检查 Veto 解除所需的数据是否存在
                has_override_data = all([ema_50_1h, ema_50_4h, volume_ratio_1h])

                # --- 检查做多 (LIMIT_BUY) Veto ---
                is_4h_bear_trend = current_price < ema_50_4h
                if action == "LIMIT_BUY" and is_4h_bear_trend:
                    
                    # [V7 例外检查]
                    is_bullish_reversal_state = has_override_data and (ema_50_1h > ema_50_4h)
                    is_volume_confirmed = has_override_data and (volume_ratio_1h > VOL_CONFIRM_THRESHOLD)
                    
                    if is_bullish_reversal_state and is_volume_confirmed:
                        # Veto 解除：允许做多
                        self.logger.warning(f"Trend Filter OVERRIDE (Bullish Reversal): {symbol} 1h EMA > 4h EMA + Vol Confirmed. Allowing LONG.")
                    else:
                        # 没有例外：保持 Veto
                        reason = f"Trend Filter Veto (ADX > 20 & Price < 4h EMA)"
                        if is_bullish_reversal_state and not is_volume_confirmed:
                            reason += f" (Reversal lacked Vol Confirmation: {volume_ratio_1h:.1f}x < {VOL_CONFIRM_THRESHOLD}x)"
                        return False, reason

                # --- 检查做空 (LIMIT_SELL) Veto ---
                is_4h_bull_trend = current_price > ema_50_4h
                if action == "LIMIT_SELL" and is_4h_bull_trend:
                    
                    # [V7 例外检查]
                    is_bearish_reversal_state = has_override_data and (ema_50_1h < ema_50_4h)
                    is_volume_confirmed = has_override_data and (volume_ratio_1h > VOL_CONFIRM_THRESHOLD)
                    
                    if is_bearish_reversal_state and is_volume_confirmed:
                        # Veto 解除：允许做空
                        self.logger.warning(f"Trend Filter OVERRIDE (Bearish Reversal): {symbol} 1h EMA < 4h EMA + Vol Confirmed. Allowing SHORT.")
                    else:
                        # 没有例外：保持 Veto
                        reason = f"Trend Filter Veto (ADX > 20 & Price > 4h EMA)"
                        if is_bearish_reversal_state and not is_volume_confirmed:
                            reason += f" (Reversal lacked Vol Confirmation: {volume_ratio_1h:.1f}x < {VOL_CONFIRM_THRESHOLD}x)"
                        return False, reason
            
            else:
                self.logger.info(f"4H EMA Filter: Skipped (Ranging Market ADX {adx_1h:.1f} <= 20).")
        else:
            self.logger.info(f"4H EMA Filter is DISABLED. Skipping Trend Veto for {symbol}.")
            
        # 3. [VETO] OI Matrix Check (Rule 3)
        if action == "LIMIT_BUY" and current_price > ema_50_4h and oi_regime == "Falling":
             taker_confirms_veto = (taker_regime == "Selling")
             log_msg = f"OI Matrix Veto (P↑O↓ - Short Squeeze)"
             if taker_confirms_veto: 
                 log_msg += " [Taker Confirmed]"
             return False, log_msg
             
        if action == "LIMIT_SELL" and current_price < ema_50_4h and oi_regime == "Falling":
             return False, f"OI Matrix Veto (P↓O↓ - Long Squeeze)"

        # 4. [VETO] R:R Check (Python 的最后防线, 即使 AI Rule 4.5 失败)
        try:
            if action == "LIMIT_BUY":
                risk = limit_price - stop_loss_price
                reward = take_profit_price - limit_price
            else: # LIMIT_SELL
                risk = stop_loss_price - limit_price
                reward = limit_price - take_profit_price
            
            if risk <= 1e-9:
                return False, "Risk is zero or negative (SL too close)"
            
            rr = reward / risk
            MIN_RR = 1.5
            if rr < MIN_RR:
                return False, f"R:R Veto (RR: {rr:.2f} < {MIN_RR})"
        except Exception as e_rr:
            return False, f"R:R Calc Error: {e_rr}"

        return True, "Validation Passed"

    async def _execute_decisions(self, decisions: list, market_data: Dict[str, Dict[str, Any]]):
        """
        Python 执行者：
        1. 验证 AI 策略 (Veto 规则)
        2. 计算动态 Risk (含 F&G 惩罚 和 ATR 调整)
        3. 添加最终价格验证 (Stale Plan Veto)
        4. 添加模拟盘 (Paper Trading) 逻辑
        5. (修复 2) 新增 "UPDATE_STOPLOSS" 动作
        [V-FIX] 验证逻辑 (validate_ai_trade 和 Stale Plan Veto)
        已移至循环顶部，确保在实盘和模拟盘中都能执行。
        """
        MIN_MARGIN_USDT = futures_settings.MIN_NOMINAL_VALUE_USDT
        MIN_SIZE_BTC = 0.001 

        for order in decisions:
            try:
                action = order.get('action')
                symbol = order.get('symbol')
                
                if not action or not symbol or symbol not in self.symbols:
                    self.logger.warning(f"跳过无效或不支持的指令: {order}"); continue

                # --- [修复 2] 处理 UPDATE_STOPLOSS ---
                if action == "UPDATE_STOPLOSS":
                    new_stop_loss = 0.0
                    reason = order.get('reasoning', 'AI SL Update')
                    try:
                        nsl = order.get('new_stop_loss')
                        if nsl is None: raise ValueError("缺少 new_stop_loss")
                        new_stop_loss = float(nsl)
                        if new_stop_loss <= 0: raise ValueError("无效止损价")
                    except (ValueError, TypeError, KeyError) as e: 
                        self.logger.error(f"跳过 UPDATE_STOPLOSS 参数错误: {order}. Err: {e}"); continue
                    
                    self.logger.warning(f"AI 请求更新止损 {symbol}: {new_stop_loss:.4f}. 原因: {reason}")
                    await self.portfolio.update_position_rules(symbol, stop_loss=new_stop_loss, reason=reason)
                    continue # 处理完此订单，继续下一个
                # --- [修复 2 结束] ---

                if action not in ["LIMIT_BUY", "LIMIT_SELL"]:
                    self.logger.warning(f"跳过 AI 未知指令 (非 SL): {action}"); continue

                # --- [新位置: 验证块] ---
                # 1. 验证 AI 策略 (Veto 规则) - (适用于实盘和模拟盘)
                is_valid, reason = self._validate_ai_trade(order, market_data)
                if not is_valid:
                    self.logger.warning(f"!!! AI 策略 VETO !!! {symbol} | Action: {action} | 原因: {reason}")
                    continue
                
                self.logger.info(f"AI 策略 (Python 验证通过): {symbol} | Action: {action} | 原因: {reason}")

                # 2. 最终价格验证 (Stale Plan Veto) - (适用于实盘和模拟盘)
                limit_price = float(order.get('entry_price'))
                fresh_price = 0.0 
                is_immediate_fill = False 
                
                try:
                    fresh_ticker = await self.client.fetch_ticker(symbol)
                    fresh_price = fresh_ticker.get('last')
                    if not fresh_price or fresh_price <= 0:
                        raise ValueError(f"无法获取 {symbol} 的最新价格")

                    deviation_threshold = settings.AI_LIMIT_ORDER_DEVIATION_PERCENT / 100.0 
                    SLIPPAGE_ALLOWANCE = 0.001 # 0.1% 滑点
                    
                    if action == "LIMIT_BUY" and limit_price > (fresh_price * (1 + SLIPPAGE_ALLOWANCE)):
                        if self.is_live_trading: 
                            self.logger.error(f"STALE PLAN VETO (Immediate Fill): {symbol} 挂单价 {limit_price} > 现价 {fresh_price}。取消。")
                            continue
                        else: 
                            is_immediate_fill = True
                            
                    if action == "LIMIT_SELL" and limit_price < (fresh_price * (1 - SLIPPAGE_ALLOWANCE)):
                        if self.is_live_trading: 
                            self.logger.error(f"STALE PLAN VETO (Immediate Fill): {symbol} 挂单价 {limit_price} < 现价 {fresh_price}。取消。")
                            continue
                        else: 
                            is_immediate_fill = True
                        
                    deviation_pct = abs(fresh_price - limit_price) / limit_price
                    if deviation_pct > deviation_threshold:
                        self.logger.error(f"STALE PLAN VETO (Deviation): {symbol} 挂单价 {limit_price} 与 *最新*现价 {fresh_price} 偏离 ({deviation_pct:.2%}) > 阈值。取消。")
                        continue

                except Exception as e_fresh_price:
                    self.logger.error(f"STALE PLAN VETO (Fetch Error): 无法在下单前获取最新价格: {e_fresh_price}。取消订单。")
                    continue
                # --- [验证块结束] ---


                # 3. 计算 Risk/Leverage/Size (如果验证通过)
                stop_loss = float(order.get('stop_loss_price'))
                take_profit = float(order.get('take_profit_price'))
                confidence = order.get('confidence_level', 'Medium')
                ai_thesis = order.get('thesis', 'N/A')
                
                risk_percent_base = 0.015 # 默认 Medium
                if confidence == "High":
                    risk_percent_base = 0.025 # High
                
                risk_percent_final = risk_percent_base
                fng_value = self.fng_data.get('value', 50)
                SENTIMENT_PENALTY_FACTOR = 0.5 # 风险减半

                if action == "LIMIT_BUY" and fng_value > 75:
                    risk_percent_final = risk_percent_base * SENTIMENT_PENALTY_FACTOR
                    self.logger.warning(f"情绪惩罚: 'Extreme Greed' ({fng_value})，LIMIT_BUY 风险从 {risk_percent_base} 调降至 {risk_percent_final}")
                
                elif action == "LIMIT_SELL" and fng_value < 25:
                    risk_percent_final = risk_percent_base * SENTIMENT_PENALTY_FACTOR
                    self.logger.warning(f"情绪惩罚: 'Extreme Fear' ({fng_value})，LIMIT_SELL 风险从 {risk_percent_base} 调降至 {risk_percent_final}")
                
                try:
                    data = market_data.get(symbol)
                    atr_1h = data.get('1hour_atr_14')
                    price = data.get('current_price') 
                    
                    if atr_1h and price and price > 0:
                        atr_pct = (atr_1h / price) 
                        if atr_pct > 0.015: 
                            volatility_factor = 0.75
                            risk_percent_final = risk_percent_final * volatility_factor 
                            self.logger.info(f"动态风险调整: {symbol} 波动率高 ({atr_pct:.2%})，风险进一步调降至 {risk_percent_final}")
                        
                except Exception as e_vol:
                    self.logger.error(f"动态风险计算失败: {e_vol}，将使用(可能已被情绪惩罚的)风险 {risk_percent_final}")
                
                leverage = int(futures_settings.FUTURES_LEVERAGE) 
                
                final_size = 0.0
                try:
                    if self.is_live_trading:
                        total_equity = float(self.portfolio.equity)
                        available_cash = float(self.portfolio.cash)
                    else:
                        total_equity = float(self.portfolio.paper_equity)
                        available_cash = float(self.portfolio.paper_cash)

                    if total_equity <= 0: raise ValueError(f"无效账户状态 (Equity <= 0)")
                    
                    calculated_desired_margin = total_equity * risk_percent_final
                    
                    max_margin_cap = total_equity * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
                    if calculated_desired_margin > max_margin_cap:
                        self.logger.warning(f"!!! {action} Margin Capped !!! AI 期望保证金 {calculated_desired_margin:.2f} > 最大 {max_margin_cap:.2f}")
                        calculated_desired_margin = max_margin_cap

                    if self.is_live_trading and (calculated_desired_margin > available_cash):
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
                            if self.is_live_trading and (recalculated_margin > available_cash):
                                self.logger.error(f"!!! {action} Aborted (Cash Insufficient for Min BTC Size) !!! 最小 BTC size 需要 {recalculated_margin:.2f} 保证金 > 可用 {available_cash:.2f}")
                                continue
                    
                    if final_size <= 0: raise ValueError("最终计算 size 为 0")

                except (ValueError, TypeError, KeyError) as e: 
                    self.logger.error(f"跳过 {action} (Python 计算/参数错误): {order}. Err: {e}"); continue
                

                # 4. 执行 (实盘或模拟盘)
                if self.is_live_trading:
                    await self.portfolio.live_open_limit(
                        symbol, 
                        'long' if action == 'LIMIT_BUY' else 'short', 
                        final_size, 
                        leverage, 
                        limit_price,
                        reason=ai_thesis, 
                        stop_loss=stop_loss, 
                        take_profit=take_profit, 
                        invalidation_condition=f"AI V2.3 ({confidence})"
                    )
                
                else:
                    await self.portfolio.paper_open_limit(
                        symbol, 
                        'long' if action == 'LIMIT_BUY' else 'short', 
                        final_size, 
                        leverage, 
                        limit_price,
                        reason=ai_thesis, 
                        stop_loss=stop_loss, 
                        take_profit=take_profit, 
                        invalidation_condition=f"AI V2.3 ({confidence})"
                    )
                
            except Exception as e: 
                self.logger.error(f"处理 AI 指令时意外错误: {order}. Err: {e}", exc_info=True)

    async def _check_and_execute_hard_stops(self):
        if self.is_live_trading: return False
        self.logger.info("Checking hard TP/SL (Paper)..."); to_close = []; tickers = {}
        try: tickers = await self.exchange.fetch_tickers(self.symbols)
        except Exception as e: self.logger.error(f"Hard stop failed: Fetch Tickers err: {e}"); return False
        
        # [PaperFix] 模拟盘检查 hard stops
        for symbol, pos in list(self.paper_positions.items()):
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

    
    async def _get_ml_prediction(self, symbol: str, market_data: Dict) -> Dict:
        model = self.ml_models_strategic.get(symbol)
        scaler = self.ml_scalers_strategic.get(symbol)
        
        if not model or not scaler:
            return {'proba_up': 0.5, 'proba_down': 0.5} 

        symbol_data = market_data.get(symbol)
        if not symbol_data:
            self.logger.warning(f"ML: 无法获取 {symbol} 的 market_data")
            return {'proba_up': 0.5, 'proba_down': 0.5}

        try:
            features_live = {}
            for f in self.ml_feature_names: 
                val = symbol_data.get(f)
                if val is None or pd.isna(val):
                    val = 0.0
                features_live[f] = val

            df_live = pd.DataFrame([features_live], columns=self.ml_feature_names)
            
            X_scaled_live = scaler.transform(df_live)
            probabilities = model.predict_proba(X_scaled_live)[0]
            class_map = {cls: prob for cls, prob in zip(model.classes_, probabilities)}
            
            return {
                'proba_up': class_map.get(1, 0.0),    
                'proba_down': class_map.get(-1, 0.0) 
            }

        except Exception as e:
            self.logger.error(f"ML (Strategic) 预测失败 {symbol}: {e}", exc_info=False)
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
            response = await self.httpx_client.get(url, timeout=10.0)
            
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

        portfolio_state = self.portfolio.get_state_for_prompt(tickers, filter_rule8=True)
        
        user_prompt_string = self._build_prompt(market_data, portfolio_state, tickers)

        try:
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                symbol_list=self.formatted_symbols
            )
        except KeyError as e: self.logger.error(f"Format System Prompt failed: {e}"); return

        self.logger.info("Getting AI decision (Rule 6)...")
        ai_decision = await self._get_ai_decision(system_prompt, user_prompt_string)

        original_chain_of_thought = ai_decision.get("chain_of_thought", "AI No CoT.")
        orders = ai_decision.get("orders", [])
        self.logger.warning("--- AI CoT ---"); self.logger.warning(original_chain_of_thought)

        summary_for_ui = "AI did not provide a summary." 
        summary_keyword_pattern = re.compile(r"In summary,", re.IGNORECASE)
        parts = summary_keyword_pattern.split(original_chain_of_thought, maxsplit=1)

        if len(parts) > 1:
            extracted_summary = parts[1].strip().lstrip(' :').rstrip('`')
            if extracted_summary:
                summary_for_ui = extracted_summary 
                self.logger.info(f"Extracted English summary: '{summary_for_ui[:50]}...'")
            else:
                 summary_for_ui = "AI summary was empty."
        else:
            self.logger.warning("AI CoT did not find 'In summary,' keyword.")
        
        self.last_strategy_summary = summary_for_ui

        if orders:
            self.logger.info(f"AI proposed {len(orders)} order(s), executing...")
            await self._execute_decisions(orders, market_data)
        else:
            self.logger.info("AI proposed no orders.")

        self.logger.info("="*20 + " AI Cycle Finished " + "="*20 + "\n")


    async def websocket_risk_loop(self):
        """
        [架构升级 - WebSocket + 统一止损 V5 (Bug 修复)]
        此版本修复了一个致命缺陷 (Bug):
        最终止损检查 [I] 错误地使用了 "potential_sl" (候选 SL)，
        而不是 "active_sl" (只紧不松原则后的生效 SL)。
        
        新逻辑:
        1. [G] 计算 `potential_sl` (5 个候选者中的最优者)。
        2. [H] 决定 `active_sl` (比较 `potential_sl` 和 `current_sl`)。
        3. [I] 检查 `price` 是否突破了 `active_sl`。
        """
        self.logger.warning("🚀 实时风控循环 (WebSocket + 统一止损 V5) 已启动...")
        
        self.partial_tp_stage.clear()
        self.partial_tp_counter.clear()
        self.peak_profit_tracker.clear()
        self.logger.info("WebSocket Loop: 状态追踪器已重置。")
        
        CHANDELIER_MULTIPLIER = 2.5 # [新] 吊灯止损的 ATR 倍率 (标准值是 3.0)
        ATR_TRAIL_MULTIPLIER = 1.5 # [旧] Rule 6 的 ATR 倍率

        while True:
            try:
                open_positions = self.portfolio.position_manager.get_all_open_positions()
                open_symbols = set(open_positions.keys())
                
                for symbol in list(self.peak_profit_tracker.keys()):
                    if symbol not in open_symbols:
                        self.logger.info(f"WS Risk Loop: Removing trackers for closed position: {symbol}")
                        del self.peak_profit_tracker[symbol]
                        self.partial_tp_stage.pop(symbol, None)
                        self.partial_tp_counter.pop(symbol, None)

                if not open_symbols:
                    await asyncio.sleep(2) 
                    continue

                tickers = await self.client.watch_tickers(list(open_symbols)) 

                positions_to_close = {} 
                sl_update_tasks = [] 
                tasks_partial_close = [] 

                for symbol, ticker_data in tickers.items():
                    
                    state = open_positions.get(symbol)
                    if not state: continue
                        
                    price = ticker_data.get('last')
                    if not price or price <= 0: continue
                    
                    # ----------------------------------------------------
                    # --- [统一风控逻辑 V5] ---
                    # ----------------------------------------------------
                    try:
                        # [A] 获取所有状态
                        entry = state.get('avg_entry_price')
                        size = state.get('total_size')
                        side = state.get('side')
                        lev = state.get('leverage')
                        margin = state.get('margin')
                        initial_sl = state.get('ai_initial_stop_loss', 0.0)
                        
                        # [B] 计算含手续费的保本价格
                        fee_rate = 0.001 # 0.1% (0.001)
                        breakeven_price = 0.0
                        if entry and entry > 0 and side:
                            if side == 'long':
                                breakeven_price = entry * (1 + fee_rate)
                            elif side == 'short':
                                breakeven_price = entry * (1 - fee_rate)

                        # [C] 验证数据
                        if not all([entry, size, side, lev, margin, initial_sl, breakeven_price]) or lev <= 0 or entry <= 0 or margin <= 0 or initial_sl <= 0 or size <= 0 or breakeven_price <= 0:
                            self.logger.debug(f"WS Risk Loop: Skipping {symbol}, invalid state data.")
                            continue

                        upl = (price - entry) * size if side == 'long' else (entry - price) * size
                        rate = upl / margin if margin > 0 else 0.0 # 当前盈利率
                        is_profitable = rate > 0

                        # [D] 硬性风控检查
                        MAX_LOSS_PERCENT = settings.MAX_LOSS_CUTOFF_PERCENT / 100.0
                        if rate <= -MAX_LOSS_PERCENT:
                            reason = f"Hard Max Loss ({-MAX_LOSS_PERCENT:.0%})"
                            if symbol not in positions_to_close: positions_to_close[symbol] = reason
                            continue 
                        
                        DUST_MARGIN_USDT = 1.0
                        if margin < DUST_MARGIN_USDT:
                            reason = f"Dust Close (<{DUST_MARGIN_USDT:.1f}U)"
                            if symbol not in positions_to_close: positions_to_close[symbol] = reason
                            continue 

                        # [E] 峰值利润追踪
                        current_peak_rate = self.peak_profit_tracker.get(symbol, 0.0)
                        if rate > current_peak_rate:
                            self.peak_profit_tracker[symbol] = rate
                            current_peak_rate = rate 

                        # [F] 检查宽限期 (Grace Period)
                        is_in_grace_period = False
                        try:
                            entries_list = state.get('entries', []) 
                            if not entries_list: continue 
                            last_entry_timestamp = entries_list[-1].get('timestamp', 0) 
                            time_since_entry = (time.time() * 1000) - last_entry_timestamp
                            if time_since_entry < self.SAFETY_NET_GRACE_PERIOD_MS:
                                is_in_grace_period = True
                        except Exception as e_ts:
                            self.logger.warning(f"WS Loop: 无法获取 {symbol} 的 entry timestamp: {e_ts}。将继续检查。")

                        # =======================================================
                        # [G] 统一止损计算 (计算“候选”最优者)
                        # =======================================================
                        
                        # 候选者 1: 初始 SL
                        potential_sl = initial_sl
                        
                        # 候选者 2: 1R 保本 (B/E+Fee)
                        if abs(initial_sl - entry) > 1e-9: 
                            initial_risk_distance = abs(entry - initial_sl)
                            current_upl_distance = 0.0
                            if side == 'long' and price > entry:
                                current_upl_distance = price - entry
                            elif side == 'short' and price < entry:
                                current_upl_distance = entry - price
                            
                            if current_upl_distance >= initial_risk_distance:
                                if side == 'long':
                                    potential_sl = max(potential_sl, breakeven_price)
                                elif side == 'short':
                                    potential_sl = min(potential_sl, breakeven_price)

                        # 候选者 3: V3 阶梯止盈 (按峰值锁定利润)
                        target_profit_rate_v3 = 0.0
                        if current_peak_rate >= 0.06: # > 6% 
                            target_profit_rate_v3 = current_peak_rate * 0.85 # 锁定 85%
                        elif current_peak_rate >= 0.04: # 4% - 6%
                            target_profit_rate_v3 = current_peak_rate * 0.75 # 锁定 75%
                        elif current_peak_rate >= 0.02: # 2% - 4%
                            target_profit_rate_v3 = current_peak_rate * 0.65 # 锁定 65%
                        
                        if target_profit_rate_v3 > 0.0:
                            target_upl_v3 = target_profit_rate_v3 * margin
                            new_graded_sl_price = 0.0
                            if side == 'long':
                                new_graded_sl_price = (target_upl_v3 / size) + entry
                                potential_sl = max(potential_sl, new_graded_sl_price)
                            else: # short
                                new_graded_sl_price = entry - (target_upl_v3 / size)
                                potential_sl = min(potential_sl, new_graded_sl_price)

                        # --- [获取 ATR 和 High/Low 数据] ---
                        indicator_data = self.current_indicator_data.get(symbol)
                        
                        if is_profitable and not is_in_grace_period and indicator_data:
                            atr_15m = indicator_data.get('atr', 0.0)
                            
                            if atr_15m > 0:
                                # 候选者 4: ATR 追踪止损 (Rule 6)
                                new_atr_sl = 0.0
                                if side == 'long':
                                    new_atr_sl = price - (ATR_TRAIL_MULTIPLIER * atr_15m)
                                    potential_sl = max(potential_sl, new_atr_sl)
                                elif side == 'short':
                                    new_atr_sl = price + (ATR_TRAIL_MULTIPLIER * atr_15m)
                                    potential_sl = min(potential_sl, new_atr_sl)
                                
                                # 候选者 5: [新] 吊灯追踪止损 (Chandelier Exit)
                                new_chandelier_sl = 0.0
                                if side == 'long':
                                    high_15m_20p = indicator_data.get('high', price) # 回退到 price
                                    new_chandelier_sl = high_15m_20p - (CHANDELIER_MULTIPLIER * atr_15m)
                                    potential_sl = max(potential_sl, new_chandelier_sl)
                                elif side == 'short':
                                    low_15m_20p = indicator_data.get('low', price) # 回退到 price
                                    new_chandelier_sl = low_15m_20p + (CHANDELIER_MULTIPLIER * atr_15m)
                                    potential_sl = min(potential_sl, new_chandelier_sl)
                        
                        # =======================================================
                        # [H] 最终止损决策 ("只紧不松" 逻辑)
                        # =======================================================
                        
                        # [BUG 修复] 'active_sl' 必须是 "当前交易所生效的 SL"
                        current_sl_on_exchange = state.get('ai_suggested_stop_loss', 0.0)
                        active_sl = current_sl_on_exchange # 默认使用当前 SL

                        if (side == 'long' and potential_sl > current_sl_on_exchange) or \
                           (side == 'short' and potential_sl < current_sl_on_exchange):
                            # 新计算的 SL 更好 (更紧)
                            self.logger.info(f"WS Risk Loop (SL Update): 正在更新 {symbol} 止损至 {potential_sl:.4f} (旧: {current_sl_on_exchange:.4f})")
                            sl_update_tasks.append(
                                self.portfolio.update_position_rules(symbol, stop_loss=potential_sl, reason="WS Risk Loop: Unified SL V5")
                            )
                            state['ai_suggested_stop_loss'] = potential_sl # 更新本地状态
                            active_sl = potential_sl # [BUG 修复] 最终生效的 SL 是新的 potential_sl
                        else:
                            # 新计算的 SL 更松，拒绝更新
                            # [BUG 修复] 最终生效的 SL 必须保持为旧的 current_sl_on_exchange
                            active_sl = current_sl_on_exchange
                        
                        # =======================================================
                        # [I] 检查最终止损 (使用 'active_sl')
                        # =======================================================
                        if (side == 'long' and price <= active_sl) or (side == 'short' and price >= active_sl):
                            reason = f"Unified SL V5 Hit (Price: {price} vs Active SL: {active_sl:.4f})"
                            if symbol not in positions_to_close: positions_to_close[symbol] = reason
                            continue 

                        # [J] 检查 AI 止盈 (保持不变)
                        ai_tp = state.get('ai_suggested_take_profit')
                        if ai_tp and ai_tp > 0:
                            if (side == 'long' and price >= ai_tp) or (side == 'short' and price <= ai_tp):
                                reason = f"AI TP Hit ({ai_tp:.4f})"
                                if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                continue 

                        # =======================================================
                        # [K] 分批止盈 (V-Partial-TP) - (保持不变)
                        # =======================================================
                        current_stage = self.partial_tp_stage.get(symbol, 0)
                        counter = self.partial_tp_counter.get(symbol, 1) 

                        new_stage = current_stage
                        if current_peak_rate >= 0.05: new_stage = 3
                        elif current_peak_rate >= 0.025: new_stage = 2
                        elif current_peak_rate >= 0.01: new_stage = 1
                        
                        if new_stage > current_stage:
                            self.logger.info(f"WS Risk Loop ({symbol}): 进入阶段 {new_stage} (Peak: {current_peak_rate:.2%})。计数器重置为 1。")
                            self.partial_tp_stage[symbol] = new_stage
                            self.partial_tp_counter[symbol] = 1
                            current_stage = new_stage
                            counter = 1

                        # --- 检查阶段 1 ---
                        if current_stage == 1 and counter == 1:
                            target_profit_rate = current_peak_rate * 0.55
                            if rate < target_profit_rate:
                                self.logger.warning(f"WS Risk Loop (Stage 1): {symbol} 触发回撤! (Peak: {current_peak_rate:.2%}, Curr: {rate:.2%}).")
                                self.logger.warning(f"WS Risk Loop (Stage 1): 正在卖出 50%。")
                                tasks_partial_close.append(
                                    self.portfolio.live_partial_close(symbol, size_percent=0.50, reason="WS Risk Loop: Stage 1 TP (50%)")
                                )
                                self.partial_tp_counter[symbol] = 0 
                        
                        # --- 检查阶段 2 ---
                        elif current_stage == 2 and counter == 1:
                            target_profit_rate = current_peak_rate * 0.65
                            if rate < target_profit_rate:
                                self.logger.warning(f"WS Risk Loop (Stage 2): {symbol} 触发回撤! (Peak: {current_peak_rate:.2%}, Curr: {rate:.2%}).")
                                self.logger.warning(f"WS Risk Loop (Stage 2): 正在卖出 30%。")
                                tasks_partial_close.append(
                                    self.portfolio.live_partial_close(symbol, size_percent=0.30, reason="WS Risk Loop: Stage 2 TP (30%)")
                                )
                                self.partial_tp_counter[symbol] = 0
                        
                        # --- 检查阶段 3 ---
                        elif current_stage == 3 and counter == 1:
                            target_profit_rate = current_peak_rate * 0.75
                            if rate < target_profit_rate:
                                self.logger.warning(f"WS Risk Loop (Stage 3): {symbol} 触发回撤! (Peak: {current_peak_rate:.2%}, Curr: {rate:.2%}).")
                                self.logger.warning(f"WS Risk Loop (Stage 3): 正在卖出所有剩余仓位。")
                                if symbol not in positions_to_close:
                                    positions_to_close[symbol] = "WS Risk Loop: Stage 3 TP (Full)"
                                self.partial_tp_counter[symbol] = 0

                    except Exception as e_risk_inner:
                        self.logger.error(f"WS Risk Loop 内部错误: {e_risk_inner}", exc_info=True)
                    # ----------------------------------------------------
                    # --- [统一风控逻辑 V5 结束] ---
                    # ----------------------------------------------------

                # 7. (同旧) 批量执行任务
                if tasks_partial_close:
                    self.logger.info(f"WS Risk Loop: 正在执行 {len(tasks_partial_close)} 个部分平仓任务...")
                    await asyncio.gather(*tasks_partial_close, return_exceptions=True)
                
                if sl_update_tasks:
                    await asyncio.gather(*sl_update_tasks, return_exceptions=True)
                    
                if positions_to_close:
                     tasks_close = [self.portfolio.live_close(symbol, reason=f"WS Risk Loop: {reason}") for symbol, reason in positions_to_close.items()]
                     await asyncio.gather(*tasks_close, return_exceptions=True)
                     self.logger.info(f"WS Risk Loop: Hard Close actions executed for: {list(positions_to_close.keys())}")
                
            except asyncio.CancelledError: 
                self.logger.warning("WS Risk Loop task cancelled, shutting down..."); 
                break
            except Exception as e: 
                self.logger.critical(f"WS Risk Loop 致命错误: {e}", exc_info=True); 
                self.logger.critical("将在 5 秒后尝试重启 WebSocket...")
                await asyncio.sleep(5) # 在重试 WebSocket 连接前等待 5 秒

    async def start(self):
        self.logger.warning(f"🚀 AlphaTrader starting! Mode: {'LIVE' if self.is_live_trading else 'PAPER'}")
        if self.is_live_trading:
            self.logger.warning("!!! LIVE MODE !!! Syncing state on startup...")
            if not hasattr(self, 'client') and hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
            try: 
                await self.portfolio.sync_state(); self.logger.warning("!!! LIVE State Sync Complete !!!")
            except Exception as e_sync: self.logger.critical(f"Initial LIVE state sync failed: {e_sync}", exc_info=True)
        
        # [WebSocket 升级] 启动新的风控循环
        asyncio.create_task(self.websocket_risk_loop())
        
        LIMIT_ORDER_TIMEOUT_MS = settings.AI_LIMIT_ORDER_TIMEOUT_SECONDS * 1000
        
        try:
            while True:
                try:
                    # 步骤 1: 状态同步 (10s 周期)
                    try:
                        await self.portfolio.sync_state()
                    except Exception as e_sync:
                        self.logger.critical(f"Main loop sync_state failed: {e_sync}. Skipping AI cycle, will retry...", exc_info=True)
                        await asyncio.sleep(30) 
                        continue
                    
                    # 步骤 2: 限价单超时清理 (10s 周期)
                    if self.portfolio.pending_limit_orders:
                        now_ms = time.time() * 1000
                        orders_to_cancel = [] 
                        plans_to_remove = []  
                        try:
                            for symbol, plan in list(self.portfolio.pending_limit_orders.items()):
                                order_id = plan.get('order_id')
                                timestamp = plan.get('timestamp')
                                if not order_id:
                                    plans_to_remove.append(symbol)
                                    continue
                                if not timestamp:
                                    plans_to_remove.append(symbol)
                                    if self.is_live_trading: orders_to_cancel.append((order_id, symbol))
                                    continue
                                if (now_ms - timestamp) > LIMIT_ORDER_TIMEOUT_MS:
                                    self.logger.warning(f"TIMEOUT VETO: {symbol} 挂单 {order_id} 超时。正在取消...")
                                    plans_to_remove.append(symbol)
                                    if self.is_live_trading: orders_to_cancel.append((order_id, symbol))
                            
                            if plans_to_remove:
                                for symbol in plans_to_remove:
                                    await self.portfolio.remove_pending_limit_order(symbol)
                            
                            if self.is_live_trading and orders_to_cancel:
                                cancel_tasks = [self.client.cancel_order(oid, sym) for oid, sym in orders_to_cancel]
                                await asyncio.gather(*cancel_tasks, return_exceptions=True)
                                self.logger.info(f"Main Loop: 成功取消 {len(orders_to_cancel)} 个超时挂单。")
                        except Exception as e_timeout:
                            self.logger.error(f"Main Loop: 限价单超时清理时发生错误: {e_timeout}", exc_info=True)


                    # 步骤 3: [低频] 获取所有特征数据 (10s 周期)
                    await self._update_fear_and_greed_index()
                    market_data, tickers = await self._gather_all_market_data()

                    # --- [吊灯止损 升级] ---
                    # 步骤 2: 将 15m ATR 和 15m High/Low 数据传递给毫秒级循环
                    try:
                        temp_indicator_data = {}
                        for symbol, data in market_data.items():
                            atr_15m = data.get('15min_atr_14')
                            high_15m_20p = data.get('15min_recent_high') # 来自 20 周期
                            low_15m_20p = data.get('15min_recent_low')  # 来自 20 周期

                            if atr_15m and atr_15m > 0 and high_15m_20p and low_15m_20p:
                                temp_indicator_data[symbol] = {
                                    "atr": atr_15m,
                                    "high": high_15m_20p,
                                    "low": low_15m_20p
                                }
                        self.current_indicator_data = temp_indicator_data # 原子性更新
                        self.logger.debug(f"Indicator data (ATR/High/Low) updated for {len(self.current_indicator_data)} symbols.")
                    except Exception as e_indicator_pass:
                        self.logger.error(f"Failed to pass indicator data to WS loop: {e_indicator_pass}")
                    # --- [吊灯止损 升级结束] ---

                    # 步骤 4: [V-Ultimate 优化 V4] 动态安全网 (ADX 过滤 + 1.0*ATR 缓冲)
                    if self.is_live_trading:
                        self.logger.debug("Checking Dynamic Safety Net (V4 - ADX Filtered) for losing positions...")
                        try:
                            open_positions = self.portfolio.position_manager.get_all_open_positions()
                            positions_to_close = []
                            
                            ADX_TREND_THRESHOLD = 20 

                            for symbol, state in open_positions.items():
                                price = tickers.get(symbol, {}).get('last')
                                if not price or price <= 0: continue

                                entry = state.get('avg_entry_price')
                                side = state.get('side')
                                
                                is_losing = (side == 'long' and price < entry) or (side == 'short' and price > entry)
                                
                                if is_losing:
                                    try:
                                        entries_list = state.get('entries', [])
                                        if not entries_list: continue 
                                        
                                        last_entry_timestamp = entries_list[-1].get('timestamp', 0)
                                        time_since_entry = (time.time() * 1000) - last_entry_timestamp
                                        
                                        if time_since_entry < self.SAFETY_NET_GRACE_PERIOD_MS:
                                            self.logger.debug(f"Safety Net V4: {symbol} 仍在 {self.SAFETY_NET_GRACE_PERIOD_MS/60000:.0f} 分钟宽限期内。跳过。")
                                            continue
                                            
                                    except Exception as e_ts:
                                        self.logger.warning(f"Safety Net V4: 无法获取 {symbol} 的 entry timestamp: {e_ts}。将继续检查。")

                                    data_1h = market_data.get(symbol, {})
                                    ema_50 = data_1h.get('1hour_ema_50')
                                    atr_14 = data_1h.get('1hour_atr_14') 
                                    adx_14 = data_1h.get('1hour_adx_14') 

                                    if not ema_50 or not atr_14 or atr_14 <= 0 or not adx_14:
                                        self.logger.warning(f"Safety Net V4: 无法获取 {symbol} 的 1h EMA/ATR/ADX 数据。跳过。")
                                        continue
                                    
                                    if adx_14 < ADX_TREND_THRESHOLD:
                                        self.logger.debug(f"Safety Net V4: {symbol} 处于震荡市 (1h ADX {adx_14:.1f} < {ADX_TREND_THRESHOLD})。安全网已禁用。")
                                        continue 
                                    
                                    ATR_BUFFER_MULTIPLIER = 1.5
                                    buffer = atr_14 * ATR_BUFFER_MULTIPLIER
                                    
                                    if side == 'long':
                                        stop_line = ema_50 - buffer
                                        if price < stop_line:
                                            positions_to_close.append((symbol, f"Safety Net V4: Price < (1h_EMA - 1.0*ATR) ({price:.4f} < {stop_line:.4f})"))
                                    elif side == 'short':
                                        stop_line = ema_50 + buffer
                                        if price > stop_line:
                                            positions_to_close.append((symbol, f"Safety Net V4: Price > (1h_EMA + 1.0*ATR) ({price:.4f} > {stop_line:.4f})"))

                            if positions_to_close:
                                close_tasks = [self.portfolio.live_close(symbol, reason=reason) for symbol, reason in positions_to_close]
                                self.logger.warning(f"🔥 动态安全网 V4 (ADX Filtered) 触发: 正在平仓 {len(positions_to_close)} 个仓位。")
                                await asyncio.gather(*close_tasks, return_exceptions=True)
                                await self.portfolio.sync_state() 

                        except Exception as e_safety_net:
                            self.logger.error(f"动态安全网 V4 (1h EMA + ATR) 检查失败: {e_safety_net}", exc_info=True)
                    
                    # 步骤 4.5: 取消过时/偏差过大的限价单
                    deviation_threshold = settings.AI_LIMIT_ORDER_DEVIATION_PERCENT / 100.0
                    
                    if self.is_live_trading and self.portfolio.pending_limit_orders:
                        self.logger.debug("Checking for stale (price deviation) LIVE limit orders...")
                        try:
                            orders_to_cancel = []
                            for symbol, plan in list(self.portfolio.pending_limit_orders.items()):
                                current_price = tickers.get(symbol, {}).get('last')
                                plan_price = plan.get('limit_price')
                                order_id = plan.get('order_id')

                                if not all([current_price, plan_price, order_id]):
                                    self.logger.warning(f"Stale Price Check (Live): 缺少 {symbol} 的价格或计划数据。")
                                    continue
                                
                                deviation_pct = abs(current_price - plan_price) / plan_price
                                if deviation_pct > deviation_threshold:
                                    self.logger.warning(f"STALE PRICE VETO (Live): {symbol} 挂单价 {plan_price} 与现价 {current_price} 偏离 ({deviation_pct:.2%}) > 阈值。正在取消...")
                                    orders_to_cancel.append((order_id, symbol))
                                    await self.portfolio.remove_pending_limit_order(symbol)

                            if orders_to_cancel:
                                cancel_tasks = [self.client.cancel_order(oid, sym) for oid, sym in orders_to_cancel]
                                await asyncio.gather(*cancel_tasks, return_exceptions=True)
                                self.logger.info(f"Stale Price Check (Live): 成功取消 {len(orders_to_cancel)} 个过时挂单。")

                        except Exception as e_stale_check:
                             self.logger.error(f"过时挂单 (Live) 检查失败: {e_stale_check}", exc_info=True)
                    
                    elif (not self.is_live_trading) and self.portfolio.pending_limit_orders:
                        self.logger.debug("Checking for stale (price deviation) PAPER limit orders...")
                        try:
                            plans_to_remove = []
                            for symbol, plan in list(self.portfolio.pending_limit_orders.items()):
                                current_price = tickers.get(symbol, {}).get('last')
                                plan_price = plan.get('limit_price')

                                if not all([current_price, plan_price]):
                                    self.logger.warning(f"Stale Price Check (Paper): 缺少 {symbol} 的价格或计划数据。")
                                    continue
                                
                                deviation_pct = abs(current_price - plan_price) / plan_price
                                if deviation_pct > deviation_threshold:
                                    self.logger.warning(f"STALE PRICE VETO (Paper): {symbol} 挂单价 {plan_price} 与现价 {current_price} 偏离 ({deviation_pct:.2%}) > 阈值。正在移除...")
                                    plans_to_remove.append(symbol)

                            if plans_to_remove:
                                for symbol in plans_to_remove:
                                    await self.portfolio.remove_pending_limit_order(symbol) 
                                self.logger.info(f"Stale Price Check (Paper): 成功移除 {len(plans_to_remove)} 个过时模拟挂单。")
                        
                        except Exception as e_stale_paper_check:
                             self.logger.error(f"过时模拟挂单 (Paper) 检查失败: {e_stale_paper_check}", exc_info=True)

                    
                    # 步骤 5: [中频] Rule 6 ATR 追踪止损 (10s 周期)
                    # --- [WebSocket 升级] ---
                    # [已删除] 此模块 (Rule 6 ATR Trail) 已被移除
                    # 它的逻辑已合并到 websocket_risk_loop 的统一止损计算中
                    # --- [升级结束] ---
                    
                    
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
                                    ohlcv_15m = market_data.get(sym, {}).get('ohlcv_15m', [])
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
                                anomaly_score = await self._get_ml_anomaly_score(symbol, market_data)
                                market_data[symbol]['anomaly_score'] = anomaly_score
                                
                                ml_pred = await self._get_ml_prediction(symbol, market_data)
                                market_data[symbol]['ml_proba_up'] = ml_pred['proba_up']
                                market_data[symbol]['ml_proba_down'] = ml_pred['proba_down']
                                
                                peak_profit_rate = self.peak_profit_tracker.get(symbol, 0.0)
                                market_data[symbol]['peak_profit_achieved_percent'] = peak_profit_rate * 100.0
                                
                        self.logger.debug("ML scores pre-processed.")

                        await self.run_cycle(market_data, tickers); 
                        self.last_run_time = now
                    
                    await asyncio.sleep(10) # 10秒主循环 (低频)
                    
                except asyncio.CancelledError: 
                    self.logger.warning("Main loop task cancelled, shutting down..."); 
                    break 
                except Exception as e: 
                    self.logger.critical(f"Main loop fatal error (inside loop): {e}", exc_info=True); 
                    await asyncio.sleep(60) 
            
        finally:
            await self.httpx_client.aclose()
            self.logger.info("HTTPA(Httpx) 客户端已关闭。")
