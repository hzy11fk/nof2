# 文件: alpha_trader.py (V-Final - 放弃 Rule 8，集成 OI Regime 和 Strategist V2)
# 描述: 
# 1. (已移除) 删除了 Rule 8 策略 (_check_python_rule_8, _build_python_order)。
# 2. (已移除) 删除了 start() 循环中对 Rule 8 的调用。
# 3. (已移除) 删除了 high_frequency_risk_loop() 中对 Rule 8 的特殊追踪止损。
# 4. (已优化) _gather_all_market_data 现在获取 1h OI 数据并计算 'oi_regime_1h' (Rising/Falling)。
# 5. (已优化) SYSTEM_PROMPT_TEMPLATE 已替换为 "V2 Strategist" 版本。
# 6. (已优化) _build_prompt 现在提供 'oi_regime_1h' 而不是 'oi_change_5m_pct'。
# 7. (已新增) 新增了 _validate_ai_trade() 辅助函数，用于 Python 端的硬 Veto 规则。
# 8. (已重写) _execute_decisions() 已重写，以处理 V2 Prompt 的简化 JSON，并调用 _validate_ai_trade()。

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
    
    # --- [AI (Rule 6) 专用 PROMPT (V2 - 策略师版)] ---
    SYSTEM_PROMPT_TEMPLATE = """
    You are an **Expert Market Analyst and Trading Strategist**. Your primary goal is to **identify high-probability trading theses** (the 'Why') based on the provided market data.

    You are the 'Strategist' (the brain), and a separate, 100% reliable Python system is the 'Executor' (the hands).

    **Your Task (The 'Why'):**
    1.  Analyze the multi-timeframe data to determine the market regime (Trending, Ranging, Chop).
    2.  Identify **ONE** high-confidence trade setup (e.g., "Bullish pullback to 4h support").
    3.  Provide the **key strategic parameters** for this setup:
        * `"thesis"`: Your reasoning (e.g., "Rule 6.1 Pullback to 4h BB_Mid, 1h OI_Regime is Rising").
        * `"entry_price"`: The ideal limit order price.
        * `"stop_loss_price"`: The price where your thesis is invalidated (ATR-based).
        * `"take_profit_price"`: The initial profit target.
        * `"confidence_level"`: [CHOOSE ONE: "High", "Medium"]
    4.  Explain your reasoning in the `chain_of_thought`.

    **Python's Task (Do NOT do this yourself):**
    * Python will perform **all** final checks and calculations.
    * Python will **VETO** your trade if it fails hard rules (e.g., Trend Filter, Anomaly Score, OI Matrix).
    * Python will **calculate** all final sizing, leverage, and `risk_percent` based on your `confidence_level`.
    * Python manages all open positions (SL/TP adjustments) via a high-frequency loop.

    **Focus 100% on the quality of the thesis, not the math.**

    **Core Strategy Mandates (Your Guide):**

    1.  **Strategy: Limit Orders Only (CRITICAL):**
        * Your **only** strategy is to be patient and trade pullbacks (Rule 6.1), mean-reversion (Rule 6.2), or chop-zones (Rule 6.3).
        * You MUST and ONLY use `LIMIT_BUY` or `LIMIT_SELL`.

    2.  **Market State Recognition (Default Strategy):**
        You MUST continuously assess the market regime using the **1hour** and **4hour** timeframes. This is your **Default Strategy**.
        * **1. Strong Trend (ADX > 25):**
            * **Strategy (LIMIT ONLY):** Identify a **'Pullback Confluence Zone'**. This is a small price area where **at least two** key S/R levels overlap.
            * *Zone Example (Long):* The `1h EMA 20` is at $65,100 **AND** the `4h BB_Mid` is at $65,050.
            * **Timing:** Place the `LIMIT_BUY` (in uptrend) or `LIMIT_SELL` (in downtrend) **only if** the pullback appears exhausted (e.g., `15min RSI` < 40 for Long).
            * **[Confirmation]:** This is High-Confidence *only if* `ML_Proba` confirms (e.g., > 0.60) AND the `OI_Regime_1h` is "Rising" (confirming new money).
        * **2. Ranging (ADX < 20):**
            * **Strategy (LIMIT ONLY):** **Mean-reversion**. Issue `LIMIT_SELL` at (or near) the `BB_Upper` or `LIMIT_BUY` at (or near) the `BB_Lower`.
            * **[Confirmation]:** You MUST check the correct `ML_Proba` (e.g., `ML_Proba_DOWN > 0.60` for `LIMIT_SELL`).
        * **3. Chop (ADX 20-25):**
            * **Strategy (LIMIT ONLY):** Low-conviction. Shift focus to **15min timeframe**. May issue `LIMIT` orders at 15m BBands with 'Medium' confidence.

    3.  **Data Confirmation (CRITICAL):**
        * You MUST check the `OI_Regime_1h`.
        * **P↑ O↑ (Price Up, OI Rising):** Healthy Trend. **CONFIRMS** `LIMIT_BUY` on pullbacks.
        * **P↓ O↑ (Price Down, OI Rising):** Healthy Downtrend. **CONFIRMS** `LIMIT_SELL` on pullbacks.
        * **P↑ O↓ (Price Up, OI Falling):** Short Squeeze. **VETO** all new `LIMIT_BUY` plans.
        * **P↓ O↓ (Price Down, OI Falling):** Long Squeeze/Capitulation. **VETO** all new `LIMIT_SELL` plans.

    4.  **Smarter Stop-Loss:**
        * Your `stop_loss_price` MUST be placed relative to volatility using the **ATR**.
        * *Example (Long):* Place `stop_loss_price` at `[Confluence_Zone_Low] - (1.5 * 1h_atr_14)`.
        * *Example (Short):* Place `stop_loss_price` at `[Confluence_Zone_High] + (1.5 * 1h_atr_14)`.
        
    5.  **Market Sentiment Filter (Fear & Greed Index):**
        You MUST use the provided `Fear & Greed Index` (from the User Prompt) as a macro filter.
        * **Extreme Fear (Index < 25):** ...
        * **Fear (Index 25-45):** ...
        * **Neutral (Index 45-55):** ...
        * **Greed (Index 55-75):** ...
        * **Extreme Greed (Index > 75):** ...

    **MANDATORY OUTPUT FORMAT:**
    Your entire response must be a single JSON object with two keys: "chain_of_thought" and "orders".

    1.  `"chain_of_thought"` (string): A multi-line string containing your detailed analysis in English. It MUST follow this template precisely:
        ```
        My Current Assessment & Actions (Rule 6 - Strategist)

        Market State Analysis:
        - 1h ADX: [Value] | 4h ADX: [Value]
        - Regime: [Applying Rule 6: Trending Pullback (ADX>25) / Ranging (ADX<20) / Chop (ADX 20-25)]
        - Market Sentiment: [F&G Index value and its implication]

        Portfolio Overview:
        (Python handles all open positions. I am focused on new opportunities.)

        New Trade Opportunities Analysis (Rule 6 - Limit Orders Only):
        
        [CRITICAL STATE CHECK (NO HEDGING): Before analyzing any new symbol, you MUST check the 'Open Positions' and 'Pending Limit Orders' lists. If a position or pending order for that [SYMBOL] already exists, you MUST SKIP analysis for that symbol.]

        [EXAMPLE - RULE 6.1 (Trending Pullback):]
        ETH Multi-Timeframe Assessment (Market State: Trending Bullish, 4h ADX=28):
        - Thesis: Price is pulling back to a confluence support zone (4h BB_Mid + 1h EMA 20). 15m RSI is low (38), indicating a good entry point.
        - Data Check: The `OI_Regime_1h` is "Rising", which confirms (P↑ O↑) that new money supports this uptrend.
        - ML Confirmation: ML_Proba_UP = 0.68.
        - Final Confidence: High.
        - Plan: PREPARE LIMIT_BUY.

        [EXAMPLE - RULE 3 (OI VETO):]
        SOL Multi-Timeframe Assessment (Market State: Trending Bullish, 4h ADX=30):
        - Thesis: Price is breaking out, pulling back to new support.
        - Data Check: The `OI_Regime_1h` is "Falling".
        - Final Confidence: VETO. This is a P↑O↓ (Short Squeeze). It is too dangerous to open a new LONG.
        - Plan: ABORT.

        In summary, [**Key Instruction: Please provide your final concise decision overview directly here, in Chinese.**Final concise decision overview.]
        ```

    2.  `"orders"` (list): A list of JSON objects for trades. Empty list `[]` if holding.

    **Order Object Rules (NEW SIMPLIFIED FORMAT):**
    -   **To Open Limit (LONG - Rule 6):** `{{"action": "LIMIT_BUY", "symbol": "...", "thesis": "Rule 6.1 Pullback. Confidence: High. OI Rising.", "entry_price": [CALCULATED_PRICE], "take_profit_price": ..., "stop_loss_price": ..., "confidence_level": "High"}}`
    -   **To Open Limit (SHORT - Rule 6):** `{{"action": "LIMIT_SELL", "symbol": "...", "thesis": "Rule 6.2 Ranging. Confidence: Medium. ML Confirmed.", "entry_price": [CALCULATED_PRICE], "take_profit_price": ..., "stop_loss_price": ..., "confidence_level": "Medium"}}`
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
        
        self.fng_data: Dict[str, Any] = {"value": 50, "value_classification": "Neutral"}
        self.last_fng_fetch_time: float = 0.0
        self.FNG_CACHE_DURATION_SECONDS = 3600
        
        # 检查交易所能力
        self.has_oi_history = self.client.has.get('fetchOpenInterestHistory', False)
        # [V-Final 移除] Taker Ratio 不再使用
        # self.has_taker_ratio = self.client.has.get('fapiPublicGetTakerlongshortRatio', False)
        if not self.has_oi_history:
             self.logger.warning("Exchange does not support 'fetchOpenInterestHistory'. OI data will be unavailable.")
        # if not self.has_taker_ratio:
        #      self.logger.warning("Exchange does not support 'fapiPublicGetTakerlongshortRatio'. Taker Ratio data will be unavailable.")

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
        self.logger.debug("Gathering multi-TF market data (5m, 15m, 1h, 4h) + Indicators + OI Regime...")
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
                    # [V-Final 修改] 获取最近 50 根 1h OI
                    return await self.exchange.fetch_open_interest_history(symbol, timeframe='1h', limit=50)
                except Exception as e:
                    self.logger.error(f"Safe Fetch OI History Error ({symbol}): {e}", exc_info=False)
                    return e


        try:
            timeframes = ['5m', '15m', '1h', '4h']
            tasks = []
            for symbol in self.symbols:
                for timeframe in timeframes: 
                    limit = 150 if timeframe == '15m' else 100
                    tasks.append(_safe_fetch_ohlcv(symbol, timeframe, limit=limit))
                tasks.append(_safe_fetch_ticker(symbol))
                # [V-Final 修改] 添加 OI 任务
                tasks.append(_safe_fetch_oi_history(symbol))
               # [V-Final 移除] Taker Ratio 任务
                
            results = await asyncio.gather(*tasks)
            
            total_timeframes = len(timeframes)
            tasks_per_symbol = total_timeframes + 1 + 1 # 1 个新任务 (OI)
            
            for i, symbol in enumerate(self.symbols):
                start_index = i * tasks_per_symbol
                symbol_ohlcv_results = results[start_index : start_index + total_timeframes]
                ticker_result = results[start_index + total_timeframes]
                oi_history_result = results[start_index + total_timeframes + 1]

                if not isinstance(ticker_result, Exception) and ticker_result and ticker_result.get('last') is not None:
                    fetched_tickers[symbol] = ticker_result; market_indicators_data[symbol] = {'current_price': ticker_result.get('last')}
                else: 
                    market_indicators_data[symbol] = {'current_price': None}
                    self.logger.warning(f"Failed fetch ticker/price for {symbol} (Result: {ticker_result})")
                
                # [V-Final 修改] 处理 1h OI Regime
                market_indicators_data[symbol]['oi_regime_1h'] = "Neutral" # 默认值
                if isinstance(oi_history_result, list) and len(oi_history_result) >= 21: # 需要足够数据来计算 20-EMA
                    try:
                        # 转换数据
                        oi_values = [float(item['openInterestValue']) for item in oi_history_result]
                        oi_df = pd.DataFrame({'oi': oi_values})
                        
                        # 计算 20-EMA
                        oi_ema_20 = ta.ema(oi_df['oi'], 20)
                        
                        if oi_ema_20 is not None and not oi_ema_20.empty:
                            current_oi = oi_df['oi'].iloc[-1]
                            current_oi_ema = oi_ema_20.iloc[-1]
                            
                            # 比较当前 OI 和 EMA
                            if current_oi > current_oi_ema:
                                market_indicators_data[symbol]['oi_regime_1h'] = "Rising"
                            elif current_oi < current_oi_ema:
                                market_indicators_data[symbol]['oi_regime_1h'] = "Falling"
                    except Exception as e_oi:
                        self.logger.warning(f"Error processing 1h OI Regime for {symbol}: {e_oi}")
                # [V-Final 修改结束]


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
                                    market_indicators_data[symbol][f'{prefix}bb_upper'] = bbands.iloc[:, 2].iloc[-1] # BBU
                                    market_indicators_data[symbol][f'{prefix}bb_middle'] = bbands.iloc[:, 1].iloc[-1] # BBM
                                    market_indicators_data[symbol][f'{prefix}bb_lower'] = bbands.iloc[:, 0].iloc[-1] # BBL
                                    
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
        
       # prompt += "\n--- Previous AI Summary (For Context Only) ---\n"
      #  prompt += f"{self.last_strategy_summary}\n"
        
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
            
            # [V-Final 修改] 提供 1h OI Regime
            prompt += f"OI_Regime_1h: {d.get('oi_regime_1h', 'N/A')}\n"

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

    # [V-Final 新增] _execute_decisions 的辅助验证函数
    def _validate_ai_trade(self, order: Dict, market_data: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        [V2 核心] Python 执行者：验证 AI 策略的硬规则。
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
        
        if not all([current_price, ema_50_4h, oi_regime, limit_price, stop_loss_price, take_profit_price]):
            return False, f"Missing critical data (Price, EMA, OI, or Order Prices)"

        # 1. [VETO] Anomaly Check (Rule 1.5)
        if anomaly_score < -0.1:
            return False, f"Anomaly Veto (Score: {anomaly_score:.3f})"

        # 2. [VETO] Trend Filter Check (Rule 1.B)
        if action == "LIMIT_BUY" and current_price < ema_50_4h:
            return False, f"Trend Filter Veto (Price {current_price} < 4h EMA {ema_50_4h})"
        if action == "LIMIT_SELL" and current_price > ema_50_4h:
            return False, f"Trend Filter Veto (Price {current_price} > 4h EMA {ema_50_4h})"
            
        # 3. [VETO] OI Matrix Check (Rule 3)
        # P↑O↓ (轧空)
        if action == "LIMIT_BUY" and current_price > ema_50_4h and oi_regime == "Falling":
             return False, f"OI Matrix Veto (P↑O↓ - Short Squeeze)"
        # P↓O↓ (多杀多)
        if action == "LIMIT_SELL" and current_price < ema_50_4h and oi_regime == "Falling":
             return False, f"OI Matrix Veto (P↓O↓ - Long Squeeze)"

        # 4. [VETO] R:R Check
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

    # [V-Final 重写] _execute_decisions
    async def _execute_decisions(self, decisions: list, market_data: Dict[str, Dict[str, Any]]):
        """
        [V2 核心] Python 执行者：
        1. 验证 AI 策略 (Veto 规则)
        2. 计算 Risk/Leverage
        3. 计算 Size
        4. 执行 (调用 portfolio)
        """
        MIN_MARGIN_USDT = futures_settings.MIN_NOMINAL_VALUE_USDT
        MIN_SIZE_BTC = 0.001 

        for order in decisions:
            try:
                action = order.get('action')
                symbol = order.get('symbol')
                
                # [V-Final 修改] 只处理 AI 的新 'LIMIT' action
                if action not in ["LIMIT_BUY", "LIMIT_SELL"]:
                    self.logger.warning(f"跳过 AI 未知指令: {action}"); continue
                if not symbol or symbol not in self.symbols: 
                    self.logger.warning(f"跳过无效或不支持的 symbol: {symbol}"); continue

                # 1. 验证 AI 策略 (Veto 规则)
                is_valid, reason = self._validate_ai_trade(order, market_data)
                if not is_valid:
                    self.logger.warning(f"!!! AI 策略 VETO !!! {symbol} | Action: {action} | 原因: {reason}")
                    continue
                
                self.logger.info(f"AI 策略 (Python 验证通过): {symbol} | Action: {action} | 原因: {reason}")

                # 2. 计算 Risk/Leverage/Size (如果验证通过)
                
                # AI 提供的策略参数
                limit_price = float(order.get('entry_price'))
                stop_loss = float(order.get('stop_loss_price'))
                take_profit = float(order.get('take_profit_price'))
                confidence = order.get('confidence_level', 'Medium')
                ai_thesis = order.get('thesis', 'N/A')
                
                # a. 确定 Risk Percent
                risk_percent = 0.015 # 默认 Medium
                if confidence == "High":
                    risk_percent = 0.025 # High
                
                # b. 确定 Leverage
                # (使用配置中的固定杠杆，因为 AI 不再决定这个)
                leverage = int(futures_settings.FUTURES_LEVERAGE) 
                
                # c. 检查是否为模拟盘
                if not self.is_live_trading:
                    self.logger.warning(f"模拟盘：跳过 {action} 执行 (逻辑已验证)。"); continue
                
                # d. [复用逻辑] 计算保证金和规模
                # (此逻辑复用自你 V1 的 _execute_decisions)
                
                try:
                    total_equity = float(self.portfolio.equity)
                    available_cash = float(self.portfolio.cash)
                    if total_equity <= 0: raise ValueError(f"无效账户状态 (Equity <= 0)")
                    
                    calculated_desired_margin = total_equity * risk_percent
                    
                    max_margin_cap = total_equity * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
                    if calculated_desired_margin > max_margin_cap:
                        self.logger.warning(f"!!! {action} Margin Capped !!! AI 期望保证金 {calculated_desired_margin:.2f} > 最大 {max_margin_cap:.2f}")
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
                
                # 4. 执行 (调用 portfolio)
                # (alpha_portfolio.py 提供了这个强大的函数)
                await self.portfolio.live_open_limit(
                    symbol, 
                    'long' if action == 'LIMIT_BUY' else 'short', 
                    final_size, 
                    leverage, 
                    limit_price,
                    reason=ai_thesis, 
                    stop_loss=stop_loss, 
                    take_profit=take_profit, 
                    invalidation_condition=f"AI V2 ({confidence})" # 使用 Invalidation 字段记录信心
                )

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

    
    # [V-Final 移除] _check_python_rule_8

    # [V-Final 移除] _build_python_order


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

        # [V-Final 修复] filter_rule8=True 现在是默认行为
        portfolio_state = self.portfolio.get_state_for_prompt(tickers, filter_rule8=True)
        
        user_prompt_string = self._build_prompt(market_data, portfolio_state, tickers)

        try:
            system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                symbol_list=self.formatted_symbols
                # [V-Final 移除] 'specific_rules_or_notes' 不再需要
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
                self.logger.info(f"Extracted English summary: '{summary_for_ui[:50]}...'")
            else:
                 summary_for_ui = "AI summary was empty."
        else:
            self.logger.warning("AI CoT did not find 'In summary,' keyword.")
        
        self.last_strategy_summary = summary_for_ui

        if orders:
            self.logger.info(f"AI proposed {len(orders)} order(s), executing...")
            # [V-Final 修改] 调用 V2 (重写) 的 _execute_decisions
            await self._execute_decisions(orders, market_data)
        else:
            self.logger.info("AI proposed no orders.")

        self.logger.info("="*20 + " AI Cycle Finished " + "="*20 + "\n")


    async def high_frequency_risk_loop(self):
        self.logger.warning("🚀 高频风控循环 (HF Risk Loop) 已启动 (2s 周期)...")
        
        MAX_LOSS_PERCENT = settings.MAX_LOSS_CUTOFF_PERCENT / 100.0
        DUST_MARGIN_USDT = 1.0
        
        while True:
            try:
                if self.is_live_trading and self.portfolio.position_manager:
                    
                    open_positions = self.portfolio.position_manager.get_all_open_positions()
                    if not open_positions:
                        await asyncio.sleep(2) 
                        continue
                        
                    symbols_to_check = list(open_positions.keys())
                    tickers = {}
                    try:
                        tickers = await self.client.fetch_tickers(symbols_to_check)
                    except Exception as e_ticker:
                        self.logger.error(f"HF Risk Loop: 获取 tickers 失败: {e_ticker}")
                        await asyncio.sleep(2) 
                        continue

                    positions_to_close = {} 
                    sl_update_tasks = [] 

                    open_symbols = set(open_positions.keys())
                    for symbol in list(self.peak_profit_tracker.keys()):
                        if symbol not in open_symbols:
                            self.logger.info(f"HF Risk Loop: Removing Peak Profit tracker for closed position: {symbol}")
                            del self.peak_profit_tracker[symbol]

                    try:
                        for symbol, state in open_positions.items():
                            price = tickers.get(symbol, {}).get('last')
                            if not price or price <= 0: continue
                            
                            entry = state.get('avg_entry_price')
                            size = state.get('total_size')
                            side = state.get('side')
                            lev = state.get('leverage')
                            margin = state.get('margin')
                            
                            initial_sl = state.get('ai_suggested_stop_loss', 0.0) 
                            
                            if not all([entry, size, side, lev, margin, initial_sl]) or lev <= 0 or entry <= 0 or margin <= 0 or initial_sl <= 0 or size <= 0:
                                self.logger.warning(f"HF Risk Loop: Skipping {symbol}, invalid state data (entry, size, side, lev, margin, or initial_sl missing/zero).")
                                continue

                            upl = (price - entry) * size if side == 'long' else (entry - price) * size
                            rate = upl / margin 

                            inval_cond = state.get('invalidation_condition') or '' 
                            
                            # [V-Final 移除] Rule 8 的特殊追踪止损逻辑
                            # is_rule_8_trade = "Python Rule 8" in inval_cond
                            # if is_rule_8_trade:
                            #   ... (逻辑已删除) ...

                            if rate <= -MAX_LOSS_PERCENT:
                                reason = f"Hard Max Loss ({-MAX_LOSS_PERCENT:.0%})"
                                if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                continue 

                            if margin < DUST_MARGIN_USDT:
                                reason = f"Dust Close (<{DUST_MARGIN_USDT:.1f}U)"
                                if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                continue 

                            # [集成] 更新 Peak Profit (适用于所有策略)
                            current_peak_rate = self.peak_profit_tracker.get(symbol, 0.0)
                            if rate > current_peak_rate:
                                self.peak_profit_tracker[symbol] = rate
                                current_peak_rate = rate # 使用最新峰值

                            # [V-Final 修改] 现在所有策略都走这个逻辑
                            # 1. 1R Breakeven Check
                            if abs(initial_sl - entry) > 1e-9: 
                                initial_risk_distance = abs(entry - initial_sl)
                                current_upl_distance = 0.0
                                if side == 'long' and price > entry:
                                    current_upl_distance = price - entry
                                elif side == 'short' and price < entry:
                                    current_upl_distance = entry - price
                                    
                                if current_upl_distance >= initial_risk_distance:
                                    self.logger.warning(f"🔥 HF Risk Loop (1R Breakeven): {symbol} 达到 1R。正在将 SL 移至 {entry:.4f}")
                                    sl_update_tasks.append(
                                        self.portfolio.update_position_rules(symbol, stop_loss=entry, reason="HF Risk Loop: 1R Breakeven")
                                    )
                            
                            # 2. [集成] Graded Profit Taker (分级主动止盈)
                            target_profit_rate = 0.0
                            if current_peak_rate >= 0.15: # > 15%
                                target_profit_rate = current_peak_rate * 0.70 # 锁定 70%
                            elif current_peak_rate >= 0.08: # 8% - 15%
                                target_profit_rate = current_peak_rate * 0.75 # 锁定 75% (回撤 25%)
                            elif current_peak_rate >= 0.05: # 5% - 8%
                                target_profit_rate = current_peak_rate * 0.80 # 锁定 80% (回撤 20%)
                            elif current_peak_rate >= 0.03: # 3% - 5%
                                target_profit_rate = current_peak_rate * 0.85 # 锁定 85% (回撤 15%)
                            elif current_peak_rate >= 0.007: # 0.7% - 3%
                                target_profit_rate = current_peak_rate * 0.65 # 锁定 65% (回撤 35%)
                            
                            if target_profit_rate > 0.0:
                                target_upl = target_profit_rate * margin
                                new_graded_sl_price = 0.0
                                if side == 'long':
                                    new_graded_sl_price = (target_upl / size) + entry
                                else:
                                    new_graded_sl_price = entry - (target_upl / size)
                                
                                current_sl = state.get('ai_suggested_stop_loss', 0.0)
                                
                                if side == 'long':
                                    # 新 SL 必须高于条目 和 当前 SL
                                    final_sl_floor = max(entry, current_sl)
                                    if new_graded_sl_price > final_sl_floor:
                                        sl_update_tasks.append(
                                            self.portfolio.update_position_rules(symbol, stop_loss=new_graded_sl_price, reason="HF Risk Loop: Graded Profit Taker")
                                        )
                                elif side == 'short':
                                    # 新 SL 必须低于条目 和 当前 SL
                                    final_sl_floor = min(entry, current_sl)
                                    if new_graded_sl_price < final_sl_floor:
                                         sl_update_tasks.append(
                                            self.portfolio.update_position_rules(symbol, stop_loss=new_graded_sl_price, reason="HF Risk Loop: Graded Profit Taker")
                                        )
                            
                            # 3. 检查 AI 设定的 SL/TP (所有策略)
                            current_active_sl = state.get('ai_suggested_stop_loss') 
                            if current_active_sl and current_active_sl > 0:
                                if (side == 'long' and price <= current_active_sl) or (side == 'short' and price >= current_active_sl):
                                    reason = f"AI SL Hit ({current_active_sl:.4f})"
                                    if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                    continue 

                            ai_tp = state.get('ai_suggested_take_profit')
                            if ai_tp and ai_tp > 0:
                                if (side == 'long' and price >= ai_tp) or (side == 'short' and price <= ai_tp):
                                    reason = f"AI TP Hit ({ai_tp:.4f})"
                                    if symbol not in positions_to_close: positions_to_close[symbol] = reason
                                    continue 

                    except Exception as e_risk_inner:
                        self.logger.error(f"HF Risk Loop 内部错误: {e_risk_inner}", exc_info=True)

                    if sl_update_tasks:
                        await asyncio.gather(*sl_update_tasks, return_exceptions=True)
                    if positions_to_close:
                         tasks_close = [self.portfolio.live_close(symbol, reason=f"HF Risk Loop: {reason}") for symbol, reason in positions_to_close.items()]
                         await asyncio.gather(*tasks_close, return_exceptions=True)
                         self.logger.info(f"HF Risk Loop: Hard Close actions executed for: {list(positions_to_close.keys())}")
                    
                await asyncio.sleep(2) # 2 秒高频循环
                
            except asyncio.CancelledError: 
                self.logger.warning("HF Risk Loop task cancelled, shutting down..."); 
                break
            except Exception as e: 
                self.logger.critical(f"HF Risk Loop 致命错误: {e}", exc_info=True); 
                await asyncio.sleep(10)


    async def start(self):
        self.logger.warning(f"🚀 AlphaTrader starting! Mode: {'LIVE' if self.is_live_trading else 'PAPER'}")
        if self.is_live_trading:
            self.logger.warning("!!! LIVE MODE !!! Syncing state on startup...")
            if not hasattr(self, 'client') and hasattr(self.portfolio, 'client'): self.client = self.portfolio.client
            try: 
                await self.portfolio.sync_state(); self.logger.warning("!!! LIVE State Sync Complete !!!")
            except Exception as e_sync: self.logger.critical(f"Initial LIVE state sync failed: {e_sync}", exc_info=True)
        
        asyncio.create_task(self.high_frequency_risk_loop())
        
        LIMIT_ORDER_TIMEOUT_MS = settings.AI_LIMIT_ORDER_TIMEOUT_SECONDS * 1000
        
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
                if self.is_live_trading and self.portfolio.pending_limit_orders:
                    now_ms = time.time() * 1000
                    orders_to_cancel = []
                    try:
                        for symbol, plan in list(self.portfolio.pending_limit_orders.items()):
                            order_id = plan.get('order_id')
                            timestamp = plan.get('timestamp')
                            if not order_id:
                                await self.portfolio.remove_pending_limit_order(symbol)
                                continue
                            if not timestamp:
                                orders_to_cancel.append((order_id, symbol))
                                await self.portfolio.remove_pending_limit_order(symbol)
                                continue
                            if (now_ms - timestamp) > LIMIT_ORDER_TIMEOUT_MS:
                                orders_to_cancel.append((order_id, symbol))
                                await self.portfolio.remove_pending_limit_order(symbol)
                        
                        if orders_to_cancel:
                            cancel_tasks = [self.client.cancel_order(oid, sym) for oid, sym in orders_to_cancel]
                            await asyncio.gather(*cancel_tasks, return_exceptions=True)
                            self.logger.info(f"Main Loop: 成功取消 {len(orders_to_cancel)} 个超时/孤儿订单。")
                    except Exception as e_timeout:
                        self.logger.error(f"Main Loop: 限价单超时清理时发生错误: {e_timeout}", exc_info=True)


                # 步骤 3: [低频] 获取所有特征数据 (10s 周期)
                await self._update_fear_and_greed_index()
                market_data, tickers = await self._gather_all_market_data()

                # 步骤 4: [V-Final 移除] Python Rule 8 执行
                # (逻辑已删除)
                
                # 步骤 5: [中频] Rule 6 ATR 追踪止损 (10s 周期)
                if self.is_live_trading:
                    sl_update_tasks_rule6 = []
                    try:
                        # [V-Final 修改] 现在所有持仓都由 position_manager 管理
                        open_positions_rule6 = self.portfolio.position_manager.get_all_open_positions()
                        for symbol, state in open_positions_rule6.items():
                            
                            # [V-Final 移除] Rule 8 检查
                            # inval_cond = state.get('invalidation_condition') or '' 
                            # is_rule_8_trade = "Python Rule 8" in inval_cond
                            # if is_rule_8_trade:
                            #     continue
                                
                            price = tickers.get(symbol, {}).get('last')
                            entry = state.get('avg_entry_price')
                            side = state.get('side')
                            if not price or not entry:
                                continue
                            
                            is_profitable = (side == 'long' and price > entry) or (side == 'short' and price < entry)
                            
                            if is_profitable:
                                atr_15m = market_data.get(symbol, {}).get('15min_atr_14')
                                if not atr_15m or atr_15m <= 0:
                                    self.logger.warning(f"Main Loop (ATR Trail): 无法获取 {symbol} 的 15min_atr_14")
                                    continue
                                
                                ATR_TRAIL_MULTIPLIER = 2.0 
                                current_sl = state.get('ai_suggested_stop_loss', 0.0)
                                new_sl = 0.0

                                if side == 'long':
                                    new_sl = price - (ATR_TRAIL_MULTIPLIER * atr_15m)
                                    if new_sl > current_sl:
                                        sl_update_tasks_rule6.append(
                                            self.portfolio.update_position_rules(symbol, stop_loss=new_sl, reason="Main Loop: Rule 6 ATR Trail")
                                        )
                                elif side == 'short':
                                    new_sl = price + (ATR_TRAIL_MULTIPLIER * atr_15m)
                                    if new_sl < current_sl:
                                         sl_update_tasks_rule6.append(
                                            self.portfolio.update_position_rules(symbol, stop_loss=new_sl, reason="Main Loop: Rule 6 ATR Trail")
                                        )
                        
                        if sl_update_tasks_rule6:
                            self.logger.info(f"Main Loop (ATR Trail): 正在为 {len(sl_update_tasks_rule6)} 个 Rule 6 仓位更新追踪止损...")
                            await asyncio.gather(*sl_update_tasks_rule6, return_exceptions=True)

                    except Exception as e_atr_trail:
                        self.logger.error(f"Main Loop: Rule 6 ATR 追踪止损失败: {e_atr_trail}", exc_info=True)
                
                
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
                self.logger.critical(f"Main loop fatal error (outside sync/AI): {e}", exc_info=True); 
                await asyncio.sleep(60)
