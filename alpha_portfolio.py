# 文件: alpha_portfolio.py (V23.7 - 修复 live_partial_close 中的 get_size 错误)

import logging
import time
import json
import os
from collections import deque
import pandas as pd
from config import settings, futures_settings
from bark_notifier import send_bark_notification
from ccxt.base.errors import InsufficientFunds, ExchangeError
from typing import Optional

from exchange_client import ExchangeClient
from alpha_trade_logger import AlphaTradeLogger
from alpha_position_manager import AlphaPositionManager # 假设 V2.2

class AlphaPortfolio:
    FEE_RATE = 0.001 # 仅用于模拟盘
    MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK = 5.1 # V23.4 保留

    def __init__(self, exchange, symbols: list):
        # ... (V23.6 __init__ 代码不变) ...
        self.logger = logging.getLogger(self.__class__.__name__)
        if isinstance(exchange, ExchangeClient):
             self.client = exchange; self.exchange = exchange.exchange
        else:
             self.exchange = exchange; self.client = ExchangeClient(self.exchange)
        self.symbols = symbols
        self.is_live = settings.ALPHA_LIVE_TRADING
        self.mode_str = "[实盘]" if self.is_live else "[模拟]"
        self.trade_logger = AlphaTradeLogger(futures_settings.FUTURES_STATE_DIR)
        self.position_manager = AlphaPositionManager(futures_settings.FUTURES_STATE_DIR) # V2.2
        self.paper_cash: float = settings.ALPHA_PAPER_CAPITAL
        self.paper_equity: float = settings.ALPHA_PAPER_CAPITAL
        self.paper_positions: dict = {symbol: {} for symbol in symbols}
        self.paper_trade_history: list = []
        self.paper_equity_history: deque = deque(maxlen=30000)
        if self.is_live: self.cash, self.equity = 0.0, 0.0
        else: self.cash, self.equity = settings.ALPHA_PAPER_CAPITAL, settings.ALPHA_PAPER_CAPITAL
        self.state_file = os.path.join('data', 'alpha_portfolio_state_PAPER.json')
        if not self.is_live: self._load_paper_state()

    def _load_paper_state(self):
        # ... (V23.6 代码不变) ...
        if not os.path.exists(self.state_file): self.logger.info(f"{self.mode_str} 模拟盘状态文件不存在"); return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f: state = json.load(f)
            self.paper_cash = state.get('cash', settings.ALPHA_PAPER_CAPITAL)
            self.paper_equity = state.get('equity', settings.ALPHA_PAPER_CAPITAL)
            loaded_positions = state.get('positions', {})
            self.paper_positions = loaded_positions if isinstance(loaded_positions, dict) else {s: {} for s in self.symbols}
            loaded_trade_history = state.get('trade_history', [])
            self.paper_trade_history = loaded_trade_history if isinstance(loaded_trade_history, list) else []
            loaded_equity_history = state.get('equity_history', [])
            self.paper_equity_history = deque(loaded_equity_history if isinstance(loaded_equity_history, list) else [], maxlen=2000)
            self.cash = self.paper_cash; self.equity = self.paper_equity
            self.logger.warning("成功加载模拟投资组合状态。")
        except json.JSONDecodeError as e: self.logger.error(f"加载模拟状态失败：JSON 格式错误 - {e}", exc_info=False)
        except Exception as e: self.logger.error(f"加载模拟状态失败: {e}", exc_info=True)

    def _save_paper_state(self):
        # ... (V23.6 代码不变) ...
        state = {
            'cash': float(self.paper_cash) if self.paper_cash is not None else 0.0,
            'equity': float(self.paper_equity) if self.paper_equity is not None else 0.0,
            'positions': self.paper_positions if isinstance(self.paper_positions, dict) else {},
            'trade_history': self.paper_trade_history if isinstance(self.paper_trade_history, list) else [],
            'equity_history': list(self.paper_equity_history)
        }
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f: json.dump(state, f, indent=4, ensure_ascii=False)
        except TypeError as e: self.logger.error(f"保存模拟状态失败：类型错误 - {e}. State: {state}", exc_info=True)
        except Exception as e: self.logger.error(f"保存模拟状态失败: {e}", exc_info=True)

    async def sync_state(self):
        # ... (V23.6 代码不变, 包含 equity_history 诊断) ...
        try:
            if self.is_live:
                try:
                    balance = await self.client.fetch_balance(); usdt_balance = balance.get('USDT', {})
                    fetched_cash = usdt_balance.get('free'); fetched_equity = usdt_balance.get('total')
                    if fetched_cash is not None: self.cash = float(fetched_cash)
                    else: self.logger.error(f"{self.mode_str} sync: 无法获取 cash, 用上次值: {self.cash}")
                    if fetched_equity is not None: self.equity = float(fetched_equity)
                    else: self.logger.error(f"{self.mode_str} sync: 无法获取 equity, 用上次值: {self.equity}")
                    self.logger.debug(f"{self.mode_str} sync: Equity={self.equity:.2f}, Cash={self.cash:.2f}")
                    real_positions = await self.client.fetch_positions(self.symbols); exchange_open_symbols = set()
                    for pos in real_positions:
                        symbol = pos.get('symbol')
                        if symbol in self.symbols:
                            size_str = pos.get('contracts') or pos.get('contractSize'); size = float(size_str) if size_str else 0.0
                            side = pos.get('side').lower() if pos.get('side') else None
                            if abs(size) > 1e-9:
                                exchange_open_symbols.add(symbol)
                                if not self.position_manager.is_open(symbol):
                                    lev_val = pos.get('leverage'); lev_int = int(lev_val) if lev_val is not None and float(lev_val) > 0 else 1
                                    entry_str = pos.get('entryPrice') or pos.get('basePrice'); entry = float(entry_str) if entry_str else 0.0
                                    self.position_manager.open_position(symbol=symbol, side=side, entry_price=entry, size=abs(size), entry_fee=0.0, leverage=lev_int, stop_loss=None, take_profit=None, timestamp=int(pos.get('timestamp', time.time()*1000)), reason="live_sync", invalidation_condition=None) # 使用 None
                                    self.logger.warning(f"{self.mode_str} sync: 发现交易所持仓 {symbol}, 已同步到本地。")
                                else: self.logger.debug(f"{self.mode_str} sync: {symbol} 本地和交易所均存在。")
                    local_open_symbols = set(self.position_manager.get_all_open_positions().keys())
                    symbols_to_close_locally = local_open_symbols - exchange_open_symbols
                    for symbol in symbols_to_close_locally:
                         self.logger.warning(f"{self.mode_str} sync: 本地 {symbol} 在交易所已平仓，同步关闭。")
                         self.position_manager.close_position(symbol)
                    current_equity_to_append = self.equity
                    self.logger.debug(f"{self.mode_str} sync: 准备追加净值历史。 Equity: {current_equity_to_append}, Type: {type(current_equity_to_append)}")
                    # [V23.7 修复] 确保 pd 存在再调用 isnan
                    is_valid_equity = current_equity_to_append is not None and isinstance(current_equity_to_append, (int, float)) and (not pd or not pd.isna(current_equity_to_append))
                    if is_valid_equity:
                        history_entry = {'timestamp': time.time() * 1000, 'equity': float(current_equity_to_append)}
                        self.paper_equity_history.append(history_entry)
                        self.logger.debug(f"{self.mode_str} sync: 成功追加净值历史: {history_entry}")
                    else: self.logger.warning(f"{self.mode_str} sync: 跳过追加净值历史，Equity无效: {current_equity_to_append} (Type: {type(current_equity_to_append)})")
                except Exception as e: self.logger.critical(f"{self.mode_str} sync 失败 (实盘部分): {e}", exc_info=True)
            else: # 模拟盘
                unrealized_pnl = 0.0; total_margin = 0.0; tickers = {}
                try: tickers = await self.exchange.fetch_tickers(self.symbols)
                except Exception as e: self.logger.error(f"{self.mode_str} sync: 获取 Tickers 失败: {e}")
                for symbol, pos in list(self.paper_positions.items()):
                    if pos and isinstance(pos, dict) and pos.get('size', 0) > 0:
                        price = tickers.get(symbol, {}).get('last'); entry = pos.get('entry_price', 0.0); size = pos.get('size', 0.0); side = pos.get('side')
                        if price and isinstance(price, (int, float)) and price > 0:
                            if side=='long': pnl=(price-entry)*size
                            elif side=='short': pnl=(entry-price)*size
                            else: pnl=0.0
                            pos['unrealized_pnl'] = pnl; unrealized_pnl += pnl
                        else: unrealized_pnl += pos.get('unrealized_pnl', 0.0);
                        if not price: self.logger.warning(f"{self.mode_str} sync: 无法获取 {symbol} 价格，UPL 可能不准。")
                        total_margin += pos.get('margin', 0.0)
                    elif not isinstance(pos, dict): self.logger.error(f"{self.mode_str} sync: 无效模拟仓位 {symbol}: {pos}，清除。"); self.paper_positions[symbol] = {}
                cash_val = float(self.paper_cash) if self.paper_cash is not None else 0.0
                margin_val = float(total_margin) if total_margin is not None else 0.0
                upl_val = float(unrealized_pnl) if unrealized_pnl is not None else 0.0
                self.paper_equity = cash_val + margin_val + upl_val
                self.cash = self.paper_cash; self.equity = self.paper_equity
                current_equity_to_append = self.paper_equity
                self.logger.debug(f"{self.mode_str} sync: 准备追加净值历史。 Equity: {current_equity_to_append}, Type: {type(current_equity_to_append)}")
                # [V23.7 修复] 确保 pd 存在再调用 isnan
                is_valid_equity = current_equity_to_append is not None and isinstance(current_equity_to_append, (int, float)) and (not pd or not pd.isna(current_equity_to_append))
                if is_valid_equity:
                    history_entry = {'timestamp': time.time() * 1000, 'equity': float(current_equity_to_append)}
                    self.paper_equity_history.append(history_entry)
                    self.logger.debug(f"{self.mode_str} sync: 成功追加净值历史: {history_entry}")
                else: self.logger.warning(f"{self.mode_str} sync: 跳过追加净值历史，Equity无效: {current_equity_to_append} (Type: {type(current_equity_to_append)})")
                self._save_paper_state()
        except Exception as e: self.logger.critical(f"{self.mode_str} sync_state 顶层执行失败: {e}", exc_info=True)

# 修复：为实盘模式增加 UPL (未实现盈亏) 计算
    def get_state_for_prompt(self, tickers: dict = None):
        """
        [V23.8 修复]
        获取用于 AI Prompt 的状态。
        在实盘模式下，现在需要传入 tickers 字典来实时计算 UPL。
        """
        position_details = []
        
        if self.is_live:
            # --- [ 实盘模式 UPL 计算 ] ---
            if tickers is None: # 提供一个回退，以防万一
                tickers = {}
                self.logger.warning("get_state_for_prompt (live) 未收到 tickers! UPL 将丢失。")

            open_positions = self.position_manager.get_all_open_positions()
            for symbol, state in open_positions.items():
                
                # --- [新增 UPL 计算块] ---
                upl_str = "UPL=N/A"
                try:
                    # 从传入的 tickers 获取当前价格
                    current_price = tickers.get(symbol, {}).get('last')
                    if current_price and isinstance(current_price, (int, float)) and current_price > 0:
                        entry_price = state.get('avg_entry_price', 0.0)
                        size = state.get('total_size', 0.0)
                        side = state.get('side')
                        upl = 0.0

                        if side == 'long':
                            upl = (current_price - entry_price) * size
                        elif side == 'short':
                            upl = (entry_price - current_price) * size
                        
                        # 同时计算 PNL 百分比
                        margin = state.get('margin', 0.0) # V2.2 PositionManager 应该有这个
                        pnl_percent = (upl / margin) * 100 if margin > 0 else 0.0
                        upl_str = f"UPL={upl:.2f}$ ({pnl_percent:.2f}%)" # 包含 $ 符号和百分比
                    else:
                        upl_str = "UPL=NoPrice"
                except Exception as e:
                    self.logger.error(f"实盘 get_state_for_prompt UPL 计算失败 {symbol}: {e}")
                    upl_str = f"UPL=CalcErr"
                # --- [新增 UPL 计算块 结束] ---

                # 将 upl_str 添加到输出字符串中
                pos_str = ( f"- {symbol.split(':')[0]}: Side={state['side'].upper()}, Size={state['total_size']:.4f}, Entry={state['avg_entry_price']:.4f}, "
                            f"{upl_str}, " # <--- 新增的 UPL 信息
                            f"TP={state.get('ai_suggested_take_profit', 'N/A')}, SL={state.get('ai_suggested_stop_loss', 'N/A')}, "
                            f"Invalidation='{state.get('invalidation_condition', 'N/A')}'")
                position_details.append(pos_str)
            # --- [ 实盘模式修复结束 ] ---

        else:
            # --- [ 模拟盘模式 (不变) ] ---
            for symbol, pos in self.paper_positions.items():
                if pos and isinstance(pos, dict) and pos.get('size', 0) > 0:
                    # 模拟盘的 UPL 在 sync_state 中已算好，直接使用
                    pos_str = ( f"- {symbol.split(':')[0]}: Side={pos['side'].upper()}, Size={pos['size']:.4f}, Entry={pos['entry_price']:.4f}, "
                                f"UPL={pos.get('unrealized_pnl', 0.0):.2f}, TP={pos.get('take_profit', 'N/A')}, SL={pos.get('stop_loss', 'N/A')}, "
                                f"Invalidation='{pos.get('invalidation_condition', 'N/A')}'")
                    position_details.append(pos_str)
        
        if not position_details: position_details.append("No open positions.")
        
        initial_capital_for_calc = settings.ALPHA_LIVE_INITIAL_CAPITAL if self.is_live else settings.ALPHA_PAPER_CAPITAL
        performance_percent_str = "N/A (Invalid Initial)"
        
        if initial_capital_for_calc > 0:
            current_equity_val = float(self.equity) if self.equity is not None else 0.0
            performance_percent = (current_equity_val / initial_capital_for_calc - 1) * 100
            performance_percent_str = f"{performance_percent:.2f}%"
            
        return { "account_value_usd": f"{float(self.equity):.2f}" if self.equity is not None else "0.00",
                 "cash_usd": f"{float(self.cash):.2f}" if self.cash is not None else "0.00",
                 "performance_percent": performance_percent_str,
                 "open_positions": "\n".join(position_details)}
    # --- [ 修复结束 ] ---

    async def live_open(self, symbol, side, size, leverage, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        # ... (V23.6 代码不变, 包含最终名义价值检查 5.1U) ...
        is_adding = self.position_manager.is_open(symbol); action_type = "加仓" if is_adding else "开新仓"
        self.logger.warning(f"!!! {self.mode_str} AI 请求 {action_type}: {side.upper()} {size} {symbol} !!!")
        current_pos_state = None
        if is_adding:
            current_pos_state = self.position_manager.get_position_state(symbol)
            if not current_pos_state or current_pos_state.get('side') != side:
                self.logger.error(f"!!! {self.mode_str} {action_type} 失败: 方向 ({side}) 与现有 ({current_pos_state.get('side') if current_pos_state else 'N/A'}) 不符。将覆盖。")
                is_adding = False; current_pos_state = None
        try:
            raw_exchange = self.client.exchange
            if not raw_exchange.markets: await self.client.load_markets()
            market = raw_exchange.markets.get(symbol);
            if not market: raise ValueError(f"无市场信息 {symbol}")
            ticker = await self.client.fetch_ticker(symbol); current_price = ticker.get('last')
            if not current_price or current_price <= 0: raise ValueError(f"无有效价格 {symbol}")
            required_margin_initial = (size * current_price) / leverage
            if required_margin_initial <= 0: raise ValueError("保证金无效 (<= 0)")
            max_allowed_margin = self.cash * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
            if max_allowed_margin <= 0: raise ValueError(f"最大允许保证金无效 (<= 0), 可用现金: {self.cash}")
            adjusted_size = size; required_margin_final = required_margin_initial
            if required_margin_initial > max_allowed_margin:
                self.logger.warning(f"!!! {self.mode_str} {action_type} 保证金超限 ({required_margin_initial:.2f} > {max_allowed_margin:.2f})，缩减 !!!")
                adj_size_raw = (max_allowed_margin * leverage) / current_price
                adjusted_size = float(raw_exchange.amount_to_precision(symbol, adj_size_raw))
                min_amount = market.get('limits', {}).get('amount', {}).get('min')
                if min_amount is not None and adjusted_size < min_amount:
                     self.logger.error(f"!!! {self.mode_str} {action_type} 缩减后过小 ({adjusted_size} < {min_amount})，取消 !!!")
                     await send_bark_notification(f"⚠️ {self.mode_str} AI {action_type} 被拒", f"品种: {symbol}\n原因: 缩减后过小"); return
                self.logger.warning(f"缩减后 Size: {adjusted_size}")
                required_margin_final = (adjusted_size * current_price) / leverage
            final_notional_value = adjusted_size * current_price
            if final_notional_value < self.MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK:
                self.logger.error(f"!!! {self.mode_str} {action_type} 最终名义价值检查失败 !!!")
                self.logger.error(f"最终名义价值 {final_notional_value:.4f} USDT < 阈值 {self.MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK} USDT。取消。")
                await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 最终名义价值过低 (<{self.MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK} USDT)"); return
            estimated_fee = adjusted_size * current_price * market.get('taker', self.FEE_RATE)
            if self.cash < required_margin_final + estimated_fee:
                 self.logger.error(f"!!! {self.mode_str} {action_type} 现金不足 !!! (需 {required_margin_final + estimated_fee:.2f}, 可用 {self.cash:.2f})")
                 await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 现金不足"); return
            await self.client.set_margin_mode(futures_settings.FUTURES_MARGIN_MODE, symbol)
            await self.client.set_leverage(leverage, symbol)
            exchange_side = 'BUY' if side == 'long' else 'SELL'
            order_result = await self.client.create_market_order(symbol, exchange_side, adjusted_size)
            entry_price = float(order_result.get('average', order_result.get('price')))
            if not entry_price or entry_price <= 0: entry_price = float(order_result['price'])
            filled_size = float(order_result['filled']); timestamp = int(order_result['timestamp'])
            fee = await self._parse_fee_from_order(order_result, symbol)
            success = False
            if is_adding: success = self.position_manager.add_entry(symbol=symbol, entry_price=entry_price, size=filled_size, entry_fee=fee, leverage=leverage, stop_loss=stop_loss, take_profit=take_profit, timestamp=timestamp, invalidation_condition=invalidation_condition)
            else: self.position_manager.open_position(symbol=symbol, side=side, entry_price=entry_price, size=filled_size, entry_fee=fee, leverage=leverage, stop_loss=stop_loss, take_profit=take_profit, timestamp=timestamp, reason=reason, invalidation_condition=invalidation_condition); success = True # open_position 总是成功（覆盖）
            if success:
                 self.logger.warning(f"!!! {self.mode_str} {action_type} 成功: {side.upper()} {filled_size} {symbol} @ {entry_price} (Fee: {fee}) | AI原因: {reason}")
                 title = f"📈 {self.mode_str} AI {action_type}: {side.upper()} {symbol.split('/')[0]}"
                 final_pos_state = self.position_manager.get_position_state(symbol)
                 final_avg = final_pos_state.get('avg_entry_price', entry_price) if final_pos_state else entry_price
                 final_size = final_pos_state.get('total_size', filled_size) if final_pos_state else filled_size
                 body = f"价格: {entry_price:.4f}\n数量: {filled_size}\n杠杆: {leverage}x\n手续费: {fee:.4f}\n保证金: {required_margin_final:.2f}\nTP/SL: {take_profit}/{stop_loss}"
                 if is_adding: body += f"\n新均价: {final_avg:.4f}\n总数量: {final_size:.4f}"
                 body += f"\nAI原因: {reason}";
                 if adjusted_size != size: body += f"\n(请求 {size} 缩减至 {filled_size})"
                 await send_bark_notification(title, body); await self.sync_state()
            else: raise RuntimeError(f"{action_type} 失败但未抛异常")
        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} {action_type} 失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: self.logger.error(f"!!! {self.mode_str} {action_type} 失败: {e}", exc_info=True); await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n错误: {e}")


    async def live_partial_close(self, symbol: str, size_percent: Optional[float] = None, size_absolute: Optional[float] = None, reason: str = "N/A"):
        """[实盘] 部分平仓"""
        self.logger.warning(f"!!! {self.mode_str} AI 请求部分平仓: {symbol} | %: {size_percent} | Abs: {size_absolute} | 原因: {reason} !!!")

        pos_state = self.position_manager.get_position_state(symbol) # 获取前会 recalculate
        if not pos_state or pos_state.get('total_size', 0) <= 0:
            self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 本地无有效持仓 {symbol}。")
            return

        current_total_size = pos_state['total_size']
        size_to_close = 0.0
        if size_percent is not None and 0 < size_percent < 1: size_to_close = current_total_size * size_percent
        elif size_absolute is not None and 0 < size_absolute <= current_total_size + 1e-9: # 允许略大于当前size
             size_to_close = min(size_absolute, current_total_size) # 不能超过当前size
        else: self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 无效数量参数..."); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n原因: 无效数量参数"); return

        # 检查交易所精度和最小量
        try:
             raw_exchange = self.client.exchange
             if not raw_exchange.markets: await self.client.load_markets()
             market = raw_exchange.markets.get(symbol)
             if not market: raise ValueError(f"无法找到市场信息 {symbol}")
             size_to_close = float(raw_exchange.amount_to_precision(symbol, size_to_close))
             min_amount = market.get('limits', {}).get('amount', {}).get('min')
             if min_amount is not None and size_to_close < min_amount:
                 # 如果计算出的量小于最小量，但大于0，尝试平掉最小量（如果接近全平则全平）
                 if size_to_close > 1e-9:
                      if current_total_size - size_to_close < min_amount: # 如果剩余量也小于最小量，不如全平
                           self.logger.warning(f"{self.mode_str} 部分平仓 {symbol}: 计算量 {size_to_close} < 最小量 {min_amount} 且剩余量也小，转为全平。")
                           await self.live_close(symbol, reason=f"{reason} (转为全平)")
                           return
                      else:
                           self.logger.warning(f"{self.mode_str} 部分平仓 {symbol}: 计算量 {size_to_close} < 最小量 {min_amount}，尝试平最小量。")
                           size_to_close = min_amount
                 else: # 如果计算量本身就接近0，则取消
                      self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 数量过小 ({size_to_close})"); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n原因: 平仓数量过小"); return
             if size_to_close <= 0: self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 计算数量为 0"); return
        except Exception as e: self.logger.error(f"!!! {self.mode_str} 部分平仓失败 (检查数量时出错): {e}", exc_info=True); return

        try:
            internal_side = pos_state['side']; avg_entry_price = pos_state['avg_entry_price']
            open_fee_total = pos_state['total_entry_fee']; leverage = pos_state.get('leverage', 0)
            total_margin = pos_state.get('margin', 0.0) # V2.2 _recalculate 会计算
            margin_per_unit = total_margin / current_total_size if current_total_size > 0 else 0
            margin_for_this_part = margin_per_unit * size_to_close

            exchange_close_side = 'SELL' if internal_side == 'long' else 'BUY'
            params = {'reduceOnly': True}
            order_result = await self.client.create_market_order(symbol, exchange_close_side, size_to_close, params=params)

            exit_price = float(order_result.get('average', order_result.get('price')))
            if not exit_price or exit_price <= 0: exit_price = float(order_result['price'])
            filled_size = float(order_result['filled']); timestamp = int(order_result['timestamp'])

            # [V23.6 修改] 调用 async _parse_fee_from_order
            close_fee = await self._parse_fee_from_order(order_result, symbol)

            # 对应开仓费（按比例）
            open_fee_for_this_part = (open_fee_total / current_total_size) * filled_size if current_total_size > 0 else 0

            if internal_side == 'long': gross_pnl_part = (exit_price - avg_entry_price) * filled_size
            else: gross_pnl_part = (avg_entry_price - exit_price) * filled_size
            net_pnl_part = gross_pnl_part - open_fee_for_this_part - close_fee

            # --- [V23.6 核心修改] 增加保证金计算验证 ---
            # 基于订单成交额计算的近似保证金
            order_notional = filled_size * exit_price # 或使用 avg_entry_price 更接近真实保证金占用？这里用成交价估算
            margin_calc_by_order = order_notional / leverage if leverage > 0 else 0.0

            trade_data = {
                'symbol': symbol, 'side': internal_side, 'entry_price': avg_entry_price,
                'exit_price': exit_price, 'size': filled_size,
                'net_pnl': net_pnl_part, 'fees': open_fee_for_this_part + close_fee,
                'margin': margin_for_this_part, # 记录基于 PositionManager 计算的保证金释放量
                'margin_calc_by_order': margin_calc_by_order, # 新增：记录基于订单估算的保证金
                'leverage': leverage,
                'open_reason': pos_state.get('entry_reason', 'N/A'), 'close_reason': reason,
                'timestamp': timestamp, 'partial': True
            }
            # --- [V23.6 修改结束] ---

            self.trade_logger.record_trade(trade_data)
            success = self.position_manager.reduce_position(symbol, filled_size)

            if success:
                 # --- [V23.7 核心修复] ---
                 # 在 reduce_position 后，重新获取状态以获得正确的剩余 size
                 updated_pos_state = self.position_manager.get_position_state(symbol)
                 remaining_size = updated_pos_state.get('total_size', 0.0) if updated_pos_state else 0.0
                 # --- [修复结束] ---

                 self.logger.warning(f"!!! {self.mode_str} 部分平仓成功: {symbol} | 平掉 {filled_size} @ {exit_price:.4f} (Fee: {close_fee}) | 本次净盈亏: {net_pnl_part:.2f} USDT | 剩余 {remaining_size:.8f} | 原因: {reason}") # 使用 .8f 显示剩余量
                 pnl_prefix = "盈利" if net_pnl_part >= 0 else "亏损"; title = f"💰 {self.mode_str} AI 部分平仓: {pnl_prefix} {abs(net_pnl_part):.2f} USDT"
                 body = (f"品种: {symbol.split('/')[0]}\n方向: {internal_side.upper()}\n平仓价格: {exit_price:.4f}\n平仓数量: {filled_size}\n手续费: {close_fee:.4f}\n剩余数量: {remaining_size:.8f}\n原因: {reason}") # 使用 .8f
                 await send_bark_notification(title, body); await self.sync_state()
            else: raise RuntimeError("position_manager.reduce_position 返回失败")
        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} 部分平仓失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: self.logger.error(f"!!! {self.mode_str} 部分平仓失败: {e}", exc_info=True); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n错误: {e}")


    async def live_close(self, symbol, reason: str = "N/A"):
        """[实盘] 全平指定symbol的仓位"""
        self.logger.warning(f"!!! {self.mode_str} 正在尝试(全)平仓: {symbol} | 原因: {reason} !!!")
        pos_state = self.position_manager.get_position_state(symbol) # 获取前会 recalculate
        if not pos_state or pos_state.get('total_size', 0) <= 0:
            self.logger.error(f"!!! {self.mode_str} (全)平仓失败: 本地无有效持仓 {symbol}。")
            return

        try:
            internal_side = pos_state['side']; size_to_close = pos_state['total_size']
            avg_entry_price = pos_state['avg_entry_price']; open_fee_total = pos_state['total_entry_fee']
            leverage = pos_state.get('leverage', 0); margin_to_record = pos_state.get('margin', 0.0) # V2.2 _recalculate 会计算
            entry_reason = pos_state.get('entry_reason', 'N/A')

            exchange_close_side = 'SELL' if internal_side == 'long' else 'BUY'
            params = {'reduceOnly': True}
            order_result = await self.client.create_market_order(symbol, exchange_close_side, size_to_close, params=params)

            exit_price = float(order_result.get('average', order_result.get('price')))
            if not exit_price or exit_price <= 0: exit_price = float(order_result['price'])
            filled_size = float(order_result['filled']); timestamp = int(order_result['timestamp'])

            # [V23.6 修改] 调用 async _parse_fee_from_order
            close_fee = await self._parse_fee_from_order(order_result, symbol)

            if internal_side == 'long': gross_pnl = (exit_price - avg_entry_price) * filled_size
            else: gross_pnl = (avg_entry_price - exit_price) * filled_size
            net_pnl = gross_pnl - open_fee_total - close_fee

            # --- [V23.6 核心修改] 增加保证金计算验证 ---
            order_notional = filled_size * exit_price
            margin_calc_by_order = order_notional / leverage if leverage > 0 else 0.0

            trade_data = {
                'symbol': symbol, 'side': internal_side, 'entry_price': avg_entry_price,
                'exit_price': exit_price, 'size': filled_size,
                'net_pnl': net_pnl, 'fees': open_fee_total + close_fee,
                'margin': margin_to_record, # 记录基于 PositionManager 计算的保证金
                'margin_calc_by_order': margin_calc_by_order, # 新增：记录基于订单估算的保证金
                'leverage': leverage,
                'open_reason': entry_reason, 'close_reason': reason,
                'timestamp': timestamp, 'partial': False
            }
            # --- [V23.6 修改结束] ---

            self.trade_logger.record_trade(trade_data)
            self.position_manager.close_position(symbol)

            self.logger.warning(f"!!! {self.mode_str} (全)平仓成功: {symbol} @ {exit_price:.4f} (Fee: {close_fee}), 净盈亏: {net_pnl:.2f} USDT | 原因: {reason}")
            pnl_prefix = "盈利" if net_pnl >= 0 else "亏损"
            title = f"📉 {self.mode_str} AI (全)平仓: {pnl_prefix} {abs(net_pnl):.2f} USDT"
            body = f"品种: {symbol.split('/')[0]}\n方向: {internal_side.upper()}\n平仓价格: {exit_price:.4f}\n手续费: {close_fee:.4f}\n原因: {reason}"
            await send_bark_notification(title, body); await self.sync_state()

        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} (全)平仓失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI (全)平仓失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: self.logger.error(f"!!! {self.mode_str} (全)平仓失败: {e}", exc_info=True); await send_bark_notification(f"❌ {self.mode_str} AI (全)平仓失败", f"品种: {symbol}\n错误: {e}")


    # --- paper_open, paper_close, paper_partial_close 保持 V23.3 不变 (已修复 margin/fees 记录) ---
    async def paper_open(self, symbol, side, size, price, leverage, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        # ... (V23.3 代码不变) ...
        action_type = "加仓" if self.paper_positions.get(symbol) and self.paper_positions[symbol].get('side') == side else "开新仓"
        margin_required = (size * price) / leverage; fee = size * price * self.FEE_RATE
        if self.paper_cash < (margin_required + fee): self.logger.error(f"{self.mode_str} {action_type} 失败: 资金不足"); return
        self.paper_cash -= (margin_required + fee); existing_pos = self.paper_positions.get(symbol)
        if existing_pos and isinstance(existing_pos, dict) and existing_pos.get('side') == side:
            old_size = existing_pos.get('size', 0.0); old_value = old_size * existing_pos.get('entry_price', price); new_value = size * price
            existing_pos['size'] = old_size + size; existing_pos['entry_price'] = (old_value + new_value) / existing_pos['size'] if existing_pos['size'] > 0 else price
            existing_pos['margin'] = existing_pos.get('margin', 0.0) + margin_required; existing_pos['take_profit'] = take_profit; existing_pos['stop_loss'] = stop_loss; existing_pos['invalidation_condition'] = invalidation_condition
            self.logger.warning(f"{self.mode_str} {action_type}: {side.upper()} {size:.4f} {symbol} @ {price:.4f} | 新均价: {existing_pos['entry_price']:.4f}"); title=f"🔼 {self.mode_str} AI {action_type}"; body=f"价格: {price:.4f}\n新均价: {existing_pos['entry_price']:.4f}\nTP/SL: {take_profit}/{stop_loss}"; await send_bark_notification(title, body)
        else:
            if existing_pos and isinstance(existing_pos, dict): self.logger.warning(f"{self.mode_str} 反向开仓 {symbol}，覆盖。")
            self.paper_positions[symbol] = {'side': side, 'size': size, 'entry_price': price, 'leverage': leverage, 'margin': margin_required, 'unrealized_pnl': 0.0, 'open_reason': reason, 'take_profit': take_profit, 'stop_loss': stop_loss, 'invalidation_condition': invalidation_condition}
            self.logger.warning(f"{self.mode_str} {action_type}: {side.upper()} {size:.4f} {symbol} @ {price:.4f}"); title=f"📈 {self.mode_str} AI {action_type}"; body=f"价格: {price:.4f}\n杠杆: {leverage}x\nTP/SL: {take_profit}/{stop_loss}\n原因: {reason}"; await send_bark_notification(title, body)
        await self.sync_state()

    async def paper_close(self, symbol, price, reason: str = "N/A"):
        # ... (V23.3 代码不变) ...
        pos = self.paper_positions.pop(symbol, None)
        if not pos or not isinstance(pos, dict) or pos.get('size', 0) <= 0: self.logger.error(f"{self.mode_str} (全)平仓失败: 未找到 {symbol} 持仓。"); return
        entry_price = pos.get('entry_price', 0.0); size = pos.get('size', 0.0); leverage = pos.get('leverage'); margin_recorded = pos.get('margin', 0.0)
        open_fee = size * entry_price * self.FEE_RATE; close_fee = size * price * self.FEE_RATE; total_fees = open_fee + close_fee
        if pos.get('side') == 'long': gross_pnl = (price - entry_price) * size
        elif pos.get('side') == 'short': gross_pnl = (entry_price - price) * size
        else: gross_pnl = 0.0; self.logger.error(f"{self.mode_str} 平仓 {symbol} 方向无效: {pos.get('side')}")
        net_pnl = gross_pnl - total_fees; self.paper_cash += (margin_recorded + net_pnl)
        trade_record = {'symbol': symbol, 'side': pos.get('side'), 'entry_price': entry_price, 'exit_price': price, 'size': size, 'net_pnl': net_pnl, 'fees': total_fees, 'margin': margin_recorded, 'leverage': leverage, 'open_reason': pos.get('open_reason', 'N/A'), 'close_reason': reason, 'timestamp': time.time() * 1000, 'partial': False}
        self.paper_trade_history.append(trade_record)
        self.logger.warning(f"{self.mode_str} (全)平仓: {symbol} @ {price:.4f}, 净盈亏: {net_pnl:.2f} USDT | 原因: {reason}")
        pnl_prefix = "盈利" if net_pnl >= 0 else "亏损"; title = f"📉 {self.mode_str} AI 平仓: {pnl_prefix} {abs(net_pnl):.2f} USDT"; body = f"品种: {symbol.split('/')[0]}\n方向: {pos.get('side', 'N/A').upper()}\n平仓价: {price:.4f}\n原因: {reason}"; await send_bark_notification(title, body)
        await self.sync_state()

    async def paper_partial_close(self, symbol: str, price: float, size_percent: Optional[float] = None, size_absolute: Optional[float] = None, reason: str = "N/A"):
        # ... (V23.3 代码不变) ...
        pos = self.paper_positions.get(symbol)
        if not pos or not isinstance(pos, dict) or pos.get('size', 0) <= 0: self.logger.error(f"{self.mode_str} 部分平仓失败: 未找到 {symbol} 持仓。"); return
        current_total_size = pos.get('size', 0.0); current_total_margin = pos.get('margin', 0.0); size_to_close = 0.0
        if size_percent is not None and 0 < size_percent < 1: size_to_close = current_total_size * size_percent
        elif size_absolute is not None and 0 < size_absolute < current_total_size: size_to_close = size_absolute
        else: self.logger.error(f"{self.mode_str} 部分平仓失败: 无效数量参数"); return
        if size_to_close <= 0: self.logger.error(f"{self.mode_str} 部分平仓失败: 计算数量为 0"); return
        entry_price = pos.get('entry_price', 0.0); leverage = pos.get('leverage'); margin_per_unit = current_total_margin / current_total_size if current_total_size > 0 else 0
        margin_to_release = margin_per_unit * size_to_close; open_fee_per_unit = (entry_price * self.FEE_RATE); open_fee_for_part = open_fee_per_unit * size_to_close; close_fee_for_part = size_to_close * price * self.FEE_RATE; total_fees_for_part = open_fee_for_part + close_fee_for_part
        if pos.get('side') == 'long': gross_pnl_part = (price - entry_price) * size_to_close
        elif pos.get('side') == 'short': gross_pnl_part = (entry_price - price) * size_to_close
        else: gross_pnl_part = 0.0; self.logger.error(f"{self.mode_str} 部分平仓 {symbol} 方向无效")
        net_pnl_part = gross_pnl_part - total_fees_for_part; self.paper_cash += (margin_to_release + net_pnl_part)
        trade_record = {'symbol': symbol, 'side': pos.get('side'), 'entry_price': entry_price, 'exit_price': price, 'size': size_to_close, 'net_pnl': net_pnl_part, 'fees': total_fees_for_part, 'margin': margin_to_release, 'leverage': leverage, 'open_reason': pos.get('open_reason', 'N/A'), 'close_reason': reason, 'timestamp': time.time() * 1000, 'partial': True}
        self.paper_trade_history.append(trade_record)
        pos['size'] = current_total_size - size_to_close; pos['margin'] = current_total_margin - margin_to_release
        if pos['size'] < 1e-9: self.logger.warning(f"{self.mode_str} 部分平仓后 {symbol} 剩余过小，视为全平。"); self.paper_positions[symbol] = {}
        else: self.logger.warning(f"{self.mode_str} 部分平仓: {symbol} | 平掉 {size_to_close:.4f} @ {price:.4f} | 本次净盈亏: {net_pnl_part:.2f} | 剩余: {pos['size']:.4f}"); pnl_prefix = "盈利" if net_pnl_part >= 0 else "亏损"; title = f"💰 {self.mode_str} AI 部分平仓: {pnl_prefix} {abs(net_pnl_part):.2f}"; body = (f"品种:{symbol.split('/')[0]}\n方向:{pos.get('side','N/A').upper()}\n平仓价:{price:.4f}\n数量:{size_to_close:.4f}\n剩余:{pos['size']:.4f}\n原因:{reason}"); await send_bark_notification(title, body)
        await self.sync_state()

    # --- [V23.6 核心修复] _parse_fee_from_order (async + BNB 转换) ---
    async def _parse_fee_from_order(self, order_result: dict, symbol: str) -> float:
        """从交易所订单结果中解析手续费 (尝试转换为 USDT 等值)"""
        fees_paid_usdt = 0.0
        if not order_result: return fees_paid_usdt

        self.logger.debug(f"Fee Parsing Debug: Raw order_result for {symbol}: {order_result}")

        fee_currency = None
        fee_cost = None

        # 优先尝试 'fee' 结构
        if 'fee' in order_result and isinstance(order_result['fee'], dict):
            fee_info = order_result['fee']
            if 'cost' in fee_info and 'currency' in fee_info:
                try: fee_cost = float(fee_info['cost']); fee_currency = fee_info['currency']; self.logger.debug(f"Fee Parsing: Found 'fee': {fee_cost} {fee_currency}")
                except (ValueError, TypeError): self.logger.warning(f"无法解析 'fee.cost': {fee_info}"); fee_cost = None
        # 其次尝试 'fees' 列表
        elif 'fees' in order_result and isinstance(order_result['fees'], list) and len(order_result['fees']) > 0:
            first_valid_fee = next((f for f in order_result['fees'] if f and 'cost' in f and 'currency' in f), None)
            if first_valid_fee:
                 try:
                    fee_cost = float(first_valid_fee['cost']); fee_currency = first_valid_fee['currency']
                    if len(order_result['fees']) > 1: self.logger.warning(f"{symbol} 含多个费用条目，仅处理第一个: {order_result['fees']}")
                    self.logger.debug(f"Fee Parsing: Found 'fees' list: {fee_cost} {fee_currency}")
                 except (ValueError, TypeError) as e: self.logger.warning(f"解析 'fees'列表出错: {e}"); fee_cost = None
            else: self.logger.warning(f"{symbol} 'fees'列表为空或缺字段: {order_result['fees']}")

        # --- 处理解析出的费用 ---
        if fee_cost is not None and fee_currency is not None:
            if fee_currency == 'USDT':
                fees_paid_usdt = fee_cost
                self.logger.debug(f"Fee Parsing: Fee is USDT: {fees_paid_usdt}")
            elif fee_currency == 'BNB':
                self.logger.warning(f"检测到 {symbol} 手续费以 BNB 支付: {fee_cost} BNB。尝试获取 BNB/USDT 价格进行转换...")
                try:
                    # --- 获取实时 BNB/USDT 价格 ---
                    bnb_ticker = await self.client.fetch_ticker('BNB/USDT')
                    bnb_price = bnb_ticker.get('last')
                    if bnb_price and bnb_price > 0:
                        fees_paid_usdt = fee_cost * bnb_price
                        self.logger.warning(f"BNB 手续费已转换为 USDT: {fee_cost} BNB * {bnb_price} USD/BNB = {fees_paid_usdt:.4f} USDT")
                    else:
                        self.logger.error("无法获取有效的 BNB/USDT 价格，BNB 手续费将记录为 0 USDT。")
                        fees_paid_usdt = 0.0
                except ExchangeError as e:
                     self.logger.error(f"获取 BNB/USDT ticker 时交易所错误: {e}。BNB 手续费将记录为 0 USDT。")
                     fees_paid_usdt = 0.0
                except Exception as e:
                    self.logger.error(f"获取 BNB/USDT 价格或转换时发生意外错误: {e}。BNB 手续费将记录为 0 USDT。", exc_info=True)
                    fees_paid_usdt = 0.0
            else: # 其他币种
                self.logger.warning(f"检测到 {symbol} 手续费以非 USDT/BNB 币种支付: {fee_cost} {fee_currency}。将记录为 0 USDT。")
                fees_paid_usdt = 0.0 # 暂不处理其他币种转换
        else:
            self.logger.warning(f"未能从 {symbol} 订单结果解析费用。将使用 0.0 USDT。")

        return fees_paid_usdt
    # --- [V23.6 修复结束] ---

    # --- equity_history, trade_history properties 保持 V23.3 不变 ---
    @property
    def equity_history(self):
        return self.paper_equity_history

    @property
    def trade_history(self):
        if self.is_live: return self.trade_logger.get_history()
        else: return self.paper_trade_history

    # --- update_position_rules 保持 V23.3 不变 ---
    async def update_position_rules(self, symbol: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None, invalidation_condition: Optional[str] = None, reason: str = "AI update"):
        # ... (V23.3 代码不变) ...
        if self.is_live:
            success = self.position_manager.update_rules(symbol, stop_loss, take_profit, invalidation_condition) # 调用 V2.2 的方法
            if success: self.logger.info(f"{self.mode_str} 更新规则 {symbol}: SL={stop_loss}, TP={take_profit}, Inval='{invalidation_condition}'. R: {reason}")
            else: self.logger.error(f"{self.mode_str} 更新规则 {symbol} 失败 (无持仓?)")
        else:
            pos = self.paper_positions.get(symbol)
            if pos and isinstance(pos, dict) and pos.get('size', 0) > 0:
                if stop_loss is not None: pos['stop_loss'] = stop_loss
                if take_profit is not None: pos['take_profit'] = take_profit
                if invalidation_condition is not None: pos['invalidation_condition'] = invalidation_condition
                self.logger.info(f"{self.mode_str} 更新规则 {symbol}: SL={pos.get('stop_loss')}, TP={pos.get('take_profit')}, Inval='{pos.get('invalidation_condition')}''. R: {reason}")
                await self.sync_state()
            else: self.logger.error(f"{self.mode_str} 更新规则 {symbol} 失败 (无持仓?)")
