# 文件: alpha_portfolio.py (V23 - 修复同步时的 NoneType 错误)

import logging
import time
import json
import os
from collections import deque
from config import settings, futures_settings
from bark_notifier import send_bark_notification
from ccxt.base.errors import InsufficientFunds
from typing import Optional # 增加 Optional

# 导入实盘工具箱, 日志器, 持仓管理器 V2
from exchange_client import ExchangeClient
from alpha_trade_logger import AlphaTradeLogger
from alpha_position_manager import AlphaPositionManager # 使用 V2

class AlphaPortfolio:
    FEE_RATE = 0.001 # 仅用于模拟盘

    def __init__(self, exchange, symbols: list):
        self.logger = logging.getLogger(self.__class__.__name__)
        # 保证 self.exchange 是原始 ccxt 对象
        if isinstance(exchange, ExchangeClient):
             self.client = exchange
             self.exchange = exchange.exchange
        else:
             self.exchange = exchange
             self.client = ExchangeClient(self.exchange)
        self.symbols = symbols
        self.is_live = settings.ALPHA_LIVE_TRADING

        # 实盘组件 (使用 V2 持仓管理器)
        self.trade_logger = AlphaTradeLogger(futures_settings.FUTURES_STATE_DIR)
        self.position_manager = AlphaPositionManager(futures_settings.FUTURES_STATE_DIR) # 使用 V2

        # --- 模拟盘组件 ---
        self.paper_cash: float = settings.ALPHA_PAPER_CAPITAL
        self.paper_equity: float = settings.ALPHA_PAPER_CAPITAL
        self.paper_positions: dict = {symbol: {} for symbol in symbols} # 模拟盘结构不变
        self.paper_trade_history: list = []
        self.paper_equity_history: deque = deque(maxlen=2000)
        # --- 模拟盘组件结束 ---

        # 根据模式初始化 equity 和 cash
        if self.is_live:
            self.cash: float = 0.0
            self.equity: float = 0.0
        else:
            self.cash: float = settings.ALPHA_PAPER_CAPITAL
            self.equity: float = settings.ALPHA_PAPER_CAPITAL

        self.state_file = os.path.join('data', 'alpha_portfolio_state_PAPER.json')
        if not self.is_live:
            self._load_paper_state()

    # --- _load_paper_state, _save_paper_state 保持不变 ---
    def _load_paper_state(self):
        # ... (加载模拟盘状态逻辑不变) ...
        if not os.path.exists(self.state_file): return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f: state = json.load(f)
            self.paper_cash = state.get('cash', settings.ALPHA_PAPER_CAPITAL)
            self.paper_equity = state.get('equity', settings.ALPHA_PAPER_CAPITAL)
            self.paper_positions = state.get('positions', self.paper_positions) # 模拟盘结构
            self.paper_trade_history = state.get('trade_history', [])
            self.paper_equity_history = deque(state.get('equity_history', []), maxlen=2000)
            self.cash = self.paper_cash
            self.equity = self.paper_equity
            self.logger.warning("成功从文件加载模拟投资组合状态。")
        except Exception as e:
            self.logger.error(f"加载模拟投资组合状态失败: {e}", exc_info=True)

    def _save_paper_state(self):
        # ... (保存模拟盘状态逻辑不变) ...
        state = {'cash': self.paper_cash, 'equity': self.paper_equity, 'positions': self.paper_positions, 'trade_history': self.paper_trade_history, 'equity_history': list(self.paper_equity_history)}
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f: json.dump(state, f, indent=4)
        except Exception as e:
            self.logger.error(f"保存模拟投资组合状态失败: {e}", exc_info=True)

    # --- [核心修改] sync_state 修复 leverage=None 问题 ---
    async def sync_state(self):
        """根据实盘或模拟盘同步状态"""
        if self.is_live:
            try:
                # 1. [实盘] 同步余额 (不变)
                balance = await self.client.fetch_balance()
                usdt_balance = balance.get('USDT', {})
                self.cash = usdt_balance.get('free', 0.0)
                self.equity = usdt_balance.get('total', 0.0)
                self.logger.debug(f"实盘同步：从交易所获取 Equity={self.equity:.2f}, Cash={self.cash:.2f}")

                # 2. [实盘] 同步持仓 (不变)
                real_positions = await self.client.fetch_positions(self.symbols)

                # 3. [实盘] 更新本地 AlphaPositionManager
                exchange_open_symbols = set()
                for pos in real_positions:
                    symbol = pos.get('symbol')
                    if symbol in self.symbols:
                        size = float(pos.get('contracts', 0.0))
                        side = pos.get('side').lower() if pos.get('side') else None
                        if size != 0:
                            exchange_open_symbols.add(symbol)
                            if not self.position_manager.is_open(symbol):
                                # --- [核心修复] 安全处理 leverage ---
                                leverage_val = pos.get('leverage')
                                leverage_int = int(leverage_val) if leverage_val is not None else 0
                                # --- [核心修复结束] ---
                                self.position_manager.open_position( # 调用 V2 open_position
                                    symbol=symbol, side=side, entry_price=float(pos.get('entryPrice', 0.0)),
                                    size=abs(size), entry_fee=0.0,
                                    leverage=leverage_int, # 使用修复后的值
                                    stop_loss=0.0, take_profit=0.0,
                                    timestamp=int(pos.get('timestamp', time.time() * 1000)),
                                    reason="live_sync", invalidation_condition="N/A"
                                )
                                self.logger.warning(f"实盘同步：发现交易所持仓 {symbol}，已强制同步到本地管理器 V2。")

                local_open_symbols = set(self.position_manager.get_all_open_positions().keys())
                symbols_to_close_locally = local_open_symbols - exchange_open_symbols
                for symbol in symbols_to_close_locally:
                     self.logger.warning(f"实盘同步：发现本地管理器 V2 持仓 {symbol} 已在交易所平仓，已强制同步关闭。")
                     self.position_manager.close_position(symbol) # 调用 V2 close_position

                # 4. [实盘] 更新净值历史 (不变)
                self.paper_equity_history.append({'timestamp': time.time() * 1000, 'equity': self.equity})

            except Exception as e:
                # [修改] 打印更具体的错误位置
                self.logger.critical(f"实盘同步失败 (sync_state): {e}", exc_info=True)
        else:
            # --- [模拟盘] 逻辑保持不变 ---
            # ... (模拟盘同步逻辑不变) ...
            unrealized_pnl = 0.0
            tickers = await self.exchange.fetch_tickers(self.symbols) # 模拟盘直接用 self.exchange 获取 tickers
            for symbol, pos in self.paper_positions.items():
                if pos and tickers.get(symbol):
                    current_price = tickers[symbol]['last']
                    pnl = (current_price - pos['entry_price']) * pos['size'] if pos['side'] == 'long' else (pos['entry_price'] - current_price) * pos['size']
                    pos['unrealized_pnl'] = pnl
                    unrealized_pnl += pnl
            total_margin = sum(p.get('margin', 0) for p in self.paper_positions.values() if p)
            self.paper_equity = self.paper_cash + total_margin + unrealized_pnl
            self.cash = self.paper_cash
            self.equity = self.paper_equity
            self.paper_equity_history.append({'timestamp': time.time() * 1000, 'equity': self.paper_equity})
            self._save_paper_state()

    # --- get_state_for_prompt 保持不变 ---
    def get_state_for_prompt(self):
        """根据实盘或模拟盘提供状态"""
        position_details = []

        if self.is_live:
            # --- [实盘] 从 AlphaPositionManager V2 获取状态 ---
            open_positions = self.position_manager.get_all_open_positions() # 获取含计算属性的状态
            for symbol, state in open_positions.items():
                    # V2 返回的状态已包含 total_size 和 avg_entry_price
                    pos_str = (
                        f"- {symbol.split(':')[0]}: Side={state['side'].upper()}, Size={state['total_size']:.4f}, Entry={state['avg_entry_price']:.4f}, "
                        f"TP={state.get('ai_suggested_take_profit', 'N/A')}, SL={state.get('ai_suggested_stop_loss', 'N/A')}, "
                        f"Invalidation_Condition='{state.get('invalidation_condition', 'N/A')}'"
                    )
                    position_details.append(pos_str)
        else:
            # --- [模拟盘] 从 paper_positions 获取状态 (不变) ---
            # ... (模拟盘部分不变) ...
            for symbol, pos in self.paper_positions.items():
                if pos:
                    pos_str = (
                        f"- {symbol.split(':')[0]}: Side={pos['side'].upper()}, Size={pos['size']:.4f}, Entry={pos['entry_price']:.4f}, "
                        f"UPL={pos.get('unrealized_pnl', 0.0):.2f}, TP={pos.get('take_profit', 'N/A')}, SL={pos.get('stop_loss', 'N/A')}, "
                        f"Invalidation_Condition='{pos.get('invalidation_condition', 'N/A')}'"
                    )
                    position_details.append(pos_str)


        if not position_details:
            position_details.append("You have no open positions.")

        initial_capital_for_calc = settings.ALPHA_LIVE_INITIAL_CAPITAL if self.is_live else settings.ALPHA_PAPER_CAPITAL
        if initial_capital_for_calc <= 0:
            performance_percent_str = "N/A (Invalid Initial Capital)"
        else:
            performance_percent = (self.equity / initial_capital_for_calc - 1) * 100
            performance_percent_str = f"{performance_percent:.2f}%"


        return {
            "account_value_usd": f"{self.equity:.2f}",
            "cash_usd": f"{self.cash:.2f}",
            "performance_percent": performance_percent_str,
            "open_positions": "\n".join(position_details)
        }


    # --- live_open 保持不变 ---
    async def live_open(self, symbol, side, size, leverage, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        is_adding = self.position_manager.is_open(symbol) # 检查是否已有持仓
        action_type = "加仓" if is_adding else "开新仓"
        self.logger.warning(f"!!! [实盘] AI 请求 {action_type}: {side.upper()} {size} {symbol} !!!")

        current_pos_state = None
        if is_adding:
            current_pos_state = self.position_manager.get_position_state(symbol)
            if not current_pos_state or current_pos_state.get('side') != side:
                self.logger.error(f"!!! [实盘] {action_type} 失败: 请求方向 ({side}) 与现有持仓方向 ({current_pos_state.get('side') if current_pos_state else 'N/A'}) 不符。将尝试作为新仓处理（覆盖）。")
                is_adding = False
                current_pos_state = None

        try:
            # --- 服务器端验证 (不变) ---
            raw_exchange = self.client.exchange
            if not raw_exchange.markets: await self.client.load_markets()
            market = raw_exchange.markets.get(symbol)
            if not market: raise ValueError(f"无法找到市场信息 {symbol}")

            ticker = await self.client.fetch_ticker(symbol)
            current_price = ticker.get('last')
            if not current_price or current_price <= 0: raise ValueError(f"无法获取有效价格 {symbol}")

            required_margin_for_this_order = (size * current_price) / leverage
            if required_margin_for_this_order <= 0: raise ValueError("保证金无效 (<= 0)")

            max_allowed_margin_per_order = self.cash * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
            if max_allowed_margin_per_order <= 0: raise ValueError(f"最大允许保证金无效 (<= 0), 可用现金: {self.cash}")

            adjusted_size = size
            if required_margin_for_this_order > max_allowed_margin_per_order:
                self.logger.warning(f"!!! [实盘] {action_type} 保证金超限，将自动缩减 !!! ...")
                adjusted_size_raw = (max_allowed_margin_per_order * leverage) / current_price
                adjusted_size = float(raw_exchange.amount_to_precision(symbol, adjusted_size_raw))
                min_amount = market.get('limits', {}).get('amount', {}).get('min')
                if min_amount is not None and adjusted_size < min_amount:
                     self.logger.error(f"!!! [实盘] {action_type} 自动缩减后仓位过小 ({adjusted_size} < {min_amount})，订单取消 !!!")
                     await send_bark_notification(f"⚠️ [实盘] AI {action_type} 被拒", f"品种: {symbol}\n原因: 缩减后仓位过小")
                     return
                self.logger.warning(f"自动缩减后的 Size: {adjusted_size}")
                required_margin_for_this_order = (adjusted_size * current_price) / leverage

            estimated_fee = adjusted_size * current_price * market.get('taker', self.FEE_RATE * 2)
            if self.cash < required_margin_for_this_order + estimated_fee:
                 self.logger.error(f"!!! [实盘] {action_type} 现金不足 !!! ...")
                 await send_bark_notification(f"❌ [实盘] AI {action_type} 失败", f"品种: {symbol}\n原因: 现金不足")
                 return
            # --- 验证结束 ---

            # 设置杠杆和保证金模式 (不变)
            await self.client.set_margin_mode(futures_settings.FUTURES_MARGIN_MODE, symbol)
            await self.client.set_leverage(leverage, symbol)

            # 执行市价单 (不变)
            exchange_side = 'BUY' if side == 'long' else 'SELL'
            order_result = await self.client.create_market_order(symbol, exchange_side, adjusted_size)

            # 解析订单结果 (不变)
            entry_price = float(order_result.get('average', order_result.get('price')))
            if not entry_price or entry_price <= 0: entry_price = float(order_result['price'])
            filled_size = float(order_result['filled'])
            timestamp = int(order_result['timestamp'])
            fee_info = order_result.get('fee')
            fee = 0.0
            if fee_info and isinstance(fee_info, dict) and 'cost' in fee_info:
                try: fee = float(fee_info['cost'])
                except (ValueError, TypeError): self.logger.warning(f"无法解析手续费 'cost': {fee_info}")
            # ... (省略其他 fee 检查)

            # 根据 is_adding 调用不同方法 (不变)
            success = False
            if is_adding:
                 success = self.position_manager.add_entry(
                     symbol=symbol, entry_price=entry_price, size=filled_size, entry_fee=fee,
                     leverage=leverage, stop_loss=stop_loss, take_profit=take_profit,
                     timestamp=timestamp, invalidation_condition=invalidation_condition
                 )
            else:
                 self.position_manager.open_position(
                     symbol=symbol, side=side, entry_price=entry_price, size=filled_size,
                     entry_fee=fee, leverage=leverage, stop_loss=stop_loss, take_profit=take_profit,
                     timestamp=timestamp, reason="ai_entry", invalidation_condition=invalidation_condition
                 )
                 success = True

            if success:
                 self.logger.warning(f"!!! [实盘] {action_type} 成功: {side.upper()} {filled_size} {symbol} @ {entry_price} (Fee: {fee}) | AI原因: {reason}")
                 title = f"📈 [实盘] AI {action_type}: {side.upper()} {symbol.split('/')[0]}"
                 final_pos_state = self.position_manager.get_position_state(symbol)
                 final_avg_price = final_pos_state.get('avg_entry_price', entry_price) if final_pos_state else entry_price
                 final_total_size = final_pos_state.get('total_size', filled_size) if final_pos_state else filled_size
                 body = f"价格: {entry_price:.4f}\n数量: {filled_size}\n杠杆: {leverage}x\n手续费: {fee:.4f}\n本次保证金: {required_margin_for_this_order:.2f}\n止盈/止损: {take_profit}/{stop_loss}"
                 if is_adding: body += f"\n新均价: {final_avg_price:.4f}\n总数量: {final_total_size:.4f}"
                 body += f"\nAI原因: {reason}"
                 if adjusted_size != size: body += f"\n(请求 {size} 被缩减至 {filled_size})"
                 await send_bark_notification(title, body)
            else:
                 raise RuntimeError(f"{action_type} 失败，但未抛出明确异常。")

        except InsufficientFunds as e:
             self.logger.error(f"!!! [实盘] {action_type} 失败 (交易所返回资金不足): {e}", exc_info=False)
             await send_bark_notification(f"❌ [实盘] AI {action_type} 失败", f"品种: {symbol}\n原因: 交易所报告资金不足")
        except Exception as e:
            self.logger.error(f"!!! [实盘] {action_type} 失败: {e}", exc_info=True)
            await send_bark_notification(f"❌ [实盘] AI {action_type} 失败", f"品种: {symbol}\n错误: {e}")

    # --- live_partial_close 保持不变 ---
    async def live_partial_close(self, symbol: str, size_percent: Optional[float] = None, size_absolute: Optional[float] = None, reason: str = "N/A"):
        self.logger.warning(f"!!! [实盘] AI 请求部分平仓: {symbol} | %: {size_percent} | Abs: {size_absolute} | 原因: {reason} !!!")

        pos_state = self.position_manager.get_position_state(symbol)
        if not pos_state or pos_state.get('total_size', 0) <= 0:
            self.logger.error(f"!!! [实盘] 部分平仓失败: 本地管理器未找到 {symbol} 的有效持仓记录。")
            return

        current_total_size = pos_state['total_size']
        size_to_close = 0.0
        if size_percent is not None and 0 < size_percent < 1: size_to_close = current_total_size * size_percent
        elif size_absolute is not None and 0 < size_absolute < current_total_size: size_to_close = size_absolute
        else: self.logger.error(f"!!! [实盘] 部分平仓失败: 无效数量参数..."); await send_bark_notification(f"❌ [实盘] AI 部分平仓失败", f"品种: {symbol}\n原因: 无效数量参数"); return

        try:
             raw_exchange = self.client.exchange
             if not raw_exchange.markets: await self.client.load_markets()
             market = raw_exchange.markets.get(symbol)
             if not market: raise ValueError(f"无法找到市场信息 {symbol}")
             size_to_close = float(raw_exchange.amount_to_precision(symbol, size_to_close))
             min_amount = market.get('limits', {}).get('amount', {}).get('min')
             if min_amount is not None and size_to_close < min_amount: self.logger.error(f"!!! [实盘] 部分平仓失败: 数量过小..."); await send_bark_notification(f"❌ [实盘] AI 部分平仓失败", f"品种: {symbol}\n原因: 平仓数量过小"); return
             if size_to_close <= 0: self.logger.error(f"!!! [实盘] 部分平仓失败: 计算数量为 0"); return
        except Exception as e: self.logger.error(f"!!! [实盘] 部分平仓失败 (检查数量时出错): {e}", exc_info=True); return

        try:
            internal_side = pos_state['side']; avg_entry_price = pos_state['avg_entry_price']
            open_fee_total = pos_state['total_entry_fee']; leverage = pos_state.get('leverage', 0)
            exchange_close_side = 'SELL' if internal_side == 'long' else 'BUY'
            order_result = await self.client.create_market_order(symbol, exchange_close_side, size_to_close, params={'reduceOnly': True})
            exit_price = float(order_result.get('average', order_result.get('price')))
            if not exit_price or exit_price <= 0: exit_price = float(order_result['price'])
            filled_size = float(order_result['filled']); timestamp = int(order_result['timestamp'])
            fee_info = order_result.get('fee'); close_fee = 0.0
            if fee_info and isinstance(fee_info, dict) and 'cost' in fee_info:
                try: close_fee = float(fee_info['cost'])
                except (ValueError, TypeError): self.logger.warning(f"无法解析部分平仓手续费 'cost': {fee_info}")
            # ... (省略其他 fee 检查)
            open_fee_for_this_part = (open_fee_total / current_total_size) * filled_size if current_total_size > 0 else 0
            if internal_side == 'long': gross_pnl_part = (exit_price - avg_entry_price) * filled_size
            else: gross_pnl_part = (avg_entry_price - exit_price) * filled_size
            net_pnl_part = gross_pnl_part - open_fee_for_this_part - close_fee
            trade_data = { 'symbol': symbol, 'side': internal_side, 'entry_price': avg_entry_price, 'exit_price': exit_price, 'size': filled_size, 'net_pnl': net_pnl_part, 'fees': open_fee_for_this_part + close_fee, 'margin': 0, 'leverage': leverage, 'open_reason': pos_state.get('entry_reason', 'N/A'), 'close_reason': reason, 'timestamp': timestamp, 'partial': True }
            self.trade_logger.record_trade(trade_data)
            success = self.position_manager.reduce_position(symbol, filled_size)
            if success:
                 remaining_size = self.position_manager.get_size(symbol)
                 self.logger.warning(f"!!! [实盘] 部分平仓成功: {symbol} | 平掉 {filled_size} @ {exit_price:.4f} (Fee: {close_fee}) | 本次净盈亏: {net_pnl_part:.2f} USDT | 剩余 {remaining_size:.4f} | 原因: {reason}")
                 pnl_prefix = "盈利" if net_pnl_part >= 0 else "亏损"; title = f"💰 [实盘] AI 部分平仓: {pnl_prefix} {abs(net_pnl_part):.2f} USDT"
                 body = (f"品种: {symbol.split('/')[0]}\n方向: {internal_side.upper()}\n平仓价格: {exit_price:.4f}\n平仓数量: {filled_size}\n手续费: {close_fee:.4f}\n剩余数量: {remaining_size:.4f}\n原因: {reason}")
                 await send_bark_notification(title, body)
            else: raise RuntimeError("position_manager.reduce_position 返回失败")
        except InsufficientFunds as e: self.logger.error(f"!!! [实盘] 部分平仓失败 (交易所返回资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ [实盘] AI 部分平仓失败", f"品种: {symbol}\n原因: 交易所报告资金不足")
        except Exception as e: self.logger.error(f"!!! [实盘] 部分平仓失败: {e}", exc_info=True); await send_bark_notification(f"❌ [实盘] AI 部分平仓失败", f"品种: {symbol}\n错误: {e}")


    # --- live_close (全平) 保持不变 ---
    async def live_close(self, symbol, reason: str = "N/A"):
        # ... (全平逻辑不变, 内部已使用 V2 管理器方法) ...
        self.logger.warning(f"!!! [实盘] 正在尝试(全)平仓: {symbol} | 原因: {reason} !!!")
        pos_state = self.position_manager.get_position_state(symbol)
        if not pos_state or pos_state.get('total_size', 0) <= 0: # 使用 total_size
            self.logger.error(f"!!! [实盘] (全)平仓失败: 本地管理器未找到 {symbol} 的有效持仓记录。")
            return

        try:
            internal_side = pos_state['side']
            size_to_close = pos_state['total_size'] # 使用 total_size

            exchange_close_side = 'SELL' if internal_side == 'long' else 'BUY'
            order_result = await self.client.create_market_order(symbol, exchange_close_side, size_to_close, params={'reduceOnly': True})

            exit_price = float(order_result.get('average', order_result.get('price')))
            if not exit_price or exit_price <= 0: exit_price = float(order_result['price'])
            filled_size = float(order_result['filled'])
            timestamp = int(order_result['timestamp'])
            fee_info = order_result.get('fee')
            close_fee = 0.0
            if fee_info and isinstance(fee_info, dict) and 'cost' in fee_info:
                try: close_fee = float(fee_info['cost'])
                except (ValueError, TypeError): self.logger.warning(f"无法解析平仓手续费 'cost': {fee_info}")
            # ... (省略其他 fee 检查)

            avg_entry_price = pos_state['avg_entry_price'] # 使用均价
            open_fee_total = pos_state['total_entry_fee'] # 使用总开仓费
            leverage = pos_state.get('leverage', 0)

            if internal_side == 'long': gross_pnl = (exit_price - avg_entry_price) * filled_size
            else: gross_pnl = (avg_entry_price - exit_price) * filled_size
            net_pnl = gross_pnl - open_fee_total - close_fee # 减去总开仓费

            trade_data = {
                'symbol': symbol, 'side': internal_side, 'entry_price': avg_entry_price, # 记录均价
                'exit_price': exit_price, 'size': filled_size, 'net_pnl': net_pnl,
                'fees': open_fee_total + close_fee, 'margin': 0, 'leverage': leverage,
                'open_reason': pos_state.get('entry_reason', 'N/A'),
                'close_reason': reason, 'timestamp': timestamp,
                'partial': False # 标记为全平
            }

            self.trade_logger.record_trade(trade_data)
            self.position_manager.close_position(symbol) # 调用 V2 全平

            self.logger.warning(f"!!! [实盘] (全)平仓成功: {symbol} @ {exit_price:.4f} (Fee: {close_fee}), 净盈亏: {net_pnl:.2f} USDT | 原因: {reason}")
            pnl_prefix = "盈利" if net_pnl >= 0 else "亏损"
            title = f"📉 [实盘] AI (全)平仓: {pnl_prefix} {abs(net_pnl):.2f} USDT"
            body = f"品种: {symbol.split('/')[0]}\n方向: {internal_side.upper()}\n平仓价格: {exit_price:.4f}\n手续费: {close_fee:.4f}\n原因: {reason}"
            await send_bark_notification(title, body)

        except InsufficientFunds as e:
             self.logger.error(f"!!! [实盘] (全)平仓失败 (交易所返回资金不足): {e}", exc_info=False)
             await send_bark_notification(f"❌ [实盘] AI (全)平仓失败", f"品种: {symbol}\n原因: 交易所报告资金不足")
        except Exception as e:
            self.logger.error(f"!!! [实盘] (全)平仓失败: {e}", exc_info=True)
            await send_bark_notification(f"❌ [实盘] AI (全)平仓失败", f"品种: {symbol}\n错误: {e}")


    # --- 模拟盘函数 (保持不变) ---
    async def paper_open(self, symbol, side, size, price, leverage, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        # ... (模拟开仓/加仓逻辑不变) ...
        margin_required = (size * price) / leverage
        fee = size * price * self.FEE_RATE
        if self.paper_cash < (margin_required + fee):
            self.logger.error(f"模拟开仓/加仓失败: 资金不足 (需 {margin_required+fee:.2f}, 可用 {self.paper_cash:.2f})")
            return
        self.paper_cash -= fee
        existing_pos = self.paper_positions.get(symbol)
        if existing_pos and existing_pos.get('side') == side:
            old_size = existing_pos['size']
            old_value = old_size * existing_pos['entry_price']
            new_value = size * price
            existing_pos['size'] += size
            existing_pos['entry_price'] = (old_value + new_value) / existing_pos['size']
            existing_pos['margin'] += margin_required
            self.paper_cash -= margin_required
            existing_pos['take_profit'] = take_profit
            existing_pos['stop_loss'] = stop_loss
            existing_pos['invalidation_condition'] = invalidation_condition
            self.logger.warning(f"模拟加仓: {side.upper()} {size:.4f} {symbol} @ {price:.4f} | 新均价: {existing_pos['entry_price']:.4f} | 新TP/SL: {take_profit}/{stop_loss}")
            title = f"🔼 AI 加仓提醒: {side.upper()} {symbol.split('/')[0]}"
            body = f"价格: {price:.4f}\n新均价: {existing_pos['entry_price']:.4f}\n新TP/SL: {take_profit}/{stop_loss}"
            await send_bark_notification(title, body)
        else:
            self.paper_cash -= margin_required
            self.paper_positions[symbol] = { 'side': side, 'size': size, 'entry_price': price, 'leverage': leverage, 'margin': margin_required, 'unrealized_pnl': 0.0, 'open_reason': reason, 'take_profit': take_profit, 'stop_loss': stop_loss, 'invalidation_condition': invalidation_condition }
            self.logger.warning(f"模拟开仓: {side.upper()} {size:.4f} {symbol} @ {price:.4f} | TP: {take_profit} | SL: {stop_loss} | AI原因: {reason}")
            title = f"📈 AI 开仓提醒: {side.upper()} {symbol.split('/')[0]}"
            body = f"价格: {price:.4f}\n杠杆: {leverage}x\n止盈/止损: {take_profit}/{stop_loss}\nAI原因: {reason}"
            await send_bark_notification(title, body)

    async def paper_close(self, symbol, price, reason: str = "N/A"):
        # ... (模拟全平逻辑不变) ...
        pos = self.paper_positions.get(symbol)
        if not pos: return
        gross_pnl = (price - pos['entry_price']) * pos['size'] if pos['side'] == 'long' else (pos['entry_price'] - price) * pos['size']
        close_fee = pos['size'] * price * self.FEE_RATE
        open_fee = pos['size'] * pos['entry_price'] * self.FEE_RATE # 估算开仓费
        pnl_val_after_fees = gross_pnl - open_fee - close_fee
        self.paper_cash += (pos['margin'] + pnl_val_after_fees)
        self.paper_trade_history.append({ 'symbol': symbol, 'side': pos['side'], 'entry_price': pos['entry_price'], 'exit_price': price, 'size': pos['size'], 'pnl': pnl_val_after_fees, 'fees': open_fee + close_fee, 'margin': pos['margin'], 'leverage': pos.get('leverage'), 'open_reason': pos.get('open_reason', 'N/A'), 'close_reason': reason, 'timestamp': time.time() * 1000, 'partial': False })
        self.paper_positions[symbol] = {}
        self.logger.warning(f"模拟(全)平仓: {symbol} @ {price:.4f}, 净盈亏: {pnl_val_after_fees:.2f} USDT | 原因: {reason}")
        pnl_prefix = "盈利" if pnl_val_after_fees >= 0 else "亏损"
        title = f"📉 AI 平仓提醒: {pnl_prefix} {abs(pnl_val_after_fees):.2f} USDT"
        body = f"品种: {symbol.split('/')[0]}\n方向: {pos['side'].upper()}\n平仓价格: {price:.4f}\n原因: {reason}"
        await send_bark_notification(title, body)

    # --- [新增] 模拟盘部分平仓函数 ---
    async def paper_partial_close(self, symbol: str, price: float, size_percent: Optional[float] = None, size_absolute: Optional[float] = None, reason: str = "N/A"):
        pos = self.paper_positions.get(symbol)
        if not pos:
            self.logger.error(f"模拟部分平仓失败: 未找到 {symbol} 的持仓。")
            return

        current_total_size = pos.get('size', 0.0)
        if current_total_size <= 0:
            self.logger.error(f"模拟部分平仓失败: {symbol} 当前无持仓。")
            return

        size_to_close = 0.0
        if size_percent is not None and 0 < size_percent < 1:
             size_to_close = current_total_size * size_percent
        elif size_absolute is not None and 0 < size_absolute < current_total_size:
             size_to_close = size_absolute
        else:
             self.logger.error(f"模拟部分平仓失败: 无效的平仓数量参数 (percent={size_percent}, absolute={size_absolute}, current={current_total_size})")
             return

        # 模拟盘不需要处理精度和最小量
        if size_to_close <= 0:
             self.logger.error(f"模拟部分平仓失败: 计算出的平仓数量为 0 或负数")
             return

        # 计算本次部分平仓的盈亏
        entry_price = pos['entry_price'] # 模拟盘使用当前均价
        margin_per_unit = pos['margin'] / current_total_size if current_total_size > 0 else 0
        open_fee_per_unit = (entry_price * self.FEE_RATE) # 估算单位开仓费
        close_fee_for_part = size_to_close * price * self.FEE_RATE

        if pos['side'] == 'long': gross_pnl_part = (price - entry_price) * size_to_close
        else: gross_pnl_part = (entry_price - price) * size_to_close
        net_pnl_part = gross_pnl_part - (open_fee_per_unit * size_to_close) - close_fee_for_part

        # 释放对应比例的保证金
        margin_to_release = margin_per_unit * size_to_close
        self.paper_cash += (margin_to_release + net_pnl_part)

        # 记录交易
        self.paper_trade_history.append({
            'symbol': symbol, 'side': pos['side'], 'entry_price': entry_price, # 记录均价
            'exit_price': price, 'size': size_to_close, # 本次平掉的数量
            'pnl': net_pnl_part, # 本次平仓的净 PNL
            'fees': (open_fee_per_unit * size_to_close) + close_fee_for_part,
            'margin': margin_to_release,
            'leverage': pos.get('leverage'),
            'open_reason': pos.get('open_reason', 'N/A'),
            'close_reason': reason, 'timestamp': time.time() * 1000,
            'partial': True # 标记为部分平仓
        })

        # 更新剩余仓位
        pos['size'] -= size_to_close
        pos['margin'] -= margin_to_release
        # 模拟盘均价不变

        # 检查剩余 size 是否过小
        if pos['size'] < 1e-9:
             self.logger.warning(f"模拟部分平仓后 {symbol} 剩余数量过小，视为全平。")
             self.paper_positions[symbol] = {} # 直接清空
        else:
             self.logger.warning(f"模拟部分平仓: {symbol} | 平掉 {size_to_close:.4f} @ {price:.4f} | 本次净盈亏: {net_pnl_part:.2f} USDT | 剩余: {pos['size']:.4f} | 原因: {reason}")
             pnl_prefix = "盈利" if net_pnl_part >= 0 else "亏损"
             title = f"💰 AI 部分平仓提醒: {pnl_prefix} {abs(net_pnl_part):.2f} USDT"
             body = (f"品种: {symbol.split('/')[0]}\n方向: {pos['side'].upper()}\n"
                     f"平仓价格: {price:.4f}\n平仓数量: {size_to_close:.4f}\n"
                     f"剩余数量: {pos['size']:.4f}\n原因: {reason}")
             await send_bark_notification(title, body)

    @property
    def equity_history(self):
        """统一返回净值历史，无论实盘还是模拟盘"""
        return self.paper_equity_history

    @property
    def trade_history(self):
        """统一返回交易历史，无论实盘还是模拟盘"""
        if self.is_live:
            return self.trade_logger.get_history()
        else:
            return self.paper_trade_history
