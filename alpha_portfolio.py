# 文件: alpha_portfolio.py (V45.38 - 状态持久化修复)
# 1. [V45.38 修复] 新增 _load/_save_pending_limits，使挂单计划持久化，解决重启后状态丢失问题。
# 2. [V45.37 修复] sync_state 不再调用高风险的 fetch_open_orders() (无参数)。
# 3. [V45.37 优化] sync_state 现在只精确地、并行地获取 self.pending_limit_orders 中品种的挂单。
# 4. (V45.36 修复 - 保留) 修复了限价单杠杆/本金错误和成交通知。
# 5. (V45.33 修复 - 保留) live_close 包含 filled_size > 0 检查。
# 6. [GEMINI V3 修复] live_open_limit 已升级，支持限价加仓。
# 7. [GEMINI V4 修复] live_open_limit 现在存储 'limit_price' 以支持价格偏离取消。

import logging
import time
import json
import os
import asyncio
from collections import deque
import pandas as pd
from config import settings, futures_settings
from bark_notifier import send_bark_notification
from ccxt.base.errors import InsufficientFunds, ExchangeError, OrderNotFound
from typing import Optional, Dict, List

from exchange_client import ExchangeClient
from alpha_trade_logger import AlphaTradeLogger
from alpha_position_manager import AlphaPositionManager # 假设 V2.2

class AlphaPortfolio:
    FEE_RATE = 0.001 # 仅用于模拟盘
    MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK = 5.1 # V23.4 保留

    def __init__(self, exchange, symbols: list):
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

        # [V45.38 修复] 挂单持久化
        # 结构: { "BTC/USDT:USDT": {"order_id": "123", "timestamp": ..., "leverage": 10, "limit_price": 65000, ...}, ... }
        self.pending_limit_orders: Dict[str, Dict] = {}
        # [V45.38 修复] 定义持久化文件路径
        self.pending_limits_file = os.path.join(futures_settings.FUTURES_STATE_DIR, 'alpha_pending_limits.json')
        # [V45.38 修复] 在启动时加载
        self._load_pending_limits()


    def _load_paper_state(self):
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

    # --- [V45.38 修复] 新增挂单持久化函数 ---
    def _load_pending_limits(self):
        """[V45.38 修复] 从 JSON 加载待处理的限价单"""
        if not self.is_live: return # 模拟盘不需要
        if not os.path.exists(self.pending_limits_file):
            self.logger.info(f"{self.mode_str} 待处理限价单文件不存在，跳过加载。")
            return
        try:
            with open(self.pending_limits_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict):
                self.pending_limit_orders = loaded_data
                self.logger.warning(f"成功加载 {len(self.pending_limit_orders)} 个待处理限价单计划。")
            else:
                self.logger.error(f"加载待处理限价单失败：文件内容不是一个字典。")
        except json.JSONDecodeError as e:
            self.logger.error(f"加载待处理限价单失败：JSON 格式错误 - {e}", exc_info=False)
        except Exception as e:
            self.logger.error(f"加载待处理限价单失败: {e}", exc_info=True)

    async def _save_pending_limits(self):
        """[V45.38 修复] 异步保存待处理的限价单到 JSON"""
        if not self.is_live: return # 模拟盘不需要
        
        # (这是一个简化的异步保存，在真实的多线程环境中可能需要锁)
        try:
            os.makedirs(os.path.dirname(self.pending_limits_file), exist_ok=True)
            # 使用异步 IO 写入 (如果可用)，或者回退到同步写入
            # (为简单起见，这里使用同步写入，因为它足够快)
            with open(self.pending_limits_file, 'w', encoding='utf-8') as f:
                json.dump(self.pending_limit_orders, f, indent=4, ensure_ascii=False)
            self.logger.debug(f"已保存 {len(self.pending_limit_orders)} 个待处理限价单。")
        except Exception as e:
            self.logger.error(f"保存待处理限价单失败: {e}", exc_info=True)

    async def add_pending_limit_order(self, symbol: str, plan: Dict):
        """[V45.38 修复] 安全地添加一个挂单计划并保存"""
        self.pending_limit_orders[symbol] = plan
        await self._save_pending_limits()

    async def remove_pending_limit_order(self, symbol: str) -> Optional[Dict]:
        """[V45.38 修复] 安全地移除一个挂单计划并保存"""
        plan = self.pending_limit_orders.pop(symbol, None)
        await self._save_pending_limits()
        return plan
    # --- [V45.38 修复结束] ---


    async def sync_state(self):
        """
        [V45.38 修复] 修改 pop S' S' S' 
        [V45.37 策略A 重大修复]
        1. 不再调用高风险的 fetch_open_orders() (无参数)。
        2. 仅并行获取 self.pending_limit_orders 中品种的挂单。
        [GEMINI V5 修复] 增加了 'elif pending_plan:' 逻辑块，以正确处理 "限价加仓" 订单的成交。
        """
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

                    # --- [V45.37 策略A 步骤 1: 精确同步待处理订单] ---
                    open_order_ids = set()
                    if self.pending_limit_orders:
                        symbols_to_check = list(self.pending_limit_orders.keys())
                        self.logger.debug(f"Sync: 正在检查 {len(symbols_to_check)} 个品种的待处理订单: {symbols_to_check}")
                        
                        # 并行获取所有相关品种的挂单
                        fetch_tasks = [self.client.fetch_open_orders(symbol=s) for s in symbols_to_check]
                        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                        
                        all_open_orders: List[Dict] = []
                        for i, res in enumerate(results):
                            symbol = symbols_to_check[i]
                            if isinstance(res, Exception):
                                self.logger.error(f"Sync: 获取 {symbol} 的挂单失败: {res}", exc_info=False)
                            elif res:
                                all_open_orders.extend(res)
                        
                        open_order_ids = {o['id'] for o in all_open_orders}
                        
                        # 迭代副本以安全删除
                        for symbol, plan in list(self.pending_limit_orders.items()):
                            plan_order_id = plan.get('order_id')
                            if not plan_order_id:
                                self.logger.warning(f"Sync: 待处理计划 {symbol} 缺少 order_id，已移除。")
                                # [V45.38 修复]
                                await self.remove_pending_limit_order(symbol)
                                continue

                            if plan_order_id not in open_order_ids:
                                # 这个订单不再是 "open" 状态，它可能已成交 (将在下面被同步) 或被取消/超时
                                self.logger.warning(f"Sync: 待处理订单 {plan_order_id} ({symbol}) 不再 'open'。已从待处理列表移除。")
                                # [V45.38 修复]
                                await self.remove_pending_limit_order(symbol)
                    # --- [V45.37 步骤 1 结束] ---

                    real_positions = await self.client.fetch_positions(self.symbols); exchange_open_symbols = set()
                    for pos in real_positions:
                        symbol = pos.get('symbol')
                        if symbol in self.symbols:
                            size_str = pos.get('contracts') or pos.get('contractSize'); size = float(size_str) if size_str else 0.0
                            side = pos.get('side').lower() if pos.get('side') else None
                            if abs(size) > 1e-9:
                                exchange_open_symbols.add(symbol)
                                
                                # --- [V45.38 修复] 检查限价单成交 ---
                                # 检查此持仓是否由待处理的限价单触发
                                # (注意: 我们在检查时 'pop'，如果匹配，它将被移除并保存)
                                pending_plan = await self.remove_pending_limit_order(symbol)
                                
                                if not self.position_manager.is_open(symbol):
                                    # --- [V45.36 策略A 步骤 2: 修复杠杆和通知] ---
                                    # 这是一个 "新" 持仓
                                    self.logger.warning(f"{self.mode_str} sync: 发现交易所新持仓 {symbol}，正在同步到本地...")
                                    
                                    entry_str = pos.get('entryPrice') or pos.get('basePrice'); entry = float(entry_str) if entry_str else 0.0
                                    
                                    plan_reason = "live_sync"
                                    plan_sl = None
                                    plan_tp = None
                                    plan_inval = None
                                    
                                    exchange_lev_val = pos.get('leverage')
                                    final_leverage = int(exchange_lev_val) if exchange_lev_val is not None and float(exchange_lev_val) > 0 else 1

                                    if pending_plan:
                                        self.logger.warning(f"Sync: 新持仓 {symbol} 匹配到一个AI限价单计划。正在应用 SL/TP/Reason...")
                                        plan_reason = pending_plan.get('reason', 'live_sync_with_plan')
                                        plan_sl = pending_plan.get('stop_loss')
                                        plan_tp = pending_plan.get('take_profit')
                                        plan_inval = pending_plan.get('invalidation_condition')
                                        
                                        plan_leverage = pending_plan.get('leverage')
                                        if plan_leverage and isinstance(plan_leverage, (int, float)) and plan_leverage > 0:
                                            # [V45.38 修复] 这是关键！使用计划中的杠杆！
                                            self.logger.info(f"Sync: 使用AI计划的杠杆 {plan_leverage}x (交易所报告为 {exchange_lev_val}x)")
                                            final_leverage = int(plan_leverage)
                                        else:
                                            self.logger.warning(f"Sync: AI计划中无杠杆，使用交易所杠杆 {final_leverage}x")

                                        try:
                                            title = f"✅ {self.mode_str} AI 限价单成交: {side.upper()} {symbol.split('/')[0]}"
                                            body = f"成交价格: {entry:.4f}\n数量: {abs(size)}\n杠杆: {final_leverage}x\nTP/SL: {plan_tp}/{plan_sl}\nAI原因: {plan_reason}"
                                            await send_bark_notification(title, body)
                                        except Exception as e_notify:
                                            self.logger.error(f"Sync: 发送成交通知失败: {e_notify}")
                                    
                                    else:
                                        # [V45.38 修复] 如果 'pending_plan' 是 None (因为重启丢失了)，
                                        # 我们也必须把刚 'pop' 失败的 symbol 加回去，因为它还在交易所挂着
                                        # (哦，不... 如果它成交了，它就不在交易所挂着了... 
                                        # 这里的逻辑是：如果 'pop' 失败，说明本地没有这个计划。)
                                        # (这是正确的行为：我们没有计划，所以我们只能用交易所的杠杆)
                                        self.logger.warning(f"Sync: 新持仓 {symbol} 未匹配到AI计划，使用默认值同步 (杠杆 {final_leverage}x)。")

                                    self.position_manager.open_position(
                                        symbol=symbol, 
                                        side=side, 
                                        entry_price=entry, 
                                        size=abs(size), 
                                        entry_fee=0.0, 
                                        leverage=final_leverage, 
                                        stop_loss=plan_sl, 
                                        take_profit=plan_tp, 
                                        timestamp=int(pos.get('timestamp', time.time()*1000)), 
                                        reason=plan_reason, 
                                        invalidation_condition=plan_inval
                                    )
                                    # --- [V45.36 步骤 2 结束] ---
                                    
                                # --- [GEMINI V5 修复] 新增 'elif pending_plan:' 块来处理限价加仓 ---
                                elif pending_plan:
                                    # 这是一个 "限价加仓"
                                    self.logger.warning(f"{self.mode_str} sync: 发现交易所持仓 {symbol} 变动，匹配到AI限价加仓计划。")
                                    
                                    # 我们需要从交易所获取 *最新* 的均价和总数
                                    entry_str = pos.get('entryPrice') or pos.get('basePrice');
                                    current_avg_price = float(entry_str) if entry_str else 0.0
                                    current_total_size = abs(size)
                                    
                                    # 从本地获取 *旧* 的状态
                                    old_state = self.position_manager.get_position_state(symbol)
                                    old_total_size = old_state.get('total_size', 0.0) if old_state else 0.0
                                    
                                    # 计算本次加仓的数量和价格
                                    added_size = current_total_size - old_total_size
                                    
                                    if added_size > 1e-9:
                                        self.logger.info(f"Sync: 本次加仓 {added_size} (Exch: {current_total_size}, Local: {old_total_size})")
                                        
                                        # 价格计算 (反推)
                                        old_avg_price = old_state.get('avg_entry_price', 0.0) if old_state else 0.0
                                        
                                        add_price = 0.0
                                        if added_size > 0:
                                            add_price = ((current_avg_price * current_total_size) - (old_avg_price * old_total_size)) / added_size
                                        
                                        if add_price <= 0:
                                             self.logger.warning(f"Sync: 无法反推加仓价格 (AddPrice: {add_price})。将使用交易所均价 {current_avg_price} 作为近似值。")
                                             add_price = current_avg_price

                                        plan_sl = pending_plan.get('stop_loss')
                                        plan_tp = pending_plan.get('take_profit')
                                        plan_inval = pending_plan.get('invalidation_condition')
                                        plan_reason = pending_plan.get('reason', 'live_sync_add_with_plan')
                                        # [V45.38 修复] 使用计划中的杠杆
                                        plan_leverage = pending_plan.get('leverage')
                                        
                                        # 使用 position_manager.add_entry 来正确更新均价和规则
                                        self.position_manager.add_entry(
                                            symbol=symbol,
                                            entry_price=add_price,
                                            size=added_size,
                                            entry_fee=0.0, # TODO: 费用无法反推，暂时记为0
                                            leverage=plan_leverage, # 使用计划的杠杆
                                            stop_loss=plan_sl,
                                            take_profit=plan_tp,
                                            timestamp=int(pos.get('timestamp', time.time()*1000)),
                                            invalidation_condition=plan_inval
                                        )
                                        
                                        # 获取 *更新后* 的本地状态，以显示正确的均价
                                        final_state = self.position_manager.get_position_state(symbol)
                                        final_avg_price = final_state.get('avg_entry_price', current_avg_price)
                                        final_total_size = final_state.get('total_size', current_total_size)
                                        
                                        try:
                                            title = f"🔼 {self.mode_str} AI 限价加仓成交: {side.upper()} {symbol.split('/')[0]}"
                                            body = f"成交价格: {add_price:.4f}\n数量: {added_size}\n杠杆: {plan_leverage}x\n新均价: {final_avg_price:.4f}\n新总量: {final_total_size}\nAI原因: {plan_reason}"
                                            await send_bark_notification(title, body)
                                        except Exception as e_notify:
                                            self.logger.error(f"Sync: 发送加仓成交通知失败: {e_notify}")
                                            
                                    else:
                                        self.logger.warning(f"Sync: 匹配到限价单 {symbol}，但计算出的 added_size 为 0 或负数 ({added_size})。不同步加仓。")
                                        # [V45.38 修复] 如果加仓失败，我们必须把 'pop' 出来的计划加回去
                                        await self.add_pending_limit_order(symbol, pending_plan)
                                        
                                # --- [GEMINI V5 修复结束] ---
                                else:
                                    # 本地和交易所均存在，且没有匹配到限价单
                                    self.logger.debug(f"{self.mode_str} sync: {symbol} 本地和交易所均存在。")
                    
                    local_open_symbols = set(self.position_manager.get_all_open_positions().keys())
                    symbols_to_close_locally = local_open_symbols - exchange_open_symbols
                    for symbol in symbols_to_close_locally:
                         self.logger.warning(f"{self.mode_str} sync: 本地 {symbol} 在交易所已平仓，同步关闭。")
                         self.position_manager.close_position(symbol)
                    
                    current_equity_to_append = self.equity
                    self.logger.debug(f"{self.mode_str} sync: 准备追加净值历史。 Equity: {current_equity_to_append}, Type: {type(current_equity_to_append)}")
                    is_valid_equity = current_equity_to_append is not None and isinstance(current_equity_to_append, (int, float)) and (not pd or not pd.isna(current_equity_to_append))
                    if is_valid_equity:
                        history_entry = {'timestamp': time.time() * 1000, 'equity': float(current_equity_to_append)}
                        self.paper_equity_history.append(history_entry)
                        self.logger.debug(f"{self.mode_str} sync: 成功追加净值历史: {history_entry}")
                    else: self.logger.warning(f"{self.mode_str} sync: 跳过追加净值历史，Equity无效: {current_equity_to_append} (Type: {type(current_equity_to_append)})")
                except Exception as e: self.logger.critical(f"{self.mode_str} sync 失败 (实盘部分): {e}", exc_info=True)
            else: # 模拟盘
                # ... (模拟盘逻辑无变化) ...
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
                is_valid_equity = current_equity_to_append is not None and isinstance(current_equity_to_append, (int, float)) and (not pd or not pd.isna(current_equity_to_append))
                if is_valid_equity:
                    history_entry = {'timestamp': time.time() * 1000, 'equity': float(current_equity_to_append)}
                    self.paper_equity_history.append(history_entry)
                    self.logger.debug(f"{self.mode_str} sync: 成功追加净值历史: {history_entry}")
                else: self.logger.warning(f"{self.mode_str} sync: 跳过追加净值历史，Equity无效: {current_equity_to_append} (Type: {type(current_equity_to_append)})")
                self._save_paper_state()
        except Exception as e: self.logger.critical(f"{self.mode_str} sync_state 顶层执行失败: {e}", exc_info=True)


    def get_state_for_prompt(self, tickers: dict = None):
        # ... (此函数无变化) ...
        position_details = []
        
        if self.is_live:
            if tickers is None: 
                tickers = {}
                self.logger.warning("get_state_for_prompt (live) 未收到 tickers! UPL 将丢失。")

            open_positions = self.position_manager.get_all_open_positions()
            for symbol, state in open_positions.items():
                
                upl_str = "UPL=N/A"
                try:
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
                        
                        margin = state.get('margin', 0.0) 
                        pnl_percent = (upl / margin) * 100 if margin > 0 else 0.0
                        upl_str = f"UPL={upl:.2f}$ ({pnl_percent:.2f}%)" 
                    else:
                        upl_str = "UPL=NoPrice"
                except Exception as e:
                    self.logger.error(f"实盘 get_state_for_prompt UPL 计算失败 {symbol}: {e}")
                    upl_str = f"UPL=CalcErr"

                pos_str = ( f"- {symbol.split(':')[0]}: Side={state['side'].upper()}, Size={state['total_size']:.4f}, Entry={state['avg_entry_price']:.4f}, "
                            f"{upl_str}, " 
                            f"TP={state.get('ai_suggested_take_profit', 'N/A')}, SL={state.get('ai_suggested_stop_loss', 'N/A')}, "
                            f"Invalidation='{state.get('invalidation_condition', 'N/A')}'")
                position_details.append(pos_str)

        else:
            for symbol, pos in self.paper_positions.items():
                if pos and isinstance(pos, dict) and pos.get('size', 0) > 0:
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
    
    async def live_open(self, symbol, side, size, leverage, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        # ... (此函数无变化) ...
        is_adding = self.position_manager.is_open(symbol); action_type = "加仓" if is_adding else "开新仓"
        self.logger.warning(f"!!! {self.mode_str} AI 请求 {action_type} (市价): {side.upper()} {size} {symbol} !!!")
        
        current_pos_state = None
        final_leverage_to_record = int(leverage) 

        if is_adding:
            current_pos_state = self.position_manager.get_position_state(symbol)
            if not current_pos_state or current_pos_state.get('side') != side:
                self.logger.error(f"!!! {self.mode_str} {action_type} 失败: 方向 ({side}) 与现有 ({current_pos_state.get('side') if current_pos_state else 'N/A'}) 不符。将覆盖。")
                is_adding = False; current_pos_state = None
            else:
                current_leverage = current_pos_state.get('leverage')
                if current_leverage and isinstance(current_leverage, (int, float)) and current_leverage > 0:
                    self.logger.warning(f"{self.mode_str} {action_type}: 检测到现有杠杆 {current_leverage}x。将忽略 AI 请求的 {leverage}x 并使用现有杠杆。")
                    final_leverage_to_record = int(current_leverage) 
                else:
                    self.logger.error(f"{self.mode_str} {action_type}: 无法获取现有杠杆！将回退使用 AI 杠杆 {leverage}x。")

        try:
            raw_exchange = self.client.exchange
            if not raw_exchange.markets: await self.client.load_markets()
            market = raw_exchange.markets.get(symbol);
            if not market: raise ValueError(f"无市场信息 {symbol}")
            ticker = await self.client.fetch_ticker(symbol); current_price = ticker.get('last')
            if not current_price or current_price <= 0: raise ValueError(f"无有效价格 {symbol}")

            required_margin_initial = (size * current_price) / final_leverage_to_record
            if required_margin_initial <= 0: raise ValueError("保证金无效 (<= 0)")
            
            max_allowed_margin = self.cash * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
            if max_allowed_margin <= 0: raise ValueError(f"最大允许保证金无效 (<= 0), 可用现金: {self.cash}")
            
            adjusted_size = size; required_margin_final = required_margin_initial
            
            if required_margin_initial > max_allowed_margin:
                self.logger.warning(f"!!! {self.mode_str} {action_type} 保证金超限 ({required_margin_initial:.2f} > {max_allowed_margin:.2f})，缩减 !!!")
                adj_size_raw = (max_allowed_margin * final_leverage_to_record) / current_price 
                adjusted_size = float(raw_exchange.amount_to_precision(symbol, adj_size_raw))
                min_amount = market.get('limits', {}).get('amount', {}).get('min')
                if min_amount is not None and adjusted_size < min_amount:
                     self.logger.error(f"!!! {self.mode_str} {action_type} 缩减后过小 ({adjusted_size} < {min_amount})，取消 !!!")
                     await send_bark_notification(f"⚠️ {self.mode_str} AI {action_type} 被拒", f"品种: {symbol}\n原因: 缩减后过小"); return
                self.logger.warning(f"缩减后 Size: {adjusted_size}")
                required_margin_final = (adjusted_size * current_price) / final_leverage_to_record
            
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

            if not is_adding:
                self.logger.debug(f"{self.mode_str} {action_type}: 正在设置 AI 杠杆 {final_leverage_to_record}x。")
                await self.client.set_leverage(final_leverage_to_record, symbol)
            else:
                self.logger.info(f"{self.mode_str} {action_type}: 正在使用现有杠杆 {final_leverage_to_record}x，不发送 set_leverage。")

            exchange_side = 'BUY' if side == 'long' else 'SELL'
            order_result = await self.client.create_market_order(symbol, exchange_side, adjusted_size)
            
            entry_price = float(order_result.get('average', order_result.get('price')))
            if not entry_price or entry_price <= 0: entry_price = float(order_result['price'])
            filled_size = float(order_result['filled']); timestamp = int(order_result['timestamp'])
            
            if filled_size <= 0:
                self.logger.error(f"!!! {self.mode_str} {action_type} 失败: 交易所返回成交量为 0 (Filled=0)。")
                return

            fee = await self._parse_fee_from_order(order_result, symbol)
            success = False
            
            if is_adding: 
                success = self.position_manager.add_entry(symbol=symbol, entry_price=entry_price, size=filled_size, entry_fee=fee, leverage=final_leverage_to_record, stop_loss=stop_loss, take_profit=take_profit, timestamp=timestamp, invalidation_condition=invalidation_condition)
            else: 
                self.position_manager.open_position(symbol=symbol, side=side, entry_price=entry_price, size=filled_size, entry_fee=fee, leverage=final_leverage_to_record, stop_loss=stop_loss, take_profit=take_profit, timestamp=timestamp, reason=reason, invalidation_condition=invalidation_condition); success = True 
            
            if success:
                 self.logger.warning(f"!!! {self.mode_str} {action_type} 成功: {side.upper()} {filled_size} {symbol} @ {entry_price} (Fee: {fee}) | AI原因: {reason}")
                 title = f"📈 {self.mode_str} AI {action_type}: {side.upper()} {symbol.split('/')[0]}"
                 final_pos_state = self.position_manager.get_position_state(symbol)
                 final_avg = final_pos_state.get('avg_entry_price', entry_price) if final_pos_state else entry_price
                 final_size = final_pos_state.get('total_size', filled_size) if final_pos_state else filled_size
                 body = f"价格: {entry_price:.4f}\n数量: {filled_size}\n杠杆: {final_leverage_to_record}x\n手续费: {fee:.4f}\n保证金: {required_margin_final:.2f}\nTP/SL: {take_profit}/{stop_loss}"
                 if is_adding: body += f"\n新均价: {final_avg:.4f}\n总数量: {final_size:.4f}"
                 body += f"\nAI原因: {reason}";
                 if adjusted_size != size: body += f"\n(请求 {size} 缩减至 {filled_size})"
                 await send_bark_notification(title, body); await self.sync_state()
            else: raise RuntimeError(f"{action_type} 失败但未抛异常")
        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} {action_type} 失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: self.logger.error(f"!!! {self.mode_str} {action_type} 失败: {e}", exc_info=True); await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n错误: {e}")

    # --- [GEMINI V4 修复] 升级此函数，使其支持 "限价加仓" (Pyramiding) 并存储 "limit_price" ---
    async def live_open_limit(self, symbol, side, size, leverage, limit_price: float, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        """[实盘] 挂一个限价开仓单，并将 SL/TP/Leverage 计划存储起来。
        [GEMINI V3 修复] 此函数现在支持对同向持仓进行限价加仓。
        [GEMINI V4 修复] 此函数现在存储 'limit_price'。
        [V45.38 修复] 此函数现在调用持久化方法。
        """
        action_type = "限价开仓" # 默认为开新仓
        self.logger.warning(f"!!! {self.mode_str} AI 请求 {action_type} (初步): {side.upper()} {size} {symbol} @ {limit_price} !!!")
        
        # --- [GEMINI V3 修复] ---
        # 检查是否已有持仓，以区分 "开仓" 和 "加仓"
        if self.position_manager.is_open(symbol):
            pos_state = self.position_manager.get_position_state(symbol)
            
            if pos_state and pos_state.get('side') == side:
                # 1. 方向一致：这是允许的 "限价加仓" (Pyramiding)
                action_type = "限价加仓"
                self.logger.warning(f"!!! {self.mode_str} AI 请求 {action_type}: {side.upper()} {size} {symbol} @ {limit_price} !!!")
                
                # 关键：加仓时，必须强制使用现有杠杆，忽略 AI 请求的杠杆
                current_leverage = pos_state.get('leverage')
                if current_leverage and int(current_leverage) > 0:
                    if int(leverage) != int(current_leverage):
                         self.logger.warning(f"{action_type}: AI 请求杠杆 {leverage}x, 但将强制使用现有杠杆 {current_leverage}x 以规避 -4161 错误。")
                         leverage = int(current_leverage) # 强制覆盖
                else:
                    self.logger.error(f"{action_type}: 无法获取 {symbol} 的现有杠杆！将冒险使用 AI 请求的 {leverage}x。")
            
            else:
                # 2. 方向相反：这是 "对冲"，我们不允许
                self.logger.error(f"!!! {self.mode_str} 限价单失败: {symbol} 已有 *相反* 持仓 (已有 {pos_state.get('side')}, 请求 {side})。")
                await send_bark_notification(f"❌ {self.mode_str} AI 限价单失败", f"品种: {symbol}\n原因: 已有相反持仓")
                return
        # --- [GEMINI V3 修复结束] ---

        try:
            # --- (此处的逻辑与您 V45.36 版的 live_open_limit 相同) ---
            # --- 检查是否已有旧的限价单，有则取消 ---
            if symbol in self.pending_limit_orders:
                # [V45.38 修复]
                old_plan = await self.remove_pending_limit_order(symbol)
                old_order_id = old_plan.get('order_id') if old_plan else None
                if old_order_id:
                    self.logger.warning(f"{self.mode_str} {action_type}: 发现旧的待处理订单 {old_order_id}。正在取消...")
                    try:
                        await self.client.cancel_order(old_order_id, symbol)
                        self.logger.info(f"成功取消旧订单 {old_order_id}。")
                    except OrderNotFound:
                        self.logger.info(f"旧订单 {old_order_id} 已不在交易所 (可能已成交或已取消)。")
                    except Exception as e_cancel:
                        self.logger.error(f"取消旧订单 {old_order_id} 失败: {e_cancel}。继续尝试设置新订单...")

            # --- (此处的计算逻辑与您 V45.36 版的 live_open_limit 相同) ---
            raw_exchange = self.client.exchange
            if not raw_exchange.markets: await self.client.load_markets()
            market = raw_exchange.markets.get(symbol);
            if not market: raise ValueError(f"无市场信息 {symbol}")

            # [GEMINI V3 修复] 杠杆 (leverage) 变量可能在上面 "加仓" 逻辑中被修改了
            required_margin_initial = (size * limit_price) / leverage
            if required_margin_initial <= 0: raise ValueError(f"保证金无效 (<= 0) | Size: {size}, Price: {limit_price}, Lev: {leverage}")

            max_allowed_margin = self.cash * futures_settings.MAX_MARGIN_PER_TRADE_RATIO
            if max_allowed_margin <= 0: raise ValueError(f"最大允许保证金无效 (<= 0), 可用现金: {self.cash}")

            adjusted_size = size; required_margin_final = required_margin_initial

            if required_margin_initial > max_allowed_margin:
                self.logger.warning(f"!!! {self.mode_str} {action_type} 保证金超限 ({required_margin_initial:.2f} > {max_allowed_margin:.2f})，缩减 !!!")
                adj_size_raw = (max_allowed_margin * leverage) / limit_price 
                adjusted_size = float(raw_exchange.amount_to_precision(symbol, adj_size_raw))
                min_amount = market.get('limits', {}).get('amount', {}).get('min')
                if min_amount is not None and adjusted_size < min_amount:
                     self.logger.error(f"!!! {self.mode_str} {action_type} 缩减后过小 ({adjusted_size} < {min_amount})，取消 !!!")
                     await send_bark_notification(f"⚠️ {self.mode_str} AI {action_type} 被拒", f"品种: {symbol}\n原因: 缩减后过小"); return
                self.logger.warning(f"缩减后 Size: {adjusted_size}")
                required_margin_final = (adjusted_size * limit_price) / leverage

            final_notional_value = adjusted_size * limit_price
            if final_notional_value < self.MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK:
                self.logger.error(f"!!! {self.mode_str} {action_type} 最终名义价值检查失败 !!!")
                self.logger.error(f"最终名义价值 {final_notional_value:.4f} USDT < 阈值 {self.MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK} USDT。取消。")
                await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 最终名义价值过低 (<{self.MIN_NOTIONAL_VALUE_USDT_FINAL_CHECK} USDT)"); return

            estimated_fee = adjusted_size * limit_price * market.get('taker', self.FEE_RATE)
            if self.cash < required_margin_final + estimated_fee:
                 self.logger.error(f"!!! {self.mode_str} {action_type} 现金不足 !!! (需 {required_margin_final + estimated_fee:.2f}, 可用 {self.cash:.2f})")
                 await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 现金不足"); return

            # --- (此处的下单逻辑与您 V45.36 版的 live_open_limit 相同) ---
            await self.client.set_margin_mode(futures_settings.FUTURES_MARGIN_MODE, symbol)
            
            # [GEMINI V3 修复] 杠杆 (leverage) 变量可能已被修改
            # 只有在不是加仓时才设置杠杆 (加仓时杠杆已匹配)
            if action_type == "限价开仓":
                 await self.client.set_leverage(leverage, symbol)
            else:
                 self.logger.info(f"{action_type}: 正在使用现有杠杆 {leverage}x，不发送 set_leverage。")


            exchange_side = 'BUY' if side == 'long' else 'SELL'
            
            order_result = await self.client.create_limit_order(symbol, exchange_side, adjusted_size, limit_price)
            
            order_id = order_result.get('id')
            if not order_id:
                raise ValueError(f"交易所未返回 order_id: {order_result}")

            # --- [GEMINI V4 修复：存储 limit_price] ---
            pending_plan = {
                'order_id': order_id,
                'side': side,
                'leverage': int(leverage), # <-- [GEMINI V3 修复] 存储最终使用的杠杆
                'limit_price': limit_price, # <-- [GEMINI V4 修复] 存储挂单价格
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'invalidation_condition': invalidation_condition,
                'reason': reason,
                'timestamp': time.time() * 1000 
            }
            # --- [修复结束] ---
            
            # [V45.38 修复]
            await self.add_pending_limit_order(symbol, pending_plan)
            
            self.logger.warning(f"!!! {self.mode_str} {action_type} 挂单成功: {side.upper()} {adjusted_size} {symbol} @ {limit_price} (Order ID: {order_id})")
            self.logger.info(f"    SL: {stop_loss}, TP: {take_profit}, Inval: {invalidation_condition}")
            
            # --- [V45.36 修复：添加挂单通知] ---
            title_prefix = "⌛" if action_type == "限价开仓" else "🔼" # 开仓用沙漏, 加仓用箭头
            title = f"{title_prefix} {self.mode_str} AI {action_type}: {side.upper()} {symbol.split('/')[0]}"
            body = f"价格: {limit_price:.4f}\n数量: {adjusted_size}\n杠杆: {leverage}x\nTP/SL: {take_profit}/{stop_loss}\nAI原因: {reason}"
            if adjusted_size != size: body += f"\n(请求 {size} 缩减至 {adjusted_size})"
            await send_bark_notification(title, body)
            # --- [修复结束] ---

        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} {action_type} 失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: 
            self.logger.error(f"!!! {self.mode_str} {action_type} 失败: {e}", exc_info=True); 
            await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n错误: {e}")
            # [V45.38 修复]
            await self.remove_pending_limit_order(symbol)
    # --- [V45.34/36 修复结束] ---


    async def live_partial_close(self, symbol: str, size_percent: Optional[float] = None, size_absolute: Optional[float] = None, reason: str = "N/A"):
        # ... (此函数无变化) ...
        self.logger.warning(f"!!! {self.mode_str} AI 请求部分平仓: {symbol} | %: {size_percent} | Abs: {size_absolute} | 原因: {reason} !!!")

        pos_state = self.position_manager.get_position_state(symbol)
        if not pos_state or pos_state.get('total_size', 0) <= 0:
            self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 本地无有效持仓 {symbol}。")
            return

        current_total_size = pos_state['total_size']
        size_to_close = 0.0
        if size_percent is not None and 0 < size_percent <= 1: # 允许 1.0 (100%)
            if abs(size_percent - 1.0) < 1e-9:
                 self.logger.warning(f"{self.mode_str} 部分平仓请求 100%，转为全平。")
                 await self.live_close(symbol, reason=f"{reason} (转为全平)")
                 return
            size_to_close = current_total_size * size_percent
        elif size_absolute is not None and 0 < size_absolute <= current_total_size + 1e-9: 
             if abs(size_absolute - current_total_size) < 1e-9:
                 self.logger.warning(f"{self.mode_str} 部分平仓请求绝对数量等于全仓，转为全平。")
                 await self.live_close(symbol, reason=f"{reason} (转为全平)")
                 return
             size_to_close = min(size_absolute, current_total_size) 
        else: self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 无效数量参数..."); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n原因: 无效数量参数"); return

        try:
             raw_exchange = self.client.exchange
             if not raw_exchange.markets: await self.client.load_markets()
             market = raw_exchange.markets.get(symbol)
             if not market: raise ValueError(f"无法找到市场信息 {symbol}")
             size_to_close = float(raw_exchange.amount_to_precision(symbol, size_to_close))
             min_amount = market.get('limits', {}).get('amount', {}).get('min')
             if min_amount is not None and size_to_close < min_amount:
                 if size_to_close > 1e-9:
                      if current_total_size - size_to_close < min_amount: 
                           self.logger.warning(f"{self.mode_str} 部分平仓 {symbol}: 计算量 {size_to_close} < 最小量 {min_amount} 且剩余量也小，转为全平。")
                           await self.live_close(symbol, reason=f"{reason} (转为全平)")
                           return
                      else:
                           self.logger.warning(f"{self.mode_str} 部分平仓 {symbol}: 计算量 {size_to_close} < 最小量 {min_amount}，尝试平最小量。")
                           size_to_close = min_amount
                 else: 
                      self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 数量过小 ({size_to_close})"); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n原因: 平仓数量过小"); return
             if size_to_close <= 0: self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 计算数量为 0"); return
        except Exception as e: self.logger.error(f"!!! {self.mode_str} 部分平仓失败 (检查数量时出错): {e}", exc_info=True); return

        try:
            internal_side = pos_state['side']; avg_entry_price = pos_state['avg_entry_price']
            open_fee_total = pos_state['total_entry_fee']; leverage = pos_state.get('leverage', 0)
            total_margin = pos_state.get('margin', 0.0) 
            margin_per_unit = total_margin / current_total_size if current_total_size > 0 else 0
            margin_for_this_part = margin_per_unit * size_to_close

            exchange_close_side = 'SELL' if internal_side == 'long' else 'BUY'
            params = {'reduceOnly': True}
            order_result = await self.client.create_market_order(symbol, exchange_close_side, size_to_close, params=params)

            exit_price = float(order_result.get('average', order_result.get('price')))
            if not exit_price or exit_price <= 0: exit_price = float(order_result['price'])
            filled_size = float(order_result['filled']); timestamp = int(order_result['timestamp'])
            
            if filled_size <= 0:
                self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 交易所返回成交量为 0 (Filled=0)。仓位可能过小。")
                return

            close_fee = await self._parse_fee_from_order(order_result, symbol)

            open_fee_for_this_part = (open_fee_total / current_total_size) * filled_size if current_total_size > 0 else 0

            if internal_side == 'long': gross_pnl_part = (exit_price - avg_entry_price) * filled_size
            else: gross_pnl_part = (avg_entry_price - exit_price) * filled_size
            net_pnl_part = gross_pnl_part - open_fee_for_this_part - close_fee

            order_notional = filled_size * exit_price 
            margin_calc_by_order = order_notional / leverage if leverage > 0 else 0.0

            trade_data = {
                'symbol': symbol, 'side': internal_side, 'entry_price': avg_entry_price,
                'exit_price': exit_price, 'size': filled_size,
                'net_pnl': net_pnl_part, 'fees': open_fee_for_this_part + close_fee,
                'margin': margin_for_this_part, 
                'margin_calc_by_order': margin_calc_by_order, 
                'leverage': leverage,
                'open_reason': pos_state.get('entry_reason', 'N/A'), 'close_reason': reason,
                'timestamp': timestamp, 'partial': True
            }

            self.trade_logger.record_trade(trade_data)
            success = self.position_manager.reduce_position(symbol, filled_size)

            if success:
                 updated_pos_state = self.position_manager.get_position_state(symbol)
                 remaining_size = updated_pos_state.get('total_size', 0.0) if updated_pos_state else 0.0

                 self.logger.warning(f"!!! {self.mode_str} 部分平仓成功: {symbol} | 平掉 {filled_size} @ {exit_price:.4f} (Fee: {close_fee}) | 本次净盈亏: {net_pnl_part:.2f} USDT | 剩余 {remaining_size:.8f} | 原因: {reason}") 
                 pnl_prefix = "盈利" if net_pnl_part >= 0 else "亏损"; title = f"💰 {self.mode_str} AI 部分平仓: {pnl_prefix} {abs(net_pnl_part):.2f} USDT"
                 body = (f"品种: {symbol.split('/')[0]}\n方向: {internal_side.upper()}\n平仓价格: {exit_price:.4f}\n平仓数量: {filled_size}\n手续费: {close_fee:.4f}\n剩余数量: {remaining_size:.8f}\n原因: {reason}")
                 await send_bark_notification(title, body); await self.sync_state()
            else: raise RuntimeError("position_manager.reduce_position 返回失败")
        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} 部分平仓失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: self.logger.error(f"!!! {self.mode_str} 部分平仓失败: {e}", exc_info=True); await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n错误: {e}")


    async def live_close(self, symbol, reason: str = "N/A"):
        # ... (此函数无变化) ...
        self.logger.warning(f"!!! {self.mode_str} 正在尝试(全)平仓: {symbol} | 原因: {reason} !!!")
        pos_state = self.position_manager.get_position_state(symbol) 
        if not pos_state or pos_state.get('total_size', 0) <= 0:
            self.logger.error(f"!!! {self.mode_str} (全)平仓失败: 本地无有效持仓 {symbol}。")
            return

        try:
            internal_side = pos_state['side']; size_to_close = pos_state['total_size']
            avg_entry_price = pos_state['avg_entry_price']; open_fee_total = pos_state['total_entry_fee']
            leverage = pos_state.get('leverage', 0); margin_to_record = pos_state.get('margin', 0.0) 
            entry_reason = pos_state.get('entry_reason', 'N/A')

            exchange_close_side = 'SELL' if internal_side == 'long' else 'BUY'
            params = {'reduceOnly': True}
            order_result = await self.client.create_market_order(symbol, exchange_close_side, size_to_close, params=params)

            exit_price = float(order_result.get('average', order_result.get('price')))
            if not exit_price or exit_price <= 0: exit_price = float(order_result['price'])
            filled_size = float(order_result['filled']); timestamp = int(order_result['timestamp'])

            # --- [ V45.33 核心修复 ] ---
            if filled_size <= 0:
                self.logger.error(f"!!! {self.mode_str} (全)平仓失败: 交易所返回成交量为 0 (Filled=0)。仓位可能过小 (Dust) 或API错误。")
                self.logger.error("!!! 本地状态未改变，等待下一次 sync_state 或风控循环。")
                return
            # --- [ 修复结束 ] ---

            close_fee = await self._parse_fee_from_order(order_result, symbol)

            if internal_side == 'long': gross_pnl = (exit_price - avg_entry_price) * filled_size
            else: gross_pnl = (avg_entry_price - exit_price) * filled_size
            net_pnl = gross_pnl - open_fee_total - close_fee

            order_notional = filled_size * exit_price
            margin_calc_by_order = order_notional / leverage if leverage > 0 else 0.0

            trade_data = {
                'symbol': symbol, 'side': internal_side, 'entry_price': avg_entry_price,
                'exit_price': exit_price, 'size': filled_size,
                'net_pnl': net_pnl, 'fees': open_fee_total + close_fee,
                'margin': margin_to_record, 
                'margin_calc_by_order': margin_calc_by_order, 
                'leverage': leverage,
                'open_reason': entry_reason, 'close_reason': reason,
                'timestamp': timestamp, 'partial': False
            }

            self.trade_logger.record_trade(trade_data)
            self.position_manager.close_position(symbol)

            self.logger.warning(f"!!! {self.mode_str} (全)平仓成功: {symbol} @ {exit_price:.4f} (Fee: {close_fee}), 净盈亏: {net_pnl:.2f} USDT | 原因: {reason}")
            pnl_prefix = "盈利" if net_pnl >= 0 else "亏损"
            title = f"📉 {self.mode_str} AI (全)平仓: {pnl_prefix} {abs(net_pnl):.2f} USDT"
            body = f"品种: {symbol.split('/')[0]}\n方向: {internal_side.upper()}\n平仓价格: {exit_price:.4f}\n手续费: {close_fee:.4f}\n原因: {reason}"
            await send_bark_notification(title, body); await self.sync_state()

        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} (全)平仓失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI (全)平仓失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: self.logger.error(f"!!! {self.mode_str} (全)平仓失败: {e}", exc_info=True); await send_bark_notification(f"❌ {self.mode_str} AI (全)平仓失败", f"品种: {symbol}\n错误: {e}")


    async def paper_open(self, symbol, side, size, price, leverage, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        # ... (此函数无变化) ...
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
        # ... (此函数无变化) ...
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
        # ... (此函数无变化) ...
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

    async def _parse_fee_from_order(self, order_result: dict, symbol: str) -> float:
        # ... (此函数无变化) ...
        fees_paid_usdt = 0.0
        if not order_result: return fees_paid_usdt

        self.logger.debug(f"Fee Parsing Debug: Raw order_result for {symbol}: {order_result}")

        fee_currency = None
        fee_cost = None

        if 'fee' in order_result and isinstance(order_result['fee'], dict):
            fee_info = order_result['fee']
            if 'cost' in fee_info and 'currency' in fee_info:
                try: 
                    fee_cost = float(fee_info['cost'])
                    fee_currency = fee_info['currency']
                    self.logger.debug(f"Fee Parsing: Found 'fee': {fee_cost} {fee_currency}")
                except (ValueError, TypeError): 
                    self.logger.warning(f"无法解析 'fee.cost': {fee_info}"); fee_cost = None
        elif 'fees' in order_result and isinstance(order_result['fees'], list) and len(order_result['fees']) > 0:
            first_valid_fee = next((f for f in order_result['fees'] if f and 'cost' in f and 'currency' in f), None)
            if first_valid_fee:
                 try:
                    fee_cost = float(first_valid_fee['cost'])
                    fee_currency = first_valid_fee['currency']
                    if len(order_result['fees']) > 1: 
                        self.logger.warning(f"{symbol} 含多个费用条目，仅处理第一个: {order_result['fees']}")
                    self.logger.debug(f"Fee Parsing: Found 'fees' list: {fee_cost} {fee_currency}")
                 except (ValueError, TypeError) as e: 
                    self.logger.warning(f"解析 'fees'列表出错: {e}"); fee_cost = None
            else: 
                self.logger.warning(f"{symbol} 'fees'列表为空或缺字段: {order_result['fees']}")

        if fee_cost is not None and fee_currency is not None:
            if fee_currency == 'USDT':
                fees_paid_usdt = fee_cost
                self.logger.debug(f"Fee Parsing: Fee is USDT: {fees_paid_usdt}")
            
            elif fee_currency == 'BNB':
                self.logger.warning(f"检测到 {symbol} 手续费以 BNB 支付: {fee_cost} BNB。尝试获取 BNB/USDT:USDT 价格进行转换...")
                
                bnb_contract_symbol = 'BNB/USDT:USDT' 
                
                try:
                    if bnb_contract_symbol not in self.symbols:
                        self.logger.error(f"BNB 手续费转换失败: '{bnb_contract_symbol}' 不在 self.symbols 列表中。")
                        fees_paid_usdt = 0.0 
                    else:
                        bnb_ticker = await self.client.fetch_ticker(bnb_contract_symbol) 
                        bnb_price = bnb_ticker.get('last')
                        
                        if bnb_price and bnb_price > 0:
                            fees_paid_usdt = fee_cost * bnb_price
                            self.logger.warning(f"BNB 手续费已转换为 USDT: {fee_cost:.6f} BNB * {bnb_price} USD/BNB = {fees_paid_usdt:.4f} USDT")
                        else:
                            self.logger.error(f"无法获取有效的 {bnb_contract_symbol} 价格，BNB 手续费将记录为 0 USDT。")
                            fees_paid_usdt = 0.0
                
                except ExchangeError as e:
                     self.logger.error(f"获取 {bnb_contract_symbol} ticker 时交易所错误: {e}。BNB 手续费将记录为 0 USDT。")
                     fees_paid_usdt = 0.0
                except Exception as e:
                    self.logger.error(f"获取 {bnb_contract_symbol} 价格或转换时发生意外错误: {e}。BNB 手续费将记录为 0 USDT。", exc_info=True)
                    fees_paid_usdt = 0.0
                    
            else: 
                self.logger.warning(f"检测到 {symbol} 手续费以非 USDT/BNB 币种支付: {fee_cost} {fee_currency}。将记录为 0 USDT。")
                fees_paid_usdt = 0.0 
        else:
            self.logger.warning(f"未能从 {symbol} 订单结果解析费用。将使用 0.0 USDT。")

        return fees_paid_usdt

    @property
    def equity_history(self):
        return self.paper_equity_history

    @property
    def trade_history(self):
        if self.is_live: return self.trade_logger.get_history()
        else: return self.paper_trade_history

    async def update_position_rules(self, symbol: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None, invalidation_condition: Optional[str] = None, reason: str = "AI update"):
        # ... (此函数无变化) ...
        if self.is_live:
            success = self.position_manager.update_rules(symbol, stop_loss, take_profit, invalidation_condition) 
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
