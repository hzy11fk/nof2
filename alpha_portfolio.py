# 文件: alpha_portfolio.py (V-Ultimate + PaperFix + FeeFix + OrphanFix)
# 1. [V45.40 修复] get_state_for_prompt 现已支持 'filter_rule8' 参数。
# 2. [V-Ultimate BUG 修复] sync_state (实盘) 现在会根据实际成交价重新计算 SL/TP，防止“有毒”仓位。
# 3. [V-Ultimate PaperFix] __init__, _load_pending_limits, _save_pending_limits 现在在所有模式下都运行。
# 4. [V-Ultimate PaperFix] 新增 paper_open_limit 函数，用于接收模拟盘的 AI 限价单计划。
# 5. [V-Ultimate PaperFix] sync_state (模拟盘) 现在会检查 pending_limit_orders 并模拟限价单成交。
# 6. [FEE FIX (User)] _parse_fee_from_order 现已修复 BNB 换算逻辑。
# 7. [FEE FIX (User)] sync_state 现已修复限价单手续费获取逻辑 (不再是 0.0)。
# 8. [ORPHAN FIX (User)] 所有平仓函数 (live_close, live_partial_close, paper_close, paper_partial_close) 现在会自动取消待处理的限价单。

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

        self.pending_limit_orders: Dict[str, Dict] = {}
        self.pending_limits_file = os.path.join(futures_settings.FUTURES_STATE_DIR, 'alpha_pending_limits.json')
        # [V-Ultimate PaperFix] 修复 1: 始终加载 pending_limits
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

    def _load_pending_limits(self):
        # [V-Ultimate PaperFix] 修复 2a: 移除 'if not self.is_live: return'
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
        # [V-Ultimate PaperFix] 修复 2b: 移除 'if not self.is_live: return'
        
        try:
            os.makedirs(os.path.dirname(self.pending_limits_file), exist_ok=True)
            with open(self.pending_limits_file, 'w', encoding='utf-8') as f:
                json.dump(self.pending_limit_orders, f, indent=4, ensure_ascii=False)
            self.logger.debug(f"已保存 {len(self.pending_limit_orders)} 个待处理限价单。")
        except Exception as e:
            self.logger.error(f"保存待处理限价单失败: {e}", exc_info=True)

    async def add_pending_limit_order(self, symbol: str, plan: Dict):
        self.pending_limit_orders[symbol] = plan
        await self._save_pending_limits()

    async def remove_pending_limit_order(self, symbol: str) -> Optional[Dict]:
        plan = self.pending_limit_orders.pop(symbol, None)
        await self._save_pending_limits()
        return plan


    async def sync_state(self):
        """
        [V-Ultimate PaperFix] 模拟盘逻辑现在会检查并模拟成交 pending_limit_orders。
        [V-Ultimate BUG 修复] 实盘逻辑现在会根据实际成交价重新计算 SL/TP。
        [FEE FIX (User)] 实盘逻辑现在会获取已成交限价单的实际手续费。
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
                        
                        # [V45.39 修复] 迭代副本，但不删除
                        for symbol, plan in list(self.pending_limit_orders.items()):
                            plan_order_id = plan.get('order_id')
                            if not plan_order_id:
                                self.logger.warning(f"Sync: 待处理计划 {symbol} 缺少 order_id，已移除。")
                                await self.remove_pending_limit_order(symbol)
                                continue

                            if plan_order_id not in open_order_ids:
                                # [V45.39 修复] 只记录日志，不删除
                                self.logger.debug(f"Sync: 待处理订单 {plan_order_id} ({symbol}) 不再 'open'。等待持仓同步...")
                                # await self.remove_pending_limit_order(symbol) # <--- [V45.39] 已修复：注释掉此行
                    # --- [V45.37 步骤 1 结束] ---

                    real_positions = await self.client.fetch_positions(self.symbols); exchange_open_symbols = set()
                    for pos in real_positions:
                        symbol = pos.get('symbol')
                        if symbol in self.symbols:
                            size_str = pos.get('contracts') or pos.get('contractSize'); size = float(size_str) if size_str else 0.0
                            side = pos.get('side').lower() if pos.get('side') else None
                            if abs(size) > 1e-9:
                                exchange_open_symbols.add(symbol)
                                
                                pending_plan = await self.remove_pending_limit_order(symbol)
                                
                                if not self.position_manager.is_open(symbol):
                                    # --- [V45.36 策略A 步骤 2: 修复杠杆和通知] ---
                                    self.logger.warning(f"{self.mode_str} sync: 发现交易所新持仓 {symbol}，正在同步到本地...")
                                    
                                    entry_str = pos.get('entryPrice') or pos.get('basePrice'); entry = float(entry_str) if entry_str else 0.0 #
                                    
                                    plan_reason = "live_sync"
                                    plan_sl = None
                                    plan_tp = None
                                    plan_inval = None
                                    
                                    exchange_lev_val = pos.get('leverage')
                                    final_leverage = int(exchange_lev_val) if exchange_lev_val is not None and float(exchange_lev_val) > 0 else 1

                                    # --- [FEE FIX START (新开仓)] ---
                                    calculated_entry_fee = 0.0 # 默认手续费
                                    
                                    if pending_plan:
                                        order_id = pending_plan.get('order_id')
                                        if order_id:
                                            try:
                                                self.logger.info(f"Sync: 正在为新持仓 {symbol} (Order ID: {order_id}) 获取成交手续费...")
                                                # 从交易所获取已成交订单的详情
                                                order_result = await self.client.fetch_order(order_id, symbol) 
                                                
                                                if order_result and order_result.get('status') in ['closed', 'filled']:
                                                    # 调用您已有的手续费解析函数
                                                    calculated_entry_fee = await self._parse_fee_from_order(order_result, symbol)
                                                    self.logger.warning(f"Sync: 成功获取 {symbol} (Order ID: {order_id}) 的手续费: {calculated_entry_fee:.4f} USDT")
                                                else:
                                                    self.logger.warning(f"Sync: 无法从 {order_id} (Status: {order_result.get('status') if order_result else 'N/A'}) 获取手续费，将使用 0.0。")
                                            
                                            except Exception as e_fetch_fee:
                                                self.logger.error(f"Sync: 尝试为 {order_id} 获取手续费时出错: {e_fetch_fee}。将使用 0.0。")
                                        else:
                                            self.logger.warning(f"Sync: 匹配到AI计划，但计划中无 Order ID。手续费将为 0.0。")
                                    else:
                                        self.logger.warning(f"Sync: 新持仓 {symbol} 未匹配到AI计划。手续费将为 0.0。")
                                    # --- [FEE FIX END] ---

                                    # --- [V-Ultimate BUG 修复：重新计算 SL/TP] ---
                                    if pending_plan:
                                        self.logger.warning(f"Sync: 新持仓 {symbol} 匹配到一个AI限价单计划。正在应用 SL/TP/Reason...") #
                                        plan_reason = pending_plan.get('reason', 'live_sync_with_plan')
                                        plan_inval = pending_plan.get('invalidation_condition') #
                                        
                                        try:
                                            plan_limit_price = pending_plan.get('limit_price') #
                                            original_sl = pending_plan.get('stop_loss') #
                                            original_tp = pending_plan.get('take_profit') #
                                            plan_side = pending_plan.get('side') #
                                            
                                            # 'entry' 是从交易所获取的实际成交均价
                                            
                                            if plan_limit_price and original_sl and original_tp and plan_side == side:
                                                self.logger.info(f"Sync: 正在为 {symbol} 重新计算 SL/TP。")
                                                self.logger.info(f"Sync: 实际成交价: {entry} (计划价: {plan_limit_price})")
                                                
                                                risk_distance = 0.0
                                                reward_distance = 0.0

                                                if side == 'long':
                                                    # 计算原始的风险/回报“距离”
                                                    risk_distance = plan_limit_price - original_sl #
                                                    reward_distance = original_tp - plan_limit_price
                                                    
                                                    # 将“距离”应用到新的实际成交价上
                                                    plan_sl = entry - risk_distance #
                                                    plan_tp = entry + reward_distance #
                                                    
                                                elif side == 'short':
                                                    # 计算原始的风险/回报“距离”
                                                    risk_distance = original_sl - plan_limit_price
                                                    reward_distance = plan_limit_price - original_tp
                                                    
                                                    # 将“距离”应用到新的实际成交价上
                                                    plan_sl = entry + risk_distance
                                                    plan_tp = entry - reward_distance

                                                # 最终安全检查：确保新的SL是有效的
                                                if (side == 'long' and plan_sl >= entry) or (side == 'short' and plan_sl <= entry):
                                                    self.logger.error(f"Sync: 重新计算的 SL ({plan_sl}) 对成交价 ({entry}) 无效！")
                                                    self.logger.error("Sync: 这可能是由于止损距离为0或负数。将使用原始SL值作为回退。")
                                                    plan_sl = original_sl # 回退
                                                else:
                                                    self.logger.warning(f"Sync: SL/TP 已重新计算。")
                                                    self.logger.warning(f"Sync: 原始 SL/TP: {original_sl}/{original_tp} -> 新 SL/TP: {plan_sl}/{plan_tp}")
                                                
                                            else:
                                                # 如果缺少数据或边不匹配，回退到旧的（有风险的）逻辑
                                                self.logger.warning(f"Sync: 无法重新计算 SL/TP (缺少数据或边不匹配)。使用原始计划值。")
                                                plan_sl = pending_plan.get('stop_loss') #
                                                plan_tp = pending_plan.get('take_profit') #

                                        except Exception as e_recalc:
                                            self.logger.error(f"Sync: 重新计算 SL/TP 时出错: {e_recalc}。将使用原始计划值。")
                                            plan_sl = pending_plan.get('stop_loss') #
                                            plan_tp = pending_plan.get('take_profit') #
                                        # --- [V-Ultimate BUG 修复结束] ---
                                        
                                        plan_leverage = pending_plan.get('leverage')
                                        if plan_leverage and isinstance(plan_leverage, (int, float)) and plan_leverage > 0:
                                            self.logger.info(f"Sync: 使用AI计划的杠杆 {plan_leverage}x (交易所报告为 {exchange_lev_val}x)")
                                            final_leverage = int(plan_leverage)
                                        else:
                                            self.logger.warning(f"Sync: AI计划中无杠杆，使用交易所杠杆 {final_leverage}x")

                                        try:
                                            title = f"✅ {self.mode_str} AI 限价单成交: {side.upper()} {symbol.split('/')[0]}"
                                            body = f"成交价格: {entry:.4f}\n数量: {abs(size)}\n杠杆: {final_leverage}x\nTP/SL: {plan_tp}/{plan_sl}\nAI原因: {plan_reason}\n手续费: {calculated_entry_fee:.4f} USDT" # [FEE FIX] 添加手续费到通知
                                            await send_bark_notification(title, body)
                                        except Exception as e_notify:
                                            self.logger.error(f"Sync: 发送成交通知失败: {e_notify}")
                                    
                                    else:
                                        self.logger.warning(f"Sync: 新持仓 {symbol} 未匹配到AI计划，使用默认值同步 (杠杆 {final_leverage}x)。")

                                    self.position_manager.open_position( #
                                        symbol=symbol, 
                                        side=side, 
                                        entry_price=entry, 
                                        size=abs(size), 
                                        entry_fee=calculated_entry_fee, # <--- [FEE FIX] 应用获取到的手续费
                                        leverage=final_leverage, 
                                        stop_loss=plan_sl, 
                                        take_profit=plan_tp, 
                                        timestamp=int(pos.get('timestamp', time.time()*1000)), 
                                        reason=plan_reason, 
                                        invalidation_condition=plan_inval
                                    )
                                    # --- [V45.36 步骤 2 结束] ---
                                    
                                elif pending_plan:
                                    # --- [GEMINI V5 修复] 处理限价加仓 ---
                                    self.logger.warning(f"{self.mode_str} sync: 发现交易所持仓 {symbol} 变动，匹配到AI限价加仓计划。")
                                    
                                    entry_str = pos.get('entryPrice') or pos.get('basePrice');
                                    current_avg_price = float(entry_str) if entry_str else 0.0
                                    current_total_size = abs(size)
                                    
                                    old_state = self.position_manager.get_position_state(symbol)
                                    old_total_size = old_state.get('total_size', 0.0) if old_state else 0.0
                                    
                                    added_size = current_total_size - old_total_size
                                    
                                    if added_size > 1e-9:
                                        self.logger.info(f"Sync: 本次加仓 {added_size} (Exch: {current_total_size}, Local: {old_total_size})")
                                        
                                        old_avg_price = old_state.get('avg_entry_price', 0.0) if old_state else 0.0
                                        
                                        add_price = 0.0
                                        if added_size > 0:
                                            add_price = ((current_avg_price * current_total_size) - (old_avg_price * old_total_size)) / added_size
                                        
                                        if add_price <= 0:
                                             self.logger.warning(f"Sync: 无法反推加仓价格 (AddPrice: {add_price})。将使用交易所均价 {current_avg_price} 作为近似值。")
                                             add_price = current_avg_price
                                        
                                        # --- [FEE FIX START (加仓)] ---
                                        calculated_entry_fee = 0.0 # 默认手续费
                                        order_id = pending_plan.get('order_id')
                                        if order_id:
                                            try:
                                                self.logger.info(f"Sync (Add): 正在为 {symbol} (Order ID: {order_id}) 获取成交手续费...")
                                                order_result = await self.client.fetch_order(order_id, symbol) 
                                                
                                                if order_result and order_result.get('status') in ['closed', 'filled']:
                                                    calculated_entry_fee = await self._parse_fee_from_order(order_result, symbol)
                                                    self.logger.warning(f"Sync (Add): 成功获取 {symbol} (Order ID: {order_id}) 的手续费: {calculated_entry_fee:.4f} USDT")
                                                else:
                                                    self.logger.warning(f"Sync (Add): 无法从 {order_id} (Status: {order_result.get('status') if order_result else 'N/A'}) 获取手续费，将使用 0.0。")
                                            
                                            except Exception as e_fetch_fee_add:
                                                self.logger.error(f"Sync (Add): 尝试为 {order_id} 获取手续费时出错: {e_fetch_fee_add}。将使用 0.0。")
                                        else:
                                            self.logger.warning(f"Sync (Add): 匹配到AI计划，但计划中无 Order ID。手续费将为 0.0。")
                                        # --- [FEE FIX END (加仓)] ---


                                        # [V-Ultimate BUG 修复] 加仓也需要重新计算 SL/TP
                                        plan_sl = None
                                        plan_tp = None
                                        try:
                                            plan_limit_price = pending_plan.get('limit_price')
                                            original_sl = pending_plan.get('stop_loss')
                                            original_tp = pending_plan.get('take_profit')
                                            
                                            if plan_limit_price and original_sl and original_tp:
                                                if side == 'long':
                                                    risk_distance = plan_limit_price - original_sl
                                                    reward_distance = original_tp - plan_limit_price
                                                    plan_sl = add_price - risk_distance
                                                    plan_tp = add_price + reward_distance
                                                elif side == 'short':
                                                    risk_distance = original_sl - plan_limit_price
                                                    reward_distance = plan_limit_price - original_tp
                                                    plan_sl = add_price + risk_distance
                                                    plan_tp = add_price - reward_distance
                                                
                                                self.logger.info(f"Sync (Add): SL/TP 已重新计算。")
                                                self.logger.info(f"Sync (Add): 原始 SL/TP: {original_sl}/{original_tp} -> 新 SL/TP: {plan_sl}/{plan_tp}")
                                            else:
                                                plan_sl = pending_plan.get('stop_loss')
                                                plan_tp = pending_plan.get('take_profit')
                                        except Exception as e_recalc_add:
                                            self.logger.error(f"Sync (Add): 重新计算 SL/TP 时出错: {e_recalc_add}。")
                                            plan_sl = pending_plan.get('stop_loss')
                                            plan_tp = pending_plan.get('take_profit')
                                        # [BUG 修复结束]

                                        plan_inval = pending_plan.get('invalidation_condition')
                                        plan_reason = pending_plan.get('reason', 'live_sync_add_with_plan')
                                        plan_leverage = pending_plan.get('leverage')
                                        
                                        self.position_manager.add_entry(
                                            symbol=symbol,
                                            entry_price=add_price,
                                            size=added_size,
                                            entry_fee=calculated_entry_fee, # <--- [FEE FIX] 应用获取到的手续费
                                            leverage=plan_leverage, 
                                            stop_loss=plan_sl,
                                            take_profit=plan_tp,
                                            timestamp=int(pos.get('timestamp', time.time()*1000)),
                                            invalidation_condition=plan_inval
                                        )
                                        
                                        final_state = self.position_manager.get_position_state(symbol)
                                        final_avg_price = final_state.get('avg_entry_price', current_avg_price)
                                        final_total_size = final_state.get('total_size', current_total_size)

                                        # --- [新逻辑: 移动止损至新的 (含手续费) 成本价] ---
                                        try:
                                            fee_rate = 0.001 # 0.1%
                                            new_breakeven_sl = 0.0
                                            if side == 'long':
                                                new_breakeven_sl = final_avg_price * (1 + fee_rate)
                                            elif side == 'short':
                                                new_breakeven_sl = final_avg_price * (1 - fee_rate)
                                            
                                            if new_breakeven_sl > 0:
                                                self.logger.warning(f"Sync (Add): 正在将止损移动到新的 (含手续费) 成本价: {new_breakeven_sl:.4f}")
                                                # 直接调用 position_manager.update_rules (它在 portfolio 中)
                                                self.position_manager.update_rules(
                                                    symbol, 
                                                    stop_loss=new_breakeven_sl, 
                                                    reason="Pyramiding: SL to new B/E+Fee"
                                                )
                                            else:
                                                self.logger.error(f"Sync (Add): 计算新的保本止损失败 (Price: {new_breakeven_sl:.4f})")
                                        except Exception as e_breakeven:
                                            self.logger.error(f"Sync (Add): 更新止损至新成本价时出错: {e_breakeven}")
                                        # --- [新逻辑结束] ---
                                        
                                        try:
                                            title = f"🔼 {self.mode_str} AI 限价加仓成交: {side.upper()} {symbol.split('/')[0]}"
                                            body = f"成交价格: {add_price:.4f}\n数量: {added_size}\n杠杆: {plan_leverage}x\n新均价: {final_avg_price:.4f}\n新总量: {final_total_size}\nAI原因: {plan_reason}\n手续费: {calculated_entry_fee:.4f} USDT" # [FEE FIX] 添加手续费到通知
                                            await send_bark_notification(title, body)
                                        except Exception as e_notify:
                                            self.logger.error(f"Sync: 发送加仓成交通知失败: {e_notify}")
                                            
                                    else:
                                        self.logger.warning(f"Sync: 匹配到限价单 {symbol}，但计算出的 added_size 为 0 或负数 ({added_size})。不同步加仓。")
                                        await self.add_pending_limit_order(symbol, pending_plan)
                                        
                                else:
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
                
                # --- [V-Ultimate 模拟盘修复] 模拟限价单成交检查 ---
                if self.pending_limit_orders:
                    try:
                        symbols_to_check = list(self.pending_limit_orders.keys())
                        if symbols_to_check: # 仅在有待处理订单时获取 tickers
                            tickers_for_paper = await self.exchange.fetch_tickers(symbols_to_check)
                            
                            # 迭代副本以允许在循环中删除
                            for symbol, plan in list(self.pending_limit_orders.items()):
                                current_price_data = tickers_for_paper.get(symbol)
                                if not current_price_data or not current_price_data.get('last'):
                                    self.logger.warning(f"{self.mode_str} 模拟成交: 无法获取 {symbol} 的市价，跳过。")
                                    continue
                                    
                                current_price = current_price_data.get('last')
                                limit_price = plan.get('limit_price')
                                side = plan.get('side')
                                
                                is_fill = False
                                if side == 'long' and current_price <= limit_price:
                                    self.logger.warning(f"✅ {self.mode_str} 模拟成交: LONG {symbol} 挂单 {limit_price} 已被市价 {current_price} 触发。")
                                    is_fill = True
                                elif side == 'short' and current_price >= limit_price:
                                    self.logger.warning(f"✅ {self.mode_str} 模拟成交: SHORT {symbol} 挂单 {limit_price} 已被市价 {current_price} 触发。")
                                    is_fill = True

                                if is_fill:
                                    # 1. 从待处理中移除
                                    plan = await self.remove_pending_limit_order(symbol)
                                    if not plan: continue # 万一并发
                                    
                                    # 2. [V-Ultimate BUG 修复] 重新计算 SL/TP
                                    entry_price = plan.get('limit_price')
                                    original_sl = plan.get('stop_loss')
                                    original_tp = plan.get('take_profit')
                                    
                                    new_sl = original_sl
                                    new_tp = original_tp
                                    
                                    # 检查市价是否 *好于* 限价 (滑点)
                                    if (side == 'long' and current_price < entry_price) or (side == 'short' and current_price > entry_price):
                                        self.logger.info(f"{self.mode_str} 模拟成交: 成交价 {current_price} 优于挂单价 {entry_price}。使用 {current_price}。")
                                        entry_price = current_price # 获得更好的价格
                                    else:
                                        self.logger.info(f"{self.mode_str} 模拟成交: 成交价 {entry_price} (挂单价)。")

                                    # 重新计算 SL/TP (应用 Bug 修复)
                                    try:
                                        if side == 'long':
                                            risk_distance = plan.get('limit_price') - original_sl
                                            reward_distance = original_tp - plan.get('limit_price')
                                            new_sl = entry_price - risk_distance
                                            new_tp = entry_price + reward_distance
                                        elif side == 'short':
                                            risk_distance = original_sl - plan.get('limit_price')
                                            reward_distance = plan.get('limit_price') - original_tp
                                            new_sl = entry_price + risk_distance
                                            new_tp = entry_price - reward_distance
                                        self.logger.info(f"{self.mode_str} 模拟成交: SL/TP 已重新计算为 {new_sl}/{new_tp} (基于成交价 {entry_price})")
                                    except Exception as e_recalc:
                                        self.logger.error(f"{self.mode_str} 模拟成交: SL/TP 重算失败: {e_recalc}，使用原始值。")

                                    # 3. 调用 paper_open (市价模拟器) 来执行
                                    await self.paper_open(
                                        symbol=symbol,
                                        side=plan.get('side'),
                                        size=plan.get('size'),
                                        price=entry_price, # 使用我们的成交价
                                        leverage=plan.get('leverage'),
                                        reason=plan.get('reason', 'paper_limit_fill'),
                                        stop_loss=new_sl,
                                        take_profit=new_tp,
                                        invalidation_condition=plan.get('invalidation_condition')
                                    )
                    except Exception as e_paper_fill:
                        self.logger.error(f"{self.mode_str} 模拟限价单成交检查失败: {e_paper_fill}", exc_info=True)
                # --- [模拟盘修复结束] ---

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


    # --- [V45.40 修复] ---
    def get_state_for_prompt(self, tickers: dict = None, filter_rule8: bool = False):
        """
        [V45.40 修复]
        1. 增加 filter_rule8=False 参数
        2. 过滤掉 Rule 8 持仓 (如果 filter_rule8=True)
        3. 增加 'pending_limit_orders' 键
        4. 将 'open_positions' 重命名为 'open_positions_rule6'
        """
        position_details = []
        
        if self.is_live:
            if tickers is None: 
                tickers = {}
                self.logger.warning("get_state_for_prompt (live) 未收到 tickers! UPL 将丢失。")

            open_positions = self.position_manager.get_all_open_positions()
            for symbol, state in open_positions.items():
                
                # --- [V45.40 修复] 过滤 Rule 8 持仓 ---
                if filter_rule8:
                    inval_cond = state.get('invalidation_condition') or '' 
                    is_rule_8_trade = "Python Rule 8" in inval_cond
                    if is_rule_8_trade:
                        continue # 跳过 Rule 8 持仓, 不发送给 AI (LLM)
                # --- [修复结束] ---
                
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

        else: # 模拟盘
            for symbol, pos in self.paper_positions.items():
                if pos and isinstance(pos, dict) and pos.get('size', 0) > 0:
                    
                    # --- [V45.40 修复] 过滤 Rule 8 持仓 (模拟盘) ---
                    if filter_rule8:
                        inval_cond = pos.get('invalidation_condition') or '' 
                        is_rule_8_trade = "Python Rule 8" in inval_cond
                        if is_rule_8_trade:
                            continue # 跳过 Rule 8 持仓
                    # --- [修复结束] ---
                    
                    pos_str = ( f"- {symbol.split(':')[0]}: Side={pos['side'].upper()}, Size={pos['size']:.4f}, Entry={pos['entry_price']:.4f}, "
                                f"UPL={pos.get('unrealized_pnl', 0.0):.2f}, TP={pos.get('take_profit', 'N/A')}, SL={pos.get('stop_loss', 'N/A')}, "
                                f"Invalidation='{pos.get('invalidation_condition', 'N/A')}'")
                    position_details.append(pos_str)
        
        if not position_details: position_details.append("No open positions.")
        
        # --- [V-Pending 修复] 新增挂单详情 ---
        pending_orders_details = []
        # (V-Ultimate PaperFix: 现在模拟盘也支持挂单)
        if self.pending_limit_orders:
            for symbol, plan in self.pending_limit_orders.items():
                try:
                    plan_str = ( f"- {symbol.split(':')[0]}: Side={plan.get('side', 'N/A').upper()}, "
                                 f"Price={plan.get('limit_price', 0.0):.4f}, "
                                 f"Reason='{plan.get('reason', 'N/A')}'" )
                    pending_orders_details.append(plan_str)
                except Exception as e:
                    self.logger.error(f"Error formatting pending order {symbol}: {e}")
                    pending_orders_details.append(f"- {symbol.split(':')[0]}: Error formatting plan.")
        
        if not pending_orders_details:
            pending_orders_details.append("No pending limit orders.")
        # --- [V-Pending 修复结束] ---
        
        initial_capital_for_calc = settings.ALPHA_LIVE_INITIAL_CAPITAL if self.is_live else settings.ALPHA_PAPER_CAPITAL
        performance_percent_str = "N/A (Invalid Initial)"
        
        if initial_capital_for_calc > 0:
            current_equity_val = float(self.equity) if self.equity is not None else 0.0
            performance_percent = (current_equity_val / initial_capital_for_calc - 1) * 100
            performance_percent_str = f"{performance_percent:.2f}%"
            
        return { "account_value_usd": f"{float(self.equity):.2f}" if self.equity is not None else "0.00",
                 "cash_usd": f"{float(self.cash):.2f}" if self.cash is not None else "0.00",
                 "performance_percent": performance_percent_str,
                 "open_positions_rule6": "\n".join(position_details), # [V45.40 修复] 更改键名
                 "pending_limit_orders": "\n".join(pending_orders_details) # [V-Pending 修复] 新增键
               }
    # --- [V45.40 修复结束] ---
    
    
    async def live_open(self, symbol, side, size, leverage, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
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

    async def live_open_limit(self, symbol, side, size, leverage, limit_price: float, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        action_type = "限价开仓"
        self.logger.warning(f"!!! {self.mode_str} AI 请求 {action_type} (初步): {side.upper()} {size} {symbol} @ {limit_price} !!!")
        
        if self.position_manager.is_open(symbol):
            pos_state = self.position_manager.get_position_state(symbol)
            
            if pos_state and pos_state.get('side') == side:
                action_type = "限价加仓"
                self.logger.warning(f"!!! {self.mode_str} AI 请求 {action_type}: {side.upper()} {size} {symbol} @ {limit_price} !!!")
                
                current_leverage = pos_state.get('leverage')
                if current_leverage and int(current_leverage) > 0:
                    if int(leverage) != int(current_leverage):
                         self.logger.warning(f"{action_type}: AI 请求杠杆 {leverage}x, 但将强制使用现有杠杆 {current_leverage}x 以规避 -4161 错误。")
                         leverage = int(current_leverage)
                else:
                    self.logger.error(f"{action_type}: 无法获取 {symbol} 的现有杠杆！将冒险使用 AI 请求的 {leverage}x。")
            
            else:
                self.logger.error(f"!!! {self.mode_str} 限价单失败: {symbol} 已有 *相反* 持仓 (已有 {pos_state.get('side')}, 请求 {side})。")
                await send_bark_notification(f"❌ {self.mode_str} AI 限价单失败", f"品种: {symbol}\n原因: 已有相反持仓")
                return

        try:
            if symbol in self.pending_limit_orders:
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

            raw_exchange = self.client.exchange
            if not raw_exchange.markets: await self.client.load_markets()
            market = raw_exchange.markets.get(symbol);
            if not market: raise ValueError(f"无市场信息 {symbol}")

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

            await self.client.set_margin_mode(futures_settings.FUTURES_MARGIN_MODE, symbol)
            
            if action_type == "限价开仓":
                 await self.client.set_leverage(leverage, symbol)
            else:
                 self.logger.info(f"{action_type}: 正在使用现有杠杆 {leverage}x，不发送 set_leverage。")


            exchange_side = 'BUY' if side == 'long' else 'SELL'
            
            order_result = await self.client.create_limit_order(symbol, exchange_side, adjusted_size, limit_price)
            
            order_id = order_result.get('id')
            if not order_id:
                raise ValueError(f"交易所未返回 order_id: {order_result}")

            pending_plan = {
                'order_id': order_id,
                'side': side,
                'size': adjusted_size, # [V-Ultimate PaperFix] 存储最终的 adjusted_size
                'leverage': int(leverage),
                'limit_price': limit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'invalidation_condition': invalidation_condition,
                'reason': reason,
                'timestamp': time.time() * 1000 
            }
            
            await self.add_pending_limit_order(symbol, pending_plan)
            
            self.logger.warning(f"!!! {self.mode_str} {action_type} 挂单成功: {side.upper()} {adjusted_size} {symbol} @ {limit_price} (Order ID: {order_id})")
            self.logger.info(f"    SL: {stop_loss}, TP: {take_profit}, Inval: {invalidation_condition}")
            
            title_prefix = "⌛" if action_type == "限价开仓" else "🔼"
            title = f"{title_prefix} {self.mode_str} AI {action_type}: {side.upper()} {symbol.split('/')[0]}"
            body = f"价格: {limit_price:.4f}\n数量: {adjusted_size}\n杠杆: {leverage}x\nTP/SL: {take_profit}/{stop_loss}\nAI原因: {reason}"
            if adjusted_size != size: body += f"\n(请求 {size} 缩减至 {adjusted_size})"
            await send_bark_notification(title, body)

        except InsufficientFunds as e: self.logger.error(f"!!! {self.mode_str} {action_type} 失败 (资金不足): {e}", exc_info=False); await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n原因: 资金不足")
        except Exception as e: 
            self.logger.error(f"!!! {self.mode_str} {action_type} 失败: {e}", exc_info=True); 
            await send_bark_notification(f"❌ {self.mode_str} AI {action_type} 失败", f"品种: {symbol}\n错误: {e}")
            await self.remove_pending_limit_order(symbol)

    # --- [V-Ultimate 模拟盘修复] 新增 PAPEPR_OPEN_LIMIT 函数 ---
    async def paper_open_limit(self, symbol, side, size, leverage, limit_price: float, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        """
        模拟盘：接收 AI 的限价单计划，并将其存入待处理列表以供 'sync_state' 模拟。
        """
        action_type = "模拟限价开仓"
        
        # 检查是否已有持仓 (与 live_open_limit 逻辑相同)
        if self.paper_positions.get(symbol) and self.paper_positions[symbol].get('size', 0) > 0:
            pos_state = self.paper_positions[symbol]
            if pos_state and pos_state.get('side') == side:
                action_type = "模拟限价加仓"
            else:
                self.logger.error(f"!!! {self.mode_str} 模拟限价单失败: {symbol} 已有 *相反* 持仓。")
                return

        self.logger.warning(f"!!! {self.mode_str} AI 请求 {action_type}: {side.upper()} {size} {symbol} @ {limit_price} !!!")

        # 检查是否已有一个待处理订单
        if symbol in self.pending_limit_orders:
            old_plan = await self.remove_pending_limit_order(symbol)
            old_order_id = old_plan.get('order_id') if old_plan else "N/A"
            self.logger.warning(f"{self.mode_str} {action_type}: 发现旧的待处理订单 {old_order_id}。正在覆盖...")
            
        # 模拟盘不需要复杂的保证金检查，因为我们假设计划总是好的
        # 我们只在 'sync_state' 中检查 fill
        
        # 创建一个假的 order_id
        order_id = f"PAPER-{symbol}-{int(time.time() * 1000)}"

        pending_plan = {
            'order_id': order_id, # 模拟盘 ID
            'side': side,
            'size': size, # 存储计划的 size
            'leverage': int(leverage),
            'limit_price': limit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'invalidation_condition': invalidation_condition,
            'reason': reason,
            'timestamp': time.time() * 1000 
        }
        
        # 将计划存入待处理列表
        await self.add_pending_limit_order(symbol, pending_plan)
        
        self.logger.warning(f"!!! {self.mode_str} {action_type} 挂单(模拟)成功: {side.upper()} {size} {symbol} @ {limit_price}")
        self.logger.info(f"    SL: {stop_loss}, TP: {take_profit}, Inval: {invalidation_condition}")
        
        title_prefix = "⌛" if action_type == "模拟限价开仓" else "🔼"
        title = f"{title_prefix} {self.mode_str} AI {action_type}: {side.upper()} {symbol.split('/')[0]}"
        body = f"价格: {limit_price:.4f}\n数量: {size}\n杠杆: {leverage}x\nTP/SL: {take_profit}/{stop_loss}\nAI原因: {reason}"
        await send_bark_notification(title, body)

    # --- [修复结束] ---

    async def live_partial_close(self, symbol: str, size_percent: Optional[float] = None, size_absolute: Optional[float] = None, reason: str = "N/A"):
        # --- [ORPHAN FIX START] ---
        # 在执行部分平仓时，自动取消所有相关的“待处理”限价单 (例如AI的加仓计划)
        # 因为部分平仓意味着原始的仓位结构已改变，AI应在下一个周期重新评估是否加仓。
        self.logger.warning(f"!!! {self.mode_str} [ORPHAN FIX] (部分平仓) 检查并取消 {symbol} 的待处理限价单 (如有)...")
        try:
            pending_plan = await self.remove_pending_limit_order(symbol)
            if pending_plan:
                order_id = pending_plan.get('order_id')
                if order_id:
                    self.logger.warning(f"[ORPHAN FIX] 正在取消与 {symbol} 相关的待处理订单 {order_id}...")
                    await self.client.cancel_order(order_id, symbol)
                else:
                    self.logger.warning(f"[ORPHAN FIX] {symbol} 有一个待处理计划但没有 order_id。")
        except OrderNotFound:
            self.logger.info(f"[ORPHAN FIX] 待处理订单 {order_id} 在交易所未找到 (可能已成交/取消)。")
        except Exception as e_cancel:
            self.logger.error(f"[ORPHAN FIX] 取消待处理订单 {order_id} 失败: {e_cancel}。继续部分平仓...")
        # --- [ORPHAN FIX END] ---

        self.logger.warning(f"!!! {self.mode_str} AI 请求部分平仓: {symbol} | %: {size_percent} | Abs: {size_absolute} | 原因: {reason} !!!")

        pos_state = self.position_manager.get_position_state(symbol)
        if not pos_state or pos_state.get('total_size', 0) <= 0:
            self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 本地无有效持仓 {symbol}。")
            return

        current_total_size = pos_state['total_size']
        size_to_close = 0.0
        if size_percent is not None and 0 < size_percent <= 1: 
            if abs(size_percent - 1.0) < 1e-9:
                 self.logger.warning(f"{self.mode_str} 部分平仓请求 100%，转为全平。")
                 await self.live_close(symbol, reason=f"{reason} (转为全平)") # live_close 会处理孤儿单
                 return
            size_to_close = current_total_size * size_percent
        elif size_absolute is not None and 0 < size_absolute <= current_total_size + 1e-9: 
             if abs(size_absolute - current_total_size) < 1e-9:
                 self.logger.warning(f"{self.mode_str} 部分平仓请求绝对数量等于全仓，转为全平。")
                 await self.live_close(symbol, reason=f"{reason} (转为全平)") # live_close 会处理孤儿单
                 return
             size_to_close = min(size_absolute, current_total_size) 
        else: 
            self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 无效数量参数..."); 
            await send_bark_notification(f"❌ {self.mode_str} AI 部分平仓失败", f"品种: {symbol}\n原因: 无效数量参数"); 
            return

        try:
             raw_exchange = self.client.exchange
             if not raw_exchange.markets: await self.client.load_markets()
             market = raw_exchange.markets.get(symbol)
             if not market: raise ValueError(f"无法找到市场信息 {symbol}")

             # --- [BUG 修复 V2 (按用户要求调整) 开始] ---
             
             # 1. 提前获取最小下单量
             min_amount = market.get('limits', {}).get('amount', {}).get('min')
             if min_amount is None:
                 self.logger.warning(f"无法获取 {symbol} 的 min_amount，将跳过最小量检查。")

             # 2. 检查计算出的 size_to_close 是否小于 min_amount
             if min_amount is not None and size_to_close < min_amount:
                 self.logger.warning(f"!!! {self.mode_str} 部分平仓: 计算量 {size_to_close:.8f} < 交易所最小量 {min_amount}。")
                 
                 # 3. [用户请求] 尝试将数量增加到 min_amount，而不是跳过
                 
                 # 3a. (Edge Case) 检查 min_amount 是否大于或等于我们的总持仓
                 if min_amount >= current_total_size:
                     self.logger.warning(f"!!! {self.mode_str} 最小量 {min_amount} >= 总持仓 {current_total_size}。转为全平。")
                     await self.live_close(symbol, reason=f"{reason} (Partial < Min, convert to Full)")
                     return # 任务完成，退出函数
                 
                 # 3b. (正常) 增加到 min_amount
                 else:
                     self.logger.warning(f"!!! {self.mode_str} 正在将平仓量从 {size_to_close:.8f} 增加到 {min_amount} (交易所最小量)。")
                     size_to_close = min_amount
             
             # 4. 检查 (可能已调整的) 数量是否仍为 0 (例如 size_percent=0 导致)
             if size_to_close <= 0: 
                 self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 最终计算数量为 0"); 
                 return

             # 5. [安全] 现在，在所有检查和调整之后，才调用 amount_to_precision
             size_to_close = float(raw_exchange.amount_to_precision(symbol, size_to_close))
             
             # 6. [最终安全检查] 再次检查格式化后的值
             if min_amount is not None and size_to_close < min_amount:
                 self.logger.error(f"!!! {self.mode_str} 部分平仓失败 (Precision Fallback): 格式化后 {size_to_close} < {min_amount}。")
                 return
             if size_to_close <= 0: 
                 self.logger.error(f"!!! {self.mode_str} 部分平仓失败 (Precision Fallback): 格式化后数量为 0。")
                 return
             # --- [BUG 修复 V2 结束] ---

        except Exception as e: 
            self.logger.error(f"!!! {self.mode_str} 部分平仓失败 (检查数量时出错): {e}", exc_info=True); 
            return

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
                self.logger.error(f"!!! {self.mode_str} 部分平仓失败: 交易所返回成交量为 0 (Filled=0)。")
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
        # --- [ORPHAN FIX START] ---
        # 在执行(全)平仓时，自动取消所有相关的“待处理”限价单
        self.logger.warning(f"!!! {self.mode_str} [ORPHAN FIX] (全平仓) 检查并取消 {symbol} 的待处理限价单 (如有)...")
        try:
            pending_plan = await self.remove_pending_limit_order(symbol)
            if pending_plan:
                order_id = pending_plan.get('order_id')
                if order_id:
                    self.logger.warning(f"[ORPHAN FIX] 正在取消与 {symbol} 相关的待处理订单 {order_id}...")
                    await self.client.cancel_order(order_id, symbol)
                else:
                    self.logger.warning(f"[ORPHAN FIX] {symbol} 有一个待处理计划但没有 order_id。")
        except OrderNotFound:
            self.logger.info(f"[ORPHAN FIX] 待处理订单 {order_id} 在交易所未找到 (可能已成交/取消)。")
        except Exception as e_cancel:
            self.logger.error(f"[ORPHAN FIX] 取消待处理订单 {order_id} 失败: {e_cancel}。继续全平仓...")
        # --- [ORPHAN FIX END] ---

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

            if filled_size <= 0:
                self.logger.error(f"!!! {self.mode_str} (全)平仓失败: 交易所返回成交量为 0 (Filled=0)。仓位可能过小 (Dust) 或API错误。")
                self.logger.error("!!! 本地状态未改变，等待下一次 sync_state 或风控循环。")
                return

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
        # --- [ORPHAN FIX START] ---
        # (模拟盘) 在全平仓时，移除所有相关的待处理限价单
        await self.remove_pending_limit_order(symbol)
        # --- [ORPHAN FIX END] ---

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
        # --- [ORPHAN FIX START] ---
        # (模拟盘) 在部分平仓时，移除所有相关的待处理限价单
        await self.remove_pending_limit_order(symbol)
        # --- [ORPHAN FIX END] ---
        
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
                    # --- [FIX START] ---
                    # 移除了 'if bnb_contract_symbol not in self.symbols:' 的检查
                    # 无论 self.symbols 中是否包含 BNB，我们都将尝试获取其价格
                    
                    self.logger.debug(f"Fee Parsing: 正在强制获取 {bnb_contract_symbol} Ticker (无论是否在 self.symbols 中)...")
                    bnb_ticker = await self.client.fetch_ticker(bnb_contract_symbol) 
                    bnb_price = bnb_ticker.get('last')
                    
                    if bnb_price and bnb_price > 0:
                        fees_paid_usdt = fee_cost * bnb_price
                        self.logger.warning(f"BNB 手续费已转换为 USDT: {fee_cost:.6f} BNB * {bnb_price} USD/BNB = {fees_paid_usdt:.4f} USDT")
                    else:
                        self.logger.error(f"无法获取有效的 {bnb_contract_symbol} 价格，BNB 手续费将记录为 0 USDT。")
                        fees_paid_usdt = 0.0
                    # --- [FIX END] ---

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
