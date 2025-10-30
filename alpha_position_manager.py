# 文件: alpha_position_manager.py (V2.2 - 添加 _recalculate 方法并修复调用)

import logging
import os
import json
import time
from typing import Dict, Optional, List, Any

class AlphaPositionManager:
    """
    V2.2: 支持多次入场和部分平仓的状态管理器。
    - 添加了 _recalculate_position 和 _recalculate_all_positions 方法。
    - get_position_state 和 get_all_open_positions 调用计算方法确保数据最新。
    - 包含 update_rules 方法。
    """
    # 状态文件路径常量
    STATE_FILENAME = "alpha_live_positions_v2.json" # V2 使用的文件名

    def __init__(self, state_dir: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_dir = state_dir
        self.state_file = self._get_state_file_path(state_dir)
        # _positions 结构保持 V2.1 定义
        # {
        #   "symbol": {
        #     "status": "open" | "closed",
        #     "side": "long" | "short",
        #     "leverage": int,
        #     "entries": [ {"price": float, "size": float, "fee": float, "timestamp": int}, ... ],
        #     "total_size": float, # 需计算
        #     "avg_entry_price": float, # 需计算
        #     "total_entry_fee": float, # 需计算
        #     "margin": float, # 需计算
        #     "ai_initial_stop_loss": Optional[float],
        #     "ai_initial_take_profit": Optional[float],
        #     "ai_suggested_stop_loss": Optional[float],
        #     "ai_suggested_take_profit": Optional[float],
        #     "invalidation_condition": Optional[str],
        #     "entry_reason": Optional[str]
        #   }, ...
        # }
        self._positions: Dict[str, Dict[str, Any]] = {} # 重命名为 _positions 强调内部使用
        self._load_state()

    def _get_state_file_path(self, state_dir) -> str:
        """构建状态文件的完整路径。"""
        return os.path.join(state_dir, self.STATE_FILENAME)

    # --- [V2.2 核心新增] 属性计算方法 ---
    def _calculate_total_size(self, entries: List[Dict[str, Any]]) -> float:
        """根据入场记录计算总大小"""
        if not isinstance(entries, list): return 0.0
        return sum(e.get('size', 0.0) for e in entries)

    def _calculate_avg_entry_price(self, entries: List[Dict[str, Any]], total_size: float) -> float:
        """根据入场记录和总大小计算平均价格"""
        if not isinstance(entries, list) or total_size <= 1e-9: return 0.0
        total_value = sum(e.get('price', 0.0) * e.get('size', 0.0) for e in entries)
        return total_value / total_size

    def _calculate_total_entry_fee(self, entries: List[Dict[str, Any]]) -> float:
        """根据入场记录计算总费用"""
        if not isinstance(entries, list): return 0.0
        return sum(e.get('fee', 0.0) for e in entries)

    def _calculate_margin(self, avg_price: float, total_size: float, leverage: int) -> float:
        """计算近似保证金"""
        # [V2.2 修复] 确保 leverage > 0
        if leverage <= 0:
            self.logger.warning("Margin calculation skipped: Leverage is zero or negative.")
            return 0.0
        return (avg_price * total_size) / leverage if avg_price > 0 and total_size > 0 else 0.0

    def _recalculate_position(self, symbol: str):
        """根据 entries 重新计算指定 symbol 的 avg_entry_price, total_size, margin 等聚合数据"""
        state = self._positions.get(symbol)
        # 只计算 open 状态的仓位
        if not state or state.get('status') != 'open':
            # 如果仓位存在但不是 open，确保计算值为 0
            if state:
                state['total_size'] = 0.0
                state['avg_entry_price'] = 0.0
                state['total_entry_fee'] = 0.0
                state['margin'] = 0.0
            return

        entries = state.get('entries', [])
        # 确保 entries 是列表
        if not isinstance(entries, list):
             self.logger.error(f"Recalculate error for {symbol}: 'entries' is not a list. Resetting state.")
             state['entries'] = []
             entries = []

        total_size = self._calculate_total_size(entries)
        avg_entry_price = self._calculate_avg_entry_price(entries, total_size)
        total_entry_fee = self._calculate_total_entry_fee(entries)
        leverage = state.get('leverage', 1) # 提供默认值

        # 如果计算后 total_size 极小，视为已平仓
        if total_size < 1e-9:
            self.logger.warning(f"Recalculate: Position {symbol} size is near zero ({total_size}). Marking as closed.")
            state['status'] = 'closed'
            state['total_size'] = 0.0
            state['avg_entry_price'] = 0.0
            state['total_entry_fee'] = 0.0
            state['margin'] = 0.0
        else:
            state['total_size'] = total_size
            state['avg_entry_price'] = avg_entry_price
            state['total_entry_fee'] = total_entry_fee
            state['margin'] = self._calculate_margin(avg_entry_price, total_size, leverage)

            self.logger.debug(f"Recalculated {symbol}: Size={state['total_size']:.8f}, AvgPrice={state['avg_entry_price']:.4f}, Margin={state['margin']:.2f}, Fee={state['total_entry_fee']:.4f}")

    def _recalculate_all_positions(self):
        """重新计算所有仓位的聚合数据"""
        self.logger.debug("Recalculating all positions...")
        # 使用 list 避免在迭代时修改字典（虽然这里只是修改内部值）
        for symbol in list(self._positions.keys()):
            self._recalculate_position(symbol)
        self.logger.debug("Finished recalculating all positions.")
    # --- [V2.2 新增结束] ---


    # --- 状态加载/保存 (调用 _recalculate_all_positions) ---
    def _load_state(self):
        """从 JSON 文件加载所有持仓状态。"""
        if not os.path.exists(self.state_file):
            self.logger.info(f"未找到 V2 持仓状态文件 ({self.state_file})，初始化为空。")
            self._positions = {} # 确保初始化为空
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)
            if not isinstance(loaded_state, dict):
                 raise ValueError("Loaded state is not a dictionary.")
            self._positions = loaded_state
            # [V2.2 修改] 加载后重新计算一次聚合数据
            self._recalculate_all_positions()
            open_count = sum(1 for state in self._positions.values() if state.get('status') == 'open')
            self.logger.warning(f"成功从 {self.state_file} 恢复 {len(self._positions)} 交易对状态，{open_count} 个当前持仓。")
        except (json.JSONDecodeError, ValueError) as e:
             self.logger.error(f"加载 V2 持仓状态失败：文件格式或内容错误 - {e}", exc_info=True)
             self._handle_load_error()
        except Exception as e:
            self.logger.error(f"加载 V2 持仓状态失败: {e}", exc_info=True)
            self._handle_load_error()

    def _save_state(self):
        """将当前所有持仓状态保存到 JSON 文件。"""
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            # [V2.2 修改] 保存前确保聚合数据是计算过的 (虽然每次操作后都会算，但这里再算一次更保险)
            # self._recalculate_all_positions() # 可以在这里加，但可能影响性能，看情况
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self._positions, f, indent=4, ensure_ascii=False)
            self.logger.debug(f"持仓状态已保存到: {self.state_file}")
        except TypeError as e:
             self.logger.error(f"保存持仓状态失败：类型错误 - {e}. State: {self._positions}", exc_info=True)
        except Exception as e:
            self.logger.error(f"保存 V2 持仓状态失败: {e}", exc_info=True)

    def _handle_load_error(self):
         """处理加载状态失败的情况，备份旧文件并初始化。"""
         self.logger.warning("尝试备份损坏的 V2 持仓状态文件并重新初始化。")
         try:
             backup_path = self.state_file + f".backup_{int(time.time())}"
             if os.path.exists(self.state_file):
                 os.rename(self.state_file, backup_path)
                 self.logger.info(f"损坏文件已备份至: {backup_path}")
         except Exception as rename_e:
             self.logger.error(f"备份损坏文件失败: {rename_e}")
         self._positions = {}
         self._save_state()

    # --- 核心操作方法 (调用 _recalculate_position) ---
    def open_position(self, symbol: str, side: str, entry_price: float, size: float,
                      entry_fee: float, leverage: int, stop_loss: Optional[float], take_profit: Optional[float],
                      timestamp: int, reason: Optional[str] = None, invalidation_condition: Optional[str] = None):
        """记录一个新的开仓状态 (覆盖旧的)。"""
        if symbol in self._positions and self._positions[symbol].get('status') == 'open':
            self.logger.warning(f"开仓 {symbol} 时已存在持仓，将覆盖。")

        entry_record = {"price": entry_price, "size": size, "fee": entry_fee, "timestamp": timestamp}
        state = {
            "status": "open", "side": side.lower(), "leverage": leverage,
            "entries": [entry_record],
            # AI 规则
            "ai_initial_stop_loss": stop_loss, "ai_initial_take_profit": take_profit,
            "ai_suggested_stop_loss": stop_loss, "ai_suggested_take_profit": take_profit,
            "invalidation_condition": invalidation_condition, "entry_reason": reason
        }
        self._positions[symbol] = state
        # [V2.2 修改] 开仓后立即计算一次聚合数据
        self._recalculate_position(symbol)
        self.logger.info(f"记录开仓: {symbol} | Side={side.upper()}, Size={size}, Entry={entry_price}, Lev={leverage}x, Margin={state.get('margin', 0.0):.2f}")
        self._save_state()

    def add_entry(self, symbol: str, entry_price: float, size: float, entry_fee: float,
                   leverage: int, stop_loss: Optional[float], take_profit: Optional[float], timestamp: int,
                   invalidation_condition: Optional[str] = None) -> bool:
        """向现有持仓添加一次入场记录 (加仓)。"""
        state = self._positions.get(symbol)
        if not state or state.get('status') != 'open':
            self.logger.error(f"加仓失败：未找到打开的持仓 {symbol}")
            return False
        if state.get('side') is None:
             self.logger.error(f"加仓失败：持仓 {symbol} 状态异常 (side is None)。"); return False
        if 'entries' not in state or not isinstance(state['entries'], list):
            self.logger.warning(f"持仓 {symbol} entries 丢失或格式错误，重新初始化。"); state['entries'] = []

        entry_record = {'price': entry_price, 'size': size, 'fee': entry_fee, 'timestamp': timestamp}
        state['entries'].append(entry_record)

        if state.get('leverage') != leverage:
            self.logger.warning(f"加仓 {symbol} 杠杆变化 ({state.get('leverage')}x -> {leverage}x)。"); state['leverage'] = leverage
        state['ai_suggested_stop_loss'] = stop_loss
        state['ai_suggested_take_profit'] = take_profit
        if invalidation_condition is not None: state['invalidation_condition'] = invalidation_condition

        # [V2.2 修改] 加仓后重新计算
        self._recalculate_position(symbol)

        self.logger.info(f"记录加仓: {symbol} | AddSize={size}, NewAvgPrice={state['avg_entry_price']:.4f}, NewTotalSize={state['total_size']:.4f}, NewMargin={state['margin']:.2f}")
        self._save_state()
        return True

    def reduce_position(self, symbol: str, reduce_size: float) -> bool:
        """按比例减少持仓中的入场记录 (部分平仓后调用)。"""
        state = self._positions.get(symbol)
        # [V2.2 修改] 先重新计算，确保 current_total_size 最准
        self._recalculate_position(symbol)
        current_total_size = state.get('total_size', 0.0) if state else 0.0

        if not state or state.get('status') != 'open':
            self.logger.error(f"减仓失败：未找到打开的持仓 {symbol}"); return False
        if reduce_size <= 0:
            self.logger.warning(f"请求减仓数量无效 {symbol}: {reduce_size}"); return False
        if reduce_size > current_total_size + 1e-9:
            self.logger.warning(f"请求减仓 {reduce_size} 大于当前 {current_total_size} for {symbol}。视为全平。"); reduce_size = current_total_size

        if reduce_size >= current_total_size - 1e-9:
             self.close_position(symbol); return True

        remaining_ratio = (current_total_size - reduce_size) / current_total_size
        new_entries = []; entries = state.get('entries', [])
        if not isinstance(entries, list):
             self.logger.error(f"减仓失败：{symbol} entries 格式错误。清空仓位。"); self.close_position(symbol); return False

        for entry in entries:
            new_size = entry.get('size', 0.0) * remaining_ratio
            if new_size > 1e-9:
                 new_entry = entry.copy(); new_entry['size'] = new_size; new_entries.append(new_entry)

        state['entries'] = new_entries

        # [V2.2 修改] 减仓后重新计算
        self._recalculate_position(symbol)

        if state['total_size'] < 1e-9:
             self.logger.warning(f"减仓后 {symbol} 数量接近 0，视为全平。"); self.close_position(symbol) # close_position 内部会再次计算并保存
        else:
             self.logger.info(f"记录减仓: {symbol} | Reduced={reduce_size:.4f} | Remaining={state['total_size']:.8f}")
             self._save_state()

        return True


    def close_position(self, symbol: str):
        """标记指定 symbol 的仓位为已关闭。"""
        state = self._positions.get(symbol)
        if state:
            state['status'] = 'closed'
            state['entries'] = []
            # [V2.2 修改] 主动调用计算以清零聚合数据
            self._recalculate_position(symbol)
            self.logger.info(f"记录平仓: {symbol}")
            self._save_state()
        else:
            self.logger.warning(f"尝试平仓不存在的交易对: {symbol}")

    def is_open(self, symbol: str) -> bool:
        """检查指定交易对当前是否为打开状态且有持仓量"""
        state = self._positions.get(symbol)
        # [V2.2 修改] 检查 status 和 total_size (total_size 由 _recalculate 更新)
        return bool(state and state.get('status') == 'open' and state.get('total_size', 0.0) > 1e-9)

    def get_position_state(self, symbol: str) -> Optional[Dict]:
        """获取指定交易对的当前状态字典 (包含实时计算的聚合数据)。"""
        state = self._positions.get(symbol)
        if not state or state.get('status') != 'open':
            return None
        # [V2.2 修改] 获取前确保数据最新
        self._recalculate_position(symbol)
        # [V2.2 修复] 返回 state 的副本，防止外部修改内部状态
        return state.copy() if state else None

    def get_all_open_positions(self) -> Dict[str, Dict]:
        """获取所有当前打开状态的持仓及其完整状态字典 (包含实时计算的聚合数据)。"""
        open_positions = {}
        # [V2.2 修改] 获取前确保所有数据最新
        self._recalculate_all_positions() # <--- 修复 AttributeError 的关键调用
        for symbol, state in self._positions.items():
            # [V2.2 修改] 检查 status 和 total_size
            if state and state.get('status') == 'open' and state.get('total_size', 0.0) > 1e-9:
                # [V2.2 修复] 返回 state 的副本
                open_positions[symbol] = state.copy()
        return open_positions

    # --- [V2.1 核心新增] update_rules 方法 ---
    def update_rules(self, symbol: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None, invalidation_condition: Optional[str] = None) -> bool:
        """更新指定持仓的 AI 相关规则。"""
        state = self._positions.get(symbol)
        if not state or state.get('status') != 'open':
            self.logger.error(f"更新规则失败：未找到打开的持仓 {symbol}"); return False

        updated = False
        if stop_loss is not None and isinstance(stop_loss, (int, float)) and stop_loss > 0 and state.get('ai_suggested_stop_loss') != stop_loss:
            state['ai_suggested_stop_loss'] = stop_loss; updated = True; self.logger.info(f"持仓 {symbol} AI止损更新为: {stop_loss}")
        if take_profit is not None and isinstance(take_profit, (int, float)) and take_profit > 0 and state.get('ai_suggested_take_profit') != take_profit:
            state['ai_suggested_take_profit'] = take_profit; updated = True; self.logger.info(f"持仓 {symbol} AI止盈更新为: {take_profit}")
        if invalidation_condition is not None and state.get('invalidation_condition') != invalidation_condition:
            state['invalidation_condition'] = invalidation_condition; updated = True; self.logger.info(f"持仓 {symbol} 失效条件更新为: '{invalidation_condition}'")

        if updated: self._save_state()
        else: self.logger.debug(f"请求更新 {symbol} 规则，但值未改变或无效。")
        return updated
    # --- [新增结束] ---

    # [V2.1 新增] 用于 sync_state 更新交易所数据 (V23.3 中已移除调用, 此方法保留但不再被 portfolio 调用)
    def update_position_from_exchange(self, symbol: str, current_size: float, current_entry_price: float, current_leverage: int):
        """(V2.2 不再被 portfolio 调用) 用交易所数据更新本地记录的部分字段"""
        state = self._positions.get(symbol)
        if state and state.get('status') == 'open':
            updated = False
            if state.get('leverage') != current_leverage:
                self.logger.warning(f"Sync Update: {symbol} leverage changed (Local: {state.get('leverage')}, Exch: {current_leverage}). Updating.")
                state['leverage'] = current_leverage; updated = True
                self._recalculate_position(symbol) # 杠杆变了要重新算保证金
            # 不再强制更新 size 或 avg_entry_price，依赖内部计算
            if updated: self._save_state()
