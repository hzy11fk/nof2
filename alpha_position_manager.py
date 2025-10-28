# 文件: alpha_position_manager.py (V2 - 支持多次入场和部分平仓)

import logging
import os
import json
import time
from typing import Dict, Optional, List, Any

class AlphaPositionManager:
    """
    V2: 支持多次入场（加仓）和部分平仓的状态管理器。
    """
    def __init__(self, state_dir: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_dir = state_dir
        self.state_file = self._get_state_file_path(state_dir)
        # 内存中的持仓状态: { "symbol": { state_dict }, ... }
        # state_dict 结构:
        # {
        #   "side": "long" | "short" | None,
        #   "entries": [ {"price": float, "size": float, "fee": float, "timestamp": int}, ... ],
        #   "leverage": int, # 记录最后一次设置的杠杆
        #   "ai_initial_stop_loss": float,
        #   "ai_initial_take_profit": float,
        #   "ai_suggested_stop_loss": float,
        #   "ai_suggested_take_profit": float,
        #   "invalidation_condition": str,
        #   "entry_reason": str # 首次开仓原因
        # }
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._load_state()

    def _get_state_file_path(self, state_dir) -> str:
        """构建状态文件的完整路径。"""
        # 使用新文件名以避免与旧格式冲突
        return os.path.join(state_dir, 'alpha_live_positions_v2.json')

    # --- 属性计算 ---
    def get_size(self, symbol: str) -> float:
        """计算指定 symbol 的总持仓数量。"""
        state = self.positions.get(symbol, {})
        # 确保 entries 是列表
        entries = state.get('entries', [])
        if not isinstance(entries, list): return 0.0
        return sum(e.get('size', 0.0) for e in entries)

    def get_avg_entry_price(self, symbol: str) -> float:
        """计算指定 symbol 的平均入场价格。"""
        state = self.positions.get(symbol, {})
        entries = state.get('entries', [])
        if not isinstance(entries, list): return 0.0
        total_size = self.get_size(symbol)
        if not entries or total_size == 0:
            return 0.0
        total_value = sum(e.get('price', 0.0) * e.get('size', 0.0) for e in entries)
        # 避免除零错误
        return total_value / total_size if total_size else 0.0

    def get_total_entry_fee(self, symbol: str) -> float:
        """计算指定 symbol 的总入场手续费。"""
        state = self.positions.get(symbol, {})
        entries = state.get('entries', [])
        if not isinstance(entries, list): return 0.0
        return sum(e.get('fee', 0.0) for e in entries)

    # --- 状态加载/保存 (基本不变, 处理新结构) ---
    def _load_state(self):
        """从 JSON 文件加载所有持仓状态。"""
        if not os.path.exists(self.state_file):
            self.logger.info("未找到 V2 实盘持仓状态文件，将创建新的记录。")
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded_positions = json.load(f)
            # 基本验证
            if not isinstance(loaded_positions, dict):
                 raise ValueError("Loaded state is not a dictionary.")
            # TODO: 可以添加更详细的结构验证，例如检查 'entries' 是否是列表等
            self.positions = loaded_positions
            open_count = sum(1 for symbol, state in self.positions.items() if self.get_size(symbol) > 0) # 使用 get_size 判断
            self.logger.info(f"成功从 V2 文件恢复 {len(self.positions)} 个交易对的持仓状态，其中 {open_count} 个当前有持仓。")
        except (json.JSONDecodeError, ValueError) as e:
             self.logger.error(f"加载 V2 实盘持仓状态失败：文件格式或内容错误 - {e}", exc_info=True)
             self._handle_load_error()
        except Exception as e:
            self.logger.error(f"加载 V2 实盘持仓状态失败: {e}", exc_info=True)
            self._handle_load_error()

    def _save_state(self):
        """将当前所有持仓状态保存到 JSON 文件。"""
        try:
            os.makedirs(self.state_dir, exist_ok=True) # 确保目录存在
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.positions, f, indent=4)
        except Exception as e:
            self.logger.error(f"保存 V2 实盘持仓状态失败: {e}", exc_info=True)

    def _handle_load_error(self):
         """处理加载状态失败的情况，备份旧文件并初始化。"""
         self.logger.warning("将尝试备份损坏的 V2 实盘持仓状态文件并重新初始化。")
         try:
             backup_path = self.state_file + f".backup_{int(time.time())}"
             if os.path.exists(self.state_file):
                 os.rename(self.state_file, backup_path)
                 self.logger.info(f"损坏的文件已备份至: {backup_path}")
         except Exception as rename_e:
             self.logger.error(f"备份损坏的 V2 实盘持仓状态文件失败: {rename_e}")
         self.positions = {}
         self._save_state() # 保存空的字典

    # --- 核心操作方法 ---
    def open_position(self, symbol: str, side: str, entry_price: float, size: float,
                      entry_fee: float, leverage: int, stop_loss: float, take_profit: float,
                      timestamp: int, reason: str, invalidation_condition: str):
        """
        记录一个新的开仓状态 (覆盖旧的，如果存在)。
        """
        state = {
            "side": side,
            "entries": [{'price': entry_price, 'size': size, 'fee': entry_fee, 'timestamp': timestamp}],
            "leverage": leverage,
            "ai_initial_stop_loss": stop_loss, # 记录初始AI建议
            "ai_initial_take_profit": take_profit,
            "ai_suggested_stop_loss": stop_loss, # 初始时，最新=初始
            "ai_suggested_take_profit": take_profit,
            "invalidation_condition": invalidation_condition,
            "entry_reason": reason # 首次开仓原因
        }
        self.positions[symbol] = state
        self.logger.info(f"实盘持仓状态已创建 (开新仓): {symbol} | Side: {side} | Size: {size}")
        self._save_state()

    def add_entry(self, symbol: str, entry_price: float, size: float, entry_fee: float,
                   leverage: int, stop_loss: float, take_profit: float, timestamp: int,
                   invalidation_condition: str):
        """
        向现有持仓添加一次入场记录 (加仓)。
        """
        state = self.positions.get(symbol)
        # 确保 state 存在且非空字典
        if not state or not isinstance(state, dict) or state.get('side') is None or self.get_size(symbol) <= 0:
            self.logger.error(f"尝试向一个不存在或已平仓的头寸加仓: {symbol}")
            return False # 表示加仓失败

        # 确保 entries 是列表
        if 'entries' not in state or not isinstance(state['entries'], list):
            state['entries'] = [] # 如果丢失或格式错误，则初始化

        # 更新 entries 列表
        state['entries'].append({
            'price': entry_price,
            'size': size,
            'fee': entry_fee,
            'timestamp': timestamp
        })

        # 以最新一次加仓时的设置为准，更新整体仓位的规则
        state['leverage'] = leverage # 更新杠杆记录
        state['ai_suggested_stop_loss'] = stop_loss
        state['ai_suggested_take_profit'] = take_profit
        state['invalidation_condition'] = invalidation_condition
        # 注意：ai_initial_stop_loss 和 ai_initial_take_profit 保持不变

        self.logger.info(f"实盘持仓状态已更新 (加仓): {symbol} | 新增 Size: {size} | 新均价: {self.get_avg_entry_price(symbol):.4f}")
        self._save_state()
        return True # 表示加仓成功

    def reduce_position(self, symbol: str, reduce_size: float) -> bool:
        """
        根据部分平仓的数量，按比例减少持仓中的入场记录。
        返回操作是否成功。
        """
        state = self.positions.get(symbol)
        current_total_size = self.get_size(symbol)

        # 确保 state 存在且非空字典
        if not state or not isinstance(state, dict) or current_total_size <= 0:
            self.logger.error(f"尝试部分平仓一个不存在或已平仓的头寸: {symbol}")
            return False
        if reduce_size <= 0:
            self.logger.warning(f"请求的部分平仓数量无效 (<= 0): {reduce_size}")
            return False
        if reduce_size >= current_total_size:
            # 如果请求平仓数量大于等于当前持仓，则视为全平
            self.close_position(symbol)
            return True

        # 计算剩余比例
        remaining_ratio = (current_total_size - reduce_size) / current_total_size
        if remaining_ratio <= 0: # 再次检查以防浮点数问题
             self.close_position(symbol)
             return True

        # 按比例缩减每个 entry 的 size
        new_entries = []
        entries = state.get('entries', [])
        if not isinstance(entries, list): # 防御性检查
             self.logger.error(f"部分平仓失败：{symbol} 的 entries 格式错误。")
             self.close_position(symbol) # 数据损坏，直接平仓
             return False

        for entry in entries:
            new_size = entry.get('size', 0.0) * remaining_ratio
            # 可以选择移除 size 过小的 entry，防止累积过多零碎记录
            if new_size > 1e-9: # 设定一个极小的阈值
                 new_entry = entry.copy()
                 new_entry['size'] = new_size
                 new_entries.append(new_entry)

        state['entries'] = new_entries

        # 检查缩减后是否还有有效 entry
        if not state['entries'] or self.get_size(symbol) <= 1e-9:
             self.logger.warning(f"部分平仓后 {symbol} 剩余数量过小，视为全平。")
             self.close_position(symbol)
             return True
        else:
             self.logger.info(f"实盘持仓状态已更新 (部分平仓): {symbol} | 减少 Size: {reduce_size} | 剩余 Size: {self.get_size(symbol):.8f}")
             self._save_state()
             return True


    def close_position(self, symbol: str):
        """完全平仓一个交易对 (清空状态)。"""
        if symbol in self.positions:
            # 清空该 symbol 的状态字典
            self.positions[symbol] = {}
            # 或者可以选择保留最后一次的状态，但标记 side=None, entries=[]
            # self.positions[symbol] = {"side": None, "entries": [], ...保留其他信息...}
            self.logger.info(f"实盘持仓状态已清空 (全平): {symbol}")
            self._save_state()
        else:
            self.logger.warning(f"尝试平仓一个不存在于管理器中的交易对: {symbol}")

    def is_open(self, symbol: str) -> bool:
        """检查指定交易对当前是否有持仓。"""
        # 使用 get_size 判断
        return self.get_size(symbol) > 1e-9 # 使用阈值判断

    def get_position_state(self, symbol: str) -> Optional[Dict]:
        """获取指定交易对的当前状态字典 (包含计算后的均价和总数量)。"""
        state = self.positions.get(symbol)
        if not state or not isinstance(state, dict) or not self.is_open(symbol):
            return None

        # 返回包含计算属性的完整状态
        calculated_state = state.copy()
        calculated_state['avg_entry_price'] = self.get_avg_entry_price(symbol)
        calculated_state['total_size'] = self.get_size(symbol)
        calculated_state['total_entry_fee'] = self.get_total_entry_fee(symbol)
        return calculated_state


    def get_all_open_positions(self) -> Dict[str, Dict]:
        """获取所有当前有持仓的交易对及其完整状态字典 (包含计算属性)。"""
        open_positions = {}
        for symbol, state in self.positions.items():
             # 确保 state 是字典
            if isinstance(state, dict) and self.is_open(symbol):
                calculated_state = state.copy()
                calculated_state['avg_entry_price'] = self.get_avg_entry_price(symbol)
                calculated_state['total_size'] = self.get_size(symbol)
                calculated_state['total_entry_fee'] = self.get_total_entry_fee(symbol)
                open_positions[symbol] = calculated_state
        return open_positions

    def update_ai_suggestions(self, symbol: str, new_sl: Optional[float] = None, new_tp: Optional[float] = None):
        """更新指定交易对的 AI 止损/止盈建议。"""
        state = self.positions.get(symbol)
        # 确保 state 存在且非空字典
        if state and isinstance(state, dict) and self.is_open(symbol): # 只更新开仓中的
            updated = False
            if new_sl is not None and isinstance(new_sl, (int, float)) and new_sl > 0 and state.get('ai_suggested_stop_loss') != new_sl:
                state['ai_suggested_stop_loss'] = new_sl
                updated = True
                self.logger.info(f"更新 {symbol} AI 建议止损 -> {new_sl}")
            if new_tp is not None and isinstance(new_tp, (int, float)) and new_tp > 0 and state.get('ai_suggested_take_profit') != new_tp:
                state['ai_suggested_take_profit'] = new_tp
                updated = True
                self.logger.info(f"更新 {symbol} AI 建议止盈 -> {new_tp}")

            if updated:
                self._save_state()
        else:
            self.logger.debug(f"尝试更新一个未持仓或不存在的交易对的 AI 建议: {symbol}")
