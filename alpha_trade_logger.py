# 文件: alpha_trade_logger.py (V1 - 专用实盘交易记录器)

import logging
import os
import json
import time
from typing import List, Dict

class AlphaTradeLogger:
    """
    专门用于持久化存储 AlphaTrader 实盘交易记录的类。
    """
    def __init__(self, state_dir: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_dir = state_dir
        self.log_file = self._get_log_file_path(state_dir)
        self.trades_history: List[Dict] = [] # 内存中的交易历史列表

        # 确保目录存在
        os.makedirs(state_dir, exist_ok=True)
        self._load_history()

    def _get_log_file_path(self, state_dir) -> str:
        """构建日志文件的完整路径。"""
        return os.path.join(state_dir, 'alpha_live_trades.json')

    def _load_history(self):
        """从 JSON 文件加载交易历史。"""
        if not os.path.exists(self.log_file):
            self.logger.info("未找到实盘交易记录文件，将创建新的记录。")
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                self.trades_history = json.load(f)
            if not isinstance(self.trades_history, list):
                 self.logger.error("加载的交易记录不是一个列表，将重新初始化。")
                 self.trades_history = []
                 self._save_history() # 保存空的列表
            else:
                 self.logger.info(f"成功从文件恢复 {len(self.trades_history)} 条实盘交易记录。")
        except json.JSONDecodeError as e:
             self.logger.error(f"加载实盘交易记录失败：JSON文件格式错误 - {e}", exc_info=True)
             self._handle_load_error()
        except Exception as e:
            self.logger.error(f"加载实盘交易记录失败: {e}", exc_info=True)
            self._handle_load_error()

    def _save_history(self):
        """将当前交易历史保存到 JSON 文件。"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.trades_history, f, indent=4)
        except Exception as e:
            self.logger.error(f"保存实盘交易记录失败: {e}", exc_info=True)

    def _handle_load_error(self):
         """处理加载状态失败的情况，备份旧文件并初始化。"""
         self.logger.warning("将尝试备份损坏的实盘交易记录文件并重新初始化。")
         try:
             backup_path = self.log_file + f".backup_{int(time.time())}"
             if os.path.exists(self.log_file):
                os.rename(self.log_file, backup_path)
                self.logger.info(f"损坏的文件已备份至: {backup_path}")
         except Exception as rename_e:
             self.logger.error(f"备份损坏的实盘交易记录文件失败: {rename_e}")

         # 重置状态为空列表
         self.trades_history = []
         self._save_history() # 保存空的列表

    def record_trade(self, trade_data: dict):
        """
        记录一笔新的实盘交易。
        """
        # 可以添加一些验证逻辑，确保 trade_data 格式正确
        if not isinstance(trade_data, dict) or 'symbol' not in trade_data or 'net_pnl' not in trade_data:
            self.logger.error(f"尝试记录无效的交易数据: {trade_data}")
            return
            
        self.trades_history.append(trade_data)
        self.logger.info(f"记录一笔新的实盘交易: {trade_data.get('symbol')} PNL={trade_data.get('net_pnl'):.2f}。当前共 {len(self.trades_history)} 条记录。")
        self._save_history() # 立即保存

    def get_history(self) -> List[Dict]:
        """获取当前内存中的所有交易历史记录。"""
        return self.trades_history
