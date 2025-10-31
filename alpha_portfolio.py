# --- [V45.36 策略A 修复] ---
    # [GEMINI V3 修复] 升级此函数，使其支持 "限价加仓" (Pyramiding)
    async def live_open_limit(self, symbol, side, size, leverage, limit_price: float, reason: str = "N/A", stop_loss: float = None, take_profit: float = None, invalidation_condition: str = "N/A"):
        """[实盘] 挂一个限价开仓单，并将 SL/TP/Leverage 计划存储起来。
        [GEMINI V3 修复] 此函数现在支持对同向持仓进行限价加仓。
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
                old_plan = self.pending_limit_orders.pop(symbol, None)
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

            # --- [V45.36 修复：存储杠杆] ---
            pending_plan = {
                'order_id': order_id,
                'side': side,
                'leverage': int(leverage), # <-- [GEMINI V3 修复] 存储最终使用的杠杆
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'invalidation_condition': invalidation_condition,
                'reason': reason,
                'timestamp': time.time() * 1000 
            }
            # --- [修复结束] ---
            self.pending_limit_orders[symbol] = pending_plan
            
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
            self.pending_limit_orders.pop(symbol, None)
    # --- [V45.34/36 修复结束] ---
