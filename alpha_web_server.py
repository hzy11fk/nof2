# 文件: alpha_web_server.py (V22.2 - 修复资产/日志卡片高度不一致)

import logging
import asyncio
import os
import json
from aiohttp import web
import numpy as np
import pandas as pd
import math
from config import settings
# 导入 AlphaPortfolio 以便进行类型检查 (可选但推荐)
from alpha_portfolio import AlphaPortfolio # 假设 V18 或更新版本

# 配置日志记录器
logger = logging.getLogger(__name__) # 使用模块级日志记录器

def sanitize_data(data):
    # (此函数保持不变)
    if isinstance(data, dict): return {k: sanitize_data(v) for k, v in data.items()}
    if isinstance(data, list): return [sanitize_data(i) for i in data]
    if isinstance(data, (float, np.floating)):
        # [增强] 明确处理 None, NaN 和 Inf
        if data is None or math.isinf(data) or math.isnan(data): return None
        return float(data)
    if isinstance(data, np.integer): return int(data)
    # [增强] 如果不是字典、列表或需要处理的数字类型，直接返回值
    return data


async def get_alpha_trader_status(request):
    # (此函数保持 V22.1 不变, 包含收益率计算)
    try:
        alpha_trader = request.app['alpha_trader']
        portfolio: AlphaPortfolio = alpha_trader.portfolio

        tickers = {}
        try:
            tickers = await alpha_trader.exchange.fetch_tickers(alpha_trader.symbols)
        except Exception as e:
            logger.warning(f"无法为Web服务器获取TICKERS: {e}")

        ticker_data = {}
        for s, t in tickers.items():
            ticker_data[s] = {
                'last': t.get('last', 0),
                'percentage': t.get('percentage', 0)
            }

        trade_history = []
        try:
            raw_trade_history = portfolio.trade_history
            if isinstance(raw_trade_history, list): trade_history = raw_trade_history
            else: logger.warning(f"portfolio.trade_history 返回的不是列表: {type(raw_trade_history)}")
        except Exception as e: logger.error(f"获取 trade_history 时出错: {e}", exc_info=True)

        total_pnl = 0.0; winning_trades = 0; total_trades = 0
        try:
            total_pnl = sum(t.get('net_pnl', t.get('pnl', 0)) for t in trade_history if isinstance(t, dict))
            winning_trades = sum(1 for t in trade_history if isinstance(t, dict) and t.get('net_pnl', t.get('pnl', 0)) > 0)
            total_trades = len(trade_history)
        except Exception as e: logger.error(f"计算 PNL 或胜率时出错: {e}", exc_info=True)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

        formatted_positions = []
        try:
            if portfolio.is_live:
                if hasattr(portfolio, 'position_manager') and portfolio.position_manager:
                    open_positions_dict = portfolio.position_manager.get_all_open_positions() 
                    for s, pos_data in open_positions_dict.items():
                        try:
                            if not isinstance(pos_data, dict): continue
                            frontend_pos = {'symbol': s}
                            frontend_pos['side'] = pos_data.get('side')
                            frontend_pos['entry_price'] = pos_data.get('avg_entry_price')
                            frontend_pos['size'] = pos_data.get('total_size')
                            frontend_pos['leverage'] = pos_data.get('leverage')
                            current_price = tickers.get(s, {}).get('last', 0)
                            frontend_pos['unrealized_pnl'] = 0.0
                            if current_price and current_price > 0 and frontend_pos['entry_price'] and frontend_pos['size'] and frontend_pos['side']:
                                entry = frontend_pos['entry_price']
                                size = frontend_pos['size']
                                side = frontend_pos['side']
                                if side == 'long': frontend_pos['unrealized_pnl'] = (current_price - entry) * size
                                elif side == 'short': frontend_pos['unrealized_pnl'] = (entry - current_price) * size
                            frontend_pos['margin'] = 0.0
                            leverage = frontend_pos.get('leverage')
                            entry_price_for_margin = frontend_pos.get('entry_price')
                            size_for_margin = frontend_pos.get('size')
                            if leverage and leverage > 0 and entry_price_for_margin and entry_price_for_margin > 0 and size_for_margin:
                                frontend_pos['margin'] = (size_for_margin * entry_price_for_margin) / leverage
                            frontend_pos['profit_rate'] = 0.0
                            margin_val = frontend_pos.get('margin', 0.0)
                            upl_val = frontend_pos.get('unrealized_pnl', 0.0)
                            if margin_val and margin_val > 0: 
                                frontend_pos['profit_rate'] = (upl_val / margin_val) * 100.0
                            frontend_pos['take_profit'] = pos_data.get('ai_suggested_take_profit')
                            frontend_pos['stop_loss'] = pos_data.get('ai_suggested_stop_loss')
                            frontend_pos['invalidation_condition'] = pos_data.get('invalidation_condition')
                            formatted_positions.append(frontend_pos) 
                        except Exception as inner_e:
                            logger.error(f"处理实盘持仓 {s} 时出错: {inner_e}", exc_info=True)
                else:
                    logger.error("实盘模式但 portfolio.position_manager 不存在或无效。")
            else:
                if hasattr(portfolio, 'paper_positions') and isinstance(portfolio.paper_positions, dict):
                    for s, p in portfolio.paper_positions.items():
                        try:
                            if p and isinstance(p, dict):
                                pos_copy = p.copy()
                                pos_copy['symbol'] = s
                                pos_copy['unrealized_pnl'] = pos_copy.get('unrealized_pnl', 0.0)
                                pos_copy['margin'] = pos_copy.get('margin', 0.0)
                                pos_copy['profit_rate'] = 0.0
                                margin_val = pos_copy.get('margin', 0.0)
                                upl_val = pos_copy.get('unrealized_pnl', 0.0)
                                if margin_val and margin_val > 0: 
                                    pos_copy['profit_rate'] = (upl_val / margin_val) * 100.0
                                pos_copy['leverage'] = pos_copy.get('leverage')
                                pos_copy['take_profit'] = pos_copy.get('take_profit')
                                pos_copy['stop_loss'] = pos_copy.get('stop_loss')
                                pos_copy['invalidation_condition'] = pos_copy.get('invalidation_condition')
                                formatted_positions.append(pos_copy)
                        except Exception as inner_e:
                             logger.error(f"处理模拟持仓 {s} 时出错: {inner_e}", exc_info=True)
                else:
                    logger.error("模拟盘模式但 portfolio.paper_positions 不存在或无效。")
        except Exception as e:
            logger.error(f"获取或处理持仓列表时出错: {e}", exc_info=True)

        performance_percent = 0.0
        initial_capital_for_calc = 0.0
        try:
            initial_capital_for_calc = settings.ALPHA_LIVE_INITIAL_CAPITAL if portfolio.is_live else settings.ALPHA_PAPER_CAPITAL
            if initial_capital_for_calc > 0 and portfolio.equity is not None: 
                 performance_percent = (portfolio.equity / initial_capital_for_calc - 1) * 100
        except Exception as e:
             logger.error(f"计算表现百分比时出错: {e}", exc_info=True)

        equity_history_list = []
        try:
            if hasattr(portfolio, 'equity_history') and portfolio.equity_history is not None and hasattr(portfolio.equity_history, '__iter__'):
                equity_history_list = list(portfolio.equity_history)
            else:
                logger.warning(f"portfolio.equity_history 不可用或不可迭代: {getattr(portfolio, 'equity_history', 'Attribute Missing')}")
        except Exception as e:
            logger.error(f"获取 equity_history 时出错: {e}", exc_info=True)

        logs_list = []
        try:
             if hasattr(alpha_trader, 'log_deque') and alpha_trader.log_deque is not None and hasattr(alpha_trader.log_deque, '__iter__'):
                 logs_list = list(alpha_trader.log_deque)
             else:
                 logger.warning(f"alpha_trader.log_deque 不可用或不可迭代: {getattr(alpha_trader, 'log_deque', 'Attribute Missing')}")
        except Exception as e:
             logger.error(f"获取 logs 时出错: {e}", exc_info=True)

        strategy_summary_str = ""
        try:
             strategy_summary_str = getattr(alpha_trader, 'last_strategy_summary', "") or "" 
        except Exception as e:
             logger.error(f"获取 strategy_summary 时出错: {e}", exc_info=True)

        status_data = {
            "is_live": getattr(portfolio, 'is_live', False), 
            "initial_capital": initial_capital_for_calc,
            "portfolio_equity": getattr(portfolio, 'equity', 0.0), 
            "available_cash": getattr(portfolio, 'cash', 0.0), 
            "total_pnl": total_pnl,
            "performance_percent": performance_percent,
            "stats": {"win_rate": win_rate, "total_trades": total_trades},
            "open_positions": formatted_positions, 
            "equity_history": equity_history_list,
            "trade_history": trade_history[-15:],
            "logs": logs_list,
            "strategy_summary": strategy_summary_str,
            "symbols": getattr(alpha_trader, 'symbols', []), 
            "ticker_data": ticker_data
        }
        return web.json_response(sanitize_data(status_data))

    except Exception as e:
        logger.critical(f"处理 /api/alpha/status 请求时发生严重错误: {e}", exc_info=True)
        error_response = { "error": "Internal Server Error", "message": str(e) }
        return web.json_response(error_response, status=500)


async def handle_root(request):
    """提供仪表盘主页的HTML。"""
    # [V22.2 核心修改]
    # 1. 移除日志卡片 (Log Card) 的 "h-96"
    # 2. 将 "h-96" 添加到日志容器 (logs-container)
    # 3. 移除日志容器的 "flex-1 min-h-0" (因为不再是 flex 布局的子元素)
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Alpha Trader Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
        <style>
            body { background-color: #f8fafc; color: #1e293b; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }
            .profit { color: #16a34a; } .loss { color: #dc2626; } .side-long { color: #2563eb; } .side-short { color: #f97316; }
            .card { background-color: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1); }
            .table-header { background-color: #f1f5f9; color: #475569; }
            .live-badge { background-color: #dc2626; color: white; } .paper-badge { background-color: #2563eb; color: white; }
            #ticker-bar { background-color: #ffffff; border-bottom: 1px solid #e2e8f0; }
            #logs-container { font-family: 'Courier New', Courier, monospace; font-size: 0.8rem; line-height: 1.4; color: #475569; white-space: pre-wrap; word-break: break-all; background-color: #f8fafc; }
        </style>
    </head>
    <body class="font-sans">
        <div id="loader" class="fixed inset-0 bg-white flex items-center justify-center z-50"><p class="text-xl text-gray-600">正在连接AI Alpha Trader...</p></div>
        <div id="ticker-bar" class="flex overflow-x-auto sticky top-0 z-40"></div>
        <div class="container mx-auto p-4 md:p-8">
            <header class="mb-8 flex justify-between items-center">
                <div><h1 class="text-3xl font-bold text-slate-900">AI Alpha Trader</h1><p class="text-gray-500">自主多币种合约交易系统</p></div>
                <div id="mode-badge" class="text-sm font-bold py-1 px-3 rounded-full"></div>
            </header>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div class="card p-6 rounded-lg"><h2 class="text-sm text-gray-500">投资组合净值</h2><p id="portfolio-equity" class="text-4xl font-bold text-slate-800 mt-2">--</p></div>
                <div class="card p-6 rounded-lg"><h2 class="text-sm text-gray-500">可用现金</h2><p id="available-cash" class="text-4xl font-bold text-slate-800 mt-2">--</p></div>
                <div class="card p-6 rounded-lg"><h2 class="text-sm text-gray-500">已实现总盈亏</h2><p id="total-pnl" class="text-4xl font-bold mt-2">--</p></div>
                <div class="card p-6 rounded-lg"><h2 class="text-sm text-gray-500">表现</h2><p id="performance" class="text-4xl font-bold mt-2">--</p></div>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                 <div class="lg:col-span-2 card p-6 rounded-lg"> 
                     <h2 class="text-xl font-bold text-slate-800 mb-4">资产净值曲线</h2> 
                     <div class="h-96"> <canvas id="equity-chart"></canvas> </div> 
                 </div>
                 <div class="lg:col-span-1 card p-6 rounded-lg"> 
                     <h2 class="text-xl font-bold text-slate-800 mb-4">运行日志</h2> 
                     <div id="logs-container" class="overflow-y-auto bg-slate-50 p-3 rounded h-96"></div> 
                 </div>
            </div>
            <div class="card p-6 rounded-lg mb-8"><h2 class="text-xl font-bold text-slate-800 mb-4">AI 策略摘要</h2><p id="strategy-summary" class="text-lg text-gray-800 leading-relaxed">--</p></div>
            <div class="grid grid-cols-1 xl:grid-cols-2 gap-8">
                <div class="card p-6 rounded-lg"><h2 class="text-xl font-bold text-slate-800 mb-4">当前持仓</h2><div class="overflow-x-auto"><table class="w-full text-sm text-left"> <thead class="table-header"><tr> <th class="p-3">币种</th><th class="p-3">方向</th><th class="p-3">数量</th> <th class="p-3">开仓价</th><th class="p-3">杠杆</th><th class="p-3">保证金</th> <th class="p-3">浮动盈亏</th> <th class="p-3">收益率</th> <th class="p-3">目标价 (TP)</th><th class="p-3">止损价 (SL)</th> <th class="p-3">失效条件</th> </tr></thead> <tbody id="positions-table" class="divide-y divide-slate-200"></tbody> </table></div></div>
                <div class="card p-6 rounded-lg"><h2 class="text-xl font-bold text-slate-800 mb-4">最近平仓记录</h2><div class="overflow-x-auto"><table class="w-full text-sm text-left"> <thead class="table-header"><tr> <th class="p-3">币种</th><th class="p-3">方向</th><th class="p-3">杠杆</th> <th class="p-3">保证金</th><th class="p-3">手续费</th><th class="p-3">净盈亏</th> </tr></thead> <tbody id="trades-table" class="divide-y divide-slate-200"></tbody> </table></div></div>
            </div>
        </div>
        <script>
            // (JavaScript 保持 V22.1 不变)
            let equityChart;
            const CHART_TICK_COLOR = '#64748b'; const CHART_GRID_COLOR = '#e2e8f0'; const CHART_LINE_COLOR = '#2563eb';

            function formatNumber(num, digits = 2) {
                if (num === null || typeof num === 'undefined' || isNaN(num)) return '--';
                if (Math.abs(num) < 1e-9) return (0).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
                return num.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
            }
            function formatNumberFlexibleDigits(num, maxDigits = 4) {
                 if (num === null || typeof num === 'undefined' || isNaN(num)) return '--';
                 if (Math.abs(num) < 1e-9) return '0';
                 const absNum = Math.abs(num);
                 let digits = 2;
                 if (absNum < 0.01) digits = maxDigits;
                 else if (absNum < 1) digits = 4;
                 else if (absNum < 100) digits = 2;
                 else digits = 2; 
                 digits = Math.min(digits, maxDigits); 
                 return num.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
            }

            function updateData() {
                fetch('/api/alpha/status').then(response => {
                    if (!response.ok) {
                         return response.json().then(errData => { throw new Error(`HTTP ${response.status}: ${errData.message || response.statusText}`); })
                         .catch(() => { throw new Error(`HTTP ${response.status}: ${response.statusText}`); });
                    }
                    return response.json();
                }).then(data => {
                     if (data.error) { throw new Error(`Backend API Error: ${data.message || data.error}`); }

                    const mode = document.getElementById('mode-badge');
                    mode.textContent = data.is_live ? '实盘' : '模拟';
                    mode.className = (data.is_live ? 'live-badge' : 'paper-badge') + ' text-sm font-bold py-1 px-3 rounded-full';

                    document.getElementById('portfolio-equity').textContent = '$' + formatNumber(data.portfolio_equity);
                    document.getElementById('available-cash').textContent = '$' + formatNumber(data.available_cash);

                    const pnlEl = document.getElementById('total-pnl');
                    pnlEl.textContent = (data.total_pnl >= 0 ? '+' : '') + '$' + formatNumber(data.total_pnl);
                    pnlEl.className = (data.total_pnl >= 0 ? 'profit' : 'loss') + ' text-4xl font-bold mt-2';

                    const perfEl = document.getElementById('performance');
                    perfEl.textContent = (data.performance_percent >= 0 ? '+' : '') + formatNumber(data.performance_percent, 2) + '%'; 
                    perfEl.className = (data.performance_percent >= 0 ? 'profit' : 'loss') + ' text-4xl font-bold mt-2';

                    document.getElementById('strategy-summary').textContent = data.strategy_summary || '等待AI首次决策...';

                    const tickerBar = document.getElementById('ticker-bar');
                    tickerBar.innerHTML = '';
                    if (data.symbols && data.ticker_data) {
                        data.symbols.forEach(symbol => {
                            const ticker = data.ticker_data[symbol];
                            if (!ticker) return;
                            const shortName = symbol.split('/')[0];
                            const changeClass = ticker.percentage >= 0 ? 'profit' : 'loss';
                            const sign = ticker.percentage >= 0 ? '+' : '';
                            tickerBar.innerHTML += `<div class="flex-shrink-0 flex items-baseline space-x-2 border-r border-slate-200 px-4 py-2"><span class="text-sm font-bold text-gray-800">${shortName}</span><span class="text-sm font-medium text-gray-900">$${formatNumber(ticker.last, 2)}</span><span class="text-sm font-bold ${changeClass}">${sign}${formatNumber(ticker.percentage, 2)}%</span></div>`;
                        });
                    }

                    const posTable = document.getElementById('positions-table');
                    posTable.innerHTML = '';
                    const openPositions = Array.isArray(data.open_positions) ? data.open_positions : [];
                    if (openPositions.length > 0) {
                        openPositions.forEach(pos => {
                            if (typeof pos !== 'object' || pos === null) return;
                            const pnl = pos.unrealized_pnl; 
                            const pnlClass = (pnl !== null && pnl >= 0) ? 'profit' : 'loss';
                            const sideClass = pos.side === 'long' ? 'side-long' : 'side-short';
                            const profit_rate = pos.profit_rate;
                            const rateClass = (profit_rate !== null && profit_rate >= 0) ? 'profit' : 'loss';
                            const symbol = (pos.symbol || 'N/A').split(':')[0];
                            const side = (pos.side || 'N/A').toUpperCase();
                            const size = pos.size; 
                            const entry_price = pos.entry_price; 
                            const leverage = pos.leverage || 'N/A';
                            const margin = pos.margin; 
                            const tp = pos.take_profit || pos.ai_suggested_take_profit || 'N/A';
                            const sl = pos.stop_loss || pos.ai_suggested_stop_loss || 'N/A';
                            const invalidCondition = pos.invalidation_condition || 'N/A';

                            posTable.innerHTML += `<tr class="hover:bg-slate-50">
                                <td class="p-3 font-bold text-slate-800">${symbol}</td>
                                <td class="p-3 ${sideClass}">${side}</td>
                                <td class="p-3">${formatNumberFlexibleDigits(size, 8)}</td> <td class="p-3">${formatNumberFlexibleDigits(entry_price, 4)}</td>
                                <td class="p-3">${leverage}${typeof leverage === 'number' ? 'x' : ''}</td>
                                <td class="p-3">${formatNumber(margin)}</td>
                                <td class="p-3 ${pnlClass}">${formatNumber(pnl)}</td>
                                <td class="p-3 ${rateClass} font-bold">${formatNumber(profit_rate, 2)}%</td>
                                <td class="p-3">${formatNumberFlexibleDigits(tp, 4)}</td>
                                <td class="p-3">${formatNumberFlexibleDigits(sl, 4)}</td>
                                <td class="p-3 text-xs text-slate-500 max-w-[150px] overflow-hidden truncate" title="${invalidCondition}">${invalidCondition}</td>
                            </tr>`;
                        });
                    } else { 
                        posTable.innerHTML = '<tr><td colspan="11" class="p-3 text-center text-slate-400">当前无持仓</td></tr>'; 
                    }

                    const tradesTable = document.getElementById('trades-table');
                    tradesTable.innerHTML = '';
                    const historyToDisplay = Array.isArray(data.trade_history) ? data.trade_history : [];
                    if (historyToDisplay.length > 0) {
                        historyToDisplay.slice().reverse().forEach(trade => {
                             if (typeof trade !== 'object' || trade === null) return;
                            const pnl = trade.net_pnl || trade.pnl; 
                            const pnlClass = (pnl !== null && pnl >= 0) ? 'profit' : 'loss';
                            const sideClass = trade.side === 'long' ? 'side-long' : 'side-short';
                            const leverage = trade.leverage || 'N/A';
                            tradesTable.innerHTML += `<tr class="hover:bg-slate-50"> <td class="p-3 font-bold text-slate-800">${(trade.symbol || 'N/A').split(':')[0]}</td> <td class="p-3 ${sideClass}">${(trade.side || 'N/A').toUpperCase()}</td> <td class="p-3">${leverage}${typeof leverage === 'number' ? 'x' : ''}</td> <td class="p-3">${formatNumber(trade.margin)}</td> <td class="p-3">${formatNumber(trade.fees)}</td> <td class="p-3 ${pnlClass}">${(pnl !== null && pnl >= 0 ? '+' : '') + formatNumber(pnl)}</td> </tr>`;
                        });
                    } else { tradesTable.innerHTML = '<tr><td colspan="6" class="p-3 text-center text-slate-400">暂无平仓记录</td></tr>'; }

                    const logsToDisplay = Array.isArray(data.logs) ? data.logs : [];
                    const logsContainer = document.getElementById('logs-container');
                    const currentScrollTop = logsContainer.scrollTop;
                    const isScrolledToBottom = logsContainer.scrollHeight - logsContainer.clientHeight <= logsContainer.scrollTop + 1;
                    logsContainer.innerHTML = logsToDisplay.join('\\n'); 
                    setTimeout(() => { if (isScrolledToBottom) { logsContainer.scrollTop = logsContainer.scrollHeight; } else { logsContainer.scrollTop = currentScrollTop; } }, 0);

                    if (!equityChart) {
                        const ctx = document.getElementById('equity-chart').getContext('2d');
                        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
                        gradient.addColorStop(0, 'rgba(37, 99, 235, 0.1)'); gradient.addColorStop(1, 'rgba(37, 99, 235, 0)');
                    equityChart = new Chart(ctx, { type: 'line', data: { datasets: [{ label: '资产净值', data: [], borderColor: CHART_LINE_COLOR, borderWidth: 2, tension: 0.1, pointRadius: 0, fill: true, backgroundColor: gradient }] }, options: { maintainAspectRatio: false, scales: { x: { type: 'time', time: { }, grid: { color: CHART_GRID_COLOR, borderColor: CHART_GRID_COLOR }, ticks: { color: CHART_TICK_COLOR } }, y: { position: 'right', grid: { color: CHART_GRID_COLOR, borderColor: CHART_GRID_COLOR }, ticks: { color: CHART_TICK_COLOR, callback: function(value) { return '$' + value.toLocaleString(); } } } }, plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false, callbacks: { label: function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += '$' + context.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }); } return label; } } } }, interaction: { mode: 'index', intersect: false } } });
                    }
                    const equityHistoryData = Array.isArray(data.equity_history) ? data.equity_history : [];
                    equityChart.data.datasets[0].data = equityHistoryData.map(d => ({ x: d.timestamp, y: d.equity }));
                    equityChart.update('none');

                }).catch(err => {
                     console.error("Failed to fetch or process data:", err);
                     document.getElementById('strategy-summary').textContent = `获取或处理数据失败: ${err.message}. 请检查后端服务日志和网络连接。`;
                     document.getElementById('portfolio-equity').textContent = '--'; document.getElementById('available-cash').textContent = '--'; document.getElementById('total-pnl').textContent = '--'; document.getElementById('performance').textContent = '--';
                     document.getElementById('positions-table').innerHTML = '<tr><td colspan="11" class="p-3 text-center text-red-500">无法加载持仓数据</td></tr>';
                     document.getElementById('trades-table').innerHTML = '<tr><td colspan="6" class="p-3 text-center text-red-500">无法加载交易记录</td></tr>';
                     document.getElementById('logs-container').textContent = '无法加载日志。';
                     if (equityChart) { equityChart.data.datasets[0].data = []; equityChart.update('none'); }
                }).finally(() => { document.getElementById('loader').style.display = 'none'; });
            }
            document.addEventListener('DOMContentLoaded', () => { updateData(); setInterval(updateData, 10000); });
        </script>
    </body>
    </html>
    """
    return web.Response(text=html_content, content_type='text/html')

async def start_alpha_web_server(alpha_trader):
    # (此函数保持不变)
    app = web.Application()
    app['alpha_trader'] = alpha_trader
    app.router.add_get('/', handle_root)
    app.router.add_get('/api/alpha/status', get_alpha_trader_status)
    runner = web.AppRunner(app)
    await runner.setup()
    port = 58183
    site = web.TCPSite(runner, '0.0.0.0', port)
    try:
        await site.start()
        logging.info(f"🚀 AI Alpha Trader Dashboard 已启动: http://127.0.0.1:{port}")
        return site
    except Exception as e:
        logging.critical(f"启动 Alpha Web 服务器失败，端口 {port} 可能被占用。错误: {e}")
        await runner.cleanup()
        return None
