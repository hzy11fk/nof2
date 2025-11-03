# 文件名: train_rule8_tactical_model.py
# 描述: [新] 专门为 Rule 8 (Python) 策略训练的 "战术" 模型。
#      使用 "Meta-Labeling" 逻辑，目标是预测价格是先触及 1R 止盈还是 1R 止损。

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys

# --- 关键：允许脚本导入同目录下的模块 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import futures_settings
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保此脚本与 config.py 在同一目录下。")
    sys.exit(1)

# --- 配置 ---
SYMBOLS_LIST = [
    'BTC/USDT',
    'ETH/USDT',
    'SOL/USDT',
    'DOGE/USDT',
    'BNB/USDT',
    'XRP/USDT'
    # ... 添加您所有的币种 ...
]

DATA_DIR = 'data'
MODELS_DIR = 'models'
BASE_TIMEFRAME = '15m' 

FEATURE_NAMES = [
    '5min_rsi_14', '5min_adx_14', '5min_volume_ratio', '5min_price_change_pct',
    '15min_rsi_14', '15min_adx_14', '15min_volume_ratio', '15min_price_change_pct',
    '1hour_adx_14', '1hour_price_change_pct'
]

# (我们假设 R:R = 1:1)
RISK_REWARD_RATIO = 1.0 
# (我们检查未来 20 根 15m K线 = 5 小时内是否能决出胜负)
LOOKAHEAD_CANDLES = 20
# ---

def load_all_dataframes(symbol_safe):
    """
    加载特定币种的所有时间框架 CSV 文件
    """
    print(f"正在加载 {symbol_safe} 的数据...")
    dfs = {}
    timeframes = ['5m', '15m', '1h', '4h']
    
    for tf in timeframes:
        path = os.path.join(DATA_DIR, f"{symbol_safe}_{tf}.csv")
        if not os.path.exists(path):
            print(f"错误: 文件 {path} 未找到。请先运行 fetch_binance_data.py")
            return None
        
        df = pd.read_csv(path, index_col='ts', parse_dates=True)
        df = df.astype(float)
        dfs[tf] = df
        
    print(f"{symbol_safe} 的所有数据已加载。")
    return dfs

def create_multitimeframe_features(dfs):
    """
    创建 MTF 特征和 "Meta-Labeled" 目标
    """
    print("Creating Multi-Timeframe (MTF) features...")
    
    base_df = dfs[BASE_TIMEFRAME].copy()
    base_index = base_df.index
    
    features = pd.DataFrame(index=base_index)
    
    df_5m_resampled = dfs['5m'].reindex(base_index, method='ffill')
    df_1h_resampled = dfs['1h'].reindex(base_index, method='ffill')
    df_4h_resampled = dfs['4h'].reindex(base_index, method='ffill')
    
    # --- 1. 创建 5m 特征 ---
    features['5min_rsi_14'] = ta.rsi(df_5m_resampled['c'], 14)
    features['5min_adx_14'] = ta.adx(df_5m_resampled['h'], df_5m_resampled['l'], df_5m_resampled['c'], 14)['ADX_14']
    features['5min_volume_ratio'] = (df_5m_resampled['v'] / df_5m_resampled['v'].rolling(20).mean())
    features['5min_price_change_pct'] = df_5m_resampled['c'].pct_change()

    # --- 2. 创建 15m 特征 ---
    features['15min_rsi_14'] = ta.rsi(base_df['c'], 14)
    features['15min_adx_14'] = ta.adx(base_df['h'], base_df['l'], base_df['c'], 14)['ADX_14']
    features['15min_volume_ratio'] = (base_df['v'] / base_df['v'].rolling(20).mean())
    features['15min_price_change_pct'] = base_df['c'].pct_change()
    
    # --- 3. 创建 1h 特征 ---
    features['1hour_adx_14'] = ta.adx(df_1h_resampled['h'], df_1h_resampled['l'], df_1h_resampled['c'], 14)['ADX_14']
    features['1hour_price_change_pct'] = df_1h_resampled['c'].pct_change()
    
    
    # --- 目标工程 (Y) [已修改] ---
    print("Creating target (Y) using Meta-Labeling (SL/TP check)...")
    
    # 1. 获取止损所需的 ATR (使用 1h ATR，与 Rule 8 执行时一致)
    atr_1h = ta.atr(df_1h_resampled['h'], df_1h_resampled['l'], df_1h_resampled['c'], 14)
    atr_1h_aligned = atr_1h.reindex(base_index, method='ffill')
    
    # 2. 定义止损/止盈距离
    # (这必须与 alpha_trader.py -> _build_python_order 中的 multiplier 一致)
    ATR_MULTIPLIER = futures_settings.INITIAL_STOP_ATR_MULTIPLIER 
    
    base_df['sl_distance'] = atr_1h_aligned * ATR_MULTIPLIER
    base_df['tp_distance'] = base_df['sl_distance'] * RISK_REWARD_RATIO
    
    target = np.zeros(len(base_df))
    
    # (这个循环很慢, 但在离线训练中是可接受的)
    for i in range(len(base_df) - LOOKAHEAD_CANDLES):
        entry_price = base_df['c'].iloc[i]
        
        tp_price_long = entry_price + base_df['tp_distance'].iloc[i]
        sl_price_long = entry_price - base_df['sl_distance'].iloc[i]
        
        tp_price_short = entry_price - base_df['tp_distance'].iloc[i]
        sl_price_short = entry_price + base_df['sl_distance'].iloc[i]

        future_highs = base_df['h'].iloc[i+1 : i+1+LOOKAHEAD_CANDLES]
        future_lows = base_df['l'].iloc[i+1 : i+1+LOOKAHEAD_CANDLES]

        hit_tp_long = (future_highs >= tp_price_long).any()
        hit_sl_long = (future_lows <= sl_price_long).any()

        hit_tp_short = (future_lows <= tp_price_short).any()
        hit_sl_short = (future_highs >= sl_price_short).any()

        # (我们优先做多)
        if hit_tp_long and not hit_sl_long:
            target[i] = 1 # 做多成功
        elif hit_sl_long and not hit_tp_long:
            target[i] = -1 # 做多失败
        # (如果做多未触发, 才考虑做空)
        elif hit_tp_short and not hit_sl_short:
            target[i] = -1 # 做空成功
        elif hit_sl_short and not hit_tp_short:
             target[i] = 1 # 做空失败

    base_df['target'] = target
    
    df_final = pd.concat([features, base_df['target']], axis=1)
    
    df_final = df_final.dropna()
    
    print("Feature and target creation complete (Meta-Labeled).")
    return df_final

def train_model(df_final, symbol_safe):
    """
    训练模型并保存 Scaler 和 Model
    """
    print(f"Training TACTICAL (Rule 8) model for {symbol_safe}...")
    
    X = df_final[FEATURE_NAMES] 
    y = df_final['target']
    
    # 关键：我们只对“明确的”上涨或下跌感兴趣
    X = X[y != 0]
    y = y[y != 0]
    
    if len(y) < 100:
        print(f"错误: {symbol_safe} 训练数据不足 (UP/DOWN 样本 < 100)。跳过。")
        return

    # --- [修改] 使用时序分割 ---
    test_size_percent = 0.20
    test_point = int(len(X) * (1.0 - test_size_percent))
    
    X_train = X.iloc[:test_point]
    y_train = y.iloc[:test_point]
    X_test = X.iloc[test_point:]
    y_test = y.iloc[test_point:]
        
    print(f"训练集: {X_train.index[0]} 到 {X_train.index[-1]} (大小: {len(X_train)})")
    print(f"测试集: {X_test.index[0]} 到 {X_test.index[-1]} (大小: {len(X_test)})")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1 
    )
    model.fit(X_train_scaled, y_train)
    
    print(f"\n--- {symbol_safe} TACTICAL (Rule 8) 模型评估 (时序测试集) ---")
    y_pred = model.predict(X_test_scaled)
    
    # --- [修复] 移除了 'zero_division_report=0' ---
    print(classification_report(y_test, y_pred, target_names=['DOWN/FAIL (-1)', 'UP/FAIL (1)']))
    # --- [修复结束] ---
    
    print(f"\nSaving TACTICAL models for {symbol_safe}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # --- [修改] 保存为 TACTICAL 名称 ---
    model_path = os.path.join(MODELS_DIR, f'rf_classifier_rule8_tactical_{symbol_safe}.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_rule8_tactical_{symbol_safe}.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"战术模型已保存到: {model_path}")
    print(f"战术缩放器已保存到: {scaler_path}")
    print(f"--- {symbol_safe} 战术模型训练完成 ---")

# --- 主执行流程 ---
if __name__ == "__main__":
    
    for symbol_ccxt in SYMBOLS_LIST:
        symbol_safe = symbol_ccxt.split(':')[0].replace('/', '') 
        print(f"\n{'='*20} 正在训练战术 (Rule 8) 模型: {symbol_safe} {'='*20}")
        
        all_dfs = load_all_dataframes(symbol_safe)
        
        if all_dfs:
            model_data = create_multitimeframe_features(all_dfs)
            
            if not model_data.empty:
                train_model(model_data, symbol_safe)
            else:
                print(f"错误: 未能为 {symbol_safe} 创建模型数据。")
                
    print("\n--- 所有币种战术模型训练完毕 ---")
