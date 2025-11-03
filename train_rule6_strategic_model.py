# 文件名: train_rule6_strategic_model.py
# 描述: [新] 专门为 Rule 6 (AI) 策略训练的 "战略" 模型。
#      使用 "TARGET_LOOKAHEAD" (例如 75 分钟) 目标，为 AI 提供中期趋势确认。

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

# (这是您原始的 75 分钟预测目标)
TARGET_THRESHOLD = 0.005 # 0.5% 
TARGET_LOOKAHEAD = 5 # 预测未来 5 根K线 (5 * 15m = 75 分钟)
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
    创建 MTF 特征和 "75 分钟" 目标
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
    
    # --- 目标工程 (Y) ---
    print(f"Creating target (Y) using TARGET_LOOKAHEAD = {TARGET_LOOKAHEAD} (75 分钟)...")
    
    future_return = base_df['c'].pct_change(TARGET_LOOKAHEAD).shift(-TARGET_LOOKAHEAD)
    
    base_df['target'] = 0 
    base_df.loc[future_return > TARGET_THRESHOLD, 'target'] = 1 
    base_df.loc[future_return < -TARGET_THRESHOLD, 'target'] = -1 
    
    df_final = pd.concat([features, base_df['target']], axis=1)
    
    df_final = df_final.dropna()
    
    print("Feature and target creation complete.")
    return df_final

def train_model(df_final, symbol_safe):
    """
    训练模型并使用 "Rule 6" 名称保存
    """
    print(f"Training STRATEGIC (Rule 6) model for {symbol_safe}...")
    
    X = df_final[FEATURE_NAMES] 
    y = df_final['target']
    
    X = X[y != 0]
    y = y[y != 0]
    
    if len(y) < 100:
        print(f"错误: {symbol_safe} 训练数据不足 (UP/DOWN 样本 < 100)。跳过。")
        return

    # --- 使用时序分割 ---
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
    
    print(f"\n--- {symbol_safe} STRATEGIC (Rule 6) 模型评估 (时序测试集) ---")
    y_pred = model.predict(X_test_scaled)
    
    # --- [修复] 移除了 'zero_division_report=0' ---
    print(classification_report(y_test, y_pred, target_names=['DOWN (-1)', 'UP (1)']))
    # --- [修复结束] ---
    
    print(f"\nSaving STRATEGIC models for {symbol_safe}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, f'rf_classifier_rule6_strategic_{symbol_safe}.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_rule6_strategic_{symbol_safe}.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"战略模型已保存到: {model_path}")
    print(f"战略缩放器已保存到: {scaler_path}")
    print(f"--- {symbol_safe} 战略模型训练完成 ---")

# --- 主执行流程 ---
if __name__ == "__main__":
    
    for symbol_ccxt in SYMBOLS_LIST:
        symbol_safe = symbol_ccxt.split(':')[0].replace('/', '') 
        print(f"\n{'='*20} 正在训练战略 (Rule 6) 模型: {symbol_safe} {'='*20}")
        
        all_dfs = load_all_dataframes(symbol_safe)
        
        if all_dfs:
            model_data = create_multitimeframe_features(all_dfs)
            
            if not model_data.empty:
                train_model(model_data, symbol_safe)
            else:
                print(f"错误: 未能为 {symbol_safe} 创建模型数据。")
                
    print("\n--- 所有币种战略模型训练完毕 ---")
