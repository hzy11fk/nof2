# 文件名: train_anomaly_detector.py
# 描述: [新] 离线训练 异常检测 (IsolationForest) 模型。
#      它为 SYMBOLS_LIST 中的每个币种创建一个单独的模型。

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

# --- 配置 ---
# !!! 关键：此列表必须与您的 config.py 和其他训练脚本一致 !!!
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
BASE_TIMEFRAME = '15m' # 我们在 15m K线上检测异常

# (这是我们根据 IMG_7354.jpg 定义的特征列表)
FEATURE_NAMES = [
    'volatility_5_20_ratio',
    'volume_ratio',
    'price_deviation'
]

# IsolationForest 的“污染”参数
# 这意味着我们假设 10% 的数据点是“异常”的
CONTAMINATION_RATE = 0.1
# ---

def load_data(symbol_safe):
    """
    加载特定币种的 15m K线数据
    """
    print(f"正在加载 {symbol_safe} 的 {BASE_TIMEFRAME} 数据...")
    path = os.path.join(DATA_DIR, f"{symbol_safe}_{BASE_TIMEFRAME}.csv")
    if not os.path.exists(path):
        print(f"错误: 文件 {path} 未找到。请先运行 fetch_binance_data.py")
        return None
    
    df = pd.read_csv(path, index_col='ts', parse_dates=True)
    df = df.astype(float)
    print(f"{symbol_safe} 的数据已加载。")
    return df

def create_anomaly_features(df):
    """
    根据 IMG_7354.jpg 的建议创建特征
    """
    print("Creating anomaly features...")
    features = pd.DataFrame(index=df.index)
    
    # 1. 波动率比率 (Volatility Ratio)
    df['returns'] = df['c'].pct_change()
    rolling_vol_5 = df['returns'].rolling(5).std()
    rolling_vol_20 = df['returns'].rolling(20).std()
    features['volatility_5_20_ratio'] = rolling_vol_5 / rolling_vol_20

    # 2. 交易量比率 (Volume Ratio)
    features['volume_ratio'] = df['v'] / df['v'].rolling(20).mean()

    # 3. 价格偏离 (Price Deviation)
    rolling_mean_20 = df['c'].rolling(20).mean()
    rolling_std_20 = df['c'].rolling(20).std()
    features['price_deviation'] = (df['c'] - rolling_mean_20) / rolling_std_20
    
    # 清理所有在计算中（例如 rolling）产生的 NaN 值
    df_final = features.dropna()
    
    print("Anomaly features created.")
    return df_final

def train_anomaly_model(df_features, symbol_safe):
    """
    训练并保存 Anomaly Detector (Scaler 和 Model)
    """
    print(f"Training anomaly model for {symbol_safe}...")
    
    X = df_features[FEATURE_NAMES]
    
    if len(X) < 200:
        print(f"错误: {symbol_safe} 训练数据不足 (样本 < 200)。跳过。")
        return

    # --- 时序分割 ---
    # 我们用前 80% 的数据来“拟合”什么是“正常”
    train_size_percent = 0.80
    train_point = int(len(X) * train_size_percent)
    
    X_train = X.iloc[:train_point]
    X_test = X.iloc[train_point:] # (测试集在这里仅用于观察，模型是无监督的)
        
    print(f"训练集: {X_train.index[0]} 到 {X_train.index[-1]} (大小: {len(X_train)})")
    
    # --- 1. 缩放器 (Scaler) ---
    # 使用 RobustScaler 来处理异常值
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # --- 2. 异常检测器 (Detector) ---
    model = IsolationForest(
        contamination=CONTAMINATION_RATE, # (原文为0.1)
        random_state=42                 #
    )
    model.fit(X_train_scaled)
    
    # (可选) 检查一下模型在“未来”数据上的表现
    X_test_scaled = scaler.transform(X_test)
    scores_test = model.decision_function(X_test_scaled) #
    print(f"测试集上的平均异常得分: {np.mean(scores_test):.4f}")
    
    # --- 3. 保存模型和缩放器 ---
    print(f"\nSaving models for {symbol_safe}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, f'anomaly_detector_{symbol_safe}.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'anomaly_scaler_{symbol_safe}.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"异常检测器已保存到: {model_path}")
    print(f"异常缩放器已保存到: {scaler_path}")
    print(f"--- {symbol_safe} 异常模型训练完成 ---")

# --- 主执行流程 ---
if __name__ == "__main__":
    
    for symbol_ccxt in SYMBOLS_LIST:
        symbol_safe = symbol_ccxt.split(':')[0].replace('/', '') # 'BTC/USDT' -> 'BTCUSDT'
        print(f"\n{'='*20} 正在训练异常模型: {symbol_safe} {'='*20}")
        
        # 1. 加载 15m CSV
        df_15m = load_data(symbol_safe)
        
        if df_15m is not None:
            # 2. 创建异常特征
            df_features = create_anomaly_features(df_15m)
            
            # 3. 训练和保存
            if not df_features.empty:
                train_anomaly_model(df_features, symbol_safe)
            else:
                print(f"错误: 未能为 {symbol_safe} 创建模型数据。")
                
    print("\n--- 所有币种异常模型训练完毕 ---")
