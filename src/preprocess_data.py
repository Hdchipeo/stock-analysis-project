import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os
import json

# Helper Functions for Indicators (Manual Implementation)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def preprocess_stock_data(filename="stock_data.csv"):
    """
    Tiền xử lý dữ liệu chứng khoán: Làm sạch, kỹ thuật đặc trưng, chuẩn hóa và phân chia.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(data_processed_dir, exist_ok=True)
    results_dir = os.path.join(base_dir, "results", "figures")
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(data_raw_dir, "stock_data.csv")
    
    if not os.path.exists(filename):
        print(f"Lỗi: Không tìm thấy file {filename}")
        return

    # 1. Load data
    print("Loading data...")
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    df.index = df.index.tz_convert(None) # Remove timezone info for simpler handling
    df.sort_index(inplace=True)

    # 2. Làm sạch dữ liệu (Interpolation)
    df_clean = df.interpolate(method='linear')
    print("1. Đã xử lý dữ liệu khuyết bằng phương pháp Interpolation.")

    # 3. Xử lý ngoại lai (Outlier Detection)
    iso = IsolationForest(contamination=0.01, random_state=42)
    df_clean['Outlier'] = iso.fit_predict(df_clean[['Close', 'Volume']])
    
    # Save Outlier Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_clean.index, df_clean['Close'], label='Price', color='blue', alpha=0.5)
    outliers = df_clean[df_clean['Outlier'] == -1]
    plt.scatter(outliers.index, outliers['Close'], color='red', label='Outliers', zorder=5)
    plt.title("Stock Price & Outliers: FPT.VN", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'outliers.png'))
    plt.close()
    print("2. Đã phát hiện ngoại lai và lưu biểu đồ.")

    # 4. Kỹ thuật đặc trưng (Feature Engineering)
    # Log Returns
    df_clean['Log_Returns'] = np.log(df_clean['Close'] / df_clean['Close'].shift(1))
    
    # Technical Indicators (Manual)
    df_clean['RSI_14'] = calculate_rsi(df_clean['Close'], period=14)
    
    macd, signal = calculate_macd(df_clean['Close'])
    df_clean['MACD_12_26_9'] = macd
    df_clean['MACDs_12_26_9'] = signal

    df_clean['SMA_7'] = df_clean['Close'].rolling(window=7).mean()
    df_clean['SMA_30'] = df_clean['Close'].rolling(window=30).mean()
    
    # Fill NaN
    df_clean.fillna(method='bfill', inplace=True)
    df_clean.fillna(method='ffill', inplace=True) # Ensure clean start
    
    # Sliding Window (Lag Features for Close)
    for i in range(1, 4):
        df_clean[f'Close_Lag_{i}'] = df_clean['Close'].shift(i)
    
    # Drop NaN created by lags
    df_features = df_clean.dropna()
    print("3. Đã tạo các biến đặc trưng (RSI, MACD, MA, Lags).")

    # Save Scaling Params BEFORE Scaling
    close_min = df_features['Close'].min()
    close_max = df_features['Close'].max()
    scaling_params = {"Close_min": float(close_min), "Close_max": float(close_max)}
    
    # 5. Chuẩn hóa dữ liệu (Feature Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    cols_to_scale = df_features.columns.drop('Outlier')
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features[cols_to_scale]), 
                             columns=cols_to_scale, 
                             index=df_features.index)
    df_scaled['Outlier'] = df_features['Outlier']
    print("4. Đã chuẩn hóa dữ liệu [0, 1].")

    # 6. Phân chia tập dữ liệu
    split_idx = int(len(df_scaled) * 0.8)
    train_df = df_scaled.iloc[:split_idx]
    test_df = df_scaled.iloc[split_idx:]
    
    print(f"5. Đã chia tập dữ liệu: Train ({len(train_df)}), Test ({len(test_df)}).")

    # Lưu kết quả
    df_scaled.to_csv(os.path.join(data_processed_dir, 'preprocessed_data.csv'))
    train_df.to_csv(os.path.join(data_processed_dir, 'train_data.csv'))
    test_df.to_csv(os.path.join(data_processed_dir, 'test_data.csv'))
    
    # Save Scaling Params
    with open(os.path.join(data_processed_dir, "scaling_params.json"), "w") as f:
        json.dump(scaling_params, f)
        
    print(f"6. Đã lưu dữ liệu và tham số scaling ({scaling_params}) vào 'data/processed/'.")

    return train_df, test_df

if __name__ == "__main__":
    preprocess_stock_data()
