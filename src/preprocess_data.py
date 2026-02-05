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
    print("3. Bắt đầu Feature Engineering...")
    
    # === A. Log Returns ===
    # Tỷ suất sinh lợi logarit: r_t = ln(P_t / P_{t-1})
    # Ưu điểm: Stationary, symmetric, additive
    df_clean['Log_Returns'] = np.log(df_clean['Close'] / df_clean['Close'].shift(1))
    print("   → Đã tạo Log_Returns (Target chính cho regression)")
    
    # === B. Price Direction (Classification Target) ===
    # Biến nhị phân: 1 nếu giá tăng, 0 nếu giá giảm/không đổi
    # Mục đích: Sử dụng cho mô hình phân loại (Classification)
    df_clean['Price_Direction'] = (df_clean['Log_Returns'] > 0).astype(int)
    print("   → Đã tạo Price_Direction (Target cho classification)")
    
    # === C. Technical Indicators ===
    # RSI (14): Relative Strength Index - Đo momentum
    # Giá trị 0-100: >70 = Overbought, <30 = Oversold
    df_clean['RSI_14'] = calculate_rsi(df_clean['Close'], period=14)
    
    # MACD: Moving Average Convergence Divergence - Chỉ báo xu hướng
    # MACD Line = EMA(12) - EMA(26)
    # Signal Line = EMA(9) của MACD
    macd, signal = calculate_macd(df_clean['Close'])
    df_clean['MACD_12_26_9'] = macd
    df_clean['MACDs_12_26_9'] = signal

    # Simple Moving Averages - Đường trung bình động
    # SMA_7: Xu hướng ngắn hạn (1 tuần giao dịch)
    # SMA_30: Xu hướng trung hạn (1 tháng giao dịch)
    df_clean['SMA_7'] = df_clean['Close'].rolling(window=7).mean()
    df_clean['SMA_30'] = df_clean['Close'].rolling(window=30).mean()
    print("   → Đã tạo Technical Indicators (RSI, MACD, SMA)")
    
    # === D. Volume-based Features ===
    # ⚠️ WARNING: Granger Causality Test (2026-02-04) cho thấy Volume_Change
    # KHÔNG có mối quan hệ nhân quả với Log_Returns (tất cả p-value > 0.05).
    # Các features này được giữ lại để:
    # 1. Backward compatibility với các mô hình đã train
    # 2. Có thể hữu ích như confirmation signal (không phải leading indicator)
    # 3. Cho phép so sánh model có/không có Volume features
    # 
    # TODO: Xem xét loại bỏ trong phiên bản tiếp theo nếu không cải thiện model
    
    # D1. Volume Change - % thay đổi khối lượng giao dịch
    # Mục đích: Phát hiện sự thay đổi thanh khoản
    # ⚠️ Granger test: p-value > 0.05 - KHÔNG có nhân quả
    df_clean['Volume_Change'] = df_clean['Volume'].pct_change()
    
    # NEW: Volume Diff - Log Differencing (đã được chứng minh có Granger Causality)
    # Handle volume=0 case
    vol_adjusted = df_clean['Volume'].replace(0, 1)
    df_clean['Log_Volume'] = np.log(vol_adjusted)
    df_clean['Volume_Diff'] = df_clean['Log_Volume'].diff()
    # Fill NaN created by diff
    df_clean['Volume_Diff'] = df_clean['Volume_Diff'].fillna(0)

    # D2. Volume Shock - Phát hiện khối lượng bất thường
    # Logic: Volume > Mean + 2*Std (vượt 2 độ lệch chuẩn)
    # Ý nghĩa: Báo hiệu sự kiện quan trọng (tin tức, earnings, etc.)
    volume_mean_30 = df_clean['Volume'].rolling(window=30).mean()
    volume_std_30 = df_clean['Volume'].rolling(window=30).std()
    df_clean['Volume_Shock'] = (
        df_clean['Volume'] > (volume_mean_30 + 2 * volume_std_30)
    ).astype(int)
    
    # D3. Price Volatility - Biến động giá
    # Đo bằng rolling standard deviation của Log Returns
    # Window = 30 ngày (volatility trong 1 tháng)
    df_clean['Volatility_30'] = df_clean['Log_Returns'].rolling(window=30).std()
    print("   → Đã tạo Volume Features: Volume_Diff (New!), Volume_Change, Shock")
    
    # === E. Lag Features ===
    # Sử dụng giá trị quá khứ của Log_Returns làm features
    # PACF Analysis cho thấy significant lags tại [2, 23, 27]
    # Tuy nhiên, ta sử dụng [1, 2, 3] vì:
    # 1. Lag 23, 27 có risk overfitting cao (monthly pattern có thể là noise)
    # 2. Lag 1, 2, 3 là lựa chọn thực tiễn, ổn định qua nhiều nghiên cứu
    for i in range(1, 4):
        df_clean[f'Returns_Lag_{i}'] = df_clean['Log_Returns'].shift(i)
    
    # Lag features cho Volume_Change (giữ lại để so sánh)
    for i in range(1, 3):
        df_clean[f'Volume_Change_Lag_{i}'] = df_clean['Volume_Change'].shift(i)

    # NEW: Lag features cho Volume_Diff (Quan trọng: Lag 3, 4 có Granger Causality)
    for i in range(1, 5):
        df_clean[f'Volume_Diff_Lag_{i}'] = df_clean['Volume_Diff'].shift(i)
    
    print("   → Đã tạo Lag Features (Returns_Lag_1-3, Volume_Change_Lag_1-2)")
    
    # Fill NaN values
    df_clean.fillna(method='bfill', inplace=True)
    df_clean.fillna(method='ffill', inplace=True)
    
    # Drop remaining NaN
    df_features = df_clean.dropna()
    print(f"   ✓ Feature Engineering hoàn tất: {len(df_features.columns)} features")

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
