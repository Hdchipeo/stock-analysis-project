import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import os

def preprocess_stock_data(filename="stock_data.csv"):
    """
    Tiền xử lý dữ liệu chứng khoán: Làm sạch, kỹ thuật đặc trưng, chuẩn hóa và phân chia.
    """
    import os
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
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    df.index = df.index.tz_convert(None) # Remove timezone info for simpler handling
    df.sort_index(inplace=True)

    # 2. Làm sạch dữ liệu (Interpolation)
    # Giả sử có dữ liệu khuyết (mặc dù yfinance thường đầy đủ, nhưng đây là bước quy trình)
    df_clean = df.interpolate(method='linear')
    print("1. Đã xử lý dữ liệu khuyết bằng phương pháp Interpolation.")

    # 3. Xử lý ngoại lai (Outlier Detection) với Isolation Forest
    # Chúng ta dùng giá Close và Volume để tìm ngoại lai
    iso = IsolationForest(contamination=0.01, random_state=42)
    df_clean['Outlier'] = iso.fit_predict(df_clean[['Close', 'Volume']])
    
    # Vẽ biểu đồ ngoại lai
    plt.figure(figsize=(12, 6))
    plt.plot(df_clean.index, df_clean['Close'], label='Giá Close', color='blue', alpha=0.5)
    outliers = df_clean[df_clean['Outlier'] == -1]
    plt.scatter(outliers.index, outliers['Close'], color='red', label='Ngoại lai (Outliers)', zorder=5)
    plt.title("Stock Price & Volume: FPT.VN", fontsize=16)
    plt.xlabel('Ngày')
    plt.ylabel('Giá Close')
    plt.legend()
    plt.grid(True)
    outliers_path = os.path.join(results_dir, 'outliers.png')
    plt.savefig(outliers_path)
    plt.close()
    print(f"2. Đã phát hiện ngoại lai và lưu biểu đồ vào '{outliers_path}'.")

    # 4. Kỹ thuật đặc trưng (Feature Engineering)
    # Log Returns
    df_clean['Log_Returns'] = np.log(df_clean['Close'] / df_clean['Close'].shift(1))
    
    # Technical Indicators (RSI, MACD, MA)
    df_clean.ta.rsi(length=14, append=True)
    df_clean.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_clean.ta.sma(length=7, append=True)
    df_clean.ta.sma(length=30, append=True)
    
    # Sliding Window (Lag Features for Close)
    for i in range(1, 4):
        df_clean[f'Close_Lag_{i}'] = df_clean['Close'].shift(i)
    
    # Loại bỏ các dòng có giá trị NaN sau khi tạo feature (do shift/rolling)
    df_features = df_clean.dropna()
    print("3. Đã tạo các biến đặc trưng (RSI, MACD, MA, Log Returns, Lags).")

    # 5. Chuẩn hóa dữ liệu (Feature Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Chúng ta chuẩn hóa tất cả các cột trừ 'Outlier'
    cols_to_scale = df_features.columns.drop('Outlier')
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features[cols_to_scale]), 
                             columns=cols_to_scale, 
                             index=df_features.index)
    df_scaled['Outlier'] = df_features['Outlier']
    print("4. Đã chuẩn hóa dữ liệu về khoảng [0, 1] bằng MinMaxScaler.")

    # 6. Phân chia tập dữ liệu (Data Splitting) - 80% Train, 20% Test (Theo thời gian)
    split_idx = int(len(df_scaled) * 0.8)
    train_df = df_scaled.iloc[:split_idx]
    test_df = df_scaled.iloc[split_idx:]
    
    print(f"5. Đã chia tập dữ liệu: Train ({len(train_df)} dòng), Test ({len(test_df)} dòng).")

    # Lưu kết quả
    df_scaled.to_csv(os.path.join(data_processed_dir, 'preprocessed_data.csv'))
    train_df.to_csv(os.path.join(data_processed_dir, 'train_data.csv'))
    test_df.to_csv(os.path.join(data_processed_dir, 'test_data.csv'))
    print("6. Đã lưu dữ liệu tiền xử lý vào các file CSV trong 'data/processed/'.")

    return train_df, test_df

if __name__ == "__main__":
    preprocess_stock_data()
