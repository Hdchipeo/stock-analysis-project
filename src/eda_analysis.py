import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import os

def run_eda_analysis(input_file="preprocessed_data.csv", raw_data="stock_data.csv"):
    """
    Thực hiện phân tích EDA và vẽ biểu đồ.
    """
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_raw_dir = os.path.join(base_dir, "data", "raw")
    data_processed_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results", "figures")

    # Override defaults with correct paths if not specified or if defaults are filenames
    if input_file == "preprocessed_data.csv":
         input_file = os.path.join(data_processed_dir, "preprocessed_data.csv")
    if raw_data == "stock_data.csv":
         raw_data = os.path.join(data_raw_dir, "stock_data.csv")

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return
    
    if not os.path.exists(raw_data):
        print(f"Lỗi: Không tìm thấy file {raw_data}")
        return

    # Load preprocessed and raw data
    print("Loading data...")
    df_pre = pd.read_csv(input_file)
    df_pre['Date'] = pd.to_datetime(df_pre['Date'], utc=True)
    df_pre.set_index('Date', inplace=True)
    df_pre.index = df_pre.index.tz_convert(None)

    df_raw = pd.read_csv(raw_data)
    # Parse datetime - KHÔNG convert sang UTC để giữ đúng ngày
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    # Loại bỏ timezone info nhưng giữ nguyên giá trị ngày
    df_raw['Date'] = df_raw['Date'].dt.tz_localize(None)
    df_raw.set_index('Date', inplace=True)
    df_raw.sort_index(inplace=True)

    # 1. Trend Analysis (Candlestick + MA + Volume)
    print("1. Drawing Trend Analysis (Candlestick Chart)...")
    # Lấy toàn bộ dữ liệu để vẽ tổng quan từ 2021-2026
    df_full = df_raw.copy()
    
    # Tạo style cho mplfinance
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    
    # Vẽ biểu đồ
    fig, axes = mpf.plot(df_full, type='candle', style=s,
             mav=(30), # Moving Average 30 ngày
             volume=True, 
             title='Trend Analysis: FPT.VN Stock Price (2021-2026)',
             ylabel='Price (VND)',
             ylabel_lower='Volume',
             figsize=(14, 8),
             returnfig=True)
    fig.savefig(os.path.join(results_dir, 'trend_analysis.png'))
    plt.close(fig)
    print("   - Saved 'trend_analysis.png'")

    # 2. Distribution Analysis (Histogram + KDE for Log Returns)
    print("2. Drawing Distribution Analysis (Histogram + KDE)...")
    plt.figure(figsize=(10, 6))
    
    # Tính lại Log Returns từ raw data để chính xác hơn cho visual (hoặc dùng preprocessed)
    # Ở đây dùng preprocessed vì đã tính sẵn
    log_returns = df_pre['Log_Returns'].dropna()

    sns.histplot(log_returns, kde=True, stat="density", linewidth=0)
    plt.title('Distribution of Log Returns (Fat Tails Analysis)')
    plt.xlabel('Log Returns')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Thêm đường chuẩn (Normal Distribution) để so sánh
    from scipy.stats import norm
    mu, std = norm.fit(log_returns)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, 'distribution_analysis.png'))
    plt.close()
    print("   - Saved 'distribution_analysis.png'")

    # 3. Correlation Analysis (Heatmap)
    print("3. Drawing Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    
    # Chọn các biến quan trọng để tương quan
    corr_cols = ['Close', 'Volume', 'Log_Returns', 'RSI_14', 'MACD_12_26_9', 'SMA_7', 'SMA_30']
    # Kiểm tra xem các cột có trong df_pre không
    available_cols = [c for c in corr_cols if c in df_pre.columns]
    
    corr_matrix = df_pre[available_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("FPT.VN Price Trend with MA30", fontsize=16)
    plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
    plt.close()
    print("   - Saved 'correlation_heatmap.png'")

    # 4. Seasonality Analysis (Boxplot)
    print("4. Drawing Seasonality Analysis...")
    df_raw['Month'] = df_raw.index.month
    
    # Xử lý DayOfWeek - sử dụng dayofweek số (0=Monday, 4=Friday) rồi map sang tên
    day_names_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
    df_raw['DayOfWeek'] = df_raw.index.dayofweek.map(day_names_map)
    
    # Debug: In số lượng mỗi ngày
    print("   DEBUG - Số phiên theo ngày:")
    day_counts = df_raw['DayOfWeek'].value_counts()
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        count = day_counts.get(day, 0)
        print(f"      {day}: {count} phiên")
    
    # Lọc chỉ lấy các ngày giao dịch (Monday-Friday), loại bỏ NaN nếu có
    df_raw_filtered = df_raw[df_raw['DayOfWeek'].notna()].copy()
    
    # Sắp xếp thứ tự ngày trong tuần
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot theo Tháng (Volume) - Fix FutureWarning bằng cách sử dụng hue
    sns.boxplot(x='Month', y='Volume', data=df_raw, ax=axes[0], hue='Month', palette="viridis", legend=False)
    axes[0].set_title('Volume Distribution by Month')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Volume')
    
    # Boxplot theo Ngày trong tuần (Volume) - Fix FutureWarning
    sns.boxplot(x='DayOfWeek', y='Volume', data=df_raw_filtered, ax=axes[1], 
                order=days_order, hue='DayOfWeek', hue_order=days_order, palette="magma", legend=False)
    axes[1].set_title('Volume Distribution by Day of Week')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Volume')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'seasonality_analysis.png'))
    plt.close()
    print("   - Saved 'seasonality_analysis.png'")
    
    print("EDA Analysis Completed.")

if __name__ == "__main__":
    run_eda_analysis()
