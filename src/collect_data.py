"""
Module thu thập dữ liệu cổ phiếu FPT từ Yahoo Finance

Tính năng:
- Kiểm tra nếu đã có file dữ liệu thì KHÔNG tải lại (tiết kiệm thời gian, đảm bảo reproducibility)
- Chỉ tải lại khi xóa file hoặc force=True
"""

import yfinance as yf
import pandas as pd
import os


def collect_stock_data(ticker="FPT.VN", start="2021-01-01", end="2025-12-31", 
                       interval="1d", force=False):
    """
    Tải dữ liệu lịch sử giá cổ phiếu từ yfinance.
    
    Tham số:
    - ticker: Mã cổ phiếu (default: FPT.VN)
    - start: Ngày bắt đầu (default: 2021-01-01)
    - end: Ngày kết thúc (default: 2025-12-31)
    - interval: Khoảng thời gian (default: 1d - hàng ngày)
    - force: Nếu True, tải lại dữ liệu ngay cả khi đã có file
    
    Return:
    - DataFrame dữ liệu cổ phiếu
    """
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, "stock_data.csv")
    
    # Kiểm tra nếu file đã tồn tại
    if os.path.exists(filename) and not force:
        print(f"✓ Đã có sẵn dữ liệu: {filename}")
        print("  → Đang sử dụng dữ liệu có sẵn (không tải lại)")
        print("  → Xóa file hoặc dùng force=True để tải lại\n")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"  Số phiên giao dịch: {len(df)}")
        return df
    
    # Tải dữ liệu mới
    print(f"Đang tải dữ liệu cho {ticker}...")
    print(f"Giai đoạn: {start} đến {end}")
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end, interval=interval)
    
    if df.empty:
        print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng kiểm tra lại ticker.")
        return None
    
    print(f"Đã tải {len(df)} phiên giao dịch")
    
    # Chỉ giữ lại các cột cần thiết: Open, High, Low, Close, Volume
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Lưu vào file CSV
    df.to_csv(filename)
    print(f"Đã lưu dữ liệu vào {filename}")
    
    return df


if __name__ == "__main__":
    collect_stock_data()
