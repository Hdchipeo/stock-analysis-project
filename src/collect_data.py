import yfinance as yf
import pandas as pd
import os

def collect_stock_data(ticker="FPT.VN", start="2021-01-01", end="2025-12-31", interval="1d"):
    """
    Tải dữ liệu lịch sử giá cổ phiếu từ yfinance.
    
    Tham số:
    - ticker: Mã cổ phiếu (default: FPT.VN)
    - start: Ngày bắt đầu (default: 2021-01-01)
    - end: Ngày kết thúc (default: 2025-12-31)
    - interval: Khoảng thời gian (default: 1d - hàng ngày)
    """
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
    # Define paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    filename = os.path.join(data_dir, "stock_data.csv")
    df.to_csv(filename)
    print(f"Đã lưu dữ liệu vào {filename}")
    return df

if __name__ == "__main__":
    collect_stock_data()

