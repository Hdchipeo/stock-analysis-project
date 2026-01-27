import pandas as pd
import os

def calculate_descriptive_stats():
    """
    Tính toán các chỉ số thống kê mô tả cho Close và Volume.
    """
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(base_dir, "data", "raw", "stock_data.csv")

    if not os.path.exists(filename):
        print(f"Lỗi: Không tìm thấy file {filename}")
        return

    df = pd.read_csv(filename)
    
    # Chọn các biến cần thiết
    cols = ["Close", "Volume"]
    
    stats_list = []
    
    for col in cols:
        series = df[col]
        stats = {
            "Variable": col,
            "Mean (Trung bình)": series.mean(),
            "Median (Trung vị)": series.median(),
            "Std (Độ lệch chuẩn)": series.std(),
            "Min (Nhỏ nhất)": series.min(),
            "Max (Lớn nhất)": series.max(),
            "Skewness (Độ lệch)": series.skew(),
            "Kurtosis (Độ nhọn)": series.kurtosis()
        }
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    print("\n--- Thống kê mô tả (Descriptive Statistics) ---")
    print(stats_df.to_markdown(index=False))
    
    # Phân tích điểm nhấn (Std)
    close_std = stats_df.loc[stats_df["Variable"] == "Close", "Std (Độ lệch chuẩn)"].values[0]
    volume_std = stats_df.loc[stats_df["Variable"] == "Volume", "Std (Độ lệch chuẩn)"].values[0]
    
    print("\n--- Điểm nhấn phân tích ---")
    print(f"1. Độ lệch chuẩn của giá Close là {close_std:.2f}. ")
    print(f"   - Nếu giá trị này cao so với mức giá trung bình, điều đó cho thấy cổ phiếu có mức độ biến động lớn (High Volatility), đồng nghĩa với rủi ro cao hơn cho nhà đầu tư.")
    print(f"2. Độ lệch chuẩn của Volume là {volume_std:.2f}, phản ánh sự không ổn định trong tính thanh khoản hàng ngày.")

    return stats_df

if __name__ == "__main__":
    calculate_descriptive_stats()
