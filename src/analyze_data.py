import pandas as pd

def analyze_and_describe_variables():
    """
    Đọc dữ liệu từ file CSV và chuẩn bị bảng mô tả biến (Bảng 1).
    """
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(base_dir, "data", "raw", "stock_data.csv")
    
    if not os.path.exists(filename):
        print(f"Lỗi: Không tìm thấy file {filename}")
        return

    df = pd.read_csv(filename)
    
    print("\n--- Cấu trúc dữ liệu (Data Structure) ---")
    print(df.info())
    
    print("\n--- Bảng 1: Mô tả biến (Variable Description Table) ---")
    
    descriptions = {
        "Date": "Ngày giao dịch (Index trong dữ liệu gốc).",
        "Open": "Giá mở cửa của cổ phiếu trong ngày giao dịch.",
        "High": "Giá cao nhất mà cổ phiếu đạt được trong ngày giao dịch.",
        "Low": "Giá thấp nhất mà cổ phiếu chạm tới trong ngày giao dịch.",
        "Close": "Giá đóng cửa cuối cùng của cổ phiếu trong ngày giao dịch.",
        "Volume": "Tổng khối lượng cổ phiếu được giao dịch trong ngày."
    }
    
    data_types = {
        "Date": "Datetime/String",
        "Open": "Continuous (Định lượng)",
        "High": "Continuous (Định lượng)",
        "Low": "Continuous (Định lượng)",
        "Close": "Continuous (Định lượng)",
        "Volume": "Continuous (Định lượng)"
    }
    
    table_data = []
    for i, col in enumerate(df.columns, 1):
        table_data.append({
            "STT": i,
            "Tên biến (Variable)": col,
            "Loại dữ liệu (Data Type)": data_types.get(col, "N/A"),
            "Ý nghĩa (Description)": descriptions.get(col, "N/A")
        })
    
    description_table = pd.DataFrame(table_data)
    print(description_table.to_markdown(index=False))
    
    return description_table

import os
if __name__ == "__main__":
    analyze_and_describe_variables()
