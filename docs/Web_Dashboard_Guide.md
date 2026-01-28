# HƯỚNG DẪN SỬ DỤNG WEB DASHBOARD (Interactive Stock Analysis)

Tài liệu này hướng dẫn cách cài đặt, khởi chạy và sử dụng Dashboard phân tích chứng khoán được xây dựng bằng **Streamlit**.

---

## 1. Giới thiệu
Web Dashboard là giao diện tương tác giúp bạn:
- Theo dõi biến động giá cổ phiếu FPT.VN cùng với dự báo từ các mô hình AI (Linear Regression, XGBoost).
- Xem các chỉ số KPI quan trọng (Giá hiện tại, Thay đổi trong ngày).
- Nhận tín hiệu tham khảo (Mua/Bán) dựa trên dự báo của mô hình.
- So sánh hiệu suất giữa các mô hình khác nhau.

## 2. Yêu cầu hệ thống
Đảm bảo bạn đã cài đặt các thư viện cần thiết. Nếu chưa, hãy chạy lệnh sau trong terminal:

```bash
pip install streamlit plotly pandas
```

*(Lưu ý: Các thư viện này đã được cài đặt trong quá trình triển khai dự án).*

## 3. Cách khởi chạy (Run)
Để mở Dashboard, hãy mở Terminal tại thư mục gốc của dự án (`/Users/dangminhtam/Đồán_PTDL/`) và chạy lệnh:

```bash
streamlit run src/web_dashboard.py
```

Sau khi chạy lệnh, trình duyệt sẽ tự động mở ra địa chỉ (thường là http://localhost:8501).

## 4. Hướng dẫn sử dụng giao diện

### 4.1. Thanh điều khiển (Sidebar - Bên trái)
- **Select Prediction Model**: Chọn mô hình dự báo bạn muốn xem trên biểu đồ (ví dụ: `BiLSTM`, `LinearRegression`).
- **Days to Visualize**: Kéo thanh trượt để điều chỉnh khoảng thời gian hiển thị trên biểu đồ (ví dụ: xem 30 ngày gần nhất hoặc toàn bộ dữ liệu).

### 4.2. Các chỉ số chính (KPI Metrics - Hàng trên cùng)
- **Date**: Ngày của dữ liệu mới nhất.
- **Current Price**: Giá đóng cửa thực tế gần nhất.
- **Daily Change**: Mức thay đổi giá so với ngày hôm trước (kèm % thay đổi).
- **Model Signal**: Tín hiệu gợi ý từ mô hình:
    - 🟢 **BUY (Undervalued)**: Nếu giá dự báo tăng > 1% so với hiện tại.
    - 🔴 **SELL (Overvalued)**: Nếu giá dự báo giảm > 1% so với hiện tại.
    - ⚪ **HOLD**: Nếu biến động dự báo nhỏ.

### 4.3. Biểu đồ Tương tác (Main Chart - Trung tâm)
- Biểu đồ đường so sánh **Giá Thực tế (Màu đen)** và **Giá Dự báo (Màu xanh đứt nét)**.
- Bạn có thể:
    - Di chuột vào đường để xem giá trị chi tiết.
    - Kéo thả để phóng to (Zoom) một vùng cụ thể.
    - Nhấp đúp để reset lại view ban đầu.

### 4.4. Phân tích chi tiết (Bottom Section)
- **Model Performance**: Bảng so sánh các chỉ số RMSE, MAE, R2 của các mô hình. Mô hình tốt nhất (RMSE thấp nhất) được tô sáng.
- **Feature Importance**: Hiển thị biểu đồ các yếu tố ảnh hưởng nhất (nếu có dữ liệu từ XGBoost).
- **Investment Recommendations**: Bấm vào để mở rộng xem chiến lược đầu tư và quản trị rủi ro được đề xuất.

---

## 5. Khắc phục sự cố thường gặp
- **Lỗi `ModuleNotFoundError`**: Kiểm tra lại việc cài đặt thư viện (`pip install ...`).
- **Lỗi `FileNotFoundError`**: Đảm bảo bạn đã chạy `python3 main.py` ít nhất một lần để tạo ra các file dữ liệu (`metrics.csv`, `predictions.csv`) trong thư mục `results/`.
- **Cổng 8501 bị bận**: Streamlit sẽ tự động chuyển sang cổng khác (8502, 8503...), hãy nhìn vào terminal để lấy địa chỉ đúng.
