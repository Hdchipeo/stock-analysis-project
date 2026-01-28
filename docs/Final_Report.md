# BÁO CÁO PHÂN TÍCH VÀ DỰ BÁO GIÁ CỔ PHIẾU (Mã: FPT.VN)

---

## 1. Dữ liệu và Phương pháp (Methodology)

### 1.1. Thu thập dữ liệu
Dữ liệu lịch sử của FPT Corp. (FPT.VN) được thu thập trong giai đoạn 5 năm thông qua thư viện `yfinance`. Bộ dữ liệu bao gồm các phiên giao dịch với các trường thông tin: Open, High, Low, Close, và Volume.

### 1.2. Tiền xử lý dữ liệu (Preprocessing)
Quá trình tiền xử lý đóng vai trò then chốt trong việc đảm bảo chất lượng dữ liệu đầu vào cho các mô hình.

1.  **Làm sạch dữ liệu**: Sử dụng phương pháp nội suy tuyến tính (Linear Interpolation) để xử lý các giá trị khuyết thiếu, đảm bảo tính liên tục của chuỗi thời gian.
2.  **Phát hiện và xử lý ngoại lai (Outlier Detection)**:
    -   Sử dụng thuật toán **Isolation Forest** để nhận diện các điểm dữ liệu bất thường (Anomalies).
    -   Kết quả cho thấy các điểm ngoại lai thường xuất hiện trong các giai đoạn biến động mạnh của thị trường.

![Outlier Detection](../results/figures/outliers.png)
*Hình 1: Các điểm ngoại lai (màu đỏ) được phát hiện trên biểu đồ giá Close.*

3.  **Kỹ thuật đặc trưng (Feature Engineering)**:
    Để tăng cường khả năng học của mô hình, các biến phái sinh đã được tạo ra:
    -   **Log Returns**: $ln(P_t / P_{t-1})$. Biến này giúp ổn định phương sai và xấp xỉ phân phối chuẩn tốt hơn giá tuyệt đối.
    -   **Chỉ báo kỹ thuật**:
        -   **RSI (14)**: Chỉ số sức mạnh tương đối.
        -   **MACD**: Đường trung bình động hội tụ phân kỳ.
        -   **SMA (7, 30)**: Đường trung bình động đơn giản ngắn hạn và trung hạn.
    -   **Biến trễ (Lag Features)**: Giá trị của các bước thời gian trước ($t-1, t-2, t-3$) được đưa vào làm đầu vào để mô hình nắm bắt tính tự hồi quy.

---

## 2. Phân tích Khám phá Dữ liệu (EDA)

Phân tích khám phá giúp hiểu sâu sắc về cấu trúc và động lực của chuỗi dữ liệu.

### 2.1. Phân tích Xu hướng (Trend Analysis)
Biểu đồ Candlestick kết hợp với đường MA30 và Volume cho thấy xu hướng tăng trưởng dài hạn của FPT.VN, xen kẽ với các đợt điều chỉnh ngắn hạn.

![Trend Analysis](../results/figures/trend_analysis.png)
*Hình 2: Xu hướng giá FPT.VN với đường MA30 và Khối lượng giao dịch.*

### 2.2. Phân tích Phân phối (Distribution Analysis)
Biểu đồ Histogram và KDE của Log Returns cho thấy phân phối có dạng hình chuông (Bell curve) nhưng sở hữu đặc điểm "Đuôi béo" (Fat Tails) với chỉ số Kurtosis cao. Điều này hàm ý xác suất xảy ra các biến động giá cực đoan cao hơn so với phân phối chuẩn lý thuyết.

![Distribution Analysis](../results/figures/distribution_analysis.png)
*Hình 3: Phân phối của Log Returns so với Phân phối chuẩn.*

### 2.3. Phân tích Tương quan (Correlation Analysis)
Ma trận tương quan (Heatmap) chỉ ra rằng:
-   Các biến giá và đường trung bình động (SMA) có tương quan dương rất mạnh ($>0.99$).
-   Chỉ báo kỹ thuật như RSI và MACD có tương quan vừa phải.
-   Volume có tương quan thấp với giá, cho thấy khối lượng giao dịch là một nguồn thông tin độc lập.

![Correlation Heatmap](../results/figures/correlation_heatmap.png)
*Hình 4: Ma trận tương quan giữa các biến.*

### 2.4. Phân tích Mùa vụ (Seasonality Analysis)
Phân tích biến động Volume theo Tháng và Thứ trong tuần giúp nhận diện các mẫu hình lặp lại theo chu kỳ.

![Seasonality Analysis](../results/figures/seasonality_analysis.png)
*Hình 5: Phân phối Volume theo Tháng và Thứ trong tuần.*

---

## 3. Kết quả Mô hình hóa (Modeling Results)

Nghiên cứu áp dụng hai phương pháp tiếp cận chính: Thống kê cổ điển (Linear Regression) và Học máy (XGBoost).

### 3.1. So sánh Hiệu suất (Model Comparison)
Các mô hình được đánh giá trên tập kiểm tra (Test set - 20% dữ liệu cuối). Kết quả định lượng như sau:

| Mô hình             | RMSE       | MAE        | $R^2$ Score | MAPE    |
|:--------------------|-----------:|:-----------|:------------|:--------|
| **Linear Regression** | **0.0054** | **0.0042** | **0.9953**  | **0.62%** |
| XGBoost             | 0.0152     | 0.0114     | 0.9627      | 1.66%   |

Biểu đồ so sánh trực quan trên 100 phiên giao dịch gần nhất cho thấy sự bám sát của đường dự báo Linear Regression so với giá thực tế.

![Model Comparison](../results/figures/model_comparison.png)
*Hình 6: So sánh giá Thực tế và Dự báo của các mô hình (Zoom 100 ngày).*

### 3.2. Phân tích Tầm quan trọng của Đặc trưng (Feature Importance)
Mô hình XGBoost cung cấp cái nhìn sâu sắc về mức độ đóng góp của từng biến số.
-   **Biến quan trọng nhất**: Các biến trễ (Lag 1, Lag 2) và SMA giữ vị trí thống trị, khẳng định tính tự hồi quy mạnh của chuỗi giá.
-   **Vai trò của Volume**: Volume đóng vai trò hỗ trợ nhưng không phải là yếu tố dẫn dắt chính trong việc dự báo giá đóng cửa trực tiếp.

![Feature Importance](../results/figures/feature_importance.png)
*Hình 7: Xếp hạng tầm quan trọng của các đặc trưng đầu vào.*

---

## 4. Kết luận (Conclusion)

Nghiên cứu này đã hoàn thành việc xây dựng và kiểm chứng quy trình dự báo giá cổ phiếu FPT.VN. Kết quả thực nghiệm dẫn đến các kết luận chính sau:

1.  **Hiệu quả Mô hình**:
    -   **Linear Regression** thể hiện hiệu suất vượt trội bất ngờ trên tập dữ liệu này. Điều này gợi ý rằng xu hướng giá FPT.VN trong giai đoạn nghiên cứu có tính quán tính (momentum) rất lớn và ít có các cú sốc phi tuyến phức tạp.
    -   **Mô hình BiLSTM** hiện tại đã được tạm hoãn trong báo cáo cuối cùng để ưu tiên tốc độ triển khai và tối ưu hóa tài nguyên.

2.  **Đặc điểm Dữ liệu**:
    -   Phân tích EDA đã chỉ ra rủi ro "Đuôi béo" (Fat Tails), cảnh báo rằng việc sử dụng các mô hình giả định phân phối chuẩn hoàn toàn có thể đánh giá thấp rủi ro thị trường.
    -   Sự phụ thuộc mạnh mẽ vào dữ liệu quá khứ gần (Lag features) cho thấy thị trường có tính hiệu quả yếu.

3.  **Sản phẩm Ứng dụng**:
    -   Hệ thống đã được tích hợp thành một Dashboard tương tác (hình dưới), cho phép theo dõi tín hiệu và hiệu suất mô hình theo thời gian thực.

![Dashboard Overview](../results/figures/dashboard.png)
*Hình 8: Giao diện Dashboard phân tích và dự báo.*

---
*Báo cáo kết thúc.*
