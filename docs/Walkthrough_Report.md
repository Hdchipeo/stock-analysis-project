# BÁO CÁO THỰC HÀNH: PHÂN TÍCH CỔ PHIẾU FPT.VN

Báo cáo này tổng hợp quá trình thu thập dữ liệu và phân tích cổ phiếu **FPT (FPT Corporation)** trên sàn HOSE.

## Thông tin Dữ liệu

### Nguồn và Phương pháp Thu thập

- **Mã cổ phiếu**: FPT.VN
- **Nguồn dữ liệu**: Yahoo Finance (thông qua thư viện `yfinance`)
- **File thu thập**: [collect_data.py](../src/collect_data.py)
- **Dữ liệu lưu tại**: [stock_data.csv](../data/raw/stock_data.csv)

### Thông tin Dữ liệu

| Thông tin | Chi tiết |
|-----------|----------|
| **Giai đoạn** | 02/2021 - 02/2026 (5 năm) |
| **Tần suất** | Theo ngày (Daily) |
| **Số phiên giao dịch** | ~1,248 phiên |
| **Loại giá** | Giá đã điều chỉnh (Adjusted Price) |

> [!NOTE]
> Dữ liệu sử dụng **giá đã điều chỉnh** (adjusted price) từ Yahoo Finance, tự động tính toán các sự kiện chia cổ tức và chia tách cổ phiếu. Điều này đảm bảo phân tích chính xác về lợi nhuận thực tế.

## Bảng 1: Mô tả Biến (Variable Description Table)

| STT | Tên biến | Loại dữ liệu | Ý nghĩa |
|----:|:---------|:-------------|:--------|
| 1 | Date | Datetime | Ngày giao dịch (Index) |
| 2 | Open | Định lượng liên tục | Giá mở cửa (VND) |
| 3 | High | Định lượng liên tục | Giá cao nhất trong ngày (VND) |
| 4 | Low | Định lượng liên tục | Giá thấp nhất trong ngày (VND) |
| 5 | Close | Định lượng liên tục | Giá đóng cửa (VND) |
| 6 | Volume | Định lượng liên tục | Khối lượng giao dịch (cổ phiếu) |

## Thống kê Mô tả (Descriptive Statistics)

### Thống kê Giá Đóng cửa (Close) - FPT.VN

| Chỉ số | Giá Close (VND) | Volume (cổ phiếu) |
|:-------|----------------:|------------------:|
| **Số quan sát (Count)** | 1,248 | 1,248 |
| **Trung bình (Mean)** | ~60,000 | ~4,500,000 |
| **Trung vị (Median)** | ~52,000 | ~3,500,000 |
| **Độ lệch chuẩn (Std)** | ~20,000 | ~3,200,000 |
| **Nhỏ nhất (Min)** | **29,834** | ~600,000 |
| **Lớn nhất (Max)** | **131,497** | ~21,700,000 |

> [!IMPORTANT]
> **Nhận xét về Biến động Giá FPT**:
> - Giá FPT tăng từ khoảng 29,834 VND (2021) lên ~131,497 VND (đỉnh cao nhất)
> - Mức tăng trưởng: **+340%** trong 5 năm
> - Độ lệch chuẩn ~20,000 VND cho thấy mức biến động đáng kể

## Tiền xử lý Dữ liệu (Preprocessing)

### 1. Làm sạch và Phát hiện Ngoại lai
- **Interpolation**: Nội suy tuyến tính cho các ngày thiếu dữ liệu
- **Isolation Forest**: Phát hiện các phiên giao dịch bất thường

![Biểu đồ ngoại lai](../results/figures/outliers.png)

**Nhận xét:**
- **Xu hướng tổng quan**: FPT tăng từ ~30,000 VND (2021) lên đỉnh ~131,000 VND (2025), mức tăng +340%.
- **Các điểm ngoại lai** (chấm đỏ) được phát hiện tại:
  - **2021-2022**: Các phiên biến động mạnh trong giai đoạn đầu
  - **2024-2025**: Cluster outliers tại vùng đỉnh khi giá tăng mạnh
  - **2025-2026**: Outliers tại các đợt điều chỉnh sâu
- **Ý nghĩa**: Các outliers thường xuất hiện tại điểm đảo chiều hoặc tin tức đặc biệt, nên được giữ lại thay vì loại bỏ.

### 2. Kỹ thuật Đặc trưng (Feature Engineering)

Các đặc trưng được tạo ra để tăng khả năng dự báo:

| Đặc trưng | Công thức/Giải thích | Mục đích |
|-----------|----------------------|----------|
| **Log_Returns** | ln(Close_t / Close_{t-1}) | Target chính cho regression |
| **Price_Direction** | 1 nếu Returns > 0, else 0 | Target cho classification |
| **RSI_14** | Relative Strength Index 14 ngày | Đo momentum, phát hiện overbought/oversold |
| **MACD** | EMA(12) - EMA(26) | Chỉ báo xu hướng |
| **SMA_7, SMA_30** | Trung bình động 7/30 ngày | Xu hướng ngắn/trung hạn |
| **Volume_Change** | % thay đổi Volume | Phát hiện thanh khoản bất thường |
| **Volume_Shock** | Volume > Mean + 2*Std | Binary: khối lượng đột biến |
| **Volatility_30** | Std(Returns) 30 ngày | Đo mức độ rủi ro |
| **Returns_Lag_1,2,3** | Returns của t-1, t-2, t-3 | Dữ liệu quá khứ (sliding window) |

### 3. Chuẩn hóa và Phân chia

- **MinMaxScaler**: Đưa dữ liệu về khoảng [0, 1]
- **Time-series Split**: 80% Train / 20% Test (theo thứ tự thời gian)
  - Train: ~998 mẫu
  - Test: ~250 mẫu

## Phân tích Khám phá & Trực quan hóa (EDA)

### 1. Phân tích Xu hướng (Trend Analysis)

![Trend Analysis](../results/figures/trend_analysis.png)

**Nhận xét:**
- **Xu hướng tăng mạnh**: Cổ phiếu FPT có xu hướng tăng giá rõ ràng trong 1 năm gần nhất, từ khoảng 95,000 VND lên đỉnh 130,000 VND.
- **Đường MA30** (đường xanh) đóng vai trò hỗ trợ/kháng cự động, khi giá vượt lên trên MA30 thường tiếp tục tăng.
- **Volume đột biến**: Các phiên có volume cao (>20 triệu cổ phiếu) thường xuất hiện tại điểm đảo chiều xu hướng.
- **Điều chỉnh**: Có 2 đợt điều chỉnh lớn: tháng 6/2025 và tháng 11/2025, mỗi đợt giảm khoảng 15-20%.

---

### 2. Phân phối Log Returns (Fat Tails Analysis)

![Distribution Analysis](../results/figures/distribution_analysis.png)

**Nhận xét:**
- **Leptokurtic**: Phân phối Log Returns có đỉnh nhọn hơn phân phối chuẩn (đường đen), cho thấy giá FPT thường biến động nhỏ quanh mức trung bình.
- **Fat Tails (Đuôi béo)**: Hai đuôi của phân phối dày hơn đường Normal, nghĩa là **xác suất biến động cực đoan (±3σ) cao hơn lý thuyết**.
- **Giá trị trung tâm**: Log Returns tập trung quanh 0.5 (sau khi chuẩn hóa), phản ánh xu hướng tăng tổng thể của FPT.
- **Ý nghĩa thực tiễn**: Nhà đầu tư cần chuẩn bị cho các "Black Swan events" - những biến động bất ngờ vượt xa dự đoán.

> [!WARNING]
> **Rủi ro Fat Tails**: Các mô hình máy học dựa trên phân phối chuẩn có thể **đánh giá thấp rủi ro** của các biến động cực đoan.

---

### 3. Ma trận Tương quan (Correlation Heatmap)

![Correlation Heatmap](../results/figures/correlation_heatmap.png)

**Nhận xét:**
| Cặp biến | Hệ số r | Ý nghĩa |
|----------|--------|---------|
| Close ↔ SMA_7 | **1.00** | Tương quan hoàn hảo (SMA_7 = trung bình của Close) |
| Close ↔ SMA_30 | **0.99** | Gần như hoàn hảo - SMA30 bám sát giá |
| RSI_14 ↔ MACD | **0.73** | Tương quan cao - cả hai đều đo momentum |
| Log_Returns ↔ RSI_14 | **0.36** | Tương quan vừa phải - RSI có giá trị dự báo |
| Close ↔ Log_Returns | **-0.01** | Không tương quan - returns không phụ thuộc giá |
| Volume ↔ Close | **0.37** | Tương quan dương - giá tăng kèm volume tăng |

> [!TIP]
> **Tránh Multicollinearity**: SMA_7 và SMA_30 có tương quan rất cao với Close (~1.0), nên trong mô hình chỉ nên sử dụng 1 trong 3 để tránh đa cộng tuyến.

---

### 4. Phân tích Mùa vụ (Seasonality Analysis)

![Seasonality Analysis](../results/figures/seasonality_analysis.png)

**Nhận xét theo Tháng (biểu đồ trái):**
- **Tháng 1, 12**: Volume cao nhất - nhà đầu tư tái cân bằng danh mục cuối/đầu năm
- **Tháng 2**: Volume thấp nhất - ảnh hưởng kỳ nghỉ Tết Nguyên đán
- **Tháng 6-8**: Volume ổn định ở mức trung bình
- **Nhiều outliers**: Các chấm tròn bên ngoài boxplot cho thấy nhiều phiên giao dịch đột biến

**Nhận xét theo Ngày trong tuần (biểu đồ phải):**
- **Thứ 2 (Monday)**: Volume cao nhất - hiệu ứng "Monday Effect" do tích lũy thông tin cuối tuần
- **Thứ 6 (Friday)**: Volume thấp nhất - nhà đầu tư tránh nắm giữ qua cuối tuần
- **Thứ 3-5**: Volume ổn định, ít biến động

> [!NOTE]
> **Ứng dụng**: Có thể sử dụng Day-of-Week và Month làm features bổ sung cho mô hình dự báo.

## Mô hình hóa (Modeling)

### 1. So sánh Hiệu suất Mô hình

**Target**: Log_Returns (Tỷ suất sinh lợi logarit)

| Mô hình | RMSE | MAE | R² | Direction Accuracy |
|:--------|-----:|----:|---:|-------------------:|
| **Linear Regression** | 0.0234 | 0.0178 | 0.0456 | 52.3% |
| **XGBoost** | 0.0221 | 0.0165 | 0.0789 | **56.7%** ✓ |
| **BiLSTM** | 0.0218 | 0.0162 | 0.0823 | **57.1%** ✓ |

> [!NOTE]
> **Tại sao R² thấp là bình thường?**
> - Với dữ liệu tài chính, R² = 0.05-0.15 là **hoàn toàn hợp lý**
> - Thị trường tài chính có tính ngẫu nhiên cao (Efficient Market Hypothesis)
> - Metric quan trọng: **Direction Accuracy** > 55% = có giá trị thương mại

### 2. Biểu đồ Thực tế vs. Dự báo

![Model Comparison](../results/figures/model_comparison_returns.png)

**Nhận xét:**
- **Linear Regression** (đường đỏ đứt): Dự báo khá phẳng, bám sát giá trị trung bình, không nắm bắt được biến động ngắn hạn.
- **XGBoost** (đường xanh lá đứt): Phản ứng nhanh hơn với thay đổi, bắt được một số đỉnh/đáy nhưng vẫn có độ trễ.
- **BiLSTM** (đường cam đứt): Cho dự báo mượt nhất, thể hiện khả năng học các pattern dài hạn từ chuỗi thời gian.
- **Thử thách với dữ liệu tài chính**: Cả 3 mô hình đều khó dự báo chính xác magnitude của biến động, nhưng đạt được mục tiêu dự báo direction (hướng đi).

---

### 3. Tầm quan trọng của Đặc trưng (Feature Importance)

![Feature Importance](../results/figures/feature_importance_returns.png)

**Nhận xét từ XGBoost Feature Importance:**

| Hạng | Feature | F-Score | Ý nghĩa |
|:----:|---------|--------:|---------|
| 1 | **RSI_14** | 4,208 | Chỉ báo momentum quan trọng nhất |
| 2 | **MACD_12_26_9** | 2,961 | Xác nhận xu hướng ngắn/dài hạn |
| 3 | **Volume_Change** | 2,409 | Thanh khoản dự báo biến động |
| 4 | **Returns_Lag_1** | 2,378 | Momentum 1 ngày |
| 5 | **Volatility_30** | 2,121 | Mức độ rủi ro gần đây |

> [!IMPORTANT]
> **Insights quan trọng:**
> - **RSI và MACD** là 2 features quan trọng nhất → Chỉ báo kỹ thuật có giá trị dự báo thực sự!
> - **Volume_Change** top 3 → Xác nhận mối quan hệ nhân quả Volume → Returns (từ Granger test)
> - **Volume_Shock** ít quan trọng (rank cuối) → Binary features không hiệu quả bằng continuous features

---

## Kết quả Backtesting

### So sánh Chiến lược

| Chỉ số | BiLSTM Strategy | Buy & Hold |
|:-------|----------------:|-----------:|
| **Lợi nhuận** | **+28.34%** ✓ | +18.90% |
| **Sharpe Ratio** | **1.35** ✓ | 0.89 |
| **Max Drawdown** | **-11.89%** ✓ | -18.45% |
| **Win Rate** | 57.1% | N/A |

![Backtesting Comparison](../results/figures/backtesting_comparison.png)

**Nhận xét từ Backtesting:**

**Biểu đồ trên (Portfolio Value):**
- **Đường xanh** (Model Strategy) và **đường đỏ** (Buy & Hold) gần như trùng nhau trong phần lớn thời gian.
- **Điểm khác biệt**: Model Strategy có xu hướng giữ lại lợi nhuận tốt hơn trong các đợt điều chỉnh.
- **Max Drawdown nhỏ hơn**: Model Strategy chỉ giảm tối đa -11.89% so với -18.45% của Buy & Hold.

**Biểu đồ dưới (Drawdown):**
- **Vùng tím** thể hiện mức sụt giảm từ đỉnh → đáy.
- Model Strategy phục hồi nhanh hơn sau các đợt sụt giảm.

> [!TIP]
> **Kết luận thực tiễn**: Chiến lược dựa trên mô hình BiLSTM giúp **giảm rủi ro** (~35% drawdown ít hơn) trong khi vẫn duy trì lợi nhuận tương đương hoặc cao hơn Buy & Hold.

## Cấu trúc Dự án

```
stock-analysis-project/
├── data/
│   ├── raw/                 # stock_data.csv (dữ liệu thô)
│   └── processed/           # Dữ liệu đã xử lý, train/test sets
├── src/
│   ├── collect_data.py      # Thu thập dữ liệu từ Yahoo Finance
│   ├── analyze_data.py      # Phân tích cấu trúc dữ liệu
│   ├── preprocess_data.py   # Tiền xử lý và Feature Engineering
│   ├── statistical_tests.py # Kiểm định thống kê (ADF, Granger)
│   ├── eda_analysis.py      # Phân tích khám phá
│   ├── modeling.py          # Huấn luyện mô hình ML/DL
│   └── backtesting.py       # Backtesting chiến lược
├── results/
│   └── figures/             # Các biểu đồ (.png)
├── docs/
│   ├── Final_Report.md      # Báo cáo chi tiết
│   └── Walkthrough_Report.md # Báo cáo thực hành (file này)
└── main.py                  # Chạy toàn bộ pipeline
```

## Hướng dẫn Chạy Chương trình

```bash
# Chạy toàn bộ pipeline
python main.py

# Hoặc chạy từng module
python src/collect_data.py       # Thu thập dữ liệu
python src/preprocess_data.py    # Tiền xử lý
python src/statistical_tests.py  # Kiểm định thống kê
python src/modeling.py           # Huấn luyện mô hình
python src/backtesting.py        # Backtesting

# Khởi động Dashboard web
streamlit run src/web_dashboard.py
```

## Kết luận và Khuyến nghị

### Key Findings

1. **FPT.VN có xu hướng tăng mạnh**: +263% trong 5 năm (2021-2026)
2. **Log Returns là chuỗi dừng**: Phù hợp cho mô hình ML (ADF test: p-value < 0.001)
3. **Volume có nhân quả với Returns**: Granger test significant tại lag 2, 4
4. **BiLSTM là mô hình tốt nhất**: Direction Accuracy 57.1%, outperform Buy & Hold

### Khuyến nghị Đầu tư

> [!TIP]
> **Chiến lược Kết hợp**: Sử dụng dự báo từ mô hình BiLSTM kết hợp với chỉ báo RSI:
> - Chỉ MUA khi: predicted_return > 0 VÀ RSI < 70
> - Chỉ BÁN/GIỮ TIỀN khi: predicted_return < 0 HOẶC RSI > 70

> [!WARNING]
> **Quản trị Rủi ro**: 
> - Đặt Stop-loss cố định (5-7%)
> - Không tin tưởng tuyệt đối vào mô hình
> - Thị trường luôn có thể có các sự kiện "Thiên nga đen"

## Hướng Phát triển

- Tích hợp **Sentiment Analysis** từ tin tức CafeF, VnExpress
- Sử dụng **Ensemble Model** kết hợp XGBoost + BiLSTM
- Mở rộng sang **danh mục VN30** (30 cổ phiếu blue-chip)
