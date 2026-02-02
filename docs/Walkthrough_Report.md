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

### 1. Phân tích Xu hướng

![Trend Analysis](../results/figures/trend_analysis.png)

> [!NOTE]
> FPT có xu hướng tăng mạnh trong giai đoạn 2021-2026, với các đợt điều chỉnh ngắn hạn.

### 2. Phân phối Log Returns

![Distribution Analysis](../results/figures/distribution_analysis.png)

> [!IMPORTANT]
> **Fat Tails (Đuôi béo)**: Phân phối Log Returns của FPT có đuôi dày hơn phân phối chuẩn, nghĩa là xác suất xảy ra biến động lớn cao hơn mong đợi.

### 3. Ma trận Tương quan

![Correlation Heatmap](../results/figures/correlation_heatmap.png)

### 4. Phân tích Mùa vụ

![Seasonality Analysis](../results/figures/seasonality_analysis.png)

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

### 3. Tầm quan trọng của Đặc trưng (Feature Importance)

![Feature Importance](../results/figures/feature_importance_returns.png)

**Top 5 Features quan trọng nhất:**
1. **Returns_Lag_1**: Momentum ngắn hạn
2. **Volatility_30**: Mức độ rủi ro/biến động
3. **RSI_14**: Chỉ báo overbought/oversold
4. **Returns_Lag_2**: Pattern 2 ngày
5. **Volume_Change_Lag_2**: Xác nhận xu hướng Volume

## Kết quả Backtesting

### So sánh Chiến lược

| Chỉ số | BiLSTM Strategy | Buy & Hold |
|:-------|----------------:|-----------:|
| **Lợi nhuận** | **+28.34%** ✓ | +18.90% |
| **Sharpe Ratio** | **1.35** ✓ | 0.89 |
| **Max Drawdown** | **-11.89%** ✓ | -18.45% |
| **Win Rate** | 57.1% | N/A |

![Backtesting Comparison](../results/figures/backtesting_comparison.png)

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
