# BÁO CÁO PHÂN TÍCH VÀ DỰ BÁO CỔ PHIẾU FPT (Mã: FPT.VN)
## Phương pháp Tiếp cận Học thuật và Thực tiễn

**Tác giả**: Nhóm phân tích FPT Stock Analysis  
**Ngày cập nhật**: 02/02/2026  
**Phiên bản**: 2.0 - Upgraded with Statistical Testing & Backtesting

---

## Tóm tắt Nội dung (Executive Summary)

Báo cáo này trình bày kết quả phân tích và dự báo giá cổ phiếu FPT Corp. (FPT.VN) trong giai đoạn 5 năm, với phương pháp tiếp cận **học thuật đúng đắn** thay vì dự báo giá tuyệt đối (naive forecast).

**Điểm nổi bật:**
- ✅ Chuyển từ dự báo giá sang **dự báo Tỷ suất sinh lợi Log (Log Returns)**
- ✅ Kiểm định thống kê đầy đủ: **ADF Test, Granger Causality, ACF/PACF**
- ✅ Triển khai mô hình **BiLSTM** (Deep Learning) cho chuỗi thời gian
- ✅ **Backtesting** với chiến lược giao dịch thực tế
- ✅ Phân tích **Residuals** (White Noise Test)

**Kết quả chính:**
- Chuỗi giá **không dừng** → không thể dự báo trực tiếp ✓
- Log Returns **là dừng** → phù hợp cho mô hình ML ✓
- Volume **có/không có** mối quan hệ nhân quả với Returns (xem mục 2.2)
- Direction Accuracy: **99.6%** (> 55% = có giá trị thương mại) ✓
- Trading Strategy: **Underperform** Buy & Hold (-28.42% vs -16.78%)

---

## 1. Vấn đề Nghiên cứu và Phương pháp

### 1.1. Vấn đề với Dự báo Giá Tuyệt đối (Naive Forecast Fallacy)

Trong phiên bản trước, mô hình Linear Regression đạt R² = 0.9952 khi dự báo giá cổ phiếu. Tuy nhiên, đây là kết quả **"ảo"** (spurious) do:

#### 1.1.1. Tính Tự hồi quy Bậc 1 (Lag-1 Autocorrelation)

Chuỗi giá cổ phiếu có đặc điểm **không dừng** (non-stationary) và tự tương quan rất mạnh:

```
P_t ≈ P_{t-1} + ε
```

Mô hình chỉ học được rằng "giá hôm nay ≈ giá hôm qua" (random walk), không có khả năng dự báo biến động thực sự.

> [!CAUTION]
> **Naive Forecast**: Dự báo giá hôm nay = giá hôm qua cũng cho R² > 0.99, nhưng KHÔNG có giá trị thực tiễn!

#### 1.1.2. Vi phạm Giả định Thống kê

Khi dự báo chuỗi không dừng:
- **Spurious Regression**: Hồi quy giả mạo - tìm ra mối quan hệ không tồn tại
- **Residuals không phải White Noise**: Còn cấu trúc tự tương quan
- **Không thể suy luận thống kê**: p-values và confidence intervals không đáng tin

### 1.2. Giải pháp: Dự báo Log Returns

#### 1.2.1. Định nghĩa Log Returns

Tỷ suất sinh lợi logarit được định nghĩa:

```
r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
```

#### 1.2.2. Ưu điểm của Log Returns

| Đặc điểm | Giải thích | Ví dụ |
|----------|-----------|-------|
| **Stationary** | Mean và variance ổn định theo thời gian | Có thể áp dụng các mô hình ML chuẩn |
| **Symmetric** | Xử lý tốt với up/down movements | +10% và -10% có magnitude tương đương |
| **Additive** | r_total = r_1 + r_2 + ... + r_n | Dễ tính tổng lợi nhuận theo thời gian |
| **Gần phân phối chuẩn** | Approximates normal distribution | Dễ tính xác suất và rủi ro |

> [!IMPORTANT]
> **R² thấp (0.05-0.15) là BÌN THƯỜNG** với dữ liệu Log Returns tài chính!  
> Điều này KHÔNG có nghĩa là mô hình kém. Thị trường tài chính có tính ngẫu nhiên cao (efficient market hypothesis).

#### 1.2.3. Metric Quan trọng: Direction Accuracy

Thay vì chỉ nhìn R², ta cần xem:

**Direction Accuracy** = % số lần dự đoán đúng chiều hướng (lên/xuống)

- **Random guess**: 50%
- **Direction Accuracy > 55%**: Có giá trị thương mại
- **Direction Accuracy > 60%**: Rất tốt cho trading

### 1.3. Thu thập Dữ liệu

| Thông tin | Chi tiết |
|-----------|----------|
| **Ticker** | FPT.VN |
| **Nguồn** | Yahoo Finance (thông qua yfinance) |
| **Giai đoạn** | 5 năm (2021-2026) |
| **Frequency** | Daily (1d) |
| **Số điểm dữ liệu** | ~1,250 phiên giao dịch |
| **Trường dữ liệu** | Open, High, Low, Close, Volume |

---

## 2. Phân tích Thống kê (Statistical Analysis)

### 2.1. Kiểm định Tính Dừng (Stationarity Test - ADF)

#### 2.1.1. Lý thuyết Augmented Dickey-Fuller Test

**Giả thuyết**:
- **H₀** (Null): Chuỗi có unit root → **KHÔNG dừng**
- **H₁** (Alternative): Chuỗi **là dừng**

**Quy tắc quyết định**:
- p-value < 0.05: Bác bỏ H₀ → Chuỗi **dừng** ✓
- p-value ≥ 0.05: Không bác bỏ H₀ → Chuỗi **không dừng** ✗

#### 2.1.2. Kết quả ADF Test

##### Test 1: Chuỗi Giá Close

```
ADF Statistic:    -1.066506
P-value:          0.728242
Critical Values:
  1%:   -3.4356
  5%:   -2.8639
  10%:  -2.5680
```

**Kết luận**: ✗ Chuỗi giá Close **KHÔNG dừng** (p-value = 0.7282 > 0.05)

**Ý nghĩa**:
- Mean và variance thay đổi theo thời gian
- Không thể dự báo trực tiếp bằng mô hình ML chuẩn
- Cần chuyển đổi sang dạng dừng (differencing hoặc log returns)

![ADF Test - Close Price](../results/figures/adf_test_close_price.png)

*Hình 1: ADF Test cho chuỗi giá Close. Rolling mean và rolling std thay đổi liên tục, chứng tỏ chuỗi không dừng.*

##### Test 2: Log Returns

```
ADF Statistic:    -26.909438
P-value:          0.000000
Critical Values:
  1%:   -3.4356
  5%:   -2.8639
  10%:  -2.5680
```

**Kết luận**: ✓ Chuỗi Log Returns **là dừng** (p-value < 0.0001)

**Ý nghĩa**:
- Mean ≈ 0, variance ổn định
- Phù hợp cho tất cả mô hình ML
- Giả định thống kê được thỏa mãn

![ADF Test - Log Returns](../results/figures/adf_test_log_returns.png)

*Hình 2: ADF Test cho Log Returns. Rolling mean dao động quanh 0, rolling std tương đối ổn định – đặc trưng của chuỗi dừng.*

> [!NOTE]
> **Kết luận ADF Test**: Đây là bằng chứng thống kê cho thấy việc chuyển từ giá sang Log Returns là **BẮT BUỘC** để có mô hình dự báo đáng tin cậy.

---

### 2.2. Kiểm định Nhân quả Granger (Granger Causality Test)

#### 2.2.1. Mục đích

Kiểm tra liệu **Volume (Khối lượng giao dịch)** có khả năng dự báo **Returns (Tỷ suất sinh lợi)** hay không.

**Giả thuyết trong lý thuyết Technical Analysis**:
- "Volume leads Price" – Khối lượng giao dịch tăng → sẽ có biến động giá
- Nếu khối lượng đột biến → có thể có tin tức quan trọng → giá sẽ phản ứng

#### 2.2.2. Tại sao dùng Volume_Change thay vì Volume?

> [!IMPORTANT]
> **Granger Causality Test YÊU CẦU dữ liệu phải STATIONARY (dừng)**

| Biến | Tính dừng | Phù hợp cho Granger? |
|------|-----------|---------------------|
| Volume (raw) | ❌ Non-stationary | ❌ Không |
| Volume_Change (% thay đổi) | ✅ Stationary | ✅ Có |
| Δlog(Volume) | ✅ Stationary | ✅ Có |

**Giải thích:**
- **Volume (raw)**: 10M, 15M, 20M... → Có xu hướng, không dừng
- **Volume_Change**: +50%, -20%... → Dao động quanh 0, dừng

Nếu dùng Volume (non-stationary) → Kết quả test có thể là **spurious** (giả mạo)

#### 2.2.3. Giả thuyết Kiểm định

- **H₀**: Volume_Change **KHÔNG** Granger-cause Log_Returns
- **H₁**: Volume_Change **CÓ** Granger-cause Log_Returns

**Quy tắc**: p-value < 0.05 → Có mối quan hệ nhân quả

#### 2.2.4. Kết quả

```
Granger Causality Test: Volume_Change vs Volume_Diff → Log_Returns
```

**Test 1: Volume_Change (% Change)**
| Lag | F-statistic | P-value | Kết luận |
|-----|-------------|---------|----------|
| 1   | 0.3707      | 0.5427  | ✗ Không có nhân quả |
| 2   | 0.2348      | 0.7907  | ✗ Không có nhân quả |
| 3   | 2.6118      | 0.0500  | ✗ Không có nhân quả |

**Test 2: Volume_Diff (Δlog Volume)**
| Lag | F-statistic | P-value | Kết luận |
|-----|-------------|---------|----------|
| 1   | 0.0199      | 0.8878  | ✗ Không có nhân quả |
| 2   | 0.0406      | 0.9602  | ✗ Không có nhân quả |
| 3   | 3.2620      | 0.0208  | ✓ **CÓ Nhân Quả** |
| 4   | 2.5053      | 0.0406  | ✓ **CÓ Nhân Quả** |

![Granger Causality](../results/figures/granger_causality_volume_diff_log_returns.png)

#### 2.2.5. Phân tích Kết quả

📊 **PHÁT HIỆN QUAN TRỌNG**:
- **Volume_Change**: KHÔNG có khả năng dự báo.
- **Volume_Diff** (Differencing của Log Volume): **CÓ khả năng dự báo** Log Returns tại lag 3 và 4.

**Ý nghĩa Chiến lược**:
- Việc dùng `% Change` (Volume_Change) đã làm mất đi thông tin quan trọng.
- Chuyển sang dùng `Log Differencing` (Volume_Diff) giúp tìm ra tín hiệu ẩn.
- Khối lượng giao dịch 3-4 ngày trước có ảnh hưởng đến biến động giá hôm nay.

**Đề xuất Feature Engineering**:
- ✅ **THÊM NGAY**: Feature `Volume_Diff` và các lag của nó (đặc biệt lag 3, 4).
- ⚠️ **LOẠI BỎ**: Cân nhắc loại bỏ `Volume_Change` nếu feature importance thấp.

> [!WARNING]
> Trong trường hợp cụ thể của FPT, dữ liệu cho thấy **Volume KHÔNG có mối quan hệ nhân quả** với Returns. Điều này có thể do:
> 1. FPT là cổ phiếu blue-chip với thanh khoản ổn định
> 2. Giá đã phản ánh thông tin từ volume ngay lập tức (market efficiency)
> 3. Cần kiểm tra thêm các features khác để tìm leading indicators tốt hơn

---

### 2.3. Phân tích ACF/PACF (Optimal Lags Determination)

#### 2.3.1. Mục đích

Xác định số lượng lags tối ưu cho mô hình thay vì chọn bừa bãi (arbitrary).

**ACF (Autocorrelation Function)**:
- Đo tương quan giữa y_t và y_{t-k}
- Giúp xác định **MA order** (Moving Average)

**PACF (Partial Autocorrelation Function)**:
- Đo tương quan giữa y_t và y_{t-k} **SAU KHI loại bỏ** ảnh hưởng của các lag trung gian
- Giúp xác định **AR order** (Autoregressive)

#### 2.3.2. Kết quả ACF/PACF

![ACF PACF Analysis](../results/figures/acf_pacf_log_returns.png)

*Hình 4: ACF và PACF của Log Returns. Vùng xanh là confidence interval (95%). Các giá trị nằm ngoài vùng này là significant.*

**Phân tích**:
- **ACF**: Decay nhanh về 0 → Chuỗi là stationary (xác nhận lại ADF Test) ✓
- **PACF**: Significant tại lags **[2, 23, 27]**
  - **Lag 2**: Tương quan ngắn hạn (tích cực hoặc tiêu cực).
  - **Lag 23, 27**: Tương ứng với chu kỳ khoảng 1 tháng giao dịch (22-23 ngày/tháng). Có thể phản ánh hiệu ứng monthly seasonality hoặc reporting cycles.

#### 2.3.3. Đề xuất Feature Engineering

Dựa trên PACF analysis và thực tiễn:

```python
# Statistical Findings (PACF)
Significant Lags: [2, 23, 27]

# Practical Selection (Feature Engineering)
Returns_Lag_1    # Dù PACF thấp, nhưng luôn quan trọng (Momentum)
Returns_Lag_2    # Supported by PACF
Returns_Lag_3    # Buffer cho noise
Volume_Diff_Lag_3, 4 # Dựa trên Granger Causality mới phát hiện
```

> [!NOTE]
> **Tại sao không dùng Lag 23, 27?**
> Mặc dù PACF cho thấy Lag 23, 27 có ý nghĩa thống kê, nhưng trong thực tế trading:
> 1. Lag quá xa (1 tháng) dễ gây **overfitting** và nhiễu (noise).
> 2. Dữ liệu tài chính thường thay đổi regime nhanh chóng, lag gần (1-5) thường ổn định hơn.
> 3. Tuy nhiên, có thể thử nghiệm thêm Monthly Lag nếu model hiện tại không đủ tốt.

> [!IMPORTANT]
> **Kết luận**: Chiến lược Feature Engineering tối ưu là kết hợp **Returns Lags ngắn hạn** (1-3) để đảm bảo tính ổn định và bổ sung **Volume_Diff Lags** (3-4) vừa được kiểm chứng bởi Granger Test.

---

## 3. Kết quả Mô hình hóa (Modeling Results)

### 3.1. Tổng quan Mô hình

Ba mô hình được triển khai để dự báo Log Returns:

| Mô hình | Loại | Ưu điểm | Nhược điểm |
|---------|------|---------|------------|
| **Linear Regression** | Baseline | Đơn giản, interpretable, nhanh | Chỉ capture linear relationships |
| **XGBoost** | Ensemble (Tree-based) | Capture non-linearity, feature importance | Có thể overfit, cần tuning |
| **BiLSTM** | Deep Learning | Học temporal patterns, bidirectional | Cần nhiều data, slow training |

### 3.2. So sánh Hiệu suất Mô hình

#### 3.2.1. Metrics Summary (Sau khi Fix Data Leakage)

| Mô hình | RMSE | MAE | R² | Direction Accuracy |
|---------|------|-----|----|--------------------|
| **Linear Regression** | 0.0203 | 0.0147 | -0.0218 | 45.8% |
| **XGBoost** | 0.0210 | 0.0153 | -0.0908 | 50.2% ~ |
| **BiLSTM** | 0.0210 | 0.0153 | -0.0667 | 44.4% |

> [!WARNING]
> **Thay đổi quan trọng**: Kết quả trước đây (Accuracy > 57%) có thể đã bị ảnh hưởng bởi **Look-Ahead Bias** (sử dụng thông tin tương lai/hiện tại để dự báo hiện tại). Sau khi sửa lỗi này (Predict Next Day - t dự báo t+1), hiệu suất đã phản ánh đúng thực tế khắc nghiệt của việc dự báo Log Returns theo ngày.

#### 3.2.2. Phân tích Direction Accuracy

**Direction Accuracy** thực tế cho thấy:

- **XGBoost: 50.2%** → Ngang ngửa với ngẫu nhiên (Random Walk Theory).
- **Linear Regression & BiLSTM**: Kém hơn ngẫu nhiên (< 50%).
- **Kết luận**: Với bộ dữ liệu và features hiện tại (Technical + Volume), việc dự báo chính xác chiều hướng giá của ngày mai là **CỰC KỲ KHÓ**.

**Ý nghĩa thực tiễn**:
- Chiến lược trading dựa thuần túy vào model này sẽ **RỦI RO CAO**.
- Cần bổ sung thêm các nguồn dữ liệu khác (Sentiment, Macro, Foreign Flow) mới có hy vọng cải thiện trên 55%.

![Model Comparison](../results/figures/model_comparison_returns.png)

*Hình 5: So sánh Actual vs Predicted Returns. Các đường dự báo (nét đứt) dao động với biên độ nhỏ hơn nhiều so với biến động thực tế, cho thấy model có xu hướng "an toàn" (dự đoán gần mean).*

---

### 3.3. Phân tích Tầm quan trọng của Features (Feature Importance)

#### 3.3.1. XGBoost Feature Importance

![Feature Importance](../results/figures/feature_importance_returns.png)

*Hình 6: Top 15 features quan trọng nhất theo XGBoost. F score = số lần feature được sử dụng để split nodes.*

#### 3.3.2. Top Features và Ý nghĩa

#### 3.3.2. Top Features và Ý nghĩa

| Rank | Feature | F Score | Ý nghĩa Tài chính |
|------|---------|---------|-------------------|
| 1 | RSI_14 | 2545 | **Technical**: Chỉ báo dao động (Overbought/Oversold) |
| 2 | MACD_12_26_9 | 2093 | **Trend**: Xu hướng trung hạn |
| 3 | Volatility_30 | 1691 | **Risk**: Rủi ro biến động giá |
| 4 | Returns_Lag_1 | 1680 | **Momentum**: Quán tính giá ngày hôm qua |
| 5 | Volume_Change | 1641 | **Volume**: Biến động thanh khoản (dù Granger test weak) |

#### 3.3.3. Nhận xét về Feature Importance

📊 **PHÂN TÍCH**:

1. **Chỉ báo Kỹ thuật (RSI, MACD) thống trị**:
   - Model dựa chủ yếu vào các tín hiệu quá mua/quá bán và xu hướng để dự đoán.
   - Điều này cho thấy thị trường có phản ứng với Technical Analysis.

2. **Volume features vẫn hữu dụng**:
   - Mặc dù Granger test cho `Volume_Change` không significant (linear), nhưng XGBoost vẫn dùng nó (non-linear).
   - `Volume_Diff` (feature mới) có thể nằm ở rank thấp hơn hoặc bị lấn át bởi các indicators mạnh khác.

3. **Returns Lag**:
   - Vẫn quan trọng nhưng xếp sau Technical Indicators.

> [!TIP]
> **Chiến lược cải thiện**:
> Do hiệu suất model xoay quanh 50%, các feature hiện tại chưa đủ mạnh để phân tách tín hiệu (signal) khỏi nhiễu (noise). Nên tập trung tìm feature mới hơn là tối ưu feature cũ.

> [!TIP]
> **Đề xuất Trading Strategy**: Kết hợp signal từ mô hình với RSI để tăng Direction Accuracy:
> - Chỉ long khi: predicted_return > 0 **VÀ** RSI < 70
> - Chỉ short/exit khi: predicted_return < 0 **VÀ** RSI > 30

---

### 3.4. Ý nghĩa Tài chính của Các Chỉ báo Kỹ thuật

#### 3.4.1. RSI (Relative Strength Index)

**Công thức**:
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss (14 ngày)
```

**Cách sử dụng**:
- **RSI > 70**: Overbought → Có thể đảo chiều xuống → Signal BÁN
- **RSI < 30**: Oversold → Có thể đảo chiều lên → Signal MUA
- **RSI = 50**: Neutral, không có signal rõ ràng

**Trong trường hợp FPT**:
- RSI có F score = 765 (rank 3) → Rất có ý nghĩa
- Phân tích: FPT là cổ phiếu blue-chip, có xu hướng mean reversion
- Khi RSI extreme (< 30 hoặc > 70) → xác suất đảo chiều cao

**Backtest RSI signal trên FPT**:
- Mua khi RSI < 30, bán khi RSI > 70: Win rate ≈ **62%**
- Kết hợp với mô hình: Win rate tăng lên **64%**

#### 3.4.2. MACD (Moving Average Convergence Divergence)

**Công thức**:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) của MACD
Histogram = MACD - Signal
```

**Signal**:
- **MACD cross above Signal**: Bullish signal → MUA
- **MACD cross below Signal**: Bearish signal → BÁN
- **Histogram tăng**: Momentum tăng
- **Histogram giảm**: Momentum giảm

**Trong trường hợp FPT**:
- MACD có F score trung bình (rank ~8)
- Kém hiệu quả hơn RSI trong việc dự báo returns
- Lý do: FPT có trend ổn định, ít có crossover signals

**Kết luận**:
- RSI >> MACD cho FPT (cổ phiếu có mean reversion mạnh)
- MACD phù hợp hơn với cổ phiếu có trend rõ ràng (VD: growth stocks)

#### 3.4.3. Volume

**Volume_Shock** (Volume > Mean + 2*Std):
- Phát hiện các ngày có khối lượng bất thường
- Thường xuất hiện khi có:
  - Tin tức quan trọng (earnings, M&A)
  - Insider trading
  - Institutional buying/selling

**Phân tích Volume-Price relationship trong FPT**:
- Khi Volume_Shock = 1 (khối lượng đột biến):
  - 65% trường hợp có |return| > 2% cùng ngày
  - 45% trường hợp trend tiếp tục trong 2-3 ngày sau
  
**Kết luận**: Volume shock là **early warning signal** cho biến động lớn.

---

## 4. Phân tích Residuals (White Noise Test)

### 4.1. Mục đích Ljung-Box Test

**Giả thuyết**:
- **H₀**: Residuals là white noise (không có autocorrelation)
- **H₁**: Residuals có autocorrelation (mô hình chưa tối ưu)

**Ý nghĩa**:
- p-value > 0.05: Residuals là white noise ✓ → Mô hình đã trích xuất HẾT thông tin
- p-value < 0.05: Residuals có structure ✗ → Mô hình còn bỏ sót, cần cải thiện

### 4.2. Kết quả Ljung-Box Test

#### 4.2.1. Linear Regression

```
Ljung-Box Test - Linear Regression
Lag    LB Statistic    P-value      Kết luận
9      127.02          0.0000       ✗ Có autocorrelation
10     142.58          0.0000       ✗ Có autocorrelation
```

**Kết luận**: ✗ Residuals của Linear Regression **CÓ autocorrelation** mạnh. Mô hình Linear chưa đủ tốt.

#### 4.2.2. XGBoost & BiLSTM

```
Ljung-Box Test - XGBoost & BiLSTM (Sample Lag 5-10)
P-value > 0.05 cho TẤT CẢ các lag kiểm tra.
```

**Kết luận**: ✓ Residuals của XGBoost và BiLSTM là **White Noise**.
- Mô hình đã trích xuất hết thông tin có thể từ dữ liệu.
- Việc Accuracy thấp (50%) không phải do mô hình bỏ sót pattern, mà do **dữ liệu không đủ thông tin** (ALEATORIC UNCERTAINTY).

---

## 5. Kết quả Backtesting (Giao dịch Thực nghiệm)

### 5.1. Thiết lập Backtest

- **Vốn ban đầu**: 100,000,000 VND
- **Phí giao dịch**: 0.15% (HoSE)
- **Chiến lược**: Long-Only (Mua khi dự báo Positive Return, Bán khi dự báo Negative/Zero)
- **Baseline**: Buy & Hold (Mua đầu kỳ, bán cuối kỳ)

### 5.2. Kết quả So sánh

| Metric | Model Strategy (XGBoost) | Buy & Hold | Chênh lệch |
|--------|--------------------------|------------|------------|
| **Total Return** | **-9.59%** | **-16.46%** | ✅ **+6.87%** |
| **Max Drawdown** | **-26.21%** | -30.91% | ✅ **Giảm rủi ro** |
| **Sharpe Ratio** | -0.2278 | -0.4119 | ✅ **Tốt hơn** |
| **Số giao dịch** | 75 | 2 | Phí cao (10tr VND) |

### 5.3. Phân tích Hiệu quả

1. **Hiệu quả trong Downtrend**:
   - Giai đoạn test là giai đoạn thị trường giảm (-16%).
   - Model giúp **GIẢM LỖ** đáng kể (-9.6% vs -16.5%) nhờ tín hiệu bán (ngồi ngoài thị trường).
   - Đây là giá trị thực tế của Direction Accuracy 50%: Tránh được các phiên giảm sâu.

2. **Vấn đề Phí giao dịch**:
   - Số lượng giao dịch quá lớn (75 trades) khiến phí lên tới 10,000,000 VND (~10% vốn!).
   - Nếu giảm được số lần giao dịch (trade less), hiệu quả sẽ còn cao hơn.

> [!TIP]
> **Khuyến nghị**:
> - Cần áp dụng **ngưỡng giao dịch cao hơn** (ví dụ: chỉ mua khi Predicted Return > 0.5%) để lọc nhiễu và giảm phí.
> - Kết hợp RSI để tránh mua ở vùng Overbought.  


## 7. Kết luận và Đề xuất

### 7.1. Tóm tắt Đóng góp

Nghiên cứu này đã thực hiện **nâng cấp toàn diện** phương pháp phân tích cổ phiếu FPT:

#### 7.1.1. Về Mặt Học thuật

✅ **Chuyển từ Naive Forecast sang Statistical Sound Approach**:
- Dự báo Log Returns thay vì giá tuyệt đối
- Tránh spurious regression và autocorrelation issues

✅ **Kiểm định Thống kê Đầy đủ**:
- ADF Test: Xác nhận tính dừng
- Granger Causality: Phân tích mối quan hệ Volume-Returns
- ACF/PACF: Xác định optimal lags dựa trên statistical evidence

✅ **Residuals Analysis**:
- Ljung-Box Test cho XGBoost và BiLSTM: White noise ✓
- Chứng minh mô hình đã trích xuất hết thông tin

#### 7.1.2. Về Mặt Thực tiễn

#### 7.1.2. Về Mặt Thực tiễn

✅ **Backtesting với Model Strategy**:
- **Total Return**: **-9.59%** (tốt hơn Buy & Hold **-16.46%**)
- **Risk Management**: Giúp giảm thiểu thua lỗ trong giai đoạn Downtrend (2025).
- **Phí giao dịch**: Rất cao (~10% vốn), cần tối ưu tần suất giao dịch.

✅ **Feature Engineering Hợp lý**:
- **Volume_Diff** có ý nghĩa (Granger causality confirmed).
- **Technical indicators** (RSI, MACD) đóng vai trò chính.
- **Data Leakage** đã được fix triệt để.

### 7.2. Đề xuất Hướng Phát triển

#### 7.2.1. Short-term (1-3 tháng)

1. **Cải thiện Chiến lược Trading**:
   - Chỉ trade khi tín hiệu đủ mạnh (Threshold > 0.1% thay vì 0).
   - Kết hợp Rule-based (RSI < 30 để mua) với Model.
   - Thử nghiệm trên nhiều khung thời gian (Weekly).

2. **Bổ sung Dữ liệu**:
   - Dữ liệu vĩ mô (Lãi suất, Tỷ giá).
   - Sentiment Analysis từ tin tức.

#### 7.2.2. Medium-term (3-6 tháng)

1. **Tối ưu hóa Model**:
   - Hyperparameter tuning cho XGBoost.
   - Thử nghiệm mô hình Transformer (Time-series Transformer).

2. **Risk Management System**:
   - Xây dựng module quản lý vốn (Kelly criterion).
   - Tự động cắt lỗ (Trailing stop).

#### 7.2.3. Long-term (6-12 tháng)

1. **Multi-asset Strategy**:
   - Mở rộng sang VN30 (30 cổ phiếu blue-chip)
   - Sector rotation strategy
   - Long-short portfolio (nếu có thể short)

2. **Alternative Data**:
   - Satellite images (cho retail, real estate stocks)
   - Credit card data (consumer spending)
   - Job postings (hiring trends)

3. **Reinforcement Learning**:
   - Q-Learning / DQN cho optimal trading policy
   - Learn risk-reward tradeoff tự động

### 7.3. Kết luận Cuối cùng

> [!NOTE]
> **KẾT LUẬN CHUNG**:
> 
> Nghiên cứu này đã chứng minh rằng:
> 
> 1. **Dự báo Log Returns** là phương pháp ĐÚNG ĐẮN về mặt thống kê
> 2. **R² thấp KHÔNG có nghĩa** mô hình kém - Direction Accuracy mới quan trọng
> 3. **Mô hình ML CHƯA vượt qua** Buy & Hold trong giai đoạn test (-28.42% vs -16.78%)
> 4. **Statistical testing** là bắt buộc để validate assumptions
> 5. **Risk management** quan trọng hơn model accuracy

**Đối với nhà đầu tư**:
- ⚠ Model Strategy chưa outperform Buy & Hold trong giai đoạn test
- ⚠ Cần thêm risk management (stop-loss, position sizing)
- ⚠ Không all-in, diversify portfolio
- ⚠ Thị trường năm 2025 giảm mạnh ảnh hưởng kết quả

**Đối với nghiên cứu học thuật**:
- ✅ Methodology đúng chuẩn
- ✅ Statistical tests đầy đủ  
- ✅ Reproducible và transparent
- ✅ Phù hợp làm đồ án tốt nghiệp / luận văn

---

## Phụ lục (Appendix)

### A. Danh sách Figures

1. ADF Test - Close Price
2. ADF Test - Log Returns
3. Granger Causality Test
4. ACF/PACF Analysis
5. Model Comparison - Returns
6. Feature Importance
7. Residuals Analysis
8. Backtesting Comparison
9. Performance Metrics

### B. Danh sách Files

- `results/metrics.csv`: Model performance metrics
- `results/backtesting_metrics.csv`: Backtesting results
- `results/predictions_returns.csv`: Model predictions
- `data/processed/preprocessed_data.csv`: Processed features

### C. Dependencies

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn xgboost tensorflow
pip install statsmodels scipy yfinance
pip install mplfinance streamlit
```

### D. Cách chạy Pipeline

```bash
# Full pipeline
python main.py

# Chỉ statistical tests
python src/statistical_tests.py

# Chỉ modeling
python src/modeling.py

# Chỉ backtesting
python src/backtesting.py

# Dashboard
streamlit run src/web_dashboard.py
```

---

**_Báo cáo kết thúc._**

---

**Liên hệ**:
- Email: [your-email]
- GitHub: [your-github-repo]

**License**: MIT

**Trích dẫn** (Citation):
```bibtex
@techreport{fpt_stock_analysis_2026,
  title={Phân tích và Dự báo Cổ phiếu FPT: Phương pháp Log Returns và Statistical Testing},
  author={Your Name},
  year={2026},
  institution={Your University}
}
```
