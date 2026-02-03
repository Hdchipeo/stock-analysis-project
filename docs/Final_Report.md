# BÁO CÁO PHÂN TÍCH VÀ DỰ BÁO CỔ PHIẾU FPT (Mã: FPT.VN)

**Môn học**: Phân tích dữ liệu / Machine Learning  
**Giai đoạn dữ liệu**: 01/01/2021 - 31/12/2025  
**Ngày báo cáo**: 03/02/2026

---

## 1. Giới thiệu

### 1.1. Mục tiêu đồ án

Đồ án thực hiện phân tích và dự báo cổ phiếu FPT Corp. (FPT.VN) sử dụng các phương pháp Machine Learning.

**Mục tiêu cụ thể:**
1. Thu thập và tiền xử lý dữ liệu cổ phiếu FPT
2. Thực hiện kiểm định thống kê (ADF, Granger, ACF/PACF)
3. Xây dựng mô hình dự báo Log Returns
4. So sánh hiệu quả Backtesting giữa năm 2024 và 2025

### 1.2. Tại sao dùng Log Returns?

**Công thức**: `Log Return = ln(P_t / P_{t-1})`

| Đặc điểm | Giá tuyệt đối | Log Returns |
|----------|---------------|-------------|
| Tính dừng | ❌ Không | ✅ Có |
| R² | 0.99 (ảo) | 0.05-0.10 (thực) |
| Phù hợp ML | ❌ | ✅ |

---

## 2. Dữ liệu

| Thông tin | Chi tiết |
|-----------|----------|
| **Ticker** | FPT.VN |
| **Nguồn** | Yahoo Finance |
| **Giai đoạn** | 01/01/2021 - 31/12/2025 |
| **Số phiên** | 1,246 phiên |
| **Train** | 748 phiên (2021-2023) |
| **Test** | 498 phiên (2024-2025) |

---

## 3. Kiểm định Thống kê

### 3.1. ADF Test

| Chuỗi | P-value | Kết luận |
|-------|---------|----------|
| Close Price | 0.626 | ❌ Không dừng |
| Log Returns | 0.000 | ✅ Dừng |

### 3.2. Granger Causality

Volume_Change → Log_Returns: **KHÔNG** có mối quan hệ nhân quả

### 3.3. ACF/PACF

Optimal lags: **[2, 7, 27]**

---

## 4. Mô hình Dự báo

### 4.1. Kết quả

| Mô hình | RMSE | R² | Direction Accuracy |
|---------|------|----|--------------------|
| Linear Regression | 0.020 | 0.011 | 52.81% |
| XGBoost | 0.019 | 0.051 | 52.21% |
| BiLSTM | 0.020 | 0.020 | 50.41% |

**Lưu ý**: R² thấp là **BÌNH THƯỜNG** với dữ liệu tài chính.

---

## 5. Backtesting - So sánh 2024 vs 2025

### 5.1. Thiết lập

- **Vốn ban đầu**: 100,000,000 VND
- **Phí giao dịch**: 0.15%
- **Chiến lược**: Long-Only (mua khi dự báo tăng)

### 5.2. Kết quả NĂM 2024 (Thị trường TĂNG)

| Metric | Model Strategy | Buy & Hold |
|--------|----------------|------------|
| **Lợi nhuận** | +74.90% | +137.84% |
| **Sharpe Ratio** | 2.24 | 2.77 |
| **Max Drawdown** | -10.66% | -19.40% |
| **Số giao dịch** | 109 | 2 |

**Nhận xét 2024:** 
- Thị trường TĂNG mạnh (+137%)
- Model lãi +74.9% nhưng **KÉM HƠN** Buy & Hold
- Model có Max Drawdown **THẤP HƠN** (rủi ro ít hơn)

### 5.3. Kết quả NĂM 2025 (Thị trường GIẢM)

| Metric | Model Strategy | Buy & Hold |
|--------|----------------|------------|
| **Lợi nhuận** | -27.69% | -32.26% |
| **Sharpe Ratio** | -0.87 | -0.73 |
| **Max Drawdown** | -31.38% | -43.23% |
| **Số giao dịch** | 127 | 2 |

**Nhận xét 2025:** 
- Thị trường GIẢM mạnh (-32%)
- Model lỗ -27.7% nhưng **TỐT HƠN** Buy & Hold 4.57%
- Model có Max Drawdown **THẤP HƠN** 12% (bảo vệ vốn tốt hơn)

### 5.4. So sánh Tổng hợp

| Metric | 2024 | 2025 |
|--------|------|------|
| Model vs B&H | Kém 62.94% | **Tốt hơn 4.57%** |
| Model Max DD | -10.66% | -31.38% |
| B&H Max DD | -19.40% | -43.23% |

**Kết luận:**
- **Thị trường tăng (bull)**: Buy & Hold thắng
- **Thị trường giảm (bear)**: Model Strategy thắng
- Model **giảm thiểu rủi ro** (Max Drawdown thấp hơn) trong mọi điều kiện

---

## 6. Kết luận

### 6.1. Những điểm đạt được

✅ Thu thập và xử lý 1,246 phiên giao dịch (2021-2025)  
✅ Kiểm định thống kê đầy đủ (ADF, Granger, ACF/PACF)  
✅ Xây dựng 3 mô hình: LR, XGBoost, BiLSTM  
✅ Backtesting so sánh 2024 vs 2025  

### 6.2. Insight chính

1. **Model hoạt động tốt hơn trong thị trường giảm**
2. **Model giảm rủi ro** (Max Drawdown thấp hơn Buy & Hold)
3. **Trade-off**: Lợi nhuận thấp hơn khi thị trường tăng, nhưng bảo vệ vốn tốt hơn khi giảm

### 6.3. Đề xuất

- Kết hợp với **Regime Detection** để chuyển đổi chiến lược theo xu hướng thị trường
- Áp dụng model trong thị trường biến động/giảm
- Sử dụng Buy & Hold khi thị trường tăng rõ ràng

---

## 7. Phụ lục

### 7.1. Cách Chạy

```bash
# Chạy pipeline
python main.py

# Xem dashboard
streamlit run src/web_dashboard.py
```

### 7.2. Output Files

- `results/metrics.csv` - Kết quả mô hình
- `results/backtesting_metrics.csv` - Kết quả backtesting 2024 vs 2025
- `results/figures/backtesting_yearly_comparison.png` - Biểu đồ so sánh

---

**--- HẾT ---**
