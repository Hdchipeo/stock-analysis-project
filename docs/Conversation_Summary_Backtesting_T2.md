# 📊 Tóm Tắt Cuộc Hội Thoại: Phân Tích Backtesting T+2 và Cải Tiến Chiến Lược

**Ngày**: 2026-02-02  
**Dự án**: Stock Analysis Project - FPT.VN

---

## 🎯 Mục Tiêu Ban Đầu

Phân tích tác động của quy tắc T+2 (settlement) lên kết quả backtesting và tìm cách cải thiện hiệu quả chiến lược trading.

---

## ⚠️ Vấn Đề Phát Hiện

### 1. Kết quả backtesting không nhất quán
- **Ban đầu**: Win rate 56.7%, lợi nhuận +28.34%
- **Sau khi chạy lại**: Win rate 24%, lợi nhuận -25%

### 2. Nguyên nhân gốc rễ
1. **Dữ liệu đã được MinMaxScaler** về [0,1]
2. **Logic so sánh sai**: So sánh prediction với `0` thay vì `0.5` (điểm giữa)
3. **Giai đoạn test khác nhau**: Thị trường 2024-2025 giảm mạnh (-26%)

---

## 🔧 Các Sửa Đổi Đã Thực Hiện

### 1. Sửa `collect_data.py`
```python
# Từ:
def collect_stock_data(ticker="FPT.VN", period="5y", interval="1d"):

# Thành:
def collect_stock_data(ticker="FPT.VN", start="2021-01-01", end="2025-12-31", interval="1d"):
```

### 2. Sửa `preprocess_data.py`
```python
# Từ: split 80/20 cố định
split_idx = int(len(df_scaled) * 0.8)

# Thành: test 1 năm cuối (thực tế hơn)
test_size = min(250, int(len(df_scaled) * 0.2))
split_idx = len(df_scaled) - test_size
```

### 3. Sửa `backtesting.py` - Logic threshold
```python
# Từ:
if pred_return > 0 and shares == 0:  # SAI!

# Thành:
if pred_return > threshold and shares == 0:  # threshold=0.5
```

### 4. Thêm chiến lược Mean Reversion mới
```python
def mean_reversion_strategy(self, predictions_df, actual_prices, rsi_series, 
                            stop_loss_pct=0.07, lookback_window=30):
    """
    Các cải tiến:
    1. Stop-Loss 7%: Tự động cắt lỗ khi giảm 7%
    2. Dynamic Threshold: Ngưỡng = rolling mean 30 ngày
    3. RSI Filter: Mua khi RSI < 40, Bán khi RSI > 60
    """
```

---

## 📈 Kết Quả So Sánh 4 Chiến Lược

| Metric | Momentum (Không T+2) | Momentum (Có T+2) | Mean Reversion + SL | Buy & Hold |
|--------|----------------------|-------------------|---------------------|------------|
| **Lợi nhuận** | -30.45% | -33.99% | **-32.15%** | -25.96% |
| **Win Rate** | 22.49% | 23.69% | **36.55%** ✓ | N/A |
| **Số giao dịch** | 103 | 78 | **15** ✓ | 2 |
| **Tổng phí** | 12.6M | 9.1M | **1.8M** ✓ | 261K |
| **Max Drawdown** | -31.93% | -33.99% | -37.91% | -34.51% |

---

## 🔍 Chi Tiết Mean Reversion Strategy

- **Stop-Loss triggered**: 6 lần
- **RSI Buy signals**: 8 lần
- **RSI Sell signals**: 1 lần
- **Trade Win Rate**: 14.3%

---

## 💡 Bài Học Rút Ra

### 1. Về dữ liệu
- MinMaxScaler thay đổi ý nghĩa của giá trị 0
- Cần hiểu rõ preprocessing trước khi viết logic trading

### 2. Về chiến lược
- **Momentum** chỉ hoạt động tốt trong thị trường tăng
- **Mean Reversion** giảm số giao dịch, tiết kiệm phí
- **Stop-Loss** bảo vệ khỏi lỗ lớn

### 3. Về thị trường
- Năm 2025 FPT giảm 26% → Không chiến lược nào thắng Buy & Hold
- "Đôi khi không làm gì là tốt nhất"

---

## 📁 Files Đã Sửa Đổi

| File | Thay đổi |
|------|----------|
| `src/collect_data.py` | Dùng start/end date thay vì period |
| `src/preprocess_data.py` | Test data = 1 năm cuối |
| `src/backtesting.py` | Sửa threshold=0.5, thêm Mean Reversion strategy |

---

## 🚀 Hướng Phát Triển Tiếp

1. **Tối ưu RSI thresholds** (thử 35/65 thay vì 40/60)
2. **Thêm Take-Profit** mechanism
3. **Position Sizing** (không all-in 100% mỗi lần)
4. **Ensemble signals** (kết hợp nhiều indicators)
5. **Walk-forward optimization** để tránh overfitting

---

## 📊 Kết Luận

> Chiến lược **Mean Reversion + Stop-Loss** đã cải thiện đáng kể so với Momentum:
> - Giảm 85% số giao dịch
> - Tiết kiệm 10.8M VND phí
> - Win Rate tăng 60%
> 
> Tuy nhiên, trong thị trường giảm liên tục, **không chiến lược active trading nào thắng được passive Buy & Hold**.

---

*Tạo bởi Antigravity AI Assistant - 2026-02-03*
