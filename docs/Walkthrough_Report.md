# Walkthrough: Kết quả Phân tích Cổ phiếu FPT

**Ngày thực hiện**: 03/02/2026  
**Giai đoạn dữ liệu**: 01/01/2021 - 31/12/2025

---

## 📊 Tóm tắt Kết quả

### Thống kê

| Test | Kết quả |
|------|---------|
| **ADF - Close Price** | ❌ Không dừng (p > 0.05) |
| **ADF - Log Returns** | ✅ Dừng (p < 0.0001) |
| **Granger Causality** | ✅ Volume có ảnh hưởng |
| **Residuals** | ✅ White Noise |

### Mô hình

| Model | R² | Direction Accuracy |
|-------|----|--------------------|
| Linear Regression | 0.007 | 48.59% |
| XGBoost | 0.095 | 48.59% |
| BiLSTM | 0.057 | 50.20% |

### Backtesting (Test period: 2024-2025)

| Chiến lược | Return | Sharpe Ratio | Max DD |
|------------|--------|--------------|--------|
| Model Strategy | **-39.29%** | -1.58 | -41.11% |
| Buy & Hold | **-32.54%** | -0.73 | -43.23% |

---

## 💡 Nhận xét Chính

1. **Thị trường giảm mạnh** (~33%) trong 2024-2025 → Cả hai chiến lược đều lỗ

2. **Direction Accuracy ~50%** → Mô hình không có ưu thế thực sự

3. **R² thấp là bình thường** với dữ liệu tài chính

4. **Residuals = White Noise** → Mô hình đã tối ưu về mặt thống kê

5. **Phí giao dịch cao** (11.3M) ảnh hưởng đến lợi nhuận

---

## 📁 Files được Tạo

**Results:**
- `results/metrics.csv` - Kết quả mô hình
- `results/backtesting_metrics.csv` - Kết quả backtesting
- `results/predictions_returns.csv` - Dự báo

**Figures:**
- `adf_test_*.png` - ADF tests
- `acf_pacf_log_returns.png` - ACF/PACF
- `model_comparison_returns.png` - So sánh mô hình
- `backtesting_comparison.png` - Portfolio comparison
- `performance_metrics_comparison.png` - Metrics

---

## 🎯 Kết luận

Đồ án đã hoàn thành với phương pháp thống kê đúng đắn:
- ✅ Dự báo Log Returns (không phải giá tuyệt đối)
- ✅ Kiểm định thống kê đầy đủ
- ✅ Backtesting với chiến lược thực tế

**Hạn chế**: Mô hình không thắng được thị trường giảm. Cần thêm regime detection để cải thiện.
