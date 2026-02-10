# Stock Price Analysis & Prediction Project (FPT.VN)

## 📌 Introduction
Dự án thực hiện thu thập dữ liệu, phân tích biến động giá và khối lượng giao dịch của cổ phiếu FPT thông qua các mô hình học máy (XGBoost, BiLSTM). Hệ thống giúp tối ưu hóa chiến lược đầu tư và quản trị rủi ro dựa trên các chỉ báo kỹ thuật và kiểm định thống kê chuyên sâu.

## 📂 Project Structure
```
project_root/
├── data/               # Raw and processed data
├── src/                # Source code (collect, analyze, model, dashboard)
├── results/            # Figures (.png) and Metrics (.csv)
├── docs/               # Detailed documentation and reports
├── main.py             # Unified entry point
└── README.md           # Project documentation
```

## 🛠️ Installation

Cài đặt toàn bộ các thư viện cần thiết:

```bash
pip install pandas numpy yfinance matplotlib seaborn scikit-learn xgboost tensorflow plotly streamlit tabulate mplfinance pandas_ta statsmodels
```

## 🏃 Usage

### 1. Run the Full Pipeline
Chạy lệnh sau để thực hiện lại toàn bộ các bước.

**Lưu ý:** Dữ liệu có thể thay đổi khi chạy lại này vì thời gian collect data, training model, backtesting có thể thay đổi.
```bash
python3 main.py
```

### 2. Web Dashboard
Chạy lệnh sau để mở dashboard:
```bash
streamlit run src/web_dashboard.py
```