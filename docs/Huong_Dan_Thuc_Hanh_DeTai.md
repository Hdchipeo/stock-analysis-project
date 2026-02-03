# Hướng dẫn Thực hành Đồ án

## 1. Cài đặt

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn xgboost tensorflow
pip install statsmodels scipy yfinance mplfinance streamlit
```

## 2. Chạy Pipeline

```bash
cd e:\application\python\stock-analysis-project
python main.py
```

**Pipeline bao gồm:**
1. Thu thập dữ liệu (yfinance)
2. Tiền xử lý (feature engineering)
3. Kiểm định thống kê (ADF, Granger, ACF/PACF)
4. EDA & Visualization
5. Modeling (LR, XGBoost, BiLSTM)
6. Backtesting

## 3. Xem Dashboard

```bash
streamlit run src/web_dashboard.py
```

## 4. Đọc Báo cáo

Xem file `docs/Final_Report.md`

## 5. Cấu trúc Thư mục

```
stock-analysis-project/
├── data/raw/               # Dữ liệu thô
├── data/processed/         # Dữ liệu đã xử lý
├── src/                    # Source code
├── results/figures/        # Biểu đồ
├── docs/                   # Tài liệu
└── main.py                 # Entry point
```

## 6. Modules Chính

| Module | Chức năng |
|--------|-----------|
| `collect_data.py` | Thu thập từ yfinance |
| `preprocess_data.py` | Feature engineering |
| `statistical_tests.py` | ADF, Granger, ACF/PACF |
| `modeling.py` | Train LR, XGBoost, BiLSTM |
| `backtesting.py` | Đánh giá chiến lược |
| `web_dashboard.py` | Dashboard Streamlit |
