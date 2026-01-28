# Stock Price Analysis & Prediction Project (FPT.VN)

![Dashboard Preview](results/figures/dashboard.png)

## 📌 Introduction
This project performs an in-depth analysis and prediction of FPT Corp. (FPT.VN) stock prices over a 5-year period. It leverages statistical analysis and Machine Learning (XGBoost & Linear Regression) to forecast future trends.

The project includes a complete pipeline: **Data Collection -> Preprocessing -> EDA -> Modeling -> Dashboarding**.

## 🚀 Key Features
-   **Automated Data Pipeline**: Fetches and processes historical data using `yfinance`.
-   **Advanced EDA**: Visualizes Trends, Distributions, Correlations, and Seasonality.
-   **Multi-Model Forecasting**:
    -   **Linear Regression** (Baseline)
    -   **XGBoost** (Feature Importance Analysis)
-   **Interactive Dashboard**: A Streamlit-based web app for real-time analysis and investment signals.

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

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Hdchipeo/stock-analysis-project.git
    cd stock-analysis-project
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas numpy yfinance matplotlib seaborn scikit-learn xgboost tensorflow plotly streamlit
    ```

## 🏃 Usage

### 1. Run the Full Pipeline
To collect data, train models, and generate static reports:
```bash
python3 main.py
```

### 2. Launch the Web Dashboard
To interact with the charts and signals:
```bash
streamlit run src/web_dashboard.py
```

## 📊 Results Overview
| Model | RMSE | R2 Score |
| :--- | :--- | :--- |
| **Linear Regression** | 0.0054 | 0.9953 |
| **XGBoost** | 0.0152 | 0.9627 |

> **Conclusion**: Linear Regression performs exceptionally well (R2 > 0.99), indicating strong momentum in FPT.VN's stock price. XGBoost provides valuable insights into feature importance but with slightly lower accuracy.

## 📝 Documentation
-   [Final Report (Full Analysis)](docs/Final_Report.md)
-   [Web Dashboard Guide](docs/Web_Dashboard_Guide.md)
-   [Walkthrough Report](docs/Walkthrough_Report.md)

---