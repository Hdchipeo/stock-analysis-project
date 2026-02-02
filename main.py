import sys
import os

# Add src to python path to facilitate imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import collect_data
import analyze_data
import descriptive_stats
import preprocess_data
import eda_analysis
import statistical_tests  # NEW: Statistical testing module
import modeling
import backtesting  # NEW: Backtesting module

def main():
    """
    UPGRADED STOCK ANALYSIS PIPELINE - FPT.VN
    
    Nâng cấp chính:
    - Dự báo Log Returns thay vì giá tuyệt đối
    - Kiểm định thống kê đầy đủ (ADF, Granger, ACF/PACF)
    - Mô hình BiLSTM cho time series
    - Backtesting với chiến lược giao dịch thực tế
    """
    print("="*80)
    print(" " * 20 + "STOCK ANALYSIS PIPELINE - UPGRADED VERSION")
    print(" " * 30 + "FPT.VN Analysis")
    print("="*80)

    # Phase 1: Data Collection
    print("\n" + "█"*80)
    print("PHASE 1: DATA COLLECTION")
    print("█"*80)
    collect_data.collect_stock_data(ticker="FPT.VN")
    analyze_data.analyze_and_describe_variables()

    # Phase 2: Descriptive Statistics
    print("\n" + "█"*80)
    print("PHASE 2: DESCRIPTIVE STATISTICS")
    print("█"*80)
    descriptive_stats.calculate_descriptive_stats()

    # Phase 3: Data Preprocessing
    print("\n" + "█"*80)
    print("PHASE 3: DATA PREPROCESSING")
    print("█"*80)
    print("Bao gồm:")
    print("  - Feature Engineering: Log Returns, Volume features, Price Direction")
    print("  - Technical Indicators: RSI, MACD, SMA")
    print("  - Lag Features: Returns_Lag, Volume_Change_Lag")
    preprocess_data.preprocess_stock_data()

    # Phase 4: Statistical Tests (NEW!)
    print("\n" + "█"*80)
    print("PHASE 4: STATISTICAL TESTING (NEW!)")
    print("█"*80)
    print("Kiểm định:")
    print("  - ADF Test: Kiểm tra tính dừng (Stationarity)")
    print("  - Granger Causality: Volume có dự báo được Returns không?")
    print("  - ACF/PACF: Xác định optimal lags")
    statistical_tests.run_all_statistical_tests()

    # Phase 5: EDA & Visualization
    print("\n" + "█"*80)
    print("PHASE 5: EDA & VISUALIZATION")
    print("█"*80)
    eda_analysis.run_eda_analysis()

    # Phase 6: Modeling (UPGRADED!)
    print("\n" + "█"*80)
    print("PHASE 6: MODELING - LOG RETURNS PREDICTION (UPGRADED!)")
    print("█"*80)
    print("Mô hình:")
    print("  - Linear Regression (Baseline)")
    print("  - XGBoost (Gradient Boosting)")
    print("  - BiLSTM (Deep Learning)")
    print("\nLưu ý: R² thấp (<0.1) là BÌN THƯỜNG với dữ liệu tài chính!")
    modeling.run_modeling()

    # Phase 7: Backtesting (NEW!)
    print("\n" + "█"*80)
    print("PHASE 7: BACKTESTING - TRADING STRATEGY (NEW!)")
    print("█"*80)
    print("So sánh:")
    print("  - Model Strategy (dùng dự báo để giao dịch)")
    print("  - Buy & Hold (mua và giữ)")
    print("\nMetrics: Sharpe Ratio, Max Drawdown, Win Rate")
    
    # Note: backtesting uses predictions from modeling
    # Make sure modeling has generated predictions_returns.csv
    backtesting.run_backtesting(
        predictions_file="predictions_returns.csv",
        test_data_file="test_data.csv"
    )

    print("\n" + "="*80)
    print(" " * 25 + "PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print("\n📊 KẾT QUẢ ĐƯỢC LƯU TẠI:")
    print("   - results/metrics.csv: Hiệu suất mô hình")
    print("   - results/backtesting_metrics.csv: Kết quả backtesting")
    print("   - results/figures/: Tất cả biểu đồ")
    print("   - docs/Final_Report.md: Báo cáo chi tiết")
    print("\n💡 BƯỚC TIẾP THEO:")
    print("   - Đọc Final_Report.md để hiểu kết quả")
    print("   - Chạy 'streamlit run src/web_dashboard.py' để xem dashboard")
    print()

if __name__ == "__main__":
    main()

