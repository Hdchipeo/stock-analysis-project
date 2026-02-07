"""
FPT Stock Analysis Dashboard - Phiên bản đồ án môn học
Đơn giản hóa và dễ hiểu cho mục đích học thuật
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FPT Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    """Load all necessary data files"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    data_dir = os.path.join(base_dir, "data", "processed")
    figures_dir = os.path.join(results_dir, "figures")
    
    # Load metrics
    metrics_df = None
    metrics_path = os.path.join(results_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
    
    # Load predictions
    preds_df = None
    preds_path = os.path.join(results_dir, "predictions_returns.csv")
    if os.path.exists(preds_path):
        preds_df = pd.read_csv(preds_path, index_col=0, parse_dates=True)
    
    # Load backtesting results
    backtest_df = None
    backtest_path = os.path.join(results_dir, "backtesting_metrics.csv")
    if os.path.exists(backtest_path):
        backtest_df = pd.read_csv(backtest_path)
    
    # Load test data for prices
    test_df = None
    test_path = os.path.join(data_dir, "test_data.csv")
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    
    return metrics_df, preds_df, backtest_df, test_df, figures_dir

metrics_df, preds_df, backtest_df, test_df, figures_dir = load_data()

# --- HEADER ---
st.title("📈 FPT Stock Analysis Dashboard")
st.markdown("**Đồ án Phân tích và Dự báo Cổ phiếu FPT.VN**")
st.markdown("Giai đoạn: 01/01/2021 - 31/12/2025")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Điều hướng")
    
    page = st.radio(
        "Chọn trang:",
        ["Tổng quan", "Hiệu suất Mô hình", "Backtesting", "Biểu đồ"]
    )
    
    st.divider()
    st.info("**Đồ án môn học**\n\nPhân tích cổ phiếu sử dụng Machine Learning")

# ==================== PAGE: TỔNG QUAN ====================
if page == "Tổng quan":
    st.header("📊 Tổng quan Dự án")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Mục tiêu")
        st.markdown("""
        - Thu thập dữ liệu cổ phiếu FPT.VN (2021-2026)
        - Phân tích thống kê và EDA
        - Xây dựng mô hình dự báo Log Returns
        - Đánh giá hiệu quả qua Backtesting
        """)
        
        st.subheader("📈 Tại sao dùng Log Returns?")
        st.markdown("""
        - **Stationary**: Chuỗi dừng, phù hợp cho ML
        - **Symmetric**: Xử lý tốt tăng/giảm
        - **Additive**: Dễ tính tổng lợi nhuận
        """)
    
    with col2:
        st.subheader("📁 Dữ liệu")
        if test_df is not None:
            st.metric("Số phiên test", len(test_df))
        if metrics_df is not None:
            st.metric("Số mô hình", len(metrics_df))
        if backtest_df is not None:
            st.metric("Backtesting", "✅ Đã chạy")

# ==================== PAGE: HIỆU SUẤT MÔ HÌNH ====================
elif page == "Hiệu suất Mô hình":
    st.header("🤖 Hiệu suất Mô hình")
    
    if metrics_df is not None and not metrics_df.empty:
        st.subheader("📊 Bảng so sánh")
        st.dataframe(metrics_df, use_container_width=True)
        
        st.info("""
        **Lưu ý**: R² thấp (0.05-0.15) là **BÌN THƯỜNG** với dữ liệu tài chính!
        
        **Direction Accuracy > 55%** = Mô hình có giá trị thực tiễn cho trading.
        """)
        
        # Bar chart comparison
        st.subheader("📈 So sánh Direction Accuracy")
        
        if 'Direction_Accuracy' in metrics_df.columns:
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df['Direction_Accuracy'],
                    marker_color=['#2E86AB', '#A23B72', '#F18F01'][:len(metrics_df)]
                )
            ])
            fig.add_hline(y=55, line_dash="dash", line_color="red", annotation_text="Ngưỡng 55%")
            fig.update_layout(
                title="Direction Accuracy theo Mô hình", yaxis_title="Accuracy (%)", height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Chưa có dữ liệu metrics. Vui lòng chạy `python main.py` trước.")
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    feat_path = os.path.join(figures_dir, "feature_importance_returns.png")
    if os.path.exists(feat_path):
        st.image(feat_path, width=800)
    else:
        st.info("Biểu đồ feature importance sẽ hiển thị sau khi chạy pipeline.")

# ==================== PAGE: BACKTESTING ====================
elif page == "Backtesting":
    st.header("💰 Kết quả Backtesting")
    
    if backtest_df is not None and not backtest_df.empty:
        st.subheader("📊 So sánh chiến lược")
        st.dataframe(backtest_df, use_container_width=True)
        
        st.subheader("📈 Biểu đồ Portfolio")
        backtest_chart = os.path.join(figures_dir, "backtesting_comparison.png")
        if os.path.exists(backtest_chart):
            st.image(backtest_chart, width=900)
        
        st.subheader("📊 Performance Metrics")
        perf_chart = os.path.join(figures_dir, "performance_metrics_comparison.png")
        if os.path.exists(perf_chart):
            st.image(perf_chart, width=800)
        
        st.markdown("""
        ### 💡 Giải thích các chỉ số
        
        | Chỉ số | Ý nghĩa | Đánh giá tốt |
        |--------|---------|--------------|
        | **Total Return** | Tổng lợi nhuận | Càng cao càng tốt |
        | **Sharpe Ratio** | Lợi nhuận/Rủi ro | > 1.0 = Tốt |
        | **Max Drawdown** | Mức giảm tối đa | < -15% = Chấp nhận |
        | **Win Rate** | % ngày có lời | > 50% = Tốt |
        """)
    else:
        st.warning("⚠️ Chưa có kết quả backtesting. Vui lòng chạy `python main.py` trước.")

# ==================== PAGE: BIỂU ĐỒ ====================
elif page == "Biểu đồ":
    st.header("📊 Các Biểu đồ Phân tích")
    
    st.subheader("1️⃣ ADF Test - Kiểm định tính dừng")
    col1, col2 = st.columns(2)
    
    with col1:
        adf_close = os.path.join(figures_dir, "adf_test_close_price.png")
        if os.path.exists(adf_close):
            st.image(adf_close, caption="Close Price - Không dừng ❌")
    
    with col2:
        adf_returns = os.path.join(figures_dir, "adf_test_log_returns.png")
        if os.path.exists(adf_returns):
            st.image(adf_returns, caption="Log Returns - Dừng ✅")
    
    st.divider()
    
    st.subheader("2️⃣ ACF/PACF - Xác định Lags tối ưu")
    acf_path = os.path.join(figures_dir, "acf_pacf_log_returns.png")
    if os.path.exists(acf_path):
        st.image(acf_path, width=900)
    
    st.divider()
    
    st.subheader("3️⃣ Predictions vs Actual")
    model_comp = os.path.join(figures_dir, "model_comparison_returns.png")
    if os.path.exists(model_comp):
        st.image(model_comp, width=900)
    
    # List all available figures
    st.divider()
    st.subheader("📁 Tất cả biểu đồ có sẵn")
    
    if os.path.exists(figures_dir):
        figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        if figures:
            selected_fig = st.selectbox("Chọn biểu đồ:", figures)
            if selected_fig:
                st.image(os.path.join(figures_dir, selected_fig), width=900)
        else:
            st.info("Chưa có biểu đồ. Vui lòng chạy `python main.py` trước.")

# --- FOOTER ---
st.divider()
st.caption("📊 FPT Stock Analysis Dashboard | Đồ án môn học | 2021-2025")
