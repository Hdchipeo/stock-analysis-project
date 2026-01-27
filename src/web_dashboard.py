import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    metrics_path = os.path.join(results_dir, "metrics.csv")
    predictions_path = os.path.join(results_dir, "predictions.csv")
    figures_dir = os.path.join(results_dir, "figures")
    
    # Load Metrics
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = pd.DataFrame(columns=["Model", "RMSE", "R2", "MAPE"])
        
    # Load Predictions
    if os.path.exists(predictions_path):
        preds_df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    else:
        preds_df = pd.DataFrame()
        
    return metrics_df, preds_df, figures_dir

metrics_df, preds_df, figures_dir = load_data()

# --- TITLE & SIDEBAR ---
st.title("📈 Stock Price Analysis & Prediction Dashboard")
st.markdown("Professional analysis of **AAPL** stock using Machine Learning & Deep Learning models.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selection
    model_options = [col for col in preds_df.columns if col != 'Actual']
    selected_model = st.selectbox("Select Prediction Model", model_options, index=len(model_options)-1 if model_options else 0)
    
    # Date Range
    days_to_show = st.slider("Days to Visualize", min_value=30, max_value=len(preds_df), value=100)
    
    st.divider()
    st.info("Built with [Streamlit](https://streamlit.io) • Phase 6b")

# --- KPI METRICS ---
if not preds_df.empty:
    latest_date = preds_df.index[-1]
    last_actual = preds_df['Actual'].iloc[-1]
    prev_actual = preds_df['Actual'].iloc[-2]
    
    daily_change = last_actual - prev_actual
    daily_change_pct = (daily_change / prev_actual) * 100
    
    # Signal Logic
    # Simple strategy: If Model predicts Up for *next* step vs current actual.
    # Note: Our predictions are aligned. So pred[t] is prediction for time t.
    # To simulate a signal, we look at if pred[t] > actual[t-1] implies upward movement expected?
    # Or simply: compare Predicted Current vs Actual Current to see Over/Undervalued?
    # Let's use: Trend direction.
    
    # Signal: If Model Prediction > Actual Price => Undervalued/Buy signal? 
    # Or: If Predicted Trend (Pred[t] vs Pred[t-1]) is UP.
    
    last_pred = preds_df[selected_model].iloc[-1]
    signal_val = "HOLD"
    signal_color = "normal"
    
    if last_pred > last_actual * 1.01: # Predicted > 1% of actual
        signal_val = "BUY (Undervalued)"
        signal_color = "normal" # Streamlit metric delta handles color
    elif last_pred < last_actual * 0.99:
        signal_val = "SELL (Overvalued)"
        signal_color = "inverse"
    else:
        signal_val = "HOLD"
        signal_color = "off"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Date", latest_date.strftime('%Y-%m-%d'))
    col2.metric("Current Price", f"${last_actual:.2f}")
    col3.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:.2f}%")
    col4.metric("Model Signal", signal_val, delta_color=signal_color)

# --- MAIN CHART ---
st.subheader(f"Price History: Actual vs {selected_model}")

if not preds_df.empty:
    subset = preds_df.iloc[-days_to_show:]
    
    fig = go.Figure()
    
    # Actual Price
    fig.add_trace(go.Scatter(
        x=subset.index, y=subset['Actual'],
        mode='lines', name='Actual Price',
        line=dict(color='black', width=2)
    ))
    
    # Predicted Price
    fig.add_trace(go.Scatter(
        x=subset.index, y=subset[selected_model],
        mode='lines', name=f'Predicted ({selected_model})',
        line=dict(color='#00CC96', width=2, dash='dash')
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- DETAILED ANALYSIS ---
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Model Performance")
    st.dataframe(metrics_df.style.highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE"], color="#90EE90"), use_container_width=True)
    
    st.info(f"**Best Model**: check RMSE column (Lower is better).")

with c2:
    st.subheader("Feature Importance")
    fet_imp_path = os.path.join(figures_dir, "feature_importance.png")
    if os.path.exists(fet_imp_path):
        st.image(fet_imp_path, caption="XGBoost Feature Importance", use_container_width=True)
    else:
        st.warning("Feature importance image not found.")

# --- INVESTMENT RECOMMENDATIONS ---
st.subheader("💡 Investment Recommendations & Conclusion")

with st.expander("See Detailed Strategy", expanded=True):
    st.markdown("""
    ### 1. Strategy
    *   **Trend Following**: Use the **BiLSTM** model to identify the major trend direction.
    *   **Entry/Exit**: Refine signals using **RSI**.
        *   **Identify Buy**: Model predicts UP + RSI < 30 (Oversold).
        *   **Identify Sell**: Model predicts DOWN + RSI > 70 (Overbought).
    
    ### 2. Risk Management
    *   Always use a **Stop-loss (5-7%)** to protect capital against market crashes ("Black Swan" events).
    *   Do not rely solely on the AI model; use it as a confirmation tool.
    
    ### 3. Conclusion
    The **BiLSTM** model demonstrates superior performance ($R^2 > 0.99$) in capturing non-linear price patterns compared to traditional models.
    """)
