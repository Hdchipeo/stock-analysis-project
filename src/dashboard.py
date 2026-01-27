import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def create_dashboard(ticker="AAPL"):
    """
    Tổng hợp kết quả thành một Dashboard duy nhất.
    """
    # Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    
    metrics_path = os.path.join(results_dir, "metrics.csv")
    predictions_path = os.path.join(results_dir, "predictions.csv")
    
    output_path = os.path.join(figures_dir, "dashboard.png")

    if not os.path.exists(metrics_path) or not os.path.exists(predictions_path):
        print("Lỗi: Không tìm thấy file metrics.csv hoặc predictions.csv. Hãy chạy main.py hoặc modeling.py trước.")
        return

    # Load Data
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    
    # Tìm mô hình tốt nhất (dựa trên RMSE thấp nhất)
    best_model_row = metrics.sort_values(by="RMSE").iloc[0]
    best_model_name = best_model_row["Model"]
    # Handle potential name mismatch (e.g., "Linear Regression" vs "LinearRegression")
    best_model_col = best_model_name.replace(" ", "") if "Linear" in best_model_name else best_model_name

    # Setup Figure Layout
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Top Left: Model Metrics Table & Summary
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    # Title
    ax1.text(0.0, 0.9, f"Stock Analysis Dashboard: {ticker}", fontsize=24, fontweight='bold', color='darkblue')
    ax1.text(0.0, 0.8, "Performance Metrics (Best Model Highlighted)", fontsize=14, color='gray')
    
    # Table Data
    table_data = [] # Header later
    cell_colors = []
    
    for _, row in metrics.iterrows():
        is_best = row["Model"] == best_model_name
        color = "#e6fffa" if is_best else "white"
        cell_colors.append([color] * 4)
        table_data.append([
            row["Model"], 
            f"{row['RMSE']:.4f}", 
            f"{row['R2']:.4f}", 
            f"{row['MAPE']:.2f}%"
        ])
        
    cols = ["Model", "RMSE", "R2", "MAPE"]
    
    table = ax1.table(cellText=table_data, colLabels=cols, cellColours=cell_colors, 
                      loc='center', cellLoc='center', bbox=[0.05, 0.4, 0.9, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Best Model Summary Text
    summary_text = (
        f"★ Best Model: {best_model_name}\n"
        f"• Lowest RMSE: {best_model_row['RMSE']:.4f}\n"
        f"• Highest R2: {best_model_row['R2']:.4f}\n\n"
        "Recommendation:\n"
        "Use this model for trend prediction alongside RSI/MACD signals."
    )
    ax1.text(0.05, 0.1, summary_text, fontsize=12, bbox=dict(facecolor='#f0f0f0', alpha=0.5, pad=10))

    # 2. Top Right: Actual vs Predicted Chart (Last 100 Days)
    ax2 = fig.add_subplot(gs[0, 1])
    subset = preds.iloc[-100:]
    
    ax2.plot(subset.index, subset['Actual'], label='Actual Price', color='black', linewidth=2)
    ax2.plot(subset.index, subset[best_model_col], label=f'Predicted ({best_model_name})', color='green', linestyle='--')
    
    ax2.set_title(f"Price Prediction: Actual vs {best_model_name} (Last 100 Days)", fontsize=16)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Normalized Price")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Calculate simple signals (Buy if Predicted > Actual + Threshold?) - Just visual for now
    # Or Buy if Trend is Up. Let's mark crossover points simply.
    
    # 3. Bottom Left: Feature Importance (Placeholder image loading if XGBoost exists)
    ax3 = fig.add_subplot(gs[1, 0])
    feature_img_path = os.path.join(figures_dir, "feature_importance.png")
    
    if os.path.exists(feature_img_path):
        img = plt.imread(feature_img_path)
        ax3.imshow(img)
        ax3.axis('off')
        ax3.set_title("XGBoost Feature Importance", fontsize=16)
    else:
        ax3.text(0.5, 0.5, "Feature Importance Image Not Found", ha='center')

    # 4. Bottom Right: Trend / Outlier Context (Load Outliers or Trend image)
    ax4 = fig.add_subplot(gs[1, 1])
    trend_img_path = os.path.join(figures_dir, "outliers.png") # Or trend_analysis.png
    
    if os.path.exists(trend_img_path):
        img = plt.imread(trend_img_path)
        ax4.imshow(img)
        ax4.axis('off')
        ax4.set_title("Market Anomalies & Outliers", fontsize=16)
    else:
        ax4.text(0.5, 0.5, "Outlier Image Not Found", ha='center')

    # Save Dashboard
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Dashboard saved to: {output_path}")

if __name__ == "__main__":
    create_dashboard()
