import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping
import os

def evaluate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"--- {model_name} Metrics ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("-" * 30)
    
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

def run_modeling(train_file="train_data.csv", test_file="test_data.csv"):
    """
    Huấn luyện và đánh giá các mô hình: Linear Regression, XGBoost, BiLSTM.
    """
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_processed_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results", "figures")
    metrics_path = os.path.join(base_dir, "results", "metrics.csv")
    scaling_params_path = os.path.join(data_processed_dir, "scaling_params.json")

    # Override defaults with correct paths if not specified or if defaults are filenames
    if train_file == "train_data.csv":
         train_file = os.path.join(data_processed_dir, "train_data.csv")
    if test_file == "test_data.csv":
         test_file = os.path.join(data_processed_dir, "test_data.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Lỗi: Không tìm thấy file dữ liệu train/test.")
        return

    # Load Scaling Params
    import json
    scaling_params = None
    if os.path.exists(scaling_params_path):
        with open(scaling_params_path, "r") as f:
            scaling_params = json.load(f)
        print(f"Loaded scaling params: {scaling_params}")
    else:
        print("Warning: Scaling params not found. Predictions will be normalized.")

    # Helper for inversion
    def inverse_scale(val):
        if scaling_params:
            min_val = scaling_params["Close_min"]
            max_val = scaling_params["Close_max"]
            return val * (max_val - min_val) + min_val
        return val

    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_csv(train_file, index_col='Date', parse_dates=True)
    test_df = pd.read_csv(test_file, index_col='Date', parse_dates=True)

    # Xác định Features (X) và Target (y)
    # Target là 'Close'. Features là tất cả trừ 'Close', 'Outlier'.
    
    features = [c for c in train_df.columns if c not in ['Close', 'Outlier']]
    target = 'Close'
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Inverse Transform Target for Validation/Reporting
    y_test_original = inverse_scale(y_test)

    results = []
    predictions = pd.DataFrame(index=X_test.index)
    predictions['Actual'] = y_test_original


    # 2. Baseline Model: Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr_scaled = lr_model.predict(X_test)
    y_pred_lr = inverse_scale(y_pred_lr_scaled)
    predictions['LinearRegression'] = y_pred_lr
    results.append(evaluate_metrics(y_test_original, y_pred_lr, "Linear Regression"))

    # 3. Machine Learning Model: XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False) # Train on scaled data
    y_pred_xgb_scaled = xgb_model.predict(X_test)
    y_pred_xgb = inverse_scale(y_pred_xgb_scaled)
    predictions['XGBoost'] = y_pred_xgb
    results.append(evaluate_metrics(y_test_original, y_pred_xgb, "XGBoost"))
    
    # Feature Importance (XGBoost)
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close() # Đóng figure để tránh conflict
    print("   - Saved 'feature_importance.png'")

    # 4. BiLSTM Disabled (Due to environment constraints)
    # print("\nTraining BiLSTM...")
    # ... code removed ...
    predictions['BiLSTM'] = 0 # Placeholder
    # results.append(...) # Disabled

    # 5. Summarize Results & Save
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison Summary ---")
    print(results_df.to_markdown(index=False))
    results_df.to_csv(metrics_path, index=False)
    print(f"   - Saved metrics to '{metrics_path}'")

    # 6. Visualization: Actual vs Predicted (Zoom last 100 points)
    print("\nDrawing Model Comparison Chart...")
    plt.figure(figsize=(14, 7))
    subset = predictions.iloc[-100:] # Lấy 100 điểm cuối
    
    plt.plot(subset.index, subset['Actual'], label='Actual', color='black', linewidth=2)
    plt.plot(subset.index, subset['LinearRegression'], label='Linear Regression', linestyle='--')
    plt.plot(subset.index, subset['XGBoost'], label='XGBoost', linestyle='-.')
    plt.plot(subset.index, subset['BiLSTM'], label='BiLSTM', linestyle=':')
    
    plt.title('Model Comparison: Actual vs Predicted (Last 100 Days)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
    plt.close()
    print("   - Saved 'model_comparison.png'")
    
    # Save predictions for Dashboard
    predictions.to_csv(os.path.join(base_dir, "results", "predictions.csv"))
    print("   - Saved 'predictions.csv'")

    print("Modeling Completed.")

if __name__ == "__main__":
    run_modeling()
