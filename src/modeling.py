import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
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

    # Override defaults with correct paths if not specified or if defaults are filenames
    if train_file == "train_data.csv":
         train_file = os.path.join(data_processed_dir, "train_data.csv")
    if test_file == "test_data.csv":
         test_file = os.path.join(data_processed_dir, "test_data.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Lỗi: Không tìm thấy file dữ liệu train/test.")
        return

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

    results = []
    predictions = pd.DataFrame(index=X_test.index)
    predictions['Actual'] = y_test

    # 2. Baseline Model: Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    predictions['LinearRegression'] = y_pred_lr
    results.append(evaluate_metrics(y_test, y_pred_lr, "Linear Regression"))

    # 3. Machine Learning Model: XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    predictions['XGBoost'] = y_pred_xgb
    results.append(evaluate_metrics(y_test, y_pred_xgb, "XGBoost"))
    
    # Feature Importance (XGBoost)
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close() # Đóng figure để tránh conflict
    print("   - Saved 'feature_importance.png'")

    # 4. Deep Learning Model: BiLSTM
    print("\nTraining BiLSTM...")
    # Reshape input for LSTM [Samples, Timesteps, Features]
    # Ở đây Timesteps = 1 vì dữ liệu đã load dạng 2D (Features làm cột)
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm_model = Sequential()
    # Upgrade to Bidirectional LSTM
    lstm_model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(1, X_train.shape[1])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32, 
                   validation_data=(X_test_lstm, y_test), 
                   callbacks=[early_stop], verbose=0) # Tắt verbose cho gọn
    
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    predictions['BiLSTM'] = y_pred_lstm.flatten()
    results.append(evaluate_metrics(y_test, y_pred_lstm.flatten(), "BiLSTM"))

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
