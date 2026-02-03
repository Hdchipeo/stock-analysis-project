import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# RANDOM SEED - Đảm bảo kết quả lặp lại được (Reproducibility)
# ============================================================
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)


def evaluate_returns_metrics(y_true_returns, y_pred_returns, model_name):
    """
    Đánh giá metrics cho dự báo Log Returns
    
    Tại sao khác với dự báo giá:
    - R² trên returns thường THẤP (0.01-0.15) nhưng đây là BÌN THƯỜNG với dữ liệu tài chính
    - Direction Accuracy > 55% đã là có giá trị thương mại
    - RMSE/MAE đo trên returns (scale nhỏ hơn nhiều so với giá)
    
    Metrics:
    - RMSE: Root Mean Squared Error - đo sai số trung bình
    - MAE: Mean Absolute Error - sai số tuyệt đối trung bình
    - R²: Coefficient of Determination - tỷ lệ variance được giải thích
    - Direction Accuracy: % dự đoán đúng chiều hướng (lên/xuống)
      → Đây là metric QUAN TRỌNG NHẤT cho trading!
    """
    rmse = np.sqrt(mean_squared_error(y_true_returns, y_pred_returns))
    mae = mean_absolute_error(y_true_returns, y_pred_returns)
    r2 = r2_score(y_true_returns, y_pred_returns)
    
    # Direction Accuracy - Tỷ lệ dự đoán đúng chiều hướng
    # Nếu cả actual và predicted cùng dấu (+/+) hoặc (-/-) → Đúng chiều hướng
    correct_direction = np.sum(np.sign(y_true_returns) == np.sign(y_pred_returns))
    direction_accuracy = (correct_direction / len(y_true_returns)) * 100
    
    print(f"\n{'─'*70}")
    print(f"METRICS - {model_name} (Log Returns Prediction)")
    print(f"{'─'*70}")
    print(f"RMSE:                  {rmse:.6f}")
    print(f"MAE:                   {mae:.6f}")
    print(f"R²:                    {r2:.4f}")
    print(f"Direction Accuracy:    {direction_accuracy:.2f}%")
    print(f"{'─'*70}")
    
    # Nhận xét
    print(f"\n📊 NHẬN XÉT:")
    if r2 < 0:
        print(f"   ⚠ R² âm: Mô hình KÉME hơn cả việc dự đoán mean constant")
    elif r2 < 0.05:
        print(f"   ℹ R² thấp ({r2:.4f}) nhưng BÌN THƯỜNG với dữ liệu tài chính")
    else:
        print(f"   ✓ R² = {r2:.4f}: Mô hình có khả năng giải thích {r2*100:.2f}% variance")
    
    if direction_accuracy > 55:
        print(f"   ✓ Direction Accuracy {direction_accuracy:.2f}% > 55%: CÓ giá trị thương mại")
    elif direction_accuracy > 50:
        print(f"   ~ Direction Accuracy {direction_accuracy:.2f}%: Hơi tốt hơn ngẫu nhiên")
    else:
        print(f"   ✗ Direction Accuracy {direction_accuracy:.2f}% ≤ 50%: KHÔNG tốt hơn ngẫu nhiên")
    
    print()
    
    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Direction_Accuracy": direction_accuracy
    }


def create_lstm_sequences(data, features, target, lookback=10):
    """
    Tạo sequences cho LSTM
    
    LSTM cần input dạng 3D: [samples, timesteps, features]
    - samples: Số điểm dữ liệu
    - timesteps: Số bước thời gian nhìn lại (lookback window)
    - features: Số lượng features
    
    Tham số:
    - data: DataFrame chứa features và target
    - features: List tên các features
    - target: Tên cột target
    - lookback: Số ngày nhìn lại (default=10, có thể điều chỉnh từ PACF)
    
    Ví dụ:
    - Lookback=10: Dùng dữ liệu 10 ngày trước để dự báo ngày hôm nay
    - Với 1000 ngày data → có 990 samples (vì 10 ngày đầu không đủ lookback)
    """
    X, y = [], []
    
    for i in range(lookback, len(data)):
        # Lấy lookback ngày trước đó
        X.append(data[features].iloc[i-lookback:i].values)
        # Target là giá trị tại thời điểm i
        y.append(data[target].iloc[i])
    
    return np.array(X), np.array(y)


def run_modeling(train_file="train_data.csv", test_file="test_data.csv"):
    """
    Huấn luyện và đánh giá mô hình dự báo Log Returns
    
    THAY ĐỔI QUAN TRỌNG so với version cũ:
    - Target: Log_Returns thay vì Close price
    - Không dùng inverse scaling (vì returns không cần scale back)
    - Thêm Direction Accuracy metric
    - Implement BiLSTM (đã bỏ comment)
    - Phân tích residuals với Ljung-Box test
    
    Mô hình:
    1. Linear Regression (Baseline)
    2. XGBoost (Tree-based, capture non-linearity)
    3. BiLSTM (Deep Learning for sequential data)
    """
    print("\n" + "="*80)
    print(" " * 25 + "MODELING - LOG RETURNS PREDICTION")
    print("="*80 + "\n")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_processed_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results", "figures")
    metrics_path = os.path.join(base_dir, "results", "metrics.csv")

    if train_file == "train_data.csv":
        train_file = os.path.join(data_processed_dir, "train_data.csv")
    if test_file == "test_data.csv":
        test_file = os.path.join(data_processed_dir, "test_data.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Lỗi: Không tìm thấy file dữ liệu train/test.")
        return

    # 1. Load Data
    print("📂 Loading data...")
    train_df = pd.read_csv(train_file, index_col='Date', parse_dates=True)
    test_df = pd.read_csv(test_file, index_col='Date', parse_dates=True)
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test:  {len(test_df)} samples\n")

    # === THAY ĐỔI QUAN TRỌNG: Target là Log_Returns ===
    target = 'Log_Returns'
    
    # Features: Loại bỏ các cột không dùng
    exclude_cols = [
        'Log_Returns',  # Target
        'Close',  # Giá tuyệt đối (không dùng nữa)
        'Outlier',  # Flag
        'Price_Direction',  # Target cho classification (dùng riêng)
        'Open', 'High', 'Low', 'Volume'  # Raw values (đã có derived features)
    ]
    features = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"🎯 Target: {target}")
    print(f"📊 Features ({len(features)}): {features[:5]}... (showing first 5)\n")
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    results = []
    predictions = pd.DataFrame(index=X_test.index)
    predictions['Actual_Returns'] = y_test

    # ========================================================================
    # 2. LINEAR REGRESSION (Baseline)
    # ========================================================================
    print("\n" + "█"*70)
    print("MODEL 1: LINEAR REGRESSION (Baseline)")
    print("█"*70)
    print("\nĐây là baseline model đơn giản nhất.")
    print("Giả định: Mối quan hệ tuyến tính giữa features và returns")
    print("Huấn luyện...\n")
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    predictions['LR_Returns'] = y_pred_lr
    
    lr_metrics = evaluate_returns_metrics(y_test, y_pred_lr, "Linear Regression")
    results.append(lr_metrics)

    # ========================================================================
    # 3. XGBOOST
    # ========================================================================
    print("\n" + "█"*70)
    print("MODEL 2: XGBOOST (Gradient Boosting)")
    print("█"*70)
    print("\nXGBoost là ensemble of decision trees.")
    print("Ưu điểm: Capture non-linearity, feature importance, robust to outliers")
    print("Tham số:")
    print("  - n_estimators=1000: Số cây quyết định")
    print("  - learning_rate=0.01: Tốc độ học (thấp = học chậm nhưng ổn định)")
    print("  - max_depth=5: Độ sâu tối đa của mỗi cây (tránh overfitting)")
    print("Huấn luyện...\n")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred_xgb = xgb_model.predict(X_test)
    predictions['XGBoost_Returns'] = y_pred_xgb
    
    xgb_metrics = evaluate_returns_metrics(y_test, y_pred_xgb, "XGBoost")
    results.append(xgb_metrics)
    
    # Feature Importance
    print("\n📊 Vẽ Feature Importance...")
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=15,
                        title='XGBoost Feature Importance (Top 15)\nĐộ quan trọng của từng feature trong việc dự báo Log Returns')
    plt.xlabel('F score (số lần feature được sử dụng để split)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance_returns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   → Đã lưu: feature_importance_returns.png")
    
    print("\n📝 NHẬN XÉT Feature Importance:")
    feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (feat, score) in enumerate(top_features, 1):
        # Map feature index to name
        feat_idx = int(feat[1:]) if feat.startswith('f') else -1
        feat_name = features[feat_idx] if 0 <= feat_idx < len(features) else feat
        print(f"   {i}. {feat_name}: {score:.0f} lần sử dụng")
    
    # ========================================================================
    # 4. BiLSTM (Bidirectional LSTM)
    # ========================================================================
    print("\n" + "█"*70)
    print("MODEL 3: BiLSTM (Bidirectional Long Short-Term Memory)")
    print("█"*70)
    print("\nBiLSTM là mô hình Deep Learning cho dữ liệu chuỗi thời gian.")
    print("Ưu điểm:")
    print("  - Capture long-term dependencies (phụ thuộc dài hạn)")
    print("  - Bidirectional: học từ cả quá khứ VÀ tương lai")
    print("  - Tự động học features từ sequences")
    print("\nCấu trúc:")
    print("  - Layer 1: BiLSTM(64 units) + Dropout(0.2)")
    print("  - Layer 2: BiLSTM(32 units) + Dropout(0.2)")
    print("  - Layer 3: Dense(16, relu)")
    print("  - Output: Dense(1) - dự báo Log Return")
    print("\nChuẩn bị sequences (lookback=10 ngày)...\n")
    
    # Tạo sequences
    lookback = 10  # Có thể điều chỉnh dựa trên PACF analysis
    
    X_train_lstm, y_train_lstm = create_lstm_sequences(
        train_df, features, target, lookback
    )
    X_test_lstm, y_test_lstm = create_lstm_sequences(
        test_df, features, target, lookback
    )
    
    print(f"   LSTM Train shape: {X_train_lstm.shape}")
    print(f"   LSTM Test shape:  {X_test_lstm.shape}")
    print(f"   Format: (samples, timesteps={lookback}, features={len(features)})\n")
    
    # Build model
    print("Xây dựng BiLSTM model...")
    lstm_model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), 
                     input_shape=(lookback, len(features))),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Output: Log Returns
    ])
    
    lstm_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print(lstm_model.summary())
    
    # Train với EarlyStopping
    print("\nHuấn luyện BiLSTM (có thể mất vài phút)...")
    es = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    history = lstm_model.fit(
        X_train_lstm, y_train_lstm,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[es],
        shuffle=False,  # Đảm bảo reproducibility
        verbose=1
    )
    
    # Predict
    y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    
    # Add to predictions (align indices)
    predictions['BiLSTM_Returns'] = np.nan
    predictions.iloc[lookback:lookback+len(y_pred_lstm), 
                    predictions.columns.get_loc('BiLSTM_Returns')] = y_pred_lstm
    
    lstm_metrics = evaluate_returns_metrics(y_test_lstm, y_pred_lstm, "BiLSTM")
    results.append(lstm_metrics)
    
    # Plot training history
    print("\n📊 Vẽ Training History...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('BiLSTM Training History - Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('MSE Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_title('BiLSTM Training History - MAE', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('MAE', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lstm_training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   → Đã lưu: lstm_training_history.png")

    # ========================================================================
    # 5. COMPARE MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("SO SÁNH MÔ HÌNH")
    print("="*80 + "\n")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))
    results_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Đã lưu metrics vào: {metrics_path}")

    # ========================================================================
    # 6. VISUALIZATION: Actual vs Predicted Returns
    # ========================================================================
    print("\n📊 Vẽ biểu đồ so sánh Actual vs Predicted Returns...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Zoom last 100 days
    subset = predictions.iloc[-100:]
    
    # Plot 1: Linear Regression
    axes[0].plot(subset.index, subset['Actual_Returns'], label='Actual Returns', 
                color='black', linewidth=2, alpha=0.7)
    axes[0].plot(subset.index, subset['LR_Returns'], label='LR Prediction', 
                color='#E63946', linestyle='--', linewidth=1.5)
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_title('Linear Regression: Actual vs Predicted Log Returns (Last 100 Days)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Log Returns', fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: XGBoost
    axes[1].plot(subset.index, subset['Actual_Returns'], label='Actual Returns', 
                color='black', linewidth=2, alpha=0.7)
    axes[1].plot(subset.index, subset['XGBoost_Returns'], label='XGBoost Prediction', 
                color='#2A9D8F', linestyle='--', linewidth=1.5)
    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_title('XGBoost: Actual vs Predicted Log Returns (Last 100 Days)', 
                     fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Log Returns', fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: BiLSTM
    subset_lstm = subset.dropna(subset=['BiLSTM_Returns'])
    axes[2].plot(subset_lstm.index, subset_lstm['Actual_Returns'], label='Actual Returns', 
                color='black', linewidth=2, alpha=0.7)
    axes[2].plot(subset_lstm.index, subset_lstm['BiLSTM_Returns'], label='BiLSTM Prediction', 
                color='#F4A261', linestyle='--', linewidth=1.5)
    axes[2].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_title('BiLSTM: Actual vs Predicted Log Returns (Last 100 Days)', 
                     fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Date', fontweight='bold')
    axes[2].set_ylabel('Log Returns', fontweight='bold')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison_returns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   → Đã lưu: model_comparison_returns.png")
    
    # ========================================================================
    # 7. RESIDUALS ANALYSIS
    # ========================================================================
    print("\n" + "█"*70)
    print("PHÂN TÍCH RESIDUALS (White Noise Test)")
    print("█"*70 + "\n")
    
    from statistical_tests import StatisticalTests
    tester = StatisticalTests(results_dir=results_dir)
    
    # Test residuals for each model
    for model_name, pred_col in [('Linear Regression', 'LR_Returns'),
                                   ('XGBoost', 'XGBoost_Returns'),
                                   ('BiLSTM', 'BiLSTM_Returns')]:
        residuals = predictions['Actual_Returns'] - predictions[pred_col]
        residuals = residuals.dropna()
        
        if len(residuals) > 20:
            tester.ljung_box_test(residuals, lags=10, name=model_name)
    
    # ========================================================================
    # 8. SAVE PREDICTIONS
    # ========================================================================
    predictions_path = os.path.join(base_dir, "results", "predictions_returns.csv")
    predictions.to_csv(predictions_path)
    print(f"\n✓ Đã lưu predictions vào: {predictions_path}")

    print("\n" + "="*80)
    print(" " * 30 + "MODELING HOÀN THÀNH")
    print("="*80 + "\n")
    
    print("📝 TÓM TẮT:")
    print("   - Đã chuyển từ dự báo GIÁ TUYỆT ĐỐI sang DỰ BÁO LOG RETURNS")
    print("   - R² thấp (< 0.1) là BÌN THƯỜNG với dữ liệu tài chính")
    print("   - Direction Accuracy > 55% = CÓ giá trị thương mại")
    print("   - Residuals analysis kiểm tra xem mô hình có bỏ sót thông tin không")
    print()


if __name__ == "__main__":
    run_modeling()
