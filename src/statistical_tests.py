import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import os
import warnings
warnings.filterwarnings('ignore')

class StatisticalTests:
    """
    Module kiểm định thống kê cho chuỗi thời gian tài chính.
    
    Mục đích:
    - Kiểm định tính dừng (Stationarity) của chuỗi dữ liệu
    - Kiểm định mối quan hệ nhân quả giữa các biến (Granger Causality)
    - Xác định số lượng lags tối ưu thông qua ACF/PACF
    - Kiểm tra Residuals có phải White Noise không
    """
    
    def __init__(self, results_dir=None):
        if results_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.results_dir = os.path.join(base_dir, "results", "figures")
        else:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
    
    def adf_test(self, series, name="Series"):
        """
        Augmented Dickey-Fuller Test - Kiểm định tính dừng (Stationarity Test)
        
        Giả thuyết:
        - H0 (Null Hypothesis): Chuỗi có unit root (non-stationary) - KHÔNG dừng
        - H1 (Alternative): Chuỗi là stationary - Dừng
        
        Tham số:
        - series: Chuỗi dữ liệu cần kiểm định (pandas Series)
        - name: Tên của chuỗi để hiển thị trong báo cáo
        
        Ý nghĩa:
        - p-value < 0.05: Bác bỏ H0, chuỗi là DỪNG ✓
        - p-value >= 0.05: Không bác bỏ H0, chuỗi KHÔNG DỪNG ✗
        
        Tại sao quan trọng:
        - Chuỗi không dừng (như giá cổ phiếu) có mean/variance thay đổi theo thời gian
        - Hồi quy trên chuỗi không dừng dẫn đến "spurious regression" (hồi quy giả mạo)
        - Log Returns thường là dừng → phù hợp cho mô hình ML
        """
        series_clean = series.dropna()
        
        try:
            result = adfuller(series_clean, autolag='AIC')
            
            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            print(f"\n{'='*70}")
            print(f"ADF TEST - {name}")
            print(f"{'='*70}")
            print(f"ADF Statistic:        {adf_statistic:.6f}")
            print(f"P-value:              {p_value:.6f}")
            print(f"\nCritical Values:")
            for key, value in critical_values.items():
                print(f"  {key:>4s}: {value:.4f}")
            
            # Kết luận
            if p_value < 0.05:
                conclusion = f"✓ ChuỖi '{name}' là STATIONARY (Dừng) - p-value = {p_value:.6f} < 0.05"
                is_stationary = True
            else:
                conclusion = f"✗ Chuỗi '{name}' là NON-STATIONARY (Không dừng) - p-value = {p_value:.6f} >= 0.05"
                is_stationary = False
            
            print(f"\n{conclusion}")
            print(f"{'='*70}\n")
            
            # Vẽ biểu đồ Rolling Mean và Rolling Std để minh họa tính dừng
            self._plot_rolling_statistics(series_clean, name, is_stationary)
            
            return {
                'name': name,
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'conclusion': conclusion
            }
        except Exception as e:
            print(f"Lỗi khi thực hiện ADF test cho {name}: {e}")
            return None
    
    def _plot_rolling_statistics(self, series, name, is_stationary):
        """
        Vẽ Rolling Mean và Rolling Std để minh họa tính dừng
        
        Chuỗi dừng: Rolling mean và std ổn định theo thời gian
        Chuỗi không dừng: Rolling mean/std thay đổi liên tục
        """
        plt.figure(figsize=(14, 8))
        
        # Original Series
        plt.subplot(3, 1, 1)
        plt.plot(series, label='Original Series', color='blue', alpha=0.7)
        plt.title(f'{name} - Original Series', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rolling Mean
        rolling_mean = series.rolling(window=30).mean()
        plt.subplot(3, 1, 2)
        plt.plot(rolling_mean, label='Rolling Mean (30 days)', color='red', linewidth=2)
        plt.title('Rolling Mean - Kiểm tra xu hướng thay đổi', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rolling Std
        rolling_std = series.rolling(window=30).std()
        plt.subplot(3, 1, 3)
        plt.plot(rolling_std, label='Rolling Std (30 days)', color='green', linewidth=2)
        plt.title('Rolling Standard Deviation - Kiểm tra biến động thay đổi', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        status = "STATIONARY ✓" if is_stationary else "NON-STATIONARY ✗"
        plt.suptitle(f'ADF Test: {name} - {status}', fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        filename = f"adf_test_{name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   → Đã lưu biểu đồ: {filename}")
    
    def granger_causality_test(self, data, target_col, cause_col, max_lag=5):
        """
        Granger Causality Test - Kiểm định nhân quả Granger
        
        Mục đích:
        Kiểm tra liệu biến X (cause_col) có "gây ra" (predict) biến Y (target_col) hay không
        
        Giả thuyết:
        - H0: X KHÔNG Granger-cause Y (X không giúp dự báo Y)
        - H1: X Granger-cause Y (X có khả năng dự báo Y)
        
        Tham số:
        - data: DataFrame chứa cả hai cột
        - target_col: Biến mục tiêu (Y) - ví dụ: 'Log_Returns'
        - cause_col: Biến nguyên nhân (X) - ví dụ: 'Volume_Change'
        - max_lag: Số lags tối đa để kiểm tra (default=5)
        
        Ý nghĩa tài chính:
        - Nếu Volume Granger-cause Returns: Khối lượng giao dịch có thể dự báo biến động giá
        - Điều này hỗ trợ giả thuyết: "Volume leads Price" trong technical analysis
        
        Kết quả:
        - p-value < 0.05 tại lag k: Volume tại t-k có thể dự báo Returns tại t
        """
        print(f"\n{'='*70}")
        print(f"GRANGER CAUSALITY TEST")
        print(f"Nguyên nhân (X): {cause_col} → Kết quả (Y): {target_col}")
        print(f"{'='*70}")
        
        # Chuẩn bị dữ liệu: Hai cột [target, cause]
        test_data = data[[target_col, cause_col]].dropna()
        
        if len(test_data) < max_lag + 10:
            print(f"Không đủ dữ liệu để thực hiện test (cần ít nhất {max_lag + 10} điểm)")
            return None
        
        try:
            # Thực hiện test
            gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            
            # Tổng hợp kết quả
            results_summary = []
            
            print(f"\n{'Lag':<6} {'F-stat':<12} {'P-value':<12} {'Kết luận':<30}")
            print("-" * 70)
            
            for lag in range(1, max_lag + 1):
                # Lấy F-test result
                f_test = gc_result[lag][0]['ssr_ftest']
                f_stat = f_test[0]
                p_value = f_test[1]
                
                # Kết luận
                if p_value < 0.05:
                    conclusion = f"✓ {cause_col} CÓ Granger-cause {target_col}"
                    has_causality = True
                else:
                    conclusion = f"✗ KHÔNG có Granger causality"
                    has_causality = False
                
                print(f"{lag:<6} {f_stat:<12.4f} {p_value:<12.6f} {conclusion}")
                
                results_summary.append({
                    'lag': lag,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'has_causality': has_causality
                })
            
            print(f"{'='*70}\n")
            
            # Vẽ biểu đồ
            self._plot_granger_results(results_summary, cause_col, target_col)
            
            # Nhận xét tổng quát
            significant_lags = [r['lag'] for r in results_summary if r['has_causality']]
            if significant_lags:
                print(f"📊 NHẬN XÉT: {cause_col} có khả năng dự báo {target_col} tại các lag: {significant_lags}")
                print(f"   → Ý nghĩa: Nên đưa {cause_col} vào mô hình với lag features {significant_lags}")
            else:
                print(f"📊 NHẬN XÉT: KHÔNG tìm thấy bằng chứng thống kê cho mối quan hệ nhân quả")
                print(f"   → Ý nghĩa: {cause_col} có thể không hữu ích cho việc dự báo {target_col}")
            print()
            
            return results_summary
            
        except Exception as e:
            print(f"Lỗi khi thực hiện Granger Causality Test: {e}")
            return None
    
    def _plot_granger_results(self, results_summary, cause_col, target_col):
        """Vẽ biểu đồ kết quả Granger Causality Test"""
        lags = [r['lag'] for r in results_summary]
        p_values = [r['p_value'] for r in results_summary]
        
        plt.figure(figsize=(10, 6))
        
        # Vẽ p-values
        plt.bar(lags, p_values, color=['green' if p < 0.05 else 'red' for p in p_values], 
                alpha=0.7, edgecolor='black')
        
        # Đường significance level
        plt.axhline(y=0.05, color='blue', linestyle='--', linewidth=2, 
                   label='Significance Level (α=0.05)')
        
        plt.xlabel('Lag', fontsize=12, fontweight='bold')
        plt.ylabel('P-value', fontsize=12, fontweight='bold')
        plt.title(f'Granger Causality Test: {cause_col} → {target_col}\n' + 
                 f'(Green = Có nhân quả, Red = Không có nhân quả)', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(lags)
        
        plt.tight_layout()
        filename = f"granger_causality_{cause_col.lower()}_{target_col.lower()}.png"
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   → Đã lưu biểu đồ: {filename}")
    
    def acf_pacf_analysis(self, series, name="Series", lags=40):
        """
        ACF/PACF Analysis - Phân tích tự tương quan
        
        Mục đích:
        Xác định số lượng lags tối ưu cho mô hình ARIMA và lag features
        
        ACF (Autocorrelation Function):
        - Đo tương quan giữa y_t và y_{t-k}
        - Giúp xác định MA order (Moving Average)
        
        PACF (Partial Autocorrelation Function):
        - Đo tương quan giữa y_t và y_{t-k} SAU KHI loại bỏ ảnh hưởng của y_{t-1},...,y_{t-k+1}
        - Giúp xác định AR order (Autoregressive)
        
        Tham số:
        - series: Chuỗi dữ liệu (pandas Series)
        - name: Tên chuỗi
        - lags: Số lượng lags để hiển thị (default=40)
        
        Ý nghĩa:
        - Nếu PACF significant đến lag 5 → Sử dụng 5 lag features trong mô hình
        - Nếu ACF decay chậm → Chuỗi có thể không dừng
        """
        print(f"\n{'='*70}")
        print(f"ACF/PACF ANALYSIS - {name}")
        print(f"{'='*70}")
        
        series_clean = series.dropna()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # ACF Plot
        plot_acf(series_clean, lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'ACF (Autocorrelation Function) - {name}\n' + 
                         'Đo tương quan giữa y_t và y_{t-k}', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Lag', fontsize=11)
        axes[0].set_ylabel('Correlation', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # PACF Plot
        plot_pacf(series_clean, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
        axes[1].set_title(f'PACF (Partial Autocorrelation Function) - {name}\n' + 
                         'Đo tương quan SAU KHI loại bỏ ảnh hưởng của các lag trung gian', 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Lag', fontsize=11)
        axes[1].set_ylabel('Partial Correlation', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"acf_pacf_{name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   → Đã lưu biểu đồ: {filename}")
        
        # Phân tích và đề xuất số lags
        from statsmodels.tsa.stattools import acf, pacf
        acf_values = acf(series_clean, nlags=lags, fft=False)
        pacf_values = pacf(series_clean, nlags=lags, method='ywm')
        
        # Tìm significant lags trong PACF (|value| > 1.96/sqrt(n))
        n = len(series_clean)
        threshold = 1.96 / np.sqrt(n)
        
        significant_lags_pacf = [i for i in range(1, lags+1) if abs(pacf_values[i]) > threshold]
        
        print(f"\n📊 PHÂN TÍCH KẾT QUẢ:")
        print(f"   - Significant lags trong PACF: {significant_lags_pacf[:10]}")  # Top 10
        print(f"   - Ngưỡng significance: ±{threshold:.4f}")
        
        if len(significant_lags_pacf) > 0:
            optimal_lags = significant_lags_pacf[:5]  # Lấy top 5
            print(f"\n   💡 ĐỀ XUẤT: Sử dụng {len(optimal_lags)} lag features: {optimal_lags}")
            print(f"      Lý do: Các lag này có tương quan riêng phần (PACF) vượt ngưỡng significance")
        else:
            print(f"\n   💡 ĐỀ XUẤT: Chuỗi có thể là white noise hoặc số lags quá lớn")
        
        print(f"{'='*70}\n")
        
        return {
            'name': name,
            'significant_lags': significant_lags_pacf,
            'optimal_lags': optimal_lags if len(significant_lags_pacf) > 0 else [1, 2, 3],
            'acf_values': acf_values,
            'pacf_values': pacf_values
        }
    
    def ljung_box_test(self, residuals, lags=10, name="Model"):
        """
        Ljung-Box Test - Kiểm định White Noise cho Residuals
        
        Mục đích:
        Kiểm tra xem residuals (phần dư) của mô hình có phải là white noise không
        
        Giả thuyết:
        - H0: Residuals là white noise (KHÔNG có autocorrelation)
        - H1: Residuals CÓ autocorrelation
        
        Tham số:
        - residuals: Phần dư của mô hình (y_true - y_pred)
        - lags: Số lags để kiểm tra
        - name: Tên mô hình
        
        Ý nghĩa:
        - p-value > 0.05: Residuals là white noise ✓
          → Mô hình đã trích xuất HẾT thông tin từ dữ liệu
        - p-value < 0.05: Residuals CÓ cấu trúc tự tương quan ✗
          → Mô hình còn bỏ sót thông tin, cần cải thiện
        
        Tại sao quan trọng:
        - Residuals có autocorrelation nghĩa là mô hình chưa tối ưu
        - Có thể cần thêm features hoặc thay đổi cấu trúc mô hình
        """
        print(f"\n{'='*70}")
        print(f"LJUNG-BOX TEST (WHITE NOISE TEST) - {name}")
        print(f"{'='*70}")
        
        residuals_clean = residuals.dropna()
        
        if len(residuals_clean) < lags + 10:
            print("Không đủ dữ liệu để thực hiện Ljung-Box test")
            return None
        
        try:
            # Thực hiện test
            lb_result = acorr_ljungbox(residuals_clean, lags=lags, return_df=True)
            
            print(f"\n{'Lag':<6} {'LB Statistic':<15} {'P-value':<12} {'Kết luận':<30}")
            print("-" * 70)
            
            all_white_noise = True
            
            for i, row in lb_result.iterrows():
                lag = i + 1
                lb_stat = row['lb_stat']
                p_val = row['lb_pvalue']
                
                if p_val > 0.05:
                    conclusion = "✓ White Noise (Tốt)"
                else:
                    conclusion = "✗ Có autocorrelation (Chưa tối ưu)"
                    all_white_noise = False
                
                print(f"{lag:<6} {lb_stat:<15.4f} {p_val:<12.6f} {conclusion}")
            
            print(f"{'='*70}")
            
            # Kết luận tổng quát
            if all_white_noise:
                print(f"\n✓ KẾT LUẬN: Residuals của {name} là WHITE NOISE")
                print(f"  → Mô hình đã trích xuất hết thông tin có thể từ dữ liệu")
                print(f"  → Không cần thêm features hoặc lags")
            else:
                print(f"\n✗ KẾT LUẬN: Residuals của {name} CÓ AUTOCORRELATION")
                print(f"  → Mô hình chưa tối ưu, còn bỏ sót thông tin")
                print(f"  → Đề xuất: Thêm lag features, thử mô hình phức tạp hơn (LSTM/ARIMA)")
            print()
            
            # Vẽ biểu đồ residuals
            self._plot_residuals_analysis(residuals_clean, lb_result, name)
            
            return lb_result
            
        except Exception as e:
            print(f"Lỗi khi thực hiện Ljung-Box test: {e}")
            return None
    
    def _plot_residuals_analysis(self, residuals, lb_result, name):
        """Vẽ biểu đồ phân tích Residuals"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. Residuals over time
        axes[0].plot(residuals, color='blue', alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title(f'Residuals Over Time - {name}\n(Nên dao động quanh 0 nếu mô hình tốt)', 
                         fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Distribution of Residuals
        axes[1].hist(residuals, bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title('Phân phối Residuals\n(Nên có dạng chuẩn - hình chuông)', 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Residual Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. Ljung-Box p-values
        lags = range(1, len(lb_result) + 1)
        p_values = lb_result['lb_pvalue'].values
        
        colors = ['green' if p > 0.05 else 'red' for p in p_values]
        axes[2].bar(lags, p_values, color=colors, alpha=0.7, edgecolor='black')
        axes[2].axhline(y=0.05, color='blue', linestyle='--', linewidth=2, 
                       label='Significance Level (α=0.05)')
        axes[2].set_title('Ljung-Box Test P-values\n(Green = White Noise ✓, Red = Autocorrelation ✗)', 
                         fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Lag', fontsize=11)
        axes[2].set_ylabel('P-value', fontsize=11)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f"residuals_analysis_{name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   → Đã lưu biểu đồ: {filename}")


def run_all_statistical_tests():
    """
    Chạy tất cả các kiểm định thống kê trên dữ liệu FPT
    
    Pipeline:
    1. Load dữ liệu preprocessed
    2. ADF Test cho Close và Log_Returns
    3. Granger Causality Test (Volume → Returns)
    4. ACF/PACF Analysis cho Log_Returns
    
    Kết quả:
    - Các biểu đồ được lưu trong results/figures/
    - Kết quả được in ra console với giải thích chi tiết
    """
    print("\n" + "="*80)
    print(" " * 20 + "STATISTICAL TESTS - FPT STOCK ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "processed", "preprocessed_data.csv")
    
    if not os.path.exists(data_file):
        print(f"Lỗi: Không tìm thấy file {data_file}")
        print("Vui lòng chạy preprocess_data.py trước")
        return
    
    df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    print(f"✓ Đã load dữ liệu: {len(df)} điểm dữ liệu\n")
    
    # Khởi tạo test module
    tester = StatisticalTests()
    
    # 1. ADF Test
    print("\n" + "█"*80)
    print("PHẦN 1: KIỂM ĐỊNH TÍNH DỪNG (STATIONARITY TEST)")
    print("█"*80)
    
    adf_close = tester.adf_test(df['Close'], name="Close Price")
    adf_returns = tester.adf_test(df['Log_Returns'], name="Log Returns")
    
    # 2. Granger Causality Test
    print("\n" + "█"*80)
    print("PHẦN 2: KIỂM ĐỊNH NHÂN QUẢ GRANGER (CAUSALITY TEST)")
    print("█"*80)
    
    # Tạo Volume_Change nếu chưa có
    if 'Volume_Change' not in df.columns:
        df['Volume_Change'] = df['Volume'].pct_change()
    
    gc_result = tester.granger_causality_test(
        df, 
        target_col='Log_Returns', 
        cause_col='Volume_Change',
        max_lag=5
    )
    
    # 3. ACF/PACF Analysis
    print("\n" + "█"*80)
    print("PHẦN 3: PHÂN TÍCH ACF/PACF (OPTIMAL LAGS DETERMINATION)")
    print("█"*80)
    
    acf_result = tester.acf_pacf_analysis(df['Log_Returns'], name="Log Returns", lags=40)
    
    print("\n" + "="*80)
    print(" " * 25 + "HOÀN THÀNH TẤT CẢ KIỂM ĐỊNH")
    print("="*80)
    print(f"\n✓ Tất cả biểu đồ đã được lưu trong: {tester.results_dir}")
    print("\nKết quả tóm tắt:")
    print(f"  1. Close Price: {'Dừng ✓' if adf_close['is_stationary'] else 'KHÔNG dừng ✗'}")
    print(f"  2. Log Returns: {'Dừng ✓' if adf_returns['is_stationary'] else 'KHÔNG dừng ✗'}")
    if gc_result:
        has_causality = any(r['has_causality'] for r in gc_result)
        print(f"  3. Volume → Returns: {'Có nhân quả ✓' if has_causality else 'KHÔNG có nhân quả ✗'}")
    print(f"  4. Optimal lags: {acf_result['optimal_lags']}")
    print()


if __name__ == "__main__":
    run_all_statistical_tests()
