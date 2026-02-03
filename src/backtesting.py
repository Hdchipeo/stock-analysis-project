import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


class BacktestingEngine:
    """
    Backtesting Engine - Đánh giá hiệu quả giao dịch của mô hình dự báo
    
    Mục đích:
    - Kiểm tra xem mô hình dự báo có thực sự sinh lời trong giao dịch thực tế không
    - So sánh với chiến lược Buy & Hold (mua và giữ)
    - Tính toán các chỉ số tài chính: Sharpe Ratio, Max Drawdown, Win Rate
    
    Tại sao quan trọng:
    - R² cao không đảm bảo lợi nhuận thực tế
    - Cần kiểm tra khả năng dự báo CHIỀU HƯỚNG giá (lên/xuống)
    - Transaction costs và slippage ảnh hưởng lớn đến lợi nhuận
    """
    
    def __init__(self, initial_capital=100_000_000, commission_rate=0.0015):
        """
        Khởi tạo Backtesting Engine
        
        Tham số:
        - initial_capital: Vốn ban đầu (VND) - default: 100 triệu
        - commission_rate: Phí giao dịch (%) - default: 0.15% (phí HoSE)
        
        Ý nghĩa:
        - Phí 0.15% là tổng phí mua + bán trên sàn HoSE
        - Vốn 100 triệu là mức vừa phải cho nhà đầu tư cá nhân
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        print(f"\n{'='*70}")
        print(f"BACKTESTING ENGINE INITIALIZED")
        print(f"{'='*70}")
        print(f"Vốn ban đầu:     {initial_capital:,.0f} VND")
        print(f"Phí giao dịch:   {commission_rate*100:.2f}%")
        print(f"{'='*70}\n")
    
    def simple_long_strategy(self, predictions_df, actual_prices, threshold=0.5):
        """
        Chiến lược Long-Only đơn giản
        
        Logic:
        - Nếu predicted_return > threshold: MUA (Long) - kỳ vọng giá tăng
        - Nếu predicted_return <= threshold: GIỮ TIỀN MẶT - tránh rủi ro giá giảm
        
        Tham số:
        - predictions_df: DataFrame với cột 'Predicted_Returns'
        - actual_prices: Series giá thực tế (để tính lợi nhuận thực)
        - threshold: Ngưỡng quyết định (default 0.5 vì data đã MinMaxScale về [0,1])
        
        Lưu ý:
        - Đây là chiến lược BẢO THỦ (không short)
        - Phù hợp với thị trường VN (không cho phép short dễ dàng)
        - Không tính đòn bẩy (leverage)
        - threshold=0.5 vì Log_Returns đã scale: 0.5 = không tăng không giảm
        """
        print(f"\n{'█'*70}")
        print(f"BACKTESTING: SIMPLE LONG-ONLY STRATEGY")
        print(f"{'█'*70}\n")
        
        capital = self.initial_capital
        shares = 0  # Số cổ phiếu đang nắm giữ
        portfolio_values = [capital]
        positions = []  # Lưu lịch sử giao dịch
        cash_history = [capital]
        shares_history = [0]
        
        for i in range(len(predictions_df)):
            pred_return = predictions_df['Predicted_Returns'].iloc[i]
            current_price = actual_prices.iloc[i]
            next_price = actual_prices.iloc[i+1] if i+1 < len(actual_prices) else current_price
            
            # Tính giá trị portfolio hiện tại
            current_portfolio_value = capital + shares * current_price
            
            # Quyết định giao dịch
            if pred_return > threshold and shares == 0:
                # Signal: MUA - Dự báo giá tăng
                # Mua tối đa số cổ phiếu có thể với số tiền hiện có
                shares_to_buy = int((capital * (1 - self.commission_rate)) / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    commission = cost * self.commission_rate
                    capital -= (cost + commission)
                    shares += shares_to_buy
                    
                    positions.append({
                        'date': predictions_df.index[i],
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'commission': commission,
                        'capital': capital
                    })
            
            elif pred_return <= threshold and shares > 0:
                # Signal: BÁN - Dự báo giá giảm hoặc không tăng
                # Bán toàn bộ cổ phiếu, chuyển sang tiền mặt
                revenue = shares * current_price
                commission = revenue * self.commission_rate
                capital += (revenue - commission)
                
                positions.append({
                    'date': predictions_df.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'commission': commission,
                    'capital': capital
                })
                
                shares = 0
            
            # Cập nhật giá trị portfolio
            portfolio_value = capital + shares * next_price
            portfolio_values.append(portfolio_value)
            cash_history.append(capital)
            shares_history.append(shares)
        
        # Tính toán metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe Ratio (Annualized)
        # Công thức: (Mean Return - Risk-free Rate) / Std of Returns * sqrt(252)
        # Risk-free rate ≈ 0 (để đơn giản)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Maximum Drawdown
        # Đo lường mức sụt giảm lớn nhất từ đỉnh cao nhất
        max_dd = self._calculate_max_drawdown(portfolio_values)
        
        # Total Return
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Win Rate (% số ngày có lợi nhuận)
        winning_days = sum(1 for r in returns if r > 0)
        win_rate = (winning_days / len(returns)) * 100 if len(returns) > 0 else 0
        
        # Total commission paid
        total_commission = sum(p['commission'] for p in positions)
        
        results = {
            'portfolio_values': portfolio_values,
            'cash_history': cash_history,
            'shares_history': shares_history,
            'positions': positions,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_return_pct': total_return,
            'win_rate': win_rate,
            'total_commission': total_commission,
            'num_trades': len(positions),
            'final_capital': portfolio_values[-1]
        }
        
        # In kết quả
        self._print_strategy_results(results, "SIMPLE LONG-ONLY STRATEGY")
        
        return results
    
    def buy_and_hold_strategy(self, actual_prices):
        """
        Chiến lược Buy & Hold (Baseline)
        
        Logic:
        - Mua cổ phiếu ở đầu kỳ
        - Giữ cho đến cuối kỳ
        - Không giao dịch trong suốt thời gian nắm giữ
        
        Mục đích:
        - So sánh xem chiến lược dự báo có vượt qua được "mua và chờ" không
        - Nếu không vượt qua Buy & Hold → Mô hình không có giá trị thực tiễn
        """
        print(f"\n{'█'*70}")
        print(f"BASELINE: BUY & HOLD STRATEGY")
        print(f"{'█'*70}\n")
        
        # Mua tối đa cổ phiếu ở ngày đầu tiên
        first_price = actual_prices.iloc[0]
        shares = int((self.initial_capital * (1 - self.commission_rate)) / first_price)
        cost = shares * first_price
        commission_buy = cost * self.commission_rate
        remaining_cash = self.initial_capital - (cost + commission_buy)
        
        # Tính giá trị portfolio theo thời gian
        portfolio_values = [self.initial_capital]
        for price in actual_prices:
            portfolio_value = remaining_cash + shares * price
            portfolio_values.append(portfolio_value)
        
        # Bán ở ngày cuối
        last_price = actual_prices.iloc[-1]
        revenue = shares * last_price
        commission_sell = revenue * self.commission_rate
        final_capital = remaining_cash + revenue - commission_sell
        
        # Metrics
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        max_dd = self._calculate_max_drawdown(portfolio_values)
        
        # Sharpe Ratio
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        results = {
            'portfolio_values': portfolio_values,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_return_pct': total_return,
            'total_commission': commission_buy + commission_sell,
            'num_trades': 2,  # Buy + Sell
            'final_capital': final_capital
        }
        
        self._print_strategy_results(results, "BUY & HOLD STRATEGY")
        
        return results
    
    def _calculate_max_drawdown(self, portfolio_values):
        """
        Tính Maximum Drawdown - Mức sụt giảm lớn nhất từ đỉnh cao nhất
        
        Công thức:
        DD_t = (Portfolio_t - Peak_t) / Peak_t
        Max DD = min(DD_t)
        
        Ý nghĩa:
        - Đo lường rủi ro lớn nhất mà nhà đầu tư phải chịu
        - Ví dụ: Max DD = -15% nghĩa là tài khoản từng giảm 15% từ đỉnh cao
        """
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) * 100  # Convert to percentage
        
        return max_drawdown
    
    def _print_strategy_results(self, results, strategy_name):
        """In kết quả của chiến lược"""
        print(f"\n{'─'*70}")
        print(f"KẾT QUẢ: {strategy_name}")
        print(f"{'─'*70}")
        print(f"Vốn ban đầu:           {self.initial_capital:>15,.0f} VND")
        print(f"Vốn cuối kỳ:           {results['final_capital']:>15,.0f} VND")
        print(f"Tổng lợi nhuận:        {results['total_return_pct']:>15.2f}%")
        print(f"Sharpe Ratio:          {results['sharpe_ratio']:>15.4f}")
        print(f"Max Drawdown:          {results['max_drawdown']:>15.2f}%")
        
        if 'win_rate' in results:
            print(f"Win Rate:              {results['win_rate']:>15.2f}%")
        
        print(f"Số lần giao dịch:      {results['num_trades']:>15}")
        print(f"Tổng phí giao dịch:    {results['total_commission']:>15,.0f} VND")
        print(f"{'─'*70}\n")
    
    def compare_strategies(self, model_results, baseline_results):
        """
        So sánh Model Strategy vs Buy & Hold
        
        Mục đích:
        - Xem chiến lược dự báo có vượt trội không
        - Đánh giá risk-adjusted return (Sharpe Ratio)
        """
        print(f"\n{'='*70}")
        print(f"SO SÁNH CHIẾN LƯỢC")
        print(f"{'='*70}\n")
        
        comparison = pd.DataFrame({
            'Metric': [
                'Total Return (%)',
                'Sharpe Ratio',
                'Max Drawdown (%)',
                'Số giao dịch',
                'Tổng phí (VND)'
            ],
            'Model Strategy': [
                f"{model_results['total_return_pct']:.2f}%",
                f"{model_results['sharpe_ratio']:.4f}",
                f"{model_results['max_drawdown']:.2f}%",
                model_results['num_trades'],
                f"{model_results['total_commission']:,.0f}"
            ],
            'Buy & Hold': [
                f"{baseline_results['total_return_pct']:.2f}%",
                f"{baseline_results['sharpe_ratio']:.4f}",
                f"{baseline_results['max_drawdown']:.2f}%",
                baseline_results['num_trades'],
                f"{baseline_results['total_commission']:,.0f}"
            ]
        })
        
        print(comparison.to_string(index=False))
        print(f"\n{'='*70}")
        
        # Kết luận
        print("\n📊 NHẬN XÉT:")
        
        # Total Return comparison
        if model_results['total_return_pct'] > baseline_results['total_return_pct']:
            diff = model_results['total_return_pct'] - baseline_results['total_return_pct']
            print(f"   ✓ Model Strategy VƯỢT TRỘI hơn Buy & Hold: {diff:.2f}%")
        else:
            diff = baseline_results['total_return_pct'] - model_results['total_return_pct']
            print(f"   ✗ Model Strategy KÉMHƠN Buy & Hold: {diff:.2f}%")
        
        # Sharpe Ratio comparison
        if model_results['sharpe_ratio'] > baseline_results['sharpe_ratio']:
            print(f"   ✓ Risk-adjusted return TỐT HƠN (Sharpe Ratio cao hơn)")
        else:
            print(f"   ✗ Risk-adjusted return KÉMUẢ (Sharpe Ratio thấp hơn)")
        
        # Max Drawdown comparison (càng nhỏ càng tốt)
        if model_results['max_drawdown'] > baseline_results['max_drawdown']:
            print(f"   ✗ RỦI RO cao hơn (Max Drawdown lớn hơn)")
        else:
            print(f"   ✓ RỦI RO thấp hơn (Max Drawdown nhỏ hơn)")
        
        print()
        
        return comparison


def plot_backtest_comparison(model_results, baseline_results, save_path="results/figures"):
    """
    Vẽ biểu đồ so sánh hiệu quả backtesting
    
    Bao gồm:
    1. Portfolio value theo thời gian
    2. Drawdown chart
    3. Monthly returns comparison
    """
    os.makedirs(save_path, exist_ok=True)
    
    # === Figure 1: Portfolio Value Comparison ===
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Portfolio Value
    model_values = model_results['portfolio_values']
    baseline_values = baseline_results['portfolio_values']
    
    axes[0].plot(model_values, label='Model Strategy', linewidth=2.5, color='#2E86AB')
    axes[0].plot(baseline_values, label='Buy & Hold', linewidth=2.5, color='#A23B72', linestyle='--')
    axes[0].axhline(y=model_results['portfolio_values'][0], color='gray', 
                    linestyle=':', alpha=0.5, label='Initial Capital')
    
    axes[0].set_title('Backtesting: Portfolio Value Over Time\nModel Strategy vs Buy & Hold', 
                      fontsize=16, fontweight='bold', pad=20)
    axes[0].set_ylabel('Portfolio Value (VND)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='plain', axis='y')
    
    # Format y-axis with Vietnamese number format
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Plot 2: Drawdown Comparison
    model_dd = calculate_drawdown(model_values)
    baseline_dd = calculate_drawdown(baseline_values)
    
    axes[1].fill_between(range(len(model_dd)), model_dd, 0, 
                          alpha=0.4, color='#2E86AB', label='Model Strategy DD')
    axes[1].fill_between(range(len(baseline_dd)), baseline_dd, 0, 
                          alpha=0.4, color='#A23B72', label='Buy & Hold DD')
    axes[1].set_title('Drawdown Comparison (% giảm từ đỉnh cao nhất)', 
                      fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Trading Days', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'backtesting_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   → Đã lưu biểu đồ: backtesting_comparison.png")
    
    # === Figure 2: Performance Metrics Bar Chart ===
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['Total Return\n(%)', 'Sharpe\nRatio', 'Max Drawdown\n(%)']
    model_metrics = [
        model_results['total_return_pct'],
        model_results['sharpe_ratio'],
        abs(model_results['max_drawdown'])
    ]
    baseline_metrics = [
        baseline_results['total_return_pct'],
        baseline_results['sharpe_ratio'],
        abs(baseline_results['max_drawdown'])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model_metrics, width, label='Model Strategy', 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, baseline_metrics, width, label='Buy & Hold', 
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_metrics_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   → Đã lưu biểu đồ: performance_metrics_comparison.png")


def calculate_drawdown(portfolio_values):
    """Helper function to calculate drawdown series"""
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100
    return drawdown


def run_backtesting(predictions_file="predictions_returns.csv", test_data_file="test_data.csv", year_label=""):
    """
    Chạy backtesting cho mô hình dự báo
    
    Input:
    - predictions_file: File chứa dự báo của mô hình
    - test_data_file: File chứa dữ liệu test (giá thực tế)
    - year_label: Nhãn năm để hiển thị (vd: "2024", "2025")
    
    Output:
    - Dict kết quả backtesting
    """
    label = f" ({year_label})" if year_label else ""
    print(f"\n{'='*80}")
    print(f"{' '*25}BACKTESTING MODULE{label}")
    print("="*80 + "\n")
    
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predictions_path = os.path.join(base_dir, "results", predictions_file)
    test_data_path = os.path.join(base_dir, "data", "processed", test_data_file)
    scaling_params_path = os.path.join(base_dir, "data", "processed", "scaling_params.json")
    
    if not os.path.exists(predictions_path):
        print(f"Lỗi: Không tìm thấy {predictions_path}")
        print("Vui lòng chạy modeling.py trước")
        return None
    
    if not os.path.exists(test_data_path):
        print(f"Lỗi: Không tìm thấy {test_data_path}")
        return None
    
    # Load predictions
    predictions_df = pd.read_csv(predictions_path, index_col='Date', parse_dates=True)
    test_df = pd.read_csv(test_data_path, index_col='Date', parse_dates=True)
    
    # Load scaling params to get actual prices
    with open(scaling_params_path, 'r') as f:
        scaling_params = json.load(f)
    
    # Inverse scale Close prices
    def inverse_scale(val):
        return val * (scaling_params['Close_max'] - scaling_params['Close_min']) + scaling_params['Close_min']
    
    actual_prices = test_df['Close'].apply(inverse_scale)
    
    # Align predictions with test data dates
    common_dates = predictions_df.index.intersection(test_df.index)
    if len(common_dates) == 0:
        print(f"Lỗi: Không có ngày chung giữa predictions và test data cho {year_label}")
        return None
    
    predictions_df = predictions_df.loc[common_dates]
    actual_prices = actual_prices.loc[common_dates]
    
    print(f"Số phiên giao dịch: {len(common_dates)}")
    print(f"Từ {common_dates.min().strftime('%Y-%m-%d')} đến {common_dates.max().strftime('%Y-%m-%d')}")
    
    # Get predicted returns
    if 'XGBoost_Returns' in predictions_df.columns:
        pred_returns = predictions_df['XGBoost_Returns']
    elif 'BiLSTM_Returns' in predictions_df.columns:
        pred_returns = predictions_df['BiLSTM_Returns']
    else:
        print("Lỗi: Không tìm thấy cột dự báo trong predictions file")
        return None
    
    # Create predictions dataframe for backtesting
    backtest_df = pd.DataFrame({
        'Predicted_Returns': pred_returns[:len(actual_prices)-1]
    }, index=actual_prices.index[:len(pred_returns)])
    
    # Initialize backtesting engine
    engine = BacktestingEngine(initial_capital=100_000_000, commission_rate=0.0015)
    
    # Run Model Strategy
    model_results = engine.simple_long_strategy(backtest_df, actual_prices)
    
    # Run Buy & Hold Strategy  
    baseline_results = engine.buy_and_hold_strategy(actual_prices)
    
    # Compare strategies
    comparison = engine.compare_strategies(model_results, baseline_results)
    
    return {
        'model_results': model_results,
        'baseline_results': baseline_results,
        'comparison': comparison,
        'year': year_label
    }


def run_yearly_comparison():
    """
    Chạy backtesting so sánh 2 năm: 2024 và 2025
    """
    print("\n" + "█"*80)
    print(" "*20 + "SO SÁNH BACKTESTING: 2024 vs 2025")
    print("█"*80 + "\n")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results", "figures")
    
    # Chạy backtesting cho năm 2024
    print("\n" + "="*60)
    print(" "*20 + "NĂM 2024")
    print("="*60)
    results_2024 = run_backtesting(
        predictions_file="predictions_returns.csv",
        test_data_file="test_2024.csv",
        year_label="2024"
    )
    
    # Chạy backtesting cho năm 2025
    print("\n" + "="*60)
    print(" "*20 + "NĂM 2025")
    print("="*60)
    results_2025 = run_backtesting(
        predictions_file="predictions_returns.csv",
        test_data_file="test_2025.csv",
        year_label="2025"
    )
    
    # So sánh 2 năm
    if results_2024 and results_2025:
        print("\n" + "█"*80)
        print(" "*25 + "BẢNG SO SÁNH TỔNG HỢP")
        print("█"*80 + "\n")
        
        comparison_data = {
            'Metric': [
                'Vốn cuối kỳ (VND)',
                'Total Return (%)',
                'Sharpe Ratio',
                'Max Drawdown (%)',
                'Win Rate (%)',
                'Số giao dịch',
                'Tổng phí (VND)',
                'Buy & Hold Return (%)',
                'Alpha (%)'
            ],
            '2024 Model': [
                f"{results_2024['model_results']['final_capital']:,.0f}",
                f"{results_2024['model_results']['total_return_pct']:.2f}%",
                f"{results_2024['model_results']['sharpe_ratio']:.4f}",
                f"{results_2024['model_results']['max_drawdown']:.2f}%",
                f"{results_2024['model_results']['win_rate']:.2f}%",
                f"{results_2024['model_results']['num_trades']}",
                f"{results_2024['model_results']['total_commission']:,.0f}",
                f"{results_2024['baseline_results']['total_return_pct']:.2f}%",
                f"{results_2024['model_results']['total_return_pct'] - results_2024['baseline_results']['total_return_pct']:.2f}%"
            ],
            '2025 Model': [
                f"{results_2025['model_results']['final_capital']:,.0f}",
                f"{results_2025['model_results']['total_return_pct']:.2f}%",
                f"{results_2025['model_results']['sharpe_ratio']:.4f}",
                f"{results_2025['model_results']['max_drawdown']:.2f}%",
                f"{results_2025['model_results']['win_rate']:.2f}%",
                f"{results_2025['model_results']['num_trades']}",
                f"{results_2025['model_results']['total_commission']:,.0f}",
                f"{results_2025['baseline_results']['total_return_pct']:.2f}%",
                f"{results_2025['model_results']['total_return_pct'] - results_2025['baseline_results']['total_return_pct']:.2f}%"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Đánh giá
        print("\n" + "─"*80)
        print("📊 ĐÁNH GIÁ:")
        print("─"*80)
        
        ret_2024 = results_2024['model_results']['total_return_pct']
        ret_2025 = results_2025['model_results']['total_return_pct']
        bh_2024 = results_2024['baseline_results']['total_return_pct']
        bh_2025 = results_2025['baseline_results']['total_return_pct']
        
        if ret_2024 > 0:
            print(f"   ✅ 2024: Model có lãi {ret_2024:.2f}%")
        else:
            print(f"   ❌ 2024: Model lỗ {ret_2024:.2f}%")
            
        if ret_2025 > 0:
            print(f"   ✅ 2025: Model có lãi {ret_2025:.2f}%")
        else:
            print(f"   ❌ 2025: Model lỗ {ret_2025:.2f}%")
        
        if ret_2024 > bh_2024:
            print(f"   ✅ 2024: Model THẮNG Buy & Hold ({ret_2024:.2f}% vs {bh_2024:.2f}%)")
        else:
            print(f"   ❌ 2024: Model THUA Buy & Hold ({ret_2024:.2f}% vs {bh_2024:.2f}%)")
            
        if ret_2025 > bh_2025:
            print(f"   ✅ 2025: Model THẮNG Buy & Hold ({ret_2025:.2f}% vs {bh_2025:.2f}%)")
        else:
            print(f"   ❌ 2025: Model THUA Buy & Hold ({ret_2025:.2f}% vs {bh_2025:.2f}%)")
        
        print("─"*80)
        
        # Lưu kết quả
        results_path = os.path.join(base_dir, "results", "backtesting_yearly_comparison.csv")
        comparison_df.to_csv(results_path, index=False)
        print(f"\n✓ Đã lưu kết quả so sánh vào: {results_path}")
        
        # Vẽ biểu đồ so sánh
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Return comparison
        years = ['2024', '2025']
        model_returns = [ret_2024, ret_2025]
        bh_returns = [bh_2024, bh_2025]
        
        x = np.arange(len(years))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, model_returns, width, label='Model Strategy', color='#2ecc71' if ret_2024 > 0 else '#e74c3c')
        axes[0, 0].bar(x + width/2, bh_returns, width, label='Buy & Hold', color='#3498db')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].set_title('So sánh Return: Model vs Buy & Hold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(years)
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Plot 2: Win Rate
        win_rates = [results_2024['model_results']['win_rate'], results_2025['model_results']['win_rate']]
        colors = ['#2ecc71' if wr > 50 else '#e74c3c' for wr in win_rates]
        axes[0, 1].bar(years, win_rates, color=colors)
        axes[0, 1].axhline(y=50, color='black', linestyle='--', alpha=0.3, label='Random (50%)')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].set_title('Win Rate theo năm')
        axes[0, 1].legend()
        
        # Plot 3: Number of trades
        num_trades = [results_2024['model_results']['num_trades'], results_2025['model_results']['num_trades']]
        axes[1, 0].bar(years, num_trades, color='#9b59b6')
        axes[1, 0].set_ylabel('Số giao dịch')
        axes[1, 0].set_title('Số lượng giao dịch theo năm')
        
        # Plot 4: Max Drawdown
        max_dd = [results_2024['model_results']['max_drawdown'], results_2025['model_results']['max_drawdown']]
        axes[1, 1].bar(years, max_dd, color='#e74c3c')
        axes[1, 1].set_ylabel('Max Drawdown (%)')
        axes[1, 1].set_title('Max Drawdown theo năm')
        
        plt.tight_layout()
        chart_path = os.path.join(results_dir, 'yearly_comparison.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   → Đã lưu biểu đồ: yearly_comparison.png")
    
    print("\n" + "█"*80)
    print(" "*25 + "BACKTESTING HOÀN THÀNH")
    print("█"*80 + "\n")
    
    return results_2024, results_2025


if __name__ == "__main__":
    run_yearly_comparison()
