"""
Backtesting Module - Đánh giá hiệu quả giao dịch của mô hình dự báo

Phiên bản: Đồ án môn học
- Chiến lược Long-Only đơn giản
- So sánh với Buy & Hold
- Chia theo 2 năm: 2024 và 2025 để so sánh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class BacktestingEngine:
    """Backtesting Engine cho đánh giá chiến lược giao dịch"""
    
    def __init__(self, initial_capital=100_000_000, commission_rate=0.0015):
        """
        Khởi tạo Backtesting Engine
        
        Tham số:
        - initial_capital: Vốn ban đầu (VND) - default: 100 triệu
        - commission_rate: Phí giao dịch (%) - default: 0.15% (phí HoSE)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
    
    def simple_long_strategy(self, predictions_df, actual_prices, threshold=0.5):
        """
        Chiến lược Long-Only đơn giản
        
        Logic:
        - predicted_return > threshold: MUA
        - predicted_return <= threshold: BÁN
        """
        capital = self.initial_capital
        shares = 0
        portfolio_values = [capital]
        positions = []
        
        for i in range(len(predictions_df)):
            pred_return = predictions_df['Predicted_Returns'].iloc[i]
            current_price = actual_prices.iloc[i]
            next_price = actual_prices.iloc[i+1] if i+1 < len(actual_prices) else current_price
            
            if pred_return > threshold and shares == 0:
                # MUA
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
                        'commission': commission
                    })
            
            elif pred_return <= threshold and shares > 0:
                # BÁN
                revenue = shares * current_price
                commission = revenue * self.commission_rate
                capital += (revenue - commission)
                positions.append({
                    'date': predictions_df.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'commission': commission
                })
                shares = 0
            
            portfolio_value = capital + shares * next_price
            portfolio_values.append(portfolio_value)
        
        # Metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_dd = self._calculate_max_drawdown(portfolio_values)
        win_rate = (sum(1 for r in returns if r > 0) / len(returns)) * 100 if len(returns) > 0 else 0
        total_commission = sum(p['commission'] for p in positions)
        
        return {
            'portfolio_values': portfolio_values,
            'positions': positions,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': len(positions),
            'total_commission': total_commission,
            'final_capital': portfolio_values[-1]
        }
    
    def buy_and_hold_strategy(self, actual_prices):
        """Chiến lược Buy & Hold (Baseline)"""
        first_price = actual_prices.iloc[0]
        shares = int((self.initial_capital * (1 - self.commission_rate)) / first_price)
        cost = shares * first_price
        commission_buy = cost * self.commission_rate
        remaining_cash = self.initial_capital - (cost + commission_buy)
        
        portfolio_values = [self.initial_capital]
        for price in actual_prices:
            portfolio_value = remaining_cash + shares * price
            portfolio_values.append(portfolio_value)
        
        last_price = actual_prices.iloc[-1]
        revenue = shares * last_price
        commission_sell = revenue * self.commission_rate
        final_capital = remaining_cash + revenue - commission_sell
        
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        max_dd = self._calculate_max_drawdown(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        win_rate = (sum(1 for r in returns if r > 0) / len(returns)) * 100 if len(returns) > 0 else 0
        
        return {
            'portfolio_values': portfolio_values,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': 2,
            'total_commission': commission_buy + commission_sell,
            'final_capital': final_capital
        }
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Tính Maximum Drawdown"""
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown) * 100


def run_backtesting_by_year(predictions_file="predictions_returns.csv", test_data_file="test_data.csv"):
    """
    Chạy backtesting chia theo năm (2024 và 2025)
    """
    print(f"\n{'='*70}")
    print("BACKTESTING MODULE - SO SÁNH THEO NĂM")
    print(f"{'='*70}\n")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    data_dir = os.path.join(base_dir, "data", "processed")
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load predictions
    predictions_path = os.path.join(results_dir, predictions_file)
    if not os.path.exists(predictions_path):
        print(f"❌ Không tìm thấy file: {predictions_path}")
        return None
    
    predictions_df = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
    print(f"✓ Đã load predictions: {len(predictions_df)} samples")
    
    # Load test data
    test_data_path = os.path.join(data_dir, test_data_file)
    if not os.path.exists(test_data_path):
        print(f"❌ Không tìm thấy file: {test_data_path}")
        return None
    
    test_df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
    print(f"✓ Đã load test data: {len(test_df)} samples")
    
    # Align data
    common_index = predictions_df.index.intersection(test_df.index)
    predictions_df = predictions_df.loc[common_index]
    actual_prices = test_df.loc[common_index, 'Close']
    
    # Tìm cột dự báo
    pred_cols = [col for col in predictions_df.columns if 'XGBoost' in col or 'Returns' in col]
    if pred_cols:
        predictions_df['Predicted_Returns'] = predictions_df[pred_cols[0]]
    
    # ============================================================
    # CHIA DỮ LIỆU THEO NĂM
    # ============================================================
    print(f"\n{'─'*70}")
    print("CHIA DỮ LIỆU THEO NĂM")
    print(f"{'─'*70}")
    
    # Filter by year
    mask_2024 = predictions_df.index.year == 2024
    mask_2025 = predictions_df.index.year == 2025
    
    preds_2024 = predictions_df[mask_2024]
    preds_2025 = predictions_df[mask_2025]
    prices_2024 = actual_prices[mask_2024]
    prices_2025 = actual_prices[mask_2025]
    
    print(f"  Năm 2024: {len(preds_2024)} phiên ({preds_2024.index.min().strftime('%Y-%m-%d')} → {preds_2024.index.max().strftime('%Y-%m-%d')})")
    print(f"  Năm 2025: {len(preds_2025)} phiên ({preds_2025.index.min().strftime('%Y-%m-%d')} → {preds_2025.index.max().strftime('%Y-%m-%d')})")
    
    # ============================================================
    # BACKTESTING NĂM 2024
    # ============================================================
    print(f"\n{'█'*70}")
    print("BACKTESTING NĂM 2024")
    print(f"{'█'*70}")
    
    engine_2024 = BacktestingEngine(initial_capital=100_000_000)
    
    if len(preds_2024) > 0:
        model_2024 = engine_2024.simple_long_strategy(preds_2024, prices_2024)
        bh_2024 = engine_2024.buy_and_hold_strategy(prices_2024)
        
        print(f"\n  Model Strategy 2024:")
        print(f"    - Lợi nhuận: {model_2024['total_return_pct']:.2f}%")
        print(f"    - Sharpe Ratio: {model_2024['sharpe_ratio']:.4f}")
        print(f"    - Max Drawdown: {model_2024['max_drawdown']:.2f}%")
        print(f"    - Số giao dịch: {model_2024['num_trades']}")
        
        print(f"\n  Buy & Hold 2024:")
        print(f"    - Lợi nhuận: {bh_2024['total_return_pct']:.2f}%")
        print(f"    - Sharpe Ratio: {bh_2024['sharpe_ratio']:.4f}")
        print(f"    - Max Drawdown: {bh_2024['max_drawdown']:.2f}%")
    else:
        print("  ⚠️ Không có dữ liệu năm 2024")
        model_2024 = bh_2024 = None
    
    # ============================================================
    # BACKTESTING NĂM 2025
    # ============================================================
    print(f"\n{'█'*70}")
    print("BACKTESTING NĂM 2025")
    print(f"{'█'*70}")
    
    engine_2025 = BacktestingEngine(initial_capital=100_000_000)
    
    if len(preds_2025) > 0:
        model_2025 = engine_2025.simple_long_strategy(preds_2025, prices_2025)
        bh_2025 = engine_2025.buy_and_hold_strategy(prices_2025)
        
        print(f"\n  Model Strategy 2025:")
        print(f"    - Lợi nhuận: {model_2025['total_return_pct']:.2f}%")
        print(f"    - Sharpe Ratio: {model_2025['sharpe_ratio']:.4f}")
        print(f"    - Max Drawdown: {model_2025['max_drawdown']:.2f}%")
        print(f"    - Số giao dịch: {model_2025['num_trades']}")
        
        print(f"\n  Buy & Hold 2025:")
        print(f"    - Lợi nhuận: {bh_2025['total_return_pct']:.2f}%")
        print(f"    - Sharpe Ratio: {bh_2025['sharpe_ratio']:.4f}")
        print(f"    - Max Drawdown: {bh_2025['max_drawdown']:.2f}%")
    else:
        print("  ⚠️ Không có dữ liệu năm 2025")
        model_2025 = bh_2025 = None
    
    # ============================================================
    # SO SÁNH 2024 vs 2025
    # ============================================================
    print(f"\n{'='*70}")
    print("SO SÁNH HIỆU QUẢ: 2024 vs 2025")
    print(f"{'='*70}\n")
    
    comparison_data = {
        'Metric': ['Model Return (%)', 'Model Sharpe', 'Model Max DD (%)', 
                   'B&H Return (%)', 'B&H Sharpe', 'B&H Max DD (%)']
    }
    
    if model_2024 and bh_2024:
        comparison_data['Năm 2024'] = [
            f"{model_2024['total_return_pct']:.2f}",
            f"{model_2024['sharpe_ratio']:.4f}",
            f"{model_2024['max_drawdown']:.2f}",
            f"{bh_2024['total_return_pct']:.2f}",
            f"{bh_2024['sharpe_ratio']:.4f}",
            f"{bh_2024['max_drawdown']:.2f}"
        ]
    
    if model_2025 and bh_2025:
        comparison_data['Năm 2025'] = [
            f"{model_2025['total_return_pct']:.2f}",
            f"{model_2025['sharpe_ratio']:.4f}",
            f"{model_2025['max_drawdown']:.2f}",
            f"{bh_2025['total_return_pct']:.2f}",
            f"{bh_2025['sharpe_ratio']:.4f}",
            f"{bh_2025['max_drawdown']:.2f}"
        ]
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # ============================================================
    # VẼ BIỂU ĐỒ SO SÁNH
    # ============================================================
    print(f"\n📊 Đang tạo biểu đồ...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Portfolio 2024
    if model_2024:
        axes[0, 0].plot(model_2024['portfolio_values'], label='Model Strategy', linewidth=2, color='#2E86AB')
        axes[0, 0].plot(bh_2024['portfolio_values'], label='Buy & Hold', linewidth=2, color='#A23B72', linestyle='--')
        axes[0, 0].axhline(y=100_000_000, color='gray', linestyle=':', alpha=0.5)
        axes[0, 0].set_title(f'Portfolio Value - Năm 2024\nModel: {model_2024["total_return_pct"]:.1f}% | B&H: {bh_2024["total_return_pct"]:.1f}%', fontweight='bold')
        axes[0, 0].set_ylabel('Portfolio Value (VND)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # Plot 2: Portfolio 2025
    if model_2025:
        axes[0, 1].plot(model_2025['portfolio_values'], label='Model Strategy', linewidth=2, color='#2E86AB')
        axes[0, 1].plot(bh_2025['portfolio_values'], label='Buy & Hold', linewidth=2, color='#A23B72', linestyle='--')
        axes[0, 1].axhline(y=100_000_000, color='gray', linestyle=':', alpha=0.5)
        axes[0, 1].set_title(f'Portfolio Value - Năm 2025\nModel: {model_2025["total_return_pct"]:.1f}% | B&H: {bh_2025["total_return_pct"]:.1f}%', fontweight='bold')
        axes[0, 1].set_ylabel('Portfolio Value (VND)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    
    # Plot 3: Return Comparison Bar Chart
    if model_2024 and model_2025:
        years = ['2024', '2025']
        model_returns = [model_2024['total_return_pct'], model_2025['total_return_pct']]
        bh_returns = [bh_2024['total_return_pct'], bh_2025['total_return_pct']]
        
        x = np.arange(len(years))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x - width/2, model_returns, width, label='Model Strategy', color='#2E86AB')
        bars2 = axes[1, 0].bar(x + width/2, bh_returns, width, label='Buy & Hold', color='#A23B72')
        axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
        axes[1, 0].set_title('So sánh Lợi nhuận theo Năm', fontweight='bold')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(years)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar in bars1:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
        for bar in bars2:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Sharpe Ratio Comparison
    if model_2024 and model_2025:
        model_sharpe = [model_2024['sharpe_ratio'], model_2025['sharpe_ratio']]
        bh_sharpe = [bh_2024['sharpe_ratio'], bh_2025['sharpe_ratio']]
        
        bars1 = axes[1, 1].bar(x - width/2, model_sharpe, width, label='Model Strategy', color='#2E86AB')
        bars2 = axes[1, 1].bar(x + width/2, bh_sharpe, width, label='Buy & Hold', color='#A23B72')
        axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[1, 1].set_title('So sánh Sharpe Ratio theo Năm', fontweight='bold')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(years)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'backtesting_yearly_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   → Đã lưu: backtesting_yearly_comparison.png")
    
    # ============================================================
    # LƯU KẾT QUẢ
    # ============================================================
    metrics_data = {
        'Metric': ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                   'Win Rate (%)', 'Num Trades', 'Total Commission (VND)']
    }
    
    if model_2024:
        metrics_data['Model_2024'] = [
            model_2024['total_return_pct'], model_2024['sharpe_ratio'], 
            model_2024['max_drawdown'], model_2024['win_rate'],
            model_2024['num_trades'], model_2024['total_commission']
        ]
        metrics_data['BuyHold_2024'] = [
            bh_2024['total_return_pct'], bh_2024['sharpe_ratio'], 
            bh_2024['max_drawdown'], bh_2024['win_rate'],
            bh_2024['num_trades'], bh_2024['total_commission']
        ]
    
    if model_2025:
        metrics_data['Model_2025'] = [
            model_2025['total_return_pct'], model_2025['sharpe_ratio'], 
            model_2025['max_drawdown'], model_2025['win_rate'],
            model_2025['num_trades'], model_2025['total_commission']
        ]
        metrics_data['BuyHold_2025'] = [
            bh_2025['total_return_pct'], bh_2025['sharpe_ratio'], 
            bh_2025['max_drawdown'], bh_2025['win_rate'],
            bh_2025['num_trades'], bh_2025['total_commission']
        ]
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(results_dir, "backtesting_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Đã lưu kết quả: {metrics_path}")
    
    # ============================================================
    # NHẬN XÉT
    # ============================================================
    print(f"\n{'='*70}")
    print("📊 NHẬN XÉT")
    print(f"{'='*70}")
    
    if model_2024 and model_2025:
        if model_2024['total_return_pct'] > model_2025['total_return_pct']:
            print(f"  → Năm 2024 hiệu quả hơn năm 2025 cho Model Strategy")
        else:
            print(f"  → Năm 2025 hiệu quả hơn năm 2024 cho Model Strategy")
        
        if model_2024['total_return_pct'] > bh_2024['total_return_pct']:
            print(f"  → Năm 2024: Model VƯỢT TRỘI so với Buy & Hold")
        else:
            print(f"  → Năm 2024: Model KÉM HƠN Buy & Hold")
        
        if model_2025['total_return_pct'] > bh_2025['total_return_pct']:
            print(f"  → Năm 2025: Model VƯỢT TRỘI so với Buy & Hold")
        else:
            print(f"  → Năm 2025: Model KÉM HƠN Buy & Hold")
    
    print(f"\n{'='*70}")
    print("BACKTESTING HOÀN TẤT!")
    print(f"{'='*70}\n")
    
    return {
        '2024': {'model': model_2024, 'buyhold': bh_2024} if model_2024 else None,
        '2025': {'model': model_2025, 'buyhold': bh_2025} if model_2025 else None
    }


# Alias for backward compatibility
def run_backtesting(*args, **kwargs):
    return run_backtesting_by_year(*args, **kwargs)


if __name__ == "__main__":
    run_backtesting_by_year()
