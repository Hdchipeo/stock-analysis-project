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
        Chiến lược Long-Only đơn giản (KHÔNG có T+2 - để so sánh)
        
        Logic:
        - Nếu predicted_return > threshold: MUA (Long) - kỳ vọng giá tăng
        - Nếu predicted_return <= threshold: BÁN/GIỮ TIỀN - tránh rủi ro
        
        Tham số:
        - predictions_df: DataFrame với cột 'Predicted_Returns'
        - actual_prices: Series giá thực tế (để tính lợi nhuận thực)
        - threshold: Ngưỡng quyết định (default 0.5 cho dữ liệu MinMaxScaled)
        
        Lưu ý:
        - Đây là chiến lược BẢO THỦ (không short)
        - Dữ liệu đã được MinMaxScale về [0,1], nên 0.5 = không đổi
        - KHÔNG áp dụng T+2 (phiên bản đơn giản)
        """
        print(f"\n{'█'*70}")
        print(f"BACKTESTING: SIMPLE LONG-ONLY STRATEGY (Không T+2)")
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
            
            # Quyết định giao dịch (0.5 = điểm giữa của MinMaxScaler)
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
        self._print_strategy_results(results, "SIMPLE LONG-ONLY (Không T+2)")
        
        return results
    
    def simple_long_strategy_t2(self, predictions_df, actual_prices, threshold=0.5):
        """
        Chiến lược Long-Only với quy tắc T+2 (Thị trường Việt Nam)
        
        Quy tắc T+2:
        - T+0: Đặt lệnh mua/bán
        - T+2: Cổ phiếu/tiền về tài khoản (2 ngày làm việc sau)
        
        Logic:
        - Sau khi MUA, phải đợi 2 ngày mới được BÁN (cổ phiếu chưa về)
        - Sau khi BÁN, phải đợi 2 ngày mới được MUA lại (tiền chưa về)
        
        Tham số:
        - predictions_df: DataFrame với cột 'Predicted_Returns'
        - actual_prices: Series giá thực tế (để tính lợi nhuận thực)
        - threshold: Ngưỡng quyết định (default 0.5 cho dữ liệu MinMaxScaled)
        """
        print(f"\n{'█'*70}")
        print(f"BACKTESTING: LONG-ONLY STRATEGY VỚI T+2 (Việt Nam)")
        print(f"{'█'*70}\n")
        
        T_PLUS = 2  # Số ngày chờ settlement
        
        capital = self.initial_capital
        available_capital = capital  # Tiền có thể sử dụng ngay
        pending_cash = []  # [(ngày_về, số_tiền)] - Tiền đang chờ về
        
        shares = 0  # Cổ phiếu đã settled (có thể bán)
        pending_shares = []  # [(ngày_về, số_cổ_phiếu)] - Cổ phiếu đang chờ về
        
        portfolio_values = [capital]
        positions = []
        cash_history = [capital]
        shares_history = [0]
        
        for i in range(len(predictions_df)):
            current_day = i
            pred_return = predictions_df['Predicted_Returns'].iloc[i]
            current_price = actual_prices.iloc[i]
            next_price = actual_prices.iloc[i+1] if i+1 < len(actual_prices) else current_price
            
            # === SETTLEMENT: Kiểm tra cổ phiếu/tiền đã về chưa ===
            # Cập nhật cổ phiếu đã settled
            new_pending_shares = []
            for settle_day, share_count in pending_shares:
                if current_day >= settle_day:
                    shares += share_count  # Cổ phiếu đã về, có thể bán
                else:
                    new_pending_shares.append((settle_day, share_count))
            pending_shares = new_pending_shares
            
            # Cập nhật tiền đã settled
            new_pending_cash = []
            for settle_day, cash_amount in pending_cash:
                if current_day >= settle_day:
                    available_capital += cash_amount  # Tiền đã về, có thể dùng
                else:
                    new_pending_cash.append((settle_day, cash_amount))
            pending_cash = new_pending_cash
            
            # Tính tổng tài sản (bao gồm cả pending)
            total_pending_shares = shares + sum(s for _, s in pending_shares)
            total_pending_cash = available_capital + sum(c for _, c in pending_cash)
            
            # === TRADING DECISION (0.5 = điểm giữa của MinMaxScaler) ===
            if pred_return > threshold and shares == 0 and len(pending_shares) == 0:
                # Signal: MUA - Dự báo giá tăng
                # Chỉ mua nếu có tiền khả dụng
                if available_capital > current_price:
                    shares_to_buy = int((available_capital * (1 - self.commission_rate)) / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        commission = cost * self.commission_rate
                        available_capital -= (cost + commission)
                        
                        # Cổ phiếu sẽ về sau T+2 ngày
                        settle_day = current_day + T_PLUS
                        pending_shares.append((settle_day, shares_to_buy))
                        
                        positions.append({
                            'date': predictions_df.index[i],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'commission': commission,
                            'settle_date': predictions_df.index[min(i + T_PLUS, len(predictions_df)-1)] if i + T_PLUS < len(predictions_df) else 'N/A',
                            'capital': available_capital
                        })
            
            elif pred_return <= threshold and shares > 0:
                # Signal: BÁN - Dự báo giá giảm
                # Chỉ bán cổ phiếu đã settled (T+2)
                if shares > 0:
                    revenue = shares * current_price
                    commission = revenue * self.commission_rate
                    net_revenue = revenue - commission
                    
                    # Tiền sẽ về sau T+2 ngày
                    settle_day = current_day + T_PLUS
                    pending_cash.append((settle_day, net_revenue))
                    
                    positions.append({
                        'date': predictions_df.index[i],
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'commission': commission,
                        'settle_date': predictions_df.index[min(i + T_PLUS, len(predictions_df)-1)] if i + T_PLUS < len(predictions_df) else 'N/A',
                        'capital': available_capital
                    })
                    
                    shares = 0
            
            # Cập nhật giá trị portfolio (bao gồm pending)
            total_shares = shares + sum(s for _, s in pending_shares)
            total_cash = available_capital + sum(c for _, c in pending_cash)
            portfolio_value = total_cash + total_shares * next_price
            portfolio_values.append(portfolio_value)
            cash_history.append(total_cash)
            shares_history.append(total_shares)
        
        # Tính toán metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        max_dd = self._calculate_max_drawdown(portfolio_values)
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        
        winning_days = sum(1 for r in returns if r > 0)
        win_rate = (winning_days / len(returns)) * 100 if len(returns) > 0 else 0
        
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
        self._print_strategy_results(results, "LONG-ONLY VỚI T+2 (Việt Nam)")
        
        return results
    
    def mean_reversion_strategy(self, predictions_df, actual_prices, rsi_series, 
                                  stop_loss_pct=0.07, lookback_window=30):
        """
        Chiến lược Mean Reversion với Stop-Loss và Dynamic Threshold
        
        Nguyên lý Mean Reversion:
        - Giá cổ phiếu có xu hướng quay về giá trị trung bình
        - Mua khi giá giảm quá mức (oversold), bán khi tăng quá mức (overbought)
        
        Các cải tiến:
        1. Stop-Loss 7%: Tự động cắt lỗ khi giảm 7% so với giá mua
        2. Dynamic Threshold: Ngưỡng mua/bán thích ứng theo thị trường
        3. RSI Filter: Chỉ mua khi RSI < 40 (oversold), bán khi RSI > 60 (overbought)
        
        Tham số:
        - predictions_df: DataFrame với cột 'Predicted_Returns'
        - actual_prices: Series giá thực tế
        - rsi_series: Series RSI tương ứng
        - stop_loss_pct: Ngưỡng cắt lỗ (default 7%)
        - lookback_window: Số ngày tính dynamic threshold (default 30)
        """
        print(f"\n{'█'*70}")
        print(f"BACKTESTING: MEAN REVERSION + STOP-LOSS STRATEGY")
        print(f"{'█'*70}")
        print(f"   📉 Stop-Loss: {stop_loss_pct*100:.0f}%")
        print(f"   📊 Dynamic Threshold: {lookback_window}-day rolling mean")
        print(f"   🔍 RSI Filter: Buy < 40, Sell > 60")
        print(f"{'█'*70}\n")
        
        capital = self.initial_capital
        shares = 0
        buy_price = 0  # Giá mua để tính stop-loss
        portfolio_values = [capital]
        positions = []
        cash_history = [capital]
        shares_history = [0]
        
        # Tính Dynamic Threshold = rolling mean của predictions
        pred_returns = predictions_df['Predicted_Returns']
        dynamic_threshold = pred_returns.rolling(window=lookback_window, min_periods=10).mean()
        dynamic_threshold = dynamic_threshold.fillna(0.5)  # Default = 0.5 cho những ngày đầu
        
        stop_loss_triggered = 0
        rsi_buy_signals = 0
        rsi_sell_signals = 0
        
        for i in range(len(predictions_df)):
            pred_return = predictions_df['Predicted_Returns'].iloc[i]
            current_price = actual_prices.iloc[i]
            next_price = actual_prices.iloc[i+1] if i+1 < len(actual_prices) else current_price
            current_threshold = dynamic_threshold.iloc[i]
            
            # Lấy RSI (cần kiểm tra index)
            try:
                current_rsi = rsi_series.iloc[i] if i < len(rsi_series) else 50
            except:
                current_rsi = 50  # Default nếu lỗi
            
            # === STOP-LOSS CHECK ===
            if shares > 0 and buy_price > 0:
                loss_pct = (current_price - buy_price) / buy_price
                if loss_pct <= -stop_loss_pct:
                    # Stop-loss triggered! Bán ngay
                    revenue = shares * current_price
                    commission = revenue * self.commission_rate
                    capital += (revenue - commission)
                    
                    positions.append({
                        'date': predictions_df.index[i],
                        'action': 'STOP-LOSS',
                        'price': current_price,
                        'shares': shares,
                        'commission': commission,
                        'loss_pct': loss_pct * 100,
                        'capital': capital
                    })
                    
                    shares = 0
                    buy_price = 0
                    stop_loss_triggered += 1
                    
                    # Cập nhật và continue (không xét signal nữa)
                    portfolio_value = capital + shares * next_price
                    portfolio_values.append(portfolio_value)
                    cash_history.append(capital)
                    shares_history.append(shares)
                    continue
            
            # === MEAN REVERSION TRADING LOGIC ===
            # Điều kiện MUA (Mean Reversion):
            # - Prediction > Dynamic Threshold (tín hiệu phục hồi)
            # - RSI < 40 (oversold - giá đã giảm nhiều, có khả năng tăng lại)
            # - Chưa có vị thế
            should_buy = (pred_return > current_threshold and 
                         current_rsi < 40 and 
                         shares == 0)
            
            # Điều kiện BÁN (Mean Reversion):
            # - Prediction <= Dynamic Threshold (tín hiệu yếu đi)
            # - RSI > 60 (overbought - giá đã tăng nhiều, có khả năng giảm)
            # - Đang có vị thế
            should_sell = (pred_return <= current_threshold and 
                          current_rsi > 60 and 
                          shares > 0)
            
            if should_buy:
                shares_to_buy = int((capital * (1 - self.commission_rate)) / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    commission = cost * self.commission_rate
                    capital -= (cost + commission)
                    shares += shares_to_buy
                    buy_price = current_price  # Lưu giá mua cho stop-loss
                    
                    positions.append({
                        'date': predictions_df.index[i],
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'commission': commission,
                        'rsi': current_rsi,
                        'threshold': current_threshold,
                        'capital': capital
                    })
                    rsi_buy_signals += 1
            
            elif should_sell:
                revenue = shares * current_price
                commission = revenue * self.commission_rate
                capital += (revenue - commission)
                
                # Tính lời/lỗ
                pnl_pct = (current_price - buy_price) / buy_price * 100 if buy_price > 0 else 0
                
                positions.append({
                    'date': predictions_df.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'commission': commission,
                    'rsi': current_rsi,
                    'pnl_pct': pnl_pct,
                    'capital': capital
                })
                
                shares = 0
                buy_price = 0
                rsi_sell_signals += 1
            
            # Cập nhật giá trị portfolio
            portfolio_value = capital + shares * next_price
            portfolio_values.append(portfolio_value)
            cash_history.append(capital)
            shares_history.append(shares)
        
        # Tính toán metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        max_dd = self._calculate_max_drawdown(portfolio_values)
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        
        winning_days = sum(1 for r in returns if r > 0)
        win_rate = (winning_days / len(returns)) * 100 if len(returns) > 0 else 0
        
        total_commission = sum(p['commission'] for p in positions)
        
        # Tính win rate của các giao dịch (không phải theo ngày)
        sell_trades = [p for p in positions if p['action'] in ['SELL', 'STOP-LOSS']]
        profitable_trades = sum(1 for p in sell_trades if p.get('pnl_pct', p.get('loss_pct', -100)) > 0)
        trade_win_rate = (profitable_trades / len(sell_trades) * 100) if sell_trades else 0
        
        results = {
            'portfolio_values': portfolio_values,
            'cash_history': cash_history,
            'shares_history': shares_history,
            'positions': positions,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_return_pct': total_return,
            'win_rate': win_rate,
            'trade_win_rate': trade_win_rate,
            'total_commission': total_commission,
            'num_trades': len(positions),
            'stop_loss_count': stop_loss_triggered,
            'final_capital': portfolio_values[-1]
        }
        
        # In kết quả chi tiết
        self._print_strategy_results(results, "MEAN REVERSION + STOP-LOSS")
        print(f"\n   📊 Chi tiết bổ sung:")
        print(f"      - Stop-Loss triggered: {stop_loss_triggered} lần")
        print(f"      - RSI Buy signals: {rsi_buy_signals} lần")  
        print(f"      - RSI Sell signals: {rsi_sell_signals} lần")
        print(f"      - Trade Win Rate: {trade_win_rate:.1f}%")
        
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


def run_backtesting(predictions_file="predictions.csv", test_data_file="test_data.csv"):
    """
    Chạy backtesting cho mô hình dự báo
    
    Input:
    - predictions_file: File chứa dự báo của mô hình
    - test_data_file: File chứa dữ liệu test (giá thực tế)
    
    Output:
    - Kết quả backtesting được lưu vào results/backtesting_metrics.csv
    - Các biểu đồ so sánh
    """
    print("\n" + "="*80)
    print(" " * 30 + "BACKTESTING MODULE")
    print("="*80 + "\n")
    
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predictions_path = os.path.join(base_dir, "results", predictions_file)
    test_data_path = os.path.join(base_dir, "data", "processed", test_data_file)
    scaling_params_path = os.path.join(base_dir, "data", "processed", "scaling_params.json")
    
    if not os.path.exists(predictions_path):
        print(f"Lỗi: Không tìm thấy {predictions_path}")
        print("Vui lòng chạy modeling.py trước")
        return
    
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
    
    # Get predicted returns (assuming model predicted Log_Returns)
    # Need to check if predictions contain Log_Returns or Close
    if 'XGBoost_Returns' in predictions_df.columns:
        pred_returns = predictions_df['XGBoost_Returns']
    elif 'XGBoost' in predictions_df.columns:
        # Convert price predictions to returns
        pred_returns = predictions_df['XGBoost'].pct_change()
    else:
        print("Lỗi: Không tìm thấy cột dự báo trong predictions file")
        return
    
    # Create predictions dataframe for backtesting
    backtest_df = pd.DataFrame({
        'Predicted_Returns': pred_returns[:len(actual_prices)-1]  # -1 because we need next price
    }, index=actual_prices.index[:len(pred_returns)])
    
    # Initialize backtesting engine
    engine = BacktestingEngine(initial_capital=100_000_000, commission_rate=0.0015)
    
    # =========================================================================
    # CHẠY 4 CHIẾN LƯỢC ĐỂ SO SÁNH
    # =========================================================================
    
    # 1. Chiến lược Momentum KHÔNG T+2 (phiên bản đơn giản)
    model_results_no_t2 = engine.simple_long_strategy(backtest_df, actual_prices)
    
    # 2. Chiến lược Momentum CÓ T+2 (theo quy định thị trường VN)
    model_results_t2 = engine.simple_long_strategy_t2(backtest_df, actual_prices)
    
    # 3. Chiến lược Mean Reversion + Stop-Loss (CHIẾN LƯỢC CẢI TIẾN)
    # Cần RSI từ test data
    if 'RSI_14' in test_df.columns:
        # Inverse scale RSI (nếu đã scale)
        rsi_min, rsi_max = 0, 100  # RSI gốc trong khoảng 0-100
        # RSI đã được scale về [0,1], cần inverse lại
        rsi_series = test_df['RSI_14'] * (rsi_max - rsi_min) + rsi_min
        rsi_series = rsi_series[:len(backtest_df)]
        
        mean_reversion_results = engine.mean_reversion_strategy(
            backtest_df, 
            actual_prices, 
            rsi_series,
            stop_loss_pct=0.07,  # Stop-Loss 7%
            lookback_window=30   # Dynamic threshold 30 ngày
        )
    else:
        print("⚠️ Không tìm thấy RSI_14 trong test data, bỏ qua Mean Reversion strategy")
        mean_reversion_results = None
    
    # 4. Buy & Hold Strategy (baseline)
    baseline_results = engine.buy_and_hold_strategy(actual_prices)
    
    # =========================================================================
    # SO SÁNH TẤT CẢ CHIẾN LƯỢC
    # =========================================================================
    print(f"\n{'='*80}")
    print(f" " * 20 + "SO SÁNH TẤT CẢ CHIẾN LƯỢC")
    print(f"{'='*80}\n")
    
    comparison_data = {
        'Metric': [
            'Total Return (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Số giao dịch',
            'Tổng phí (VND)'
        ],
        'Momentum (Không T+2)': [
            f"{model_results_no_t2['total_return_pct']:.2f}%",
            f"{model_results_no_t2['sharpe_ratio']:.4f}",
            f"{model_results_no_t2['max_drawdown']:.2f}%",
            f"{model_results_no_t2['win_rate']:.2f}%",
            model_results_no_t2['num_trades'],
            f"{model_results_no_t2['total_commission']:,.0f}"
        ],
        'Momentum (Có T+2)': [
            f"{model_results_t2['total_return_pct']:.2f}%",
            f"{model_results_t2['sharpe_ratio']:.4f}",
            f"{model_results_t2['max_drawdown']:.2f}%",
            f"{model_results_t2['win_rate']:.2f}%",
            model_results_t2['num_trades'],
            f"{model_results_t2['total_commission']:,.0f}"
        ],
        'Buy & Hold': [
            f"{baseline_results['total_return_pct']:.2f}%",
            f"{baseline_results['sharpe_ratio']:.4f}",
            f"{baseline_results['max_drawdown']:.2f}%",
            "N/A",
            baseline_results['num_trades'],
            f"{baseline_results['total_commission']:,.0f}"
        ]
    }
    
    # Thêm Mean Reversion nếu có
    if mean_reversion_results:
        comparison_data['Mean Reversion + SL'] = [
            f"{mean_reversion_results['total_return_pct']:.2f}%",
            f"{mean_reversion_results['sharpe_ratio']:.4f}",
            f"{mean_reversion_results['max_drawdown']:.2f}%",
            f"{mean_reversion_results['win_rate']:.2f}%",
            mean_reversion_results['num_trades'],
            f"{mean_reversion_results['total_commission']:,.0f}"
        ]
    
    comparison = pd.DataFrame(comparison_data)
    
    print(comparison.to_string(index=False))
    
    # Nhận xét
    print(f"\n{'─'*80}")
    print("📊 PHÂN TÍCH TÁC ĐỘNG CỦA QUY TẮC T+2:")
    print(f"{'─'*80}")
    
    diff_return = model_results_no_t2['total_return_pct'] - model_results_t2['total_return_pct']
    diff_trades = model_results_no_t2['num_trades'] - model_results_t2['num_trades']
    
    print(f"   📉 Chênh lệch lợi nhuận (Không T+2 - Có T+2): {diff_return:+.2f}%")
    print(f"   📊 Chênh lệch số giao dịch: {diff_trades} lần")
    
    if diff_return > 0:
        print(f"\n   ⚠️  CẢNH BÁO: Kết quả không có T+2 bị PHÓNG ĐẠI {diff_return:.2f}%")
        print(f"       Trong thực tế, bạn sẽ đạt được khoảng {model_results_t2['total_return_pct']:.2f}%")
    else:
        print(f"\n   ✓ Chiến lược T+2 thực tế tốt hơn dự kiến!")
    
    print(f"\n   💡 LƯU Ý:")
    print(f"      - Quy tắc T+2 giảm tần suất giao dịch → Ít cơ hội lướt sóng")
    print(f"      - Cổ phiếu mua xong phải đợi 2 ngày mới bán được")
    print(f"      - Tiền bán xong phải đợi 2 ngày mới mua lại được")
    
    # Save results
    results_path = os.path.join(base_dir, "results", "backtesting_metrics.csv")
    comparison.to_csv(results_path, index=False)
    print(f"\n✓ Đã lưu kết quả backtesting vào: {results_path}")
    
    # Plot comparison (sử dụng T+2 results làm model chính)
    results_dir = os.path.join(base_dir, "results", "figures")
    plot_backtest_comparison(model_results_t2, baseline_results, results_dir)
    
    print("\n" + "="*80)
    print(" " * 28 + "BACKTESTING HOÀN THÀNH")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_backtesting()
