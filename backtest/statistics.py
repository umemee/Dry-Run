# backtest/statistics.py
"""
백테스팅 결과 통계 분석
- 기본 지표:  Win Rate, Profit Factor, Avg Win/Loss
- 리스크 지표: MDD, Sharpe Ratio, VaR
- 경고 지표: Top 5 의존도, 연속 손실, 복구 기간
"""

import numpy as np
import pandas as pd

class BacktestStatistics:
    
    @staticmethod
    def calculate_all_metrics(daily_results):
        """
        전체 통계 계산
        
        Args:
            daily_results:  [
                {'date': '2025-12-30', 'pnl': 123.45, 'trades': 2, ... },
                ... 
            ]
            
        Returns:
            dict: 모든 통계 지표
        """
        if not daily_results:
            return BacktestStatistics._empty_metrics()
        
        df = pd.DataFrame(daily_results)
        
        # 기본 지표
        total_pnl = df['pnl'].sum()
        total_trades = df['trades'].sum()
        
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        
        win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        # Profit Factor
        total_wins = wins['pnl'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # MDD (Maximum Drawdown)
        cumulative = df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # MDD 발생 날짜
        mdd_date = df. iloc[drawdown.idxmin()]['date'] if len(df) > 0 else None
        
        # Sharpe Ratio (일간 수익률 기준)
        if df['pnl'].std() > 0:
            sharpe_daily = df['pnl'].mean() / df['pnl'].std()
            sharpe_annualized = sharpe_daily * np.sqrt(252)
        else:
            sharpe_annualized = 0
        
        # VaR (Value at Risk) - 하위 5%
        var_5pct = np.percentile(df['pnl'], 5)
        
        # 상위 5개 거래 의존도
        if 'trade_details' in df.columns and total_pnl > 0:
            all_trades = []
            for trades in df['trade_details']:
                if isinstance(trades, list):
                    all_trades.extend(trades)
            
            if all_trades:
                trade_pnls = [t['net_pnl'] for t in all_trades]
                top_5 = sorted(trade_pnls, reverse=True)[:5]
                top_5_sum = sum(top_5)
                top_5_dependency = (top_5_sum / total_pnl * 100) if total_pnl > 0 else 0
            else:
                top_5_dependency = 0
        else: 
            top_5_dependency = 0
        
        # 거래 없는 날
        zero_trade_days = len(df[df['trades'] == 0])
        
        # 연속 손실
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for pnl in df['pnl']:
            if pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # 복구 기간 (MDD 발생 후 회복까지)
        recovery_days = 0
        max_recovery_days = 0
        in_drawdown = False
        
        for dd in drawdown:
            if dd < 0:
                in_drawdown = True
                recovery_days += 1
            elif in_drawdown:
                max_recovery_days = max(max_recovery_days, recovery_days)
                recovery_days = 0
                in_drawdown = False
        
        return {
            # 기본 지표
            'total_pnl': round(total_pnl, 2),
            'total_trades': int(total_trades),
            'trading_days': len(df),
            'win_rate': round(win_rate, 1),
            'avg_win':  round(avg_win, 2),
            'avg_loss':  round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            
            # 리스크 지표
            'max_drawdown': round(max_drawdown, 2),
            'mdd_date': str(mdd_date),
            'sharpe_ratio':  round(sharpe_annualized, 2),
            'var_5pct': round(var_5pct, 2),
            
            # 경고 지표
            'top_5_dependency_pct': round(top_5_dependency, 1),
            'zero_trade_days': int(zero_trade_days),
            'max_consecutive_losses': int(max_consecutive_losses),
            'max_recovery_days': int(max_recovery_days)
        }
    
    @staticmethod
    def _empty_metrics():
        """빈 결과"""
        return {
            'total_pnl': 0,
            'total_trades': 0,
            'trading_days': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'mdd_date': None,
            'sharpe_ratio': 0,
            'var_5pct': 0,
            'top_5_dependency_pct': 0,
            'zero_trade_days':  0,
            'max_consecutive_losses': 0,
            'max_recovery_days':  0
        }
    
    @staticmethod
    def analyze_regime(df, regime_type='volatility'):
        """
        시장 환경별 성과 분석
        
        Args:
            df: 데이터프레임 (분봉)
            regime_type: 'volatility' | 'trend'
            
        Returns:
            필터링된 데이터프레임
        """
        if regime_type == 'low_volatility':
            # 일간 변동폭 < 5%
            df['daily_range'] = (df. groupby('date')['high']. transform('max') - 
                                 df.groupby('date')['low'].transform('min')) / df.groupby('date')['open']. transform('first')
            return df[df['daily_range'] < 0.05]
        
        elif regime_type == 'bear_market':
            # 전일 대비 하락
            df['prev_close'] = df. groupby('date')['close'].shift(1)
            df['daily_change'] = (df['close'] - df['prev_close']) / df['prev_close']
            return df[df['daily_change'] < 0]
        
        return df