# backtest/stress_scenarios.py
"""
스트레스 시나리오 - 최악의 상황 강제 재현
"""

import random
from .execution_sim import ExecutionSimulator

class StressScenarios: 
    
    @staticmethod
    def worst_case_execution(limit_price, current_bar):
        """
        최악의 체결 조건 강제 적용
        - 50% 체결 실패 확률
        - +1% 슬리피지
        - 30%만 부분 체결
        """
        if random.random() < 0.5:
            return {
                'filled': False,
                'fill_price': None,
                'fill_qty_pct': 0.0,
                'latency_seconds': 10,
                'reason': 'Stress test:  forced rejection'
            }
        
        slippage = 0.01  # 1% 슬리피지
        return {
            'filled': True,
            'fill_price': limit_price * (1 + slippage),
            'fill_qty_pct': 0.3,  # 30%만 체결
            'latency_seconds': 10,
            'reason': 'Stress test: worst case fill'
        }
    
    @staticmethod
    def force_consecutive_losses(trade_results, count=10):
        """
        최악의 연속 손실 시나리오
        
        Args:
            trade_results: 전체 거래 결과
            count: 연속 손실 개수
            
        Returns:
            {
                'scenario': str,
                'total_loss':  float,
                'survival':  bool
            }
        """
        # 손실 거래만 추출
        losses = [t for t in trade_results if t['net_pnl'] < 0]
        
        if len(losses) < count:
            return {
                'scenario': f'연속 손실 {count}개',
                'total_loss': 0,
                'survival': True,
                'message': f'데이터 부족 (손실 거래 {len(losses)}개만 존재)'
            }
        
        # 최악의 손실 N개 선택
        worst_losses = sorted(losses, key=lambda x: x['net_pnl'])[:count]
        total_loss = sum([t['net_pnl'] for t in worst_losses])
        
        # 생존 여부 (일일 한도 $600 기준)
        survival = total_loss > -600
        
        return {
            'scenario': f'최악 {count}연속 손실',
            'total_loss': round(total_loss, 2),
            'survival': survival,
            'trades':  [{'symbol': t['symbol'], 'pnl': t['net_pnl']} for t in worst_losses]
        }
    
    @staticmethod
    def remove_top_n_trades(trade_results, n=5):
        """
        상위 N개 거래 제거 후 수익률
        
        Args:
            trade_results: 전체 거래 결과
            n:  제거할 상위 거래 개수
            
        Returns:
            {
                'original_pnl': float,
                'after_removal_pnl': float,
                'dependency_pct': float
            }
        """
        if not trade_results:
            return {'original_pnl': 0, 'after_removal_pnl': 0, 'dependency_pct': 0}
        
        sorted_trades = sorted(trade_results, key=lambda x: x['net_pnl'], reverse=True)
        
        original_pnl = sum([t['net_pnl'] for t in trade_results])
        removed_trades = sorted_trades[:n]
        remaining_trades = sorted_trades[n:]
        remaining_pnl = sum([t['net_pnl'] for t in remaining_trades])
        
        dependency = (1 - remaining_pnl / original_pnl) * 100 if original_pnl > 0 else 0
        
        return {
            'original_pnl': round(original_pnl, 2),
            'after_removal_pnl': round(remaining_pnl, 2),
            'dependency_pct': round(dependency, 1),
            'removed_trades': [
                {'symbol': t['symbol'], 'pnl': t['net_pnl']} 
                for t in removed_trades
            ]
        }