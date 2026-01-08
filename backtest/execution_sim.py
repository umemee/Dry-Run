# backtest/execution_sim. py
"""
실전에 가까운 체결 시뮬레이션
- 지연 (Latency): 1~10초
- 슬리피지 (Slippage): 0.1~0.5% (불리한 방향)
- 부분 체결 (Partial Fill): 거래량 기반 30~100%
- 체결 실패 (Rejection): 극단 변동 시 10% 확률
"""

import random

class ExecutionSimulator:
    
    @staticmethod
    def simulate_fill(limit_price, current_bar, symbol_volatility=0.02, mode='realistic'):
        """
        체결 시뮬레이션
        
        Args:
            limit_price: 지정가
            current_bar: {'open', 'high', 'low', 'close', 'volume'}
            symbol_volatility: 종목 변동성 (0.0~1.0)
            mode: 'optimistic' | 'realistic' | 'pessimistic'
            
        Returns:
            {
                'filled': bool,
                'fill_price': float,
                'fill_qty_pct': float (0.0~1.0),
                'latency_seconds': int,
                'reason': str
            }
        """
        result = {
            'filled': False,
            'fill_price':  None,
            'fill_qty_pct': 0.0,
            'latency_seconds': 0,
            'reason': 'No fill'
        }
        
        # Mode별 파라미터 조정
        if mode == 'optimistic':
            tolerance = 1.01
            slippage_range = (0.0, 0.001)
            rejection_prob = 0.0
        elif mode == 'realistic':
            tolerance = 1.005
            slippage_range = (0.001, 0.005)
            rejection_prob = 0.10
        else:  # pessimistic
            tolerance = 1.003
            slippage_range = (0.005, 0.015)
            rejection_prob = 0.20
        
        # 1. 체결 가능 여부 (허용 범위 내 터치)
        if current_bar['low'] > limit_price * tolerance:
            result['reason'] = 'Price too high'
            return result
        
        # 2. 지연 시간 (급등주일수록 증가)
        base_latency = random.randint(1, 3)
        volatility_latency = int(symbol_volatility * 50)
        result['latency_seconds'] = min(base_latency + volatility_latency, 10)
        
        # 3. 극단 변동 시 체결 실패
        price_range = abs(current_bar['high'] - current_bar['low'])
        price_move_pct = price_range / current_bar['open'] if current_bar['open'] > 0 else 0
        
        if price_move_pct > 0.10 and random.random() < rejection_prob:
            result['reason'] = 'Extreme volatility rejection'
            return result
        
        # 4. 체결 확정
        result['filled'] = True
        
        # 5. 체결 가격 (슬리피지 포함 - 불리한 방향)
        slippage = random.uniform(*slippage_range)
        
        if current_bar['open'] < limit_price: 
            # 시가가 지정가보다 낮으면 시가 체결 (운 좋음)
            result['fill_price'] = current_bar['open']
            result['reason'] = 'Filled at open (favorable)'
        else:
            # 시가가 지정가보다 높으면 지정가 + 슬리피지
            result['fill_price'] = limit_price * (1 + slippage)
            result['reason'] = f'Filled with {slippage*100:.2f}% slippage'
        
        # 6. 부분 체결 (거래량 기반)
        volume = current_bar.get('volume', 1000)
        
        if volume < 500:
            result['fill_qty_pct'] = random.uniform(0.3, 0.6)
        elif volume < 2000:
            result['fill_qty_pct'] = random.uniform(0.6, 0.9)
        else:
            result['fill_qty_pct'] = random.uniform(0.9, 1.0)
        
        return result
    
    @staticmethod
    def simulate_market_sell(position_qty, current_bar, urgency='normal'):
        """
        시장가 매도 시뮬레이션
        
        Args: 
            position_qty: 보유 수량
            current_bar: 현재 봉
            urgency: 'normal' | 'panic' (손절 등)
            
        Returns:
            {
                'fill_price': float,
                'fill_qty':  int,
                'slippage_pct': float
            }
        """
        if urgency == 'panic': 
            # 손절 매도 = 더 불리한 가격
            slippage = random.uniform(0.005, 0.020)  # 0.5~2%
            fill_price = current_bar['close'] * (1 - slippage)
        else:
            # 일반 매도
            slippage = random.uniform(0.001, 0.005)  # 0.1~0.5%
            fill_price = current_bar['close'] * (1 - slippage)
        
        return {
            'fill_price':  fill_price,
            'fill_qty': position_qty,
            'slippage_pct': slippage * 100
        }