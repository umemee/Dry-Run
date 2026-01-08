# backtest/trading_costs.py
"""
미국 주식 거래 비용 계산 (2024년 기준)
- Commission: 주당 $0.005 (최소 $1.0, 최대 거래액의 0.5%)
- SEC Fee: 거래액의 0.00221%
- FINRA TAF: 주당 $0.000145 (매도 시만)
"""

class TradingCosts:
    # 수수료 구조
    COMMISSION_PER_SHARE = 0.005
    COMMISSION_MIN = 1.0
    COMMISSION_MAX_PCT = 0.005
    
    # 규제 수수료
    SEC_FEE_RATE = 0.0000221
    FINRA_TAF = 0.000145
    
    @staticmethod
    def calculate_entry_cost(qty, price):
        """
        매수 비용 계산
        
        Args:
            qty:  수량
            price: 가격
            
        Returns:
            총 비용 (USD)
        """
        notional = qty * price
        
        # Commission
        commission = max(
            TradingCosts. COMMISSION_MIN,
            min(
                qty * TradingCosts. COMMISSION_PER_SHARE,
                notional * TradingCosts. COMMISSION_MAX_PCT
            )
        )
        
        # SEC Fee
        sec_fee = notional * TradingCosts. SEC_FEE_RATE
        
        return round(commission + sec_fee, 2)
    
    @staticmethod
    def calculate_exit_cost(qty, price):
        """
        매도 비용 계산 (TAF 추가)
        
        Args:
            qty: 수량
            price: 가격
            
        Returns:
            총 비용 (USD)
        """
        notional = qty * price
        
        # Commission
        commission = max(
            TradingCosts.COMMISSION_MIN,
            min(
                qty * TradingCosts.COMMISSION_PER_SHARE,
                notional * TradingCosts.COMMISSION_MAX_PCT
            )
        )
        
        # SEC Fee
        sec_fee = notional * TradingCosts. SEC_FEE_RATE
        
        # FINRA TAF (매도 시만)
        taf = qty * TradingCosts. FINRA_TAF
        
        return round(commission + sec_fee + taf, 2)
    
    @staticmethod
    def total_round_trip_cost(qty, entry_price, exit_price):
        """왕복 거래 비용"""
        entry = TradingCosts.calculate_entry_cost(qty, entry_price)
        exit_ = TradingCosts.calculate_exit_cost(qty, exit_price)
        return entry + exit_