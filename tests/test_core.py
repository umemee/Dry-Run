# tests/test_core.py
"""
핵심 기능 유닛 테스트
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import TradingCosts, ExecutionSimulator, MarketCalendar, BacktestStatistics

def test_trading_costs():
    """거래 비용 계산 테스트"""
    print("🧪 Testing TradingCosts...")
    
    # Entry Cost
    entry_cost = TradingCosts.calculate_entry_cost(qty=100, price=10.0)
    assert entry_cost > 0, "Entry cost should be > 0"
    print(f"  ✅ Entry cost (100 shares @ $10): ${entry_cost:.2f}")
    
    # Exit Cost
    exit_cost = TradingCosts.calculate_exit_cost(qty=100, price=11.0)
    assert exit_cost > entry_cost, "Exit cost should be higher (TAF included)"
    print(f"  ✅ Exit cost (100 shares @ $11): ${exit_cost:.2f}")
    
    # Round Trip
    total_cost = TradingCosts.total_round_trip_cost(100, 10.0, 11.0)
    print(f"  ✅ Round trip cost:  ${total_cost:.2f}")

def test_execution_simulator():
    """체결 시뮬레이션 테스트"""
    print("\n🧪 Testing ExecutionSimulator...")
    
    bar = {'open': 10.0, 'high': 10.5, 'low': 9.8, 'close': 10.2, 'volume': 5000}
    
    # Realistic Mode
    result = ExecutionSimulator.simulate_fill(limit_price=10.0, current_bar=bar, mode='realistic')
    
    print(f"  Filled:  {result['filled']}")
    print(f"  Fill Price: ${result['fill_price']:.2f}" if result['filled'] else "  Not filled")
    print(f"  Fill Qty %: {result['fill_qty_pct']*100:.0f}%" if result['filled'] else "")
    print(f"  Reason: {result['reason']}")
    
    assert 'filled' in result, "Result should have 'filled' key"
    print("  ✅ Execution simulation OK")

def test_market_calendar():
    """시장 시간 테스트"""
    print("\n🧪 Testing MarketCalendar...")
    
    assert MarketCalendar.is_market_hours("0930") == True
    assert MarketCalendar.is_market_hours("1600") == False
    assert MarketCalendar.is_market_hours("0400") == True
    
    print("  ✅ Market hours check OK")
    
    # 파일명 파싱
    date = MarketCalendar.extract_date_from_filename("20251230_AEHL.csv")
    symbol = MarketCalendar.extract_symbol_from_filename("20251230_AEHL.csv")
    
    assert date == "2025-12-30", f"Date parsing failed: {date}"
    assert symbol == "AEHL", f"Symbol parsing failed: {symbol}"
    
    print(f"  ✅ Filename parsing:  {date}, {symbol}")

def test_statistics():
    """통계 계산 테스트"""
    print("\n🧪 Testing BacktestStatistics...")
    
    daily_results = [
        {'date': '2025-12-30', 'pnl': 123.45, 'trades': 2, 'trade_details': []},
        {'date': '2025-12-31', 'pnl': -45.67, 'trades':  1, 'trade_details': []},
        {'date': '2026-01-02', 'pnl': 89.12, 'trades': 3, 'trade_details': []}
    ]
    
    stats = BacktestStatistics.calculate_all_metrics(daily_results)
    
    print(f"  Total PnL:  ${stats['total_pnl']:.2f}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  Max Drawdown: ${stats['max_drawdown']:.2f}")
    
    assert stats['total_pnl'] > 0, "Total PnL should be positive"
    assert stats['win_rate'] > 0, "Win rate should be > 0"
    
    print("  ✅ Statistics calculation OK")

if __name__ == "__main__": 
    print("="*70)
    print("🧪 Running Unit Tests")
    print("="*70)
    
    test_trading_costs()
    test_execution_simulator()
    test_market_calendar()
    test_statistics()
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)