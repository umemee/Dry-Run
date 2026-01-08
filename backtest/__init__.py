# backtest/__init__.py
"""
Dry-Run Backtesting Framework
실전 매매 환경을 최대한 재현하는 백테스팅 시스템
"""

__version__ = "2.0.0"
__author__ = "umemee"

from .trading_costs import TradingCosts
from .execution_sim import ExecutionSimulator
from .market_calendar import MarketCalendar
from .statistics import BacktestStatistics
from .stress_scenarios import StressScenarios

__all__ = [
    'TradingCosts',
    'ExecutionSimulator',
    'MarketCalendar',
    'BacktestStatistics',
    'StressScenarios'
]