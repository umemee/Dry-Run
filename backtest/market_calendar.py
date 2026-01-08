# backtest/market_calendar.py
"""
시장 시간 관리 및 날짜 유틸리티
"""

import pandas as pd
from datetime import datetime, timedelta

class MarketCalendar: 
    
    # 거래 시간 (ET 기준)
    PREMARKET_START = "0400"
    MARKET_OPEN = "0930"
    MARKET_CLOSE = "1600"
    AFTERMARKET_END = "2000"
    
    @staticmethod
    def is_market_hours(time_str):
        """
        장 운영 시간 여부 (04:00 ~ 16:00)
        
        Args:
            time_str: "0930" 형식
            
        Returns:
            bool
        """
        return MarketCalendar.PREMARKET_START <= time_str < MarketCalendar.MARKET_CLOSE
    
    @staticmethod
    def is_regular_hours(time_str):
        """정규장 시간 (09:30 ~ 16:00)"""
        return MarketCalendar.MARKET_OPEN <= time_str < MarketCalendar.MARKET_CLOSE
    
    @staticmethod
    def extract_date_from_filename(filename):
        """
        파일명에서 날짜 추출
        
        Args: 
            filename: "20251230_AEHL.csv"
            
        Returns: 
            "2025-12-30" 또는 None
        """
        import os
        basename = os.path.basename(filename).replace('.csv', '')
        parts = basename.split('_')
        
        for part in parts:
            if part.isdigit() and len(part) == 8:
                try:
                    dt = datetime.strptime(part, '%Y%m%d')
                    return dt.strftime('%Y-%m-%d')
                except:
                    pass
        return None
    
    @staticmethod
    def extract_symbol_from_filename(filename):
        """
        파일명에서 종목명 추출
        
        Args:
            filename: "20251230_AEHL.csv"
            
        Returns: 
            "AEHL"
        """
        import os
        basename = os.path.basename(filename).replace('.csv', '')
        parts = basename.split('_')
        
        # 날짜가 아닌 부분이 종목명
        for part in parts:
            if not (part.isdigit() and len(part) == 8):
                return part
        
        return parts[0] if parts else "UNKNOWN"
    
    @staticmethod
    def should_scan_now(time_str, interval_minutes=10):
        """
        스캔 시점 여부 (10분, 20분, 30분, ...)
        
        Args:
            time_str: "0930"
            interval_minutes: 10
            
        Returns:
            bool
        """
        if len(time_str) != 4:
            return False
        
        minute = int(time_str[-2:])
        return minute % interval_minutes == 0