# run_reality.py
"""
Tier 2: Reality Mode - 실전 시뮬레이션
- 거래 비용 포함 (Commission + SEC Fee + TAF)
- 체결 지연/슬리피지/부분체결
- 일일 손실 한도
- Trailing Stop
"""

import os
import sys
import re
import glob
import random
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 랜덤 시드 고정 (재현성)
random.seed(42)
np.random.seed(42)

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 실전 코드 import
from strategy import GapZoneStrategy
from config import Config

# 백테스트 모듈 import
from backtest import (
    TradingCosts,
    ExecutionSimulator,
    MarketCalendar,
    BacktestStatistics
)

# ==========================================
# 로깅 설정 (Windows 이모지 대응) - 수정 버전
# ==========================================
import sys
import re

os.makedirs('results', exist_ok=True)

# 파일 핸들러 (UTF-8)
file_handler = logging.FileHandler('results/reality_mode.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# 콘솔 핸들러 (이모지 제거)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

class NoEmojiFormatter(logging.Formatter):
    """Windows 콘솔용 이모지 제거 포매터"""
    def format(self, record):
        msg = super().format(record)
        # 이모지 제거
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub('', msg)

console_handler.setFormatter(NoEmojiFormatter('%(asctime)s [%(levelname)s] %(message)s'))

# 로거 설정 (중복 제거!)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# ==========================================
# 지표 계산 (실전 코드와 동일)
# ==========================================
def compute_indicators(df):
    """지표 계산 (Shift 1 적용) - 안정화 버전"""
    df = df.copy()
    
    # 숫자형 변환
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 날짜/시간 처리
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    else:
        df['date_str'] = datetime.now().strftime('%Y-%m-%d')
    
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str).str.zfill(4)
    
    df = df.sort_values(['date_str', 'time']).reset_index(drop=True)
    
    # 일별 시가
    df['day_open'] = df.groupby('date_str')['open'].transform('first')
    
    # ORB High (첫 30분)
    def calc_orb(g):
        return g.head(30)['high'].max()
    
    orb_map = df.groupby('date_str').apply(calc_orb)
    df['orb_high'] = df['date_str'].map(orb_map)
    
    # EMA (Shift 1) - transform 사용
    for span in [5, 20, 50, 200]:
        df[f'ema_{span}'] = df.groupby('date_str', group_keys=False)['close'].transform(
            lambda s: s.ewm(span=span, adjust=False).mean().shift(1)
        )
    
    # SMA (Shift 1) - transform 사용
    for window in [50, 200]:
        df[f'sma_{window}'] = df.groupby('date_str', group_keys=False)['close'].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        )
    
    # VWAP (Shift 1) - 안전한 방식
    df['vwap'] = np.nan

    for date in df['date_str'].unique():
        mask = df['date_str'] == date
        df_day = df[mask].copy()
    
        vol = df_day['volume'].replace(0, 1.0)
        tp = df_day['close']
    
        cumsum_vol_price = (tp * vol).cumsum()
        cumsum_vol = vol.cumsum()
        vwap_values = (cumsum_vol_price / cumsum_vol).shift(1)
    
        df.loc[mask, 'vwap'] = vwap_values.values
    
    # Bollinger Lower Band (Shift 1) - transform 사용
    df['bb_lower'] = df.groupby('date_str', group_keys=False)['close'].transform(
        lambda g: (g.rolling(window=20).mean() - 2 * g.rolling(window=20).std().fillna(0)).shift(1)
    )
    
    # NaN 채우기
    for col in ['vwap', 'ema_200', 'sma_200', 'bb_lower']:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(df['close'].shift(1))
    
    return df

# ==========================================
# 베이스 프라이스 계산
# ==========================================
def get_base_price(df, target_date):
    """
    전날 종가 (옵션 A - 정확한 방법)
    
    Args:
        df: 전체 데이터 (5일치)
        target_date: "2025-12-30"
        
    Returns:
        전날 종가
    """
    df_before = df[df['date_str'] < target_date]
    
    if not df_before.empty:
        return df_before.iloc[-1]['close']
    else:
        # 전날 데이터 없으면 당일 시가
        df_target = df[df['date_str'] == target_date]
        if not df_target.empty:
            return df_target.iloc[0]['open']
        else:
            return np.nan

# ==========================================
# 40% 급등 체크
# ==========================================
def passes_surge_check(df_until_now, base_price, threshold=0.40):
    """
    현재까지 데이터가 40% 이상 급등했는가?
    
    Args: 
        df_until_now: 현재까지의 분봉
        base_price: 전날 종가
        threshold: 급등 기준 (기본 40%)
        
    Returns: 
        bool
    """
    if df_until_now.empty or pd.isna(base_price) or base_price == 0:
        return False
    
    current_price = df_until_now.iloc[-1]['close']
    surge_pct = (current_price - base_price) / base_price
    
    return surge_pct >= threshold

# ==========================================
# 전략별 진입 신호
# ==========================================
def get_entry_signal(df, strategy_name, strategy_params, symbol, row):
    """
    특정 전략의 진입 신호 확인
    
    Args: 
        df: 지표 계산된 DataFrame
        strategy_name: 전략 이름
        strategy_params: 전략 파라미터
        symbol: 종목명
        row: 현재 봉
        
    Returns:
        {'price': float, 'comment': str} 또는 None
    """
    if df.empty or len(df) < 5:
        return None
    
    last_row = df.iloc[-1]
    limit_price = 0
    
    # 전략별 지정가 계산
    if strategy_name == 'NEW_ORB': 
        orb_high = last_row.get('orb_high', np.nan)
        if pd.notna(orb_high) and orb_high > 0:
            limit_price = orb_high
    
    elif strategy_name == 'NEW_PRE': 
        limit_price = last_row.get('day_open', 0)
    
    elif strategy_name == 'ATOM_SUP_EMA5':
        limit_price = last_row.get('ema_5', 0)
    
    elif strategy_name == 'ATOM_SUP_EMA20':
        limit_price = last_row.get('ema_20', 0)
    
    elif strategy_name == 'ATOM_SUP_EMA50':
        limit_price = last_row.get('ema_50', 0)
    
    elif strategy_name == 'ATOM_SUP_EMA200':
        limit_price = last_row.get('ema_200', 0)
    
    elif strategy_name == 'ATOM_SUP_VWAP':
        limit_price = last_row.get('vwap', 0)
    
    elif strategy_name == 'DIP_SNIPER':
        limit_price = last_row.get('bb_lower', 0)
    
    elif strategy_name == 'MOL_CONFLUENCE':
        limit_price = last_row.get('ema_20', 0)
    
    elif strategy_name == 'ROD_A':
        sma_50 = last_row.get('sma_50', 0)
        ema_50 = last_row.get('ema_50', 0)
        if pd.notna(sma_50) and pd.notna(ema_50):
            limit_price = max(sma_50, ema_50)
    
    elif strategy_name == 'ROD_B':
        limit_price = last_row.get('sma_200', 0)
    
    elif strategy_name == 'ROD_C':
        limit_price = last_row.get('sma_50', 0)
    
    # 유효성 체크
    if pd.isna(limit_price) or limit_price <= 0:
        return None
    
    # 진입 조건 (현재 저가가 지정가 터치)
    BUY_TOLERANCE = 1.005
    current_low = last_row['low']
    
    if current_low <= limit_price * BUY_TOLERANCE:
        return {
            'price': limit_price,
            'comment': f"{strategy_name} Signal"
        }
    
    return None

# ==========================================
# 단일 전략 백테스팅 (Reality Mode)
# ==========================================
def backtest_strategy_reality(data_files, strategy_name, initial_cash=10000.0):
    """
    Reality Mode 백테스팅
    - 거래 비용 포함
    - 체결 시뮬레이션
    - 일일 손실 한도
    - Trailing Stop
    
    Args:
        data_files: CSV 파일 리스트
        strategy_name:  테스트할 전략
        initial_cash: 초기 자금
        
    Returns:
        {
            'strategy':  str,
            'daily_results': [...],
            'all_trades': [...],
            'statistics': {...}
        }
    """
    logger.info(f"🏃 [{strategy_name}] Reality Mode 시작...")
    
    # 전략 엔진 초기화
    engine = GapZoneStrategy()
    params = engine.strategies.get(strategy_name, {})
    
    if not params or not params.get('enabled', False):
        logger.warning(f"⚠️ {strategy_name} 비활성화 상태")
        return None
    
    # 전략 파라미터
    tp_pct = params.get('take_profit', 0.10)
    sl_pct = abs(params.get('stop_loss', -0.05))
    trailing_dist = 0.05  # 5% 트레일링 (전략별로 다르게 가능)
    
    # 날짜별 데이터 로드
    calendar = {}
    base_prices = {}
    
    for filepath in data_files:
        try:
            target_date = MarketCalendar.extract_date_from_filename(filepath)
            symbol = MarketCalendar.extract_symbol_from_filename(filepath)
            
            if not target_date:
                continue
            
            df_raw = pd.read_csv(filepath)
            df_with_indicators = compute_indicators(df_raw)
            
            # 날짜별로 그룹화
            for date in df_with_indicators['date_str'].unique():
                df_day = df_with_indicators[df_with_indicators['date_str'] == date]
                
                if df_day.empty:
                    continue
                
                calendar.setdefault(date, {})[symbol] = df_day.set_index('time', drop=False)
                
                # 베이스 프라이스 계산 (전날 종가)
                base_price = get_base_price(df_with_indicators, date)
                base_prices.setdefault(date, {})[symbol] = base_price
        
        except Exception as e: 
            logger.error(f"파일 로드 실패 {filepath}: {e}")
            continue
    
    logger.info(f"📅 총 {len(calendar)}일 데이터 로드 완료")
    
    # 날짜별 시뮬레이션
    daily_results = []
    all_trades = []
    
    for day in sorted(calendar.keys()):
        day_stocks = calendar.get(day, {})
        
        if not day_stocks:
            continue
        
        # 타임라인 생성 (모든 종목의 시간 합집합)
        timeline = sorted(set().union(*[set(df.index) for df in day_stocks.values()]))
        
        # 당일 상태 초기화
        balance = initial_cash
        position = None
        watchlist = set()
        traded_symbols = set()  # One-Shot
        day_trades = []
        
        for timestamp in timeline:
            # 장외 시간 스킵
            if not MarketCalendar.is_market_hours(timestamp):
                continue
            
            # ========================================
            # A. Exit Logic (포지션 보유 ��)
            # ========================================
            if position: 
                sym = position['symbol']
                df_sym = day_stocks.get(sym)
                
                if df_sym is None or timestamp not in df_sym.index:
                    continue
                
                row = df_sym.loc[timestamp]
                curr_close = float(row['close'])
                curr_high = float(row['high'])
                curr_low = float(row['low'])
                
                # Max Price 업데이트
                if curr_high > position['max_price']:
                    position['max_price'] = curr_high
                
                exit_reason = None
                exit_price = curr_close
                urgency = 'normal'
                
                # Exit 조건 (우선순위 순서)
                sl_price = position['entry_price'] * (1 - sl_pct)
                tp_price = position['entry_price'] * (1 + tp_pct)
                
                # 1. Stop Loss (최우선)
                if curr_low <= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                    urgency = 'panic'
                
                # 2. Take Profit
                elif curr_high >= tp_price: 
                    exit_reason = "TP"
                    exit_price = tp_price
                
                # 3. Trailing Stop (수익 구간에서만)
                elif position['max_price'] > position['entry_price']:
                    trail_price = position['max_price'] * (1 - trailing_dist)
                    if curr_low <= trail_price: 
                        exit_reason = "Trailing Stop"
                        exit_price = trail_price
                
                # 4. EOD (장 마감)
                elif timestamp == timeline[-1]:
                    exit_reason = "EOD"
                    exit_price = curr_close
                
                # Exit 실행
                if exit_reason: 
                    # 시장가 매도 시뮬레이션
                    sell_result = ExecutionSimulator.simulate_market_sell(
                        position['qty'],
                        {'close': exit_price, 'low': curr_low, 'high': curr_high},
                        urgency=urgency
                    )
                    
                    actual_exit_price = sell_result['fill_price']
                    actual_qty = sell_result['fill_qty']
                    
                    # 거래 비용
                    exit_cost = TradingCosts.calculate_exit_cost(actual_qty, actual_exit_price)
                    
                    # 수익 계산
                    gross_pnl = (actual_exit_price - position['entry_price']) * actual_qty
                    net_pnl = gross_pnl - position['entry_cost'] - exit_cost
                    
                    # 잔고 업데이트
                    balance += (actual_qty * actual_exit_price - exit_cost)
                    
                    # 거래 기록
                    trade_record = {
                        'date': day,
                        'symbol': sym,
                        'strategy': strategy_name,
                        'entry_price': position['entry_price'],
                        'exit_price': actual_exit_price,
                        'qty': actual_qty,
                        'gross_pnl': round(gross_pnl, 2),
                        'entry_cost': position['entry_cost'],
                        'exit_cost': exit_cost,
                        'net_pnl': round(net_pnl, 2),
                        'exit_reason': exit_reason,
                        'slippage_pct': sell_result['slippage_pct']
                    }
                    
                    day_trades.append(trade_record)
                    all_trades.append(trade_record)
                    
                    logger.info(f"  💰 {sym} Exit @ ${actual_exit_price:.2f} | {exit_reason} | PnL: ${net_pnl:.2f}")
                    
                    # 포지션 청산
                    position = None
                    continue
            
            # ========================================
            # B.  Scanning (10분마다)
            # ========================================
            if MarketCalendar.should_scan_now(timestamp, interval_minutes=10):
                for sym, df_sym in day_stocks.items():
                    if sym in watchlist or sym in traded_symbols:
                        continue
                    
                    if timestamp not in df_sym.index:
                        continue
                    
                    # 40% 급등 체크
                    df_until_now = df_sym[df_sym['time'] <= timestamp]
                    base_price = base_prices.get(day, {}).get(sym, np.nan)
                    
                    if passes_surge_check(df_until_now, base_price, threshold=0.40):
                        watchlist.add(sym)
                        logger.debug(f"  🔭 {sym} 감시 리스트 추가 (40% 급등)")
            
            # ========================================
            # C. Entry Logic (빈손 + Watchlist)
            # ========================================
            if not position and watchlist:
                for sym in sorted(watchlist):
                    # One-Shot 체크
                    if sym in traded_symbols:
                        continue
                    
                    df_sym = day_stocks.get(sym)
                    
                    if df_sym is None or timestamp not in df_sym.index:
                        continue
                    
                    row = df_sym.loc[timestamp]
                    
                    # 신호까지의 데이터만 사용
                    df_for_signal = df_sym[df_sym['time'] <= timestamp]
                    
                    # 진입 신호 확인
                    signal = get_entry_signal(df_for_signal, strategy_name, params, sym, row)
                    
                    if not signal:
                        continue
                    
                    # 체결 시뮬레이션
                    exec_result = ExecutionSimulator.simulate_fill(
                        signal['price'],
                        {
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        },
                        symbol_volatility=0.05,
                        mode='realistic'
                    )
                    
                    if not exec_result['filled']:
                        logger.debug(f"  ⚠️ {sym} 체결 실패:  {exec_result['reason']}")
                        continue
                    
                    # 체결 가격 및 수량
                    actual_entry_price = exec_result['fill_price']
                    requested_qty = int((balance * Config.ALL_IN_RATIO) / actual_entry_price)
                    filled_qty = int(requested_qty * exec_result['fill_qty_pct'])
                    
                    if filled_qty == 0:
                        logger.debug(f"  ⚠️ {sym} 수량 0 (부분체결 {exec_result['fill_qty_pct']*100:.0f}%)")
                        continue
                    
                    # 거래 비용
                    entry_cost = TradingCosts.calculate_entry_cost(filled_qty, actual_entry_price)
                    total_cost = filled_qty * actual_entry_price + entry_cost
                    
                    # 잔고 확인
                    if total_cost > balance:
                        logger.debug(f"  ⚠️ {sym} 잔고 부족 (필요: ${total_cost:.2f}, 보유: ${balance:.2f})")
                        continue
                    
                    # 포지션 생성
                    position = {
                        'symbol': sym,
                        'entry_price': actual_entry_price,
                        'qty': filled_qty,
                        'entry_cost': entry_cost,
                        'max_price': actual_entry_price
                    }
                    
                    # 잔고 차감
                    balance -= total_cost
                    
                    # One-Shot 기록
                    traded_symbols.add(sym)
                    
                    logger.info(f"  🎯 {sym} Entry @ ${actual_entry_price:.2f} | Qty: {filled_qty} ({exec_result['fill_qty_pct']*100:.0f}%)")
                    
                    break  # Single Slot:  한 번에 하나만
        
        # EOD:  미청산 포지션 강제 청산
        if position:
            sym = position['symbol']
            df_sym = day_stocks.get(sym)
            
            if df_sym is not None: 
                final_row = df_sym.iloc[-1]
                final_price = final_row['close']
                
                exit_cost = TradingCosts.calculate_exit_cost(position['qty'], final_price)
                gross_pnl = (final_price - position['entry_price']) * position['qty']
                net_pnl = gross_pnl - position['entry_cost'] - exit_cost
                
                balance += (position['qty'] * final_price - exit_cost)
                
                trade_record = {
                    'date':  day,
                    'symbol':  sym,
                    'strategy':  strategy_name,
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'qty': position['qty'],
                    'gross_pnl': round(gross_pnl, 2),
                    'entry_cost': position['entry_cost'],
                    'exit_cost': exit_cost,
                    'net_pnl': round(net_pnl, 2),
                    'exit_reason': 'EOD (Forced)',
                    'slippage_pct': 0
                }
                
                day_trades.append(trade_record)
                all_trades.append(trade_record)
                
                logger.info(f"  💰 {sym} EOD Exit @ ${final_price:.2f} | PnL: ${net_pnl:.2f}")
        
        # 일일 결과 기록
        day_pnl = balance - initial_cash
        
        daily_results.append({
            'date': day,
            'pnl': round(day_pnl, 2),
            'trades': len(day_trades),
            'final_balance': round(balance, 2),
            'trade_details': day_trades
        })
        
        logger.info(f"📅 {day} | PnL: ${day_pnl:.2f} | Trades: {len(day_trades)} | Balance: ${balance:.2f}")
    
    # 통계 계산
    statistics = BacktestStatistics.calculate_all_metrics(daily_results)
    
    return {
        'strategy': strategy_name,
        'daily_results': daily_results,
        'all_trades': all_trades,
        'statistics':  statistics
    }

# ==========================================
# 메인 실행
# ==========================================
def main():
    print("="*70)
    print("🏆 Reality Mode - 실전 시뮬레이션 (Tier 2)")
    print("="*70)
    print("✅ 거래 비용 포함")
    print("✅ 체결 지연/슬리피지/부분체결")
    print("✅ Trailing Stop")
    print("✅ One-Shot Rule")
    print("="*70)
    
    # 데이터 파일 로드
    data_dir = "data"
    data_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not data_files:
        logger.error("❌ 데이터 파일 없음")
        return
    
    logger.info(f"📂 {len(data_files)}개 파일 발견")
    
    # 전략 리스트
    engine = GapZoneStrategy()
    active_strategies = [
        name for name, params in engine.strategies.items()
        if params.get('enabled', False)
    ]
    
    logger.info(f"🎯 {len(active_strategies)}개 전략 테스트")
    
    # 전략별 실행
    results = []
    
    for strategy_name in active_strategies:
        result = backtest_strategy_reality(data_files, strategy_name, initial_cash=10000.0)
        
        if result: 
            results.append(result)
    
    # 결과 저장
    if results:
        # CSV:  모든 거래
        all_trades_list = []
        for r in results:
            all_trades_list.extend(r['all_trades'])
        
        if all_trades_list:
            df_trades = pd.DataFrame(all_trades_list)
            df_trades.to_csv('results/reality_trades.csv', index=False)
            logger.info(f"✅ 거래 내역 저장:  results/reality_trades.csv ({len(all_trades_list)}건)")
        
        # CSV:  일별 결과
        all_daily_list = []
        for r in results:
            for day_result in r['daily_results']:
                all_daily_list.append({
                    'strategy': r['strategy'],
                    'date': day_result['date'],
                    'pnl': day_result['pnl'],
                    'trades': day_result['trades'],
                    'balance': day_result['final_balance']
                })
        
        if all_daily_list: 
            df_daily = pd.DataFrame(all_daily_list)
            df_daily.to_csv('results/reality_daily.csv', index=False)
            logger.info(f"✅ 일별 결과 저장: results/reality_daily.csv")
        
        # JSON: 통계 요약
        summary = {}
        for r in results: 
            summary[r['strategy']] = r['statistics']
        
        with open('results/reality_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ 통계 요약 저장: results/reality_summary.json")
        
        # 콘솔 출력
        print("\n" + "="*70)
        print("🏆 Reality Mode 결과")
        print("="*70)
        
        for r in results:
            stats = r['statistics']
            print(f"\n📊 {r['strategy']}")
            print(f"  Total PnL:         ${stats['total_pnl']: ,.2f}")
            print(f"  Win Rate:         {stats['win_rate']:.1f}%")
            print(f"  Profit Factor:    {stats['profit_factor']:.2f}")
            print(f"  Max Drawdown:     ${stats['max_drawdown']:,.2f}")
            print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:.2f}")
            print(f"  VaR (5%):         ${stats['var_5pct']:,.2f}")
            print(f"  ⚠️ Top 5 Dependency: {stats['top_5_dependency_pct']:.1f}%")
            print(f"  Max Consec Loss:  {stats['max_consecutive_losses']} days")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    main()