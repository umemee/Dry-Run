# run_championship.py (FIXED VERSION)
import os
import sys
import glob
import json
import logging
import re
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy import GapZoneStrategy
from config import Config
from backtest import MarketCalendar, BacktestStatistics, ExecutionSimulator

# ==========================================
# 로깅 설정 (이모지 제거)
# ==========================================
os.makedirs('results', exist_ok=True)

file_handler = logging.FileHandler('results/championship.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

class NoEmojiFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
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
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# ==========================================
# 지표 계산
# ==========================================
def compute_indicators(df):
    df = df.copy()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    else:
        df['date_str'] = datetime.now().strftime('%Y-%m-%d')
    
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str).str.zfill(4)
    
    df = df.sort_values(['date_str', 'time']).reset_index(drop=True)
    
    df['day_open'] = df.groupby('date_str')['open'].transform('first')
    
    def calc_orb(g):
        return g.head(30)['high'].max()
    orb_map = df.groupby('date_str').apply(calc_orb)
    df['orb_high'] = df['date_str'].map(orb_map)
    
    for span in [5, 20, 50, 200]:
        df[f'ema_{span}'] = df.groupby('date_str', group_keys=False)['close'].transform(
            lambda s: s.ewm(span=span, adjust=False).mean().shift(1)
        )
    
    for window in [50, 200]:
        df[f'sma_{window}'] = df.groupby('date_str', group_keys=False)['close'].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        )
    
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
    
    df['bb_lower'] = df.groupby('date_str', group_keys=False)['close'].transform(
        lambda g: (g.rolling(window=20).mean() - 2 * g.rolling(window=20).std().fillna(0)).shift(1)
    )
    
    for col in ['vwap', 'ema_200', 'sma_200', 'bb_lower']:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(df['close'].shift(1))
    
    return df

def get_base_price(df, target_date):
    df_before = df[df['date_str'] < target_date]
    if not df_before.empty:
        return df_before.iloc[-1]['close']
    else:
        df_target = df[df['date_str'] == target_date]
        if not df_target.empty:
            return df_target.iloc[0]['open']
        else:
            return np.nan

def passes_surge_check(df_until_now, base_price, threshold=0.40):
    if df_until_now.empty or pd.isna(base_price) or base_price == 0:
        return False
    current_price = df_until_now.iloc[-1]['close']
    surge_pct = (current_price - base_price) / base_price
    return surge_pct >= threshold

def get_entry_signal(df, strategy_name, row):
    if df.empty or len(df) < 5:
        return None
    
    last_row = df.iloc[-1]
    limit_price = 0
    
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
    
    if pd.isna(limit_price) or limit_price <= 0:
        return None
    
    BUY_TOLERANCE = 1.005
    current_low = last_row['low']
    
    if current_low <= limit_price * BUY_TOLERANCE:
        return {
            'price': limit_price,
            'comment': f"{strategy_name} Signal"
        }
    
    return None

# ==========================================
# 백테스팅 함수 (수정됨)
# ==========================================
def backtest_strategy_championship(data_files, strategy_name, initial_cash=10000.0):
    logger.info(f"[{strategy_name}] Championship Mode 시작...")
    
    engine = GapZoneStrategy()
    params = engine.strategies.get(strategy_name, {})
    
    if not params or not params.get('enabled', False):
        logger.warning(f"{strategy_name} 비활성화")
        return None
    
    tp_pct = params.get('take_profit', 0.10)
    sl_pct = abs(params.get('stop_loss', -0.05))
    
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
            
            for date in df_with_indicators['date_str'].unique():
                df_day = df_with_indicators[df_with_indicators['date_str'] == date]
                
                if df_day.empty:
                    continue
                
                calendar.setdefault(date, {})[symbol] = df_day.set_index('time', drop=False)
                base_price = get_base_price(df_with_indicators, date)
                base_prices.setdefault(date, {})[symbol] = base_price
        except Exception:
            continue
    
    logger.info(f"[{strategy_name}] {len(calendar)}일 데이터 로드")
    
    daily_results = []
    all_trades = []  # ✅ 수정: 변수명 확인
    
    for day in sorted(calendar.keys()):
        day_stocks = calendar.get(day, {})
        if not day_stocks:
            continue
        
        timeline = sorted(set().union(*[set(df.index) for df in day_stocks.values()]))
        
        balance = initial_cash
        position = None
        watchlist = set()
        traded_symbols = set()  # ✅ One-Shot
        day_trades = []
        
        for timestamp in timeline:
            if not MarketCalendar.is_market_hours(timestamp):
                continue
            
            # Exit
            if position:
                sym = position['symbol']
                df_sym = day_stocks.get(sym)
                
                if df_sym is None or timestamp not in df_sym.index:
                    continue
                
                row = df_sym.loc[timestamp]
                curr_close = float(row['close'])
                curr_high = float(row['high'])
                curr_low = float(row['low'])
                
                if curr_high > position['max_price']:
                    position['max_price'] = curr_high
                
                exit_reason = None
                exit_price = curr_close
                
                sl_price = position['entry_price'] * (1 - sl_pct)
                tp_price = position['entry_price'] * (1 + tp_pct)
                
                if curr_low <= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                elif curr_high >= tp_price:
                    exit_reason = "TP"
                    exit_price = tp_price
                elif timestamp == timeline[-1]:
                    exit_reason = "EOD"
                    exit_price = curr_close
                
                if exit_reason:
                    pnl = (exit_price - position['entry_price']) * position['qty']
                    balance += (position['qty'] * exit_price)
                    
                    trade_record = {
                        'date': day,
                        'symbol': sym,
                        'strategy': strategy_name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'qty': position['qty'],
                        'net_pnl': round(pnl, 2),
                        'exit_reason': exit_reason
                    }
                    
                    day_trades.append(trade_record)
                    all_trades.append(trade_record)  # ✅ 수정: 올바른 메서드 호출
                    
                    position = None
                    continue
            
            # Scanning
            if MarketCalendar.should_scan_now(timestamp, interval_minutes=10):
                for sym, df_sym in day_stocks.items():
                    if sym in watchlist or sym in traded_symbols:
                        continue
                    
                    if timestamp not in df_sym.index:
                        continue
                    
                    df_until_now = df_sym[df_sym['time'] <= timestamp]
                    base_price = base_prices.get(day, {}).get(sym, np.nan)
                    
                    if passes_surge_check(df_until_now, base_price, threshold=0.40):
                        watchlist.add(sym)
            
            # Entry
            if not position and watchlist:
                for sym in sorted(watchlist):
                    if sym in traded_symbols:  # ✅ One-Shot 체크
                        continue
                    
                    df_sym = day_stocks.get(sym)
                    if df_sym is None or timestamp not in df_sym.index:
                        continue
                    
                    row = df_sym.loc[timestamp]
                    df_for_signal = df_sym[df_sym['time'] <= timestamp]
                    
                    signal = get_entry_signal(df_for_signal, strategy_name, row)
                    if not signal:
                        continue
                    
                    entry_price = signal['price']
                    qty = int((balance * 0.98) / entry_price)
                    
                    if qty > 0:
                        position = {
                            'symbol': sym,
                            'entry_price': entry_price,
                            'qty': qty,
                            'max_price': entry_price
                        }
                        
                        balance -= (qty * entry_price)
                        traded_symbols.add(sym)  # ✅ One-Shot 기록
                        break
        
        # EOD
        if position:
            sym = position['symbol']
            df_sym = day_stocks.get(sym)
            
            if df_sym is not None:
                final_row = df_sym.iloc[-1]
                final_price = final_row['close']
                
                pnl = (final_price - position['entry_price']) * position['qty']
                balance += (position['qty'] * final_price)
                
                trade_record = {
                    'date': day,
                    'symbol': sym,
                    'strategy': strategy_name,
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'qty': position['qty'],
                    'net_pnl': round(pnl, 2),
                    'exit_reason': 'EOD'
                }
                
                day_trades.append(trade_record)
                all_trades.append(trade_record)  # ✅ 수정
        
        day_pnl = balance - initial_cash
        daily_results.append({
            'date': day,
            'pnl': round(day_pnl, 2),
            'trades': len(day_trades),
            'final_balance': round(balance, 2),
            'trade_details': day_trades
        })
    
    statistics = BacktestStatistics.calculate_all_metrics(daily_results)
    
    return {
        'strategy': strategy_name,
        'daily_results': daily_results,
        'all_trades': all_trades,
        'statistics': statistics
    }

def main():
    print("="*70)
    print("Championship Mode - FIXED")
    print("="*70)
    
    data_dir = "data"
    data_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not data_files:
        logger.error("데이터 파일 없음")
        return
    
    logger.info(f"{len(data_files)}개 파일 발견")
    
    engine = GapZoneStrategy()
    active_strategies = [
        name for name, params in engine.strategies.items()
        if params.get('enabled', False)
    ]
    
    logger.info(f"{len(active_strategies)}개 전략 테스트")
    
    results = []
    for strategy_name in active_strategies:
        result = backtest_strategy_championship(data_files, strategy_name, initial_cash=10000.0)
        if result:
            results.append(result)
    
    if results:
        all_trades_list = []
        for r in results:
            all_trades_list.extend(r['all_trades'])
        
        if all_trades_list:
            df_trades = pd.DataFrame(all_trades_list)
            df_trades.to_csv('results/championship_trades_fixed.csv', index=False)
            logger.info(f"거래 내역 저장: results/championship_trades_fixed.csv")
        
        summary = {}
        for r in results:
            summary[r['strategy']] = r['statistics']
        
        with open('results/championship_summary_fixed.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*70)
        print("Championship 결과 (PnL 순)")
        print("="*70)
        
        results_sorted = sorted(results, key=lambda x: x['statistics']['total_pnl'], reverse=True)
        
        for rank, r in enumerate(results_sorted, 1):
            stats = r['statistics']
            print(f"\n{rank}. {r['strategy']}")
            print(f"  Total PnL:     ${stats['total_pnl']:,.2f}")
            print(f"  Win Rate:     {stats['win_rate']:.1f}%")
            print(f"  Trades:       {stats['total_trades']}")

if __name__ == "__main__":
    main()
