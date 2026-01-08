"""
🚪 Gate 1: Truth Extraction Stage (FINAL PRODUCTION)
목적: 현실적인 백테스트 - 갭상승 + 장중 급등 모두 포착

[FIX] 스캔 조건:  전일 종가 대비 10% 이상 (실시간)
[FIX] 시간 필터 제거 (모든 틱에서 감지)
[FIX] 전일 종가 체인 로직 추가
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 📝 Simple Logger
# ==========================================
class SimpleLogger: 
    def __init__(self, name):
        self.name = name
    
    def info(self, msg):
        print(f"[{self.name}] {msg}")
    
    def warning(self, msg):
        print(f"⚠️  [{self.name}] {msg}")
    
    def error(self, msg):
        print(f"❌ [{self.name}] {msg}")


logger = SimpleLogger("Gate1")


# ==========================================
# 💰 CostCalculator
# ==========================================
class CostCalculator:
    COMMISSION_RATE = 0.0001  # 0.01%
    SEC_FEE_RATE = 0.0000278  # SEC Fee
    
    @classmethod
    def calculate(cls, side, price, qty):
        notional = price * qty
        commission = notional * cls.COMMISSION_RATE
        
        sec_fee = 0
        if side == 'SELL':
            sec_fee = notional * cls.SEC_FEE_RATE
        
        return commission + sec_fee


# ==========================================
# 📊 지표 계산 (간소화 버전)
# ==========================================
def compute_indicators_for_df(df):
    """
    실전 전략에 필요한 최소 지표만 계산
    """
    df = df.copy()
    
    # 기본 컬럼 숫자형 변환
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 1.0 if col == "volume" else np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 날짜/시간 처리
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    else:
        df["date_str"] = datetime.now().strftime("%Y%m%d")

    # 시간 컬럼 정규화 (549 -> 0549)
    if "time" in df.columns:
        df["time"] = df["time"].astype(str).apply(lambda x: x.split('.')[0].zfill(4))
    else:
        df["time"] = "0000"

    # 정렬
    df = df.sort_values(["date_str", "time"]).reset_index(drop=True)

    # === 지표 계산 ===
    df["day_open"] = df.groupby("date_str")["open"].transform("first")
    
    # ORB High (오전 첫 30봉 고가)
    def calc_orb_high(g):
        return g.head(30)["high"].max()
    orb_map = df.groupby("date_str").apply(calc_orb_high)
    df["orb_high"] = df["date_str"].map(orb_map)

    # EMA (단순화)
    for span in [5, 20, 50, 200]:
        df[f"ema_{span}"] = df.groupby("date_str")["close"].transform(
            lambda s: s.ewm(span=span, adjust=False).mean().shift(1)
        )

    # SMA
    for window in [50, 200]:
        df[f"sma_{window}"] = df.groupby("date_str")["close"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # VWAP (간소화)
    df["vwap"] = df.groupby("date_str").apply(
        lambda g: ((g["close"] * g["volume"]).cumsum() / g["volume"].cumsum()).shift(1)
    ).reset_index(level=0, drop=True)

    # Bollinger Lower Band
    df["sma_20"] = df.groupby("date_str")["close"].transform(
        lambda s: s.rolling(window=20).mean()
    )
    df["std_20"] = df.groupby("date_str")["close"].transform(
        lambda s: s.rolling(window=20).std().fillna(0)
    )
    df["bb_lower"] = (df["sma_20"] - 2 * df["std_20"]).shift(1)
    df.drop(columns=["sma_20", "std_20"], inplace=True)

    # NaN 처리
    cols_to_fill = ["vwap", "ema_200", "sma_200", "bb_lower"]
    for c in cols_to_fill: 
        if c in df.columns:
            df[c] = df[c].ffill().fillna(df["close"].shift(1))

    return df


# ==========================================
# 📋 전략 설정
# ==========================================
def get_strategy_configs():
    return {
        'NEW_PRE':  {'enabled': True, 'priority': 1, 'stop_loss': -0.05, 'take_profit': 0.07},
        'ATOM_SUP_EMA200': {'enabled': True, 'priority': 2, 'stop_loss': -0.05, 'take_profit': 0.10},
        'NEW_ORB':  {'enabled': True, 'priority': 3, 'stop_loss': -0.04, 'take_profit': 0.15},
        'DIP_SNIPER': {'enabled': True, 'priority': 4, 'stop_loss': -0.05, 'take_profit': 0.10},
        'ROD_B': {'enabled': True, 'priority': 5, 'stop_loss': -0.08, 'take_profit': 0.10},
    }


# ==========================================
# 🏃 Gate 1 Main Runner
# ==========================================
def run_gate1():
    print("="*70)
    print("🚪 GATE 1: Truth Extraction (FINAL PRODUCTION)")
    print("   [수정] 전일 종가 기준 40% 실시간 감지")
    print("="*70)

    # 데이터 로드
    files = glob.glob("data/*.csv")
    if not files:
        logger.error("데이터 파일이 없습니다.")
        return

    symbol_raw = {}
    for f in files:
        base = os.path.basename(f).replace(".csv", "")
        parts = base.split("_")
        sym = parts[1] if (len(parts) >= 2 and parts[0].isdigit()) else parts[0]
        try:
            symbol_raw[sym] = pd.read_csv(f)
        except Exception as e:
            logger.warning(f"파일 로드 실패: {f} ({e})")

    print(f"📦 로드된 종목: {len(symbol_raw)}개")
    print("⚙️ 지표 계산 중...")

    # 캘린더 구성
    calendar = {}
    last_closes = {}  # 🔑 핵심:  전일 종가 메모리
    processed_count = 0

    for sym, df in symbol_raw.items():
        try:
            df_inds = compute_indicators_for_df(df)
            
            for day in df_inds["date_str"].unique():
                df_day = df_inds[df_inds["date_str"] == day]
                if df_day.empty or len(df_day) < 5:
                    continue
                
                calendar.setdefault(day, {})[sym] = df_day.set_index("time", drop=False)
                processed_count += 1
                
        except Exception as e:
            logger.warning(f"지표 계산 실패: {sym} ({e})")

    logger.info(f"처리 완료: {processed_count}개")

    sorted_days = sorted(calendar.keys())
    if not sorted_days:
        logger.error("처리 가능한 데이터가 없습니다")
        return
        
    print(f"📅 테스트 기간: {sorted_days[0]} ~ {sorted_days[-1]} ({len(sorted_days)}일)")

    # 전략 매핑
    strategy_map = {
        "NEW_ORB": "orb_high",
        "NEW_PRE": "day_open",
        "ATOM_SUP_EMA5": "ema_5",
        "ATOM_SUP_EMA20": "ema_20",
        "ATOM_SUP_EMA50": "ema_50",
        "ATOM_SUP_EMA200": "ema_200",
        "ATOM_SUP_VWAP": "vwap",
        "DIP_SNIPER": "bb_lower",
        "ROD_A": "sma_50",
        "ROD_B": "sma_200",
    }

    strategies_config = get_strategy_configs()
    active_strategies = [s for s in strategy_map.keys() if strategies_config.get(s, {}).get('enabled', True)]
    leaderboard = []

    # === 전략별 백테스팅 ===
    for strat_name in active_strategies:
        config = strategies_config.get(strat_name, {})
        sl_pct = abs(config.get('stop_loss', -0.05))
        tp_pct = config.get('take_profit', 0.10)

        total_pnl = 0.0
        total_costs = 0.0
        trade_count = 0
        win_count = 0
        scan_count = 0
        entry_attempts = 0

        # 전일 종가 초기화
        prev_day_closes = {}

        for day in sorted_days:
            day_stocks = calendar.get(day, {})
            if not day_stocks:
                continue
            
            timeline = sorted(set().union(*(d.index for d in day_stocks.values())))
            
            watchlist = set()
            position = None
            balance = 10000.0

            # === [핵심] 당일 기준가 설정 ===
            daily_base_prices = {}
            for sym, df_sym in day_stocks.items():
                # 전일 종가가 있으면 사용, 없으면 당일 시가
                if sym in prev_day_closes:
                    daily_base_prices[sym] = prev_day_closes[sym]
                else:
                    daily_base_prices[sym] = df_sym.iloc[0]["open"]
                
                # 오늘 종가를 내일을 위해 저장
                prev_day_closes[sym] = df_sym.iloc[-1]["close"]

            for t in timeline:
                # [A] Exit Logic
                if position: 
                    sym = position["symbol"]
                    df_sym = day_stocks.get(sym)
                    if df_sym is None or t not in df_sym.index:
                        continue
                    
                    row = df_sym.loc[t]
                    curr_close = float(row["close"])
                    curr_high = float(row["high"])
                    curr_low = float(row["low"])
                    
                    if curr_high > position["max_price"]:
                        position["max_price"] = curr_high

                    exit_reason = None
                    exit_price = curr_close

                    sl_price = position["entry"] * (1 - sl_pct)
                    tp_price = position["entry"] * (1 + tp_pct)

                    if curr_low <= sl_price: 
                        exit_reason = "SL"
                        exit_price = sl_price
                    elif curr_high >= tp_price: 
                        exit_reason = "TP"
                        exit_price = tp_price
                    elif t == timeline[-1]:
                        exit_reason = "EOD"
                        exit_price = curr_close

                    if exit_reason:
                        sell_cost = CostCalculator.calculate('SELL', exit_price, position["qty"])
                        
                        pnl = (exit_price - position["entry"]) * position["qty"]
                        net_pnl = pnl - position["entry_cost"] - sell_cost
                        
                        total_pnl += net_pnl
                        total_costs += (position["entry_cost"] + sell_cost)
                        trade_count += 1
                        
                        if net_pnl > 0:
                            win_count += 1
                        
                        position = None
                    continue

                # [B] Scanning Logic (🔥 수정:  모든 틱에서 검사)
                for sym, df_sym in day_stocks.items():
                    if sym in watchlist or t not in df_sym.index:
                        continue
                    
                    curr_p = float(df_sym.loc[t]["close"])
                    base_p = daily_base_prices.get(sym, curr_p)
                    
                    # 🔥 핵심:  전일 종가 대비 40% 이상 (Gap or Intraday)
                    if base_p > 0 and (curr_p - base_p) / base_p >= 0.40:
                        watchlist.add(sym)
                        scan_count += 1

                # [C] Entry Logic
                if not position:
                    for sym in sorted(watchlist):
                        df_sym = day_stocks.get(sym)
                        if df_sym is None or t not in df_sym.index:
                            continue
                        
                        row = df_sym.loc[t]
                        
                        limit_col = strategy_map.get(strat_name)
                        limit_price = np.nan
                        
                        try:
                            if strat_name == "ROD_A":
                                v1 = row.get("sma_50", np.nan)
                                v2 = row.get("ema_50", np.nan)
                                if pd.notna(v1) or pd.notna(v2):
                                    limit_price = max(v1 if pd.notna(v1) else 0, v2 if pd.notna(v2) else 0)
                            else:
                                limit_price = row.get(limit_col, np.nan)
                        except:
                            pass

                        if pd.isna(limit_price) or limit_price <= 0:
                            continue

                        entry_attempts += 1

                        if float(row["low"]) <= limit_price * 1.005:
                            entry_exec = min(limit_price, float(row["open"]))
                            qty = int((balance * 0.98) / entry_exec)
                            
                            if qty > 0:
                                entry_cost = CostCalculator.calculate('BUY', entry_exec, qty)
                                
                                position = {
                                    "symbol": sym,
                                    "entry": entry_exec,
                                    "qty": qty,
                                    "max_price": entry_exec,
                                    "entry_cost": entry_cost
                                }
                                break

        # 결과 집계
        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
        
        print(f"👉 {strat_name:<20} | Net PnL: ${total_pnl:>10,.2f} | Costs: ${total_costs:>8,.2f} | Trades: {trade_count:>4} | Win:  {win_rate:>5.1f}% | 스캔:{scan_count} | 진입시도:{entry_attempts}")
        
        leaderboard.append({
            "Strategy": strat_name,
            "Net_PnL": round(total_pnl, 2),
            "Total_Costs": round(total_costs, 2),
            "Trades": trade_count,
            "Win_Rate": f"{win_rate:.1f}%",
            "Scans": scan_count
        })

    # === 최종 리포트 ===
    if leaderboard:
        df_res = pd.DataFrame(leaderboard).sort_values("Net_PnL", ascending=False)
        
        print("\n" + "="*70)
        print("🏆 GATE 1 RESULTS (현실 반영 버전)")
        print("="*70)
        print(df_res.to_string(index=False))
        print("="*70)
        
        df_res.to_csv("gate1_results.csv", index=False)
        
        if df_res.head(3)["Net_PnL"].min() > 0:
            print("\n✅ Gate 1 통과:  상위 3개 전략 수익 유지")
        else:
            print("\n⚠️  Gate 1: 일부 전략 수익 소멸 (추가 최적화 필요)")


if __name__ == "__main__":
    run_gate1()