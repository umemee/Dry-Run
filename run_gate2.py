"""
🚪 Gate 2: Execution Reality Test
목적: 슬리피지 + 체결지연 반영 후 전략 생존 여부 판정

[추가] 슬리피지 모델 (0.1% ~ 0.5%)
[추가] 체결 지연 (1~3틱)
[추가] 부분 체결 확률 (30%)
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import random

warnings.filterwarnings("ignore")


# ==========================================
# 📝 Logger & Cost Calculator (Gate 1과 동일)
# ==========================================
class SimpleLogger:
    def __init__(self, name): self.name = name
    def info(self, msg): print(f"[{self.name}] {msg}")
    def error(self, msg): print(f"❌ [{self.name}] {msg}")

logger = SimpleLogger("Gate2")

class CostCalculator:
    COMMISSION_RATE = 0.0001
    SEC_FEE_RATE = 0.0000278
    
    @classmethod
    def calculate(cls, side, price, qty):
        notional = price * qty
        commission = notional * cls.COMMISSION_RATE
        sec_fee = notional * cls.SEC_FEE_RATE if side == 'SELL' else 0
        return commission + sec_fee


# ==========================================
# 🎲 슬리피지 & 체결 모델
# ==========================================
class ExecutionSimulator:
    """실전 체결 환경 시뮬레이터"""
    
    @staticmethod
    def apply_slippage(price, side, volatility_factor=1.0):
        """
        슬리피지 적용
        - 매수: 불리하게 (가격 상승)
        - 매도: 불리하게 (가격 하락)
        - 변동성 비례: 급등주일수록 슬리피지 증가
        """
        # 기본 슬리피지:  0.1% ~ 0.5%
        base_slip = random.uniform(0.001, 0.005)
        # 변동성 가중치 (급등주는 최대 2배)
        slip_rate = base_slip * volatility_factor
        
        if side == 'BUY': 
            return price * (1 + slip_rate)  # 더 비싸게 산다
        else:  # SELL
            return price * (1 - slip_rate)  # 더 싸게 판다
    
    @staticmethod
    def get_fill_delay():
        """
        체결 지연 시간 (틱 단위)
        - 70%:  1틱 지연
        - 20%: 2틱 지연
        - 10%:  3틱 지연
        """
        rand = random.random()
        if rand < 0.7:
            return 1
        elif rand < 0.9:
            return 2
        else:
            return 3
    
    @staticmethod
    def is_partial_fill():
        """
        부분 체결 여부
        - 30% 확률로 주문 수량의 50~80%만 체결
        """
        return random.random() < 0.3
    
    @staticmethod
    def get_fill_ratio():
        """부분 체결 시 체결 비율"""
        return random.uniform(0.5, 0.8)


# ==========================================
# 📊 지표 계산 (Gate 1과 동일)
# ==========================================
def compute_indicators_for_df(df):
    df = df.copy()
    
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 1.0 if col == "volume" else np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    else:
        df["date_str"] = datetime.now().strftime("%Y%m%d")

    if "time" in df.columns:
        df["time"] = df["time"].astype(str).apply(lambda x: x.split('.')[0].zfill(4))
    else:
        df["time"] = "0000"

    df = df.sort_values(["date_str", "time"]).reset_index(drop=True)

    # 지표 계산 (간소화)
    df["day_open"] = df.groupby("date_str")["open"].transform("first")
    
    # ORB
    orb_map = df.groupby("date_str").apply(lambda x: x.head(30)["high"].max())
    df["orb_high"] = df["date_str"].map(orb_map)

    # EMA
    for span in [200]:   # Gate 2에서는 필요한 것만
        df[f"ema_{span}"] = df.groupby("date_str")["close"].transform(
            lambda s: s.ewm(span=span, adjust=False).mean().shift(1)
        )

    # SMA
    for window in [200]:
        df[f"sma_{window}"] = df.groupby("date_str")["close"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # NaN 처리
    df["ema_200"] = df["ema_200"].ffill().fillna(df["close"].shift(1))
    df["sma_200"] = df["sma_200"].ffill().fillna(df["close"].shift(1))

    return df


# ==========================================
# 🏃 Gate 2 Main Runner
# ==========================================
def run_gate2():
    print("="*70)
    print("🚪 GATE 2: Execution Reality Test")
    print("   [추가] 슬리피지 + 체결지연 + 부분체결")
    print("   [대상] 상위 3개 전략만 테스트")
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
        except: pass

    print(f"📦 로드된 종목: {len(symbol_raw)}개")
    print("⚙️ 지표 계산 중...")

    # 캘린더 구성
    calendar = {}
    prev_day_closes = {}
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
            logger.info(f"지표 계산 실패:  {sym} ({e})")

    logger.info(f"처리 완료: {processed_count}개")

    sorted_days = sorted(calendar.keys())
    if not sorted_days:
        logger.error("처리 가능한 데이터가 없습니다")
        return
        
    print(f"📅 테스트 기간: {sorted_days[0]} ~ {sorted_days[-1]} ({len(sorted_days)}일)")

    # === 전략 설정 (상위 3개만) ===
    strategies = {
        "NEW_ORB": {
            "enabled": True,
            "indicator": "orb_high",
            "stop_loss": 0.04,
            "take_profit":  0.15
        },
        "ATOM_SUP_EMA200": {
            "enabled": True,
            "indicator": "ema_200",
            "stop_loss": 0.05,
            "take_profit":  0.10
        },
        "ROD_B": {
            "enabled": True,
            "indicator":  "sma_200",
            "stop_loss": 0.08,
            "take_profit":  0.10
        }
    }

    exec_sim = ExecutionSimulator()
    leaderboard = []

    # === 전략별 백테스팅 ===
    for strat_name, config in strategies.items():
        if not config["enabled"]:
            continue

        sl_pct = config["stop_loss"]
        tp_pct = config["take_profit"]
        indicator_col = config["indicator"]

        total_pnl = 0.0
        total_costs = 0.0
        trade_count = 0
        win_count = 0
        scan_count = 0
        partial_fills = 0
        slippage_loss = 0.0

        # 전일 종가 초기화
        last_closes = {}

        for day in sorted_days: 
            day_stocks = calendar.get(day, {})
            if not day_stocks:
                continue
            
            timeline = sorted(set().union(*(d.index for d in day_stocks.values())))
            
            watchlist = set()
            position = None
            balance = 10000.0
            pending_order = None  # 체결 대기 주문

            # 당일 기준가 설정
            daily_base_prices = {}
            for sym, df_sym in day_stocks.items():
                if sym in last_closes:
                    daily_base_prices[sym] = last_closes[sym]
                else:
                    daily_base_prices[sym] = df_sym.iloc[0]["open"]
                
                last_closes[sym] = df_sym.iloc[-1]["close"]

            for tick_idx, t in enumerate(timeline):
                
                # [0] 대기 중인 주문 체결 처리
                if pending_order and tick_idx >= pending_order["fill_tick"]:
                    sym = pending_order["symbol"]
                    df_sym = day_stocks.get(sym)
                    
                    if df_sym is not None and t in df_sym.index:
                        row = df_sym.loc[t]
                        curr_p = float(row["close"])
                        
                        # 변동성 계산 (급등률)
                        base_p = daily_base_prices.get(sym, curr_p)
                        volatility = (curr_p - base_p) / base_p if base_p > 0 else 0
                        vol_factor = min(1 + abs(volatility) * 2, 2.0)  # 최대 2배
                        
                        # 슬리피지 적용
                        fill_price = exec_sim.apply_slippage(
                            pending_order["target_price"], 
                            'BUY',
                            vol_factor
                        )
                        
                        # 부분 체결 확인
                        qty = pending_order["qty"]
                        if exec_sim.is_partial_fill():
                            fill_ratio = exec_sim.get_fill_ratio()
                            qty = int(qty * fill_ratio)
                            partial_fills += 1
                        
                        if qty > 0:
                            entry_cost = CostCalculator.calculate('BUY', fill_price, qty)
                            slippage_loss += (fill_price - pending_order["target_price"]) * qty
                            
                            position = {
                                "symbol":  sym,
                                "entry":  fill_price,
                                "qty": qty,
                                "max_price": fill_price,
                                "entry_cost": entry_cost
                            }
                    
                    pending_order = None  # 체결 완료

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
                        # 매도 시에도 슬리피지 적용
                        base_p = daily_base_prices.get(sym, curr_close)
                        volatility = (curr_close - base_p) / base_p if base_p > 0 else 0
                        vol_factor = min(1 + abs(volatility) * 2, 2.0)
                        
                        actual_exit = exec_sim.apply_slippage(exit_price, 'SELL', vol_factor)
                        sell_cost = CostCalculator.calculate('SELL', actual_exit, position["qty"])
                        
                        slippage_loss += (exit_price - actual_exit) * position["qty"]
                        
                        pnl = (actual_exit - position["entry"]) * position["qty"]
                        net_pnl = pnl - position["entry_cost"] - sell_cost
                        
                        total_pnl += net_pnl
                        total_costs += (position["entry_cost"] + sell_cost)
                        trade_count += 1
                        
                        if net_pnl > 0:
                            win_count += 1
                        
                        position = None
                    continue

                # [B] Scanning Logic
                for sym, df_sym in day_stocks.items():
                    if sym in watchlist or t not in df_sym.index:
                        continue
                    
                    curr_p = float(df_sym.loc[t]["close"])
                    base_p = daily_base_prices.get(sym, curr_p)
                    
                    if base_p > 0 and (curr_p - base_p) / base_p >= 0.40:
                        watchlist.add(sym)
                        scan_count += 1

                # [C] Entry Logic (체결 지연 적용)
                if not position and not pending_order:
                    for sym in sorted(watchlist):
                        df_sym = day_stocks.get(sym)
                        if df_sym is None or t not in df_sym.index:
                            continue
                        
                        row = df_sym.loc[t]
                        limit_price = row.get(indicator_col, np.nan)

                        if pd.isna(limit_price) or limit_price <= 0:
                            continue

                        if float(row["low"]) <= limit_price * 1.005:
                            entry_exec = min(limit_price, float(row["open"]))
                            qty = int((balance * 0.98) / entry_exec)
                            
                            if qty > 0:
                                # 체결 지연 적용
                                delay_ticks = exec_sim.get_fill_delay()
                                fill_tick = tick_idx + delay_ticks
                                
                                pending_order = {
                                    "symbol": sym,
                                    "target_price": entry_exec,
                                    "qty": qty,
                                    "fill_tick": fill_tick
                                }
                                break

        # 결과 집계
        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
        
        print(f"👉 {strat_name: <20} | Net PnL: ${total_pnl:>10,.2f} | Trades: {trade_count:>4} | Win:  {win_rate:>5.1f}% | 부분체결:{partial_fills} | 슬리피지손실: ${slippage_loss:>8,.2f}")
        
        leaderboard.append({
            "Strategy": strat_name,
            "Net_PnL": round(total_pnl, 2),
            "Gate1_PnL": {
                "NEW_ORB": 145498.63,
                "ATOM_SUP_EMA200": 67181.16,
                "ROD_B": 53183.63
            }[strat_name],
            "Degradation": 0,  # 나중에 계산
            "Trades": trade_count,
            "Win_Rate": f"{win_rate:.1f}%",
            "Partial_Fills": partial_fills,
            "Slippage_Loss": round(slippage_loss, 2)
        })

    # === 최종 리포트 ===
    if leaderboard:
        for item in leaderboard:
            item["Degradation"] = round(
                (item["Gate1_PnL"] - item["Net_PnL"]) / item["Gate1_PnL"] * 100, 1
            )
        
        df_res = pd.DataFrame(leaderboard).sort_values("Net_PnL", ascending=False)
        
        print("\n" + "="*70)
        print("🏆 GATE 2 RESULTS (슬리피지 + 체결지연 반영)")
        print("="*70)
        print(df_res[["Strategy", "Net_PnL", "Gate1_PnL", "Degradation", "Trades", "Win_Rate"]].to_string(index=False))
        print("="*70)
        
        df_res.to_csv("gate2_results.csv", index=False)
        
        # 생존 판정
        survivors = df_res[df_res["Net_PnL"] > 0]
        
        print(f"\n✅ Gate 2 생존 전략: {len(survivors)}/{len(leaderboard)}개")
        if len(survivors) > 0:
            print("   → Gate 3 (일일 리셋 모드) 진행 가능")
        else:
            print("   ❌ 모든 전략 수익 소멸 → 전략 재설계 필요")


if __name__ == "__main__":
    run_gate2()