import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    from config import Config
    Config.MODE = "SIMULATION"
except Exception:
    pass

from infra.utils import get_logger

logger = get_logger("Championship") if "get_logger" in globals() else None
RESULT_FILE = "championship_final_report.txt"

# ------------------------
# Utility Functions
# ------------------------
def safe_to_datetime_series(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.to_datetime(s.astype(str), errors="coerce")

def compute_indicators_for_df(df):
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 1.0 if col == "volume" else np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = safe_to_datetime_series(df["date"])
        df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    else:
        df["date_str"] = datetime.now().strftime("%Y%m%d")

    if "time" in df.columns:
        df["time"] = df["time"].astype(str).str.zfill(4)
    else:
        df["time"] = df.groupby("date_str").cumcount().astype(str).str.zfill(4)

    df = df.sort_values(["date_str", "time"]).reset_index(drop=True)

    # Indicators (Shifted by 1 for look-ahead bias prevention)
    df["day_open"] = df.groupby("date_str")["open"].transform("first")
    
    def calc_orb_high(g):
        return g.head(30)["high"].max()
    orb_map = df.groupby("date_str").apply(calc_orb_high)
    df["orb_high"] = df["date_str"].map(orb_map)

    df["ema_5"] = df.groupby("date_str")["close"].transform(lambda s: s.ewm(span=5, adjust=False).mean().shift(1))
    df["ema_20"] = df.groupby("date_str")["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean().shift(1))
    df["ema_50"] = df.groupby("date_str")["close"].transform(lambda s: s.ewm(span=50, adjust=False).mean().shift(1))
    df["ema_200"] = df.groupby("date_str")["close"].transform(lambda s: s.ewm(span=200, adjust=False).mean().shift(1))
    df["sma_50"] = df.groupby("date_str")["close"].transform(lambda s: s.rolling(window=50, min_periods=1).mean().shift(1))
    df["sma_200"] = df.groupby("date_str")["close"].transform(lambda s: s.rolling(window=200, min_periods=1).mean().shift(1))

    def per_day_vwap(g):
        vol = g["volume"].replace(0, np.nan).fillna(0.0)
        tp = g["close"]
        return ((tp * vol).cumsum() / vol.cumsum()).shift(1)
    df["vwap"] = df.groupby("date_str").apply(per_day_vwap).reset_index(level=0, drop=True)

    def bb_lower(g):
        ma = g["close"].rolling(window=20).mean()
        sd = g["close"].rolling(window=20).std().fillna(0.0)
        return (ma - 2 * sd).shift(1)
    df["bb_lower"] = df.groupby("date_str").apply(bb_lower).reset_index(level=0, drop=True)

    cols_to_fill = ["vwap", "ema_200", "sma_200", "bb_lower"]
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = df[c].fillna(method="ffill").fillna(df["close"].shift(1))

    return df

def run_championship():
    print("🚀 Championship 시뮬레이션 시작 (Syntax Error Fixed)")

    files = glob.glob("data/*.csv")
    if not files:
        print("❌ 데이터 파일이 없습니다.")
        return

    symbol_raw = {}
    for f in files:
        base = os.path.basename(f).replace(".csv", "")
        parts = base.split("_")
        sym = parts[1] if (len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 8) else parts[0]
        try:
            symbol_raw[sym] = pd.read_csv(f)
        except Exception: pass

    print(f"📦 로드된 종목: {len(symbol_raw)}개")
    print("⚙️ 지표 계산 중...")

    calendar = {}
    base_prices = {}

    for sym, df in symbol_raw.items():
        try:
            df_inds = compute_indicators_for_df(df)
            for day in df_inds["date_str"].unique():
                df_day = df_inds[df_inds["date_str"] == day]
                if df_day.empty: continue
                calendar.setdefault(day, {})[sym] = df_day.set_index("time", drop=False)
                
                prev = df_inds[df_inds["date_str"] < day]
                base = prev.iloc[-1]["close"] if not prev.empty else df_day.iloc[0]["open"]
                base_prices.setdefault(day, {})[sym] = base
        except Exception: pass

    sorted_days = sorted(calendar.keys())
    
    strategy_map = {
        "NEW_ORB": "orb_high", "NEW_PRE": "day_open",
        "ATOM_SUP_EMA5": "ema_5", "ATOM_SUP_EMA20": "ema_20", 
        "ATOM_SUP_EMA50": "ema_50", "ATOM_SUP_EMA200": "ema_200",
        "ATOM_SUP_VWAP": "vwap", "DIP_SNIPER": "bb_lower",
        "MOL_CONFLUENCE": "ema_20", "ROD_A": "sma_50", 
        "ROD_B": "sma_200", "ROD_C": "sma_50"
    }

    try:
        from strategy import GapZoneStrategy
        engine = GapZoneStrategy()
        strategies_config = engine.strategies
    except Exception:
        strategies_config = {}

    active_strategies = [s for s in strategy_map.keys() if strategies_config.get(s, {}).get('enabled', True)]
    leaderboard = []

    for strat_name in active_strategies:
        config = strategies_config.get(strat_name, {})
        sl_pct = abs(config.get('stop_loss', -0.05))
        tp_pct = config.get('take_profit', 0.10)
        trailing_dist = config.get('trailing_stop', 0.0)

        total_pnl = 0.0
        trade_count = 0
        win_count = 0

        print(f"🏃 [{strat_name}] 테스트 중...", end="\r")

        for day in sorted_days:
            day_stocks = calendar.get(day, {})
            if not day_stocks: continue
            
            timeline = sorted(set().union(*(d.index for d in day_stocks.values())))
            watchlist = set()
            position = None
            balance = 10000.0

            for t in timeline:
                # [A] Exit Logic
                if position:
                    sym = position["symbol"]
                    df_sym = day_stocks.get(sym)
                    if df_sym is None or t not in df_sym.index: continue
                    
                    row = df_sym.loc[t]
                    curr_close = float(row["close"])
                    curr_high = float(row["high"])
                    curr_low = float(row["low"])
                    
                    if curr_high > position["max_price"]:
                        position["max_price"] = curr_high

                    exit_reason = None
                    exit_price = curr_close

                    # 미리 가격 계산 (IF문 밖으로 뺌)
                    sl_price = position["entry"] * (1 - sl_pct)
                    tp_price = position["entry"] * (1 + tp_pct)
                    trail_price = position["max_price"] * (1 - trailing_dist)

                    # --- [FIXED LOGIC START] ---
                    if curr_low <= sl_price:
                        exit_reason = "SL"
                        exit_price = sl_price
                    elif curr_high >= tp_price:
                        exit_reason = "TP"
                        exit_price = tp_price
                    elif trailing_dist > 0 and curr_low <= trail_price and trail_price > position["entry"]:
                        exit_reason = "TS"
                        exit_price = trail_price
                    elif t == timeline[-1]:
                        exit_reason = "EOD"
                        exit_price = curr_close
                    # --- [FIXED LOGIC END] ---

                    if exit_reason:
                        pnl = (exit_price - position["entry"]) * position["qty"]
                        total_pnl += pnl
                        trade_count += 1
                        if pnl > 0: win_count += 1
                        position = None
                    continue

                # [B] Scanning Logic
                if t.endswith("0"):
                    for sym, df_sym in day_stocks.items():
                        if sym in watchlist or t not in df_sym.index: continue
                        curr_p = float(df_sym.loc[t]["close"])
                        base_p = base_prices.get(day, {}).get(sym, curr_p)
                        if base_p > 0 and (curr_p - base_p) / base_p >= 0.40:
                            watchlist.add(sym)

                # [C] Entry Logic
                if not position:
                    for sym in sorted(watchlist):
                        df_sym = day_stocks.get(sym)
                        if df_sym is None or t not in df_sym.index: continue
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
                        except: pass

                        if pd.isna(limit_price) or limit_price <= 0: continue

                        if float(row["low"]) <= limit_price * 1.005:
                            entry_exec = min(limit_price, float(row["open"]))
                            qty = int((balance * 0.98) / entry_exec)
                            if qty > 0:
                                position = {"symbol": sym, "entry": entry_exec, "qty": qty, "max_price": entry_exec}
                                break

        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0
        print(f"👉 {strat_name: <15} | PnL: ${total_pnl:,.2f} | Trades: {trade_count} | Win: {win_rate:.1f}%")
        leaderboard.append({"Strategy": strat_name, "PnL": round(total_pnl, 2), "Trades": trade_count, "WinRate": f"{win_rate:.1f}%"})

    if leaderboard:
        df_res = pd.DataFrame(leaderboard).sort_values("PnL", ascending=False)
        print("\n" + "="*60 + "\n🏆 CHAMPIONSHIP LEADERBOARD (Fixed)\n" + df_res.to_string(index=False) + "\n" + "="*60)
        df_res.to_csv(RESULT_FILE, index=False)

if __name__ == "__main__":
    run_championship()