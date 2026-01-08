# bulk_alpaca_loader.py
import os
import time
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime, timedelta
import pytz

# --- [설정] ---
# (기존에 사용하시던 API 키를 그대로 유지하세요)
API_KEY = "PKEVP9MF37N172VZ66P0" 
SECRET_KEY = "iAoJINgI9ic0KqLVILinxD3lNpeIRfkxBI0nWz5Q"
BASE_URL = "https://paper-api.alpaca.markets" # v2 제거 (SDK가 알아서 처리함)

TARGET_FILE = "targets.txt"
OUTPUT_DIR = "data"

def setup_api():
    return REST(API_KEY, SECRET_KEY, BASE_URL)

def parse_target_line(line):
    """ 
    '20251218_TMDE' 또는 'TMDE_20251218' 형식을 유연하게 파싱 
    """
    line = line.strip()
    if not line or "_" not in line: return None, None
    
    parts = line.split("_")
    
    # [수정] 어느 쪽이 날짜(숫자)인지 자동 판별
    part1 = parts[0].strip()
    part2 = parts[1].strip()
    
    symbol = ""
    date_str = ""
    
    # part1이 숫자(날짜)인 경우 (예: 20251218_TMDE)
    if part1.isdigit() and len(part1) == 8:
        date_str = part1
        symbol = part2.upper()
    # part2가 숫자(날짜)인 경우 (예: TMDE_20251218)
    elif part2.isdigit() and len(part2) == 8:
        symbol = part1.upper()
        date_str = part2
    else:
        # 알 수 없는 형식이면 기본적으로 앞을 심볼로 가정
        symbol = part1.upper()
        date_str = part2

    return symbol, date_str

def download_data(api, symbol, target_date_raw):
    # 날짜 변환 (YYYYMMDD -> YYYY-MM-DD)
    try:
        target_dt = datetime.strptime(target_date_raw, "%Y%m%d")
    except ValueError:
        print(f"⚠️ 날짜 형식 오류 (YYYYMMDD 필요): {target_date_raw} (Symbol: {symbol})")
        return False

    # 데이터 수집 기간 설정 (타겟 날짜 하루 전 ~ 하루 후, 넉넉하게)
    start_dt = target_dt - timedelta(days=5)
    end_dt = target_dt + timedelta(days=0)
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    print(f"📥 [{symbol}] Downloading... ({start_str} ~ {end_str})")

    try:
        # Alpaca API로 데이터 요청 (1Min Bar)
        bars = api.get_bars(
            symbol, 
            TimeFrame.Minute, 
            start=start_str, 
            end=end_str, 
            adjustment='raw',
            feed='iex'
        ).df

        if bars.empty:
            print(f"⚠️ 데이터 없음: {symbol}")
            return False

        # Timezone 처리 (UTC -> New York)
        ny_tz = pytz.timezone('America/New_York')
        if bars.index.tzinfo is None:
            bars.index = bars.index.tz_localize('UTC').tz_convert(ny_tz)
        else:
            bars.index = bars.index.tz_convert(ny_tz)

        # 포맷 정리
        bars = bars.reset_index()
        bars.columns = [c.lower() for c in bars.columns] # 컬럼 소문자화
        
        # timestamp 컬럼을 date와 time으로 분리
        bars['date'] = bars['timestamp'].dt.strftime('%Y-%m-%d')
        bars['time'] = bars['timestamp'].dt.strftime('%H%M').astype(int)
        
        # 필요한 컬럼만 선택
        final_df = bars[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]
        
        # [수정] 파일명 형식 변경: 날짜_티커.csv
        filename = f"{target_date_raw}_{symbol}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        final_df.to_csv(filepath, index=False)
        
        print(f"✅ 성공: {filename}")
        return True

    except Exception as e:
        print(f"❌ 실패 ({symbol}): {e}")
        return False

def main():
    print("🚀 [Data Miner] targets.txt 기반 데이터 수집 시작")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(TARGET_FILE):
        print(f"❌ '{TARGET_FILE}' 파일이 없습니다.")
        return

    with open(TARGET_FILE, "r") as f:
        lines = f.readlines()

    print(f"📋 총 {len(lines)}개의 타겟을 읽었습니다.\n")

    api = setup_api()
    
    success_count = 0
    for line in lines:
        symbol, date_str = parse_target_line(line)
        if not symbol or not date_str:
            continue
            
        if download_data(api, symbol, date_str):
            success_count += 1
            # API 제한 고려 (짧은 대기)
            time.sleep(0.5)

    print(f"\n🎉 모든 작업 완료. (성공: {success_count} / 전체: {len(lines)})")

if __name__ == "__main__":
    main()