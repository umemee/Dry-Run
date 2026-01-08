"""
🔍 GATE 1: DEBUG MODE (디버깅 전용)
목적: 전략 실행이 아니라, "데이터가 왜 무시되는지" 추적
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def run_debug_mode():
    print("="*70)
    print("🐞 GATE 1: 초정밀 디버깅 모드 시작")
    print("   모든 필터를 끄고 날 것의 데이터를 확인합니다.")
    print("="*70)

    # 1. 파일 로드
    files = glob.glob("data/*.csv")
    if not files:
        print("❌ 파일이 없습니다. 경로를 확인하세요.")
        return

    print(f"📂 발견된 파일: {len(files)}개")
    
    # 2. 첫 3개 파일만 집중 분석 (전체를 다 돌면 로그가 너무 많음)
    target_files = files[:3] 
    print(f"🔬 테스트 대상 파일: {[os.path.basename(f) for f in target_files]}")
    
    for f in target_files:
        filename = os.path.basename(f)
        print(f"\n" + "-"*50)
        print(f"📄 분석 파일: {filename}")
        print("-"*50)

        # 종목명 파싱
        try:
            sym = filename.split('_')[1].replace(".csv", "") if "_" in filename else filename.replace(".csv", "")
        except:
            sym = "UNKNOWN"
            
        df = pd.read_csv(f)
        
        # (1) 날짜/시간 컬럼 확인
        print(f"   👉 컬럼 목록: {list(df.columns)}")
        
        # 날짜 변환 시도
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            unique_dates = df['date'].dt.strftime('%Y-%m-%d').unique()
            print(f"   👉 포함된 날짜({len(unique_dates)}일): {unique_dates[:3]} ...")
        else:
            print("   ❌ 'date' 컬럼이 없습니다! 분석 불가.")
            continue

        # 시간 변환 시도
        if 'time' in df.columns:
            # 시간 형식 샘플 출력
            sample_time = df['time'].iloc[0]
            print(f"   👉 시간 포맷 샘플: {sample_time} (Type: {type(sample_time)})")
        else:
            print("   ❌ 'time' 컬럼이 없습니다! 분석 불가.")
            continue

        # (2) 갭(Gap) 계산 시뮬레이션
        # 날짜별로 루프를 돌면서 '전일 종가' vs '당일 시가' 비교
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        grouped = df.groupby('date_str')
        
        sorted_dates = sorted(list(grouped.groups.keys()))
        last_close = None # 전일 종가

        print("\n   [📉 갭(Gap) 계산 추적]")
        
        gap_found_count = 0
        
        for date in sorted_dates:
            day_data = grouped.get_group(date).sort_values('time')
            
            # 당일 시가 / 종가
            day_open = day_data.iloc[0]['open']
            day_close = day_data.iloc[-1]['close']
            
            # 기준가 설정 (전일 종가가 없으면 당일 시가 사용 -> 갭 0%)
            base_price = last_close if last_close is not None else day_open
            
            # 갭 계산
            gap_rate = (day_open - base_price) / base_price * 100
            
            # 디버깅 출력 (갭이 10% 이상이거나, 첫 3일간은 무조건 출력)
            if gap_rate > 10 or sorted_dates.index(date) < 3:
                status = "🔥급등발견" if gap_rate > 10 else "일반"
                print(f"     📅 {date}: 전일종가 {base_price:.2f} -> 시가 {day_open:.2f} | 갭: {gap_rate:.2f}% [{status}]")
                
                if gap_rate > 10:
                    gap_found_count += 1
            
            # 다음 날을 위해 종가 저장
            last_close = day_close
            
        print(f"\n   ✅ {filename} 분석 완료: 10% 이상 급등 {gap_found_count}회 발견")

    print("\n" + "="*70)
    print("🏁 디버깅 완료.")
    print("만약 위 로그에서 '갭: 0.00%'만 계속 나온다면 -> 날짜 정렬이나 전일 종가 연동 문제")
    print("만약 위 로그에서 '급등발견'이 뜬다면 -> 기존 코드의 '스캔' 로직 문제")
    print("="*70)

if __name__ == "__main__":
    run_debug_mode()