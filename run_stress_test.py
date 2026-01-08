# run_stress_test.py
"""
Tier 3: Stress Test Mode - 최악 시나리오 강제 재현
- 최악 10연속 손실
- 상위 5개 거래 제거 후 수익률
- 저변동성 장 (VIX < 15 대용)
- 하락장 시뮬레이션
- 최악의 체결 조건
"""

import os
import sys
import re
import glob
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 실전 코드 import
from strategy import GapZoneStrategy
from config import Config

# 백테스트 모듈 import
from backtest import StressScenarios, BacktestStatistics

os.makedirs('results', exist_ok=True)

file_handler = logging.FileHandler('results/stress_test.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)

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
# Stress Test 실행
# ==========================================
def run_stress_tests(strategy_name):
    """
    특정 전략에 대한 스트레스 테스트
    
    Args:
        strategy_name: 테스트할 전략
        
    Returns:
        {
            'strategy':  str,
            'tests': {
                'consecutive_losses': {...},
                'top_5_removal': {...},
                'low_volatility': {...},
                'bear_market': {...}
            }
        }
    """
    logger.info(f"🔥 [{strategy_name}] Stress Test 시작...")
    
    # Reality Mode 결과 로드 (사전에 run_reality.py 실행 필요)
    try:
        df_trades = pd.read_csv('results/reality_trades.csv')
        df_trades = df_trades[df_trades['strategy'] == strategy_name]
        
        if df_trades.empty:
            logger.warning(f"⚠️ {strategy_name} 거래 내역 없음")
            return None
        
        trades = df_trades.to_dict('records')
        logger.info(f"📊 {len(trades)}건 거래 로드")
    
    except FileNotFoundError:
        logger.error("❌ results/reality_trades.csv 없음. run_reality.py를 먼저 실행하세요.")
        return None
    
    # ========================================
    # Test 1: 최악 10연속 손실
    # ========================================
    logger.info("🧪 Test 1: 최악 10연속 손실")
    test_consecutive = StressScenarios.force_consecutive_losses(trades, count=10)
    
    logger.info(f"  Total Loss: ${test_consecutive['total_loss']:.2f}")
    logger.info(f"  Survival:  {'✅ YES' if test_consecutive['survival'] else '❌ NO'}")
    
    # ========================================
    # Test 2: 상위 5개 거래 제거
    # ========================================
    logger.info("🧪 Test 2: 상위 5개 거래 제거")
    test_top5 = StressScenarios.remove_top_n_trades(trades, n=5)
    
    logger.info(f"  Original PnL: ${test_top5['original_pnl']:.2f}")
    logger.info(f"  After Removal: ${test_top5['after_removal_pnl']:.2f}")
    logger.info(f"  Dependency: {test_top5['dependency_pct']:.1f}%")
    
    if test_top5['dependency_pct'] > 50:
        logger.warning("  ⚠️ 상위 5개 거래 의존도 50% 초과 (위험)")
    
    # ========================================
    # Test 3: 저변동성 장 (Low Volatility)
    # ========================================
    logger.info("🧪 Test 3: 저변동성 장 시뮬레이션")
    
    try:
        # 일별 변동폭 < 5% 필터링
        df_daily = pd.read_csv('results/reality_daily.csv')
        df_daily = df_daily[df_daily['strategy'] == strategy_name]
        
        # 변동폭 계산 (간이 버전:  거래 없는 날 = 저변동성)
        # 실제로는 데이터에서 high-low 범위 계산 필요
        low_vol_days = df_daily[df_daily['trades'] == 0]
        
        low_vol_pnl = low_vol_days['pnl'].sum()
        low_vol_count = len(low_vol_days)
        
        test_low_vol = {
            'days': low_vol_count,
            'total_pnl': round(low_vol_pnl, 2),
            'avg_pnl': round(low_vol_pnl / low_vol_count, 2) if low_vol_count > 0 else 0
        }
        
        logger.info(f"  Days:  {test_low_vol['days']}")
        logger.info(f"  Total PnL:  ${test_low_vol['total_pnl']:.2f}")
        logger.info(f"  Avg PnL/Day: ${test_low_vol['avg_pnl']:.2f}")
    
    except Exception as e: 
        logger.error(f"  ❌ 저변동성 테스트 실패: {e}")
        test_low_vol = {'error': str(e)}
    
    # ========================================
    # Test 4: 하락장 (Bear Market)
    # ========================================
    logger.info("🧪 Test 4: 하락장 시뮬레이션")
    
    try:
        # 손실��만 추출
        loss_days = df_daily[df_daily['pnl'] < 0]
        
        bear_pnl = loss_days['pnl'].sum()
        bear_count = len(loss_days)
        
        test_bear = {
            'days': bear_count,
            'total_pnl': round(bear_pnl, 2),
            'avg_pnl':  round(bear_pnl / bear_count, 2) if bear_count > 0 else 0
        }
        
        logger.info(f"  Days: {test_bear['days']}")
        logger.info(f"  Total PnL:  ${test_bear['total_pnl']:.2f}")
        logger.info(f"  Avg PnL/Day:  ${test_bear['avg_pnl']:.2f}")
        
        if test_bear['total_pnl'] < -500:
            logger.warning("  ⚠️ 하락장 손실 $500 초과 (취약)")
    
    except Exception as e:
        logger.error(f"  ❌ 하락장 테스트 실패: {e}")
        test_bear = {'error': str(e)}
    
    return {
        'strategy': strategy_name,
        'tests': {
            'consecutive_losses': test_consecutive,
            'top_5_removal': test_top5,
            'low_volatility': test_low_vol,
            'bear_market': test_bear
        }
    }

# ==========================================
# 메인 실행
# ==========================================
def main():
    print("="*70)
    print("🔥 Stress Test Mode - 최악 시나리오 (Tier 3)")
    print("="*70)
    print("🧪 Test 1: 최악 10연속 손실")
    print("🧪 Test 2: 상위 5개 거래 제거")
    print("🧪 Test 3: 저변동성 장")
    print("🧪 Test 4: 하락장")
    print("="*70)
    
    # Reality Mode 결과 확인
    if not os.path.exists('results/reality_trades.csv'):
        logger.error("❌ results/reality_trades.csv 없음")
        logger.error("   run_reality.py를 먼저 실행하세요.")
        return
    
    # 전략 리스트
    engine = GapZoneStrategy()
    active_strategies = [
        name for name, params in engine.strategies.items()
        if params.get('enabled', False)
    ]
    
    logger.info(f"🎯 {len(active_strategies)}개 전략 스트레스 테스트")
    
    # 전략별 실행
    results = []
    
    for strategy_name in active_strategies:
        result = run_stress_tests(strategy_name)
        
        if result:
            results.append(result)
    
    # 결과 저장
    if results:
        # JSON 저장
        with open('results/stress_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("✅ 결과 저장:  results/stress_test_results.json")
        
        # 콘솔 출력
        print("\n" + "="*70)
        print("🔥 Stress Test 결과")
        print("="*70)
        
        for r in results:
            print(f"\n📊 {r['strategy']}")
            print("-"*70)
            
            # Test 1: 연속 손실
            t1 = r['tests']['consecutive_losses']
            print(f"\n🧪 최악 10연속 손실")
            print(f"  Total Loss: ${t1['total_loss']:.2f}")
            print(f"  Survival:  {'✅ YES' if t1['survival'] else '❌ NO'}")
            
            # Test 2: Top 5 제거
            t2 = r['tests']['top_5_removal']
            print(f"\n🧪 상위 5개 거래 제거")
            print(f"  Original:  ${t2['original_pnl']:.2f}")
            print(f"  After Removal: ${t2['after_removal_pnl']:.2f}")
            print(f"  Dependency: {t2['dependency_pct']:.1f}%")
            
            if t2['dependency_pct'] > 50:
                print(f"  ⚠️ WARNING: 상위 거래 의존도 높음")
            
            # Test 3: 저변동성
            t3 = r['tests']['low_volatility']
            if 'error' not in t3:
                print(f"\n🧪 저변동성 장")
                print(f"  Days: {t3['days']}")
                print(f"  Total PnL: ${t3['total_pnl']:.2f}")
                print(f"  Avg/Day: ${t3['avg_pnl']:.2f}")
            
            # Test 4: 하락장
            t4 = r['tests']['bear_market']
            if 'error' not in t4:
                print(f"\n🧪 하락장")
                print(f"  Days: {t4['days']}")
                print(f"  Total PnL: ${t4['total_pnl']:.2f}")
                print(f"  Avg/Day: ${t4['avg_pnl']:.2f}")
                
                if t4['total_pnl'] < -500:
                    print(f"  ⚠️ WARNING:  하락장 취약")
        
        print("\n" + "="*70)
        print("🎯 최종 판정")
        print("="*70)
        
        for r in results:
            t1 = r['tests']['consecutive_losses']
            t2 = r['tests']['top_5_removal']
            
            survival = t1['survival']
            dependency_ok = t2['dependency_pct'] < 50
            
            if survival and dependency_ok:
                print(f"✅ {r['strategy']}:  PASS (실전 투입 가능)")
            elif survival:
                print(f"⚠️ {r['strategy']}: CONDITIONAL (상위 거래 의존도 주의)")
            else:
                print(f"❌ {r['strategy']}:  FAIL (연속 손실 견딜 수 없음)")
        
        print("="*70)

if __name__ == "__main__":
    main()