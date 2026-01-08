# Dry-Run Backtesting System v2.0

실전 매매 환경을 최대한 재현하는 3-Tier 백테스팅 시스템

## 🎯 설계 철학

> "이 백테스터는 수익을 증명하는 도구가 아니라,  
> 실전에서 망할 가능성을 최대한 앞당겨 보여주는 도구다."

## 📦 시스템 구조

```
Tier 1: Championship Mode (전략 선택)
   ↓
Tier 2: Reality Mode (실전 시뮬레이션)
   ↓
Tier 3: Stress Test (최악 시나리오)
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install pandas numpy python-dotenv
```

### 2. 데이터 준비

`data/` 폴더에 CSV 파일 배치:
- 파일명 형식: `YYYYMMDD_SYMBOL.csv`
- 예: `20251230_AEHL.csv`

### 3. 실행

```bash
# Tier 1: 전략 간 비교 (빠름, 낙관적)
python run_championship.py

# Tier 2: 실전 시뮬레이션 (비용·마찰 포함)
python run_reality.py

# Tier 3: 스트레스 테스트 (최악 시나리오)
python run_stress_test.py
```

### 4. 유닛 테스트

```bash
python tests/test_core.py
```

## 📊 결과 파일

- `results/championship_trades.csv` - Tier 1 거래 내역
- `results/reality_trades.csv` - Tier 2 거래 내역
- `results/reality_summary.json` - 통계 요약
- `results/stress_test_results.json` - 스트레스 테스트 결과

## 🔧 설정

`config.py`에서 다음 값 조정: 

```python
ALL_IN_RATIO = 0.98          # 예수금 사용 비율
MAX_DAILY_LOSS_PCT = 6.0     # 일일 손실 한도
MIN_CHANGE_PCT = 40.0        # 급등 필터 (40%)
```

## 📈 통계 지표

- **기본**:  Total PnL, Win Rate, Profit Factor
- **리스크**: MDD, Sharpe Ratio, VaR (5%)
- **경고**: Top 5 의존도, 연속 손실, 복구 기간

## ⚠️ 주의사항

1. **Championship vs Reality 차이**
   - Championship: 거래 비용 없음, 즉시 체결
   - Reality: 비용·슬리피지·부분체결 포함
   - Reality가 20~30% 낮게 나오는 것이 정상

2. **Stress Test 필수**
   - 상위 5개 거래 의존도 > 50% → 위험
   - 연속 10회 손실 견딜 수 없음 → 실전 투입 불가

3. **실전 코드 동기화**
   - `strategy.py`, `config.py`는 auto-sell-system과 동일해야 함
   - 주기적으로 복사 또는 symlink 사용

## 🧪 검증 체크리스트

- [ ] Championship 우승 전략 선택
- [ ] Reality Mode에서도 수익 유지
- [ ] Stress Test 통과 (연속 손실 생존)
- [ ] Top 5 의존도 < 50%
- [ ] 하락장·저변동성 장 대응 확인

## 📚 추가 문서

- `GapZone Dry-Run 시스템 백서.md` - 설계 철학
- `results/reality_mode.log` - 실행 로그

## 🤝 기여

이슈 및 PR 환영합니다.

## 📄 라이선스

MIT License