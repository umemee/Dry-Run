# 📜 The GapZone Backtesting Constitution

**Version: 1.4.0 (Field-Tested Amendment - Performance Metrics & Parameter Optimization)**

**Effective Date: 2026-01-10**

**Amendment ID: GZ-CONST-AMD-005 (Essential Fix for Production Reliability)**

**Authority: Supreme Quant Architecture Board (Approved by System Commander)**

---

## 🏛️ Preamble (전문)

GapZone 시스템의 백테스팅은 수익률을 자랑하기 위한 '마케팅 도구'가 아니라, **'실전에서의 생존 가능성을 검증하는 극한의 스트레스 테스트'**이다.

우리는 과거의 데이터(Data)와 현재의 로직(Logic)을 결합하여 미래의 불확실성을 통제한다. 이에 우리는 시간의 불가역성(Irreversibility of Time), 자원의 유한성(Finiteness of Resources), 실행의 마찰력(Friction of Execution)을 시스템의 3대 핵심 가치로 규정하고 본 헌법을 제정한다.

**v1.4.0 개정 배경:** 그간의 실무 디버깅과 최적화 과정에서 발견된 **"입법적 허점"**을 반영한다:
- **판타지 수익률 버그 (100% 승률):** 부호 오류로 인한 손절/익절 역전
- **누적 손익 증발:** 일일 리셋 후 누적 수익 미기록
- **데이터 누락:** 필수 필드 부재로 시각화 실패
- **폭주 거래:** 종목별 거래 제한 이중 정의 필요

이제부터는 어떤 개발자가 구현하더라도 **동일한 결과 재현**이 가능해야 한다.

이 시각 이후, 본 헌법을 위반하여 생성된 모든 시뮬레이션 결과는 **'무효(Null and Void)'**이며, 어떠한 전략적 의사결정의 근거로도 사용할 수 없다.

---

## Article 1. System Definition (시스템의 물리적 정의)

### 1.1 Spacetime Coordinates (시공간 좌표)

**정의:** 시뮬레이션의 기준 시간은 **KST(한국 표준시)**를 따르며, 거래 대상은 **미국 주식 시장(NASDAQ, NYSE, AMEX)**으로 한정한다.

**이유:** 실전 코드(config.py, kis_api.py)가 KST 기준으로 장 운영 시간(18:00~06:00)을 통제하므로, 시차 변환으로 인한 연산 오류를 원천 차단하기 위함이다.

**구현 규칙:**

- 모든 Input Data(CSV)의 타임스탬프는 ISO8601 포맷의 KST로 파싱한다.
- **Pre-Market:** 18:00 ~ 23:30 (제한적 진입/감시)
- **Main-Market:** 23:30 ~ 04:00 (메인 트레이딩)
- **Cooldown:** 04:00 ~ 06:00 (신규 진입 금지, 청산 집중)

---

### 1.2 Atomic Data Unit (데이터의 최소 단위)

**정의:** 의사결정의 최소 단위는 **1분 봉(1-Minute Candle)**이다.

**이유:** 1분 미만의 틱 데이터 처리는 과도한 연산 부하와 노이즈를 유발하며, 실전 API의 호출 제한(Rate Limit)을 고려할 때 현실적이지 않다.

---

### 1.3 Daily Trading Boundary (일일 매매 경계)

**정의:** 주식 데이터 파일은 파일명의 날짜를 기준으로 하며, 실제 매매는 해당 날짜에만 수행된다.

**이유:** 미국 주식 시장의 일일 정산(Daily Settlement)과 급등주 매매의 특성상, 파일이 제공하는 과거 데이터(히스토리)는 이동평균선 등 지표 계산용이며, 실제 거래 신호는 명시된 거래 날짜에만 유효하다.

**구현 규칙:**

- 파일명 형식: `YYYYMMDD_SYMBOL` (예: `20251230_FLYE`)
- 실제 거래 가능 날짜: 파일명의 YYYYMMDD (예: 20251230)
- 파일 내 포함된 과거 데이터: **이동평균선, RSI 등 기술적 지표 계산 용도**
- 파일 내 미래 데이터: 사용 금지 (Future Leakage 방지)

---

### 1.4 Moving Average Timeframe Specification (이동평균선 기준 명시)

**정의:** 모든 기술적 지표(Moving Average, RSI, MACD 등)의 계산 기준 시간프레임은 **1분 봉(1-Minute Candle)**이다.

**이유:** Article 1.2에서 의사결정의 최소 단위가 1분 봉으로 정의되었으므로, 지표도 동일 시간프레임에서 계산되어야 시뮬레이션과 실전의 신호 일관성이 보장된다.

**구현 규칙:**

- 예시: 200EMA = 최근 200개의 1분 봉 Close 값을 기반으로 계산
- 10분 스캔 시점에서도 **1분 봉 기준**의 200EMA를 사용
- 다중 시간프레임 지표 금지 (1분, 5분, 15분 혼재 불가)

---

## Article 2. Chronological Event Processing (시간 흐름 및 이벤트 처리)

### 2.1 The Law of Chronological Sovereignty (시간 주권의 법칙)

**정의:** 시뮬레이션 루프의 주체는 '종목(Stock)'이 아니라 **'시간(Time)'**이다. `for stock in stocks` 패턴은 영구히 폐기한다.

**이유:** 종목별 루프는 타 종목의 기회비용을 무시하는 '신(God)의 시점' 오류(Look-ahead Bias)를 범한다. 오직 시간 순서대로 처리해야만 **\"자금이 없어 매수하지 못하는 상황\"**을 재현할 수 있다.

**구현 규칙:**

```python
# Mandatory Loop Structure
sorted_timestamps = get_all_unique_timestamps()
for timestamp in sorted_timestamps:
    update_market_data(timestamp)         # 현재 시간의 가격 정보 업데이트
    
    if portfolio.is_empty():
        scan_and_signal(timestamp)        # 빈손일 때만 탐색
    else:
        manage_position(timestamp)        # 보유 중이면 관리만 수행
```

---

### 2.2 Dynamic Rolling Watch (동적 순환 감시)

**정의:** 감시(Scanning)는 장 시작 시 1회성으로 끝나는 것이 아니라, 매 10분(10-minute interval)마다 반복되는 **'상태 확인(State Check)'** 프로세스이다.

**구현 규칙:**

- 시스템은 매 10분 정각(예: 09:30, 09:40, 09:50...)마다 전체 종목을 스캔한다.
- 이때 확인하는 기준 가격은 시가(Open)가 아니라, **해당 시점의 현재가(Current Close)**이다.
- **\"늦은 출발(Late Bloomer)\" 인정:** 장 시작 시점에는 조건 미달이었더라도, 장중 상승하여 현재 스캔 시점에 기준을 충족하면 즉시 **감시 대상(Watchlist)**에 편입한다.

**데이터 결측 시 폴백 로직:**

- **상황:** 정각(예: 09:40:00)에 데이터가 없고, 이후 09:41:30에 데이터가 도래함
- **처리:** 해당 시점 이후 가장 먼저 도래하는 틱(09:41:30)을 스캔 시점으로 간주
- **중요:** 지표 계산은 여전히 직전 확정 봉(t-1) 데이터 기준으로 수행
- **결과:** 스캔 시간 연기 (09:40 → 09:41:30)는 발생하지만, 미래 데이터 참조는 없음

---

### 2.3 The Pessimistic Sequence Rule (비관적 순서 법칙)

**정의:** 단일 1분 봉 내에서 익절가(Target)와 손절가(Stop-loss)가 동시에 도달한 경우(`Low <= Stop <= High <= Target`), 무조건 **'손절(Stop-loss)'이 먼저 발생한 것**으로 간주하여 청산한다.

**이유:** 백테스팅은 항상 최악의 시나리오를 가정해야 하므로, 양쪽 조건을 동시에 만족할 경우 손실 방향을 우선한다.

---

## Article 3. Resource Constraints (자원 제약 및 슬롯)

### 3.1 Single Slot Mandate (단일 슬롯 강령)

**정의:** 시스템은 동시에 단 하나의 포지션만 보유할 수 있다. (Status: `EMPTY` or `HOLDING`)

**이유:** 자금의 유한성과 리스크 집중을 방지하고, **기회비용(Opportunity Cost)**을 백테스팅 결과에 강제로 반영하기 위함이다.

**테스트 기준:** 포지션 보유 중(`HOLDING`) 발생한 모든 타 종목의 진입 신호는 로그에 `REJECT:SLOT_FULL`로 기록하고 무조건 무시한다.

---

### 3.2 Per-Stock Trading Limit (종목별 거래 제한) - 이중 정의 [Amended v1.4]

**배경:** v1.3.0의 "당일 동일 종목 1회 거래" 규칙은 GapZone/ORB 전략에 적합하지만, EMA 같은 추세 전략에서는 반복 거래로 인한 폭주(900회 거래) 문제 발생.

**정의:** 거래 제한 방식은 전략 특성에 따라 이중으로 정의된다.

#### 3.2.1 Daily One-Shot Rule (일일 1회 거래, GapZone/ORB용)
- **정의:** 동일 종목, 동일 거래 날짜 내에 1회만 진입 가능
- **재진입:** 다음 날짜에는 동일 종목 재진입 허용
- **적용 전략:** GapZone, ORB
- **로그:** `REJECT:ALREADY_TRADED_TODAY`

#### 3.2.2 Global One-Shot Rule (전역 1회 거래, EMA 등 추세 전략용) [선택적 적용]
- **정의:** 시뮬레이션 전체 기간 동안 동일 종목 재진입 금지
- **이유:** 추세 전략은 당일 리셋과 무관하게 같은 종목을 반복 매수할 경우 과도한 거래량 발생
- **적용 조건:** 전략의 논리가 "추세 확인" 기반일 경우 명시적으로 활성화
- **로그:** `REJECT:GLOBAL_DUPLICATE`

---

### 3.3 Daily Reset & Cumulative Performance [Amended v1.4] - CRITICAL FIX

**핵심 개정:** "일일 리셋"과 "누적 수익" 혼동 문제 해결

**정의:**
- **운용 잔고(Operational Balance):** 매일 장 시작 전(09:00 KST) **$10,000으로 초기화**
- **성과 기록(Performance Record):** **별도로 영구 보존**, 모든 거래 이력은 Trade Ledger에 기록
- **최종 PnL 계산:** 마지막 날의 잔고가 아니라, **모든 거래 기록의 PnL 총합**으로 계산

**문제 사례 (v1.3.0 이전):**
```
Day 1: 거래 2회, 잔고 $10,100 (PnL +$100)
       → Daily Reset: 잔고를 $10,000으로 초기화

Day 2: 거래 3회, 잔고 $9,950 (PnL -$50)
       → 최종 출력: 잔고 $9,950 또는 Day2 PnL -$50만 출력
       → ❌ Day 1의 +$100 누락됨!
```

**해결 방안 (v1.4.0):**
```python
class Portfolio:
    def __init__(self, seed=10000):
        self.operational_balance = seed      # 매일 리셋됨
        self.trade_ledger = []              # 모든 거래 기록 (영구 보존)
    
    def daily_reset(self):
        """매일 09:00에 호출"""
        daily_pnl = self.operational_balance - self.INITIAL_SEED
        self.trade_ledger.append({'date': today, 'pnl': daily_pnl})
        self.operational_balance = self.INITIAL_SEED  # 리셋
    
    def get_cumulative_pnl(self):
        """최종 PnL = Trade Ledger 합계 (정답!)"""
        return sum(record['pnl'] for record in self.trade_ledger)

# 결과
# Day 1 PnL: +$100
# Day 2 PnL: -$50
# → 최종 누적 PnL: +$50 ✓
```

---

### 3.4 Zone-Based Resource Allocation [New v1.4]

**정의:** 시드 금액에 따라 동시 운영 가능한 전략 개수를 제한한다.

| Zone | 시드 범위 | 활성 전략 수 | 최대 포지션 | 목적 |
|------|---------|-----------|-----------|------|
| **ZONE_1** | $100 ~ $1,000 | 1개 | 1 | 최고 전략 검증 |
| **ZONE_2** | $1,000 ~ $5,000 | 2개 | 2 | 위험 분산 |
| **ZONE_3** | $5,000+ | 3개 | 3 | 완전 자동화 |

---

## Article 4. Execution & Friction Model (실행 및 마찰 모델)

### 4.0 Stop Loss & Take Profit Calculation (손절/익절 수식 명시) [New v1.4] - CRITICAL

**배경:** v1.3.0 이전 구현에서 **부호 오류**로 인한 "판타지 체결"(익절 구간에서 손절이 발생) 문제가 있었음. 이를 원천 차단하기 위해 수식을 명시화한다.

**Long Position에서의 수식:**

```
Entry_Price = 100.0

Stop Loss (SL) 계산:
  SL = Entry_Price * (1 - abs(SL_Ratio))
  예: SL_Ratio = 0.05 (5% 손절)
  → SL = 100 * (1 - 0.05) = 95.0 ✓

Take Profit (TP) 계산:
  TP = Entry_Price * (1 + abs(TP_Ratio))
  예: TP_Ratio = 0.10 (10% 익절)
  → TP = 100 * (1 + 0.10) = 110.0 ✓

Trailing Stop (TS) 계산:
  TS = Highest_Price_Since_Entry * (1 - abs(TS_Callback))
  예: Highest = 105, TS_Callback = 0.02 (2% 콜백)
  → TS = 105 * (1 - 0.02) = 102.9 ✓
```

**코드 구현 규칙 (abs() 필수):**
```python
# ✅ CORRECT (abs() 필수)
sl_price = entry_price * (1 - abs(sl_ratio))
tp_price = entry_price * (1 + abs(tp_ratio))
ts_price = highest_price * (1 - abs(ts_callback))

# ❌ WRONG (부호 미처리)
sl_price = entry_price * (1 + sl_ratio)      # -0.05를 +0.05로 계산함 (역전!)
tp_price = entry_price * (1 - tp_ratio)      # 반대 방향
```

**검증:** 모든 SL/TP 계산 코드는 반드시 `abs()` 함수를 사용해야 함.

---

### 4.1 Entry: The Aggressive Scoop (공격적 뜰채 진입)

**Order_Price 계산:**
```
Order_Price = Current_Price * 1.005 (0.5% 상향)
```

**체결 조건:**
```
Low(t) <= Order_Price → 체결
```

---

### 4.2 Exit: Market Emulation (시장가 에뮬레이션)

**구현 규칙:**
```
Sell_Price = Current_Close * 0.99 (1% 슬리피지 강제 적용)
```

---

### 4.3 Balance Protection (잔고 보호)

**Order Amount 계산:**
```
Order_Amount = Cash * 0.98 (2% 마진)
Position_Size = int(Order_Amount / Order_Price)
```

---

### 4.4 Exit: Dynamic Trailing Stop (동적 트레일링 스탑)

**구현 규칙:**
- **Activation:** 진입 후 +5% 수익 도달 시
- **Callback:** 기록된 최고가 대비 -2% 하락 시 시장가 매도

---

### 4.5 End-of-Day Protocol (장 종료 강제 청산)

**구현 규칙:**
```
if timestamp.time() >= datetime.time(5, 55, 0):  # KST 05:55
    force_liquidate_all_positions()
```

---

## Article 5. Data Integrity & Performance Metrics (데이터 기록 및 성과 측정) [Restructured v1.4]

### 5.0 Fallback Protocol (404 예외 처리)

**구현 규칙:** 등락률 순위 데이터 부재 시 → **거래량 순위 데이터**로 대체 스캔.

---

### 5.1 Trade Logs (거래 기록 명시) [New v1.4] - CRITICAL

**배경:** 시각화(visualization.py) 실패와 사후 분석 불가 문제를 해결하기 위해, 필수 필드를 헌법 레벨에서 명시화한다.

**필수 출력 파일:** `final_trades.csv`

**필수 필드 (11개):**

| 필드명 | 타입 | 예시 | 설명 |
|--------|------|------|------|
| `date` | DATE | 2025-12-30 | 거래 날짜 |
| `strategy` | STR | GapZone | 전략명 |
| `ticker` | STR | TSLA | 종목 코드 |
| `entry_time` | TIME | 23:45:00 | 진입 시간 |
| `entry_price` | FLOAT | 250.50 | 진입 단가 **(필수)** |
| `exit_time` | TIME | 00:15:00 | 청산 시간 |
| `exit_price` | FLOAT | 252.00 | 청산 단가 **(필수)** |
| `qty` | INT | 10 | 거래량 |
| `pnl` | FLOAT | 15.00 | 실현 손익 = (exit_price - entry_price) * qty - fee |
| `return_pct` | FLOAT | 0.60% | 수익률 = (exit_price - entry_price) / entry_price |
| `reason` | STR | TRAILING_STOP | 청산 사유 |

**CSV 예시:**
```csv
date,strategy,ticker,entry_time,entry_price,exit_time,exit_price,qty,pnl,return_pct,reason
2025-12-30,GapZone,TSLA,23:45:00,250.50,00:15:00,252.00,10,15.00,0.60%,TRAILING_STOP
2025-12-30,ORB,NVDA,23:50:00,180.25,00:30:00,179.00,5,-6.25,-0.69%,STOP_LOSS
```

---

### 5.2 Performance Metrics (성과 지표 정의) [New v1.4]

**정의:** 승률, 손익비, 최대 낙폭 계산의 모호성을 제거하기 위해 명확한 수식을 정의한다.

#### 5.2.1 Win Rate (승률)

```
승률 = (PnL > 0인 거래 수) / (Total Trades) * 100%

예: 10회 거래 중 6회 수익 → 승률 = 60%
```

#### 5.2.2 Profit Factor (손익비)

```
손익비 = (총 수익의 합) / (총 손실의 합)

예: 
총 수익 = $500
총 손실 = $200
손익비 = 2.5
```

#### 5.2.3 Maximum Drawdown (최대 낙폭)

```
MDD = (누적 자산의 최고점 - 그 이후 최저점) / 누적 자산 최고점

예:
Day 1: 누적 자산 $10,100 (최고점)
Day 2: 누적 자산 $9,950 (최저점)
MDD = ($10,100 - $9,950) / $10,100 = 1.49%
```

#### 5.2.4 Sharpe Ratio (샤프 비율)

```
Sharpe = (평균 일일 수익률 - 무위험 수익률) / 일일 수익률 표준편차

기준:
- Sharpe ≥ 1.0: 우수한 전략
- Sharpe ≥ 2.0: 매우 우수한 전략
```

---

### 5.3 Missing Data Handling (결측치 처리)

**정의:** 타임스탬프가 비어있는 구간(Gap)에 대해 임의의 데이터 생성(Interpolation)을 금지한다.

---

### 5.4 Simultaneous Execution Priority (동시 체결 우선순위)

**정의:** 단일 스캔 시점에서 3개 이상의 종목이 진입 신호를 동시에 발생할 경우, 슬롯 제약(Single Slot)으로 인해 1개만 체결된다.

**우선순위 로직:** **'스캔 순서 우선(First-Scanned, First-Served)'** 원칙을 따른다.

---

## Article 2 (Amended). Strategy Definitions (전략 파라미터 최적화) [New v1.4]

**배경:** optimizer.py를 통해 발견된 "최적 파라미터"를 헌법에 법제화한다. 이전 버전의 추정값은 폐기.

### 2.1 GapZone Strategy (급등갭 추격 전략)

**Gap Threshold:** 15% 이상 갭상승 **(기존 5% → 상향)**

**손절가 (Stop Loss):**
- SL = Entry_Price * (1 - 0.05) = Entry_Price * 0.95 **(기존 -3% → -5% 확장)**

**익절가 (Take Profit):**
- TP = Entry_Price * (1 + 0.10) (정적 익절, 10%)

**트레일링 스탑 (Trailing Stop):**
- **활성화:** +5% 수익 도달 시 **(기존 3% → 5% 상향)**
- **콜백:** 고점 대비 -2% 하락 시 청산 **(기존 1% → 2%)**

---

### 2.2 EMA200 Strategy (200일선 지지 전략) [Amended v1.4]

**논리 개정 (Critical Fix):**

기존: 단순히 현재가가 200EMA를 터치하면 매수
개정: **Safe Pullback** 패턴 확인 후 매수
  1. 가격이 200EMA 위에서 움직임
  2. 200EMA를 터치 또는 약간 하회 (Pullback)
  3. 그 이후 양봉이 나타남 (반등 확인)
  → 이 3가지 조건을 모두 충족할 때만 매수

**Tolerance (감지 범위):** 1% → 3% **(기존보다 유연하게 확대)**

**손절가 (Stop Loss):**
- SL = Entry_Price * (1 - 0.07) = Entry_Price * 0.93 **(기존 -3% → -7%, 휩소 방어)**

**익절가 (Take Profit):**
- TP = Entry_Price * (1 + 0.12) (정적 익절, 12%)

---

### 2.3 ORB Strategy (오픈 레인지 브레이크아웃) [Amended v1.4]

**Timeframe 개정:**
- 기존: 30분 레인지 → **신규: 15분 레인지** (빠른 진입)

**진입 기준:**
- 장초반 15분 레인지의 고점 돌파 시 매수

**익절가 (Take Profit):**
- 정적 익절: TP1 = Entry_Price * (1 + 0.15) (15% 목표)

**트레일링 스탑 (추세 추종형):**
- **활성화:** +5% 수익 도달 시
- **콜백:** 고점 대비 -5% 하락 시 청산 **(기존 -2% → -5%, 추세 추종용으로 더 관대함)**

**손절가 (Stop Loss):**
- SL = Entry_Price * (1 - 0.06) = Entry_Price * 0.94 (6% 손절)

---

## Article 6. Meta-Rules & Amendments (메타 규칙 및 개정)

### 6.1 Prohibition of Arbitrary Amendment (자의적 개정 금지)

**정의:** 본 헌법은 백테스팅 성과(수익률, 승률)의 개선이나 전략의 적합성을 이유로 개정될 수 없다.

**허용되는 개정 조건:**
1. 입증된 구조적 불일치 (Live Log vs 시뮬레이션)
2. 외생적 요인 (API 변경, 거래소 규정 변경)

---

### 6.2 Non-Retroactivity (소급 적용 금지)

**구현 규칙:** 헌법이 개정될 경우 반드시 새로운 **Version (e.g., 1.4.0)**을 부여한다.

---

## Article 7. Judgment of Validity (유효성 판정 기준)

### 7.1 The \"Five Deadly Sins\" (5대 무효 사유) [Amended v1.4]

**INVALID RUN을 판정하는 절대적 조건:**

| 번호 | 사유 | 판정 기준 |
|------|------|---------|
| **1** | **Non-Reproducibility** | 동일 데이터/로직으로 재실행 시 결과 불일치 |
| **2** | **Future Leakage** | 특정 시점에서 미래 데이터 참조 |
| **3** | **Cost Omission** | 수수료, 슬리피지 미반영 |
| **4** | **Slot Violation** | 동시 포지션 1개 규칙 위반 |
| **5** | **Sign Error in SL/TP** [New v1.4] | Stop Loss/Take Profit 수식 부호 오류 |

---

### 7.2 Automatic Discard Protocol (자동 폐기 프로토콜)

**판정:** 위 사유 발생 시 시스템은 결과 리포트 상단에 **[INVALID RUN]** 마크를 찍고, 해당 결과를 랭킹에서 제외한다.

---

## 🛡️ Validation Suite (검증 스위트) [Enhanced v1.4]

### Case 1: The \"Twin Signal\" Paradox

**상황:** 09:50:00에 종목 A와 B가 동시에 진입 신호 발생

**기대 결과:** A만 매수, B는 `REJECT:SLOT_FULL`

---

### Case 2: The \"Ghost\" Fill

**상황:** 주문 $100.00, Low $100.01

**기대 결과:** 미체결 (조건: Low ≤ Order_Price)

---

### Case 3: The \"Frozen Market\"

**상황:** 09:31~09:39 데이터 누락, 09:40 정각 데이터 있음

**기대 결과:** 09:40에서 정상 스캔 (또는 09:41:30 Fallback)

---

### Case 4: The \"Penny Stock\" Decimal

**상황:** 주가 $0.54321 → 버림 처리

**기대 결과:** $0.5432 (반올림 아닌 버림)

---

### Case 5: The \"Daily Reset\" Amnesia

**상황:** Day 1 PnL +$500 → Day 2 시드 리셋

**기대 결과:** 누적 PnL = +$500 (Trade Ledger 기반)

---

### Case 6: The \"One-Shot Trade\" Lock

**상황:** 09:50에 A 매수 → 10:20에 A 매도 → 10:50에 A 신호 재발생

**기대 결과:** 10:50 신호 `REJECT:ALREADY_TRADED_TODAY`

---

### Case 7: The \"Sign Error\" Bug [New v1.4]

**상황:** Entry = 100, SL_Ratio = -0.05

**기대 결과:** SL = 100 * (1 - 0.05) = 95.0 ✓

**오류 케이스:** SL = 100 * (1 + 0.05) = 105.0 ❌

---

## Article 9. Output Standards (출력 표준) [New v1.4]

**최종 백테스트 리포트 형식:**

```markdown
# GapZone Backtesting Report (v1.4.0)

## Strategy: GapZone
**Backtest Period:** 2025-12-24 ~ 2025-12-31
**Zone:** Zone 1 (Single Strategy)
**Seed:** $400

### Performance Metrics
- **Total Trades:** 45
- **Win Rate:** 62.2%
- **Profit Factor:** 2.15
- **Cumulative PnL:** +$280.50 ← Trade Ledger 합계
- **Sharpe Ratio:** 1.18
- **Max Drawdown:** -4.2%

### Validation Status
✅ VALID RUN (All checks passed)
```

---

## Appendix A: Amendment History (v1.0.0 ~ v1.4.0)

| Version | Date | ID | Key Changes |
|---------|------|----|----|
| v1.0.0 | 2025-01-10 | NA | 초판: Chronological Loop, Single Slot, Validation Suite |
| v1.1.0 | 2025-01-10 | AMD-001 | Trailing Stop, EOD Protocol (검증 로직 제거) |
| v1.2.0 | 2025-01-11 | AMD-002 | Dynamic Rolling Watch, Gap Dual Check |
| v1.3.0 | 2026-01-10 | AMD-003 | Daily Boundary, Moving Average, Per-Stock Limit, Validation Suite 복구 |
| **v1.4.0** | **2026-01-10** | **AMD-005** | **성과 측정, 파라미터 최적화, SL/TP 수식, 필드 명시화, Zone Framework** |

---

**END OF CONSTITUTION v1.4.0**

*"규칙이 없으면 시스템이 없고, 시스템이 없으면 신뢰가 없다."*

*"그리고 규칙이 부실하면, 구현자는 혼돈 속에서 모두 다른 결과를 만든다."*

**—— System Commander, 2026-01-10, 19:55 KST**