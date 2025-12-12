# Analysis Log

본 문서는 Firecast 프로젝트에서 수행한 주요 실험 결과와
그에 대한 해석, 그리고 다음 단계로의 의사결정 근거를 기록한다.

---

## 1. Baseline Model: Logistic Regression (Weather-only)

### 1.1 Experiment Setup
- Task: 기상 데이터 기반 산불 발생 여부 이진 분류
- Model: Logistic Regression
- Handling class imbalance: `class_weight="balanced"`
- Features:
    - TA: 평균기온
    - TMN: 최저기온
    - TMX: 최고기온
    - RN: 일강수량
- Label:
    - `fire_label = 1` if wildfire occurred at (station_id, date)
    - `fire_label = 0` otherwise

### 1.2 Baseline Performance
- ROC-AUC: **0.741**
- Recall (fire = 1): **0.73**
- Precision (fire = 1): **0.53**

기상 변수만을 사용한 단순 선형 모델임에도 불구하고,
산불 발생 신호를 일정 수준 이상 포착할 수 있음을 확인하였다.

---

## 2. Coefficient Interpretation

Logistic Regression 계수를 통해 각 feature의 영향 방향을 분석하였다.

| Feature | Coefficient | Interpretation |
|------|-----------|---------------|
| TA (Avg Temp) | +0.258 | 평균기온이 높을수록 산불 발생 확률 증가 |
| TMN (Min Temp) | -0.318 | 야간 기온이 낮을수록 산불 발생 확률 감소 |
| TMX (Max Temp) | -0.026 | 단독 영향은 미미 |
| RN (Rainfall) | -0.008 | 강수량이 많을수록 산불 발생 확률 감소 |

해당 결과는 기온 상승과 건조한 환경이 산불 위험을 증가시킨다는
도메인 지식과 정합적인 방향성을 보인다.

---

## 3. Feature Engineering: Diurnal Temperature Range (DTR)

### 3.1 Motivation
계수 분석 결과,
- TA는 양의 영향
- TMN은 음의 영향을 보였으며

이는 **일교차(Diurnal Temperature Range)**가
산불 발생과 직접적인 연관이 있을 가능성을 시사한다.

이에 따라 다음과 같은 파생 feature를 추가하였다.


---

## 4. LR Baseline with DTR Feature

### 4.1 Updated Feature Set
- TA, TMN, TMX, RN
- **DTR (TMX - TMN)**

### 4.2 Performance Comparison

| Model | ROC-AUC |
|----|--------|
| Baseline (without DTR) | 0.74119 |
| Baseline + DTR | 0.74140 |

성능 지표(ROC-AUC)는 거의 동일한 수준을 유지하였다.

---

## 5. DTR Coefficient Analysis

| Feature | Coefficient |
|------|-----------|
| DTR | **+0.097** |

- DTR은 **양(+)의 계수**를 가짐
- 즉, 일교차가 클수록 산불 발생 확률이 증가하는 경향 확인

이는
> "낮에는 고온, 밤에는 저온 → 건조 환경 조성 → 산불 위험 증가"

라는 도메인 지식과 일치한다.

다만 Logistic Regression은 선형 모델이므로,
기존 기온 변수(TA, TMN, TMX)와의 상관성으로 인해
즉각적인 성능 향상은 제한적인 것으로 판단된다.

---

## 6. Conclusion & Next Steps

- DTR feature는 **의미적으로 유효한 변수**임이 확인되었다.
- 선형 모델에서는 성능 향상이 제한적이었으나,
  이는 feature 중복 및 모델 표현력 한계로 해석된다.
- DTR과 같은 파생 기온 변수는
  **비선형 모델(RandomForest, Gradient Boosting 등)**에서
  더 큰 효과를 낼 가능성이 높다.

### Planned Next Steps
1. 누적 강수량(RN_sum_n_days) 및 연속 무강수일(dry spell) feature 추가
2. Tree-based model(RandomForest) baseline 구축
3. Feature importance 기반 추가 feature refinement

---
