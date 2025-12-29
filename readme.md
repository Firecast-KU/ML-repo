# 🔥 Firecast
**기상 데이터 기반 산불 발생 예측 파이프라인**

Firecast는 산불 발생 이력과 기상 관측 데이터를 결합하여  
산불 발생 가능성을 예측하기 위한 데이터 파이프라인 및  
Baseline 예측 모델을 구현한 프로젝트입니다.

---

## 📁 Project Structure
```
firecast/
│
├── src/ # 핵심 파이썬 소스 코드
│ ├── config/ # 전역 설정 및 경로 관리
│ │ ├── init.py
│ │ └── paths.py # 프로젝트 경로 일괄 관리
│ │
│ ├── core/ # 핵심 도메인 로직
│ │ ├── init.py
│ │ ├── fire_events.py # 산불 발생 데이터 처리
│ │ ├── stations.py # 기상 관측소 정보 처리
│ │ ├── weather_daily.py# 일별 기상 데이터 전처리
│ │ └── labeling.py # 산불 발생 라벨링 로직
│ │
│ ├── eda/ # 탐색적 데이터 분석
│ │ ├── init.py
│ │ └── data_exploration.py
│ │
│ ├── pipelines/ # 재현 가능한 데이터 파이프라인
│ │ ├── init.py
│ │ ├── build_fire_events.py
│ │ ├── match_fire_station.py
│ │ └── merge_fire_weather.py
│ │
│ ├── models/ # 모델 학습 코드
│ │ ├── init.py
│ │ ├── lr_baseline.py # Logistic Regression baseline
│ │ └── lr_firecast.py # Firecast 실험 모델
│ │
│ └── validation/ # 데이터 및 결과 검증
│ ├── init.py
│ ├── validate_fire_weather.py
│ └── RN_histogram.py
│
├── data/
│ ├── raw/ # 🔒 원본 데이터 (절대 수정 금지)
│ │ ├── fires/ # 산불 발생 shapefile
│ │ ├── weather/ # ASOS 일별 기상 데이터
│ │ └── meta/ # 관측소 메타데이터
│ │
│ ├── processed/ # 전처리 완료 데이터
│ │ ├── fire_events.parquet
│ │ ├── weather_daily.parquet
│ │ ├── fire_weather_merged.parquet
│ │ └── weather_labeled.parquet
│ │
│ └── features/ # 모델 입력용 데이터
│ └── train_test_split/
│
├── notebooks/ # 분석/실험용 노트북
│ ├── 01_data_exploration.ipynb
│ ├── 02_match_fire_station.ipynb
│ ├── 03_join_fire_weather.ipynb
│ └── LR_Firecast.ipynb
│
├── outputs/ # 결과물 (모델, 그래프 등)
│
├── docs/ # 문서
│ └── analysis.md
│
├── readme.md
└── .gitignore
```


---

## 🔄 Data Processing Pipeline

### 1️⃣ 산불 발생 데이터 처리
- Shapefile 로딩 및 좌표계 통일 (EPSG)
- 중복 이벤트 제거
- 발생 날짜 기준 정제

→ `core/fire_events.py`  
→ `pipelines/build_fire_events.py`

---

### 2️⃣ 기상 관측 데이터 전처리
- ASOS 일별 기상 데이터 로딩
- 컬럼 정규화 및 결측치 처리
- 날짜 단위 집계

→ `core/weather_daily.py`

---

### 3️⃣ 산불–관측소 매칭
- 산불 발생 좌표 기준 최근접 관측소 탐색
- 거리 기반 매칭 (미터 단위)

→ `core/stations.py`  
→ `pipelines/match_fire_station.py`

---

### 4️⃣ 산불–기상 데이터 병합
- 산불 발생일 기준 기상 데이터 Join
- 학습용 테이블 생성

→ `pipelines/merge_fire_weather.py`

---

### 5️⃣ Feature Engineering & Labeling
- 주요 기상 변수 선택
- 산불 발생 여부 이진 라벨 생성

→ `core/labeling.py`

---

## 🤖 Modeling

### Baseline Model
- **Logistic Regression**
- 클래스 불균형 고려 (Recall 중심 평가)

→ `models/lr_baseline.py`  
→ `models/lr_firecast.py`

**입력 Feature 예시**
- 기온
- 강수량
- 습도
- 풍속

**출력**
- 산불 발생 확률 (Binary Classification)

---

## 📊 Validation & Analysis

- 기상 변수 분포 검증
- 산불 발생 / 비발생 비교 시각화
- 데이터 이상치 점검

→ `validation/validate_fire_weather.py`  
→ `validation/RN_histogram.py`

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- GeoPandas
- Scikit-learn
- Matplotlib

---

## 📌 Design Principles

- `data/raw`는 **절대 수정하지 않음**
- 모든 경로는 `src/config/paths.py`에서 관리
- Notebook은 **탐색/실험용**,  
  실제 로직은 **pipeline & src 코드로 재현 가능하게 구현**

---

## 🚀 Future Work

- 시계열 기반 모델 (LSTM / Transformer)
- 공간적 특성 반영 (Grid / Spatial Encoding)
- 기상 예보 데이터 기반 **미래 산불 위험도 예측**

