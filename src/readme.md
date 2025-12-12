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
├── data/
│   ├── raw/                # 원본 데이터 (수정 금지)
│   │   ├── fires/          # 산불 발생 shapefile
│   │   │   └── TB_FFAS_FF_OCCRR_42.(shp/shx/dbf/prj/cpg)
│   │   ├── weather/        # 기상 관측 CSV
│   │   │   ├── SURFACE_ASOS_104_DAY_2020_2021.csv
│   │   │   └── SURFACE_ASOS_105_DAY_2020_2021.csv
│   │   └── meta/           # 관측소 목록, 좌표정보 등 (선택)
│   │
│   ├── processed/          # 전처리 완료 데이터 (학습용)
│   │   ├── merged_fire_weather.parquet
│   │   ├── fire_with_station.parquet
│   │   └── fire_clean.csv
│   │
│   └── features/           # 모델 입력용 numpy/csv 등
│       └── train_test_split/
│           ├── X_train.npy
│           ├── y_train.npy
│           ├── X_test.npy
│           └── y_test.npy
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_match_fire_station.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_train_LR_model.ipynb
│   └── 05_evaluate_and_visualize.ipynb
│
├── scripts/
│   ├── match_fire_station.py
│   ├── preprocess_fire_data.py
│   ├── preprocess_weather_data.py
│   └── join_fire_weather.py
│
└── outputs/
    ├── model_checkpoints/
    ├── evaluation_reports/
    └── visualizations/
```


---

## 🔄 Data Processing Pipeline

1. **산불 발생 데이터 정제**
    - Shapefile 로딩
    - 좌표계(EPSG) 통일
    - 중복 및 이상치 제거

2. **기상 관측 데이터 전처리**
    - 일별 ASOS 기상 데이터 로딩
    - 컬럼 정규화 및 결측치 처리

3. **산불–관측소 매칭**
    - 산불 발생 위치 기준 최근접 관측소 탐색
    - 거리 기반 매칭 (미터 단위)

4. **산불–기상 데이터 병합**
    - 날짜 기준 Join
    - 학습용 테이블 생성

5. **Feature Engineering & Labeling**
    - 기상 변수 선택
    - 산불 발생 여부 이진 라벨 생성

---

## 🤖 Modeling

- **Baseline Model**
    - Logistic Regression
    - 클래스 불균형 고려
- **입력 Feature**
    - 기온, 강수량, 습도, 풍속 등 기상 변수
- **출력**
    - 산불 발생 확률

---

## 📊 Validation & Analysis

- 데이터 분포 검증
- 기상 변수 히스토그램 분석
- 산불 발생/비발생 비교 시각화

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- GeoPandas
- Scikit-learn
- Matplotlib / Seaborn

---

## 📌 Notes

- `data/raw` 디렉토리는 **절대 수정하지 않습니다**
- 모든 전처리 결과는 `data/processed`에 저장됩니다
- 경로 관리는 `src/config/paths.py`에서 일괄 처리합니다

---

## 🚀 Future Work

- 시계열 기반 모델 (LSTM / Transformer)
- 공간적 특성 반영 (Grid / Spatial Encoding)
- 기상 예보 데이터 활용한 미래 예측
