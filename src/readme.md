readme

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
