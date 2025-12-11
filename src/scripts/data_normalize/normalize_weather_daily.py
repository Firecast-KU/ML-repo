# firecast/scripts/normalize_weather_daily.py
import pandas as pd
from ..paths import PROC_DIR
from ..join_fire_weather import (
    load_weather_raw,
    preprocess_weather,
)

def normalize_weather_daily():
    weather_raw = load_weather_raw()
    weather_daily = preprocess_weather(weather_raw)

    # 컬럼 이름 통일: obs_date -> date
    weather_daily = weather_daily.rename(columns={"obs_date": "date"})

    # 타입 강제
    weather_daily["station_id"] = pd.to_numeric(weather_daily["station_id"], errors="coerce").astype("Int64")
    weather_daily["date"] = pd.to_datetime(weather_daily["date"]).dt.normalize()

    out_path = PROC_DIR / "weather_daily.parquet"
    weather_daily.to_parquet(out_path, index=False)
    print("saved normalized weather_daily ->", out_path)
