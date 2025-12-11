# src/core/weather_daily.py

import glob
import os
from typing import List, Optional

import pandas as pd

from src.config.paths import PROC_DIR, WEATHER_RAW_DIR


def pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """컬럼 리스트에서 candidates 중 하나와 case-insensitive 매칭되는 컬럼 찾기"""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_weather_raw() -> pd.DataFrame:
    """raw weather CSV들을 전부 읽어서 하나의 DataFrame으로 병합"""
    csv_paths = sorted(glob.glob(os.path.join(str(WEATHER_RAW_DIR), "*.csv")))
    print("weather files:", [os.path.basename(p) for p in csv_paths])

    dfs = []
    for p in csv_paths:
        try:
            df0 = pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            df0 = pd.read_csv(p, encoding="cp949", low_memory=False)
        df0["__source_file"] = os.path.basename(p)
        dfs.append(df0)

    weather_raw = pd.concat(dfs, ignore_index=True)
    print("weather_raw shape:", weather_raw.shape)
    print("sample columns:", list(weather_raw.columns)[:20])
    return weather_raw


def preprocess_weather(weather_raw: pd.DataFrame) -> pd.DataFrame:
    """raw weather → station_id + obs_date + (TA, TMN, TMX, RN) 형태의 daily 데이터로 축약"""
    stn_col = pick_column(weather_raw.columns, ["STN", "stn", "stnid", "지점", "지점번호"])
    date_col = pick_column(
        weather_raw.columns, ["TM", "날짜", "date", "YYYYMMDD", "일시"]
    )
    tavg_col = pick_column(
        weather_raw.columns, ["TA", "TAVG", "평균기온", "avgTa", "평균기온(°C)"]
    )
    tmin_col = pick_column(
        weather_raw.columns, ["TMN", "최저기온", "최저기온(°C)"]
    )
    tmax_col = pick_column(
        weather_raw.columns, ["TMX", "최고기온", "최고기온(°C)"]
    )
    prcp_col = pick_column(
        weather_raw.columns, ["RN", "PRCP", "강수량", "일강수량(mm)"]
    )

    print("stn_col :", stn_col)
    print("date_col:", date_col)
    print(
        "tavg_col:",
        tavg_col,
        ", tmin_col:",
        tmin_col,
        ", tmax_col:",
        tmax_col,
        ", prcp_col:",
        prcp_col,
    )

    if stn_col is None or date_col is None:
        raise ValueError("지점번호 또는 날짜 컬럼을 찾을 수 없습니다.")

    keep_cols = [
        c
        for c in [stn_col, date_col, tavg_col, tmin_col, tmax_col, prcp_col, "__source_file"]
        if c is not None
    ]
    weather = weather_raw[keep_cols].copy()

    # 컬럼 이름 표준화
    rename_map = {
        stn_col: "station_id",
        date_col: "obs_datetime",
    }
    if tavg_col:
        rename_map[tavg_col] = "TA"
    if tmin_col:
        rename_map[tmin_col] = "TMN"
    if tmax_col:
        rename_map[tmax_col] = "TMX"
    if prcp_col:
        rename_map[prcp_col] = "RN"

    weather = weather.rename(columns=rename_map)

    # 날짜 파싱
    def parse_obs_date(s: str):
        s = str(s)
        if len(s) >= 8 and s[:8].isdigit():
            return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT

    weather["obs_date"] = weather["obs_datetime"].astype("string").map(parse_obs_date)
    weather["obs_date"] = weather["obs_date"].dt.normalize()

    # target station만 (104: 북강릉, 105: 강릉)
    target_stations = [104, 105]
    weather["station_id"] = pd.to_numeric(weather["station_id"], errors="coerce")
    w_sub = weather[
        weather["station_id"].isin(target_stations) & weather["obs_date"].notna()
        ].copy()

    # 일 단위로 집계
    agg_dict = {}
    for col in ["TA", "TMN", "TMX", "RN"]:
        if col in w_sub.columns:
            agg_dict[col] = "mean"

    weather_daily = (
        w_sub.groupby(["station_id", "obs_date"]).agg(agg_dict).reset_index()
    )
    return weather_daily


def build_past_n_days_features(df: pd.DataFrame, n_days: int = 3) -> pd.DataFrame:
    """station_id+date 기준으로 과거 n일치 기상 피처를 옆으로 붙이는 함수"""
    base_cols = ["station_id", "date"]
    feature_cols = [c for c in df.columns if c not in base_cols]

    out = df[base_cols].copy()
    for k in range(1, n_days + 1):
        shifted = df.copy()
        shifted["date"] = shifted["date"] + pd.Timedelta(days=k)
        rename_map = {c: f"{c}_minus{k}d" for c in feature_cols}
        shifted = shifted[base_cols + feature_cols].rename(columns=rename_map)
        out = out.merge(shifted, how="left", on=base_cols)
    return out


def normalize_weather_daily() -> pd.DataFrame:
    """raw CSV → 정규화된 weather_daily.parquet 저장"""
    weather_raw = load_weather_raw()
    weather_daily = preprocess_weather(weather_raw)

    # 컬럼 이름 통일: obs_date -> date
    weather_daily = weather_daily.rename(columns={"obs_date": "date"})

    # 타입 강제
    weather_daily["station_id"] = (
        pd.to_numeric(weather_daily["station_id"], errors="coerce").astype("Int64")
    )
    weather_daily["date"] = pd.to_datetime(weather_daily["date"]).dt.normalize()

    # RN이 있는 경우에만 빈값을 0으로 처리
    if "RN" in weather_daily.columns:
        weather_daily["RN"] = weather_daily["RN"].fillna(0.0)


    out_path = PROC_DIR / "weather_daily.parquet"
    weather_daily.to_parquet(out_path, index=False)
    print("saved normalized weather_daily ->", out_path)

    return weather_daily
