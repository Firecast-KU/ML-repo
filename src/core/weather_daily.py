import glob
import os
from typing import List, Optional

import pandas as pd

from src.config.paths import PROC_DIR, WEATHER_RAW_DIR


SKY_MAP = {"DB01": 1, "DB02": 2, "DB03": 3, "DB04": 4}
TARGET_STATIONS = [104, 105]
TARGET_HOURS = [0, 12]


def pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """컬럼 리스트에서 candidates 중 하나와 case-insensitive 매칭되는 컬럼 찾기"""
    lower_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def parse_obs_datetime(series: pd.Series) -> pd.Series:
    """Parse TM-like datetime values robustly."""
    s = series.astype("string").str.strip()

    dt = pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")
    need_retry = dt.isna()
    if need_retry.any():
        dt2 = pd.to_datetime(s[need_retry], format="%Y%m%d%H%M%S", errors="coerce")
        dt.loc[need_retry] = dt2

    need_retry = dt.isna()
    if need_retry.any():
        dt3 = pd.to_datetime(s[need_retry], errors="coerce")
        dt.loc[need_retry] = dt3

    return dt


def _mode_or_na(series: pd.Series):
    s = series.dropna().astype("string")
    if s.empty:
        return pd.NA
    mode = s.mode()
    if mode.empty:
        return s.iloc[0]
    return mode.iloc[0]


def _dtr_0012(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return pd.NA
    return float(abs(s.max() - s.min()))


def load_weather_raw() -> pd.DataFrame:
    """Load and concatenate all raw weather CSV files."""
    csv_paths = sorted(glob.glob(os.path.join(str(WEATHER_RAW_DIR), "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No weather CSV found in: {WEATHER_RAW_DIR}")

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
    """
    Build daily weather features from hourly raw schema.

    Output columns:
    - station_id, date
    - TA: mean TA at 00/12
    - TA_dtr: |TA(12)-TA(00)| proxy (implemented as max-min across 00/12)
    - POP: mean precipitation probability at 00/12
    - is_precip: max precipitation indicator at 00/12
    - WD_sin, WD_cos: mean wind direction components at 00/12
    - SKY: sky state code mapped to 1..4 (DB01..DB04), mode at 00/12
    """
    stn_col = pick_column(weather_raw.columns, ["STN", "stn", "지점", "지점번호"])
    tm_col = pick_column(weather_raw.columns, ["TM", "일시", "datetime", "date"])
    ta_col = pick_column(weather_raw.columns, ["TA", "평균기온", "평균기온(°C)"])
    pop_col = pick_column(weather_raw.columns, ["POP", "강수확률"])
    precip_col = pick_column(weather_raw.columns, ["is_precip", "IS_PRECIP"])
    wd_sin_col = pick_column(weather_raw.columns, ["WD_sin", "wd_sin"])
    wd_cos_col = pick_column(weather_raw.columns, ["WD_cos", "wd_cos"])
    sky_col = pick_column(weather_raw.columns, ["SKY", "sky"])

    if stn_col is None or tm_col is None:
        raise ValueError("Required columns not found: station/date")

    keep_cols = [
        c
        for c in [stn_col, tm_col, ta_col, pop_col, precip_col, wd_sin_col, wd_cos_col, sky_col]
        if c is not None
    ]
    weather = weather_raw[keep_cols].copy()

    rename_map = {
        stn_col: "station_id",
        tm_col: "obs_datetime",
    }
    if ta_col:
        rename_map[ta_col] = "TA"
    if pop_col:
        rename_map[pop_col] = "POP"
    if precip_col:
        rename_map[precip_col] = "is_precip"
    if wd_sin_col:
        rename_map[wd_sin_col] = "WD_sin"
    if wd_cos_col:
        rename_map[wd_cos_col] = "WD_cos"
    if sky_col:
        rename_map[sky_col] = "SKY"

    weather = weather.rename(columns=rename_map)

    for col in ["station_id", "TA", "POP", "is_precip", "WD_sin", "WD_cos"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
            weather[col] = weather[col].replace([-99, -999, -9999], pd.NA)

    if "SKY" in weather.columns:
        weather["SKY"] = weather["SKY"].astype("string").str.strip().str.upper()

    weather["obs_datetime"] = parse_obs_datetime(weather["obs_datetime"])
    weather["date"] = weather["obs_datetime"].dt.normalize()
    weather["hour"] = weather["obs_datetime"].dt.hour

    weather = weather[weather["station_id"].isin(TARGET_STATIONS)]
    weather = weather[weather["date"].notna()]

    w_0012 = weather[weather["hour"].isin(TARGET_HOURS)].copy()
    if w_0012.empty:
        w_0012 = weather.copy()

    keys = ["station_id", "date"]
    weather_daily = w_0012[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)

    if "TA" in w_0012.columns:
        ta_daily = w_0012.groupby(keys, as_index=False)["TA"].mean()
        ta_dtr = w_0012.groupby(keys, as_index=False)["TA"].agg(_dtr_0012).rename(columns={"TA": "TA_dtr"})
        weather_daily = weather_daily.merge(ta_daily, on=keys, how="left")
        weather_daily = weather_daily.merge(ta_dtr, on=keys, how="left")

    if "POP" in w_0012.columns:
        pop_daily = w_0012.groupby(keys, as_index=False)["POP"].mean()
        weather_daily = weather_daily.merge(pop_daily, on=keys, how="left")

    if "is_precip" in w_0012.columns:
        precip_daily = w_0012.groupby(keys, as_index=False)["is_precip"].max()
        weather_daily = weather_daily.merge(precip_daily, on=keys, how="left")

    if "WD_sin" in w_0012.columns:
        wd_sin_daily = w_0012.groupby(keys, as_index=False)["WD_sin"].mean()
        weather_daily = weather_daily.merge(wd_sin_daily, on=keys, how="left")

    if "WD_cos" in w_0012.columns:
        wd_cos_daily = w_0012.groupby(keys, as_index=False)["WD_cos"].mean()
        weather_daily = weather_daily.merge(wd_cos_daily, on=keys, how="left")

    if "SKY" in w_0012.columns:
        sky_daily = w_0012.groupby(keys, as_index=False)["SKY"].agg(_mode_or_na)
        sky_num = sky_daily["SKY"].map(SKY_MAP)
        sky_num_fallback = pd.to_numeric(sky_daily["SKY"], errors="coerce")
        sky_daily["SKY"] = sky_num.fillna(sky_num_fallback).astype("Int64")
        weather_daily = weather_daily.merge(sky_daily, on=keys, how="left")

    if "is_precip" in weather_daily.columns:
        weather_daily["is_precip"] = weather_daily["is_precip"].fillna(0).astype("Int64")

    return weather_daily


def build_past_n_days_features(df: pd.DataFrame, n_days: int = 3) -> pd.DataFrame:
    """Build lagged weather features (past n days) by station_id/date."""
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
    """Save normalized daily weather data to weather_daily.parquet."""
    weather_raw = load_weather_raw()
    weather_daily = preprocess_weather(weather_raw)

    weather_daily["station_id"] = pd.to_numeric(
        weather_daily["station_id"], errors="coerce"
    ).astype("Int64")
    weather_daily["date"] = pd.to_datetime(weather_daily["date"]).dt.normalize()

    out_path = PROC_DIR / "weather_daily.parquet"
    weather_daily.to_parquet(out_path, index=False)
    print("saved normalized weather_daily ->", out_path)

    return weather_daily
