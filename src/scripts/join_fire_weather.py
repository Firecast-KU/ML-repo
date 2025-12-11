import glob
import os
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from paths import PROC_DIR, WEATHER_RAW_DIR


def pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_fires_with_station():
    fire_proc_path = PROC_DIR / "fires_with_manual_station.parquet"
    fires = gpd.read_parquet(fire_proc_path)
    print("fires shape:", fires.shape)
    print("fires columns:", list(fires.columns))

    # fire_date 생성
    if "OCCRR_DATE" in fires.columns:
        fires["fire_date"] = pd.to_datetime(fires["OCCRR_DATE"])
    elif "OCCRR_DTM" in fires.columns:
        fires["OCCRR_DATE_STR"] = fires["OCCRR_DTM"].astype("string").str.slice(0, 8)
        fires["fire_date"] = pd.to_datetime(
            fires["OCCRR_DATE_STR"], format="%Y%m%d", errors="coerce"
        )
    else:
        raise ValueError(
            "산불 데이터에서 날짜 컬럼(OCCRR_DATE / OCCRR_DTM)을 찾을 수 없습니다."
        )

    fires["fire_date"] = fires["fire_date"].dt.normalize()
    return fires


def load_weather_raw():
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

    # target station만
    target_stations = [104, 105]
    weather["station_id"] = pd.to_numeric(weather["station_id"], errors="coerce")
    w_sub = weather[
        weather["station_id"].isin(target_stations) & weather["obs_date"].notna()
        ].copy()

    agg_dict = {}
    for col in ["TA", "TMN", "TMX", "RN"]:
        if col in w_sub.columns:
            agg_dict[col] = "mean"

    weather_daily = (
        w_sub.groupby(["station_id", "obs_date"]).agg(agg_dict).reset_index()
    )
    return weather_daily


def merge_fire_weather(fires: gpd.GeoDataFrame, weather_daily: pd.DataFrame):
    fires["station_id"] = pd.to_numeric(fires["station_id"], errors="coerce")

    fires_for_join = fires.copy().rename(columns={"fire_date": "date"})
    weather_for_join = weather_daily.rename(columns={"obs_date": "date"})

    merged = fires_for_join.merge(
        weather_for_join,
        how="left",
        on=["station_id", "date"],
        suffixes=("", "_w"),
    )
    return merged


def build_past_n_days_features(df: pd.DataFrame, n_days: int = 3) -> pd.DataFrame:
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


def main():
    fires = load_fires_with_station()
    weather_raw = load_weather_raw()
    weather_daily = preprocess_weather(weather_raw)

    merged = merge_fire_weather(fires, weather_daily)
    out_path_parquet = PROC_DIR / "fire_weather_merged.parquet"
    out_path_csv = PROC_DIR / "fire_weather_merged.csv"
    merged.to_parquet(out_path_parquet, index=False)
    merged.to_csv(out_path_csv, index=False)
    print("saved ->", out_path_parquet)
    print("saved ->", out_path_csv)

    weather_daily_for_features = weather_daily.rename(columns={"obs_date": "date"})
    weather_features_3d = build_past_n_days_features(weather_daily_for_features, n_days=3)
    print("weather_features_3d shape:", weather_features_3d.shape)
    # 필요하면 여기서도 저장 가능
    # weather_features_3d.to_parquet(PROC_DIR / "weather_features_3d.parquet", index=False)


if __name__ == "__main__":
    main()
