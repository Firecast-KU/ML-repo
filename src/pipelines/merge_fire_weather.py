import geopandas as gpd
import pandas as pd

from src.config.paths import PROC_DIR
from src.core.weather_daily import (
    normalize_weather_daily,
    build_past_n_days_features,
)


def load_fires_with_station() -> gpd.GeoDataFrame:
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


def merge_fire_weather(
        fires: gpd.GeoDataFrame,
        weather_daily: pd.DataFrame,
) -> pd.DataFrame:
    """정규화된 fire_events + weather_daily를 station_id+date로 조인"""
    fires["station_id"] = pd.to_numeric(fires["station_id"], errors="coerce")

    # fires: fire_date, weather_daily: date 컬럼 기준
    fires_for_join = fires.copy().rename(columns={"fire_date": "date"})

    merged = fires_for_join.merge(
        weather_daily,
        how="left",
        on=["station_id", "date"],
        suffixes=("", "_w"),
    )
    return merged


def main():
    fires = load_fires_with_station()

    weather_daily = normalize_weather_daily()

    merged = merge_fire_weather(fires, weather_daily)

    out_parquet = PROC_DIR / "fire_weather_merged.parquet"
    out_csv = PROC_DIR / "fire_weather_merged.csv"
    merged.to_parquet(out_parquet, index=False)
    merged.to_csv(out_csv, index=False)
    print("saved ->", out_parquet)
    print("saved ->", out_csv)

    # 과거 n일 피처 생성
    weather_features_3d = build_past_n_days_features(
        weather_daily,   # 이미 date 컬럼 있음
        n_days=3,
    )
    print("weather_features_3d shape:", weather_features_3d.shape)


if __name__ == "__main__":
    main()
