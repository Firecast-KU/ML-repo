# src/core/fire_events.py  (혹은 firecast/core/fire_events.py)

import geopandas as gpd
import pandas as pd
from src.config.paths import PROC_DIR


def normalize_fire_events(
        input_path=None,
        output_path=None,
) -> gpd.GeoDataFrame:
    """
    fires_with_manual_station.parquet을 정규화된 fire_events.parquet으로 변환.

    - fire_datetime/fire_date/year/month 생성
    - 좌표계 EPSG:4326으로 변환해 lon/lat 추가
    - fire_id 부여
    """
    if input_path is None:
        input_path = PROC_DIR / "fires_with_manual_station.parquet"
    if output_path is None:
        output_path = PROC_DIR / "fire_events.parquet"

    # 1) 기존 결과 읽기 (관측소 매칭까지 끝난 상태)
    gdf = gpd.read_parquet(input_path)

    # 2) 날짜/시간 통일
    if "OCCRR_DATE" in gdf.columns:
        gdf["fire_datetime"] = pd.to_datetime(gdf["OCCRR_DATE"])
    elif "OCCRR_DTM" in gdf.columns:
        # YYYYMMDDHHMM 형태 가정
        s = gdf["OCCRR_DTM"].astype("string")
        gdf["fire_datetime"] = pd.to_datetime(
            s.str.slice(0, 12),
            format="%Y%m%d%H%M",
            errors="coerce",
        )
    else:
        raise ValueError("날짜 컬럼(OCCRR_DATE / OCCRR_DTM)을 찾을 수 없습니다.")

    gdf["fire_date"] = gdf["fire_datetime"].dt.normalize()
    gdf["year"] = gdf["fire_date"].dt.year
    gdf["month"] = gdf["fire_date"].dt.month

    # 3) 좌표계 및 lon/lat 추가
    if gdf.crs is None:
        # 필요하면 수동으로 crs 지정
        # gdf = gdf.set_crs("EPSG:5179")
        raise ValueError("CRS 정보가 없습니다. shapefile CRS를 먼저 확인하세요.")

    gdf_4326 = gdf.to_crs("EPSG:4326")
    gdf["lon"] = gdf_4326.geometry.x
    gdf["lat"] = gdf_4326.geometry.y

    # 4) fire_id 추가 (없으면 index 기반)
    if "fire_id" not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf["fire_id"] = gdf.index

    # 5) 최종 컬럼 순서 정리
    cols = [
        "fire_id",
        "fire_datetime",
        "fire_date",
        "year",
        "month",
        "CTPRV_NM",
        "SGNG_NM",
        "station_id",
        "dist_m",
        "lon",
        "lat",
        "geometry",
    ]
    cols = [c for c in cols if c in gdf.columns]
    gdf = gdf[cols]

    gdf.to_parquet(output_path, index=False)
    print("saved normalized fire_events ->", output_path)

    return gdf
