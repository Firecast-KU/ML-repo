import glob
import os

import geopandas as gpd
import pandas as pd

from paths import FIRE_RAW_DIR, WEATHER_RAW_DIR


def load_fire_shapefile():
    shp_candidates = glob.glob(os.path.join(str(FIRE_RAW_DIR), "*.shp"))
    assert len(shp_candidates) == 1, f"*.shp가 1개가 아닙니다: {shp_candidates}"
    fire_shp = shp_candidates[0]
    print("FIRE_SHP:", fire_shp)

    fires = gpd.read_file(fire_shp)
    print("rows, cols:", fires.shape)
    print("CRS:", fires.crs)
    print("bounds:", fires.total_bounds)
    print("columns:", list(fires.columns))
    return fires


def add_year_month(fires: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fires["YEAR"] = fires["OCCRR_DTM"].astype(str).str[:4]
    fires = fires[fires["YEAR"].isin(["2020", "2021"])].copy()
    fires = fires.drop(columns=["YEAR"])

    # 날짜 파싱
    if "OCCRR_DTM" in fires.columns:
        fires["OCCRR_DATE_STR"] = fires["OCCRR_DTM"].astype("string").str.slice(0, 8)
        fires["OCCRR_DATE"] = pd.to_datetime(
            fires["OCCRR_DATE_STR"], errors="coerce", format="%Y%m%d"
        )
        fires["YEAR"] = fires["OCCRR_DATE"].dt.year
        fires["MONTH"] = fires["OCCRR_DATE"].dt.month
    else:
        print("날짜 컬럼(OCCRR_DTM)이 없어 파생을 건너뜁니다.")

    return fires


def quick_fire_stats(fires: gpd.GeoDataFrame):
    show_cols = [
        c
        for c in [
            "CTPRV_NM",
            "SGNG_NM",
            "EMNDN_NM",
            "OCCCR_RI",
            "OCCRR_DTM",
            "OCUR_DYWK",
        ]
        if c in fires.columns
    ]
    for c in show_cols:
        print(f"\n[{c}] top values:")
        print(fires[c].astype("string").value_counts(dropna=False).head(10))


def inspect_weather_csv_meta():
    from paths import WEATHER_RAW_DIR

    def pick(cols, candidates):
        cols_low = {c.lower(): c for c in cols}
        for c in candidates:
            if c.lower() in cols_low:
                return cols_low[c.lower()]
        return None

    csv_list = sorted(glob.glob(os.path.join(str(WEATHER_RAW_DIR), "*.csv")))
    print("weather csv files:", len(csv_list))
    for p in csv_list:
        print("-", os.path.basename(p))

    meta = []
    for p in csv_list[:2]:
        df0 = pd.read_csv(p, nrows=50, low_memory=False, encoding="cp949")
        stn_col = pick(df0.columns, ["STN", "stn", "stnid", "지점", "지점번호"])
        lat_col = pick(df0.columns, ["LAT", "lat", "위도"])
        lon_col = pick(df0.columns, ["LON", "lon", "경도"])
        tmx_col = pick(df0.columns, ["TM_X", "tm_x", "X", "x"])
        tmy_col = pick(df0.columns, ["TM_Y", "tm_y", "Y", "y"])
        meta.append(
            {
                "file": os.path.basename(p),
                "stn_col": stn_col,
                "lat_col": lat_col,
                "lon_col": lon_col,
                "tmx_col": tmx_col,
                "tmy_col": tmy_col,
                "n_rows_look": len(df0),
            }
        )
    meta_df = pd.DataFrame(meta)
    print(meta_df)


def main():
    fires = load_fire_shapefile()
    fires = add_year_month(fires)
    quick_fire_stats(fires)
    inspect_weather_csv_meta()


if __name__ == "__main__":
    main()
