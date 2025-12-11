import glob
import os

import geopandas as gpd

from paths import FIRE_RAW_DIR, PROC_DIR
from stations import (
    WeatherStationRegistry,
    attach_nearest_station,
)


def load_filtered_fires():
    shp_list = glob.glob(os.path.join(str(FIRE_RAW_DIR), "*.shp"))
    assert len(shp_list) == 1, f"*.shp 파일이 1개가 아닙니다: {shp_list}"
    fire_shp = shp_list[0]
    print("FIRE_SHP:", fire_shp)

    fires = gpd.read_file(fire_shp)
    print("fires.shape:", fires.shape)
    print("fires.crs:", fires.crs)

    fires["YEAR"] = fires["OCCRR_DTM"].astype(str).str[:4]
    fires = fires[fires["YEAR"].isin(["2020", "2021"])].copy()
    fires = fires.drop(columns=["YEAR"])

    print("filtered fires.shape:", fires.shape)
    return fires


def main():
    fires = load_filtered_fires()

    registry = WeatherStationRegistry.default_kma_gangneung()
    fires_with_station = attach_nearest_station(
        fire_gdf=fires,
        registry=registry,
        distance_col="dist_m",
    )

    print("result shape:", fires_with_station.shape)
    print("[station_id counts]")
    print(fires_with_station["station_id"].value_counts())

    print("\n[distance stats (m)]")
    print(fires_with_station["dist_m"].describe())

    out_parquet = PROC_DIR / "fires_with_manual_station.parquet"
    fires_with_station.to_parquet(out_parquet, index=False)
    print("saved parquet ->", out_parquet)

    out_shp = PROC_DIR / "fires_with_manual_station.shp"
    fires_with_station.to_file(out_shp)
    print("saved shapefile ->", out_shp)


if __name__ == "__main__":
    main()
