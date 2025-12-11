import geopandas as gpd

from paths import FIRE_RAW_DIR


def load_gangneung_fires():
    shp_path = FIRE_RAW_DIR / "TB_FFAS_FF_OCCRR_42.shp"
    gdf = gpd.read_file(shp_path)

    print("rows, cols:", gdf.shape)
    print("CRS:", gdf.crs)
    print("bounds:", gdf.total_bounds)
    print("columns:", list(gdf.columns))

    gangneung = gdf[gdf["SGNG_NM"] == "강릉시"].copy()
    print("Gangneung rows:", len(gangneung))
    return gangneung


def main():
    _ = load_gangneung_fires()
    # TODO: 여기서 fire_weather_merged.parquet 불러와서
    #      간단한 Logistic/Linear Regression 실험 코드 추가 예정


if __name__ == "__main__":
    main()
