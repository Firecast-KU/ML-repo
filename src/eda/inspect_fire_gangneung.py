import geopandas as gpd

from src.config.paths import FIRE_RAW_DIR

"""
[EDA / INSPECTION SCRIPT]

- 산불 발생 shapefile 구조 확인
- CRS, 컬럼, bounds sanity check
- 특정 지역(강릉시) subset 테스트

※ 모델 학습/예측 파이프라인과는 무관
※ 데이터 변경 시 검증용으로만 사용
"""

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
