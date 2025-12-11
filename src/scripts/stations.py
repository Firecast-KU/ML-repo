# firecast/scripts/stations.py
from dataclasses import dataclass
from typing import List
import pandas as pd
import geopandas as gpd

@dataclass
class WeatherStation:
    station_id: int
    name_kr: str
    name_en: str
    lat: float     # WGS84 lat
    lon: float     # WGS84 lon


class WeatherStationRegistry:
    """관측소 정보를 관리하는 레지스트리."""
    def __init__(self, stations: List[WeatherStation]):
        self.stations = stations

    @classmethod
    def default_kma_gangneung(cls) -> "WeatherStationRegistry":
        """북강릉/강릉 기본 레지스트리"""
        stations = [
            WeatherStation(
                station_id=104,
                name_kr='북강릉',
                name_en='Bukgangneung',
                lat=37.80456,
                lon=128.85535,
            ),
            WeatherStation(
                station_id=105,
                name_kr='강릉',
                name_en='Gangneung',
                lat=37.75146,
                lon=128.89098,
            ),
        ]
        return cls(stations)

    def to_geodataframe(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        df = pd.DataFrame([s.__dict__ for s in self.stations])
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )
        if crs != "EPSG:4326":
            gdf = gdf.to_crs(crs)
        return gdf


def attach_nearest_station(
        fire_gdf: gpd.GeoDataFrame,
        registry: WeatherStationRegistry,
        distance_col: str = "dist_m",
) -> gpd.GeoDataFrame:
    """
    산불 지점에 최근접 관측소 정보 붙이기 (거리 단위: meter, EPSG:5179 기준).
    """
    if fire_gdf.crs is None:
        raise ValueError("fire_gdf.crs 가 None입니다. CRS를 먼저 지정해 주세요.")

    stations_wgs84 = registry.to_geodataframe(crs="EPSG:4326")

    # fire 좌표 WGS84 맞추기
    if fire_gdf.crs.to_string() != "EPSG:4326":
        fires_wgs84 = fire_gdf.to_crs("EPSG:4326")
    else:
        fires_wgs84 = fire_gdf

    # 거리 계산용 투영좌표계
    fires_proj = fires_wgs84.to_crs("EPSG:5179")
    stations_proj = stations_wgs84.to_crs("EPSG:5179")

    joined = gpd.sjoin_nearest(
        fires_proj,
        stations_proj[["station_id", "name_kr", "name_en", "geometry"]],
        how="left",
        distance_col=distance_col,
    )

    # 원래 CRS로 다시 돌려서 반환
    if fire_gdf.crs.to_string() != "EPSG:5179":
        joined = joined.to_crs(fire_gdf.crs)

    return joined
