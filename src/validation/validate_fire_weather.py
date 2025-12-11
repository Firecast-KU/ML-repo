# src/validation/validate_fire_weather.py
from pathlib import Path
import pandas as pd
from src.config.paths import PROC_DIR


def validate_fire_weather():
    fire_path = PROC_DIR / "fire_events.parquet"
    merged_path = PROC_DIR / "fire_weather_merged.parquet"

    if not fire_path.exists():
        print(f"[ERROR] {fire_path} 가 없습니다. 먼저 fire_events 파이프라인을 실행하세요.")
        return
    if not merged_path.exists():
        print(f"[ERROR] {merged_path} 가 없습니다. 먼저 merge_fire_weather 파이프라인을 실행하세요.")
        return

    fires = pd.read_parquet(fire_path)
    merged = pd.read_parquet(merged_path)

    print("fires rows:", len(fires))
    print("merged rows:", len(merged))

    if len(fires) == len(merged):
        print("[OK] Row count matches (left join 기준)")
    else:
        print("[WARN] Row count mismatch!")

    # 기상 Null 비율
    weather_cols = [c for c in ["TA", "TMN", "TMX", "RN"] if c in merged.columns]
    print("\n[Null ratio for weather columns]")
    print(merged[weather_cols].isna().mean())

    # 날짜 diff
    if "fire_date" in merged.columns and "date" in merged.columns:
        merged["date_diff"] = (merged["date"] - merged["fire_date"]).dt.days
        print("\n[Date diff (date - fire_date)]")
        print(merged["date_diff"].value_counts().head())

    # station 분포
    print("\n[station_id counts]")
    print(merged["station_id"].value_counts())


if __name__ == "__main__":
    validate_fire_weather()
