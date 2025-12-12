import pandas as pd
from src.config.paths import PROC_DIR

def build_labels():
    fires = pd.read_parquet(PROC_DIR / "fire_events.parquet")
    weather = pd.read_parquet(PROC_DIR / "weather_daily.parquet")

    # fire_date를 date로 통일
    fires = fires.rename(columns={"fire_date": "date"})
    fires["date"] = pd.to_datetime(fires["date"]).dt.normalize()

    # station_id + date 기준으로 산불 발생 여부 생성
    fires['fire_label'] = 1
    labels = fires.groupby(["station_id", "date"])["fire_label"].max().reset_index()

    # weather_daily와 merge
    df = weather.merge(labels, on=["station_id", "date"], how="left")

    # NaN → 0 (산불 없음)
    df["fire_label"] = df["fire_label"].fillna(0).astype(int)

    out = PROC_DIR / "weather_labeled.parquet"
    df.to_parquet(out, index=False)
    print("saved labeled dataset ->", out)
    return df

if __name__ == "__main__":
    build_labels()
