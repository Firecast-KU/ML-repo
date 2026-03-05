import pandas as pd

from src.config.paths import PROC_DIR


def build_labels():
    fires = pd.read_parquet(PROC_DIR / "fire_events.parquet")
    weather = pd.read_parquet(PROC_DIR / "weather_daily.parquet")

    fires = fires.rename(columns={"fire_date": "date"})
    fires["date"] = pd.to_datetime(fires["date"]).dt.normalize()

    fires["fire_label"] = 1
    labels = fires.groupby(["station_id", "date"], as_index=False)["fire_label"].max()

    df = weather.merge(labels, on=["station_id", "date"], how="left")
    df["fire_label"] = df["fire_label"].fillna(0).astype(int)

    out = PROC_DIR / "weather_labeled.parquet"
    df.to_parquet(out, index=False)
    print("saved labeled dataset ->", out)
    return df


if __name__ == "__main__":
    build_labels()
