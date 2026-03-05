import joblib
import pandas as pd

from src.config.paths import MODEL_DIR, PROC_DIR


FEATURES = [
    "TA_mean",
    # "TA_dtr",  # temporarily excluded (diurnal temperature range)
    "POP",
    "is_precip",
    "WD_sin",
    "WD_cos",
    "SKY",
]
MODEL_NAME = "base_lr.joblib"


def risk_level_from_prob(p: float) -> str:
    if p <= 0.4:
        return "LOW"
    if p <= 0.6:
        return "MODERATE"
    if p <= 0.8:
        return "HIGH"
    return "EXTREME"


def predict_for_date(target_date: str, save: bool = False):
    model_path = MODEL_DIR / MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run `python -m src.models.evaluate` first."
        )

    model = joblib.load(model_path)

    df = pd.read_parquet(PROC_DIR / "weather_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    target_dt = pd.to_datetime(target_date)

    day_df = df[df["date"].dt.date == target_dt.date()].copy()
    day_df = day_df.dropna(subset=FEATURES)
    if day_df.empty:
        raise ValueError(f"No valid data for date={target_date} in weather_labeled.parquet")

    X = day_df[FEATURES]
    day_df["base_prob"] = model.predict_proba(X)[:, 1]
    day_df["risk_level"] = day_df["base_prob"].apply(risk_level_from_prob)

    cols = []
    for c in ["station_id", "date", "base_prob", "risk_level", "lat", "lon"]:
        if c in day_df.columns:
            cols.append(c)
    if not cols:
        cols = ["date", "base_prob", "risk_level"]

    result = day_df[cols].sort_values(cols[0]).reset_index(drop=True)

    if save:
        out_path = PROC_DIR / f"base_predictions_{target_dt.date()}.parquet"
        result.to_parquet(out_path, index=False)
        print(f"Saved predictions: {out_path}")

    return result


if __name__ == "__main__":
    print(predict_for_date("2021-03-25", save=False).head())
