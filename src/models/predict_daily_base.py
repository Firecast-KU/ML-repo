import joblib
import pandas as pd

from src.config.paths import PROC_DIR, MODEL_DIR


FEATURES = ["TA", "TMN", "TMX", "RN", "DTR"]
MODEL_NAME = "base_lr.joblib"


def risk_level_from_prob(p: float) -> str:
    # 확률(0~1)을 위험도 레벨로 매핑
    if p <= 0.4:
        return "LOW"
    if p <= 0.6:
        return "MODERATE"
    if p <= 0.8:
        return "HIGH"
    return "EXTREME"


def predict_for_date(target_date: str, save: bool = False):
    """
    target_date: 'YYYY-MM-DD' 형태 권장
    save=True면 outputs/ 또는 processed/ 등 원하는 곳으로 저장하도록 확장 가능.
    """
    model_path = MODEL_DIR / MODEL_NAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"먼저 `python src/models/train_base_model.py`를 실행해서 모델을 저장하세요."
        )

    model = joblib.load(model_path)

    df = pd.read_parquet(PROC_DIR / "weather_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["DTR"] = df["TMX"] - df["TMN"]

    target_dt = pd.to_datetime(target_date)

    # 날짜가 timestamp로 들어있는 경우까지 고려해서 날짜만 비교
    day_df = df[df["date"].dt.date == target_dt.date()].copy()
    if day_df.empty:
        raise ValueError(f"No data for date={target_date} in weather_labeled.parquet")

    X = day_df[FEATURES]
    day_df["base_prob"] = model.predict_proba(X)[:, 1]
    day_df["risk_level"] = day_df["base_prob"].apply(risk_level_from_prob)

    # 보통 지도 렌더링은 station_id / grid_id / lat lon 등이 필요함
    # 현재 데이터에 있는 컬럼에 맞춰 반환 컬럼을 조정해도 됨
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
        print(f"✅ Saved predictions: {out_path}")

    return result


if __name__ == "__main__":
    # 예: python src/models/predict_daily_base.py
    print(predict_for_date("2021-03-25", save=False).head())
