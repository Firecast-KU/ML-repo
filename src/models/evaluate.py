import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

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
LABEL = "fire_label"

MODEL_NAME = "base_lr.joblib"
META_NAME = "base_lr_meta.json"


def time_split_holdout(df: pd.DataFrame, holdout_days: int = 90):
    """Time-based holdout split using the latest N days as test."""
    df = df.sort_values(["date"]).reset_index(drop=True)
    cutoff = df["date"].max() - pd.Timedelta(days=holdout_days)
    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    return train_df, test_df, cutoff


def train_and_save(holdout_days: int = 90):
    df = pd.read_parquet(PROC_DIR / "weather_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])

    # Numeric safety for model input.
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [LABEL])

    train_df, test_df, cutoff = time_split_holdout(df, holdout_days=holdout_days)

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL]
    X_test = test_df[FEATURES]
    y_test = test_df[LABEL]

    model = LogisticRegression(class_weight="balanced", max_iter=500)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else None
    pr = average_precision_score(y_test, y_prob) if len(set(y_test)) > 1 else None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_NAME
    joblib.dump(model, model_path)

    meta = {
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(PROC_DIR / "weather_labeled.parquet"),
        "features": FEATURES,
        "label": LABEL,
        "split": {
            "type": "time_holdout",
            "holdout_days": holdout_days,
            "cutoff": str(cutoff.date()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "quick_metrics_on_holdout": {
            "roc_auc": roc,
            "pr_auc": pr,
        },
    }

    meta_path = MODEL_DIR / META_NAME
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Meta saved : {meta_path}")
    print(f"Holdout cutoff date: {cutoff.date()}")
    print(f"Quick ROC-AUC: {roc}")
    print(f"Quick PR-AUC : {pr}")

    return model


if __name__ == "__main__":
    train_and_save(holdout_days=90)
