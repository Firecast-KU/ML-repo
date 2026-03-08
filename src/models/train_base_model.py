import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from src.config.paths import PROC_DIR


FEATURES = [
    "TA",
    # "TA_dtr",  # temporarily excluded (diurnal temperature range)
    "POP",
    "is_precip",
    "WD_sin",
    "WD_cos",
    "SKY",
]


def run_lr_baseline():
    df = pd.read_parquet(PROC_DIR / "weather_labeled.parquet")
    df["date"] = pd.to_datetime(df["date"])

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + ["fire_label"])

    train_df = df[df["date"].dt.year < 2021]
    test_df = df[df["date"].dt.year >= 2021]

    X_train = train_df[FEATURES]
    y_train = train_df["fire_label"]
    X_test = test_df[FEATURES]
    y_test = test_df["fire_label"]

    model = LogisticRegression(class_weight="balanced", max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    coef = model.coef_[0]
    for name, w in sorted(zip(FEATURES, coef), key=lambda x: abs(x[1]), reverse=True):
        print(f"{name:>9}: {w:+.4f}")
    print("intercept:", model.intercept_[0])

    return model


if __name__ == "__main__":
    run_lr_baseline()
