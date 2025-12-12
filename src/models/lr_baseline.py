import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from src.config.paths import PROC_DIR

def run_lr_baseline():

    df = pd.read_parquet(PROC_DIR / "weather_labeled.parquet")

    # Feature & Label 선택
    feature_cols = ["TA", "TMN", "TMX", "RN"]
    X = df[feature_cols]
    y = df["fire_label"]

    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression + 클래스 불균형 보정
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=500
    )
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    return model

if __name__ == "__main__":
    run_lr_baseline()