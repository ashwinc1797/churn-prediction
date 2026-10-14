"""Train, evaluate and save the churn pipeline.

Run from the project root:
    python data/make_dataset.py   # once, creates data/churn.csv
    python train_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, POSITIVE_CLASS
from schema import TARGET
from utils.model_io import save_model

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "churn.csv"
MODEL_DIR = ROOT / "models"


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)  # the imputers/scaler/encoder learn ONLY from train

    classes = list(pipe.classes_)
    pos_idx = classes.index(POSITIVE_CLASS)
    proba = pipe.predict_proba(X_test)[:, pos_idx]
    pred = pipe.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label=POSITIVE_CLASS),
        "recall": recall_score(y_test, pred, pos_label=POSITIVE_CLASS),
        "f1": f1_score(y_test, pred, pos_label=POSITIVE_CLASS),
        "roc_auc": roc_auc_score((y_test == POSITIVE_CLASS).astype(int), proba),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1]
    feature_importance = {str(names[i]): round(float(importances[i]), 4) for i in order}

    # store numeric medians / category modes: used as the baseline for explanations
    baseline = {c: float(X_train[c].median()) for c in NUMERIC_FEATURES}
    baseline.update({c: str(X_train[c].mode()[0]) for c in CATEGORICAL_FEATURES})

    metadata = {
        "model_name": "Customer churn classifier",
        "model_type": type(pipe.named_steps["model"]).__name__,
        "task": "binary_classification",
        "features": FEATURES,
        "target": TARGET,
        "classes": classes,
        "positive_class": POSITIVE_CLASS,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "feature_importance": feature_importance,
        "baseline": baseline,
        "default_threshold": 0.5,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    meta = save_model(pipe, MODEL_DIR / "churn_pipeline.joblib",
                      MODEL_DIR / "churn_metadata.json", metadata)

    # keep held-out predictions so the app can draw ROC / PR curves
    pd.DataFrame({"y_true": y_test.values, "y_proba": proba.round(5)}).to_csv(
        MODEL_DIR / "churn_test_predictions.csv", index=False
    )
    print("Saved model", meta["artifact"], "sha256", meta["sha256"][:12])
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}")


if __name__ == "__main__":
    main()
