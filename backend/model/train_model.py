"""
Flight Delay Prediction - Model Training
Run from backend/ folder:  python model/train_model.py
Reads  : dataset/flight_dataset.csv
Saves  : model/all_models.pkl  +  model/accuracies.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "flight_dataset.csv")
PKL_PATH  = os.path.join(BASE_DIR, "all_models.pkl")
JSON_PATH = os.path.join(BASE_DIR, "accuracies.json")

FEATURE_COLS = ["Airline", "Source", "Destination", "Distance",
                "Departure_Time", "Weather_Condition"]
TARGET_COL   = "Delay_Status"
CAT_COLS     = ["Airline", "Source", "Destination", "Departure_Time", "Weather_Condition"]
NUM_COLS     = ["Distance"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].str.strip()
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ],
        remainder="drop",
    )


def get_models():
    return {
        "KNN": KNeighborsClassifier(
            n_neighbors=9, metric="euclidean"
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-8
        ),
        "SVM": SVC(
            kernel="rbf", C=2.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.5,
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def train():
    print("=" * 52)
    print("  Flight Delay Prediction - Model Training")
    print("=" * 52)

    df = load_data()
    print(f"\nDataset : {len(df)} records")
    for label, cnt in df[TARGET_COL].value_counts().items():
        print(f"  {label}: {cnt} ({cnt/len(df)*100:.1f}%)")

    X = df[FEATURE_COLS]
    y = (df[TARGET_COL] == "Delayed").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    trained_models, metrics, cv_scores = {}, {}, {}

    print("\n" + "-" * 52)
    for name, clf in get_models().items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score   (y_test, y_pred, zero_division=0)
        f1   = f1_score       (y_test, y_pred, zero_division=0)
        cv_s = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy").mean()

        trained_models[name] = pipeline
        metrics[name]   = {"accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
                           "recall":   round(float(rec),  4), "f1_score":  round(float(f1),   4)}
        cv_scores[name] = round(float(cv_s), 4)
        print(f"  Acc={acc*100:.1f}%  Prec={prec*100:.1f}%  "
              f"Rec={rec*100:.1f}%  F1={f1*100:.1f}%  CV={cv_s*100:.1f}%")

    print("-" * 52)
    best_model = max(metrics, key=lambda m: metrics[m]["accuracy"])
    print(f"\nBest Model: {best_model} ({metrics[best_model]['accuracy']*100:.1f}%)")

    # Save single PKL bundle
    joblib.dump({
        "models":          trained_models,
        "feature_columns": FEATURE_COLS,
        "label_map":       {1: "Delayed", 0: "On Time"},
    }, PKL_PATH)
    print(f"Saved: {PKL_PATH}")

    # Save accuracies.json
    with open(JSON_PATH, "w") as f:
        json.dump({
            "accuracies": {k: v["accuracy"]  for k, v in metrics.items()},
            "precision":  {k: v["precision"] for k, v in metrics.items()},
            "recall":     {k: v["recall"]    for k, v in metrics.items()},
            "f1_score":   {k: v["f1_score"]  for k, v in metrics.items()},
            "cv_scores":  cv_scores,
            "best_model": best_model,
        }, f, indent=2)
    print(f"Saved: {JSON_PATH}")
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
