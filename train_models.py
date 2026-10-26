"""
Training script for ML Assignment 2.

Dataset : Predict Students' Dropout and Academic Success (UCI id 697)
Task    : 3-class classification -> Dropout / Enrolled / Graduate

What this script does
1. Loads the raw dataset (downloads it from UCI the first time if not present).
2. Makes a stratified 80/20 hold-out split and writes the hold-out part to
   test_data.csv (features + target) - this is the file uploaded in the app.
3. Tunes and fits five classifiers (LR, DT, KNN, Gaussian NB, SVM) on the
   training part with 5-fold stratified CV. Each model is a full sklearn
   Pipeline (preprocessing + estimator) so the app can feed raw columns.
4. Evaluates every model on the hold-out set (Accuracy, AUC, Precision,
   Recall, F1, MCC) and saves models + metrics + metadata under model/.

Run:  python train_models.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "students_dropout_academic_success.csv"
TEST_CSV = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"

UCI_DATASET_ID = 697
TARGET = "Target"
SEED = 2026
TEST_FRACTION = 0.20

# Integer-coded columns whose codes are labels, not quantities
# (e.g. Course 9254 is not "bigger" than Course 33). These get one-hot encoded.
NOMINAL_COLS = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# File-name friendly keys -> display names used in README / app
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}


def load_raw_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        from ucimlrepo import fetch_ucirepo

        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        fetch_ucirepo(id=UCI_DATASET_ID).data.original.to_csv(RAW_CSV, index=False)
    df = pd.read_csv(RAW_CSV)
    # some mirrors of this dataset carry stray whitespace in header names
    df.columns = [c.strip() for c in df.columns]
    return df


def build_preprocessor(numeric_cols, dense_output=False):
    """One-hot for nominal codes (rare codes pooled), z-scaling for the rest."""
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=not dense_output,
                ),
                NOMINAL_COLS,
            ),
            ("numeric", StandardScaler(), numeric_cols),
        ],
        sparse_threshold=0.0 if dense_output else 0.3,
    )


def candidate_models(numeric_cols):
    """(pipeline, hyper-parameter grid) for each of the five classifiers."""
    pre = lambda: build_preprocessor(numeric_cols)  # noqa: E731
    return {
        "logistic_regression": (
            Pipeline([
                ("prep", pre()),
                ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
            ]),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "decision_tree": (
            Pipeline([
                ("prep", pre()),
                ("clf", DecisionTreeClassifier(random_state=SEED)),
            ]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [1, 5, 20],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "knn": (
            Pipeline([
                ("prep", pre()),
                ("clf", KNeighborsClassifier()),
            ]),
            {
                "clf__n_neighbors": [3, 5, 7, 11, 21, 31],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        # GaussianNB needs a dense matrix
        "naive_bayes": (
            Pipeline([
                ("prep", build_preprocessor(numeric_cols, dense_output=True)),
                ("clf", GaussianNB()),
            ]),
            {"clf__var_smoothing": np.logspace(-9, 0, 10)},
        ),
        # SVC(probability=True) is deprecated since sklearn 1.9; class probabilities
        # (needed for AUC) come from a Platt-calibrated wrapper instead.
        "svm": (
            Pipeline([
                ("prep", pre()),
                ("clf", CalibratedClassifierCV(
                    SVC(kernel="rbf", random_state=SEED), method="sigmoid", ensemble=False)),
            ]),
            {
                "clf__estimator__C": [1.0, 10.0],
                "clf__estimator__gamma": ["scale", 0.01],
                "clf__estimator__class_weight": [None, "balanced"],
            },
        ),
    }


def evaluate(model, X, y) -> dict:
    """The six assignment metrics. Multi-class -> macro averaging, AUC is one-vs-rest."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="macro",
                             labels=model.classes_),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_raw_data()
    feature_cols = [c for c in df.columns if c != TARGET]
    numeric_cols = [c for c in feature_cols if c not in NOMINAL_COLS]
    print(f"Dataset: {df.shape[0]} rows, {len(feature_cols)} features")
    print(df[TARGET].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[TARGET],
        test_size=TEST_FRACTION, stratify=df[TARGET], random_state=SEED,
    )
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Train: {len(X_train)} rows | Test (saved to {TEST_CSV.name}): {len(X_test)} rows\n")

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for key, (pipe, grid) in candidate_models(numeric_cols).items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        joblib.dump(model, MODEL_DIR / f"{key}.joblib", compress=3)

        scores = evaluate(model, X_test, y_test)
        rows.append({"Model": MODEL_NAMES[key], **scores})
        best_params[key] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
        print(f"{MODEL_NAMES[key]:<24} CV F1={search.best_score_:.4f}  "
              f"test F1={scores['F1']:.4f}  ({time.time() - t0:.1f}s)  {best_params[key]}")

    metrics = pd.DataFrame(rows).set_index("Model").round(4)
    metrics.to_csv(MODEL_DIR / "test_metrics.csv")

    metadata = {
        "dataset": "Predict Students' Dropout and Academic Success (UCI id 697)",
        "target": TARGET,
        "classes": sorted(df[TARGET].unique().tolist()),
        "feature_columns": feature_cols,
        "nominal_columns": NOMINAL_COLS,
        "models": MODEL_NAMES,
        "best_params": best_params,
        "split": {"test_fraction": TEST_FRACTION, "random_state": SEED,
                  "train_rows": len(X_train), "test_rows": len(X_test)},
        "averaging": "macro (Precision/Recall/F1), one-vs-rest macro (AUC)",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    print("\nHold-out test metrics")
    print(metrics.to_string())


if __name__ == "__main__":
    main()
