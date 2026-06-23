"""
Model training, hyperparameter tuning, and evaluation.

Models:
    - Logistic Regression   (GridSearchCV tuned)
    - Random Forest         (RandomizedSearchCV tuned) ← best model
    - SVC, KNN, Decision Tree  (baseline comparison)

Usage
-----
    python3 src/train.py              # train all models, save best
    python3 src/train.py --model rf   # train Random Forest only
    python3 src/train.py --model lr   # train Logistic Regression only
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import linear_model, neighbors, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from src.config import (
    LR_MODEL_PATH,
    LR_PARAMS,
    RANDOM_STATE,
    RF_MODEL_PATH,
    RF_PARAMS,
)
from src.eda import plot_roc_curve
from src.preprocess import load_and_preprocess


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, label: str) -> dict:
    """Print classification report, confusion matrix, and ROC-AUC."""
    y_pred = model.predict(X_test)
    acc    = model.score(X_test, y_test)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) \
             if hasattr(model, "predict_proba") else None

    print(f"\n── {label} ─────────────────────────────────")
    print(f"  Accuracy : {acc * 100:.2f}%")
    if auc:
        print(f"  ROC-AUC  : {auc:.4f}")
    print("\n  Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n  Classification Report:\n", classification_report(y_test, y_pred))

    return {"model": label, "accuracy": round(acc * 100, 2), "roc_auc": round(auc, 4) if auc else None}


# ── Baseline models ───────────────────────────────────────────────────────────

def run_baseline_comparison(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train 5 models with default params, compare cross-val accuracy."""
    models = {
        "Logistic Regression": linear_model.LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVC":                 svm.SVC(probability=True, random_state=RANDOM_STATE),
        "KNN":                 neighbors.KNeighborsClassifier(),
        "Decision Tree":       tree.DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=75, random_state=RANDOM_STATE),
    }
    results = []
    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy").mean()
        model.fit(X_train, y_train)
        r = evaluate(model, X_test, y_test, name)
        r["cv_accuracy"] = round(cv_acc * 100, 2)
        results.append(r)

    df = pd.DataFrame(results).set_index("model")
    print("\n── Baseline Comparison ──────────────────────")
    print(df.to_string())
    return df


# ── Tuned models ──────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, X_test, y_train, y_test, tune: bool = True):
    """Train Logistic Regression, optionally with GridSearchCV."""
    if tune:
        param_grid = {"C": np.logspace(-4, 4, 50), "penalty": ["l1", "l2"],
                      "solver": ["saga"], "max_iter": [1000]}
        clf = GridSearchCV(
            linear_model.LogisticRegression(random_state=RANDOM_STATE),
            param_grid, cv=5, n_jobs=-1, verbose=0,
        )
        clf.fit(X_train, y_train)
        print(f"Best LR params: {clf.best_params_}")
        model = clf.best_estimator_
    else:
        model = linear_model.LogisticRegression(**LR_PARAMS)
        model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, "Logistic Regression (tuned)")
    os.makedirs(os.path.dirname(LR_MODEL_PATH), exist_ok=True)
    joblib.dump(model, LR_MODEL_PATH)
    print(f"Saved → {LR_MODEL_PATH}")
    return model


def train_random_forest(X_train, X_test, y_train, y_test, tune: bool = True):
    """Train Random Forest, optionally with RandomizedSearchCV."""
    if tune:
        param_dist = {
            "n_estimators":      [5, 20, 50, 100],
            "max_features":      ["sqrt", "log2"],
            "max_depth":         [int(x) for x in np.linspace(10, 120, num=12)],
            "min_samples_split": [2, 6, 10],
            "min_samples_leaf":  [1, 3, 4],
            "bootstrap":         [True, False],
        }
        rf_random = RandomizedSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_distributions=param_dist,
            n_iter=100, cv=5, random_state=35, n_jobs=-1, verbose=1,
        )
        rf_random.fit(X_train, y_train)
        print(f"Best RF params: {rf_random.best_params_}")
        model = rf_random.best_estimator_
    else:
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, "Random Forest (tuned)")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc = roc_auc_score(y_test, model.predict(X_test))
    plot_roc_curve(fpr, tpr, auc, "Random Forest",
                   save_path="reports/figures/roc_curve_rf.png")

    os.makedirs(os.path.dirname(RF_MODEL_PATH), exist_ok=True)
    joblib.dump(model, RF_MODEL_PATH)
    print(f"Saved → {RF_MODEL_PATH}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_choice: str = "all", tune: bool = True) -> None:
    X_train, X_test, y_train, y_test, _ = load_and_preprocess(save_scaler=True)

    if model_choice == "all":
        baseline_results = run_baseline_comparison(X_train, X_test, y_train, y_test)
        baseline_results.to_csv("reports/baseline_comparison.csv")
        print("Saved → reports/baseline_comparison.csv")

    if model_choice in ("all", "rf"):
        train_random_forest(X_train, X_test, y_train, y_test, tune=tune)

    if model_choice in ("all", "lr"):
        train_logistic_regression(X_train, X_test, y_train, y_test, tune=tune)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train prospective customer prediction models.")
    parser.add_argument("--model", default="all", choices=["all", "rf", "lr"],
                        help="Model to train (default: all)")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning and use preset params from config.py")
    args = parser.parse_args()
    main(model_choice=args.model, tune=not args.no_tune)
