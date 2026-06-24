"""
Experiment tracking for model selection: baseline Logistic Regression vs.
tuned Random Forest vs. tuned XGBoost, all logged to MLflow so we can
compare runs before picking what goes into the DVC train stage.

This is deliberately separate from src/train.py -- that script trains the
one model DVC ships, this script is the "notebook-ish" exploration that
justified picking it. Run once local prepare.py has produced
data/processed/{train,test}.csv.

Usage: python src/train_mlflow.py
"""
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("student-dropout-classification")

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]


def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    X_train, y_train = train_df.drop(columns=["Target"]), train_df["Target"]
    X_test, y_test = test_df.drop(columns=["Target"]), test_df["Target"]
    return X_train, y_train, X_test, y_test


def log_run(run_name, model, params, X_train, y_train, X_test, y_test, flavor="sklearn"):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", macro_f1)

        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print(f"[{run_name}] accuracy={acc:.4f} macro_f1={macro_f1:.4f}")
        return acc, macro_f1


def main():
    X_train, y_train, X_test, y_test = load_data()

    # --- baseline: logistic regression -------------------------------
    # LR needs scaled features to converge in a reasonable number of
    # iterations, tree models below don't care so we only scale here.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline_params = {"max_iter": 1000, "C": 1.0, "class_weight": "balanced"}
    log_run(
        "baseline_logreg",
        LogisticRegression(**baseline_params),
        baseline_params,
        X_train_scaled, y_train, X_test_scaled, y_test,
    )

    # --- experiment 2: random forest, small grid ----------------------
    rf_grid = [
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 400, "max_depth": 16},
        {"n_estimators": 600, "max_depth": None},
    ]
    for cfg in rf_grid:
        params = {**cfg, "random_state": 42, "class_weight": "balanced", "n_jobs": -1}
        log_run(
            f"rf_n{cfg['n_estimators']}_d{cfg['max_depth']}",
            RandomForestClassifier(**params),
            params,
            X_train, y_train, X_test, y_test,
        )

    # --- experiment 3: xgboost, small grid -----------------------------
    xgb_grid = [
        {"learning_rate": 0.1, "max_depth": 4},
        {"learning_rate": 0.05, "max_depth": 6},
        {"learning_rate": 0.05, "max_depth": 8},
    ]
    for cfg in xgb_grid:
        params = {
            **cfg,
            "n_estimators": 300,
            "random_state": 42,
            "objective": "multi:softmax",
            "num_class": 3,
            "eval_metric": "mlogloss",
        }
        log_run(
            f"xgb_lr{cfg['learning_rate']}_d{cfg['max_depth']}",
            XGBClassifier(**params),
            params,
            X_train, y_train, X_test, y_test,
            flavor="xgboost",
        )


if __name__ == "__main__":
    main()
