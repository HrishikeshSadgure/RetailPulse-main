"""
RetailPulse retraining pipeline task functions.
Used by retraining_dag.py (Airflow) and by Day13 notebook for local testing.
"""

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "UniqueProducts", "TotalQuantity", "AvgCustomerFrequency"
]
TARGET_COL = "Churn"

DATA_PATH = "data/churn_predictions_tuned.csv"
MODEL_PATH = "models/xgb_churn_model_tuned.json"
RETRAIN_MODEL_PATH = "models/xgb_churn_model_retrained.json"
DRIFT_LOG_PATH = "data/retraining_log.csv"

RETRAIN_THRESHOLD = 0.5
MIN_AUC_THRESHOLD = 0.80


def check_drift(data_path=DATA_PATH, retrain_threshold=RETRAIN_THRESHOLD, seed=42):

    df = pd.read_csv(data_path)

    model_cols = FEATURE_COLS + ["ChurnProbability_Tuned"]

    features = df[model_cols].fillna(0).reset_index(drop=True)

    split_point = len(features) // 2

    reference = features.iloc[:split_point].copy()
    current = features.iloc[split_point:].copy()

    rng = np.random.RandomState(seed)

    current["Recency"] = current["Recency"] * rng.uniform(1.15, 1.35, len(current))
    current["Monetary"] = current["Monetary"] * rng.uniform(0.75, 0.95, len(current))
    current["AvgOrderValue"] = current["AvgOrderValue"] * rng.uniform(0.8, 1.0, len(current))

    dd = DataDefinition()

    ref_ds = Dataset.from_pandas(reference, data_definition=dd)
    cur_ds = Dataset.from_pandas(current, data_definition=dd)

    report = Report(metrics=[DataDriftPreset()])

    result = report.run(cur_ds, ref_ds)

    rd = result.dict()

    drifted_count_metric = next(
        m for m in rd["metrics"]
        if "DriftedColumnsCount" in m["metric_name"]
    )

    drift_share = drifted_count_metric["value"]["share"]

    return {
        "drift_share": drift_share,
        "retrain_needed": drift_share >= retrain_threshold
    }


def retrain_model(data_path=DATA_PATH, model_out_path=RETRAIN_MODEL_PATH):

    df = pd.read_csv(data_path)

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42
    )

    model.fit(X_train_res, y_train_res)

    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    model.save_model(model_out_path)

    return {"auc": auc, "model_path": model_out_path}


def evaluate_and_promote(retrain_result, min_auc=MIN_AUC_THRESHOLD, promote=True):

    auc = retrain_result["auc"]
    passed = auc >= min_auc
    promoted = False

    if passed and promote:
        shutil.copy(retrain_result["model_path"], MODEL_PATH)
        promoted = True

    return {
        "auc": auc,
        "min_auc_threshold": min_auc,
        "passed_threshold": passed,
        "promoted": promoted
    }


def log_pipeline_run(drift_info, retrain_info=None, eval_info=None, log_path=DRIFT_LOG_PATH):

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "drift_share": drift_info["drift_share"],
        "retrain_triggered": drift_info["retrain_needed"],
        "new_model_auc": retrain_info["auc"] if retrain_info else None,
        "min_auc_threshold": eval_info["min_auc_threshold"] if eval_info else None,
        "passed_threshold": eval_info["passed_threshold"] if eval_info else None,
        "model_promoted": eval_info["promoted"] if eval_info else None
    }

    log_df = pd.DataFrame([log_entry])

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)

    log_df.to_csv(log_path, index=False)

    return log_entry
