"""
RetailPulse - Centralised data loader.
All Streamlit pages import from here so paths are managed in one place.

Project root: C:/Users/khush/RetailPulse/RetailPulse/
Structure:
    RetailPulse/
    ├── data/
    ├── retailpulse_dashboard/
    │   ├── app.py
    │   └── utils/
    │       └── data_loader.py   <- this file
"""

import pandas as pd
import os

# ── Path resolution ─────────────────────────────────────────────────────────────
# utils/ -> retailpulse_dashboard/ -> RetailPulse/ (project root) -> data/
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../utils
_DASHBOARD_DIR = os.path.dirname(_UTILS_DIR)             # .../retailpulse_dashboard
_ROOT_DIR = os.path.dirname(_DASHBOARD_DIR)              # .../RetailPulse
DATA_DIR = os.path.join(_ROOT_DIR, "data")

def _safe_read(filename):
    """Load a CSV from the data directory; return None if the file doesn't exist."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_all_data():
    """
    Return a dict of all Week 1 + Week 2 data artefacts.
    Any missing file returns None so pages can handle it gracefully.
    """
    return {
        # Week 1
        "cleaned_retail":       _safe_read("cleaned_retail.csv"),
        "rfm_segmented":        _safe_read("rfm_segmented.csv"),
        "daily_sales":          _safe_read("daily_sales.csv"),
        "forecast_results":     _safe_read("forecast_results.csv"),
        "lstm_predictions":     _safe_read("lstm_predictions.csv"),

        # Week 2: Forecasting
        "ensemble_forecast":    _safe_read("ensemble_forecast_results.csv"),
        "ensemble_future":      _safe_read("ensemble_future_30_days.csv"),
        "ensemble_metrics":     _safe_read("ensemble_metrics.csv"),

        # Week 2: Churn
        "churn_predictions":    _safe_read("churn_predictions_tuned.csv"),
        "churn_metrics":        _safe_read("churn_metrics.csv"),
        "tuning_summary":       _safe_read("optuna_tuning_summary.csv"),
        "best_params":          _safe_read("optuna_best_params.csv"),

        # Week 2: Inventory
        "inventory_projection": _safe_read("inventory_projection.csv"),
        "inventory_summary":    _safe_read("inventory_summary.csv"),

        # Week 2: MLOps
        "drift_columns":        _safe_read("drift_column_results.csv"),
        "drift_summary":        _safe_read("drift_monitor_summary.csv"),
        "retraining_log":       _safe_read("retraining_log.csv"),

        # Targets
        "targets_summary":      _safe_read("week2_targets_summary.csv"),
    }
