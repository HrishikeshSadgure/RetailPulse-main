"""
RetailPulse - Page 4: Metrics & Alerts
Day 19: Full real-time metrics, drift monitoring, retraining pipeline status
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_all_data

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metrics & Alerts · RetailPulse",
    page_icon="🔔",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔔 Metrics & Alerts")
    st.caption("Model Performance · Drift · Pipeline")
    st.divider()

    st.markdown("### Alert Thresholds")

    mape_alert     = st.slider("MAPE alert threshold", 0.05, 0.30, 0.12, 0.01,
                                help="Alert if MAPE exceeds this value")
    auc_alert      = st.slider("AUC-ROC alert threshold", 0.70, 0.95, 0.88, 0.01,
                                help="Alert if AUC-ROC falls below this value")
    drift_alert    = st.slider("Drift share alert threshold", 0.10, 0.90, 0.50, 0.05,
                                help="Alert if drift share exceeds this value")

    st.divider()
    #st.page_link("app.py", label="🏠 Back to Home")

# ── Load data ──────────────────────────────────────────────────────────────────
data          = load_all_data()
ensemble_met  = data["ensemble_metrics"]
tuning_df     = data["tuning_summary"]
targets_df    = data["targets_summary"]
drift_cols    = data["drift_columns"]
drift_sum     = data["drift_summary"]
retrain_log   = data["retraining_log"]
churn_met     = data["churn_metrics"]
best_params   = data["best_params"]

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🔔 Metrics & Alerts")
st.caption("Model Performance · Data Drift Monitoring · Retraining Pipeline Status")

# ══════════════════════════════════════════════════════════════════════════════
# Section 0 · Live Alert Banner
# ══════════════════════════════════════════════════════════════════════════════
alerts = []

if ensemble_met is not None:
    row = ensemble_met[ensemble_met["Model"] == "Ensemble"]
    if len(row):
        mape_val = row["MAPE"].values[0]
        if mape_val > mape_alert:
            alerts.append(f"⚠️ Ensemble MAPE ({mape_val:.2%}) exceeds alert threshold ({mape_alert:.0%})")

if tuning_df is not None:
    row = tuning_df[tuning_df["Metric"] == "Tuned AUC-ROC"]
    if len(row):
        auc_val = row["Value"].values[0]
        if auc_val < auc_alert:
            alerts.append(f"⚠️ Churn AUC-ROC ({auc_val:.3f}) is below alert threshold ({auc_alert:.2f})")

if drift_sum is not None:
    row = drift_sum[drift_sum["Metric"] == "Drift Share"]
    if len(row):
        drift_val = float(row["Value"].values[0])
        if drift_val >= drift_alert:
            alerts.append(f"🔴 Data drift share ({drift_val:.0%}) exceeds alert threshold ({drift_alert:.0%}) — retraining recommended")

if alerts:
    for alert in alerts:
        st.error(alert)
else:
    st.success("✅ All metrics within acceptable thresholds. No alerts.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 · Model Performance Targets
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Model Performance vs Targets")

t1, t2, t3 = st.columns(3)

# ── Forecasting KPIs ──
with t1:
    st.markdown("#### 📈 Demand Forecasting")

    if ensemble_met is not None:
        for _, row in ensemble_met.iterrows():
            mape = row["MAPE"]
            rmse = row["RMSE"]
            model = row["Model"]
            is_ensemble = model == "Ensemble"
            delta_txt = ("✅ Target met" if mape <= 0.12 else "❌ Above target") if is_ensemble else None
            delta_col = "normal" if (is_ensemble and mape <= 0.12) else "inverse"
            st.metric(
                f"{model} MAPE",
                f"{mape:.2%}",
                delta=delta_txt,
                delta_color=delta_col if delta_txt else "off"
            )
        st.metric("Ensemble RMSE", f"{ensemble_met[ensemble_met['Model']=='Ensemble']['RMSE'].values[0]:,.0f}")
    else:
        st.info("Run Day 8 notebook.")

# ── Churn KPIs ──
with t2:
    st.markdown("#### 👥 Churn Prediction")

    if tuning_df is not None:
        for label, metric_name, target, compare in [
            ("Tuned AUC-ROC",         "Tuned AUC-ROC",              0.88, "ge"),
            ("Baseline AUC-ROC",      "Baseline AUC-ROC",           0.88, "ge"),
            ("Precision@Top20%",      "Precision@Top20% (Tuned)",   0.75, "ge"),
            ("Best CV AUC (Optuna)",  "Best CV AUC (Optuna)",       0.88, "ge"),
        ]:
            row = tuning_df[tuning_df["Metric"] == metric_name]
            if len(row):
                val = float(row["Value"].values[0])
                met = val >= target if compare == "ge" else val <= target
                st.metric(
                    label, f"{val:.3f}",
                    delta="✅ Target met" if met else "❌ Below target",
                    delta_color="normal" if met else "inverse"
                )
    else:
        st.info("Run Day 11 notebook.")

# ── Overall targets table ──
with t3:
    st.markdown("#### ✅ Week 2 Targets Summary")

    if targets_df is not None:
        def highlight_met(row):
            color = "#ccffcc" if row.get("Met", False) else "#ffcccc"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            targets_df.style.apply(highlight_met, axis=1),
            use_container_width=True,
            hide_index=True,
            height=260
        )
    else:
        st.info("Run Day 14 notebook.")

st.divider()

# ── Radar chart: all targets ──────────────────────────────────────────────────
st.markdown("#### 📡 Model Performance Radar")

radar_metrics = []
radar_actual  = []
radar_target  = []

if ensemble_met is not None:
    row = ensemble_met[ensemble_met["Model"] == "Ensemble"]
    if len(row):
        # invert MAPE so higher=better on radar
        mape_score = max(0, 1 - row["MAPE"].values[0])
        radar_metrics.append("Forecast Accuracy\n(1 - MAPE)")
        radar_actual.append(mape_score)
        radar_target.append(1 - 0.12)

if tuning_df is not None:
    for label, metric_name, target in [
        ("Churn AUC-ROC",       "Tuned AUC-ROC",            0.88),
        ("Precision @Top20%",   "Precision@Top20% (Tuned)",  0.75),
        ("CV AUC (Optuna)",     "Best CV AUC (Optuna)",      0.88),
    ]:
        row = tuning_df[tuning_df["Metric"] == metric_name]
        if len(row):
            radar_metrics.append(label)
            radar_actual.append(float(row["Value"].values[0]))
            radar_target.append(target)

if radar_metrics:
    categories = radar_metrics + [radar_metrics[0]]
    actual_vals = radar_actual + [radar_actual[0]]
    target_vals = radar_target + [radar_target[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=actual_vals, theta=categories,
        fill="toself", name="Actual",
        line=dict(color="#636EFA"), fillcolor="rgba(99,110,250,0.2)"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=target_vals, theta=categories,
        fill="toself", name="Target",
        line=dict(color="#EF553B", dash="dash"),
        fillcolor="rgba(239,85,59,0.05)"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=380,
        legend=dict(orientation="h", y=-0.1),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 · Data Drift Status
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Data Drift Status (Evidently AI)")

if drift_cols is None and drift_sum is None:
    st.warning("⚠️ No drift data found. Run **Day 12** notebook to generate drift reports.")
else:
    d1, d2 = st.columns([3, 2])

    with d1:
        if drift_cols is not None:
            # p-value bar chart
            dc = drift_cols.copy()
            dc["Color"] = dc["drift_detected"].map({True: "#EF553B", False: "#00CC96"})
            dc = dc.sort_values("drift_score_p_value")

            fig_drift = go.Figure()
            fig_drift.add_trace(go.Bar(
                x=dc["drift_score_p_value"],
                y=dc["column"],
                orientation="h",
                marker_color=dc["Color"],
                text=[f"{'DRIFTED' if d else 'OK'}" for d in dc["drift_detected"]],
                textposition="outside"
            ))
            fig_drift.add_vline(
                x=0.05, line_dash="dash",
                line_color="black", line_width=2,
                annotation_text="p=0.05 threshold",
                annotation_position="top right"
            )
            fig_drift.update_layout(
                height=360,
                title="Feature Drift (K-S Test p-values) — red = drifted",
                xaxis_title="p-value (lower = more drift)",
                yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_drift, use_container_width=True)

    with d2:
        if drift_sum is not None:
            st.markdown("#### Drift Summary")
            st.dataframe(drift_sum, use_container_width=True, hide_index=True)

        if drift_cols is not None:
            drifted   = drift_cols["drift_detected"].sum()
            total_col = len(drift_cols)
            share     = drifted / total_col if total_col > 0 else 0

            fig_donut = go.Figure(go.Pie(
                labels=["Drifted", "Stable"],
                values=[drifted, total_col - drifted],
                hole=0.55,
                marker=dict(colors=["#EF553B", "#00CC96"])
            ))
            fig_donut.update_traces(textinfo="percent+label")
            fig_donut.update_layout(
                height=260,
                title=f"{drifted}/{total_col} features drifted ({share:.0%})",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Drift decision
            retrain_row = drift_sum[drift_sum["Metric"] == "Retraining Recommended"] \
                if drift_sum is not None else None
            if retrain_row is not None and len(retrain_row):
                retrain_flag = str(retrain_row["Value"].values[0]).lower() in ["true", "1", "yes"]
                if retrain_flag:
                    st.error("🔴 **Retraining recommended** — drift share exceeds threshold.")
                else:
                    st.success("🟢 **No retraining needed** — drift within acceptable range.")

        # Link to HTML drift report
        st.markdown("---")
        st.markdown("📄 **Full Drift Report (HTML)**")
        st.caption("Open `reports/data_drift_report.html` in your browser for the complete Evidently AI report.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 · Retraining Pipeline Log
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🔄 Retraining Pipeline Log")

if retrain_log is None:
    st.warning("⚠️ No retraining log found. Run **Day 13** notebook to execute the pipeline.")
else:
    log = retrain_log.copy()

    # Parse timestamps
    if "timestamp" in log.columns:
        log["timestamp"] = pd.to_datetime(log["timestamp"])
        log["timestamp"] = log["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Status column
    def pipeline_status(row):
        if not row.get("retrain_triggered", False):
            return "⏭️ Skipped (no drift)"
        if row.get("model_promoted", False):
            return "✅ Retrained & Promoted"
        if row.get("passed_threshold", False):
            return "⚠️ Retrained (not promoted)"
        return "❌ Failed threshold"

    log["Pipeline Status"] = log.apply(pipeline_status, axis=1)

    # Colour rows
    def colour_status(row):
        s = row.get("Pipeline Status", "")
        if "✅" in s:   return ["background-color: #ccffcc"] * len(row)
        if "❌" in s:   return ["background-color: #ffcccc"] * len(row)
        if "⚠️" in s:  return ["background-color: #fff3cc"] * len(row)
        return [""] * len(row)

    display_cols = [c for c in [
        "timestamp", "drift_share", "retrain_triggered",
        "new_model_auc", "passed_threshold", "model_promoted", "Pipeline Status"
    ] if c in log.columns]

    st.dataframe(
        log[display_cols].style.apply(colour_status, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # Timeline chart if multiple runs
    if len(log) > 1 and "new_model_auc" in log.columns:
        log_plot = retrain_log.copy()
        log_plot["timestamp"] = pd.to_datetime(log_plot["timestamp"])
        log_plot = log_plot.dropna(subset=["new_model_auc"])

        if len(log_plot):
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=log_plot["timestamp"],
                y=log_plot["new_model_auc"],
                mode="lines+markers",
                name="Retrained Model AUC",
                line=dict(color="#636EFA", width=2),
                marker=dict(size=8)
            ))
            fig_log.add_hline(
                y=0.88, line_dash="dash",
                line_color="red",
                annotation_text="AUC target (0.88)"
            )
            fig_log.update_layout(
                height=280,
                title="Retrained Model AUC Over Time",
                xaxis_title="Pipeline Run",
                yaxis_title="AUC-ROC",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_log, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 · Pipeline Architecture Overview
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🏗️ MLOps Pipeline Architecture")

col_arch1, col_arch2 = st.columns(2)

with col_arch1:
    st.markdown("""
    #### Retraining Pipeline Flow (Airflow DAG)
    ```
    [Daily Schedule]
          │
    ┌─────▼──────────────┐
    │  check_drift_task   │  Evidently AI
    │  (drift detection)  │  K-S test
    └─────┬──────────────┘
          │
      drift >= 50%?
       ┌──┴──┐
      YES    NO
       │      │
    ┌──▼──┐  ┌▼──────────────┐
    │retrain│  │skip_retrain   │
    │ task  │  │ task          │
    └──┬───┘  └──────┬────────┘
       │              │
    ┌──▼──────────┐   │
    │evaluate_task│   │
    │AUC >= 0.80? │   │
    └──┬──────────┘   │
       │               │
       └─────┬─────────┘
             │
       ┌─────▼────────┐
       │   log_task    │
       │ retraining_   │
       │ log.csv       │
       └──────────────┘
    ```
    """)

with col_arch2:
    st.markdown("""
    #### Technology Stack
    | Layer | Technology |
    |-------|-----------|
    | Language | Python 3.11 |
    | Data Processing | Pandas, NumPy |
    | Forecasting | Prophet + LSTM (PyTorch) |
    | Classification | XGBoost |
    | Explainability | SHAP |
    | Hyperparameter Tuning | Optuna |
    | Experiment Tracking | MLflow |
    | Drift Detection | Evidently AI 0.7 |
    | Orchestration | Apache Airflow |
    | Dashboard | Streamlit |
    | Containerization | Docker |
    """)

    # Best params expander
    with st.expander("⚙️ Best Optuna Hyperparameters"):
        if best_params is not None:
            st.dataframe(best_params, use_container_width=True, hide_index=True)
        else:
            st.info("Run Day 11 notebook.")
