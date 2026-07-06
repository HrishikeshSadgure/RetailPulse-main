"""
RetailPulse - Page 5: Export Reports
Day 20: CSV and PDF export functionality for all modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_all_data

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Export Reports · RetailPulse",
    page_icon="📥",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📥 Export Reports")
    st.caption("CSV & PDF Report Generation")
    st.divider()

    st.markdown("### Report Options")

    include_forecasting  = st.checkbox("📈 Demand Forecasting",  value=True)
    include_churn        = st.checkbox("👥 Churn Predictions",   value=True)
    include_inventory    = st.checkbox("📦 Inventory Report",    value=True)
    include_drift        = st.checkbox("📊 Drift & MLOps",       value=True)

    st.divider()
    #st.page_link("app.py", label="🏠 Back to Home")

# ── Helpers ────────────────────────────────────────────────────────────────────
def df_to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_pdf_html(data, sections):
    """Build a styled HTML string that prints cleanly as PDF."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset='utf-8'>
    <title>RetailPulse Report</title>
    <style>
      body      {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
      h1        {{ color: #e94560; border-bottom: 3px solid #e94560; padding-bottom: 8px; }}
      h2        {{ color: #0f3460; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 32px; }}
      h3        {{ color: #333; margin-top: 20px; }}
      table     {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
      th        {{ background: #0f3460; color: white; padding: 8px 10px; text-align: left; }}
      td        {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
      tr:nth-child(even) {{ background: #f7f7f7; }}
      .kpi-grid {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
      .kpi      {{ background: #f0f4ff; border-left: 4px solid #0f3460;
                   padding: 12px 18px; border-radius: 6px; min-width: 160px; }}
      .kpi-val  {{ font-size: 22px; font-weight: bold; color: #0f3460; }}
      .kpi-lbl  {{ font-size: 12px; color: #666; }}
      .alert-ok {{ background: #e6ffed; border-left: 4px solid #00CC96;
                   padding: 10px 16px; border-radius: 4px; margin: 8px 0; }}
      .alert-warn {{ background: #fff3cd; border-left: 4px solid #FFA15A;
                     padding: 10px 16px; border-radius: 4px; margin: 8px 0; }}
      .alert-bad  {{ background: #ffe0e0; border-left: 4px solid #EF553B;
                     padding: 10px 16px; border-radius: 4px; margin: 8px 0; }}
      .footer   {{ margin-top: 48px; border-top: 1px solid #ccc;
                   padding-top: 12px; font-size: 11px; color: #999; }}
    </style>
    </head>
    <body>

    <h1>📊 RetailPulse — Analytics Report</h1>
    <p><b>Generated:</b> {now} &nbsp;|&nbsp;
       <b>Project:</b> RetailPulse – AI-Powered Customer Analytics & Demand Forecasting &nbsp;|&nbsp;
       <b>Internship:</b> Zidio Development</p>
    """

    # ── Forecasting section ──────────────────────────────────────────────────
    if "forecasting" in sections and data["ensemble_metrics"] is not None:
        html += "<h2>📈 Demand Forecasting</h2>"
        em = data["ensemble_metrics"]
        ens = em[em["Model"] == "Ensemble"]
        mape = ens["MAPE"].values[0] if len(ens) else None
        rmse = ens["RMSE"].values[0] if len(ens) else None

        html += "<div class='kpi-grid'>"
        if mape is not None:
            status = "✅ Target met" if mape <= 0.12 else "❌ Above target"
            html += f"<div class='kpi'><div class='kpi-val'>{mape:.2%}</div><div class='kpi-lbl'>Ensemble MAPE {status}</div></div>"
        if rmse is not None:
            html += f"<div class='kpi'><div class='kpi-val'>{rmse:,.0f}</div><div class='kpi-lbl'>Ensemble RMSE</div></div>"
        html += "</div>"

        html += "<h3>Model Comparison</h3>"
        html += em.to_html(index=False, float_format=lambda x: f"{x:,.4f}")

        if data["ensemble_future"] is not None:
            ef = data["ensemble_future"][["ds", "prophet_yhat", "lstm_yhat", "ensemble_yhat"]].copy()
            ef.columns = ["Date", "Prophet", "LSTM", "Ensemble"]
            total = ef["Ensemble"].sum()
            peak  = ef.loc[ef["Ensemble"].idxmax(), "Date"]
            html += f"<h3>30-Day Forecast Summary</h3>"
            html += f"<div class='kpi-grid'>"
            html += f"<div class='kpi'><div class='kpi-val'>{total:,.0f}</div><div class='kpi-lbl'>Total 30-day demand</div></div>"
            html += f"<div class='kpi'><div class='kpi-val'>{peak}</div><div class='kpi-lbl'>Peak demand day</div></div>"
            html += f"</div>"
            html += "<h3>30-Day Ensemble Forecast</h3>"
            html += ef.to_html(index=False, float_format=lambda x: f"{x:,.0f}")

    # ── Churn section ────────────────────────────────────────────────────────
    if "churn" in sections and data["churn_predictions"] is not None:
        html += "<h2>👥 Churn Prediction</h2>"
        cp = data["churn_predictions"]
        total    = len(cp)
        at_risk  = int((cp["ChurnProbability_Tuned"] >= 0.5).sum())
        avg_prob = cp["ChurnProbability_Tuned"].mean()

        html += "<div class='kpi-grid'>"
        html += f"<div class='kpi'><div class='kpi-val'>{total:,}</div><div class='kpi-lbl'>Total Customers</div></div>"
        html += f"<div class='kpi'><div class='kpi-val'>{at_risk:,}</div><div class='kpi-lbl'>At-Risk Customers (≥50%)</div></div>"
        html += f"<div class='kpi'><div class='kpi-val'>{at_risk/total:.1%}</div><div class='kpi-lbl'>Churn Rate</div></div>"
        html += f"<div class='kpi'><div class='kpi-val'>{avg_prob:.1%}</div><div class='kpi-lbl'>Avg Churn Probability</div></div>"
        html += "</div>"

        if data["tuning_summary"] is not None:
            html += "<h3>Model Performance (Optuna Tuned)</h3>"
            html += data["tuning_summary"].to_html(index=False)

        html += "<h3>Top 20 At-Risk Customers</h3>"
        cols = [c for c in ["Customer ID","Segment","ChurnProbability_Tuned",
                             "Recency","Frequency","Monetary"] if c in cp.columns]
        top20 = cp.sort_values("ChurnProbability_Tuned", ascending=False).head(20)[cols]
        html += top20.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    # ── Inventory section ────────────────────────────────────────────────────
    if "inventory" in sections:
        html += "<h2>📦 Inventory Optimization</h2>"

        if data["inventory_summary"] is not None:
            inv = data["inventory_summary"]
            for _, row in inv.iterrows():
                if row["Metric"] in ["Safety Stock","Reorder Point","Recommended Order Quantity"]:
                    html += f"<div class='kpi-grid'><div class='kpi'>"
                    html += f"<div class='kpi-val'>{float(row['Value']):,.0f}</div>"
                    html += f"<div class='kpi-lbl'>{row['Metric']}</div></div></div>"

            html += "<h3>Inventory Parameters</h3>"
            html += data["inventory_summary"].to_html(index=False)

        if data["inventory_projection"] is not None:
            proj = data["inventory_projection"]
            if "StockStatus" in proj.columns:
                status_counts = proj["StockStatus"].value_counts().reset_index()
                status_counts.columns = ["Status", "Days"]
                html += "<h3>Stock Status Summary (30-Day Window)</h3>"
                html += status_counts.to_html(index=False)

                understock_days = int((proj["StockStatus"] == "Understock Risk").sum())
                if understock_days > 0:
                    html += f"<div class='alert-bad'>⚠️ Stock falls below reorder point for {understock_days} days in the 30-day window. Place a reorder.</div>"
                else:
                    html += "<div class='alert-ok'>✅ Stock remains above reorder point throughout the 30-day forecast window.</div>"

            html += "<h3>30-Day Stock Projection</h3>"
            disp = proj.copy()
            disp_cols = [c for c in ["ds","ForecastedDemand","ProjectedStock","StockStatus"] if c in disp.columns]
            html += disp[disp_cols].to_html(index=False, float_format=lambda x: f"{x:,.0f}")

    # ── Drift & MLOps section ────────────────────────────────────────────────
    if "drift" in sections:
        html += "<h2>📊 Data Drift & MLOps</h2>"

        if data["drift_summary"] is not None:
            html += "<h3>Drift Monitor Summary</h3>"
            html += data["drift_summary"].to_html(index=False)
            retrain_row = data["drift_summary"][data["drift_summary"]["Metric"] == "Retraining Recommended"]
            if len(retrain_row):
                flag = str(retrain_row["Value"].values[0]).lower() in ["true","1","yes"]
                if flag:
                    html += "<div class='alert-bad'>🔴 Retraining recommended — significant data drift detected.</div>"
                else:
                    html += "<div class='alert-ok'>🟢 No retraining needed — drift within acceptable range.</div>"

        if data["drift_columns"] is not None:
            html += "<h3>Feature Drift Results</h3>"
            html += data["drift_columns"].to_html(index=False)

        if data["retraining_log"] is not None:
            html += "<h3>Retraining Pipeline Log</h3>"
            html += data["retraining_log"].to_html(index=False)

        html += """
        <h3>MLOps Stack</h3>
        <table>
          <tr><th>Layer</th><th>Technology</th></tr>
          <tr><td>Forecasting</td><td>Prophet + LSTM (PyTorch)</td></tr>
          <tr><td>Classification</td><td>XGBoost</td></tr>
          <tr><td>Explainability</td><td>SHAP</td></tr>
          <tr><td>Hyperparameter Tuning</td><td>Optuna</td></tr>
          <tr><td>Drift Detection</td><td>Evidently AI 0.7</td></tr>
          <tr><td>Orchestration</td><td>Apache Airflow (DAG: retraining_dag.py)</td></tr>
          <tr><td>Dashboard</td><td>Streamlit</td></tr>
        </table>
        """

    html += f"""
    <div class='footer'>
      RetailPulse · AI-Powered Customer Analytics & Demand Forecasting Platform ·
      Zidio Development Internship · Generated {now}
    </div>
    </body></html>
    """

    return html


# ── Load data ──────────────────────────────────────────────────────────────────
data = load_all_data()

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("📥 Export Reports")
st.caption("Download CSV datasets or generate a full PDF analytics report")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 · CSV Downloads
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📄 CSV Downloads")
st.caption("Download individual datasets generated across Weeks 1 & 2.")

csv_files = [
    ("📈 Ensemble Forecast Results",    "ensemble_forecast",    "ensemble_forecast_results.csv"),
    ("🔮 Future 30-Day Forecast",       "ensemble_future",      "ensemble_future_30_days.csv"),
    ("📊 Model Metrics Comparison",     "ensemble_metrics",     "ensemble_metrics.csv"),
    ("👥 Churn Predictions (Tuned)",    "churn_predictions",    "churn_predictions_tuned.csv"),
    ("🎯 Churn Metrics",               "churn_metrics",        "churn_metrics.csv"),
    ("⚙️ Optuna Tuning Summary",       "tuning_summary",       "optuna_tuning_summary.csv"),
    ("🔧 Best Hyperparameters",         "best_params",          "optuna_best_params.csv"),
    ("📦 Inventory Projection",         "inventory_projection", "inventory_projection.csv"),
    ("📋 Inventory Summary",            "inventory_summary",    "inventory_summary.csv"),
    ("🔍 Drift Column Results",         "drift_columns",        "drift_column_results.csv"),
    ("📡 Drift Monitor Summary",        "drift_summary",        "drift_monitor_summary.csv"),
    ("🔄 Retraining Pipeline Log",      "retraining_log",       "retraining_log.csv"),
    ("✅ Week 2 Targets Summary",       "targets_summary",      "week2_targets_summary.csv"),
]

# Render in 3 columns
cols = st.columns(3)
for i, (label, data_key, filename) in enumerate(csv_files):
    df = data.get(data_key)
    with cols[i % 3]:
        if df is not None:
            st.download_button(
                label=f"⬇️ {label}",
                data=df_to_csv_bytes(df),
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
                key=f"csv_{data_key}"
            )
        else:
            st.button(
                f"🔒 {label}",
                disabled=True,
                use_container_width=True,
                key=f"csv_disabled_{data_key}",
                help="Run the corresponding notebook to generate this file."
            )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 · PDF Report
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📑 Full Analytics Report (PDF-ready HTML)")
st.caption(
    "Generate a complete report as an HTML file. Open it in any browser and use "
    "**File → Print → Save as PDF** to get a clean, formatted PDF document."
)

sections = []
if include_forecasting:  sections.append("forecasting")
if include_churn:        sections.append("churn")
if include_inventory:    sections.append("inventory")
if include_drift:        sections.append("drift")

if not sections:
    st.warning("Select at least one section from the sidebar to include in the report.")
else:
    preview_col, download_col = st.columns([3, 1])

    with preview_col:
        st.markdown("#### Report will include:")
        section_labels = {
            "forecasting": "📈 Demand Forecasting — model metrics, 30-day forecast table",
            "churn":       "👥 Churn Prediction — KPIs, top 20 at-risk customers, model performance",
            "inventory":   "📦 Inventory Optimization — parameters, stock status, reorder alert",
            "drift":       "📊 Data Drift & MLOps — drift results, retraining log, stack overview"
        }
        for s in sections:
            st.markdown(f"- {section_labels[s]}")

    with download_col:
        st.markdown("#### ")

        html_report = build_pdf_html(data, sections)

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
        report_name = f"RetailPulse_Report_{timestamp}.html"

        st.download_button(
            label="⬇️ Download Report (HTML)",
            data=html_report.encode("utf-8"),
            file_name=report_name,
            mime="text/html",
            use_container_width=True,
            key="pdf_report"
        )

        st.info("💡 Open the downloaded file in Chrome/Edge, then **Ctrl+P → Save as PDF** for a clean PDF.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 · Quick Summary Card
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Project Summary")

s1, s2, s3, s4 = st.columns(4)

available = sum(1 for _, k, _ in csv_files if data.get(k) is not None)
total_csv = len(csv_files)

with s1:
    st.metric("📁 CSV Files Available", f"{available} / {total_csv}")

with s2:
    if data["ensemble_metrics"] is not None:
        ens = data["ensemble_metrics"][data["ensemble_metrics"]["Model"] == "Ensemble"]
        mape = ens["MAPE"].values[0] if len(ens) else None
        st.metric("📈 Ensemble MAPE",
                  f"{mape:.2%}" if mape else "N/A",
                  delta="✅ Target met" if (mape and mape <= 0.12) else "❌ Above target",
                  delta_color="normal" if (mape and mape <= 0.12) else "inverse")
    else:
        st.metric("📈 Ensemble MAPE", "N/A")

with s3:
    if data["tuning_summary"] is not None:
        row = data["tuning_summary"][data["tuning_summary"]["Metric"] == "Tuned AUC-ROC"]
        auc = float(row["Value"].values[0]) if len(row) else None
        st.metric("🎯 Churn AUC-ROC",
                  f"{auc:.3f}" if auc else "N/A",
                  delta="✅ Target met" if (auc and auc >= 0.88) else "❌ Below target",
                  delta_color="normal" if (auc and auc >= 0.88) else "inverse")
    else:
        st.metric("🎯 Churn AUC-ROC", "N/A")

with s4:
    if data["targets_summary"] is not None:
        met = data["targets_summary"]["Met"].sum() if "Met" in data["targets_summary"].columns else 0
        total_t = len(data["targets_summary"])
        st.metric("✅ Targets Met", f"{met} / {total_t}")
    else:
        st.metric("✅ Targets Met", "N/A")

with st.expander("📖 Full Week 2 Targets"):
    if data["targets_summary"] is not None:
        st.dataframe(data["targets_summary"], use_container_width=True, hide_index=True)
    else:
        st.info("Run Day 14 notebook to generate week2_targets_summary.csv.")

