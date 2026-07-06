
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=60)

    st.title("RetailPulse")

    st.caption("AI-Powered Customer Analytics\n& Demand Forecasting Platform")

    st.divider()

    st.markdown("### Navigation")

    
    st.page_link("pages/1_Demand_Forecasting.py", label="📈 Demand Forecasting")
    st.page_link("pages/2_Customer_Segmentation.py", label="👥 Customer Segmentation")
    st.page_link("pages/3_Inventory_Optimization.py", label="📦 Inventory Optimization")
    st.page_link("pages/4_Metrics_and_Alerts.py", label="📊 Metrics & Alerts")
    st.page_link("pages/5_Export_Reports.py", label="📄 Export Reports")
    st.divider()

    st.caption("Zidio Development · June 2026")

# ── Hero Section ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                padding: 40px 30px; border-radius: 12px; margin-bottom: 30px;'>
        <h1 style='color:#e94560; margin:0;'>📊 RetailPulse</h1>
        <h3 style='color:#a8dadc; margin:8px 0 4px 0;'>
            AI-Powered Customer Analytics & Demand Forecasting Platform
        </h3>
        <p style='color:#ccc; margin:0;'>
            Predictive Demand &nbsp;•&nbsp; Customer Segmentation &nbsp;•&nbsp;
            Churn Analysis &nbsp;•&nbsp; Inventory Optimization
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── KPI Cards Row ───────────────────────────────────────────────────────────────
#import os
#import sys

#BASE_DIR = os.path.abspath(
#    os.path.join(os.path.dirname(__file__), "..", "utils")
#)

#if BASE_DIR not in sys.path:
#    sys.path.insert(0, BASE_DIR)

#from data_loader import load_all_data
import os
import sys

# utils folder ka path
UTILS_DIR = os.path.join(os.path.dirname(__file__), "utils")

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

#from data_loader import load_all_data
from utils.data_loader import load_all_data

data = load_all_data()

print(data)
print(type(data))
print(data["ensemble_forecast"])

col1, col2, col3, col4 = st.columns(4)

with col1:

    ensemble_mape = None

    if data["ensemble_metrics"] is not None:
        row = data["ensemble_metrics"][data["ensemble_metrics"]["Model"] == "Ensemble"]
        if len(row) > 0:
            ensemble_mape = row["MAPE"].values[0]

    st.metric(
        label="📈 Forecast MAPE",
        value=f"{ensemble_mape:.2%}" if ensemble_mape is not None else "N/A",
        delta="Target ≤ 12%",
        delta_color="normal"
    )

with col2:

    auc = None

    if data["tuning_summary"] is not None:
        row = data["tuning_summary"][data["tuning_summary"]["Metric"] == "Tuned AUC-ROC"]
        if len(row) > 0:
            auc = row["Value"].values[0]

    st.metric(
        label="🎯 Churn AUC-ROC",
        value=f"{auc:.3f}" if auc is not None else "N/A",
        delta="Target ≥ 0.88",
        delta_color="normal"
    )

with col3:

    churn_count = None

    if data["churn_predictions"] is not None:
        churn_count = int(data["churn_predictions"]["PredictedChurn_Tuned"].sum())

    total_customers = None

    if data["churn_predictions"] is not None:
        total_customers = len(data["churn_predictions"])

    st.metric(
        label="⚠️ At-Risk Customers",
        value=f"{churn_count:,}" if churn_count is not None else "N/A",
        delta=f"of {total_customers:,} total" if total_customers is not None else ""
    )

with col4:

    reorder_point = None

    if data["inventory_summary"] is not None:
        row = data["inventory_summary"][data["inventory_summary"]["Metric"] == "Reorder Point"]
        if len(row) > 0:
            reorder_point = row["Value"].values[0]

    st.metric(
        label="📦 Reorder Point",
        value=f"{reorder_point:,.0f}" if reorder_point is not None else "N/A",
        delta="Units"
    )

st.divider()

# ── Module Overview Cards ────────────────────────────────────────────────────────
st.markdown("### 🗂️ Platform Modules")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.info(
        "**📈 Demand Forecasting**\n\n"
        "Prophet + LSTM hybrid ensemble model. "
        "30-day ahead demand predictions with what-if scenario analysis."
    )

with m2:
    st.success(
        "**👥 Segmentation & Churn**\n\n"
        "RFM customer segments, churn risk scoring (XGBoost + SHAP), "
        "and at-risk customer identification."
    )

with m3:
    st.warning(
        "**📦 Inventory Optimization**\n\n"
        "Safety stock, reorder point, and recommended order quantity "
        "driven by forecasted demand."
    )

with m4:
    st.error(
        "**🔔 Metrics & Alerts**\n\n"
        "Model performance tracking, data drift monitoring (Evidently AI), "
        "and retraining pipeline status."
    )

st.divider()

# ── Week 2 Targets Summary ───────────────────────────────────────────────────────
st.markdown("### ✅ Week 2 Model Targets")

if data["targets_summary"] is not None:
    st.dataframe(data["targets_summary"], use_container_width=True)
else:
    st.info("Run the Week 2 notebooks to generate target metrics.")

st.divider()

# ── Project Info ─────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this Project"):
    st.markdown(
        """
        **RetailPulse** is a 28-day end-to-end data science project built for the
        Zidio Development internship programme.

        **Dataset:** Online Retail II (UCI Machine Learning Repository)

        **Stack:** Python 3.11 · Pandas · Prophet · PyTorch LSTM · XGBoost ·
        SHAP · Optuna · Evidently AI · Airflow · Streamlit · MLflow

        **Deliverables:**
        - Demand forecasting (MAPE ≤ 12%)
        - Churn prediction (AUC-ROC ≥ 0.88, Precision@Top20% ≥ 0.75)
        - Inventory optimization (25–40% reduction in overstock/understock)
        - MLOps pipeline (drift detection + automated retraining)
        - Interactive Streamlit analytics dashboard

        **Author:** Jainam · Zidio Development · March 2026
        """
    )