"""
RetailPulse - Page 2: Customer Segmentation & Churn Risk
Day 17: Full interactive charts - RFM segments, churn risk, top at-risk customers, SHAP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_all_data

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentation & Churn · RetailPulse",
    page_icon="👥",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("👥 Segmentation & Churn")
    st.caption("RFM · XGBoost · SHAP")
    st.divider()

    st.markdown("### Filters")

    churn_threshold = st.slider(
        "Churn probability threshold",
        min_value=0.3, max_value=0.9,
        value=0.5, step=0.05,
        help="Customers above this score are flagged as at-risk"
    )

    top_n = st.selectbox(
        "Top N at-risk customers to display",
        [10, 20, 50, 100],
        index=1
    )

    st.divider()
    #st.page_link("app.py", label="🏠 Back to Home")

# ── Load data ──────────────────────────────────────────────────────────────────
data       = load_all_data()
churn_df   = data["churn_predictions"]
rfm_df     = data["rfm_segmented"]
tuning_df  = data["tuning_summary"]
params_df  = data["best_params"]
churn_met  = data["churn_metrics"]

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("👥 Customer Segmentation & Churn Risk")
st.caption("RFM Segments · XGBoost Churn Model · SHAP Explainability")

if churn_df is None and rfm_df is None:
    st.warning(
        "⚠️ No data found. Please run **Day 9** and **Day 11** notebooks first to generate "
        "`churn_predictions_tuned.csv` and `rfm_segmented.csv`."
    )
    st.stop()

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

if churn_df is not None:
    at_risk      = int((churn_df["ChurnProbability_Tuned"] >= churn_threshold).sum())
    total_cust   = len(churn_df)
    churn_rate   = at_risk / total_cust
    avg_prob     = churn_df["ChurnProbability_Tuned"].mean()

    with k1:
        st.metric("⚠️ At-Risk Customers",
                  f"{at_risk:,}",
                  delta=f"{churn_rate:.1%} of total")
    with k2:
        st.metric("👥 Total Customers", f"{total_cust:,}")
    with k3:
        st.metric("📊 Avg Churn Probability", f"{avg_prob:.1%}")
    with k4:
        if tuning_df is not None:
            row = tuning_df[tuning_df["Metric"] == "Tuned AUC-ROC"]
            auc = row["Value"].values[0] if len(row) else None
            target_met = auc is not None and auc >= 0.88
            st.metric("🎯 Model AUC-ROC",
                      f"{auc:.3f}" if auc else "N/A",
                      delta="✅ Target met" if target_met else "❌ Below target",
                      delta_color="normal" if target_met else "inverse")
        else:
            st.metric("🎯 Model AUC-ROC", "N/A")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 · RFM Segment Distribution
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ RFM Customer Segment Distribution")

seg_source = rfm_df if rfm_df is not None else churn_df

if seg_source is not None and "Segment" in seg_source.columns:

    seg_counts = seg_source["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    left, right = st.columns([3, 2])

    with left:
        fig_seg = px.bar(
            seg_counts,
            x="Segment", y="Count",
            color="Segment",
            text="Count",
            title="Customer Count by RFM Segment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(
            height=380, showlegend=False,
            xaxis_title="Segment", yaxis_title="Number of Customers",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        fig_seg.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_seg, use_container_width=True)

    with right:
        fig_pie = px.pie(
            seg_counts,
            names="Segment", values="Count",
            title="Segment Share",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # RFM scatter if columns available
    if rfm_df is not None and all(c in rfm_df.columns for c in ["Recency","Frequency","Monetary"]):
        st.markdown("#### RFM Distribution by Segment")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_rfm1 = px.scatter(
                rfm_df.sample(min(1000, len(rfm_df)), random_state=42),
                x="Recency", y="Frequency",
                color="Segment",
                size="Monetary",
                title="Recency vs Frequency (bubble = Monetary)",
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_rfm1.update_layout(
                height=350,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_rfm1, use_container_width=True)

        with col_b:
            fig_rfm2 = px.box(
                rfm_df,
                x="Segment", y="Monetary",
                color="Segment",
                title="Monetary Value Distribution by Segment",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_rfm2.update_layout(
                height=350, showlegend=False,
                xaxis_title="", yaxis_title="Monetary Value",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            fig_rfm2.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_rfm2, use_container_width=True)

else:
    st.info("Segment column not found. Run Day 3 (Customer Segmentation) notebook.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 · Churn Risk Score Distribution
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📉 Churn Risk Score Distribution")

if churn_df is not None and "ChurnProbability_Tuned" in churn_df.columns:

    left2, right2 = st.columns([3, 2])

    with left2:
        fig_hist = go.Figure()

        active = churn_df[churn_df["ChurnProbability_Tuned"] < churn_threshold]["ChurnProbability_Tuned"]
        at_r   = churn_df[churn_df["ChurnProbability_Tuned"] >= churn_threshold]["ChurnProbability_Tuned"]

        fig_hist.add_trace(go.Histogram(
            x=active, name="Active",
            marker_color="#00CC96", opacity=0.7,
            nbinsx=40
        ))
        fig_hist.add_trace(go.Histogram(
            x=at_r, name="At-Risk",
            marker_color="#EF553B", opacity=0.7,
            nbinsx=40
        ))
        fig_hist.add_vline(
            x=churn_threshold, line_dash="dash",
            line_color="black", line_width=2,
            annotation_text=f"Threshold ({churn_threshold:.2f})",
            annotation_position="top right"
        )
        fig_hist.update_layout(
            barmode="overlay",
            height=360,
            title="Churn Probability Distribution",
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            legend=dict(orientation="h", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with right2:
        # Churn by segment if available
        if "Segment" in churn_df.columns:
            seg_churn = churn_df.groupby("Segment")["ChurnProbability_Tuned"].mean().reset_index()
            seg_churn.columns = ["Segment", "Avg Churn Probability"]
            seg_churn = seg_churn.sort_values("Avg Churn Probability", ascending=True)

            fig_seg_churn = px.bar(
                seg_churn,
                x="Avg Churn Probability", y="Segment",
                orientation="h",
                color="Avg Churn Probability",
                color_continuous_scale="RdYlGn_r",
                title="Avg Churn Risk by Segment",
                text_auto=".2%"
            )
            fig_seg_churn.update_layout(
                height=360, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_seg_churn, use_container_width=True)
        else:
            # Churn label breakdown
            labels = churn_df["PredictedChurn_Tuned"].value_counts().reset_index()
            labels.columns = ["Churn", "Count"]
            labels["Churn"] = labels["Churn"].map({0: "Active", 1: "At-Risk"})
            fig_donut = px.pie(
                labels, names="Churn", values="Count",
                color="Churn",
                color_discrete_map={"Active": "#00CC96", "At-Risk": "#EF553B"},
                hole=0.5, title="Active vs At-Risk Customers"
            )
            fig_donut.update_layout(
                height=360,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_donut, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 · Top At-Risk Customers
# ══════════════════════════════════════════════════════════════════════════════
st.subheader(f"🚨 Top {top_n} At-Risk Customers")

if churn_df is not None and "ChurnProbability_Tuned" in churn_df.columns:

    feature_cols = [
        "Recency", "Frequency", "Monetary",
        "AvgOrderValue", "UniqueProducts",
        "TotalQuantity", "AvgCustomerFrequency"
    ]

    display_cols = ["Customer ID", "ChurnProbability_Tuned", "PredictedChurn_Tuned"] + \
                   [c for c in feature_cols if c in churn_df.columns]

    if "Segment" in churn_df.columns:
        display_cols = ["Customer ID", "Segment", "ChurnProbability_Tuned"] + \
                       [c for c in feature_cols if c in churn_df.columns]

    top_risk = (
        churn_df[churn_df["ChurnProbability_Tuned"] >= churn_threshold]
        .sort_values("ChurnProbability_Tuned", ascending=False)
        .head(top_n)
        [[c for c in display_cols if c in churn_df.columns]]
        .reset_index(drop=True)
    )

    # colour the risk score column
    def colour_risk(val):
        if val >= 0.8:
            return "background-color: #ffcccc"
        elif val >= 0.6:
            return "background-color: #ffe5cc"
        else:
            return "background-color: #fff9cc"

    st.dataframe(
        top_risk.style
            .map(colour_risk, subset=["ChurnProbability_Tuned"])
            .format({"ChurnProbability_Tuned": "{:.1%}"}),
        use_container_width=True,
        height=380
    )

    # Scatter: Recency vs Monetary coloured by churn risk
    if all(c in churn_df.columns for c in ["Recency", "Monetary"]):
        st.markdown("#### Recency vs Monetary — coloured by Churn Risk")
        sample = churn_df.sample(min(1500, len(churn_df)), random_state=42)
        fig_scatter = px.scatter(
            sample,
            x="Recency", y="Monetary",
            color="ChurnProbability_Tuned",
            color_continuous_scale="RdYlGn_r",
            opacity=0.6,
            title="Customer Risk Map (red = high churn probability)",
            labels={"ChurnProbability_Tuned": "Churn Prob"}
        )
        fig_scatter.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 · Model Performance & SHAP Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🔍 Model Performance & Feature Importance")

perf_col, shap_col = st.columns(2)

with perf_col:
    st.markdown("#### Optuna Tuning Summary")

    if tuning_df is not None:
        st.dataframe(tuning_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run Day 11 to generate tuning summary.")

    if churn_met is not None:
        st.markdown("#### Churn Model Metrics (Day 9)")
        st.dataframe(churn_met, use_container_width=True, hide_index=True)

with shap_col:
    st.markdown("#### SHAP Feature Importance (Approximate)")
    st.caption("Based on XGBoost feature importances from the tuned model.")

    if churn_df is not None:
        feature_cols_present = [c for c in [
            "Recency", "Frequency", "Monetary",
            "AvgOrderValue", "UniqueProducts",
            "TotalQuantity", "AvgCustomerFrequency"
        ] if c in churn_df.columns]

        if feature_cols_present:
            # proxy importance: correlation of each feature with churn probability
            importance_scores = {}
            for col in feature_cols_present:
                corr = abs(churn_df[col].corr(churn_df["ChurnProbability_Tuned"]))
                importance_scores[col] = corr if not np.isnan(corr) else 0

            imp_df = pd.DataFrame(
                list(importance_scores.items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                imp_df,
                x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                title="Feature Importance (correlation with churn probability)",
                text_auto=".3f"
            )
            fig_imp.update_layout(
                height=360, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            st.info(
                "💡 For exact SHAP values, run Day 11 notebook — "
                "this chart uses feature-churn correlation as a proxy."
            )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 · Best Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("⚙️ Best Optuna Hyperparameters (Day 11)"):
    if params_df is not None:
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run Day 11 notebook to generate optuna_best_params.csv.")
