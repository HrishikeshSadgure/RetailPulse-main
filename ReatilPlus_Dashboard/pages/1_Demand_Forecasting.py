"""
RetailPulse - Page 1: Demand Forecasting
Day 16: Full interactive charts + what-if analysis
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
    page_title="Demand Forecasting · RetailPulse",
    page_icon="📈",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Demand Forecasting")
    st.caption("Prophet + LSTM Hybrid Ensemble")
    st.divider()

    st.markdown("### Filters")

    date_range = st.selectbox(
        "Historical date range to display",
        ["Last 90 days", "Last 180 days", "All data"],
        index=1
    )

    show_confidence = st.checkbox("Show Prophet confidence interval", value=True)

    st.divider()
    #st.page_link("app.py", label="🏠 Back to Home")

# ── Load data ──────────────────────────────────────────────────────────────────
data = load_all_data()

ensemble   = data["ensemble_forecast"]
future     = data["ensemble_future"]
metrics_df = data["ensemble_metrics"]
daily      = data["daily_sales"]
prophet_fc = data["forecast_results"]

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("📈 Demand Forecasting")
st.caption("Prophet + LSTM Hybrid Ensemble · 30-Day Ahead Predictions")

# ── Guard: no data ─────────────────────────────────────────────────────────────
if ensemble is None or future is None:
    st.warning(
        "⚠️ Forecast data not found. Please run **Day 8** notebook first to generate "
        "`ensemble_forecast_results.csv` and `ensemble_future_30_days.csv`."
    )
    st.stop()

# ── Parse dates ───────────────────────────────────────────────────────────────
ensemble["ds"] = pd.to_datetime(ensemble["ds"])
future["ds"]   = pd.to_datetime(future["ds"])

if prophet_fc is not None:
    prophet_fc["ds"] = pd.to_datetime(prophet_fc["ds"])

if daily is not None:
    daily["Date"] = pd.to_datetime(daily["Date"])

# ── Date filter ───────────────────────────────────────────────────────────────
valid = ensemble.dropna(subset=["actual"]).copy()

if date_range == "Last 90 days":
    cutoff = valid["ds"].max() - pd.Timedelta(days=90)
    valid = valid[valid["ds"] >= cutoff]
elif date_range == "Last 180 days":
    cutoff = valid["ds"].max() - pd.Timedelta(days=180)
    valid = valid[valid["ds"] >= cutoff]

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

if metrics_df is not None:
    def _mape(model):
        row = metrics_df[metrics_df["Model"] == model]
        return row["MAPE"].values[0] if len(row) else None

    def _rmse(model):
        row = metrics_df[metrics_df["Model"] == model]
        return row["RMSE"].values[0] if len(row) else None

    ens_mape  = _mape("Ensemble")
    prop_mape = _mape("Prophet")
    lstm_mape = _mape("LSTM")
    ens_rmse  = _rmse("Ensemble")

    target_met = ens_mape is not None and ens_mape <= 0.12

    with c1:
        st.metric("📉 Ensemble MAPE",
                  f"{ens_mape:.2%}" if ens_mape else "N/A",
                  delta="✅ Target met" if target_met else "❌ Above target",
                  delta_color="normal" if target_met else "inverse")
    with c2:
        st.metric("🔵 Prophet MAPE",
                  f"{prop_mape:.2%}" if prop_mape else "N/A")
    with c3:
        st.metric("🟠 LSTM MAPE",
                  f"{lstm_mape:.2%}" if lstm_mape else "N/A")
    with c4:
        st.metric("📐 Ensemble RMSE",
                  f"{ens_rmse:,.0f}" if ens_rmse else "N/A")
else:
    st.info("Run Day 8 notebook to see model metrics.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 · Historical vs Ensemble Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Historical vs Ensemble Forecast")

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=valid["ds"], y=valid["actual"],
    name="Actual Sales",
    line=dict(color="#636EFA", width=2)
))

fig1.add_trace(go.Scatter(
    x=valid["ds"], y=valid["ensemble_yhat"],
    name="Ensemble Forecast",
    line=dict(color="#EF553B", width=2, dash="dot")
))

if show_confidence and prophet_fc is not None and "yhat_lower" in prophet_fc.columns:
    pf = prophet_fc[prophet_fc["ds"].isin(valid["ds"])]
    fig1.add_trace(go.Scatter(
        x=pd.concat([pf["ds"], pf["ds"][::-1]]),
        y=pd.concat([pf["yhat_upper"], pf["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(99,110,250,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Prophet 95% CI",
        showlegend=True
    ))

fig1.update_layout(
    height=420,
    xaxis_title="Date",
    yaxis_title="Daily Sales",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
fig1.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig1.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

st.plotly_chart(fig1, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 · 30-Day Future Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🔮 30-Day Future Demand Forecast")

fig2 = go.Figure()

# Last 30 days of actuals for context
if len(valid) >= 30:
    tail = valid.tail(30)
    fig2.add_trace(go.Scatter(
        x=tail["ds"], y=tail["actual"],
        name="Recent Actual (context)",
        line=dict(color="#636EFA", width=2)
    ))

fig2.add_trace(go.Scatter(
    x=future["ds"], y=future["prophet_yhat"],
    name="Prophet",
    line=dict(color="#00CC96", width=1.5, dash="dash"),
    opacity=0.7
))

fig2.add_trace(go.Scatter(
    x=future["ds"], y=future["lstm_yhat"],
    name="LSTM",
    line=dict(color="#FFA15A", width=1.5, dash="dash"),
    opacity=0.7
))

fig2.add_trace(go.Scatter(
    x=future["ds"], y=future["ensemble_yhat"],
    name="Ensemble (final)",
    line=dict(color="#EF553B", width=3)
))

# Shade the forecast region
fig2.add_vrect(
    x0=str(future["ds"].min()),
    x1=str(future["ds"].max()),
    fillcolor="rgba(239,85,59,0.05)",
    line_width=0,
    annotation_text="Forecast window",
    annotation_position="top left"
)

fig2.update_layout(
    height=420,
    xaxis_title="Date",
    yaxis_title="Predicted Daily Sales",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
fig2.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig2.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

st.plotly_chart(fig2, use_container_width=True)

col_l, col_r = st.columns(2)
with col_l:
    st.metric("📦 Total Forecast (30 days)",
              f"{future['ensemble_yhat'].sum():,.0f} units")
with col_r:
    st.metric("📅 Peak Forecast Day",
              str(future.loc[future['ensemble_yhat'].idxmax(), 'ds'].date()))

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 · Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🔬 Model Comparison: Prophet vs LSTM vs Ensemble")

left, right = st.columns([3, 2])

with left:
    if "prophet_yhat" in valid.columns and "lstm_yhat" in valid.columns:

        fig3 = go.Figure()

        sample = valid.tail(60) if len(valid) > 60 else valid

        fig3.add_trace(go.Scatter(
            x=sample["ds"], y=sample["actual"],
            name="Actual",
            line=dict(color="#636EFA", width=2)
        ))
        fig3.add_trace(go.Scatter(
            x=sample["ds"], y=sample["prophet_yhat"],
            name="Prophet",
            line=dict(color="#00CC96", width=1.5, dash="dash"),
            opacity=0.8
        ))
        fig3.add_trace(go.Scatter(
            x=sample["ds"], y=sample["lstm_yhat"],
            name="LSTM",
            line=dict(color="#FFA15A", width=1.5, dash="dash"),
            opacity=0.8
        ))
        fig3.add_trace(go.Scatter(
            x=sample["ds"], y=sample["ensemble_yhat"],
            name="Ensemble",
            line=dict(color="#EF553B", width=2.5)
        ))

        fig3.update_layout(
            height=360,
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        fig3.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig3.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Individual model predictions not found in ensemble_forecast_results.csv.")

with right:
    if metrics_df is not None:
        st.markdown("#### Model Performance")
        st.dataframe(
            metrics_df.style.format({"MAPE": "{:.2%}", "RMSE": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True
        )

        fig_bar = px.bar(
            metrics_df,
            x="Model", y="MAPE",
            color="Model",
            color_discrete_map={"Prophet": "#00CC96", "LSTM": "#FFA15A", "Ensemble": "#EF553B"},
            text_auto=".2%",
            title="MAPE by Model"
        )
        fig_bar.add_hline(y=0.12, line_dash="dot", line_color="red",
                          annotation_text="Target (12%)")
        fig_bar.update_layout(
            height=280,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 · What-If Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🎛️ What-If Analysis")
st.caption("Adjust the demand multiplier to simulate different business scenarios (promotions, seasonality, supply shocks).")

wa_col1, wa_col2 = st.columns([1, 3])

with wa_col1:

    demand_multiplier = st.slider(
        "Demand multiplier",
        min_value=0.5, max_value=2.0,
        value=1.0, step=0.05,
        help="1.0 = baseline forecast. 1.2 = +20% demand surge."
    )

    scenario_label = (
        "📈 Surge scenario" if demand_multiplier > 1.1
        else "📉 Downturn scenario" if demand_multiplier < 0.9
        else "📊 Baseline"
    )

    st.markdown(f"**Scenario:** {scenario_label}")

    adjusted_total = future["ensemble_yhat"].sum() * demand_multiplier
    baseline_total = future["ensemble_yhat"].sum()
    delta_units    = adjusted_total - baseline_total

    st.metric(
        "Adjusted 30-day demand",
        f"{adjusted_total:,.0f}",
        delta=f"{delta_units:+,.0f} vs baseline"
    )

    st.metric(
        "Suggested order quantity",
        f"{max(0, adjusted_total * 1.1):,.0f}",
        delta="+10% safety buffer"
    )

with wa_col2:

    adjusted_yhat = future["ensemble_yhat"] * demand_multiplier

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=future["ds"], y=future["ensemble_yhat"],
        name="Baseline Forecast",
        line=dict(color="#636EFA", width=2, dash="dash"),
        fill=None
    ))

    fig4.add_trace(go.Scatter(
        x=future["ds"], y=adjusted_yhat,
        name=f"Adjusted Forecast (×{demand_multiplier:.2f})",
        line=dict(color="#EF553B", width=2.5),
        fill="tonexty",
        fillcolor="rgba(239,85,59,0.08)"
    ))

    fig4.update_layout(
        height=360,
        xaxis_title="Date",
        yaxis_title="Projected Daily Sales",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig4.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig4.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 · Raw forecast data table
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

with st.expander("📋 View raw 30-day forecast data"):
    display_future = future[["ds", "prophet_yhat", "lstm_yhat", "ensemble_yhat"]].copy()
    display_future.columns = ["Date", "Prophet", "LSTM", "Ensemble"]
    display_future["Date"] = display_future["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        display_future.style.format({
            "Prophet": "{:,.0f}",
            "LSTM": "{:,.0f}",
            "Ensemble": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )
