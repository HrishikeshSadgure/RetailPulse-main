"""
RetailPulse - Page 3: Inventory Optimization
Day 18: Full interactive UI - parameters, stock projection, reorder recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_all_data

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Optimization · RetailPulse",
    page_icon="📦",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📦 Inventory Optimization")
    st.caption("Safety Stock · Reorder Point · Demand-Driven")
    st.divider()

    st.markdown("### Adjust Parameters")

    lead_time = st.slider(
        "Lead time (days)",
        min_value=1, max_value=30,
        value=7, step=1,
        help="Days between placing and receiving an order"
    )

    service_level = st.selectbox(
        "Service level",
        ["90% (Z=1.28)", "95% (Z=1.65)", "98% (Z=2.05)", "99% (Z=2.33)"],
        index=1
    )

    z_map = {
        "90% (Z=1.28)": 1.28,
        "95% (Z=1.65)": 1.65,
        "98% (Z=2.05)": 2.05,
        "99% (Z=2.33)": 2.33
    }
    z_score = z_map[service_level]

    current_stock_pct = st.slider(
        "Current stock (× avg daily demand)",
        min_value=5, max_value=60,
        value=15, step=1,
        help="How many days of average demand you currently hold in stock"
    )

    st.divider()
    #st.page_link("app.py", label="🏠 Back to Home")

# ── Load data ──────────────────────────────────────────────────────────────────
data        = load_all_data()
projection  = data["inventory_projection"]
inv_summary = data["inventory_summary"]
daily       = data["daily_sales"]
future      = data["ensemble_future"]

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("📦 Inventory Optimization")
st.caption("Safety Stock · Reorder Point · Demand-Driven Recommendations")

if projection is None and daily is None:
    st.warning(
        "⚠️ No inventory data found. Please run **Day 10** notebook first to generate "
        "`inventory_projection.csv` and `inventory_summary.csv`."
    )
    st.stop()

# ── Recalculate on-the-fly from sidebar params ─────────────────────────────────
if daily is not None:
    daily["Date"] = pd.to_datetime(daily["Date"])
    avg_demand = daily["Sales"].mean()
    std_demand = daily["Sales"].std()
else:
    # fallback from saved summary
    avg_demand = float(inv_summary.loc[inv_summary["Metric"] == "Average Daily Demand", "Value"].values[0]) \
        if inv_summary is not None else 1000
    std_demand = float(inv_summary.loc[inv_summary["Metric"] == "Std Dev Daily Demand", "Value"].values[0]) \
        if inv_summary is not None else 200

current_stock  = avg_demand * current_stock_pct
safety_stock   = z_score * std_demand * np.sqrt(lead_time)
reorder_point  = (avg_demand * lead_time) + safety_stock
overstock_ceil = reorder_point * 2

# Recalculate projection from future forecast
if future is not None:
    future["ds"] = pd.to_datetime(future["ds"])
    proj = future[["ds", "ensemble_yhat"]].copy()
    proj = proj.rename(columns={"ensemble_yhat": "ForecastedDemand"})
    proj["ForecastedDemand"] = proj["ForecastedDemand"].clip(lower=0)

    stock_levels = []
    stock = current_stock
    for d in proj["ForecastedDemand"]:
        stock = stock - d
        stock_levels.append(stock)

    proj["ProjectedStock"] = stock_levels
    proj["BelowReorderPoint"] = proj["ProjectedStock"] < reorder_point

    def classify(s):
        if s < reorder_point:   return "Understock Risk"
        if s > overstock_ceil:  return "Overstock Risk"
        return "Optimal"

    proj["StockStatus"] = proj["ProjectedStock"].apply(classify)
    total_demand_30d = proj["ForecastedDemand"].sum()

elif projection is not None:
    proj = projection.copy()
    proj["ds"] = pd.to_datetime(proj["ds"])
    total_demand_30d = proj["ForecastedDemand"].sum() if "ForecastedDemand" in proj.columns else 0
else:
    proj = None
    total_demand_30d = 0

recommended_order = max(0, total_demand_30d + safety_stock - current_stock)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("🛡️ Safety Stock",
              f"{safety_stock:,.0f}",
              delta=f"Z={z_score} · {lead_time}d lead time")
with k2:
    st.metric("🔁 Reorder Point",
              f"{reorder_point:,.0f}",
              delta="Units")
with k3:
    st.metric("📦 Current Stock",
              f"{current_stock:,.0f}",
              delta=f"{current_stock_pct}× avg daily demand")
with k4:
    st.metric("🛒 Recommended Order Qty",
              f"{recommended_order:,.0f}",
              delta="for next 30 days")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 · Inventory Parameters Summary
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⚙️ Inventory Parameters")

p1, p2, p3 = st.columns(3)

params_data = {
    "Parameter": [
        "Average Daily Demand",
        "Std Dev Daily Demand",
        "Lead Time (days)",
        "Service Level Z-score",
        "Safety Stock",
        "Reorder Point",
        "Overstock Ceiling",
        "Current Stock",
        "Total Forecasted Demand (30d)",
        "Recommended Order Quantity"
    ],
    "Value": [
        f"{avg_demand:,.1f}",
        f"{std_demand:,.1f}",
        str(lead_time),
        str(z_score),
        f"{safety_stock:,.0f}",
        f"{reorder_point:,.0f}",
        f"{overstock_ceil:,.0f}",
        f"{current_stock:,.0f}",
        f"{total_demand_30d:,.0f}",
        f"{recommended_order:,.0f}"
    ]
}

params_df_display = pd.DataFrame(params_data)

with p1:
    st.dataframe(
        params_df_display.iloc[:5],
        use_container_width=True,
        hide_index=True
    )
with p2:
    st.dataframe(
        params_df_display.iloc[5:],
        use_container_width=True,
        hide_index=True
    )
with p3:
    # Gauge for stock health
    stock_pct_of_rop = (current_stock / reorder_point * 100) if reorder_point > 0 else 0

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=stock_pct_of_rop,
        title={"text": "Current Stock vs Reorder Point (%)"},
        delta={"reference": 100},
        gauge={
            "axis": {"range": [0, 200]},
            "bar": {"color": "#636EFA"},
            "steps": [
                {"range": [0, 100],   "color": "#ffcccc"},
                {"range": [100, 150], "color": "#ccffcc"},
                {"range": [150, 200], "color": "#fff0cc"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 100
            }
        }
    ))
    fig_gauge.update_layout(height=280,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 · 30-Day Stock Projection
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📉 30-Day Stock Projection")

if proj is not None:

    fig_proj = go.Figure()

    # Stock line coloured by status
    for status, color in [
        ("Optimal", "#00CC96"),
        ("Understock Risk", "#EF553B"),
        ("Overstock Risk", "#FFA15A")
    ]:
        subset = proj[proj["StockStatus"] == status]
        if len(subset):
            fig_proj.add_trace(go.Scatter(
                x=subset["ds"],
                y=subset["ProjectedStock"],
                mode="markers+lines",
                name=status,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color)
            ))

    # Reference lines
    fig_proj.add_hline(
        y=reorder_point,
        line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"Reorder Point ({reorder_point:,.0f})",
        annotation_position="top left"
    )
    fig_proj.add_hline(
        y=safety_stock,
        line_dash="dot", line_color="orange", line_width=1.5,
        annotation_text=f"Safety Stock ({safety_stock:,.0f})",
        annotation_position="bottom right"
    )
    fig_proj.add_hline(
        y=overstock_ceil,
        line_dash="dash", line_color="#FFA15A", line_width=1.5,
        annotation_text=f"Overstock Ceiling ({overstock_ceil:,.0f})",
        annotation_position="top right"
    )

    # Shade danger zone
    fig_proj.add_hrect(
        y0=0, y1=reorder_point,
        fillcolor="rgba(239,85,59,0.05)",
        line_width=0,
        annotation_text="⚠️ Understock zone",
        annotation_position="bottom right"
    )

    fig_proj.update_layout(
        height=430,
        xaxis_title="Date",
        yaxis_title="Stock Level (units)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig_proj.update_xaxes(showgrid=True, gridcolor="#f0f0f0", tickangle=-30)
    fig_proj.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    st.plotly_chart(fig_proj, use_container_width=True)

    # Reorder trigger date
    below = proj[proj["BelowReorderPoint"]]
    if len(below) > 0:
        trigger_date = below.iloc[0]["ds"]
        days_until   = (trigger_date - proj["ds"].iloc[0]).days + 1
        st.error(
            f"🚨 **Reorder Alert:** Stock falls below the reorder point on "
            f"**{trigger_date.date()}** — in **{days_until} days**. "
            f"Place an order for **{recommended_order:,.0f} units** now."
        )
    else:
        st.success(
            "✅ Stock remains above the reorder point throughout the entire 30-day forecast window."
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 · Stock Status Classification
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ Stock Status Classification")

if proj is not None:

    left3, right3 = st.columns([2, 3])

    with left3:
        status_counts = proj["StockStatus"].value_counts().reset_index()
        status_counts.columns = ["Status", "Days"]

        color_map = {
            "Optimal":         "#00CC96",
            "Understock Risk": "#EF553B",
            "Overstock Risk":  "#FFA15A"
        }

        fig_status = px.pie(
            status_counts,
            names="Status", values="Days",
            color="Status",
            color_discrete_map=color_map,
            hole=0.45,
            title="Days in Each Status (30-day window)"
        )
        fig_status.update_traces(textinfo="percent+label+value")
        fig_status.update_layout(
            height=320,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with right3:
        fig_bar_status = px.bar(
            proj,
            x="ds",
            y="ProjectedStock",
            color="StockStatus",
            color_discrete_map=color_map,
            title="Daily Stock Status Over 30-Day Forecast",
            labels={"ds": "Date", "ProjectedStock": "Stock Level", "StockStatus": "Status"}
        )
        fig_bar_status.add_hline(
            y=reorder_point, line_dash="dash",
            line_color="red", line_width=2,
            annotation_text="Reorder Point"
        )
        fig_bar_status.update_layout(
            height=320,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig_bar_status, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 · Reorder Recommendations
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🛒 Reorder Recommendations")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("#### Order Recommendation Card")

    status_color = "🔴" if recommended_order > 0 else "🟢"

    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Current Stock | {current_stock:,.0f} units |
    | Safety Stock Required | {safety_stock:,.0f} units |
    | Reorder Point | {reorder_point:,.0f} units |
    | 30-Day Forecasted Demand | {total_demand_30d:,.0f} units |
    | **Recommended Order Qty** | **{recommended_order:,.0f} units** |
    | Lead Time | {lead_time} days |
    | Service Level | {service_level} |
    | Status | {status_color} {"Order required" if recommended_order > 0 else "No order needed"} |
    """)

with rec_col2:
    st.markdown("#### Forecasted Demand vs Stock")

    if proj is not None and "ForecastedDemand" in proj.columns:
        fig_demand = go.Figure()

        fig_demand.add_trace(go.Bar(
            x=proj["ds"],
            y=proj["ForecastedDemand"],
            name="Daily Forecasted Demand",
            marker_color="#636EFA",
            opacity=0.7
        ))

        fig_demand.add_hline(
            y=avg_demand,
            line_dash="dot", line_color="gray",
            annotation_text=f"Avg demand ({avg_demand:,.0f})",
            annotation_position="top right"
        )

        fig_demand.update_layout(
            height=280,
            xaxis_title="Date",
            yaxis_title="Units",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig_demand, use_container_width=True)

st.divider()

# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("📋 View raw 30-day inventory projection"):
    if proj is not None:
        disp = proj.copy()
        disp["ds"] = disp["ds"].dt.strftime("%Y-%m-%d")
        disp = disp.rename(columns={
            "ds": "Date",
            "ForecastedDemand": "Forecasted Demand",
            "ProjectedStock": "Projected Stock",
            "BelowReorderPoint": "Below ROP",
            "StockStatus": "Status"
        })
        fmt_cols = {"Forecasted Demand": "{:,.0f}", "Projected Stock": "{:,.0f}"}
        st.dataframe(
            disp[[c for c in ["Date","Forecasted Demand","Projected Stock","Below ROP","Status"]
                  if c in disp.columns]]
            .style.format({k: v for k, v in fmt_cols.items() if k in disp.columns}),
            use_container_width=True,
            hide_index=True
        )
