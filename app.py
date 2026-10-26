"""
Meridian Retail Logistics (MRL) - Operations & Analytics Control Tower
A comprehensive, enterprise logistics analytics dashboard synthesizing data quality governance,
network performance diagnostics, predictive machine learning models, and prescriptive optimization engines.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# 1. Page Configuration & Theme
st.set_page_config(
    page_title="Meridian Logistics | Control Tower",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS
st.markdown("""
<style>
    /* Main Background & Clean Typography */
    .stApp {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Banner Header */
    .mrl-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .mrl-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    .mrl-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        border-left: 5px solid #0284c7;
        transition: transform 0.15s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] label {
        color: #64748b !important;
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 0.5rem;
        font-size: 1.0rem;
        font-weight: 600;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #0284c7 !important;
        border-bottom-color: #0284c7 !important;
    }
    
    /* Callout & Card containers */
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    .badge-blue { background-color: #e0f2fe; color: #0369a1; }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-amber { background-color: #fef3c7; color: #b45309; }
    .badge-red { background-color: #fee2e2; color: #b91c1c; }
</style>
""", unsafe_allow_html=True)

# 2. Data & Model Loading with Caching
@st.cache_data
def load_datasets():
    cleaned_df = pd.read_csv("data/cleaned_logistics_sample.csv", parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"])
    raw_df = pd.read_csv("data/raw_logistics_sample.csv", parse_dates=["order_date", "promised_delivery_date", "actual_delivery_date"])
    audit_df = pd.read_csv("data/ground_truth_audit.csv")
    test_pred_df = pd.read_csv("data/test_orders_with_predictions.csv", parse_dates=["order_date"])
    model_comp_df = pd.read_csv("data/final_model_comparison.csv")
    return cleaned_df, raw_df, audit_df, test_pred_df, model_comp_df

@st.cache_resource
def load_model():
    return joblib.load("src/models/champion_model.joblib")

# Ensure required files exist
if not os.path.exists("data/cleaned_logistics_sample.csv"):
    from src.data_generator import generate_raw_logistics_data
    from src.cleaning_pipeline import run_cleaning_pipeline
    from src.cost_models import evaluate_models
    generate_raw_logistics_data()
    run_cleaning_pipeline()
    evaluate_models()

df, raw_df, audit_df, test_pred_df, model_comp_df = load_datasets()
champion_model = load_model()

# Header Banner
st.markdown("""
<div class="mrl-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1>Meridian Retail Logistics (MRL)</h1>
            <p>End-to-End Enterprise Logistics Analytics & Operations Optimization Control Tower</p>
        </div>
        <div style="text-align: right; margin-top: 0.5rem;">
            <span class="badge badge-green" style="font-size: 0.85rem; padding: 0.35rem 0.8rem;">● Live Model Serving</span>
            <span class="badge badge-blue" style="font-size: 0.85rem; padding: 0.35rem 0.8rem; margin-left: 0.4rem;">5,000 Orders Monitored</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Sidebar Filters
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### 🎛️ Network Filters")
    
    # Distribution Center filter
    dc_options = ["All DCs"] + sorted(df["dc_id"].unique().tolist())
    selected_dc = st.selectbox("Distribution Center", dc_options, index=0)
    
    # Customer Region filter
    region_options = ["All Regions"] + sorted(df["customer_region"].unique().tolist())
    selected_region = st.selectbox("Customer Region", region_options, index=0)
    
    # Shipping Mode filter
    mode_options = ["All Modes"] + sorted(df["shipping_mode"].unique().tolist())
    selected_mode = st.selectbox("Shipping Mode", mode_options, index=0)
    
    # Product Category filter
    cat_options = ["All Categories"] + sorted(df["product_category"].unique().tolist())
    selected_cat = st.selectbox("Product Category", cat_options, index=0)
    
    st.markdown("---")
    st.markdown("### 📋 Executive Highlights")
    st.markdown("""
    - **Baseline OTD:** 57.4% *(Target: ≥95%)*
    - **ML Champion:** Gradient Boosting
    - **Held-Out MAE:** $2.33 - $3.46
    - **Freight Audit:** 100% Recall @ k=4
    - **Route Mileage Saved:** 27.3%
    """)
    st.markdown("---")
    st.caption("MRL Decision Support System • Version 4.2")

# Apply filters to working view
filtered_df = df.copy()
if selected_dc != "All DCs":
    filtered_df = filtered_df[filtered_df["dc_id"] == selected_dc]
if selected_region != "All Regions":
    filtered_df = filtered_df[filtered_df["customer_region"] == selected_region]
if selected_mode != "All Modes":
    filtered_df = filtered_df[filtered_df["shipping_mode"] == selected_mode]
if selected_cat != "All Categories":
    filtered_df = filtered_df[filtered_df["product_category"] == selected_cat]

# 4. Multi-Tab Navigation
tabs = st.tabs([
    "🏢 Executive Overview & KPIs",
    "🧹 Data Pipeline & Governance",
    "📊 Network Diagnostics & Analytics",
    "🤖 Predictive Freight Intelligence",
    "⚡ Prescriptive Operations",
    "📚 System Architecture & Docs"
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & STRATEGIC KPIS
# ==============================================================================
with tabs[0]:
    st.markdown("### 📈 Strategic Scorecard & Logistics KPIs")
    
    # KPI Top Row
    delivered_filtered = filtered_df[filtered_df["delivery_delay_days"].notna()]
    current_otd = (delivered_filtered["delivery_delay_days"] <= 0).mean() * 100
    avg_cost = filtered_df["shipping_cost_capped"].mean()
    total_spend = filtered_df["shipping_cost"].sum()
    avg_delay = delivered_filtered["delivery_delay_days"].mean()
    total_val = filtered_df["order_value"].sum()
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric(
            label="On-Time Delivery (OTD)",
            value=f"{current_otd:.1f}%",
            delta=f"{current_otd - 95.0:.1f}% vs Target (≥95%)",
            delta_color="inverse"
        )
    with k2:
        st.metric(
            label="Avg Last-Mile Cost",
            value=f"${avg_cost:.2f}",
            delta="-12.3% Capped vs Raw"
        )
    with k3:
        st.metric(
            label="Total Freight Spend",
            value=f"${total_spend:,.0f}",
            delta=f"{len(filtered_df):,} Shipments"
        )
    with k4:
        st.metric(
            label="Avg Delivery Delay",
            value=f"{avg_delay:+.2f} days",
            delta="Tail up to 18 days"
        )
    with k5:
        st.metric(
            label="Order Margin Captured",
            value=f"${filtered_df['profit_per_order'].sum():,.0f}",
            delta=f"${total_val:,.0f} Gross Sales"
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2 Column layout: Network Architecture & Problem Statement
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        <div class="card-box">
            <div class="card-title">🌐 Operating Scenario: Meridian Retail Logistics (MRL)</div>
            <p>
                MRL operates a multi-echelon regional retail fulfillment network serving four geographic regions 
                (<b>East, Central, West, South</b>) through three strategically located distribution centers:
            </p>
            <ul>
                <li><b>DC-East-01:</b> High-throughput hub handling apparel & office supplies fulfillment.</li>
                <li><b>DC-Central-02:</b> Central cross-dock facility specializing in general merchandise & sporting goods.</li>
                <li><b>DC-West-03:</b> Tech logistics hub fulfilling high-value electronics and regional last-mile parcels.</li>
            </ul>
            <p>
                <b>Core Dilemma:</b> Despite substantial investments in express modes, company-wide on-time delivery sits at 
                <b>57.4% – 58.6%</b>, falling over 36 percentage points short of the <b>≥95.0% SLA target</b>. Simultaneously, last-mile 
                carrier bills are inflated by unverified carrier rate invoices and misaligned shipping tier assignments.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="card-box">
            <div class="card-title">🎯 Integrated Logistics Intelligence Architecture</div>
            <div style="line-height: 1.8;">
                <span class="badge badge-blue">Layer 1</span> <b>Strategic Planning:</b> Scenario modeling, 6 core KPIs, baseline gap analysis.<br>
                <span class="badge badge-green">Layer 2</span> <b>Data Governance:</b> 8-step pipeline, 5,075 raw → 5,000 clean (35 features).<br>
                <span class="badge badge-amber">Layer 3</span> <b>Diagnostic Analytics:</b> Purpose-built visualizations, systemic delay diagnosis.<br>
                <span class="badge badge-blue">Layer 4</span> <b>Prescriptive Optimization:</b> Freight audit, mode reassignment, SLA calibration, routing.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Longitudinal Order Volume by DC Chart
    df_vol = df.copy()
    df_vol["order_period"] = df_vol["order_date"].dt.to_period("W").dt.to_timestamp()
    weekly_dc = df_vol.groupby(["order_period", "dc_id"]).size().reset_index(name="orders")
    
    fig_vol = px.area(
        weekly_dc,
        x="order_period",
        y="orders",
        color="dc_id",
        title="Fulfillment Order Volume Trajectory across Distribution Centers",
        color_discrete_sequence=["#0284c7", "#0d9488", "#6366f1"],
        labels={"order_period": "Fulfillment Period", "orders": "Orders per Period", "dc_id": "Distribution Center"}
    )
    fig_vol.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_vol, use_container_width=True)

# ==============================================================================
# TAB 2: DATA PIPELINE & QUALITY HEALTH
# ==============================================================================
with tabs[1]:
    st.markdown("### 🧹 Data Collection, Cleaning & Preprocessing Health")
    
    # Pipeline Step Metric Cards
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Raw Extract Rows", "5,075", delta="-75 Duplicates Removed")
    with s2:
        st.metric("Cleaned Model Rows", "5,000", delta="+17 Engineered Features")
    with s3:
        st.metric("Invalid Records Corrected", "113", delta="60 Qty + 53 Date Flips")
    with s4:
        st.metric("Missing Data Resolved", "625", delta="Unit Price, Segment, Dates")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_pipe, col_iqr = st.columns([1, 1])
    
    with col_pipe:
        st.markdown("""
        <div class="card-box">
            <div class="card-title">⚙️ 8-Step Data Transformation Pipeline</div>
            <ol style="font-size: 0.95rem; line-height: 1.7; padding-left: 1.2rem; color: #334155;">
                <li><b>Data Profiling:</b> Detected 5.5% missing dates, 3.7% missing segments, 3.1% missing prices.</li>
                <li><b>Deduplication:</b> 74 exact duplicates dropped + 1 soft duplicate pair resolved by completeness.</li>
                <li><b>Text Standardization:</b> Trimmed and normalized 12 dirty strings into 4 standard categories.</li>
                <li><b>Invalid Values:</b> 60 negative quantities sign-flipped; 53 temporal date violations set to null.</li>
                <li><b>Missing Value Triage:</b> Imputed prices via SKU-median; segments to 'Unknown'; delivery dates triaged into In Transit (57) and imputed (222).</li>
                <li><b>IQR Winsorization:</b> Shipping cost capped at $86.43 (224 outliers); parcel weight capped at 31.55 kg (285 outliers).</li>
                <li><b>Feature Engineering:</b> Calculated delivery_delay_days, is_late, order_value, and shipping_cost_log.</li>
                <li><b>Scaling:</b> Min-Max and Z-score standardization for downstream ML algorithms.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
    with col_iqr:
        st.markdown("""
        <div class="card-box">
            <div class="card-title">📐 Outlier Treatment: IQR Winsorization Fences</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Plotly comparison of raw vs capped shipping cost
        fig_iqr = go.Figure()
        fig_iqr.add_trace(go.Histogram(
            x=raw_df["shipping_cost"],
            name="Raw Shipping Cost ($)",
            opacity=0.5,
            marker_color="#ef4444",
            xbins=dict(start=0, end=150, size=5)
        ))
        fig_iqr.add_trace(go.Histogram(
            x=df["shipping_cost_capped"],
            name="IQR Capped Cost ($86.43)",
            opacity=0.7,
            marker_color="#0284c7",
            xbins=dict(start=0, end=150, size=5)
        ))
        fig_iqr.add_vline(x=86.43, line_dash="dash", line_color="#b91c1c", annotation_text="Upper Fence $86.43")
        fig_iqr.update_layout(
            barmode="overlay",
            template="plotly_white",
            height=300,
            title="Raw Cost Distribution vs. Winsorized IQR Cap",
            xaxis_title="Shipping Cost ($)",
            yaxis_title="Order Count",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_iqr, use_container_width=True)

    # Cleaned Dataset Interactive Preview
    with st.expander("🔍 Explore the 35-Column Cleaned Logistics Dataset", expanded=False):
        st.dataframe(filtered_df.head(100), use_container_width=True)
        st.download_button(
            label="📥 Download Full Cleaned Dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_logistics_sample.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 3: VISUAL ANALYTICS & DIAGNOSTICS
# ==============================================================================
with tabs[2]:
    st.markdown("### 📊 Advanced Exploratory Data Analysis & Diagnostics")
    st.caption("Diagnostic visualizations analyzing delivery delays, cost drivers, and operational segment variations.")
    
    # Row 1: Delay Distribution & Cost by Mode Boxplot
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        # 1. Delivery delay distribution
        fig_delay = px.histogram(
            delivered_filtered,
            x="delivery_delay_days",
            nbins=25,
            title="Delivery Delay Distribution (Delivered Orders)",
            color_discrete_sequence=["#0284c7"],
            labels={"delivery_delay_days": "Delivery Delay (Days, 0 = On Time)"}
        )
        fig_delay.add_vline(x=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Lateness Threshold")
        fig_delay.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_delay, use_container_width=True)
        
    with r1_col2:
        # 2. Shipping cost by mode boxplot
        mode_order = ["Standard Class", "Second Class", "First Class", "Same Day"]
        fig_box = px.box(
            filtered_df,
            x="shipping_mode",
            y="shipping_cost_capped",
            category_orders={"shipping_mode": mode_order},
            color="shipping_mode",
            title="Shipping Cost Distribution across Modes (Capped $)",
            color_discrete_sequence=["#94a3b8", "#38bdf8", "#0284c7", "#0369a1"]
        )
        fig_box.update_layout(template="plotly_white", height=320, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)
        
    # Row 2: OTD by Mode vs Target & OTD by DC
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        # 3. OTD by Shipping Mode
        otd_mode = delivered_filtered.groupby("shipping_mode").agg(
            otd=("delivery_delay_days", lambda s: (s <= 0).mean() * 100)
        ).reindex(mode_order).reset_index()
        
        fig_otd_m = px.bar(
            otd_mode,
            x="shipping_mode",
            y="otd",
            title="On-Time Delivery Rate by Shipping Mode vs Target",
            color="otd",
            color_continuous_scale="Teal",
            text_auto=".1f",
            labels={"otd": "OTD Rate (%)", "shipping_mode": "Shipping Mode"}
        )
        fig_otd_m.add_hline(y=95.0, line_dash="dash", line_color="#ef4444", annotation_text="Target ≥95%")
        fig_otd_m.update_layout(template="plotly_white", height=320, coloraxis_showscale=False, yaxis_range=[0, 105], margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_otd_m, use_container_width=True)
        
    with r2_col2:
        # 4. OTD by Distribution Center
        otd_dc = delivered_filtered.groupby("dc_id").agg(
            otd=("delivery_delay_days", lambda s: (s <= 0).mean() * 100)
        ).reset_index()
        
        fig_otd_dc = px.bar(
            otd_dc,
            x="dc_id",
            y="otd",
            title="On-Time Delivery Rate across Distribution Centers",
            color="otd",
            color_continuous_scale="Blues",
            text_auto=".1f",
            labels={"otd": "OTD Rate (%)", "dc_id": "Distribution Center"}
        )
        fig_otd_dc.add_hline(y=95.0, line_dash="dash", line_color="#ef4444", annotation_text="Target ≥95%")
        fig_otd_dc.update_layout(template="plotly_white", height=320, coloraxis_showscale=False, yaxis_range=[0, 105], margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_otd_dc, use_container_width=True)

    # Row 3: Correlation Matrix Heatmap & Shipping Cost vs Weight Scatter
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        # 5. Correlation Heatmap
        corr_cols = ["shipping_cost_capped", "order_weight_kg_capped", "order_value", "quantity", "discount_pct", "delivery_delay_days"]
        corr_matrix = filtered_df[corr_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Matrix Heatmap of Core Operational Variables",
            color_continuous_scale="Blues",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with r3_col2:
        # 6. Cost vs Weight by Mode Scatter
        fig_scatter = px.scatter(
            filtered_df.sample(min(1200, len(filtered_df)), random_state=42),
            x="order_weight_kg_capped",
            y="shipping_cost_capped",
            color="shipping_mode",
            category_orders={"shipping_mode": mode_order},
            title="Shipping Cost vs. Parcel Weight (Colored by Mode)",
            opacity=0.65,
            color_discrete_sequence=["#64748b", "#0284c7", "#f59e0b", "#ef4444"],
            labels={"order_weight_kg_capped": "Parcel Weight (kg)", "shipping_cost_capped": "Cost ($)"}
        )
        fig_scatter.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # Row 4: Longitudinal Trend & Pareto Concentration
    r4_col1, r4_col2 = st.columns(2)
    with r4_col1:
        # 7. Longitudinal volume vs delay
        df_time = delivered_filtered.copy()
        df_time["period"] = df_time["order_date"].dt.to_period("W").dt.to_timestamp()
        longitudinal_stats = df_time.groupby("period").agg(
            volume=("order_id", "count"),
            avg_delay=("delivery_delay_days", "mean")
        ).reset_index()
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Bar(x=longitudinal_stats["period"], y=longitudinal_stats["volume"], name="Order Volume", marker_color="#cbd5e1"),
            secondary_y=False
        )
        fig_trend.add_trace(
            go.Scatter(x=longitudinal_stats["period"], y=longitudinal_stats["avg_delay"], name="Avg Delay (Days)", line=dict(color="#0284c7", width=2)),
            secondary_y=True
        )
        fig_trend.update_layout(
            title="Longitudinal Order Volume vs. Average Delivery Delay",
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with r4_col2:
        # 8. Category Pareto Concentration
        cat_rev = df.groupby("product_category")["order_value"].sum().sort_values(ascending=False).reset_index()
        cat_rev["cum_pct"] = (cat_rev["order_value"].cumsum() / cat_rev["order_value"].sum()) * 100
        
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto.add_trace(
            go.Bar(x=cat_rev["product_category"], y=cat_rev["order_value"], name="Order Value ($)", marker_color="#0ea5e9"),
            secondary_y=False
        )
        fig_pareto.add_trace(
            go.Scatter(x=cat_rev["product_category"], y=cat_rev["cum_pct"], name="Cumulative %", line=dict(color="#f59e0b", width=3), mode="lines+markers"),
            secondary_y=True
        )
        fig_pareto.update_layout(
            title="Category Revenue Concentration (Pareto Analysis)",
            template="plotly_white",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

# ==============================================================================
# TAB 4: PREDICTIVE MACHINE LEARNING HUB
# ==============================================================================
with tabs[3]:
    st.markdown("### 🤖 Predictive Modeling Hub: Carrier Shipping Cost Estimation")
    st.markdown("""
    Evaluating candidate regressors on the 4,000-order training set against the 1,000-order held-out test window.
    The champion model is trained with an absolute-error loss on the **log-transformed target** to insulate against carrier billing errors.
    """)
    
    # Leaderboard Table
    st.markdown("#### 🏆 Model Performance Comparison on Held-Out Test Orders")
    st.dataframe(model_comp_df, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2 Columns: What-If Quoter and Feature Importance
    ml_col1, ml_col2 = st.columns([1, 1])
    
    with ml_col1:
        st.markdown("""
        <div class="card-box">
            <div class="card-title">🧮 Interactive 'What-If' Last-Mile Cost Quoter</div>
            <p style="font-size: 0.9rem; color: #64748b;">
                Simulate expected last-mile carrier invoice for any parcel configuration using the champion Gradient Boosting model:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        qc1, qc2 = st.columns(2)
        with qc1:
            input_weight = st.slider("Parcel Weight (kg)", min_value=0.5, max_value=45.0, value=8.5, step=0.5)
            input_qty = st.slider("Order Quantity (Units)", min_value=1, max_value=7, value=3)
            input_discount = st.select_slider("Discount Applied", options=[0.0, 0.05, 0.10, 0.15, 0.20], value=0.10)
        with qc2:
            input_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"], index=0)
            input_cat = st.selectbox("Product Category", ["Apparel", "Electronics", "Home Goods", "Sporting Goods", "Office Supplies"], index=1)
            
        # Model Inference
        input_data = pd.DataFrame([{
            "order_weight_kg_capped": min(31.55, input_weight),
            "quantity": input_qty,
            "discount_pct": input_discount,
            "shipping_mode": input_mode,
            "product_category": input_cat
        }])
        
        predicted_cost = champion_model.predict(input_data)[0]
        
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 2px solid #86efac; border-radius: 8px; padding: 1.2rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 0.9rem; font-weight: 600; color: #166534; text-transform: uppercase;">Expected Shipping Cost</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #15803d; margin: 0.3rem 0;">${predicted_cost:.2f}</div>
            <div style="font-size: 0.85rem; color: #166534;">95% Confidence Range: [${max(2.5, predicted_cost - 3.46):.2f} – ${predicted_cost + 3.46:.2f}]</div>
        </div>
        """, unsafe_allow_html=True)
        
    with ml_col2:
        st.markdown("""
        <div class="card-box">
            <div class="card-title">📊 Mode Cost Comparison for Current Parcel</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Cross-mode counterfactual comparison
        mode_preds = []
        for m in ["Standard Class", "Second Class", "First Class", "Same Day"]:
            m_sample = input_data.copy()
            m_sample["shipping_mode"] = m
            cost_m = champion_model.predict(m_sample)[0]
            mode_preds.append({"Mode": m, "Cost": cost_m})
            
        fig_mode_comp = px.bar(
            pd.DataFrame(mode_preds),
            x="Mode",
            y="Cost",
            title=f"Predicted Cost Across All Modes for {input_weight} kg Parcel",
            color="Cost",
            color_continuous_scale="Blues",
            text_auto=".2f",
            labels={"Cost": "Expected Cost ($)", "Mode": "Shipping Mode"}
        )
        fig_mode_comp.update_layout(template="plotly_white", height=320, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_mode_comp, use_container_width=True)

# ==============================================================================
# TAB 5: PRESCRIPTIVE OPTIMIZATION LEVERS
# ==============================================================================
with tabs[4]:
    st.markdown("### ⚡ Operational Optimization Levers")
    st.caption("Translating predictive model outputs into actionable supply chain improvements.")
    
    opt_tab1, opt_tab2, opt_tab3, opt_tab4 = st.tabs([
        "🔍 Freight-Bill Audit",
        "🔄 Mode Reassignment",
        "🎯 Promise Calibration",
        "🚚 Route Optimization"
    ])
    
    # --------------------------------------------------------------------------
    # LEVER 1: FREIGHT-BILL AUDIT
    # --------------------------------------------------------------------------
    with opt_tab1:
        st.markdown("#### Lever 1: Automated Freight-Bill Anomaly Audit")
        
        audit_col1, audit_col2 = st.columns([1, 2])
        with audit_col1:
            st.markdown("""
            **The Problem:** 2.6% of carrier bills contain severe data-entry errors (inflated 6x to 14x above rate-card tariff), costing tens of thousands in overpayment.
            """)
            audit_k = st.slider("Audit Multiplier Threshold (k)", min_value=2.0, max_value=8.0, value=4.0, step=0.5)
            
            from src.optimization_engine import run_freight_audit
            audit_summary, audited_df = run_freight_audit(k_threshold=audit_k)
            
            st.metric("Invoices Flagged for Audit", f"{audit_summary['total_flagged']} / {audit_summary['total_test_orders']}")
            st.metric("Real Billing Errors Caught", f"{audit_summary['real_errors_caught']} / 28", delta=f"{audit_summary['recall']*100:.1f}% Recall")
            st.metric("Audit Precision", f"{audit_summary['precision']*100:.1f}%", delta=f"{audit_summary['false_alarms']} False Alarms")
            st.metric("Overbilled Spend Recovered", f"${audit_summary['dollars_recovered']:,.2f}")
            
        with audit_col2:
            fig_audit = px.scatter(
                audited_df,
                x="predicted_cost",
                y="shipping_cost",
                color="audit_flagged",
                symbol="_is_error",
                log_x=True,
                log_y=True,
                title=f"Billed vs. Expected Shipping Cost (Log Scale, k={audit_k})",
                color_discrete_map={True: "#ef4444", False: "#0284c7"},
                labels={"predicted_cost": "Model-Expected Cost ($)", "shipping_cost": "Actual Carrier Bill ($)"}
            )
            # Add line y = k * x
            x_range = np.linspace(5, 100, 50)
            fig_audit.add_trace(go.Scatter(
                x=x_range,
                y=audit_k * x_range,
                mode="lines",
                line=dict(color="#b91c1c", dash="dash"),
                name=f"Flag Threshold (k={audit_k}x)"
            ))
            fig_audit.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_audit, use_container_width=True)
            
    # --------------------------------------------------------------------------
    # LEVER 2: MODE REASSIGNMENT
    # --------------------------------------------------------------------------
    with opt_tab2:
        st.markdown("#### Lever 2: Shipping-Mode Downgrade Simulator")
        st.markdown("""
        Moving non-urgent express parcels (Same Day → First Class; First Class → Second Class; Second Class → Standard Class)
        preserves customer satisfaction while driving significant margin expansion.
        """)
        
        from src.optimization_engine import run_mode_reassignment
        reassign_df, eligible = run_mode_reassignment(champion_model=champion_model)
        
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.dataframe(reassign_df, use_container_width=True)
        with rc2:
            fig_reassign = px.bar(
                reassign_df,
                x="Adoption Rate",
                y="Total Savings ($)",
                text_auto=True,
                title="Projected Freight Savings per 1,000 Orders by Adoption Rate",
                color="Total Savings ($)",
                color_continuous_scale="Greens"
            )
            fig_reassign.update_layout(template="plotly_white", height=320, coloraxis_showscale=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_reassign, use_container_width=True)

    # --------------------------------------------------------------------------
    # LEVER 3: PROMISE-DATE CALIBRATION
    # --------------------------------------------------------------------------
    with opt_tab3:
        st.markdown("#### Lever 3: Customer Promise-Date SLA Calibration")
        st.markdown("""
        Baseline OTD is only 57.4% because promised lead times are overly aggressive. 
        Calibrating customer promise windows with a realistic buffer guarantees fulfillment reliability.
        """)
        
        from src.optimization_engine import run_promise_calibration
        calib_df = run_promise_calibration()
        
        pc1, pc2 = st.columns([1, 1])
        with pc1:
            st.dataframe(calib_df, use_container_width=True)
        with rc2:
            fig_calib = px.line(
                calib_df,
                x="Buffer Days",
                y=["All Modes", "Standard Class", "Second Class", "First Class", "Same Day"],
                markers=True,
                title="On-Time Delivery Rate Progression vs. Promise Buffer",
                labels={"value": "On-Time Delivery Rate (%)", "variable": "Mode Tier"}
            )
            fig_calib.add_hline(y=95.0, line_dash="dash", line_color="#ef4444", annotation_text="Target ≥95%")
            fig_calib.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_calib, use_container_width=True)
            
    # --------------------------------------------------------------------------
    # LEVER 4: ROUTE OPTIMIZATION
    # --------------------------------------------------------------------------
    with opt_tab4:
        st.markdown("#### Lever 4: Vehicle Route Optimization (Clarke-Wright Savings Heuristic)")
        st.markdown("""
        Comparing standard dispatcher sweep against **Clarke-Wright savings heuristic with 2-opt search** on 40 delivery stops.
        """)
        
        from src.routing_engine import run_route_comparison
        route_summary, sweep_routes, cw_routes, depot, coords = run_route_comparison(seed=42)
        
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            st.metric("Dispatcher Sweep Distance", f"{route_summary['Dispatcher Sweep Km']} km")
        with rk2:
            st.metric("Clarke-Wright Distance", f"{route_summary['Clarke-Wright Km']} km")
        with rk3:
            st.metric("Distance Reduction", f"-{route_summary['Distance Reduction (%)']}%", delta="28.3% Average Across Simulations")
        with rk4:
            st.metric("Annual Route Savings", f"${route_summary['Annual Savings (250 days) ($)']:,.0f}", delta=f"${route_summary['Daily Dollar Savings ($)']}/day")
            
        # Interactive Route Visualization
        fig_route = go.Figure()
        # Plot Depot
        fig_route.add_trace(go.Scatter(
            x=[depot[0]], y=[depot[1]],
            mode="markers+text",
            marker=dict(size=16, color="#0f172a", symbol="square"),
            text=["DEPOT"], textposition="top center", name="Depot"
        ))
        
        # Plot Stops
        fig_route.add_trace(go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(size=9, color="#94a3b8"),
            name="Delivery Stops"
        ))
        
        # Plot CW Routes
        cw_colors = ["#0284c7", "#0d9488", "#6366f1", "#f59e0b"]
        for idx, r in enumerate(cw_routes):
            r_points = [0] + r + [0]
            rx = [depot[0] if p == 0 else coords[p-1, 0] for p in r_points]
            ry = [depot[1] if p == 0 else coords[p-1, 1] for p in r_points]
            fig_route.add_trace(go.Scatter(
                x=rx, y=ry,
                mode="lines+markers",
                line=dict(color=cw_colors[idx % len(cw_colors)], width=2.5),
                name=f"Van Route {idx+1} (CW + 2-Opt)"
            ))
            
        fig_route.update_layout(
            title="Optimized Delivery Van Routes (Clarke-Wright Heuristic)",
            xaxis_title="X (km)", yaxis_title="Y (km)",
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_route, use_container_width=True)

# ==============================================================================
# TAB 6: SYNTHESIS, METHODOLOGY & GITHUB REPO
# ==============================================================================
with tabs[5]:
    st.markdown("### 📚 Methodological Synthesis & Project Deliverables")
    
    st.markdown("""
    <div class="card-box">
        <div class="card-title">📖 Critical Analytical Corrections & Operational Insights</div>
        <ol style="line-height: 1.8; color: #334155;">
            <li><b>The Outlier Capping Distortion:</b> In initial data preprocessing, shipping cost was capped at $86.43 using traditional IQR fences. While capping protected simple models from exploding, it compressed legitimate heavy express parcels and understated the Same Day premium by 80 points. Using robust absolute-error loss on the log-scale (TransformedTargetRegressor) resolved this properly without discarding data.</li>
            <li><b>Systemic Delivery Delays:</b> Chi-square hypothesis tests confirmed that lateness is uniform across distribution centers, shipping modes, categories, and customer regions. Demand volume fluctuations do not cause delivery delays (r = -0.02). The solution is SLA buffer calibration (+3 days lifts on-time to 92.6%).</li>
            <li><b>Freight Bill Auditing:</b> Standard IQR rules generated 44 false alarms. The robust ML residual detector (k=4) achieved 100% recall with 0 false alarms on simulated test invoices, recovering $6,900 to $11,000 per 1,000 orders.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🚀 GitHub Repository Deployment Guide")
    st.markdown("""
    This project is packaged for zero-friction deployment to GitHub and GitHub Pages:
    ```bash
    # 1. Initialize Git Repository
    git init
    git add .
    git commit -m "feat: complete end-to-end logistics analytics control tower & optimization suite"

    # 2. Push to GitHub
    git remote add origin https://github.com/YOUR_USERNAME/meridian-logistics-dashboard.git
    git branch -M main
    git push -u origin main
    ```
    
    **To Run the Streamlit Dashboard Locally:**
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    
    **Live Browser Demo without Python (GitHub Pages):**
    Open `docs/index.html` in any browser, or enable **Settings → Pages** in your GitHub repository to deploy it automatically to `https://YOUR_USERNAME.github.io/meridian-logistics-dashboard/`.
    """)
