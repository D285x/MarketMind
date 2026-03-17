"""
MarketMind — AI Sales & Marketing Decision Intelligence Platform
================================================================
Upscaled production build.  Run with:
    streamlit run MarketMind.py

Requires:
    ANTHROPIC_API_KEY environment variable (or .env file via python-dotenv)
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Resolve imports when running from repo root ──────────
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from utils.styles   import GLOBAL_CSS
from utils.analytics import (
    compute_health_score, forecast, detect_anomalies,
    rfm_segments, rich_summary, exponential_smoothing,
)
from utils.charts import (
    revenue_timeseries, spend_vs_revenue, correlation_heatmap,
    revenue_orders_dual, forecast_chart, rfm_donut,
    waterfall_chart, score_gauge, roas_trend,
)
from utils.ai_engine import (
    stream_executive_report, stream_advisor_answer, get_auto_insights,
)

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="MarketMind",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════

def fmt_number(n: float, prefix: str = "") -> str:
    if abs(n) >= 1_000_000:
        return f"{prefix}{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{prefix}{n/1_000:.1f}K"
    return f"{prefix}{n:.2f}"


def delta_html(current: float, previous: float, prefix: str = "") -> str:
    if previous == 0:
        return ""
    pct = (current - previous) / abs(previous) * 100
    cls = "up" if pct > 0 else ("down" if pct < 0 else "flat")
    arrow = "▲" if pct > 0 else ("▼" if pct < 0 else "—")
    return f'<div class="kpi-delta {cls}">{arrow} {abs(pct):.1f}% vs. prior period</div>'


def kpi_card_html(label: str, value: str, delta_html_str: str, icon: str) -> str:
    return f"""
<div class="kpi-card">
  <div class="kpi-icon">{icon}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {delta_html_str}
</div>
"""


def section(title: str):
    st.markdown(f'<div class="mm-section">{title}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<div class="mm-divider"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ MarketMind</div>', unsafe_allow_html=True)
    st.caption("v2.0 · AI Decision Intelligence")
    st.divider()

    st.markdown("**Upload Dataset**")
    uploaded = st.file_uploader(
        "CSV file", type=["csv"],
        help="Expects columns: date, revenue, orders, ad_spend, etc."
    )

    st.divider()
    st.markdown("**Configuration**")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Leave blank to use GEMINI_API_KEY env var",
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, step=7)
    anomaly_thresh   = st.slider("Anomaly Sensitivity (σ)", 1.5, 4.0, 2.5, step=0.5)

    st.divider()

    # Download sample data
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "retail_sample.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                "⬇ Download Sample CSV",
                data=f,
                file_name="retail_sample.csv",
                mime="text/csv",
            )

    st.divider()
    st.caption("Built with ❤️ using Claude & Streamlit")


# ════════════════════════════════════════════════════════
#  Hero header
# ════════════════════════════════════════════════════════

st.markdown("""
<div class="mm-hero">
  <div class="mm-badge">AI-POWERED INTELLIGENCE</div>
  <h1 class="mm-logo">MarketMind</h1>
  <p class="mm-tagline">Sales & Marketing Decision Intelligence Platform — powered by Gemini AI</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  No file uploaded
# ════════════════════════════════════════════════════════

if not uploaded:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class="kpi-card" style="text-align:center;padding:36px 24px">
  <div style="font-size:36px;opacity:.6">📊</div>
  <div class="kpi-label" style="margin-top:12px">Upload a CSV</div>
  <div style="font-size:13px;color:#3a5a8a;margin-top:8px">Revenue, orders, ad spend…</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="kpi-card" style="text-align:center;padding:36px 24px">
  <div style="font-size:36px;opacity:.6">🤖</div>
  <div class="kpi-label" style="margin-top:12px">AI Analysis</div>
  <div style="font-size:13px;color:#3a5a8a;margin-top:8px">Executive reports & strategic Q&A</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="kpi-card" style="text-align:center;padding:36px 24px">
  <div style="font-size:36px;opacity:.6">🔮</div>
  <div class="kpi-label" style="margin-top:12px">Forecasting</div>
  <div style="font-size:13px;color:#3a5a8a;margin-top:8px">Polynomial + scenario planning</div>
</div>""", unsafe_allow_html=True)

    st.info("👈  Upload a CSV using the sidebar, or download the sample dataset to get started.", icon="💡")
    st.stop()


# ════════════════════════════════════════════════════════
#  Load & normalise data
# ════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset…")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


df = load_data(uploaded.read())
cols = {c.lower() for c in df.columns}
num_cols = df.select_dtypes("number").columns.tolist()

if not num_cols:
    st.error("No numeric columns detected in this file.")
    st.stop()

primary = next(
    (c for c in ("revenue", "sales", "gmv") if c in cols),
    num_cols[0]
)

# Derived metrics
mid = len(df) // 2
first_half  = df.iloc[:mid]
second_half = df.iloc[mid:]

anomalies = detect_anomalies(df[primary], z_thresh=anomaly_thresh)
summary_text = rich_summary(df)


# ════════════════════════════════════════════════════════
#  KPI strip
# ════════════════════════════════════════════════════════

section("Key Performance Indicators")

kpis = []
if "revenue" in cols:
    rev_now  = second_half["revenue"].sum()
    rev_prev = first_half["revenue"].sum()
    kpis.append(("Total Revenue",   fmt_number(rev_now, "$"),   delta_html(rev_now, rev_prev),   "💰"))
if "orders" in cols:
    ord_now  = second_half["orders"].sum()
    ord_prev = first_half["orders"].sum()
    kpis.append(("Total Orders",    fmt_number(ord_now),         delta_html(ord_now, ord_prev),   "🛒"))
if "revenue" in cols and "ad_spend" in cols:
    roas = df["revenue"].mean() / df["ad_spend"].replace(0, np.nan).mean()
    roas_prev = first_half["revenue"].mean() / first_half["ad_spend"].replace(0, np.nan).mean()
    roas_now  = second_half["revenue"].mean() / second_half["ad_spend"].replace(0, np.nan).mean()
    kpis.append(("Avg ROAS",         f"{roas:.2f}×",             delta_html(roas_now, roas_prev),  "📈"))
if "conversion_rate" in cols:
    cr_now  = second_half["conversion_rate"].mean()
    cr_prev = first_half["conversion_rate"].mean()
    kpis.append(("Avg Conv. Rate",   f"{df['conversion_rate'].mean():.2f}%",
                  delta_html(cr_now, cr_prev),  "🎯"))
if "avg_order_value" in cols:
    aov_now  = second_half["avg_order_value"].mean()
    aov_prev = first_half["avg_order_value"].mean()
    kpis.append(("Avg Order Value",  fmt_number(aov_now, "$"),   delta_html(aov_now, aov_prev),   "🏷️"))
if "returns" in cols and "orders" in cols:
    rr = df["returns"].mean() / df["orders"].replace(0, np.nan).mean() * 100
    kpis.append(("Return Rate",      f"{rr:.1f}%",               "",                              "↩️"))
if not kpis:
    kpis.append((primary.replace("_", " ").title(),
                 fmt_number(df[primary].sum()),
                 delta_html(second_half[primary].sum(), first_half[primary].sum()), "📊"))

# Render in rows of 4
for i in range(0, len(kpis), 4):
    batch = kpis[i:i+4]
    st.markdown(
        f'<div class="kpi-grid">{"".join(kpi_card_html(*b) for b in batch)}</div>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════
#  Health Score
# ════════════════════════════════════════════════════════

divider()
section("Business Health Score")

score, criteria = compute_health_score(df)
color = "#38f5a0" if score >= 75 else ("#f5b438" if score >= 50 else "#f55870")

col_gauge, col_criteria = st.columns([1, 2])
with col_gauge:
    st.plotly_chart(score_gauge(score), use_container_width=True)

with col_criteria:
    for c in criteria:
        pct = c["earned"] / c["max"] * 100
        bar_color = "#38f5a0" if pct >= 75 else ("#f5b438" if pct >= 50 else "#f55870")
        st.markdown(f"""
<div style="margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="font-size:13px;color:#c0cfe8;font-weight:500">{c['name']}</span>
    <span style="font-size:12px;color:{bar_color};font-weight:600">{c['earned']}/{c['max']}</span>
  </div>
  <div style="height:5px;background:rgba(60,140,255,0.1);border-radius:10px;overflow:hidden">
    <div style="height:100%;width:{pct:.0f}%;background:{bar_color};border-radius:10px"></div>
  </div>
  <div style="font-size:11px;color:#3a5a8a;margin-top:3px">{c['reason']}</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  Main tabs
# ════════════════════════════════════════════════════════

divider()

tab_dash, tab_analytics, tab_forecast, tab_ai = st.tabs([
    "📊  Dashboard",
    "🔬  Analytics",
    "🔮  Forecasting",
    "🤖  AI Advisor",
])


# ────────────────────────────────────────────────────────
#  TAB 1 — DASHBOARD
# ────────────────────────────────────────────────────────

with tab_dash:

    # Anomaly banner
    n_anom = anomalies.sum()
    if n_anom > 0:
        st.warning(
            f"⚡ **{n_anom} anomalies** detected in `{primary}` "
            f"(σ threshold: {anomaly_thresh}). "
            f"Highlighted in red on the chart below.",
            icon="🚨"
        )

    section("Revenue Performance")
    st.plotly_chart(
        revenue_timeseries(df, y_col=primary,
                           date_col="date" if "date" in cols else None,
                           anomaly_mask=anomalies),
        use_container_width=True
    )

    if "revenue" in cols and "orders" in cols:
        section("Revenue & Orders Trend")
        st.plotly_chart(
            revenue_orders_dual(df, date_col="date" if "date" in cols else None),
            use_container_width=True
        )

    if "revenue" in cols and "ad_spend" in cols:
        col_roas, col_scatter = st.columns(2)
        with col_roas:
            section("ROAS Over Time")
            st.plotly_chart(roas_trend(df, date_col="date" if "date" in cols else None),
                            use_container_width=True)
        with col_scatter:
            section("Ad Spend vs Revenue")
            st.plotly_chart(spend_vs_revenue(df), use_container_width=True)


# ────────────────────────────────────────────────────────
#  TAB 2 — ANALYTICS
# ────────────────────────────────────────────────────────

with tab_analytics:

    section("Business Metric Correlations")
    if len(num_cols) >= 2:
        st.plotly_chart(correlation_heatmap(df, num_cols[:10]),
                        use_container_width=True)
    else:
        st.info("Need ≥2 numeric columns for correlation analysis.")

    col_wf, col_rfm = st.columns(2)

    with col_wf:
        if "revenue" in cols and "date" in cols:
            section("Revenue Waterfall")
            st.plotly_chart(waterfall_chart(df), use_container_width=True)

    with col_rfm:
        section("Customer Segments (RFM)")
        rfm_df = rfm_segments(df)
        if rfm_df is not None:
            st.plotly_chart(rfm_donut(rfm_df), use_container_width=True)
        else:
            st.info("RFM segmentation requires a `revenue` column.")

    divider()
    section("Raw Data Explorer")
    n_rows = st.slider("Rows to display", 5, min(200, len(df)), 20)
    st.dataframe(df.head(n_rows), use_container_width=True, height=320)

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button("⬇ Export full dataset (CSV)",
                       data=buf.getvalue(),
                       file_name="marketmind_export.csv",
                       mime="text/csv")


# ────────────────────────────────────────────────────────
#  TAB 3 — FORECASTING
# ────────────────────────────────────────────────────────

with tab_forecast:

    section(f"{primary.replace('_',' ').title()} Forecast — {forecast_horizon} days")

    col_opt = st.radio(
        "Forecast column",
        options=[c for c in num_cols if c != "date"],
        horizontal=True,
        index=0,
    )

    x_h, x_f, y_f, ci = forecast(df[col_opt], periods=forecast_horizon)

    st.plotly_chart(
        forecast_chart(x_h, df[col_opt].values, x_f, y_f, ci,
                       col_label=col_opt.replace("_", " ").title()),
        use_container_width=True
    )

    # Forecast table
    horizon_dates = (
        pd.date_range(df["date"].iloc[-1], periods=forecast_horizon + 1, freq="D")[1:]
        if "date" in cols
        else range(len(df), len(df) + forecast_horizon)
    )
    forecast_df = pd.DataFrame({
        "Period":    list(horizon_dates),
        "Forecast":  y_f.round(2),
        "Lower CI":  (y_f - ci).round(2),
        "Upper CI":  (y_f + ci).round(2),
    })
    with st.expander("📋 Forecast Table"):
        st.dataframe(forecast_df, use_container_width=True)
        buf2 = io.BytesIO()
        forecast_df.to_csv(buf2, index=False)
        st.download_button("⬇ Download forecast CSV",
                           data=buf2.getvalue(),
                           file_name="forecast.csv",
                           mime="text/csv")

    divider()
    section("What-If Scenario Planner")

    c1, c2, c3 = st.columns(3)
    with c1:
        bear = st.slider("Bear case (%)", -50, 0, -15)
    with c2:
        base = st.slider("Base case (%)",  -10, 30, 10)
    with c3:
        bull = st.slider("Bull case (%)",   0, 100, 30)

    base_val = float(df[col_opt].mean())
    sc1, sc2, sc3 = st.columns(3)
    for col_el, label, pct, color_hex in [
        (sc1, "Bear", bear, "#f55870"),
        (sc2, "Base", base, "#f5b438"),
        (sc3, "Bull", bull, "#38f5a0"),
    ]:
        projected = base_val * (1 + pct / 100)
        col_el.markdown(f"""
<div class="kpi-card" style="text-align:center;border-color:{color_hex}40">
  <div class="kpi-label">{label} Scenario</div>
  <div class="kpi-value" style="color:{color_hex}">{fmt_number(projected)}</div>
  <div style="font-size:12px;color:{color_hex};margin-top:4px">{pct:+d}% vs. current avg</div>
</div>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────
#  TAB 4 — AI ADVISOR
# ────────────────────────────────────────────────────────

with tab_ai:

    api_ready = bool(os.getenv("GEMINI_API_KEY", "").strip())

    if not api_ready:
        st.warning(
            "**API key required.** Enter your Gemini API key in the sidebar "
            "or set the `GEMINI_API_KEY` environment variable.",
            icon="🔑"
        )

    # ── Auto insights ──────────────────────────────────
    section("Auto-Generated Insights")
    if st.button("✨ Generate AI Insights", disabled=not api_ready):
        with st.spinner("Analysing your data…"):
            try:
                raw = get_auto_insights(summary_text)
                lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
                cards_html = ""
                for line in lines[:5]:
                    cards_html += f'<div class="insight-card">{line}</div>'
                st.markdown(f'<div class="insight-grid">{cards_html}</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI error: {e}")

    divider()

    # ── Executive Report ───────────────────────────────
    section("Executive Report")
    if st.button("📄 Generate Executive Report", disabled=not api_ready):
        report_box = st.empty()
        full_report = ""
        try:
            with st.spinner(""):
                for chunk in stream_executive_report(summary_text):
                    full_report += chunk
                    report_box.markdown(
                        f'<div class="report-box">{full_report}</div>',
                        unsafe_allow_html=True
                    )
            # Download
            st.download_button(
                "⬇ Download Report (TXT)",
                data=full_report,
                file_name="executive_report.txt",
                mime="text/plain",
            )
        except Exception as e:
            st.error(f"Report generation failed: {e}\n\n{traceback.format_exc()}")

    divider()

    # ── Strategic Q&A ──────────────────────────────────
    section("AI Business Advisor")

    # Suggested questions
    suggested = [
        "What is driving our highest revenue days?",
        "How should we reallocate our ad spend?",
        "What is our biggest risk over the next quarter?",
        "Which customer segment should we prioritise?",
        "How can we reduce our return rate?",
    ]

    st.caption("Suggested questions:")
    scols = st.columns(len(suggested))
    chosen_q = ""
    for i, sq in enumerate(suggested):
        with scols[i]:
            if st.button(sq, key=f"sq_{i}", use_container_width=True):
                chosen_q = sq
                st.session_state["advisor_q"] = sq

    if "advisor_q" not in st.session_state:
        st.session_state["advisor_q"] = ""

    q = st.text_input(
        "Or type your own question…",
        value=st.session_state.get("advisor_q", ""),
        placeholder="e.g. What's the ROI on increasing ad spend by 20%?",
        key="advisor_q_input",
    )

    if st.button("Ask Advisor", disabled=not api_ready or not q.strip()):
        answer_box = st.empty()
        full_answer = ""
        try:
            for chunk in stream_advisor_answer(summary_text, q.strip()):
                full_answer += chunk
                answer_box.markdown(
                    f'<div class="report-box">{full_answer}</div>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Advisor error: {e}")

    divider()

    # ── Dataset summary (collapsible) ──────────────────
    with st.expander("📋 Dataset Summary (sent to AI)"):
        st.code(summary_text, language="text")
