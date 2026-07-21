from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Prudentia",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Data and API helpers
# =============================================================================

def api_get(endpoint: str, timeout: int = 10) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(endpoint: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(detail)
    return response.json()


def api_is_available() -> bool:
    try:
        return api_get("/health", timeout=3).get("status") == "healthy"
    except requests.RequestException:
        return False


@st.cache_data
def load_csv(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_percent(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
        if np.isnan(numeric):
            return "—"
        return f"{numeric * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def safe_number(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
        if np.isnan(numeric):
            return "—"
        return f"{numeric:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render_html(content: str) -> None:
    """Render HTML without Markdown parsing it as a code block."""
    st.html(content)


def download_csv(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def status_badge(value: str) -> str:
    status = str(value).upper()
    if status in {"ALERT", "HIGH", "CRITICAL", "IMMATURE", "INVESTIGATE", "RETRAINING REVIEW"}:
        css_class = "badge-orange"
    elif status in {"WATCH", "PROVISIONAL"}:
        css_class = "badge-yellow"
    elif status in {"STABLE", "MATURE", "REFERENCE", "APPROVED"}:
        css_class = "badge-green"
    else:
        css_class = "badge-blue"
    return f'<span class="badge {css_class}">{escape(status)}</span>'


# =============================================================================
# Shared project data
# =============================================================================

model_metadata = load_json("artifacts/models/best_model_metadata.json")
calibration_metadata = load_json("artifacts/models/calibration_metadata.json")
monitoring_metadata = load_json("data/processed/phase8_monitoring/monitoring_metadata.json")
portfolio = load_csv("data/processed/phase8_monitoring/quarterly_portfolio_metrics.csv")
performance = load_csv("data/processed/phase8_monitoring/quarterly_model_performance.csv")
score_psi = load_csv("data/processed/phase8_monitoring/score_psi_by_quarter.csv")
feature_summary = load_csv("data/processed/phase8_monitoring/feature_psi_summary.csv")
feature_psi = load_csv("data/processed/phase8_monitoring/feature_psi_by_quarter.csv")
alerts = load_csv("data/processed/phase8_monitoring/drift_alerts.csv")
recommendations = load_csv("data/processed/phase8_monitoring/quarterly_monitoring_summary.csv")
shap = load_csv("data/processed/phase7_explainability/global_shap_importance.csv")
fairness = load_csv("data/processed/phase7_explainability/fairness_disparate_impact_audit.csv")
reason_examples = load_csv("data/processed/phase7_explainability/local_adverse_action_reason_codes.csv")

api_online = api_is_available()
try:
    model_info = api_get("/model-info") if api_online else {}
except requests.RequestException:
    model_info = {}

model_name = model_info.get(
    "model_name",
    model_metadata.get("best_model_name", "Unavailable"),
)
calibration = model_info.get(
    "calibration_method",
    calibration_metadata.get("selected_calibration_method", "Unavailable"),
)
threshold = float(model_info.get("decision_threshold", 0.08))

recommendation_values = set(
    recommendations.get("recommendation", pd.Series(dtype=str)).dropna().astype(str)
)
if "RETRAINING REVIEW" in recommendation_values:
    overall = "RETRAINING REVIEW"
elif "INVESTIGATE" in recommendation_values:
    overall = "INVESTIGATE"
elif "PROVISIONAL" in recommendation_values:
    overall = "STABLE WITH PROVISIONAL VINTAGES"
else:
    overall = "STABLE" if recommendation_values else "UNAVAILABLE"

latest_portfolio = portfolio.iloc[-1] if not portfolio.empty else pd.Series(dtype=object)
latest_score = score_psi.iloc[-1] if not score_psi.empty else pd.Series(dtype=object)


# =============================================================================
# Theme and CSS
# =============================================================================

THEMES = {
    "Dark": {
        "bg": "#06111f",
        "sidebar": "#081523",
        "panel": "#0b1a2a",
        "panel_alt": "#0e2033",
        "border": "#1e354c",
        "text": "#f4f7fb",
        "muted": "#9eb0c3",
        "blue": "#35a4ff",
        "purple": "#8b5cf6",
        "green": "#2ed58b",
        "orange": "#ff9418",
        "yellow": "#f3c63f",
        "red": "#ff5b67",
        "chart_bg": "#08111d",
        "grid": "#22384e",
    },
    "Light": {
        "bg": "#f4f7fb",
        "sidebar": "#ffffff",
        "panel": "#ffffff",
        "panel_alt": "#f8fbff",
        "border": "#dce6f0",
        "text": "#162235",
        "muted": "#6d7d90",
        "blue": "#1778ff",
        "purple": "#7047eb",
        "green": "#0ca56a",
        "orange": "#d96c00",
        "yellow": "#9b7600",
        "red": "#d83b4f",
        "chart_bg": "#ffffff",
        "grid": "#e7edf4",
    },
}


selected_theme = st.sidebar.selectbox("Appearance", ["Dark", "Light"], index=0)
theme = THEMES[selected_theme]

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {theme['bg']};
        --sidebar: {theme['sidebar']};
        --panel: {theme['panel']};
        --panel-alt: {theme['panel_alt']};
        --border: {theme['border']};
        --text: {theme['text']};
        --muted: {theme['muted']};
        --blue: {theme['blue']};
        --purple: {theme['purple']};
        --green: {theme['green']};
        --orange: {theme['orange']};
        --yellow: {theme['yellow']};
        --red: {theme['red']};
    }}

    html, body, [class*="css"] {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .stApp {{ background: var(--bg); color: var(--text); }}
    [data-testid="stHeader"] {{ background: transparent; height:0rem; }}
    [data-testid="stToolbar"] {{ right: 0.8rem; }}
    [data-testid="stSidebar"] {{ background: var(--sidebar); border-right: 1px solid var(--border); }}
    [data-testid="stSidebar"] * {{ color: var(--text); }}
    [data-testid="stSidebarContent"] {{ padding-top: .45rem; }}
    .block-container {{ padding: .35rem .65rem .45rem .65rem; max-width: 1900px; }}
    h1, h2, h3, h4, h5, h6, p, span, label {{ color: var(--text); }}

    div[role="radiogroup"] label {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.48rem 0.65rem;
        margin: 0.12rem 0;
    }}
    div[role="radiogroup"] label:hover {{ background: var(--panel-alt); }}
    div[role="radiogroup"] label:has(input:checked) {{
        background: linear-gradient(90deg, rgba(53,164,255,.22), rgba(53,164,255,.08));
        border-color: rgba(53,164,255,.3);
    }}

    .brand {{ display:flex; gap:10px; align-items:center; margin:0 0 .7rem; }}
    .brand-icon {{
        width:38px; height:38px; display:grid; place-items:center; border-radius:10px;
        background:linear-gradient(135deg, rgba(53,164,255,.28), rgba(139,92,246,.20));
        border:1px solid var(--border); font-size:1.35rem;
    }}
    .brand-name {{ font-size:1.28rem; font-weight:800; line-height:1; }}
    .brand-sub {{ color:var(--muted); font-size:.67rem; margin-top:4px; }}
    .api-pill {{
        display:inline-flex; align-items:center; gap:7px; border:1px solid var(--border);
        border-radius:999px; padding:7px 10px; background:var(--panel); font-size:.76rem;
    }}
    .dot {{ width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 10px rgba(46,213,139,.8); }}
    .dot.offline {{ background:var(--red); box-shadow:0 0 10px rgba(255,91,103,.65); }}

    .page-header {{ display:flex; justify-content:flex-end; align-items:center; gap:.7rem; margin:0 0 .35rem; min-height:28px; }}
    .page-title {{ font-size:1.35rem; font-weight:800; margin:0; }}
    .page-subtitle {{ color:var(--muted); font-size:.74rem; margin-top:3px; }}

    .kpi-card {{
        background:var(--panel); border:1px solid var(--border); border-radius:13px;
        padding:8px 11px; min-height:60px; box-shadow:0 6px 18px rgba(0,0,0,.10);
    }}
    .kpi-label {{ color:var(--muted); font-size:.59rem; text-transform:uppercase; letter-spacing:.055em; }}
    .kpi-value {{ color:var(--text); font-size:1.03rem; font-weight:800; margin-top:6px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .kpi-accent-blue {{ border-top:2px solid var(--blue); }}
    .kpi-accent-purple {{ border-top:2px solid var(--purple); }}
    .kpi-accent-green {{ border-top:2px solid var(--green); }}
    .kpi-accent-orange {{ border-top:2px solid var(--orange); }}

    .panel {{
        background:var(--panel); border:1px solid var(--border); border-radius:13px;
        padding:8px 11px; box-shadow:0 6px 18px rgba(0,0,0,.09); height:100%;
    }}
    .panel-title {{ font-size:.67rem; font-weight:800; letter-spacing:.045em; text-transform:uppercase; margin-bottom:5px; }}
    .kv-row {{ display:flex; justify-content:space-between; align-items:center; gap:8px; padding:4px 0; border-bottom:1px solid var(--border); font-size:.67rem; }}
    .kv-row:last-child {{ border-bottom:none; }}
    .kv-key {{ color:var(--muted); }}
    .kv-value {{ color:var(--text); font-weight:700; text-align:right; }}

    .badge {{ display:inline-block; padding:3px 8px; border-radius:6px; font-size:.65rem; font-weight:800; border:1px solid transparent; white-space:nowrap; }}
    .badge-orange {{ color:var(--orange); background:rgba(255,148,24,.12); border-color:rgba(255,148,24,.36); }}
    .badge-yellow {{ color:var(--yellow); background:rgba(243,198,63,.12); border-color:rgba(243,198,63,.36); }}
    .badge-green {{ color:var(--green); background:rgba(46,213,139,.12); border-color:rgba(46,213,139,.36); }}
    .badge-blue {{ color:var(--blue); background:rgba(53,164,255,.12); border-color:rgba(53,164,255,.36); }}

    .alert-row {{ display:grid; grid-template-columns:7px 1fr auto; gap:7px; align-items:center; padding:4px 0; border-bottom:1px solid var(--border); }}
    .alert-row:last-child {{ border-bottom:none; }}
    .alert-dot {{ width:7px; height:7px; border-radius:50%; background:var(--yellow); }}
    .alert-main {{ font-size:.64rem; font-weight:700; color:var(--text); }}
    .alert-sub {{ font-size:.55rem; color:var(--muted); margin-top:1px; }}

    .section-title {{ margin:.45rem 0 .22rem; font-size:.82rem; font-weight:800; }}
    .chart-card {{ background:var(--panel); border:1px solid var(--border); border-radius:11px; padding:6px 6px 1px; height:100%; }}
    .chart-title {{ font-size:.62rem; font-weight:800; margin:0 0 0 2px; }}
    .small-note {{ color:var(--muted); font-size:.72rem; }}

    div[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:12px; overflow:hidden; }}
    div[data-testid="stForm"] {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:1rem; }}
    div[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:10px; background:var(--panel); }}
    .stButton button, .stDownloadButton button {{ border-radius:9px; }}
    [data-testid="stSidebar"] {{ min-width: 215px !important; max-width: 215px !important; }}
    [data-testid="stSidebar"] > div:first-child {{ width: 215px !important; }}
    .top-meta {{ display:flex; justify-content:flex-end; align-items:center; gap:8px; }}
    .updated-pill {{ color:var(--muted); border:1px solid var(--border); border-radius:7px; padding:5px 8px; background:var(--panel); font-size:.58rem; }}
    .compact-table-wrap {{ background:var(--panel); border:1px solid var(--border); border-radius:11px; overflow:hidden; }}
    .compact-table {{ width:100%; border-collapse:collapse; table-layout:fixed; }}
    .compact-table th {{ color:var(--muted); background:var(--panel-alt); font-size:.54rem; font-weight:600; text-align:left; padding:3px 5px; border-bottom:1px solid var(--border); white-space:nowrap; }}
    .compact-table td {{ color:var(--text); font-size:.56rem; padding:2px 5px; border-bottom:1px solid var(--border); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .compact-table tr:last-child td {{ border-bottom:none; }}
    .compact-table td.num, .compact-table th.num {{ text-align:right; }}
    .table-head {{ display:flex; justify-content:space-between; align-items:center; padding:5px 8px 2px; background:var(--panel); }}
    .table-title {{ font-size:.64rem; font-weight:800; text-transform:uppercase; letter-spacing:.035em; }}
    .table-link {{ color:var(--blue); font-size:.55rem; }}
    div[data-testid="stVerticalBlock"] > div {{ gap: .18rem; }}
    @media (max-height: 900px) {{ .block-container {{ zoom: .88; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Altair styling
# =============================================================================

def base_chart(chart: alt.Chart, height: int = 215) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(stroke=None, fill=theme["chart_bg"])
        .configure(background=theme["chart_bg"])
        .configure_axis(
            labelColor=theme["muted"],
            titleColor=theme["muted"],
            gridColor=theme["grid"],
            domainColor=theme["grid"],
            tickColor=theme["grid"],
            labelFontSize=9,
            titleFontSize=10,
        )
        .configure_legend(
            labelColor=theme["muted"],
            titleColor=theme["muted"],
            labelFontSize=9,
            titleFontSize=9,
            orient="bottom",
        )
    )


def chart_panel(title: str) -> None:
    render_html(f'<div class="chart-title">{escape(title)}</div>')


# =============================================================================
# Sidebar navigation
# =============================================================================

status_class = "" if api_online else " offline"
status_text = "API Connected" if api_online else "API Offline"

render_html(
    """
    <div class="brand">
      <div class="brand-icon">🛡️</div>
      <div><div class="brand-name">Prudentia</div><div class="brand-sub">Credit Risk Decisioning</div></div>
    </div>
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Applicant Scoring",
        "Batch Scoring",
        "Model Governance",
        "Drift Monitoring",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {model_name}")
st.sidebar.caption(f"Calibration: {str(calibration).title()}")
st.sidebar.caption(f"PD Threshold: {threshold:.2%}")
st.sidebar.caption(f"Encoded Features: {monitoring_metadata.get('total_encoded_features', 'N/A')}")
st.sidebar.caption(f"Monitoring: {overall}")


# =============================================================================
# Compact shared header
# =============================================================================

# if page == "Executive Overview":
#     render_html(
#         """
#         # <div class="hero-header">
#         #   <div class="hero-title">Prudentia</div>
#         #   <div class="hero-subtitle">An explainable, production-style credit-risk decisioning platform combining calibrated machine learning, cost-sensitive policy, model governance, fairness auditing, and drift monitoring.</div>
#         # </div>
#         """
#     )
# else:
#     render_html(
#         f"""
#         <div class="page-header" style="justify-content:space-between;">
#           <div>
#             <div class="page-title">{escape(page)}</div>
#             <div class="page-subtitle">Calibrated credit-risk decisioning, governance and monitoring</div>
#           </div>
#           <div class="api-pill"><span class="dot{status_class}"></span>{escape(status_text)}</div>
#         </div>
#         """
#     )


# =============================================================================
# Executive Overview — compact one-screen layout
# =============================================================================

if page == "Executive Overview":
    kpi_cols = st.columns(5, gap="small")
    kpis = [
        ("Selected Model", str(model_name), "kpi-accent-blue"),
        ("Calibration", str(calibration).title(), "kpi-accent-purple"),
        ("PD Threshold", f"{threshold:.2%}", "kpi-accent-green"),
        ("Monitoring Alerts", str(len(alerts)), "kpi-accent-orange"),
        ("Current Status", overall, "kpi-accent-orange"),
    ]
    for column, (label, value, accent) in zip(kpi_cols, kpis):
        with column:
            render_html(
                f'<div class="kpi-card {accent}"><div class="kpi-label">{escape(label)}</div><div class="kpi-value">{escape(value)}</div></div>'
            )

    summary_col, portfolio_col, alert_col = st.columns([1.12, 1.0, 1.0], gap="small")

    with summary_col:
        render_html(
            f"""
            <div class="panel">
              <div class="panel-title">Model &amp; Policy Summary</div>
              <div class="kv-row"><span class="kv-key">Model</span><span class="kv-value">{escape(str(model_name))}</span></div>
              <div class="kv-row"><span class="kv-key">Calibration</span><span class="kv-value">{escape(str(calibration).title())}</span></div>
              <div class="kv-row"><span class="kv-key">Approval Rule</span><span class="kv-value">PD below {threshold:.2%}</span></div>
              <div class="kv-row"><span class="kv-key">Recommendation</span><span class="kv-value">{status_badge(overall)}</span></div>
              <div class="kv-row"><span class="kv-key">Encoded Features</span><span class="kv-value">{monitoring_metadata.get('total_encoded_features', 'N/A')}</span></div>
            </div>
            """
        )

    with portfolio_col:
        maturity = str(latest_portfolio.get("maturity_status", "N/A"))
        render_html(
            f"""
            <div class="panel">
              <div class="panel-title">Latest Portfolio Period</div>
              <div class="kv-row"><span class="kv-key">Quarter</span><span class="kv-value">{escape(str(latest_portfolio.get('quarter', 'N/A')))}</span></div>
              <div class="kv-row"><span class="kv-key">Approval Rate</span><span class="kv-value">{safe_percent(latest_portfolio.get('approval_rate'))}</span></div>
              <div class="kv-row"><span class="kv-key">Average Predicted PD</span><span class="kv-value">{safe_percent(latest_portfolio.get('average_pd'))}</span></div>
              <div class="kv-row"><span class="kv-key">High-Risk Share</span><span class="kv-value">{safe_percent(latest_portfolio.get('high_risk_share'))}</span></div>
              <div class="kv-row"><span class="kv-key">Maturity Status</span><span class="kv-value">{status_badge(maturity)}</span></div>
            </div>
            """
        )

    with alert_col:
        recent = alerts.tail(4).iloc[::-1] if not alerts.empty else pd.DataFrame()
        if recent.empty:
            alert_rows = '<div class="small-note">No monitoring alerts generated.</div>'
        else:
            items: list[str] = []
            for _, row in recent.iterrows():
                severity = str(row.get("severity", "WATCH")).upper()
                label = str(row.get("alert_type", "Monitoring alert"))
                metric = str(row.get("metric", ""))
                value_text = safe_number(row.get("observed_value"), 4)
                items.append(
                    f'<div class="alert-row"><span class="alert-dot"></span><div><div class="alert-main">{escape(label)}</div><div class="alert-sub">{escape(metric)} · {escape(value_text)}</div></div>{status_badge(severity)}</div>'
                )
            alert_rows = "".join(items)
        render_html(f'<div class="panel"><div class="panel-title">Recent Alerts</div>{alert_rows}</div>')

    monitoring_portfolio = (
        portfolio[~portfolio["quarter"].astype(str).str.contains("Reference", case=False, na=False)].copy()
        if not portfolio.empty else pd.DataFrame()
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        render_html('<div class="chart-card"><div class="chart-title">Quarterly Model Score PSI</div>')
        if not score_psi.empty:
            line = alt.Chart(score_psi).mark_line(point=alt.OverlayMarkDef(size=35), strokeWidth=2.2, color=theme["blue"]).encode(
                x=alt.X("quarter:N", title=None, sort=None, axis=alt.Axis(labelAngle=-55, labelFontSize=8)),
                y=alt.Y("score_psi:Q", title="PSI", scale=alt.Scale(domain=[0, 0.30])),
                tooltip=["quarter:N", alt.Tooltip("score_psi:Q", format=".4f")],
            )
            watch = alt.Chart(pd.DataFrame({"y": [0.10]})).mark_rule(color=theme["yellow"], strokeDash=[5, 4]).encode(y="y:Q")
            alert_rule = alt.Chart(pd.DataFrame({"y": [0.25]})).mark_rule(color=theme["red"], strokeDash=[5, 4]).encode(y="y:Q")
            st.altair_chart(base_chart(line + watch + alert_rule, 190), use_container_width=True)
        render_html('</div>')

    with c2:
        render_html('<div class="chart-card"><div class="chart-title">Top Drifted Business Features</div>')
        if not feature_psi.empty and "original_feature" in feature_psi.columns:
            top = feature_psi.groupby("original_feature")["psi"].max().nlargest(7).index
            heat = feature_psi[feature_psi["original_feature"].isin(top)].copy()
            heat = heat.groupby(["original_feature", "quarter"], as_index=False)["psi"].max()
            heatmap = alt.Chart(heat).mark_rect().encode(
                x=alt.X("quarter:N", title=None, sort=None, axis=alt.Axis(labelAngle=-55, labelFontSize=8)),
                y=alt.Y("original_feature:N", title=None, sort=list(top), axis=alt.Axis(labelLimit=105, labelFontSize=8)),
                color=alt.Color("psi:Q", scale=alt.Scale(scheme="viridis"), title="PSI", legend=alt.Legend(orient="bottom", direction="horizontal", gradientLength=115)),
                tooltip=["original_feature:N", "quarter:N", alt.Tooltip("psi:Q", format=".4f")],
            )
            st.altair_chart(base_chart(heatmap, 190), use_container_width=True)
        render_html('</div>')

    with c3:
        render_html('<div class="chart-card"><div class="chart-title">Quarterly Approval &amp; Default Rates</div>')
        if not monitoring_portfolio.empty:
            rate_data = monitoring_portfolio[["quarter", "approval_rate", "observed_default_rate"]].melt("quarter", var_name="Metric", value_name="Rate")
            rate_data["Metric"] = rate_data["Metric"].map({"approval_rate": "Approval Rate", "observed_default_rate": "Observed Default Rate"})
            rate_chart = alt.Chart(rate_data).mark_line(point=alt.OverlayMarkDef(size=34), strokeWidth=2.1).encode(
                x=alt.X("quarter:N", title=None, sort=None, axis=alt.Axis(labelAngle=-55, labelFontSize=8)),
                y=alt.Y("Rate:Q", title=None, axis=alt.Axis(format="%")),
                color=alt.Color("Metric:N", scale=alt.Scale(range=[theme["blue"], theme["orange"]]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["quarter:N", "Metric:N", alt.Tooltip("Rate:Q", format=".2%")],
            )
            st.altair_chart(base_chart(rate_chart, 190), use_container_width=True)
        render_html('</div>')

    with c4:
        render_html('<div class="chart-card"><div class="chart-title">Predicted vs Observed Default Rate</div>')
        if not monitoring_portfolio.empty:
            pd_data = monitoring_portfolio[["quarter", "average_pd", "observed_default_rate"]].melt("quarter", var_name="Metric", value_name="Rate")
            pd_data["Metric"] = pd_data["Metric"].map({"average_pd": "Average Predicted PD", "observed_default_rate": "Observed Default Rate"})
            pd_chart = alt.Chart(pd_data).mark_line(point=alt.OverlayMarkDef(size=34), strokeWidth=2.1).encode(
                x=alt.X("quarter:N", title=None, sort=None, axis=alt.Axis(labelAngle=-55, labelFontSize=8)),
                y=alt.Y("Rate:Q", title=None, axis=alt.Axis(format="%")),
                color=alt.Color("Metric:N", scale=alt.Scale(range=[theme["blue"], theme["orange"]]), legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["quarter:N", "Metric:N", alt.Tooltip("Rate:Q", format=".2%")],
            )
            st.altair_chart(base_chart(pd_chart, 190), use_container_width=True)
        render_html('</div>')

    def feature_table_html() -> str:
        if feature_summary.empty:
            return '<div class="compact-table-wrap"><div class="table-head"><span class="table-title">Most Drifted Features</span></div><div class="small-note" style="padding:8px;">No data.</div></div>'
        rows = []
        for _, row in feature_summary.head(5).iterrows():
            status = str(row.get("latest_status", ""))
            rows.append(
                '<tr>'
                f'<td style="width:27%;">{escape(str(row.get("original_feature", "")))}</td>'
                f'<td class="num">{safe_number(row.get("maximum_psi"), 4)}</td>'
                f'<td class="num">{safe_number(row.get("average_psi"), 4)}</td>'
                f'<td class="num">{int(row.get("quarters_in_watch", 0) or 0)}</td>'
                f'<td class="num">{int(row.get("quarters_in_alert", 0) or 0)}</td>'
                f'<td class="num">{safe_number(row.get("latest_psi"), 4)}</td>'
                f'<td>{status_badge(status)}</td>'
                '</tr>'
            )
        return (
            '<div class="compact-table-wrap"><div class="table-head"><span class="table-title">Most Drifted Features</span><span class="table-link">Top 5</span></div>'
            '<table class="compact-table"><thead><tr><th>Feature</th><th class="num">Max PSI</th><th class="num">Avg PSI</th><th class="num">Watch</th><th class="num">Alert</th><th class="num">Latest</th><th>Status</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>'
        )

    def portfolio_table_html() -> str:
        if portfolio.empty:
            return '<div class="compact-table-wrap"><div class="table-head"><span class="table-title">Quarterly Portfolio Monitoring</span></div><div class="small-note" style="padding:8px;">No data.</div></div>'
        rows = []
        for _, row in portfolio.head(9).iterrows():
            maturity = str(row.get("maturity_status", ""))
            rows.append(
                '<tr>'
                f'<td>{escape(str(row.get("quarter", "")))}</td>'
                f'<td class="num">{int(row.get("loans", 0) or 0):,}</td>'
                f'<td>{status_badge(maturity)}</td>'
                f'<td class="num">{safe_percent(row.get("approval_rate"))}</td>'
                f'<td class="num">{safe_percent(row.get("observed_default_rate"))}</td>'
                f'<td class="num">{safe_percent(row.get("average_pd"))}</td>'
                f'<td class="num">{safe_number(row.get("calibration_gap"), 4)}</td>'
                '</tr>'
            )
        return (
            '<div class="compact-table-wrap"><div class="table-head"><span class="table-title">Quarterly Portfolio Monitoring</span><span class="table-link">Reference + 8 quarters</span></div>'
            '<table class="compact-table"><thead><tr><th>Quarter</th><th class="num">Loans</th><th>Maturity</th><th class="num">Approval</th><th class="num">Observed DR</th><th class="num">Avg PD</th><th class="num">Cal. Gap</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>'
        )

    left_table, right_table = st.columns([0.9, 1.35], gap="small")
    with left_table:
        render_html(feature_table_html())
    with right_table:
        render_html(portfolio_table_html())


# =============================================================================
# Applicant Scoring
# =============================================================================

elif page == "Applicant Scoring":
    if not api_online:
        st.error("FastAPI is offline. Start it with: uvicorn api.main:app --reload")

    with st.form("single_scoring"):
        row1 = st.columns(4)
        loan_amnt = row1[0].number_input("Loan amount", 500.0, 100000.0, 15000.0, 500.0)
        term = row1[1].selectbox("Term", ["36 months", "60 months"])
        annual_inc = row1[2].number_input("Annual income", 1000.0, 10000000.0, 75000.0, 1000.0)
        dti = row1[3].number_input("Debt-to-income ratio", 0.0, 100.0, 18.0, 0.5)

        row2 = st.columns(4)
        revol_util = row2[0].number_input("Revolving utilization (%)", 0.0, 200.0, 45.0, 1.0)
        revol_bal = row2[1].number_input("Revolving balance", 0.0, 10000000.0, 12000.0, 500.0)
        fico_low = row2[2].number_input("FICO range low", 300, 850, 690, 5)
        fico_high = row2[3].number_input("FICO range high", 300, 850, 694, 5)

        row3 = st.columns(4)
        emp_length = row3[0].selectbox("Employment length", ["10+ years", "9 years", "8 years", "7 years", "6 years", "5 years", "4 years", "3 years", "2 years", "1 year", "< 1 year", "None"])
        home_ownership = row3[1].selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN", "OTHER", "ANY"])
        verification_status = row3[2].selectbox("Verification status", ["Verified", "Source Verified", "Not Verified"])
        purpose = row3[3].selectbox("Loan purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "small_business", "car", "moving", "vacation", "house", "renewable_energy", "wedding", "other"])

        row4 = st.columns(4)
        delinq_2yrs = row4[0].number_input("Delinquencies in last 2 years", 0, 100, 0, 1)
        inquiries = row4[1].number_input("Inquiries in last 6 months", 0, 100, 1, 1)
        open_acc = row4[2].number_input("Open accounts", 0, 200, 10, 1)
        pub_rec = row4[3].number_input("Public records", 0, 100, 0, 1)

        submitted = st.form_submit_button("Score application", use_container_width=True)

    if submitted:
        payload = {
            "loan_amnt": loan_amnt,
            "funded_amnt": loan_amnt,
            "term": term,
            "annual_inc": annual_inc,
            "dti": dti,
            "revol_util": revol_util,
            "revol_bal": revol_bal,
            "fico_range_low": fico_low,
            "fico_range_high": fico_high,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "verification_status": verification_status,
            "purpose": purpose,
            "delinq_2yrs": delinq_2yrs,
            "inq_last_6mths": inquiries,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
        }
        try:
            result = api_post("/predict", payload)
            result_cols = st.columns(5)
            values = [
                ("Predicted PD", f"{result['probability_percent']:.2f}%"),
                ("Decision", result["decision"]),
                ("Risk Tier", result["risk_tier"]),
                ("Threshold", f"{result['decision_threshold']:.2%}"),
                ("Expected Loss", f"${result['expected_loss']:,.2f}"),
            ]
            for col, (label, value) in zip(result_cols, values):
                with col:
                    render_html(f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div><div class="kpi-value">{escape(str(value))}</div></div>')
            st.subheader("Primary risk factors")
            for reason in result["reasons"]:
                st.write(f"• {reason}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# =============================================================================
# Batch Scoring
# =============================================================================

elif page == "Batch Scoring":
    st.write("Upload a CSV with at least `loan_amnt`, `annual_inc`, and `dti`.")
    uploaded = st.file_uploader("Applicant CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        st.dataframe(batch.head(25), use_container_width=True)
        missing = {"loan_amnt", "annual_inc", "dti"} - set(batch.columns)
        if missing:
            st.error("Missing required columns: " + ", ".join(sorted(missing)))
        elif st.button("Run batch scoring", use_container_width=True):
            try:
                records = batch.where(pd.notna(batch), None).to_dict(orient="records")
                result = api_post("/batch-predict", {"applications": records}, timeout=300)
                predictions = pd.DataFrame(result["predictions"])
                predictions["reasons"] = predictions["reasons"].apply(lambda x: " | ".join(x) if isinstance(x, list) else x)
                scored = pd.concat([batch.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
                st.success(f"Scored {len(scored):,} applications")
                st.dataframe(scored, use_container_width=True, height=500)
                download_csv(scored, "prudentia_scored_applications.csv", "Download scored results")
            except Exception as exc:
                st.error(f"Batch scoring failed: {exc}")


# =============================================================================
# Model Governance
# =============================================================================

elif page == "Model Governance":
    st.subheader("Global Model Drivers")
    if not shap.empty:
        columns = [col for col in ["feature", "mean_abs_shap", "plain_english_reason"] if col in shap.columns]
        left, right = st.columns([1.1, 1])
        with left:
            st.dataframe(shap[columns].head(25), hide_index=True, use_container_width=True, height=440)
        with right:
            top = shap.head(15).copy()
            bar = alt.Chart(top).mark_bar(color=theme["blue"]).encode(
                x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature:N", alt.Tooltip("mean_abs_shap:Q", format=".4f")],
            )
            st.altair_chart(base_chart(bar, 420), use_container_width=True)

    st.subheader("Fairness and Disparate-Impact Review")
    st.warning("Protected attributes are unavailable in the Lending Club data. This is a protected-adjacent segment review, not a legal fair-lending certification.")
    if not fairness.empty:
        flagged = fairness.copy()
        if "flag_80pct_rule" in flagged.columns:
            flagged = flagged[flagged["flag_80pct_rule"].astype(str).str.lower().isin(["true", "1"])]
        st.dataframe(flagged, hide_index=True, use_container_width=True, height=430)
        download_csv(fairness, "phase7_fairness_audit.csv", "Download fairness audit")

    st.subheader("Adverse-Action Reason Examples")
    if not reason_examples.empty:
        st.dataframe(reason_examples.head(30), hide_index=True, use_container_width=True, height=420)


# =============================================================================
# Drift Monitoring
# =============================================================================

elif page == "Drift Monitoring":
    cols = st.columns(4)
    latest_recommendation = recommendations.iloc[-1]["recommendation"] if not recommendations.empty else "N/A"
    values = [
        ("Latest Score PSI", safe_number(latest_score.get("score_psi"), 4)),
        ("Score Status", str(latest_score.get("status", "N/A"))),
        ("Monitoring Alerts", str(len(alerts))),
        ("Latest Recommendation", str(latest_recommendation)),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            render_html(f'<div class="kpi-card"><div class="kpi-label">{escape(label)}</div><div class="kpi-value">{escape(value)}</div></div>')

    st.subheader("Monitoring Tables")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Drift", "Portfolio", "Performance", "Alerts & Recommendations"])
    with tab1:
        st.dataframe(feature_summary, hide_index=True, use_container_width=True, height=520)
    with tab2:
        st.dataframe(portfolio, hide_index=True, use_container_width=True, height=520)
    with tab3:
        st.dataframe(performance, hide_index=True, use_container_width=True, height=520)
    with tab4:
        st.dataframe(alerts, hide_index=True, use_container_width=True, height=260)
        st.dataframe(recommendations, hide_index=True, use_container_width=True, height=320)