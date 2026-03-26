"""
TDM Trade Flow Dashboard
========================
Run locally:   streamlit run app.py
Deploy:        push files/ folder to GitHub, set main file to files/app.py on Streamlit Cloud

Data expected at:  files/data/tdm_coffee.parquet
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TDM Trade Flow Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main    { background-color: #fafafa !important; }
  [data-testid="stHeader"]         { background: transparent !important; }
  .block-container { padding-top: 1rem !important; padding-bottom: 1.5rem; max-width: 1400px; }
  html, body, [class*="css"]       { font-family: -apple-system, "Helvetica Neue", sans-serif; }
  h1, h2, h3                       { color: #1d1d1f !important; font-weight: 500 !important; }
  hr  { border: none !important; border-top: 1px solid #e8e8ed !important; margin: 0.5rem 0 !important; }
  [data-testid="stExpander"]       { border: 1px solid #e8e8ed !important; border-radius: 8px !important; background: #fff !important; }
  [data-testid="stDataFrame"]      { border-radius: 8px; overflow: hidden; }
  .stCaption                       { color: #6e6e73 !important; font-size: 0.7rem !important; }
  [data-testid="stRadio"] label    { font-size: 0.74rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTH_ORDER  = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
NUM_TO_MONTH = {i + 1: m for i, m in enumerate(MONTH_ORDER)}

COMMODITY_FILES = {
    "Coffee": "data/tdm_coffee.parquet",
    "Cocoa":  "data/tdm_cocoa.parquet",
    "Sugar":  "data/tdm_sugar.parquet",
    "Cotton": "data/tdm_cotton.parquet",
}

_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, Helvetica Neue, sans-serif", color="#1d1d1f", size=10),
)
# Muted palette for non-highlighted years
_PAL = ["#7bafd4","#f4a460","#82c982","#c9a0dc","#e8c96a","#7ec8c0","#e89090","#a0aad4",
        "#c8a06e","#90b8a0","#d4c0a0","#a8c0d8"]


def lbl(text: str) -> str:
    """Navy blue section label with light grey text."""
    return (
        f"<div style='background:#0a2463;padding:5px 13px;border-radius:5px;margin:0 0 5px 0'>"
        f"<span style='font-size:0.78rem;font-weight:500;letter-spacing:0.07em;"
        f"text-transform:uppercase;color:#dde4f0'>{text}</span></div>"
    )


# ── TOP BAR ───────────────────────────────────────────────────────────────────
top_left, top_right = st.columns([1, 7])
with top_left:
    commodity = st.selectbox("", list(COMMODITY_FILES.keys()), label_visibility="collapsed")

data_path = Path(__file__).parent / COMMODITY_FILES[commodity]


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH"]].assign(DAY=1))
    df["BAGS"] = df["GBE"] * (1000 / 60)
    return df


if not data_path.exists():
    st.info(f"{commodity} data not available yet. Place parquet at: `{COMMODITY_FILES[commodity]}`")
    st.stop()

df = load_data(str(data_path))

# ── Filters (expander) ────────────────────────────────────────────────────────
with top_right:
    with st.expander("⚙  Filters", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 1.5, 2])

        all_reporters = sorted(df["REPORTER"].unique())
        with fc1:
            sel_reporters = st.multiselect("Reporter", all_reporters, default=all_reporters)

        all_tags = sorted(df["COMMODITY_TAG"].dropna().unique()) if "COMMODITY_TAG" in df.columns else []
        with fc2:
            sel_tags = st.multiselect("Type", all_tags, default=all_tags) if all_tags else []

        all_regions = sorted(df["REGION"].unique())
        with fc3:
            sel_regions = st.multiselect("Destination", all_regions, default=all_regions)

        all_partners = sorted(df["PARTNER"].dropna().unique())
        with fc4:
            sel_partners = st.multiselect("Partner", all_partners, default=all_partners)

# ── Apply filters ─────────────────────────────────────────────────────────────
all_reporters_all = sorted(df["REPORTER"].unique())
all_regions_all   = sorted(df["REGION"].unique())
all_partners_all  = sorted(df["PARTNER"].dropna().unique())

mask = (
    df["REPORTER"].isin(sel_reporters or all_reporters_all)
    & df["REGION"].isin(sel_regions or all_regions_all)
    & df["PARTNER"].isin(sel_partners or all_partners_all)
)
if all_tags:
    mask &= df["COMMODITY_TAG"].isin(sel_tags or all_tags)

dff = df[mask].copy()

# ── Latest common month ────────────────────────────────────────────────────────
if not dff.empty:
    latest_cy           = sorted(dff["CROP_YEAR"].unique())[-1]
    lm_per_rep          = dff[dff["CROP_YEAR"] == latest_cy].groupby("REPORTER")["CROP_MONTH_NUM"].max()
    latest_common_num   = int(lm_per_rep.min()) if len(lm_per_rep) else 12
    latest_common_label = NUM_TO_MONTH[latest_common_num]
    dff_disp = dff[
        (dff["CROP_YEAR"] < latest_cy) |
        ((dff["CROP_YEAR"] == latest_cy) & (dff["CROP_MONTH_NUM"] <= latest_common_num))
    ].copy()
else:
    latest_cy           = ""
    latest_common_num   = 12
    latest_common_label = "Sep"
    dff_disp            = dff.copy()

# Previous crop year (for line colouring)
_sorted_cy = sorted(dff_disp["CROP_YEAR"].unique()) if not dff_disp.empty else []
prev_cy    = _sorted_cy[-2] if len(_sorted_cy) >= 2 else None


def cy_style(cy):
    """Return (color, line_width) for a crop year line."""
    if cy == latest_cy:
        return "#1d1d1f", 2.5
    if cy == prev_cy:
        return "#c0392b", 2.0
    return None, 1.4   # None → use palette


# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"### {commodity} Export Trade Flows &nbsp;"
    f"<span style='font-size:0.85rem;font-weight:400;color:#6e6e73'>GBE · 60kg Bags</span>",
    unsafe_allow_html=True,
)
_cy_list = sorted(dff_disp["CROP_YEAR"].unique()) if not dff_disp.empty else ["—", "—"]
st.caption(
    f"Crop years {_cy_list[0]} – {_cy_list[-1]}  ·  "
    f"{dff_disp['REPORTER'].nunique()} reporters  ·  "
    f"Latest month: {latest_common_label} ({latest_cy})  ·  "
    f"Bold black = {latest_cy}  ·  Red = {prev_cy}"
)
st.markdown("<hr>", unsafe_allow_html=True)

if dff_disp.empty:
    st.warning("No data for the current selection.")
    st.stop()

# ── Pivot helper ──────────────────────────────────────────────────────────────
def build_pivot(data: pd.DataFrame):
    grp = data.groupby(["CROP_YEAR", "CROP_MONTH_NUM"])["BAGS"].sum().reset_index()
    grp["CROP_MONTH"] = grp["CROP_MONTH_NUM"].map(NUM_TO_MONTH)
    pivot = (
        grp.pivot(index="CROP_YEAR", columns="CROP_MONTH", values="BAGS")
        .reindex(columns=MONTH_ORDER).fillna(0).sort_index(ascending=True)
    )
    complete = (pivot > 0).sum(axis=1) == 12
    return pivot, complete


pivot, complete = build_pivot(dff_disp)
complete_years  = sorted(complete[complete].index.tolist())

# ── Pre-compute YTD (needed in Row 2 and Row 4) ───────────────────────────────
ytd = (
    dff_disp[dff_disp["CROP_MONTH_NUM"] <= latest_common_num]
    .groupby("CROP_YEAR")["BAGS"].sum().reset_index()
    .sort_values("CROP_YEAR").rename(columns={"BAGS": "YTD_BAGS"})
)
ytd["YOY_PCT"] = ytd["YTD_BAGS"].pct_change() * 100

# =============================================================================
# ROW 1 — Heatmap with embedded Total column
# =============================================================================
st.markdown(lbl("Flow Heatmap — GBE in 60kg Bags · Monthly Exports by Crop Year"), unsafe_allow_html=True)
st.caption(
    f"Latest crop year ({latest_cy}) capped at {latest_common_label}  ·  "
    f"No-data cells = light grey  ·  Total = sum of available months"
)

disp = pivot[MONTH_ORDER].astype(float)
disp[disp == 0] = np.nan
# Total only for fully complete crop years (all 12 months present)
disp["Total"] = np.where(complete, disp[MONTH_ORDER].sum(axis=1), np.nan)

_fmt = lambda x: f"{x:,.0f}" if pd.notna(x) else ""

styled = (
    disp.style
    .background_gradient(cmap="RdYlGn", axis=None, subset=MONTH_ORDER)
    .highlight_null(color="#f0f0f0")
    .format(_fmt, subset=MONTH_ORDER)
    .format(_fmt, subset=["Total"])
    .set_properties(**{"text-align": "center", "font-size": "9px"})
    .set_properties(
        subset=["Total"],
        **{"font-weight": "700", "background-color": "#f5f5f7", "border-left": "2px solid #d8d8e0", "font-size": "9px"},
    )
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center"), ("font-size", "9px"), ("font-weight", "600")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-size", "9px")]},
    ])
)
tbl_h = min(340, 26 * len(disp) + 42)
st.dataframe(styled, use_container_width=True, height=tbl_h)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ROW 2 — Min/Max/Avg vs Latest  |  YTD Table
# =============================================================================
col_mm, col_ytd_tbl = st.columns([3, 2])

with col_mm:
    st.markdown(lbl("Min / Max / Avg vs Latest — GBE in 60kg Bags"), unsafe_allow_html=True)
    if complete_years:
        last5 = complete_years[-5:]
        ref   = pivot.loc[last5, MONTH_ORDER]
        mn, mx, avg = ref.min(), ref.max(), ref.mean()
        latest_x = MONTH_ORDER[:latest_common_num]
        latest_y = [
            pivot.loc[latest_cy, m] if latest_cy in pivot.index and pivot.loc[latest_cy, m] > 0 else None
            for m in latest_x
        ]
        fig_mm = go.Figure()
        fig_mm.add_trace(go.Scatter(x=MONTH_ORDER, y=mx.values,  name=f"Max (L{len(last5)}Y)", mode="lines", line=dict(color="#5a9e6f", width=1.4)))
        fig_mm.add_trace(go.Scatter(x=MONTH_ORDER, y=mn.values,  name=f"Min (L{len(last5)}Y)", mode="lines", line=dict(color="#e07b39", width=1.4), fill="tonexty", fillcolor="rgba(180,180,180,0.10)"))
        fig_mm.add_trace(go.Scatter(x=MONTH_ORDER, y=avg.values, name=f"Avg (L{len(last5)}Y)", mode="lines", line=dict(dash="dot", color="#aaaaaa", width=1.4)))
        fig_mm.add_trace(go.Scatter(x=latest_x, y=latest_y, name=latest_cy, mode="lines+markers", line=dict(color="#1d1d1f", width=2.5), marker=dict(size=5)))
        fig_mm.update_layout(
            height=210,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, showgrid=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9)),
            legend=dict(orientation="h", y=-0.42, font=dict(size=9)),
            margin=dict(t=4, b=55, l=4, r=4),
            **_D,
        )
        st.plotly_chart(fig_mm, use_container_width=True)
    else:
        st.info("No complete crop years for reference.")

with col_ytd_tbl:
    st.markdown(lbl(f"YTD · Oct – {latest_common_label} — GBE in 60kg Bags"), unsafe_allow_html=True)
    tbl2 = ytd.copy()
    tbl2["YTD_FMT"] = tbl2["YTD_BAGS"].map(lambda x: f"{x:,.0f}")
    tbl2["YOY_FMT"] = tbl2["YOY_PCT"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    st.dataframe(
        tbl2[["CROP_YEAR","YTD_FMT","YOY_FMT"]].rename(columns={
            "CROP_YEAR": "Crop Year",
            "YTD_FMT":  f"YTD Bags (Oct–{latest_common_label})",
            "YOY_FMT":  "YoY %",
        }),
        use_container_width=True, hide_index=True, height=210,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ROW 3 — Seasonal  |  Cumulative
# =============================================================================
sea = (
    dff_disp.groupby(["CROP_YEAR", "CROP_MONTH_NUM"])["BAGS"]
    .sum().reset_index().sort_values(["CROP_YEAR", "CROP_MONTH_NUM"])
)
sea["CROP_MONTH"] = sea["CROP_MONTH_NUM"].map(NUM_TO_MONTH)
all_sea_cy  = sorted(sea["CROP_YEAR"].unique())
default_sea = all_sea_cy[-6:] if len(all_sea_cy) >= 6 else all_sea_cy

# Min/Max/Avg reference bands
pivot_s, complete_s  = build_pivot(dff_disp)
complete_years_s     = sorted(complete_s[complete_s].index.tolist())
ref_band: dict = {}
if complete_years_s:
    last5_s   = complete_years_s[-5:]
    ref_s     = pivot_s.loc[last5_s, MONTH_ORDER]
    ref_band  = {"max": ref_s.max(), "min": ref_s.min(), "avg": ref_s.mean(), "n": len(last5_s)}

col_sea, col_cum = st.columns(2)

with col_sea:
    st.markdown(lbl("Seasonal — GBE in 60kg Bags · Monthly by Crop Year"), unsafe_allow_html=True)
    sel_sea = st.multiselect("Crop years", all_sea_cy, default=default_sea, key="sea_sel")
    fig3 = go.Figure()
    if ref_band:
        fig3.add_trace(go.Scatter(x=MONTH_ORDER, y=ref_band["max"].values, name=f"Max (L{ref_band['n']}Y)", mode="lines", line=dict(color="#5a9e6f", width=1.2)))
        fig3.add_trace(go.Scatter(x=MONTH_ORDER, y=ref_band["min"].values, name=f"Min (L{ref_band['n']}Y)", mode="lines", line=dict(color="#e07b39", width=1.2), fill="tonexty", fillcolor="rgba(180,180,180,0.08)"))
        fig3.add_trace(go.Scatter(x=MONTH_ORDER, y=ref_band["avg"].values, name=f"Avg (L{ref_band['n']}Y)", mode="lines", line=dict(dash="dot", color="#aaaaaa", width=1.2)))
    pal_i = 0
    for cy in sorted(sel_sea):
        color, width = cy_style(cy)
        if color is None:
            color = _PAL[pal_i % len(_PAL)]; pal_i += 1
        d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM")
        fig3.add_trace(go.Scatter(x=d["CROP_MONTH"], y=d["BAGS"], name=cy, mode="lines+markers", line=dict(color=color, width=width), marker=dict(size=3)))
    fig3.update_layout(
        height=210,
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9)),
        legend=dict(orientation="h", y=-0.45, font=dict(size=9)),
        margin=dict(t=4, b=58, l=4, r=4),
        **_D,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_cum:
    st.markdown(lbl("Cumulative Exports — GBE in 60kg Bags · by Crop Year"), unsafe_allow_html=True)
    sel_cum = st.multiselect("Crop years", all_sea_cy, default=default_sea, key="cum_sel")
    fig4 = go.Figure()
    pal_i = 0
    for cy in sorted(sel_cum):
        color, width = cy_style(cy)
        if color is None:
            color = _PAL[pal_i % len(_PAL)]; pal_i += 1
        d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM").copy()
        d["CUM_BAGS"] = d["BAGS"].cumsum()
        fig4.add_trace(go.Scatter(x=d["CROP_MONTH"], y=d["CUM_BAGS"], name=cy, mode="lines+markers", line=dict(color=color, width=width), marker=dict(size=3)))
    fig4.update_layout(
        height=210,
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9)),
        legend=dict(orientation="h", y=-0.45, font=dict(size=9)),
        margin=dict(t=4, b=58, l=4, r=4),
        **_D,
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ROW 4 — Rolling  |  YTD Trend + YoY
# =============================================================================
col_roll, col_ytd_charts = st.columns(2)

with col_roll:
    st.markdown(lbl("Rolling Exports — GBE in 60kg Bags"), unsafe_allow_html=True)
    roll_choice = st.radio("Window", ["1m","3m","6m","12m"], index=3, horizontal=True)
    window = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}[roll_choice]
    monthly = dff_disp.groupby("DATE")["BAGS"].sum().reset_index().sort_values("DATE")
    monthly["ROLLING"] = monthly["BAGS"].rolling(window).sum()
    fig5 = go.Figure(go.Scatter(
        x=monthly["DATE"], y=monthly["ROLLING"],
        mode="lines", line=dict(color="#4a7fb5", width=1.8),
        fill="tozeroy", fillcolor="rgba(74,127,181,0.07)",
    ))
    fig5.update_layout(
        height=210,
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9)),
        margin=dict(t=4, b=10, l=4, r=4),
        **_D,
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_ytd_charts:
    st.markdown(lbl("YTD Trend + YoY % Change — GBE in 60kg Bags"), unsafe_allow_html=True)
    ya, yb = st.columns(2)
    with ya:
        fig6 = go.Figure(go.Scatter(
            x=ytd["CROP_YEAR"], y=ytd["YTD_BAGS"],
            mode="lines+markers", line=dict(color="#4a7fb5", width=1.8), marker=dict(size=4),
        ))
        fig6.update_layout(
            height=210,
            xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9)),
            margin=dict(t=4, b=10, l=4, r=4),
            **_D,
        )
        st.plotly_chart(fig6, use_container_width=True)
    with yb:
        pct_df = ytd.dropna(subset=["YOY_PCT"]).copy()
        pct_df["COLOR"] = pct_df["YOY_PCT"].apply(lambda x: "#5a9e6f" if x >= 0 else "#c0392b")
        fig7 = go.Figure(go.Bar(
            x=pct_df["CROP_YEAR"], y=pct_df["YOY_PCT"],
            marker_color=pct_df["COLOR"],
            text=pct_df["YOY_PCT"].map(lambda x: f"{x:+.1f}%"),
            textposition="outside", textfont=dict(size=8),
        ))
        fig7.add_hline(y=0, line_color="#cccccc", line_width=1)
        fig7.update_layout(
            height=210,
            xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9)),
            margin=dict(t=14, b=10, l=4, r=4),
            **_D,
        )
        st.plotly_chart(fig7, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("TDM Trade Flow Dashboard  ·  ETG Softs  ·  60kg bags (GBE)")
