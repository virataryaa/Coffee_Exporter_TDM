"""
TDM Trade Flow Dashboard
========================
Run locally:
    python -m streamlit run files/app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

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
  [data-testid="stDataFrame"]      { border-radius: 8px; overflow: visible !important; }
  .stCaption                       { color: #6e6e73 !important; font-size: 0.7rem !important; }
  [data-testid="stRadio"] label    { font-size: 0.74rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTH_ORDER  = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
NUM_TO_MONTH = {i + 1: m for i, m in enumerate(MONTH_ORDER)}

FLOW_PATHS = {
    "Coffee Exports":           r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Fundamental\TDM\tdm_coffee.parquet",
    "Coffee Imports":           r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Fundamental\TDM\tdm_coffee_imports.parquet",
    "Coffee Imports (EU Only)": r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\Fundamental\TDM\coffee_imports_eu.parquet",
}

_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, Helvetica Neue, sans-serif", color="#1d1d1f", size=10),
)
_PAL = ["#7bafd4","#f4a460","#82c982","#c9a0dc","#e8c96a","#7ec8c0","#e89090","#a0aad4",
        "#c8a06e","#90b8a0","#d4c0a0","#a8c0d8"]


def lbl(text: str) -> str:
    return (
        f"<div style='background:#0a2463;padding:5px 13px;border-radius:5px;"
        f"margin:0 0 5px 0;text-align:center'>"
        f"<span style='font-size:0.78rem;font-weight:500;letter-spacing:0.07em;"
        f"text-transform:uppercase;color:#dde4f0'>{text}</span></div>"
    )


# ── Flow selector (single choice, no double-counting) ─────────────────────────
flow_choice = st.radio(
    "Flow",
    list(FLOW_PATHS.keys()),
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)
flow_label = "Exports" if flow_choice == "Coffee Exports" else "Imports"


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH"]].assign(DAY=1))
    df["BAGS"] = df["GBE"] / 60
    return df


data_path = Path(FLOW_PATHS[flow_choice])
try:
    df = load_data(str(data_path))
except Exception as e:
    st.error(str(e))
    st.stop()

# ── Filters ────────────────────────────────────────────────────────────────────
with st.expander("⚙  Filters", expanded=False):
    fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 1.5, 2])

    all_reporters = sorted(df["REPORTER"].dropna().unique())
    with fc1:
        sel_reporters = st.multiselect("Reporter", all_reporters, default=all_reporters)

    all_tags = sorted(df["COMMODITY_TAG"].dropna().unique()) if "COMMODITY_TAG" in df.columns else []
    with fc2:
        sel_tags = st.multiselect("Type", all_tags, default=all_tags) if all_tags else []

    all_regions = sorted(df["REGION"].dropna().unique())
    with fc3:
        sel_regions = st.multiselect("Partner Region", all_regions, default=all_regions)

    all_partners = sorted(df["PARTNER"].dropna().unique())
    with fc4:
        sel_partners = st.multiselect("Partner", all_partners, default=all_partners)

# ── Apply filters ──────────────────────────────────────────────────────────────
mask = (
    df["REPORTER"].isin(sel_reporters or all_reporters)
    & df["REGION"].isin(sel_regions or all_regions)
    & df["PARTNER"].isin(sel_partners or all_partners)
)
if all_tags:
    mask &= df["COMMODITY_TAG"].isin(sel_tags or all_tags)

dff = df[mask].copy()

# ── Latest common month (based on selected flow's reporters only) ──────────────
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

# Drop oldest crop year from visuals
if not dff_disp.empty and dff_disp["CROP_YEAR"].nunique() > 1:
    oldest_cy = sorted(dff_disp["CROP_YEAR"].unique())[0]
    dff_disp  = dff_disp[dff_disp["CROP_YEAR"] != oldest_cy].copy()

_sorted_cy = sorted(dff_disp["CROP_YEAR"].unique()) if not dff_disp.empty else []
prev_cy    = _sorted_cy[-2] if len(_sorted_cy) >= 2 else None


def cy_style(cy):
    if cy == latest_cy:
        return "#1d1d1f", 2.5
    if cy == prev_cy:
        return "#c0392b", 2.0
    return None, 1.4


# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"### Coffee Trade Flows &nbsp;"
    f"<span style='font-size:0.85rem;font-weight:400;color:#6e6e73'>GBE · k Bags</span>",
    unsafe_allow_html=True,
)
_cy_list = sorted(dff_disp["CROP_YEAR"].unique()) if not dff_disp.empty else ["—", "—"]
st.caption(
    f"Crop years {_cy_list[0]} – {_cy_list[-1]}  ·  "
    f"{dff_disp['REPORTER'].nunique()} reporters  ·  "
    f"Latest month: {latest_common_label} ({latest_cy})  ·  "
    f"Bold black = {latest_cy}  ·  Red = {prev_cy}  ·  All volumes in thousands of 60kg bags"
)
st.markdown("<hr>", unsafe_allow_html=True)

if dff_disp.empty:
    st.warning("No data for the current selection.")
    st.stop()


# ── Pivot helper ───────────────────────────────────────────────────────────────
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

# ── YTD ────────────────────────────────────────────────────────────────────────
ytd = (
    dff_disp[dff_disp["CROP_MONTH_NUM"] <= latest_common_num]
    .groupby("CROP_YEAR")["BAGS"].sum().reset_index()
    .sort_values("CROP_YEAR").rename(columns={"BAGS": "YTD_BAGS"})
)
ytd["YOY_PCT"] = ytd["YTD_BAGS"].pct_change() * 100

_fmt = lambda x: f"{x:,.0f}" if pd.notna(x) else ""

# =============================================================================
# ROW 1 — Heatmap with Min/Max/Avg reference rows
# =============================================================================
st.markdown(lbl(f"Flow Heatmap (GBE in k Bags) · Monthly {flow_label} by Crop Year"), unsafe_allow_html=True)
st.caption(
    f"Latest crop year ({latest_cy}) capped at {latest_common_label}  ·  "
    f"Light grey = no data  ·  Total shown only for complete Oct–Sep years  ·  "
    f"Min / Max / Avg rows based on last 5 complete crop years"
)

disp = pivot[MONTH_ORDER].astype(float)
disp[disp == 0] = np.nan
disp["Total"] = np.where(complete, disp[MONTH_ORDER].sum(axis=1), np.nan)

main_idx  = disp.index.tolist()
_REF_ROWS = []

if complete_years:
    last5         = complete_years[-5:]
    ref           = pivot.loc[last5, MONTH_ORDER].astype(float)
    annual_totals = ref.sum(axis=1)
    n             = len(last5)

    sep     = pd.Series({m: np.nan for m in MONTH_ORDER + ["Total"]}, name="  ")
    row_min = pd.Series({**ref.min().to_dict(),  "Total": annual_totals.min()},  name=f"Min (L{n}Y)")
    row_max = pd.Series({**ref.max().to_dict(),  "Total": annual_totals.max()},  name=f"Max (L{n}Y)")
    row_avg = pd.Series({**ref.mean().to_dict(), "Total": annual_totals.mean()}, name=f"Avg (L{n}Y)")

    disp_full = pd.concat([
        disp,
        sep.to_frame().T,
        row_min.to_frame().T,
        row_max.to_frame().T,
        row_avg.to_frame().T,
    ])
    _REF_ROWS = [f"Min (L{n}Y)", f"Max (L{n}Y)", f"Avg (L{n}Y)"]
else:
    disp_full = disp.copy()

all_data_idx = main_idx + _REF_ROWS

styled = (
    disp_full.style
    .background_gradient(cmap="RdYlGn", axis=None, subset=pd.IndexSlice[main_idx, MONTH_ORDER])
    .highlight_null(color="#f0f0f0")
    .format(_fmt, subset=pd.IndexSlice[all_data_idx, MONTH_ORDER + ["Total"]])
    .set_properties(**{"text-align": "center", "font-size": "8px"})
    .set_properties(
        subset=pd.IndexSlice[main_idx, ["Total"]],
        **{"font-weight": "700", "background-color": "#f5f5f7",
           "border-left": "2px solid #d8d8e0", "font-size": "8px"},
    )
    .set_table_styles([
        {"selector": "th", "props": [("text-align","center"),("font-size","8px"),("font-weight","600")]},
        {"selector": "td", "props": [("text-align","center"),("font-size","8px")]},
    ])
)

if _REF_ROWS:
    styled = styled.set_properties(
        subset=pd.IndexSlice[_REF_ROWS, :],
        **{"background-color": "#eef3fb", "font-weight": "600",
           "font-style": "italic", "color": "#2c3e6e"},
    )

st.dataframe(styled, use_container_width=True, height=min(35 * (len(disp_full.index) + 3), 900))
st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ROW 2 — Rolling  |  Min/Max/Avg vs Latest
# =============================================================================
col_roll, col_mm = st.columns(2)

with col_roll:
    st.markdown(lbl(f"Rolling {flow_label} (GBE in k Bags)"), unsafe_allow_html=True)
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

with col_mm:
    st.markdown(lbl("Min / Max / Avg vs Latest (GBE in k Bags)"), unsafe_allow_html=True)
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

pivot_s, complete_s  = build_pivot(dff_disp)
complete_years_s     = sorted(complete_s[complete_s].index.tolist())
ref_band: dict = {}
if complete_years_s:
    last5_s  = complete_years_s[-5:]
    ref_s    = pivot_s.loc[last5_s, MONTH_ORDER]
    ref_band = {"max": ref_s.max(), "min": ref_s.min(), "avg": ref_s.mean(), "n": len(last5_s)}

col_sea, col_cum = st.columns(2)

with col_sea:
    st.markdown(lbl(f"Seasonal (GBE in k Bags) · Monthly {flow_label} by Crop Year"), unsafe_allow_html=True)
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
    st.markdown(lbl(f"Cumulative {flow_label} (GBE in k Bags) · by Crop Year"), unsafe_allow_html=True)
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
# ROW 4 — YTD Table  |  YTD Trend + YoY %
# =============================================================================
col_ytd_tbl, col_ytd_charts = st.columns([2, 3])

with col_ytd_tbl:
    st.markdown(lbl(f"YTD · Oct – {latest_common_label} (GBE in k Bags)"), unsafe_allow_html=True)
    tbl2 = ytd.copy()
    tbl2["YTD_FMT"] = tbl2["YTD_BAGS"].map(lambda x: f"{x:,.0f}")
    tbl2["YOY_FMT"] = tbl2["YOY_PCT"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    st.dataframe(
        tbl2[["CROP_YEAR","YTD_FMT","YOY_FMT"]].rename(columns={
            "CROP_YEAR": "Crop Year",
            "YTD_FMT":  f"YTD k Bags (Oct–{latest_common_label})",
            "YOY_FMT":  "YoY %",
        }),
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(tbl2.index) + 2), 900),
    )

with col_ytd_charts:
    st.markdown(lbl("YTD Trend + YoY % Change (GBE in k Bags)"), unsafe_allow_html=True)
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

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("TDM Trade Flow Dashboard  ·  ETG Softs  ·  k Bags = thousands of 60kg bags (GBE)")
