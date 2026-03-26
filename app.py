"""
TDM Trade Flow Dashboard
========================
Run locally:   streamlit run app.py
Deploy:        push files/ folder to GitHub, set main file to files/app.py on Streamlit Cloud

Data expected at:  files/data/tdm_coffee.parquet
"""

import numpy as np
import pandas as pd
import plotly.express as px
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
  /* ── Base ── */
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  .main { background-color: #fafafa !important; }
  [data-testid="stHeader"] { background: transparent !important; }
  .block-container { padding-top: 1.2rem !important; padding-bottom: 2rem; max-width: 1400px; }

  /* ── Typography ── */
  html, body, [class*="css"] { font-family: -apple-system, "Helvetica Neue", sans-serif; }
  h1, h2, h3, .stMarkdown h3 { color: #1d1d1f !important; font-weight: 500 !important; }

  /* ── Divider ── */
  hr { border: none !important; border-top: 1px solid #e8e8ed !important; margin: 1rem 0 !important; }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    border: 1px solid #e8e8ed !important;
    border-radius: 8px !important;
    background: #ffffff !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

  /* ── Metric / caption ── */
  .stCaption { color: #6e6e73 !important; font-size: 0.72rem !important; }

  /* ── Radio ── */
  [data-testid="stRadio"] label { font-size: 0.75rem !important; }
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

# Light theme plot defaults
_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, Helvetica Neue, sans-serif", color="#1d1d1f", size=11),
)
_PAL = px.colors.qualitative.Pastel + px.colors.qualitative.Set2

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
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1.5, 1.5, 1.5, 2.5])

        all_reporters = sorted(df["REPORTER"].unique())
        with fc1:
            sel_reporters = st.multiselect("Reporter", all_reporters, default=all_reporters)

        all_tags = sorted(df["COMMODITY_TAG"].dropna().unique()) if "COMMODITY_TAG" in df.columns else []
        with fc2:
            sel_tags = st.multiselect("Type", all_tags, default=all_tags) if all_tags else []

        all_regions = sorted(df["REGION"].unique())
        with fc3:
            sel_regions = st.multiselect("Destination", all_regions, default=all_regions)

        with fc4:
            partner_q = st.text_input("Partner", placeholder="e.g. Germany")

        all_cy = sorted(df["CROP_YEAR"].unique())
        with fc5:
            sel_cy_range = st.select_slider(
                "Crop Year Range", options=all_cy, value=(all_cy[0], all_cy[-1])
            )

# ── Apply filters ─────────────────────────────────────────────────────────────
i0, i1    = all_cy.index(sel_cy_range[0]), all_cy.index(sel_cy_range[1])
cy_window = all_cy[i0: i1 + 1]

mask = (
    df["CROP_YEAR"].isin(cy_window)
    & df["REPORTER"].isin(sel_reporters or all_reporters)
    & df["REGION"].isin(sel_regions or all_regions)
)
if all_tags:
    mask &= df["COMMODITY_TAG"].isin(sel_tags or all_tags)
if partner_q:
    mask &= df["PARTNER"].str.contains(partner_q, case=False, na=False)

dff = df[mask].copy()

# ── Latest common month ────────────────────────────────────────────────────────
if not dff.empty:
    latest_cy           = sorted(dff["CROP_YEAR"].unique())[-1]
    lm_per_reporter     = (
        dff[dff["CROP_YEAR"] == latest_cy]
        .groupby("REPORTER")["CROP_MONTH_NUM"].max()
    )
    latest_common_num   = int(lm_per_reporter.min()) if len(lm_per_reporter) else 12
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

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"### {commodity} Export Trade Flows &nbsp; <span style='font-size:0.85rem;font-weight:400;color:#6e6e73'>GBE · 60kg Bags</span>", unsafe_allow_html=True)
st.caption(
    f"Crop years {sel_cy_range[0]} – {sel_cy_range[1]}  ·  "
    f"{dff_disp['REPORTER'].nunique()} reporters  ·  "
    f"Latest month in view: {latest_common_label} ({latest_cy})"
)
st.markdown("<hr>", unsafe_allow_html=True)

if dff_disp.empty:
    st.warning("No data for the current selection.")
    st.stop()

# ── Helper: pivot ─────────────────────────────────────────────────────────────
def build_pivot(data: pd.DataFrame):
    grp = (
        data.groupby(["CROP_YEAR","CROP_MONTH_NUM"])["BAGS"]
        .sum().reset_index()
    )
    grp["CROP_MONTH"] = grp["CROP_MONTH_NUM"].map(NUM_TO_MONTH)
    pivot = (
        grp.pivot(index="CROP_YEAR", columns="CROP_MONTH", values="BAGS")
        .reindex(columns=MONTH_ORDER)
        .fillna(0)
        .sort_index(ascending=True)
    )
    complete = (pivot > 0).sum(axis=1) == 12
    return pivot, complete


pivot, complete = build_pivot(dff_disp)
complete_years  = sorted(complete[complete].index.tolist())

# =============================================================================
# ROW 1 — Heatmap  |  Full Year Total
# =============================================================================
col_hm, col_total = st.columns([3, 1])

with col_hm:
    st.markdown("<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>Flow Heatmap — Monthly Exports by Crop Year (60kg Bags)</p>", unsafe_allow_html=True)
    st.caption(f"Latest crop year ({latest_cy}) capped at {latest_common_label} · prior years show all months")

    disp = pivot[MONTH_ORDER].astype(float)
    disp[disp == 0] = np.nan
    styled = (
        disp.style
        .background_gradient(cmap="RdYlGn", axis=None)
        .format("{:,.0f}", na_rep="")
        .set_properties(**{"text-align": "center", "font-size": "11px"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align","center"),("font-size","11px"),("font-weight","500")]},
            {"selector": "td", "props": [("text-align","center")]},
        ])
    )
    tbl_h = min(560, 34 * len(disp) + 55)
    st.dataframe(styled, use_container_width=True, height=tbl_h)

with col_total:
    st.markdown("<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>Full Year Total</p>", unsafe_allow_html=True)
    st.caption("Complete crop years · 60kg Bags")
    if complete_years:
        total_df = (
            pivot.loc[complete_years, MONTH_ORDER]
            .sum(axis=1).reset_index()
            .rename(columns={0: "Bags"})
            .sort_values("CROP_YEAR")
        )
        fig_tot = go.Figure(go.Bar(
            x=total_df["Bags"],
            y=total_df["CROP_YEAR"],
            orientation="h",
            marker_color="#a0c4a0",
            text=total_df["Bags"].map(lambda x: f"{x/1e6:.1f}M"),
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_tot.update_layout(
            height=tbl_h,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
            margin=dict(l=5, r=35, t=5, b=5),
            **_D,
        )
        st.plotly_chart(fig_tot, use_container_width=True)
    else:
        st.info("No complete crop years.")

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ROW 2 — Min/Max/Avg vs Latest  |  YTD Table
# =============================================================================
col_mm, col_ytd = st.columns([3, 2])

with col_mm:
    st.markdown("<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>Min / Max / Avg vs Latest Crop Year (60kg Bags)</p>", unsafe_allow_html=True)
    if complete_years:
        last5    = complete_years[-5:]
        ref      = pivot.loc[last5, MONTH_ORDER]
        mn, mx, avg = ref.min(), ref.max(), ref.mean()
        latest_x = MONTH_ORDER[:latest_common_num]
        latest_y = [
            pivot.loc[latest_cy, m] if latest_cy in pivot.index and pivot.loc[latest_cy, m] > 0 else None
            for m in latest_x
        ]
        fig_mm = go.Figure()
        fig_mm.add_trace(go.Scatter(x=MONTH_ORDER, y=mx.values, name=f"Max (L{len(last5)}Y)", mode="lines", line=dict(color="#5a9e6f", width=1.5)))
        fig_mm.add_trace(go.Scatter(x=MONTH_ORDER, y=mn.values, name=f"Min (L{len(last5)}Y)", mode="lines", line=dict(color="#e07b39", width=1.5), fill="tonexty", fillcolor="rgba(180,180,180,0.12)"))
        fig_mm.add_trace(go.Scatter(x=MONTH_ORDER, y=avg.values, name=f"Avg (L{len(last5)}Y)", mode="lines", line=dict(dash="dot", color="#999999", width=1.5)))
        fig_mm.add_trace(go.Scatter(x=latest_x, y=latest_y, name=latest_cy, mode="lines+markers", line=dict(color="#1d1d1f", width=2), marker=dict(size=5)))
        fig_mm.update_layout(
            height=300,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
            margin=dict(t=10, b=60, l=10, r=10),
            **_D,
        )
        st.plotly_chart(fig_mm, use_container_width=True)
    else:
        st.info("No complete crop years for reference.")

with col_ytd:
    st.markdown(f"<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>YTD · Oct – {latest_common_label} (60kg Bags)</p>", unsafe_allow_html=True)
    ytd = (
        dff_disp[dff_disp["CROP_MONTH_NUM"] <= latest_common_num]
        .groupby("CROP_YEAR")["BAGS"].sum().reset_index()
        .sort_values("CROP_YEAR").rename(columns={"BAGS": "YTD_BAGS"})
    )
    ytd["YOY_PCT"] = ytd["YTD_BAGS"].pct_change() * 100
    tbl = ytd.copy()
    tbl["YTD_FMT"] = tbl["YTD_BAGS"].map(lambda x: f"{x:,.0f}")
    tbl["YOY_FMT"] = tbl["YOY_PCT"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    st.dataframe(
        tbl[["CROP_YEAR","YTD_FMT","YOY_FMT"]].rename(columns={
            "CROP_YEAR": "Crop Year",
            "YTD_FMT":  f"YTD Bags (Oct–{latest_common_label})",
            "YOY_FMT":  "YoY %",
        }),
        use_container_width=True,
        hide_index=True,
        height=300,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ROW 3 — Seasonal  |  Cumulative
# =============================================================================
sea = (
    dff_disp.groupby(["CROP_YEAR","CROP_MONTH_NUM"])["BAGS"]
    .sum().reset_index().sort_values(["CROP_YEAR","CROP_MONTH_NUM"])
)
sea["CROP_MONTH"] = sea["CROP_MONTH_NUM"].map(NUM_TO_MONTH)
all_sea_cy = sorted(sea["CROP_YEAR"].unique())
default_sea = all_sea_cy[-6:] if len(all_sea_cy) >= 6 else all_sea_cy

col_sea, col_cum = st.columns(2)

with col_sea:
    st.markdown("<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>Seasonal — Monthly Exports by Crop Year (60kg Bags)</p>", unsafe_allow_html=True)
    sel_sea = st.multiselect("Crop years", all_sea_cy, default=default_sea, key="sea_sel")
    fig3 = go.Figure()
    pivot_s, complete_s = build_pivot(dff_disp)
    complete_years_s    = sorted(complete_s[complete_s].index.tolist())
    if complete_years_s:
        last5_s = complete_years_s[-5:]
        ref_s   = pivot_s.loc[last5_s, MONTH_ORDER]
        fig3.add_trace(go.Scatter(x=MONTH_ORDER, y=ref_s.max().values, name=f"Max (L{len(last5_s)}Y)", mode="lines", line=dict(color="#5a9e6f", width=1.5)))
        fig3.add_trace(go.Scatter(x=MONTH_ORDER, y=ref_s.min().values, name=f"Min (L{len(last5_s)}Y)", mode="lines", line=dict(color="#e07b39", width=1.5), fill="tonexty", fillcolor="rgba(180,180,180,0.10)"))
        fig3.add_trace(go.Scatter(x=MONTH_ORDER, y=ref_s.mean().values, name=f"Avg (L{len(last5_s)}Y)", mode="lines", line=dict(dash="dot", color="#999999", width=1.5)))
    for i, cy in enumerate(sorted(sel_sea)):
        d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM")
        fig3.add_trace(go.Scatter(x=d["CROP_MONTH"], y=d["BAGS"], name=cy, mode="lines+markers", line=dict(color=_PAL[i % len(_PAL)]), marker=dict(size=4)))
    fig3.update_layout(
        height=320,
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
        margin=dict(t=10, b=60, l=10, r=10),
        **_D,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_cum:
    st.markdown("<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>Cumulative Exports by Crop Year (60kg Bags)</p>", unsafe_allow_html=True)
    sel_cum = st.multiselect("Crop years", all_sea_cy, default=default_sea, key="cum_sel")
    fig4 = go.Figure()
    for i, cy in enumerate(sorted(sel_cum)):
        d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM").copy()
        d["CUM_BAGS"] = d["BAGS"].cumsum()
        fig4.add_trace(go.Scatter(x=d["CROP_MONTH"], y=d["CUM_BAGS"], name=cy, mode="lines+markers", line=dict(color=_PAL[i % len(_PAL)]), marker=dict(size=4)))
    fig4.update_layout(
        height=320,
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
        margin=dict(t=10, b=60, l=10, r=10),
        **_D,
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =============================================================================
# ROW 4 — Rolling  |  YTD Trend + YoY
# =============================================================================
col_roll, col_ytd_charts = st.columns(2)

with col_roll:
    st.markdown("<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>Rolling Exports (60kg Bags)</p>", unsafe_allow_html=True)
    roll_choice = st.radio("Window", ["1m","3m","6m","12m"], index=3, horizontal=True)
    window = {"1m":1,"3m":3,"6m":6,"12m":12}[roll_choice]
    monthly = dff_disp.groupby("DATE")["BAGS"].sum().reset_index().sort_values("DATE")
    monthly["ROLLING"] = monthly["BAGS"].rolling(window).sum()
    fig5 = go.Figure(go.Scatter(
        x=monthly["DATE"], y=monthly["ROLLING"],
        mode="lines", line=dict(color="#4a7fb5", width=2),
        fill="tozeroy", fillcolor="rgba(74,127,181,0.08)",
    ))
    fig5.update_layout(
        height=300,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(t=10, b=10, l=10, r=10),
        **_D,
    )
    st.plotly_chart(fig5, use_container_width=True)

with col_ytd_charts:
    st.markdown("<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#6e6e73;margin-bottom:4px'>YTD Trend + YoY % Change (60kg Bags)</p>", unsafe_allow_html=True)
    ya, yb = st.columns(2)
    with ya:
        fig6 = go.Figure(go.Scatter(
            x=ytd["CROP_YEAR"], y=ytd["YTD_BAGS"],
            mode="lines+markers", line=dict(color="#4a7fb5", width=2), marker=dict(size=5),
        ))
        fig6.update_layout(
            height=300,
            xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            margin=dict(t=10, b=10, l=10, r=10),
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
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig7.add_hline(y=0, line_color="#cccccc", line_width=1)
        fig7.update_layout(
            height=300,
            xaxis=dict(showgrid=False, tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            margin=dict(t=10, b=10, l=10, r=10),
            **_D,
        )
        st.plotly_chart(fig7, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("TDM Trade Flow Dashboard  ·  ETG Softs  ·  60kg bags (GBE)")
