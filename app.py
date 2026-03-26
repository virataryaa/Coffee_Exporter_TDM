"""
TDM Trade Flow Dashboard
========================
Run locally:   streamlit run app.py
Deploy:        push files/ folder to GitHub, set main file to files/app.py on Streamlit Cloud

Data expected at:  files/data/tdm_coffee.parquet  (copy from the TDM folder)
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
  .main  { background-color: #0e1117; }
  .block-container { padding-top: 0.4rem !important; padding-bottom: 1rem; }
  h1, h2, h3 { color: #e8d5b7; }

  /* ── Sticky top filter bar ──────────────────────────────────────────────── */
  section[data-testid="stMain"] div[data-testid="stMainBlockContainer"]
    > div > div > div > div:nth-child(1) {
    position: sticky !important;
    top: 0px !important;
    z-index: 999 !important;
    background-color: #0e1117 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #2a2d35 !important;
  }
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

_D   = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
_PAL = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

# ── TOP FILTER BAR (single row — sticky) ──────────────────────────────────────
fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([0.8, 2.2, 1.6, 1.6, 1.2, 2.8])

with fc1:
    commodity = st.selectbox("Commodity", list(COMMODITY_FILES.keys()))

data_path = Path(__file__).parent / COMMODITY_FILES[commodity]


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH"]].assign(DAY=1))
    df["BAGS"] = df["GBE"] * (1000 / 60)   # GBE tons → 60kg bags
    return df


if not data_path.exists():
    st.info(
        f"{commodity} data not available yet.  "
        f"Place the parquet file at: `{COMMODITY_FILES[commodity]}`"
    )
    st.stop()

df = load_data(str(data_path))

# Fill remaining filter columns (after data is loaded)
all_reporters = sorted(df["REPORTER"].unique())
with fc2:
    sel_reporters = st.multiselect("Reporter (Origin)", all_reporters, default=all_reporters)

all_tags = sorted(df["COMMODITY_TAG"].dropna().unique()) if "COMMODITY_TAG" in df.columns else []
with fc3:
    sel_tags = st.multiselect("Type", all_tags, default=all_tags) if all_tags else []

all_regions = sorted(df["REGION"].unique())
with fc4:
    sel_regions = st.multiselect("Destination", all_regions, default=all_regions)

with fc5:
    partner_q = st.text_input("Partner", placeholder="e.g. Germany")

all_cy = sorted(df["CROP_YEAR"].unique())
with fc6:
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

# ── Latest common month (global) ──────────────────────────────────────────────
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

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown("---")
st.title(f"TDM {commodity} Trade Flow Dashboard")
st.caption(
    f"Crop years {sel_cy_range[0]} – {sel_cy_range[1]}  |  "
    f"{dff_disp['REPORTER'].nunique()} reporters  |  "
    f"Latest month in view: **{latest_common_label}** ({latest_cy})  |  "
    f"All volumes in 60kg bags (GBE)"
)
st.divider()

if dff_disp.empty:
    st.warning("No data for the current selection.")
    st.stop()

# ── Helper: pivot builder (60kg bags) ─────────────────────────────────────────
def build_pivot(data: pd.DataFrame):
    grp = (
        data.groupby(["CROP_YEAR", "CROP_MONTH_NUM"])["BAGS"]
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
# SECTION 1 — Flow Heatmap
# =============================================================================
st.subheader(f"Flow Heatmap — GBE in 60kg Bags  |  Monthly Exports by Crop Year")
st.caption(
    f"Latest crop year ({latest_cy}) capped at **{latest_common_label}**. "
    f"Prior years show all available months."
)

disp = pivot[MONTH_ORDER].astype(float)
disp[disp == 0] = np.nan

styled = (
    disp.style
    .background_gradient(cmap="RdYlGn", axis=None)
    .format("{:,.0f}", na_rep="")
    .set_properties(**{"text-align": "center"})
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]},
    ])
)
st.dataframe(styled, use_container_width=True, height=min(600, 36 * len(disp) + 60))

st.divider()

# =============================================================================
# SECTION 2 — Full Year Total  |  Min / Max / Avg vs Latest  (side by side)
# =============================================================================
col_left, col_right = st.columns(2)

with col_left:
    if complete_years:
        st.subheader("Full Year Total — GBE in 60kg Bags  |  Complete Crop Years Only")
        total_df = (
            pivot.loc[complete_years, MONTH_ORDER]
            .sum(axis=1).reset_index()
            .rename(columns={0: "Bags"})
            .sort_values("CROP_YEAR")
        )
        fig_tot = go.Figure(go.Bar(
            x=total_df["CROP_YEAR"], y=total_df["Bags"],
            marker_color="#c8a96e",
            text=total_df["Bags"].map(lambda x: f"{x:,.0f}"),
            textposition="outside",
        ))
        fig_tot.update_layout(
            height=340, yaxis_title="60kg Bags", xaxis_title="Crop Year", **_D
        )
        st.plotly_chart(fig_tot, use_container_width=True)
    else:
        st.info("No complete crop years found in current selection.")

with col_right:
    st.subheader(f"Min / Max / Avg vs Latest — GBE in 60kg Bags")
    if complete_years:
        last5 = complete_years[-5:]
        ref   = pivot.loc[last5, MONTH_ORDER]
        mn, mx, avg = ref.min(), ref.max(), ref.mean()

        latest_x = MONTH_ORDER[:latest_common_num]
        if latest_cy in pivot.index:
            latest_y = [
                pivot.loc[latest_cy, m] if pivot.loc[latest_cy, m] > 0 else None
                for m in latest_x
            ]
        else:
            latest_y = [None] * len(latest_x)

        fig_mm = go.Figure()
        fig_mm.add_trace(go.Scatter(
            x=MONTH_ORDER, y=mx.values, name=f"Max (L{len(last5)}Y)",
            mode="lines", line=dict(color="green", width=2),
        ))
        fig_mm.add_trace(go.Scatter(
            x=MONTH_ORDER, y=mn.values, name=f"Min (L{len(last5)}Y)",
            mode="lines", line=dict(color="orange", width=2),
            fill="tonexty", fillcolor="rgba(150,150,150,0.10)",
        ))
        fig_mm.add_trace(go.Scatter(
            x=MONTH_ORDER, y=avg.values, name=f"Avg (L{len(last5)}Y)",
            mode="lines", line=dict(dash="dot", color="black", width=2),
        ))
        fig_mm.add_trace(go.Scatter(
            x=latest_x, y=latest_y, name=latest_cy,
            mode="lines+markers",
            line=dict(color="#e8d5b7", width=2.5), marker=dict(size=6),
        ))
        fig_mm.update_layout(
            height=340,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
            yaxis_title="60kg Bags",
            legend=dict(orientation="h", y=-0.28),
            title=f"Reference years: {', '.join(last5)}",
            **_D,
        )
        st.plotly_chart(fig_mm, use_container_width=True)
    else:
        st.info("No complete crop years for reference range.")

st.divider()

# =============================================================================
# SECTION 3 — Seasonal Analysis
# =============================================================================
sea = (
    dff_disp.groupby(["CROP_YEAR", "CROP_MONTH_NUM"])["BAGS"]
    .sum().reset_index().sort_values(["CROP_YEAR", "CROP_MONTH_NUM"])
)
sea["CROP_MONTH"] = sea["CROP_MONTH_NUM"].map(NUM_TO_MONTH)
all_sea_cy = sorted(sea["CROP_YEAR"].unique())

st.subheader("Seasonal Analysis — GBE in 60kg Bags  |  Monthly Exports by Crop Year")
sel_sea = st.multiselect(
    "Select crop years to display", all_sea_cy, default=all_sea_cy, key="sea_sel"
)

fig3 = go.Figure()
pivot_s, complete_s = build_pivot(dff_disp)
complete_years_s    = sorted(complete_s[complete_s].index.tolist())
if complete_years_s:
    last5_s = complete_years_s[-5:]
    ref_s   = pivot_s.loc[last5_s, MONTH_ORDER]
    fig3.add_trace(go.Scatter(
        x=MONTH_ORDER, y=ref_s.max().values, name=f"Max (L{len(last5_s)}Y)",
        mode="lines", line=dict(color="green", width=2),
    ))
    fig3.add_trace(go.Scatter(
        x=MONTH_ORDER, y=ref_s.min().values, name=f"Min (L{len(last5_s)}Y)",
        mode="lines", line=dict(color="orange", width=2),
        fill="tonexty", fillcolor="rgba(150,150,150,0.07)",
    ))
    fig3.add_trace(go.Scatter(
        x=MONTH_ORDER, y=ref_s.mean().values, name=f"Avg (L{len(last5_s)}Y)",
        mode="lines", line=dict(dash="dot", color="black", width=2),
    ))
for i, cy in enumerate(sorted(sel_sea)):
    d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM")
    fig3.add_trace(go.Scatter(
        x=d["CROP_MONTH"], y=d["BAGS"], name=cy,
        mode="lines+markers", line=dict(color=_PAL[i % len(_PAL)]),
    ))
fig3.update_layout(
    height=420,
    xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
    yaxis_title="60kg Bags",
    legend=dict(orientation="h", y=-0.2),
    **_D,
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 4 — Cumulative  |  Rolling  (side by side)
# =============================================================================
col_cum, col_roll = st.columns(2)

with col_cum:
    st.subheader("Cumulative Exports — GBE in 60kg Bags")
    sel_cum = st.multiselect(
        "Select crop years", all_sea_cy, default=all_sea_cy, key="cum_sel"
    )
    fig4 = go.Figure()
    for i, cy in enumerate(sorted(sel_cum)):
        d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM").copy()
        d["CUM_BAGS"] = d["BAGS"].cumsum()
        fig4.add_trace(go.Scatter(
            x=d["CROP_MONTH"], y=d["CUM_BAGS"], name=cy,
            mode="lines+markers", line=dict(color=_PAL[i % len(_PAL)]),
        ))
    fig4.update_layout(
        height=380,
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
        yaxis_title="Cumulative 60kg Bags",
        legend=dict(orientation="h", y=-0.28),
        **_D,
    )
    st.plotly_chart(fig4, use_container_width=True)

with col_roll:
    st.subheader("Rolling Exports — GBE in 60kg Bags")
    roll_choice = st.radio("Rolling window", ["1m", "3m", "6m", "12m"], index=3, horizontal=True)
    window = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}[roll_choice]
    monthly = (
        dff_disp.groupby("DATE")["BAGS"]
        .sum().reset_index().sort_values("DATE")
    )
    monthly["ROLLING"] = monthly["BAGS"].rolling(window).sum()
    fig5 = go.Figure(go.Scatter(
        x=monthly["DATE"], y=monthly["ROLLING"],
        name=f"Rolling {roll_choice}", mode="lines",
        line=dict(color="#c8a96e", width=2.5),
        fill="tozeroy", fillcolor="rgba(200,169,110,0.12)",
    ))
    fig5.update_layout(height=380, yaxis_title="60kg Bags", xaxis_title="Date", **_D)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 5 — YTD Analysis
# =============================================================================
st.subheader(f"YTD Analysis — GBE in 60kg Bags  |  Oct – {latest_common_label}")
st.info(
    f"YTD window: Oct → **{latest_common_label}** (crop month {latest_common_num} of 12) — "
    f"auto-detected from the latest common month across reporters in {latest_cy}."
)

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
    tbl[["CROP_YEAR", "YTD_FMT", "YOY_FMT"]].rename(columns={
        "CROP_YEAR": "Crop Year",
        "YTD_FMT":  f"YTD Bags (Oct–{latest_common_label})",
        "YOY_FMT":  "YoY Change",
    }),
    use_container_width=True,
    hide_index=True,
)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("YTD Trend — GBE in 60kg Bags")
    fig6 = go.Figure(go.Scatter(
        x=ytd["CROP_YEAR"], y=ytd["YTD_BAGS"],
        mode="lines+markers",
        line=dict(color="#c8a96e", width=2.5), marker=dict(size=7),
        name="YTD Bags",
    ))
    fig6.update_layout(
        height=340, xaxis_title="Crop Year", yaxis_title="60kg Bags", **_D
    )
    st.plotly_chart(fig6, use_container_width=True)

with col_b:
    st.subheader("YoY % Change — GBE in 60kg Bags")
    pct_df = ytd.dropna(subset=["YOY_PCT"]).copy()
    pct_df["COLOR"] = pct_df["YOY_PCT"].apply(lambda x: "#6eb5c8" if x >= 0 else "#e05c5c")
    fig7 = go.Figure(go.Bar(
        x=pct_df["CROP_YEAR"], y=pct_df["YOY_PCT"],
        marker_color=pct_df["COLOR"],
        text=pct_df["YOY_PCT"].map(lambda x: f"{x:+.1f}%"),
        textposition="outside",
    ))
    fig7.add_hline(y=0, line_color="white", line_width=1, opacity=0.35)
    fig7.update_layout(
        height=340, xaxis_title="Crop Year", yaxis_title="YoY %", **_D
    )
    st.plotly_chart(fig7, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("TDM Trade Flow Dashboard  |  Streamlit + Plotly  |  Unit: 60kg bags (GBE)")
