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
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .block-container { padding-top: 1.5rem; }
  h1, h2, h3 { color: #e8d5b7; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTH_ORDER   = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
NUM_TO_MONTH  = {i + 1: m for i, m in enumerate(MONTH_ORDER)}
BAGS_PER_TON  = 1000 / 60   # 1 metric ton = 16.667 × 60kg bags

COMMODITY_FILES = {
    "Coffee": "data/tdm_coffee.parquet",
    "Cocoa":  "data/tdm_cocoa.parquet",
    "Sugar":  "data/tdm_sugar.parquet",
    "Cotton": "data/tdm_cotton.parquet",
}

_D   = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
_PAL = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel

# ── Sidebar: commodity picker ─────────────────────────────────────────────────
st.sidebar.title("TDM Trade Flows")
commodity = st.sidebar.selectbox("Commodity", list(COMMODITY_FILES.keys()))
data_path = Path(__file__).parent / COMMODITY_FILES[commodity]


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["DATE"]     = pd.to_datetime(df[["YEAR", "MONTH"]].assign(DAY=1))
    df["BAGS"]     = df["GBE"] * (1000 / 60)   # convert GBE tons → 60kg bags
    return df


if not data_path.exists():
    st.info(
        f"{commodity} data not available yet.  "
        f"Place the parquet file at: `{COMMODITY_FILES[commodity]}`"
    )
    st.stop()

df = load_data(str(data_path))

# ── Sidebar: filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

all_reporters = sorted(df["REPORTER"].unique())
sel_reporters = st.sidebar.multiselect("Reporter (Origin)", all_reporters, default=all_reporters)

all_tags = sorted(df["COMMODITY_TAG"].dropna().unique()) if "COMMODITY_TAG" in df.columns else []
sel_tags = st.sidebar.multiselect("Commodity Type", all_tags, default=all_tags) if all_tags else []

all_regions = sorted(df["REGION"].unique())
sel_regions = st.sidebar.multiselect("Destination Region", all_regions, default=all_regions)

partner_q = st.sidebar.text_input("Partner (search by name)", placeholder="e.g. Germany")

all_cy = sorted(df["CROP_YEAR"].unique())
sel_cy_range = st.sidebar.select_slider(
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

# ── Latest common month (applied globally to all visuals) ─────────────────────
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

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"TDM {commodity} Trade Flow Dashboard")
st.caption(
    f"Crop years {sel_cy_range[0]} to {sel_cy_range[1]}  |  "
    f"{dff_disp['REPORTER'].nunique()} reporters  |  "
    f"Latest month in view: **{latest_common_label}** ({latest_cy})  |  "
    f"All volumes in 60kg bags"
)

st.divider()


# ── Helper: pivot builder (60kg bags) ─────────────────────────────────────────
def build_pivot(data: pd.DataFrame):
    """Returns (pivot of BAGS sorted oldest-first, complete_mask Series)."""
    grp = (
        data.groupby(["CROP_YEAR", "CROP_MONTH_NUM"])["BAGS"]
        .sum()
        .reset_index()
    )
    grp["CROP_MONTH"] = grp["CROP_MONTH_NUM"].map(NUM_TO_MONTH)
    pivot = (
        grp.pivot(index="CROP_YEAR", columns="CROP_MONTH", values="BAGS")
        .reindex(columns=MONTH_ORDER)
        .fillna(0)
        .sort_index(ascending=True)   # oldest crop year on top, newest at bottom
    )
    complete = (pivot > 0).sum(axis=1) == 12
    return pivot, complete


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Flow Heatmap",
    "Seasonal Analysis",
    "Rolling",
    "YTD Analysis",
])

# =============================================================================
# TAB 1  —  Flow Heatmap
# =============================================================================
with tab1:
    if dff_disp.empty:
        st.warning("No data for the current selection.")
    else:
        pivot, complete = build_pivot(dff_disp)
        complete_years  = sorted(complete[complete].index.tolist())

        # ── Flow Heatmap
        st.subheader("Flow Heatmap")
        st.caption(
            f"Latest crop year ({latest_cy}) capped at **{latest_common_label}**. "
            f"Prior years show all available months.  |  Unit: 60kg bags"
        )

        disp = pivot[MONTH_ORDER].astype(float)
        disp[disp == 0] = np.nan

        styled = (
            disp
            .style
            .background_gradient(cmap="RdYlGn", axis=None)
            .format("{:,.0f}", na_rep="")
            .set_properties(**{"text-align": "center"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ])
        )
        st.dataframe(styled, use_container_width=True, height=min(700, 36 * len(disp) + 60))

        # ── Full Year Total
        if complete_years:
            st.markdown("---")
            st.subheader("Full Year Total  (complete crop years only)  |  60kg bags")
            total_df = (
                pivot.loc[complete_years, MONTH_ORDER]
                .sum(axis=1)
                .reset_index()
                .rename(columns={0: "Bags"})
                .sort_values("CROP_YEAR")
            )
            fig_tot = go.Figure(go.Bar(
                x=total_df["CROP_YEAR"],
                y=total_df["Bags"],
                marker_color="#c8a96e",
                text=total_df["Bags"].map(lambda x: f"{x:,.0f}"),
                textposition="outside",
            ))
            fig_tot.update_layout(height=300, yaxis_title="60kg Bags", xaxis_title="Crop Year", **_D)
            st.plotly_chart(fig_tot, use_container_width=True)

        # ── Min / Max / Avg vs Latest
        st.markdown("---")
        st.subheader("Min / Max / Avg vs Latest")

        if complete_years:
            last5 = complete_years[-5:]
            ref   = pivot.loc[last5, MONTH_ORDER]
            mn    = ref.min()
            mx    = ref.max()
            avg   = ref.mean()

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
                x=MONTH_ORDER, y=mx.values,
                name=f"Max (L{len(last5)}Y)",
                mode="lines",
                line=dict(color="green", width=2),
            ))
            fig_mm.add_trace(go.Scatter(
                x=MONTH_ORDER, y=mn.values,
                name=f"Min (L{len(last5)}Y)",
                mode="lines",
                line=dict(color="orange", width=2),
                fill="tonexty",
                fillcolor="rgba(150,150,150,0.10)",
            ))
            fig_mm.add_trace(go.Scatter(
                x=MONTH_ORDER, y=avg.values,
                name=f"Avg (L{len(last5)}Y)",
                mode="lines",
                line=dict(dash="dot", color="black", width=2),
            ))
            fig_mm.add_trace(go.Scatter(
                x=latest_x, y=latest_y,
                name=latest_cy,
                mode="lines+markers",
                line=dict(color="#e8d5b7", width=2.5),
                marker=dict(size=7),
            ))
            fig_mm.update_layout(
                height=400,
                xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
                yaxis_title="60kg Bags",
                legend=dict(orientation="h", y=-0.2),
                title=f"Reference years: {', '.join(last5)}",
                **_D,
            )
            st.plotly_chart(fig_mm, use_container_width=True)
        else:
            st.info("No complete crop years (all 12 months) found in the current selection.")

# =============================================================================
# TAB 2  —  Seasonal Analysis
# =============================================================================
with tab2:
    if dff_disp.empty:
        st.warning("No data for the current selection.")
    else:
        sea = (
            dff_disp.groupby(["CROP_YEAR", "CROP_MONTH_NUM"])["BAGS"]
            .sum()
            .reset_index()
            .sort_values(["CROP_YEAR", "CROP_MONTH_NUM"])
        )
        sea["CROP_MONTH"] = sea["CROP_MONTH_NUM"].map(NUM_TO_MONTH)
        all_sea_cy = sorted(sea["CROP_YEAR"].unique())

        # ── Seasonal line chart
        st.subheader("Seasonal — 60kg Bags by Crop Month")
        sel_sea = st.multiselect(
            "Select crop years", all_sea_cy, default=all_sea_cy, key="sea_sel"
        )

        fig3 = go.Figure()

        pivot_s, complete_s = build_pivot(dff_disp)
        complete_years_s    = sorted(complete_s[complete_s].index.tolist())
        if complete_years_s:
            last5_s = complete_years_s[-5:]
            ref_s   = pivot_s.loc[last5_s, MONTH_ORDER]
            fig3.add_trace(go.Scatter(
                x=MONTH_ORDER, y=ref_s.max().values,
                name=f"Max (L{len(last5_s)}Y)", mode="lines",
                line=dict(color="green", width=2),
            ))
            fig3.add_trace(go.Scatter(
                x=MONTH_ORDER, y=ref_s.min().values,
                name=f"Min (L{len(last5_s)}Y)", mode="lines",
                line=dict(color="orange", width=2),
                fill="tonexty", fillcolor="rgba(150,150,150,0.07)",
            ))
            fig3.add_trace(go.Scatter(
                x=MONTH_ORDER, y=ref_s.mean().values,
                name=f"Avg (L{len(last5_s)}Y)", mode="lines",
                line=dict(dash="dot", color="black", width=2),
            ))

        for i, cy in enumerate(sorted(sel_sea)):
            d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM")
            fig3.add_trace(go.Scatter(
                x=d["CROP_MONTH"], y=d["BAGS"],
                name=cy, mode="lines+markers",
                line=dict(color=_PAL[i % len(_PAL)]),
            ))

        fig3.update_layout(
            height=460,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
            yaxis_title="60kg Bags",
            legend=dict(orientation="h", y=-0.2),
            **_D,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Cumulative line chart
        st.divider()
        st.subheader("Cumulative 60kg Bags by Crop Month")
        sel_cum = st.multiselect(
            "Select crop years", all_sea_cy, default=all_sea_cy, key="cum_sel"
        )

        fig4 = go.Figure()
        for i, cy in enumerate(sorted(sel_cum)):
            d = sea[sea["CROP_YEAR"] == cy].sort_values("CROP_MONTH_NUM").copy()
            d["CUM_BAGS"] = d["BAGS"].cumsum()
            fig4.add_trace(go.Scatter(
                x=d["CROP_MONTH"], y=d["CUM_BAGS"],
                name=cy, mode="lines+markers",
                line=dict(color=_PAL[i % len(_PAL)]),
            ))

        fig4.update_layout(
            height=440,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
            yaxis_title="Cumulative 60kg Bags",
            legend=dict(orientation="h", y=-0.2),
            **_D,
        )
        st.plotly_chart(fig4, use_container_width=True)

# =============================================================================
# TAB 3  —  Rolling
# =============================================================================
with tab3:
    if dff_disp.empty:
        st.warning("No data for the current selection.")
    else:
        st.subheader("Rolling 60kg Bags")
        roll_choice = st.radio(
            "Rolling window", ["1m", "3m", "6m", "12m"], index=3, horizontal=True
        )
        window = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}[roll_choice]

        monthly = (
            dff_disp.groupby("DATE")["BAGS"]
            .sum()
            .reset_index()
            .sort_values("DATE")
        )
        monthly["ROLLING"] = monthly["BAGS"].rolling(window).sum()

        fig5 = go.Figure(go.Scatter(
            x=monthly["DATE"],
            y=monthly["ROLLING"],
            name=f"Rolling {roll_choice}",
            mode="lines",
            line=dict(color="#c8a96e", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(200,169,110,0.12)",
        ))
        fig5.update_layout(
            height=450,
            yaxis_title="60kg Bags",
            xaxis_title="Date",
            **_D,
        )
        st.plotly_chart(fig5, use_container_width=True)

# =============================================================================
# TAB 4  —  YTD Analysis
# =============================================================================
with tab4:
    if dff_disp.empty:
        st.warning("No data for the current selection.")
    else:
        ytd_cutoff_num = latest_common_num
        ytd_label      = latest_common_label

        st.info(
            f"YTD window: Oct to **{ytd_label}** (crop month {ytd_cutoff_num} of 12) — "
            f"auto-detected from the latest common month across reporters in {latest_cy}."
        )

        ytd = (
            dff_disp[dff_disp["CROP_MONTH_NUM"] <= ytd_cutoff_num]
            .groupby("CROP_YEAR")["BAGS"]
            .sum()
            .reset_index()
            .sort_values("CROP_YEAR")
            .rename(columns={"BAGS": "YTD_BAGS"})
        )
        ytd["YOY_PCT"] = ytd["YTD_BAGS"].pct_change() * 100

        # YTD Table
        st.subheader(f"YTD 60kg Bags by Crop Year  (Oct – {ytd_label})")
        tbl = ytd.copy()
        tbl["YTD_FMT"] = tbl["YTD_BAGS"].map(lambda x: f"{x:,.0f}")
        tbl["YOY_FMT"] = tbl["YOY_PCT"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
        st.dataframe(
            tbl[["CROP_YEAR", "YTD_FMT", "YOY_FMT"]].rename(columns={
                "CROP_YEAR": "Crop Year",
                "YTD_FMT":  f"YTD Bags (Oct–{ytd_label})",
                "YOY_FMT":  "YoY Change",
            }),
            use_container_width=True,
            hide_index=True,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("YTD Trend")
            fig6 = go.Figure(go.Scatter(
                x=ytd["CROP_YEAR"], y=ytd["YTD_BAGS"],
                mode="lines+markers",
                line=dict(color="#c8a96e", width=2.5),
                marker=dict(size=7),
                name="YTD Bags",
            ))
            fig6.update_layout(
                height=370, xaxis_title="Crop Year", yaxis_title="60kg Bags", **_D,
            )
            st.plotly_chart(fig6, use_container_width=True)

        with col_b:
            st.subheader("YoY % Change")
            pct_df = ytd.dropna(subset=["YOY_PCT"]).copy()
            pct_df["COLOR"] = pct_df["YOY_PCT"].apply(
                lambda x: "#6eb5c8" if x >= 0 else "#e05c5c"
            )
            fig7 = go.Figure(go.Bar(
                x=pct_df["CROP_YEAR"],
                y=pct_df["YOY_PCT"],
                marker_color=pct_df["COLOR"],
                text=pct_df["YOY_PCT"].map(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig7.add_hline(y=0, line_color="white", line_width=1, opacity=0.35)
            fig7.update_layout(
                height=370, xaxis_title="Crop Year", yaxis_title="YoY %", **_D,
            )
            st.plotly_chart(fig7, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("TDM Trade Flow Dashboard  |  Streamlit + Plotly  |  Unit: 60kg bags")
