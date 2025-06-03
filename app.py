import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from translations import translations
import functions as fn 

st.set_page_config(page_title="💧 Water Quality Explorer", layout="wide")

lang_idx_default = 0 if st.session_state.get("language", "Deutsch") == "Deutsch" else 1
language = st.sidebar.selectbox(
    "Wähle die Sprache / Selectează limba:", ["Deutsch", "Română"], index=lang_idx_default
)
st.session_state["language"] = language
text = translations[language]

st.title(text["title"])

data = fn.load_data()

all_countries = sorted(data["Country"].dropna().unique())
all_regions = sorted(data["Region"].dropna().unique()) if "Region" in data.columns else []
min_year, max_year = int(data["Year"].min()), int(data["Year"].max())

mem = st.session_state.get(
    "filters",
    dict(
        countries=all_countries,
        regions=all_regions,
        year_range=(dt.date(min_year, 1, 1), dt.date(max_year, 12, 31)),
    ),
)

with st.sidebar.form(key="filter_form"):
    st.header(text["filter_header"])
    countries = st.multiselect(text["select_countries"], all_countries, mem["countries"])
    regions = (
        st.multiselect(text["select_region"], all_regions, mem["regions"])
        if all_regions
        else []
    )
    year_range = st.date_input(
        text["select_years"],
        value=mem["year_range"],
        min_value=dt.date(min_year, 1, 1),
        max_value=dt.date(max_year, 12, 31),
    )
    submitted = st.form_submit_button(text["apply_filters"])

if submitted or "filters" not in st.session_state:
    st.session_state["filters"] = dict(
        countries=countries or None,
        regions=regions or None,
        year_range=year_range,
    )

filt = st.session_state["filters"]

if isinstance(filt["year_range"], tuple):
    start_year, end_year = filt["year_range"][0].year, filt["year_range"][1].year
else:
    start_year = end_year = filt["year_range"].year
sel_years = list(range(start_year, end_year + 1))

filtered = fn.filter_data(
    data,
    countries=filt["countries"],
    years=sel_years,
    regions=filt["regions"],
)

st.subheader(text["filtered_data"])
st.dataframe(filtered, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if "Contaminant Level (ppm)" in filtered.columns:
        st.markdown(f"**{text['contaminant_level']}**")
        pivot_c = fn.make_pivot(filtered, "Contaminant Level (ppm)").reset_index()
        pivot_long = pivot_c.melt("Year", var_name="Region", value_name="Value")
        fig_c = px.line(
            pivot_long,
            x="Year",
            y="Value",
            color="Region",
            markers=True,
            labels={"Value": text["contaminant_level"], "Year": text["year"]},
        )
        st.plotly_chart(fig_c, use_container_width=True)

with col2:
    if "Access to Clean Water (% of Population)" in filtered.columns:
        st.markdown(f"**{text['clean_water_access']}**")
        pivot_a = fn.make_pivot(filtered, "Access to Clean Water (% of Population)").reset_index()
        pivot_long_a = pivot_a.melt("Year", var_name="Region", value_name="Value")
        fig_a = px.line(
            pivot_long_a,
            x="Year",
            y="Value",
            color="Region",
            markers=True,
            labels={"Value": text["clean_water_access"], "Year": text["year"]},
        )
        st.plotly_chart(fig_a, use_container_width=True)

st.subheader(text["forecast"])
indicator_cols = [
    c
    for c in filtered.columns
    if c not in ["Country", "Year", "Region"]
    and pd.api.types.is_numeric_dtype(filtered[c])
    and filtered[c].notna().any()
]
if not indicator_cols:
    st.info("Keine numerischen Daten für Prognose verfügbar.")
    st.stop()

indicator = st.selectbox(text["forecast_indicator"], indicator_cols)
future_years = np.arange(filtered["Year"].max() + 1, 2031)

fig_fore = go.Figure()
for region in (filt["regions"] or [None]):
    forecast_df = fn.linear_forecast(filtered, indicator, future_years, region)
    if forecast_df.empty:
        continue
    lbl = region if region else "All"
    hist_df = forecast_df.dropna(subset=["historical"])
    pred_df = forecast_df.dropna(subset=["prediction"])
    fig_fore.add_trace(
        go.Scatter(
            x=hist_df["Year"],
            y=hist_df["historical"],
            mode="lines+markers",
            name=f"{lbl} - {text['historical']}",
        )
    )
    fig_fore.add_trace(
        go.Scatter(
            x=pred_df["Year"],
            y=pred_df["prediction"],
            mode="lines",
            line=dict(dash="dash"),
            name=f"{lbl} - {text['prediction']}",
        )
    )

fig_fore.update_layout(
    title=f"{text['forecast']} '{indicator}'",
    xaxis_title=text["year"],
    yaxis_title=indicator,
)
st.plotly_chart(fig_fore, use_container_width=True)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("CSV", csv, "filtered.csv", "text/csv")

 