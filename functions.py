import pandas as pd
import streamlit as st
import csv
from sklearn.linear_model import LinearRegression
import numpy as np
from pathlib import Path

@st.cache_data(show_spinner="Loading data…")
def load_data(csv_path: str = "non.csv") -> pd.DataFrame:
    """Liest die Roh-CSV ein, beseitigt kaputte Anführungszeichen,
    wandelt 'Year' in int um und entfernt Spalten, die komplett leer sind."""
    df = pd.read_csv(
        Path(__file__).with_name(csv_path),
        engine="python",         
        sep=",",
        quoting=csv.QUOTE_NONE, 
        skipinitialspace=True,
    )

    df.columns = df.columns.str.strip().str.strip('"')

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    num_cols = df.columns.difference(["Country", "Region", "Year"])
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    empty_cols = [c for c in num_cols if df[c].notna().sum() == 0]
    return df.drop(columns=empty_cols)

def filter_data(df: pd.DataFrame, *, countries=None, years=None, regions=None) -> pd.DataFrame:
    """Filtert das DataFrame nach den drei Selektoren.
    Jeder Parameter darf None sein – dann wird nicht gefiltert."""
    mask = pd.Series(True, index=df.index)
    if countries is not None:
        mask &= df["Country"].isin(countries)
    if years is not None:
        mask &= df["Year"].isin(years)
    if regions:
        mask &= df["Region"].isin(regions)
    return df.loc[mask]


def make_pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(values=value, index="Year", columns="Region", aggfunc="mean")


def linear_forecast(df: pd.DataFrame, indicator: str, future_years: np.ndarray, region=None) -> pd.DataFrame:
    reg_df = df if region is None else df[df["Region"] == region]
    ts = reg_df.groupby("Year")[indicator].mean().reset_index().dropna()
    if len(ts) < 2:
        return pd.DataFrame()
    X, y = ts[["Year"]].values, ts[indicator].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(future_years.reshape(-1, 1))
    hist = pd.DataFrame({"Year": ts["Year"], "historical": ts[indicator], "prediction": np.nan})
    fut = pd.DataFrame({"Year": future_years, "historical": np.nan, "prediction": y_pred})
    return pd.concat([hist, fut], ignore_index=True)
 