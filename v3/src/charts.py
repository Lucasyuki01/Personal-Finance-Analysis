"""Plotly chart helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

GREEN_PIE = px.colors.sequential.Greens


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=300)
    return fig


def _month_series(df: pd.DataFrame) -> pd.Series:
    return df["Date"].dt.to_period("M").dt.to_timestamp()


def earnings_over_time(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No earnings data")
    data = df.copy()
    data["Month"] = _month_series(data)
    grouped = data.groupby("Month", as_index=False)["Amount"].sum()
    fig = px.bar(grouped, x="Month", y="Amount", title="Earnings Over Time")
    fig.update_layout(height=300)
    return fig


def earnings_by_category_pie(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No earnings data")
    grouped = df.groupby("Category", as_index=False)["Amount"].sum()
    fig = px.pie(grouped, names="Category", values="Amount", title="Earnings by Category", color_discrete_sequence=GREEN_PIE)
    fig.update_layout(height=300)
    return fig


def spendings_over_time(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No spendings data")
    data = df.copy()
    data["Month"] = _month_series(data)
    data["Spent"] = data["Amount"].abs()
    grouped = data.groupby("Month", as_index=False)["Spent"].sum()
    fig = px.bar(grouped, x="Month", y="Spent", title="Spendings Over Time")
    fig.update_layout(height=300)
    return fig


def spendings_by_category_pie(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No spendings data")
    data = df.copy()
    data["Spent"] = data["Amount"].abs()
    grouped = data.groupby("Category", as_index=False)["Spent"].sum()
    fig = px.pie(grouped, names="Category", values="Spent", title="Spendings by Category")
    fig.update_layout(height=300)
    return fig


def comparison_bar(earnings_df: pd.DataFrame, spendings_df: pd.DataFrame) -> go.Figure:
    if earnings_df.empty and spendings_df.empty:
        return _empty_fig("No comparison data")

    def _agg(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame({"Month": [], col: []})
        data = df.copy()
        data["Month"] = _month_series(data)
        return data.groupby("Month", as_index=False)[col].sum()

    earn = earnings_df.copy()
    earn = _agg(earn, "Amount")
    spend = spendings_df.copy()
    spend["Spent"] = spend["Amount"].abs() if not spend.empty else []
    spend = _agg(spend, "Spent")

    merged = pd.merge(earn, spend, on="Month", how="outer").fillna(0)
    fig = go.Figure()
    fig.add_bar(x=merged["Month"], y=merged["Amount"], name="Earnings")
    fig.add_bar(x=merged["Month"], y=merged["Spent"], name="Spendings")
    fig.update_layout(barmode="group", height=320, title="Earnings vs Spendings")
    return fig


def income_by_category_stacked(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No income data")
    data = df.copy()
    data["Month"] = _month_series(data)
    grouped = data.groupby(["Month", "Category"], as_index=False)["Amount"].sum()
    fig = px.bar(grouped, x="Month", y="Amount", color="Category", title="Income by Category")
    fig.update_layout(barmode="stack", height=320)
    return fig


def category_pie(df: pd.DataFrame, title: str) -> go.Figure:
    if df.empty:
        return _empty_fig("No data")
    grouped = df.groupby("Category", as_index=False)["Amount"].sum()
    fig = px.pie(grouped, names="Category", values="Amount", title=title)
    fig.update_layout(height=320)
    return fig


def category_pie_spent(df: pd.DataFrame, title: str) -> go.Figure:
    if df.empty:
        return _empty_fig("No data")
    data = df.copy()
    data["Spent"] = data["Amount"].abs()
    grouped = data.groupby("Category", as_index=False)["Spent"].sum()
    fig = px.pie(grouped, names="Category", values="Spent", title=title)
    fig.update_layout(height=320)
    return fig


def income_subcategory_pie(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No data")
    grouped = df.groupby("Sub-Category", as_index=False)["Amount"].sum()
    fig = px.pie(grouped, names="Sub-Category", values="Amount", title="Sub-Category Distribution")
    fig.update_layout(height=300)
    return fig


def expense_subcategory_pie(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No data")
    data = df.copy()
    data["Spent"] = data["Amount"].abs()
    grouped = data.groupby("Sub-Category", as_index=False)["Spent"].sum()
    fig = px.pie(grouped, names="Sub-Category", values="Spent", title="Sub-Category Distribution")
    fig.update_layout(height=300)
    return fig


def category_time_distribution(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    if df.empty:
        return _empty_fig("No data")
    data = df.copy()
    data["Month"] = _month_series(data)
    grouped = data.groupby("Month", as_index=False)[value_col].sum()
    fig = px.bar(grouped, x="Month", y=value_col, title=title)
    fig.update_layout(height=300)
    return fig


def spend_by_weekday_category(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No expense data")
    data = df.copy()
    data["Weekday"] = data["Date"].dt.day_name()
    data["Spent"] = data["Amount"].abs()

    daily = data.groupby(["Date", "Weekday", "Category"], as_index=False)["Spent"].sum()
    avg_daily = daily.groupby(["Weekday", "Category"], as_index=False)["Spent"].mean()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    avg_daily["Weekday"] = pd.Categorical(avg_daily["Weekday"], categories=weekday_order, ordered=True)
    avg_daily = avg_daily.sort_values("Weekday")

    fig = px.bar(
        avg_daily,
        x="Weekday",
        y="Spent",
        color="Category",
        title="Average Spend by Category per Weekday",
    )
    fig.update_layout(barmode="stack", height=320)
    return fig


def spend_over_time(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_fig("No expense data")
    data = df.copy()
    data["Month"] = _month_series(data)
    data["Spent"] = data["Amount"].abs()
    grouped = data.groupby(["Month", "Category"], as_index=False)["Spent"].sum()
    fig = px.bar(grouped, x="Month", y="Spent", color="Category", title="Spendings Over Time")
    fig.update_layout(barmode="stack", height=320)
    return fig
