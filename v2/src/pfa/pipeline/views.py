from __future__ import annotations

import pandas as pd


def view_profits(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows classified as profits."""
    flow_type = _lower_series(df, "flow_type")
    return df.loc[flow_type.eq("profit")].copy()


def view_wastes(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows classified as wastes."""
    flow_type = _lower_series(df, "flow_type")
    return df.loc[flow_type.eq("waste")].copy()


def view_card_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows tagged as card transactions."""
    channel = _lower_series(df, "channel")
    return df.loc[channel.eq("card")].copy()


def view_account_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows tagged as account expenses."""
    channel = _lower_series(df, "channel")
    amount_series = _amount_series(df)
    return df.loc[channel.eq("account") & amount_series.lt(0)].copy()


def view_fixed_wastes(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows tagged as fixed wastes."""
    if "is_fixed_waste" not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df["is_fixed_waste"].fillna(False).astype(bool)].copy()


def _lower_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index).astype(str)
    return df[column].fillna("").astype(str).str.strip().str.lower()


def _amount_series(df: pd.DataFrame) -> pd.Series:
    if "Amount" not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
