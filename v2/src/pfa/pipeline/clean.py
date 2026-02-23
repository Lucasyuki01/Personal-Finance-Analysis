from typing import Dict

import pandas as pd


def compute_missing_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return missing value counts and percentages per column."""
    if df.empty:
        return {}

    total = len(df)
    missing_counts = df.isna().sum()
    stats = {}
    for col, count in missing_counts.items():
        stats[col] = {
            "missing_count": int(count),
            "missing_pct": float(count) / total if total else 0.0,
        }
    return stats


def initial_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop optional columns and add description_norm."""
    df = df.copy()

    if "Filter" in df.columns:
        df = df.drop(columns=["Filter"])

    description = df.get("Description")
    if description is None:
        df["Description"] = ""
        description = df["Description"]

    df["description_norm"] = (
        description.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return df

