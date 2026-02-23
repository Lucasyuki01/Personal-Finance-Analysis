"""Processing pipeline for uploaded transaction files."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import secrets

import pandas as pd

from .rules import apply_pos_rules, apply_specific_rules

REQUIRED_COLUMNS = ["Date", "Amount", "Description"]


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return pd.isna(value)
    except Exception:
        return False


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip().str.lower()


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def normalize_file(df: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.copy()

    if "filter" in df.columns:
        df = df.drop(columns=["filter"])

    if "Source" not in df.columns:
        df["Source"] = filename
    else:
        df.loc[df["Source"].isna() | (df["Source"].astype("string").str.strip() == ""), "Source"] = filename

    if "Category" not in df.columns:
        df["Category"] = "none"
    df["Category"] = df["Category"].astype("string").fillna("none")
    df.loc[df["Category"].str.strip() == "", "Category"] = "none"

    if "Sub-Category" not in df.columns:
        df["Sub-Category"] = "none"
    df["Sub-Category"] = df["Sub-Category"].astype("string").fillna("none")
    df.loc[df["Sub-Category"].str.strip() == "", "Sub-Category"] = "none"

    if "Sub-description" not in df.columns:
        df["Sub-description"] = ""
    df["Sub-description"] = df["Sub-description"].astype("string").fillna("")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df["Description"] = df["Description"].astype("string").fillna("").str.strip()
    df["Sub-description"] = df["Sub-description"].astype("string").fillna("").str.strip()

    df["description_norm"] = _normalize_text_series(df["Description"])
    df["sub_description_norm"] = _normalize_text_series(df["Sub-description"])

    stats = {
        "invalid_date": int(df["Date"].isna().sum()),
        "invalid_amount": int(df["Amount"].isna().sum()),
    }

    return df, stats


def _generate_hex_id() -> str:
    return secrets.token_hex(3)


def ensure_id_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ID" not in df.columns:
        df["ID"] = ""

    df["ID"] = df["ID"].astype("string").fillna("")

    # Fill missing IDs
    missing_mask = df["ID"].str.strip() == ""
    if missing_mask.any():
        for idx in df.index[missing_mask]:
            new_id = _generate_hex_id()
            while new_id in set(df["ID"]):
                new_id = _generate_hex_id()
            df.at[idx, "ID"] = new_id

    # Ensure uniqueness for duplicates
    duplicates = df["ID"].duplicated(keep="first")
    if duplicates.any():
        for idx in df.index[duplicates]:
            new_id = _generate_hex_id()
            while new_id in set(df["ID"]):
                new_id = _generate_hex_id()
            df.at[idx, "ID"] = new_id

    return df


def process_uploaded_files(
    uploaded_files: Iterable,
    pos_rules: Dict,
    specific_rules: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    frames: List[pd.DataFrame] = []
    per_file_stats: List[Dict] = []

    for uploaded_file in uploaded_files:
        df_raw = load_uploaded_file(uploaded_file)
        df_norm, stats = normalize_file(df_raw, uploaded_file.name)
        per_file_stats.append({"file": uploaded_file.name, **stats})
        frames.append(df_norm)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    invalid_mask = merged["Date"].isna() | merged["Amount"].isna()
    invalid_rows = int(invalid_mask.sum())
    if invalid_rows:
        merged = merged.loc[~invalid_mask].copy()

    merged["Profit"] = (merged["Amount"] > 0).astype(int)

    drop_mask = merged["description_norm"].isin(["customer transfer cr.", "customer transfer dr."])
    if drop_mask.any():
        merged = merged.loc[~drop_mask].copy()

    merged = ensure_id_unique(merged)

    merged = apply_pos_rules(merged, pos_rules)
    merged = apply_specific_rules(merged, specific_rules)

    meta = {
        "per_file": per_file_stats,
        "invalid_rows_dropped": invalid_rows,
    }

    return merged, meta
