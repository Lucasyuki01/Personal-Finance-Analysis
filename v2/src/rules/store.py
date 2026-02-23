from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from rules.schema import CLASSIFICATION_RULE_COLUMNS, empty_rules_df, validate_rules_df


def default_rules_path() -> Path:
    """Return the default rules path inside v2/data/rules."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "rules" / "classification_rules.csv"


def load_rules(path: Optional[Path] = None) -> pd.DataFrame:
    """Load classification rules from CSV/Parquet."""
    rules_path = path or default_rules_path()
    if not rules_path.exists():
        return empty_rules_df()

    if rules_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(rules_path)
    else:
        df = pd.read_csv(rules_path)

    df = _coerce_types(df)
    validate_rules_df(df)
    return df


def save_rules(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save classification rules to CSV/Parquet."""
    rules_path = path or default_rules_path()
    rules_path.parent.mkdir(parents=True, exist_ok=True)

    df = _coerce_types(df)
    validate_rules_df(df)

    if rules_path.suffix.lower() == ".parquet":
        df.to_parquet(rules_path, index=False)
    else:
        df.to_csv(rules_path, index=False)

    return rules_path


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_cols = [col for col in CLASSIFICATION_RULE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "classification_rules missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    df["match_type"] = df["match_type"].astype(str).str.lower()
    df["match_field"] = df["match_field"].astype(str).str.lower()

    df["priority"] = pd.to_numeric(df["priority"], errors="raise").astype(int)

    df["is_active"] = _coerce_bool(df["is_active"], allow_empty=False, field="is_active")
    df["set_is_fixed_waste"] = _coerce_bool(
        df["set_is_fixed_waste"], allow_empty=True, field="set_is_fixed_waste"
    )

    return df


def _coerce_bool(series: pd.Series, allow_empty: bool, field: str) -> pd.Series:
    def parse(value):
        if pd.isna(value):
            if allow_empty:
                return pd.NA
            raise ValueError(f"classification_rules {field} cannot be empty.")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        raise ValueError(f"classification_rules {field} must be boolean.")

    return series.map(parse)

