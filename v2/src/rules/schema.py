from __future__ import annotations

from typing import List, Sequence, Set

import numpy as np
import pandas as pd


CLASSIFICATION_RULE_COLUMNS: List[str] = [
    "rule_id",
    "priority",
    "is_active",
    "scope_source_account",
    "match_field",
    "match_type",
    "match_value",
    "set_flow_type",
    "set_class",
    "set_sub_class",
    "set_is_fixed_waste",
    "created_at",
    "updated_at",
    "notes",
]

VALID_MATCH_TYPES: Set[str] = {"equals", "contains", "regex"}
VALID_MATCH_FIELDS: Set[str] = {"description_norm"}
BOOLEAN_FIELDS: Set[str] = {"is_active", "set_is_fixed_waste"}


def empty_rules_df() -> pd.DataFrame:
    """Return an empty classification_rules DataFrame with the canonical columns."""
    return pd.DataFrame(columns=CLASSIFICATION_RULE_COLUMNS)


def validate_rules_df(df: pd.DataFrame) -> None:
    """Validate the classification_rules schema and field types."""
    missing = [col for col in CLASSIFICATION_RULE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "classification_rules missing required columns: "
            + ", ".join(sorted(missing))
        )

    _validate_match_types(df["match_type"])
    _validate_match_fields(df["match_field"])
    _validate_priorities(df["priority"])
    _validate_boolean_field(df["is_active"], allow_empty=False, field="is_active")
    _validate_boolean_field(
        df["set_is_fixed_waste"], allow_empty=True, field="set_is_fixed_waste"
    )


def _validate_match_types(series: pd.Series) -> None:
    values = series.dropna().astype(str).str.lower()
    invalid = sorted(set(values.unique()) - VALID_MATCH_TYPES)
    if invalid:
        raise ValueError(
            "classification_rules has invalid match_type values: "
            + ", ".join(invalid)
        )


def _validate_match_fields(series: pd.Series) -> None:
    values = series.dropna().astype(str).str.lower()
    invalid = sorted(set(values.unique()) - VALID_MATCH_FIELDS)
    if invalid:
        raise ValueError(
            "classification_rules has unsupported match_field values: "
            + ", ".join(invalid)
        )


def _validate_priorities(series: pd.Series) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("classification_rules priority must be integer values.")
    fractional = numeric % 1 != 0
    if fractional.any():
        raise ValueError("classification_rules priority must be integer values.")


def _validate_boolean_field(series: pd.Series, allow_empty: bool, field: str) -> None:
    values = series
    if not allow_empty and values.isna().any():
        raise ValueError(f"classification_rules {field} cannot be empty.")

    invalid_mask = []
    for value in values:
        if pd.isna(value):
            invalid_mask.append(False if allow_empty else True)
            continue
        if isinstance(value, (bool, np.bool_)):
            invalid_mask.append(False)
            continue
        invalid_mask.append(True)

    if any(invalid_mask):
        raise ValueError(f"classification_rules {field} must be boolean.")

