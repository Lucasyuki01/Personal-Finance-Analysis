from __future__ import annotations

from typing import Optional

import pandas as pd

from rules.schema import VALID_MATCH_FIELDS, VALID_MATCH_TYPES


def build_match_mask(
    df: pd.DataFrame,
    match_field: str,
    match_type: str,
    match_value: Optional[str],
) -> pd.Series:
    """Return a boolean mask for the specified rule."""
    field = str(match_field).strip().lower()
    if field not in VALID_MATCH_FIELDS:
        raise ValueError(f"Unsupported match_field: {match_field}")
    if field not in df.columns:
        raise ValueError(f"Match field '{match_field}' missing from canonical data.")

    rule_type = str(match_type).strip().lower()
    if rule_type not in VALID_MATCH_TYPES:
        raise ValueError(f"Unsupported match_type: {match_type}")

    value = "" if match_value is None else str(match_value)
    series = df[field].fillna("").astype(str)

    if rule_type == "equals":
        if value == "":
            return pd.Series(False, index=df.index)
        return series.eq(value)
    if rule_type == "contains":
        if value == "":
            return pd.Series(False, index=df.index)
        return series.str.contains(value, regex=False)
    if rule_type == "regex":
        if value == "":
            return pd.Series(False, index=df.index)
        return series.str.contains(value, regex=True, na=False)

    raise ValueError(f"Unsupported match_type: {match_type}")

