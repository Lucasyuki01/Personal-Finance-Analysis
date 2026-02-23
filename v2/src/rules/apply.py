from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from pfa.constants import (
    DEFAULT_CLASS,
    DEFAULT_CLASSIFICATION_SOURCE,
    DEFAULT_FLOW_TYPE,
    DEFAULT_SUB_CLASS,
)
from rules.matcher import build_match_mask
from rules.schema import validate_rules_df


def apply_rules(
    canonical_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    overwrite_manual: bool = False,
) -> pd.DataFrame:
    """Apply classification rules to a copy of the canonical dataset."""
    if canonical_df.empty:
        return canonical_df.copy()
    if rules_df is None or rules_df.empty:
        return canonical_df.copy()

    rules_df = rules_df.copy()
    validate_rules_df(rules_df)

    df = canonical_df.copy()
    _ensure_required_columns(df)

    rules_active = rules_df[rules_df["is_active"]].copy()
    if rules_active.empty:
        return df

    rules_active = rules_active.sort_values(["priority", "rule_id"])

    excluded_mask = df["excluded_reason"].fillna("").astype(str).str.strip().ne("")
    base_mask = ~excluded_mask

    original_source = (
        df["classification_source"]
        .fillna(DEFAULT_CLASSIFICATION_SOURCE)
        .astype(str)
        .str.lower()
    )
    source_allows = original_source.isin(["unclassified", "rules"])

    applied_by_field = {
        "flow_type": pd.Series(False, index=df.index),
        "class": pd.Series(False, index=df.index),
        "sub_class": pd.Series(False, index=df.index),
        "is_fixed_waste": pd.Series(False, index=df.index),
    }

    for _, rule in rules_active.iterrows():
        mask = base_mask.copy()
        scope = str(rule.get("scope_source_account", "") or "").strip()
        if scope:
            mask &= df["source_account"].fillna("").astype(str).eq(scope)

        mask &= build_match_mask(
            df,
            match_field=rule["match_field"],
            match_type=rule["match_type"],
            match_value=rule["match_value"],
        )

        if not mask.any():
            continue

        rule_applied = pd.Series(False, index=df.index)
        rule_applied = _apply_rule_fields(
            df,
            rule,
            mask,
            source_allows=source_allows,
            overwrite_manual=overwrite_manual,
            rule_applied=rule_applied,
            applied_by_field=applied_by_field,
        )

        if rule_applied.any():
            df.loc[rule_applied, "classification_source"] = "rules"
            if "applied_rule_id" not in df.columns:
                df["applied_rule_id"] = pd.NA
            df.loc[rule_applied, "applied_rule_id"] = rule["rule_id"]

    return df


def _ensure_required_columns(df: pd.DataFrame) -> None:
    if "classification_source" not in df.columns:
        df["classification_source"] = DEFAULT_CLASSIFICATION_SOURCE
    if "excluded_reason" not in df.columns:
        df["excluded_reason"] = ""
    if "flow_type" not in df.columns:
        df["flow_type"] = DEFAULT_FLOW_TYPE
    if "class" not in df.columns:
        df["class"] = DEFAULT_CLASS
    if "sub_class" not in df.columns:
        df["sub_class"] = DEFAULT_SUB_CLASS
    if "is_fixed_waste" not in df.columns:
        df["is_fixed_waste"] = False
    if "source_account" not in df.columns:
        df["source_account"] = ""


def _apply_rule_fields(
    df: pd.DataFrame,
    rule: pd.Series,
    mask: pd.Series,
    source_allows: pd.Series,
    overwrite_manual: bool,
    rule_applied: pd.Series,
    applied_by_field: Dict[str, pd.Series],
) -> pd.Series:
    updates: Dict[str, Optional[object]] = {
        "flow_type": _normalize_string(rule.get("set_flow_type")),
        "class": _normalize_string(rule.get("set_class")),
        "sub_class": _normalize_string(rule.get("set_sub_class")),
    }

    for col, value in updates.items():
        if value is None:
            continue
        eligible = mask & _field_allows_update(
            df[col],
            source_allows,
            overwrite_manual,
            _default_string_values(col),
        )
        if not overwrite_manual:
            eligible &= ~applied_by_field[col]
        if eligible.any():
            changed = eligible & df[col].ne(value)
            df.loc[eligible, col] = value
            rule_applied |= changed
            applied_by_field[col] |= changed

    bool_value = rule.get("set_is_fixed_waste")
    if not pd.isna(bool_value):
        eligible = mask & _field_allows_update(
            df["is_fixed_waste"],
            source_allows,
            overwrite_manual,
            defaults=None,
        )
        if not overwrite_manual:
            eligible &= ~applied_by_field["is_fixed_waste"]
        if eligible.any():
            changed = eligible & df["is_fixed_waste"].ne(bool_value)
            df.loc[eligible, "is_fixed_waste"] = bool(bool_value)
            rule_applied |= changed
            applied_by_field["is_fixed_waste"] |= changed

    return rule_applied


def _field_allows_update(
    series: pd.Series,
    source_allows: pd.Series,
    overwrite_manual: bool,
    defaults: Optional[set],
) -> pd.Series:
    default_mask = _default_mask(series, defaults)
    source_mask = source_allows | overwrite_manual
    return default_mask | source_mask


def _default_mask(series: pd.Series, defaults: Optional[set]) -> pd.Series:
    if defaults is None:
        return series.isna() | series.eq(False)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({d.lower() for d in defaults})


def _default_string_values(column: str) -> set:
    if column == "flow_type":
        return {"", "none", DEFAULT_FLOW_TYPE}
    return {"", "none"}


def _normalize_string(value: Optional[object]) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None
