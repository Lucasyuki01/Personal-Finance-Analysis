from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from uuid import uuid4

import pandas as pd

from rules.schema import CLASSIFICATION_RULE_COLUMNS, empty_rules_df


EDIT_FIELDS: List[str] = ["flow_type", "class", "sub_class", "is_fixed_waste"]


def generate_rules_from_edits(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    min_examples: int = 2,
) -> pd.DataFrame:
    """
    Generate draft rules from repeated manual edits.

    This is a placeholder implementation that finds repeated changes by
    description_norm and suggests equals-based rules. It should be reviewed
    before activating.
    """
    if "transaction_id" not in before_df.columns or "transaction_id" not in after_df.columns:
        raise ValueError("Both dataframes must include transaction_id.")
    if "description_norm" not in after_df.columns:
        raise ValueError("after_df must include description_norm.")

    before = before_df.set_index("transaction_id")
    after = after_df.set_index("transaction_id")
    common = before.index.intersection(after.index)
    if common.empty:
        return empty_rules_df()

    before = before.loc[common]
    after = after.loc[common]

    changes = _find_changes(before, after)
    if changes.empty:
        return empty_rules_df()

    grouped = (
        changes.groupby(["description_norm"] + EDIT_FIELDS, dropna=False)
        .size()
        .reset_index(name="count")
    )
    candidates = grouped[grouped["count"] >= min_examples]
    if candidates.empty:
        return empty_rules_df()

    now = datetime.now(timezone.utc)
    rules = []
    for _, row in candidates.iterrows():
        rule = {
            "rule_id": str(uuid4()),
            "priority": 1000,
            "is_active": False,
            "scope_source_account": "",
            "match_field": "description_norm",
            "match_type": "equals",
            "match_value": row["description_norm"],
            "set_flow_type": _normalize_string(row["flow_type"]),
            "set_class": _normalize_string(row["class"]),
            "set_sub_class": _normalize_string(row["sub_class"]),
            "set_is_fixed_waste": row["is_fixed_waste"],
            "created_at": now,
            "updated_at": now,
            "notes": "generated from edits",
        }
        rules.append(rule)

    return pd.DataFrame(rules, columns=CLASSIFICATION_RULE_COLUMNS)


def _find_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    for field in EDIT_FIELDS:
        if field not in before.columns or field not in after.columns:
            raise ValueError(f"Missing required field '{field}' for edit detection.")

    before_fields = before[EDIT_FIELDS].copy()
    after_fields = after[EDIT_FIELDS].copy()

    for field in EDIT_FIELDS:
        if field == "is_fixed_waste":
            before_fields[field] = before_fields[field].fillna(False).astype(bool)
            after_fields[field] = after_fields[field].fillna(False).astype(bool)
        else:
            before_fields[field] = before_fields[field].fillna("").astype(str)
            after_fields[field] = after_fields[field].fillna("").astype(str)

    changed_mask = (before_fields != after_fields).any(axis=1)
    changed = after.loc[changed_mask].copy()
    if changed.empty:
        return changed

    return changed[["description_norm"] + EDIT_FIELDS]


def _normalize_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text if text else ""

