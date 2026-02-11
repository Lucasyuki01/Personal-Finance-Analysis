"""Rule loading, saving, and application helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
POS_RULES_PATH = DATA_DIR / "pos_rules.json"
SPECIFIC_RULES_PATH = DATA_DIR / "specific_rules.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_rules_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not POS_RULES_PATH.exists():
        POS_RULES_PATH.write_text("{}", encoding="utf-8")
    if not SPECIFIC_RULES_PATH.exists():
        empty = {"by_id": {}, "by_pattern": {}}
        SPECIFIC_RULES_PATH.write_text(json.dumps(empty, indent=2), encoding="utf-8")


def load_pos_rules() -> Dict:
    ensure_rules_files()
    try:
        return json.loads(POS_RULES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_pos_rules(pos_rules: Dict) -> None:
    ensure_rules_files()
    POS_RULES_PATH.write_text(json.dumps(pos_rules, indent=2), encoding="utf-8")


def load_specific_rules() -> Dict:
    ensure_rules_files()
    try:
        data = json.loads(SPECIFIC_RULES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {"by_id": {}, "by_pattern": {}}
    data.setdefault("by_id", {})
    data.setdefault("by_pattern", {})
    return data


def save_specific_rules(specific_rules: Dict) -> None:
    ensure_rules_files()
    specific_rules.setdefault("by_id", {})
    specific_rules.setdefault("by_pattern", {})
    SPECIFIC_RULES_PATH.write_text(json.dumps(specific_rules, indent=2), encoding="utf-8")


def load_specific_rules_from_upload(uploaded_file) -> Dict:
    try:
        content = uploaded_file.getvalue()
        data = json.loads(content.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid specific_rules file. Must be JSON.") from exc
    data.setdefault("by_id", {})
    data.setdefault("by_pattern", {})
    return data


def rule_key(profit: int, description_norm: str, sub_description_norm: str) -> str:
    return f"{profit}|{description_norm}|{sub_description_norm}"


def _is_empty_cell(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").fillna("").str.strip().str.lower()
    return normalized.isin(["", "none"])


def apply_pos_rules(df: pd.DataFrame, pos_rules: Dict) -> pd.DataFrame:
    if not pos_rules:
        return df

    df = df.copy()
    mask_pos = df["description_norm"] == "pos purchase"
    if not mask_pos.any():
        return df

    key_series = (
        df.loc[mask_pos, "Profit"].astype(str)
        + "|"
        + df.loc[mask_pos, "description_norm"]
        + "|"
        + df.loc[mask_pos, "sub_description_norm"]
    )

    mapped_category = key_series.map(lambda k: pos_rules.get(k, {}).get("category"))
    mapped_sub = key_series.map(lambda k: pos_rules.get(k, {}).get("sub_category"))

    cat_empty = _is_empty_cell(df.loc[mask_pos, "Category"])
    sub_empty = _is_empty_cell(df.loc[mask_pos, "Sub-Category"])

    pos_idx = df.index[mask_pos]
    update_cat = cat_empty & mapped_category.notna()
    update_sub = sub_empty & mapped_sub.notna()

    df.loc[pos_idx[update_cat], "Category"] = mapped_category[update_cat]
    df.loc[pos_idx[update_sub], "Sub-Category"] = mapped_sub[update_sub]

    return df


def apply_specific_rules(df: pd.DataFrame, specific_rules: Dict) -> pd.DataFrame:
    df = df.copy()
    by_id = specific_rules.get("by_id", {})
    by_pattern = specific_rules.get("by_pattern", {})

    if by_id:
        for rule_id, rule in by_id.items():
            df.loc[df["ID"] == rule_id, "Category"] = rule.get("category", "none")
            df.loc[df["ID"] == rule_id, "Sub-Category"] = rule.get("sub_category", "none")

    if by_pattern:
        key_series = (
            df["Profit"].astype(str)
            + "|"
            + df["description_norm"]
            + "|"
            + df["sub_description_norm"]
        )

        mapped_category = key_series.map(lambda k: by_pattern.get(k, {}).get("category"))
        mapped_sub = key_series.map(lambda k: by_pattern.get(k, {}).get("sub_category"))

        mask_cat = mapped_category.notna()
        mask_sub = mapped_sub.notna()

        df.loc[mask_cat, "Category"] = mapped_category[mask_cat]
        df.loc[mask_sub, "Sub-Category"] = mapped_sub[mask_sub]

    return df


def update_pos_rule(pos_rules: Dict, key: str, category: str, sub_category: str) -> Dict | None:
    previous = pos_rules.get(key)
    pos_rules[key] = {
        "category": category,
        "sub_category": sub_category,
        "updated_at": _utc_now_iso(),
    }
    return previous


def update_specific_rule_by_id(
    specific_rules: Dict, rule_id: str, category: str, sub_category: str
) -> Dict | None:
    specific_rules.setdefault("by_id", {})
    previous = specific_rules["by_id"].get(rule_id)
    specific_rules["by_id"][rule_id] = {
        "category": category,
        "sub_category": sub_category,
        "updated_at": _utc_now_iso(),
    }
    return previous


def update_specific_rule_by_pattern(
    specific_rules: Dict, key: str, category: str, sub_category: str
) -> Dict | None:
    specific_rules.setdefault("by_pattern", {})
    previous = specific_rules["by_pattern"].get(key)
    specific_rules["by_pattern"][key] = {
        "category": category,
        "sub_category": sub_category,
        "updated_at": _utc_now_iso(),
    }
    return previous


def revert_rule(
    pos_rules: Dict,
    specific_rules: Dict,
    action_type: str,
    rule_key_value: str,
    previous_rule: Dict | None,
) -> None:
    if action_type == "pos_save":
        if previous_rule is None:
            pos_rules.pop(rule_key_value, None)
        else:
            pos_rules[rule_key_value] = previous_rule
    elif action_type == "specific_save":
        specific_rules.setdefault("by_id", {})
        if previous_rule is None:
            specific_rules["by_id"].pop(rule_key_value, None)
        else:
            specific_rules["by_id"][rule_key_value] = previous_rule
