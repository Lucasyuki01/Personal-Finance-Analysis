"""Rule loading, saving, and application helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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
        empty = {"by_id": {}, "by_pattern": {}, "taxonomy": {"income": {}, "expense": {}}}
        SPECIFIC_RULES_PATH.write_text(json.dumps(empty, indent=2), encoding="utf-8")


def _pos_db_url() -> str:
    return os.getenv("POS_RULES_DATABASE_URL", "").strip()


def _using_pos_db() -> bool:
    return bool(_pos_db_url())


def _get_pos_db_connection():
    db_url = _pos_db_url()
    if not db_url:
        return None
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "POS_RULES_DATABASE_URL is set, but psycopg is not installed. Add psycopg[binary]."
        ) from exc
    return psycopg.connect(db_url)


def _ensure_pos_rules_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pos_rules (
                rule_key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                sub_category TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )


def load_pos_rules() -> Dict:
    ensure_rules_files()
    if _using_pos_db():
        with _get_pos_db_connection() as conn:
            _ensure_pos_rules_table(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT rule_key, category, sub_category, updated_at FROM pos_rules")
                rows = cur.fetchall()
        rules: Dict = {}
        for rule_key_value, category, sub_category, updated_at in rows:
            if hasattr(updated_at, "isoformat"):
                updated_at_iso = updated_at.isoformat()
            else:
                updated_at_iso = str(updated_at)
            rules[rule_key_value] = {
                "category": category,
                "sub_category": sub_category,
                "updated_at": updated_at_iso,
            }
        return rules
    try:
        return json.loads(POS_RULES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_pos_rules(pos_rules: Dict) -> None:
    ensure_rules_files()
    if _using_pos_db():
        with _get_pos_db_connection() as conn:
            _ensure_pos_rules_table(conn)
            with conn.cursor() as cur:
                for rule_key_value, rule in pos_rules.items():
                    cur.execute(
                        """
                        INSERT INTO pos_rules (rule_key, category, sub_category, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (rule_key)
                        DO UPDATE SET
                            category = EXCLUDED.category,
                            sub_category = EXCLUDED.sub_category,
                            updated_at = NOW()
                        """,
                        (
                            rule_key_value,
                            rule.get("category", "none"),
                            rule.get("sub_category", "none"),
                        ),
                    )
            conn.commit()
        return
    POS_RULES_PATH.write_text(json.dumps(pos_rules, indent=2), encoding="utf-8")


def persist_pos_rule(rule_key_value: str, rule: Dict) -> None:
    ensure_rules_files()
    if _using_pos_db():
        with _get_pos_db_connection() as conn:
            _ensure_pos_rules_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pos_rules (rule_key, category, sub_category, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (rule_key)
                    DO UPDATE SET
                        category = EXCLUDED.category,
                        sub_category = EXCLUDED.sub_category,
                        updated_at = NOW()
                    """,
                    (
                        rule_key_value,
                        rule.get("category", "none"),
                        rule.get("sub_category", "none"),
                    ),
                )
            conn.commit()
        return
    pos_rules = load_pos_rules()
    pos_rules[rule_key_value] = rule
    POS_RULES_PATH.write_text(json.dumps(pos_rules, indent=2), encoding="utf-8")


def delete_pos_rule(rule_key_value: str) -> None:
    ensure_rules_files()
    if _using_pos_db():
        with _get_pos_db_connection() as conn:
            _ensure_pos_rules_table(conn)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pos_rules WHERE rule_key = %s", (rule_key_value,))
            conn.commit()
        return
    pos_rules = load_pos_rules()
    pos_rules.pop(rule_key_value, None)
    POS_RULES_PATH.write_text(json.dumps(pos_rules, indent=2), encoding="utf-8")


def load_specific_rules() -> Dict:
    ensure_rules_files()
    try:
        data = json.loads(SPECIFIC_RULES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {"by_id": {}, "by_pattern": {}, "taxonomy": {"income": {}, "expense": {}}}
    return _ensure_specific_rules_shape(data)


def save_specific_rules(specific_rules: Dict) -> None:
    ensure_rules_files()
    specific_rules = _ensure_specific_rules_shape(specific_rules)
    SPECIFIC_RULES_PATH.write_text(json.dumps(specific_rules, indent=2), encoding="utf-8")


def load_specific_rules_from_upload(uploaded_file) -> Dict:
    try:
        content = uploaded_file.getvalue()
        data = json.loads(content.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid specific_rules file. Must be JSON.") from exc
    return _ensure_specific_rules_shape(data)


def _ensure_specific_rules_shape(data: Dict) -> Dict:
    data.setdefault("by_id", {})
    data.setdefault("by_pattern", {})
    taxonomy = data.setdefault("taxonomy", {})
    taxonomy.setdefault("income", {})
    taxonomy.setdefault("expense", {})
    return data


def normalize_name(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def dedupe_case_insensitive(items: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        norm = normalize_name(item)
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(norm)
    return deduped


def get_effective_taxonomy(
    default_income: Dict[str, List[str]],
    default_expense: Dict[str, List[str]],
    user_taxonomy: Dict,
) -> Dict[str, Dict[str, List[str]]]:
    user_taxonomy = user_taxonomy or {}
    income_user = user_taxonomy.get("income", {}) or {}
    expense_user = user_taxonomy.get("expense", {}) or {}

    return {
        "income": _merge_taxonomy(default_income, income_user),
        "expense": _merge_taxonomy(default_expense, expense_user),
    }


def _merge_taxonomy(default_tax: Dict[str, List[str]], user_tax: Dict) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}

    for category, subs in default_tax.items():
        merged[category] = dedupe_case_insensitive(list(subs or []))

    for category, subs in (user_tax or {}).items():
        category_name = normalize_name(category)
        if not category_name:
            continue
        lower = category_name.lower()
        existing_key = next((key for key in merged if key.lower() == lower), None)
        if existing_key is None:
            merged[category_name] = dedupe_case_insensitive(list(subs or []))
        else:
            merged[existing_key] = dedupe_case_insensitive(merged[existing_key] + list(subs or []))

    return merged


def taxonomy_self_check(taxonomy: Dict) -> List[str]:
    issues: List[str] = []
    if not isinstance(taxonomy, dict):
        return ["taxonomy must be a dict"]
    for scope in ("income", "expense"):
        scope_data = taxonomy.get(scope, {})
        if not isinstance(scope_data, dict):
            issues.append(f"{scope} taxonomy must be a dict")
            continue
        for category, subs in scope_data.items():
            if not isinstance(category, str) or not category.strip():
                issues.append(f"{scope} has invalid category name")
            if not isinstance(subs, list):
                issues.append(f"{scope} category '{category}' must have list of sub-categories")
                continue
            for sub in subs:
                if not isinstance(sub, str) or not sub.strip():
                    issues.append(f"{scope} category '{category}' has invalid sub-category")
                    break
    return issues


def rule_key(profit: int, description_norm: str, sub_description_norm: str) -> str:
    return f"{profit}|{description_norm}|{sub_description_norm}"


def _is_empty_cell(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").fillna("").str.strip().str.lower()
    return normalized.isin(["", "none"])


def apply_pos_rules(df: pd.DataFrame, pos_rules: Dict) -> pd.DataFrame:
    if not pos_rules:
        return df

    df = df.copy()
    mask_pos = df["description_norm"].isin(["pos purchase", "payroll"])
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
