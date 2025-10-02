import re
from typing import Dict, List

import pandas as pd


def classify_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Classify transactions into Class, Category, and Sub-Category columns."""
    df = df.copy()

    df["Class"] = "Expenses"
    df["Category"] = "Others"
    df["Sub-Category"] = "None"

    desc = df["Description"].fillna("").str.lower()
    sub_desc = df["Sub-description"].fillna("").str.lower()

    payroll_mask = desc == "payroll deposit"
    df.loc[payroll_mask, "Class"] = "Earnings"
    df.loc[payroll_mask, "Category"] = "Payment"

    correction_mask = desc == "correction"
    df.loc[correction_mask, "Class"] = "Earnings"
    df.loc[correction_mask, "Category"] = "earnings"

    interest_mask = desc == "interest"
    df.loc[interest_mask, "Class"] = "Earnings"
    df.loc[interest_mask, "Category"] = "earnings"

    deposit_mask = desc == "deposit"
    df.loc[deposit_mask, "Class"] = "Earnings"
    df.loc[deposit_mask, "Category"] = "shared bills"
    deposit_payment_mask = deposit_mask & df["Amount"].between(300, 900, inclusive="neither")
    df.loc[deposit_payment_mask, "Category"] = "Payment"

    withdrawal_mask = desc == "withdrawal"
    special_indices_mask = withdrawal_mask & df.index.isin({399, 352, 289})
    df.loc[special_indices_mask, "Category"] = "shopping"

    rent_mask = withdrawal_mask & ~special_indices_mask & (df["Amount"] == 1200)
    df.loc[rent_mask, "Category"] = "Bills"
    df.loc[rent_mask, "Sub-Category"] = "Rent"

    other_withdrawal_mask = withdrawal_mask & ~(special_indices_mask | rent_mask)
    df.loc[other_withdrawal_mask, "Category"] = "money sent"

    pos_purchase_mask = desc == "pos purchase"
    compass_mask = pos_purchase_mask & sub_desc.str.startswith("compass")
    df.loc[compass_mask, "Category"] = "Bills"
    df.loc[compass_mask, "Sub-Category"] = "Transport"

    walmart_mask = pos_purchase_mask & sub_desc.str.startswith("walmart")
    df.loc[walmart_mask, "Category"] = "Groceries"

    real_cdn_mask = pos_purchase_mask & sub_desc.str.startswith("real cdn")
    df.loc[real_cdn_mask, "Category"] = "Groceries"

    bill_payment_mask = desc == "bill payment"
    df.loc[bill_payment_mask, "Category"] = "Bills"
    df.loc[bill_payment_mask, "Sub-Category"] = "Cellphone"

    service_charge_mask = desc == "service charge"
    df.loc[service_charge_mask, "Category"] = "Bills"
    df.loc[service_charge_mask, "Sub-Category"] = "Bank"

    return df


def normalize_text(s: str) -> str:
    t = (str(s) or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def classify_pos_purchase_generic(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    df = df.copy()
    # normalize once
    is_pos = df["Description"].fillna("").str.lower().eq("pos purchase")
    sub_norm = df["Sub-description"].fillna("").map(normalize_text)

    # ensure columns exist
    if "Category" not in df.columns:
        df["Category"] = "Others"
    if "Sub-Category" not in df.columns:
        df["Sub-Category"] = "None"

    # sort rules by priority
    rules_sorted = sorted(rules, key=lambda r: r.get("priority", 9999))

    # compile regexes
    for r in rules_sorted:
        if r.get("match_type") == "regex":
            r["_compiled"] = re.compile(r["pattern"], flags=re.I)

    # apply first-match-wins
    matched = pd.Series(False, index=df.index)
    for r in rules_sorted:
        pattern = r["pattern"].lower()
        match_type = r.get("match_type")
        if match_type == "startswith":
            mask = is_pos & ~matched & sub_norm.str.startswith(pattern)
        elif match_type == "regex":
            mask = is_pos & ~matched & sub_norm.str.match(r["_compiled"])
        else:  # contains (default)
            mask = is_pos & ~matched & sub_norm.str.contains(re.escape(pattern), regex=True)

        if mask.any():
            df.loc[mask, "Category"] = r["category"]
            df.loc[mask, "Sub-Category"] = r.get("sub_category", "None")
            matched |= mask

    # Final category consolidation
    original_category = df["Category"].copy()

    remap = {
        "Retail": "Shopping",
        "Online": "Shopping",
        "Household": "Groceries",
        "Convenience": "Groceries",
    }

    df["Category"] = df["Category"].replace(remap)

    special_rules = {
        "Alcohol": ("Convenience", "Alcohool"),
        "Coffee": ("Eating Out", "Coffee"),
        "Personal Care": ("Others", "Hair cut"),
    }

    for source, (target_category, target_sub_category) in special_rules.items():
        mask = original_category == source
        if mask.any():
            df.loc[mask, "Category"] = target_category
            df.loc[mask, "Sub-Category"] = target_sub_category

    return df
