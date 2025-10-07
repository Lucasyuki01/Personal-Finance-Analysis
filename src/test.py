import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List, Dict
from typing import Dict, List

# -------------------- helpers --------------------
def normalize_text(s: str) -> str:
    t = (str(s) or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    # if you want to remove prefixes "apos/opos/fpos", uncomment:
    # t = re.sub(r"^(apos|opos|fpos)\s+", "", t)
    return t

# -------------------- POS rules engine --------------------
def classify_pos_purchase(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    """
    Applies rules ONLY to rows with Description == 'pos purchase'.
    First match wins; increasing priority.
    """
    df = df.copy()

    is_pos   = df["Description"].fillna("").map(normalize_text).eq("pos purchase")
    sub_norm = df["Sub-description"].fillna("").map(normalize_text)

    # ensure output columns
    if "Category" not in df.columns: df["Category"] = "Others"
    if "Sub-Category" not in df.columns: df["Sub-Category"] = "None"

    # sort by priority
    rules_sorted = sorted(rules, key=lambda r: r.get("priority", 9999))

    # compile regex when necessary
    for r in rules_sorted:
        if r.get("match_type") == "regex":
            r["_compiled"] = re.compile(r["pattern"], flags=re.I)

    matched = pd.Series(False, index=df.index)
    for r in rules_sorted:
        pat = r["pattern"].lower()
        mt  = r.get("match_type", "contains")

        if mt == "startswith":
            mask = is_pos & ~matched & sub_norm.str.startswith(pat)
        elif mt == "regex":
            mask = is_pos & ~matched & sub_norm.map(lambda x: bool(r["_compiled"].search(x)))
        else:  # contains
            mask = is_pos & ~matched & sub_norm.str.contains(re.escape(pat), regex=True)

        if mask.any():
            df.loc[mask, "Category"]      = r["category"]
            df.loc[mask, "Sub-Category"]  = r.get("sub_category", "None")
            matched |= mask

    # (optional) fallback: unmatched POS
    still_pos_unmatched = is_pos & ~matched
    df.loc[still_pos_unmatched, ["Category","Sub-Category"]] = ["Others", "None"]

    return df

# -------------------- general classification --------------------
def classify_transactions(df: pd.DataFrame, pos_rules: List[Dict]) -> pd.DataFrame:
    """Classifies Class, Category and Sub-Category and calls the POS engine."""
    df = df.copy()

    # defaults
    df["Class"]         = "Expenses"
    df["Category"]      = "Others"
    df["Sub-Category"]  = "None"

    # normalizations
    desc    = df["Description"].fillna("").map(normalize_text)

    # --- Earnings ---
    mask_payroll   = desc.eq("payroll deposit")
    mask_correct   = desc.eq("correction")
    mask_interest  = desc.eq("interest")
    mask_deposit   = desc.eq("deposit")

    df.loc[mask_payroll,  ["Class","Category"]] = ["Earnings","Payment"]
    df.loc[mask_correct,  ["Class","Category"]] = ["Earnings","earnings"]
    df.loc[mask_interest, ["Class","Category"]] = ["Earnings","earnings"]

    df.loc[mask_deposit, "Class"]    = "Earnings"
    df.loc[mask_deposit, "Category"] = "shared bills"
    df.loc[mask_deposit & df["Amount"].between(300, 900, inclusive="neither"), "Category"] = "Payment"

    # --- Withdrawal (specific order -> general) ---
    mask_withdraw = desc.eq("withdrawal")
    df.loc[mask_withdraw & df.index.isin([399, 352, 289]), "Category"] = "Shopping"
    df.loc[mask_withdraw & (df["Amount"] == -1200), ["Category","Sub-Category"]] = ["Bills","Rent"]
    df.loc[mask_withdraw & df["Category"].eq("Others"), "Category"] = "Money Sent"

    # --- Bill payment / Service charge ---
    df.loc[desc.eq("bill payment"),   ["Category","Sub-Category"]] = ["Bills","Cellphone"]
    df.loc[desc.eq("service charge"), ["Category","Sub-Category"]] = ["Bills","Bank"]

    # --- POS purchase: everything is handled inside the dedicated function ---
    df = classify_pos_purchase(df, pos_rules)

    return df

# Example of “rules” (can come from CSV)
pos_rules = [
    # Transport (Compass)
    {"pattern": "compass", "category": "Bills", "sub_category": "Transport", "priority": 1, "match_type": "startswith"},
    {"pattern": "compass vending", "category": "Bills", "sub_category": "Transport", "priority": 1, "match_type": "contains"},
    # Groceries (supermarkets)
    {"pattern": "walmart", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "real cdn supers", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "save on foods", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "t t supermarket", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "h-mart", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "costco wholesal", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "urban fare", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    # Discount / Household
    {"pattern": "dollarama", "category": "Groceries", "sub_category": "Discount", "priority": 6, "match_type": "contains"},
    {"pattern": "dollar tree", "category": "Groceries", "sub_category": "Discount", "priority": 6, "match_type": "contains"},
    {"pattern": "daiso", "category": "Groceries", "sub_category": "Discount", "priority": 6, "match_type": "contains"},
    # Eating out / Coffee / Desserts
    {"pattern": "mcdonald", "category": "Eating Out", "sub_category": "Fast Food", "priority": 10, "match_type": "contains"},
    {"pattern": "kfc", "category": "Eating Out", "sub_category": "Fast Food", "priority": 10, "match_type": "contains"},
    {"pattern": "popeyes", "category": "Eating Out", "sub_category": "Fast Food", "priority": 10, "match_type": "contains"},
    {"pattern": "subway", "category": "Eating Out", "sub_category": "Fast Food", "priority": 10, "match_type": "contains"},
    {"pattern": "fatburger", "category": "Eating Out", "sub_category": "Burgers", "priority": 11, "match_type": "contains"},
    {"pattern": "freshslice", "category": "Eating Out", "sub_category": "Pizza", "priority": 11, "match_type": "contains"},
    {"pattern": "ramen", "category": "Eating Out", "sub_category": "Asian", "priority": 12, "match_type": "contains"},
    {"pattern": "sushi", "category": "Eating Out", "sub_category": "Asian", "priority": 12, "match_type": "contains"},
    {"pattern": "shawarma", "category": "Eating Out", "sub_category": "Middle Eastern", "priority": 12, "match_type": "contains"},
    {"pattern": "tim hortons", "category": "Eating Out", "sub_category": "Cafe", "priority": 13, "match_type": "contains"},
    {"pattern": "starbucks", "category": "Eating Out", "sub_category": "Cafe", "priority": 13, "match_type": "contains"},
    {"pattern": "lee's donut", "category": "Eating Out", "sub_category": "Dessert", "priority": 14, "match_type": "contains"},
    {"pattern": "rain or shi", "category": "Eating Out", "sub_category": "Dessert", "priority": 14, "match_type": "contains"},
    # Convenience / Liquor
    {"pattern": "7-eleven", "category": "Groceries", "sub_category": "None", "priority": 20, "match_type": "contains"},
    {"pattern": "ins market", "category": "Groceries", "sub_category": "None", "priority": 20, "match_type": "contains"},
    {"pattern": "liquor", "category": "Groceries", "Alcohool": "None", "priority": 21, "match_type": "contains"},
    # Retail
    {"pattern": "best buy", "category": "Groceries", "sub_category": "Electronics", "priority": 30, "match_type": "contains"},
    {"pattern": "london drugs", "category": "Groceries", "sub_category": "Pharmacy", "priority": 30, "match_type": "contains"},
    {"pattern": "winners", "category": "Groceries", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "sephora", "category": "Groceries", "sub_category": "Beauty", "priority": 30, "match_type": "contains"},
    {"pattern": "decathlon", "category": "Groceries", "sub_category": "Sports", "priority": 30, "match_type": "contains"},
    {"pattern": "sport chek", "category": "Groceries", "sub_category": "Sports", "priority": 30, "match_type": "contains"},
    {"pattern": "uniqlo", "category": "Groceries", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "nike", "category": "Groceries", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "north face", "category": "Groceries", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "browns shoes", "category": "Groceries", "sub_category": "Shoes", "priority": 30, "match_type": "contains"},
    # Online merchants
    {"pattern": "amzn mktp", "category": "Shopping", "sub_category": "Amazon", "priority": 40, "match_type": "contains"},
    {"pattern": "aliexpress", "category": "Shopping", "sub_category": "AliExpress", "priority": 40, "match_type": "contains"},
    {"pattern": "temu", "category": "Shopping", "sub_category": "Temu", "priority": 40, "match_type": "contains"},
    {"pattern": "name-cheap", "category": "Shopping", "sub_category": "Domains", "priority": 41, "match_type": "contains"},
    {"pattern": "openai", "category": "Shopping", "sub_category": "Services", "priority": 41, "match_type": "contains"},
    # Hair / Personal care
    {"pattern": "hair salon", "category": "Others", "sub_category": "Salon", "priority": 50, "match_type": "contains"},
]
#-----------------------------------------------------------------------------------------------------

acc1 = pd.read_csv('data/acc1.csv')
acc2 = pd.read_csv('data/acc2.csv')

# adding here a label in each dataframe before concat
acc1["Account"] = "Chequing"
acc2["Account"] = "Savings"

# concat
bank_df_raw = pd.concat([acc1, acc2], ignore_index=True)

# Removing columns that are not that important for what i would like to analyse
bank_df = bank_df_raw.drop(columns=["Filter", "Type of Transaction"])

# Removing rows that represents internal transactions between my accounts
bank_df = bank_df[bank_df["Description"] != "customer transfer cr."]
bank_df = bank_df[bank_df["Description"] != "customer transfer dr."]

# The only column that has some missing values is the sub-category, actually is not empty cause there is spaces there
# The solution was implementing this script just to make sure
bank_df["Sub-description"] = (
    bank_df["Sub-description"]
    .fillna("none")         # fill NaN
    .replace("", "none")    # fill empty strings
)

# fill when there is just spaces 
bank_df.loc[bank_df["Sub-description"].str.strip() == "", "Sub-description"] = "none"

# Right here I'm removing the initial deposits, to avoid contamination of the analysis
# This one is to remove the deposits done in the scotia unit
bank_df = bank_df[~((bank_df["Description"] == "deposit") & (bank_df["Sub-description"] == "none"))]

# Here also because all the abm deposits of this one were as "initial deposits"
bank_df = bank_df[bank_df["Description"] != "abm deposit"]

# These were the transfers that I did using my wise account 
bank_df = bank_df.drop(index=[400, 397, 389, 349])

# running the classification function
bank_df = classify_transactions(bank_df, pos_rules)

# ensure Category is filled
if "Category" not in bank_df.columns:
    raise ValueError("Category column not found. Run classification first.")

# aggregating expenses by category (using absolute value of Amount)
category_expenses = (
    bank_df[bank_df["Class"] == "Expenses"]
    .groupby("Category")["Amount"]
    .sum()
    .abs()
)

# sort to look nicer
category_expenses = category_expenses.sort_values(ascending=False)

# pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    category_expenses,
    labels=category_expenses.index,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)
plt.title("Expenses by Category", fontsize=14)
plt.show()

print(bank_df[bank_df['Category'] == 'Shopping'])