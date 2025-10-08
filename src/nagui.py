import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
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
    # ===================== Transport =====================
    {"pattern": "compass",           "category": "Bills",      "sub_category": "Transport", "priority": 1, "match_type": "startswith"},
    {"pattern": "compass vending",   "category": "Bills",      "sub_category": "Transport", "priority": 1, "match_type": "contains"},
    {"pattern": "bcf-dep self se",   "category": "Services",  "sub_category": "Nanaimo",   "priority": 2, "match_type": "contains"},
    {"pattern": "bcf-dep self se",   "category": "Services",  "sub_category": "Nanaimo",   "priority": 2, "match_type": "contains"},

    # ===================== Bills =====================
    {"pattern": "revenue services bc", "category": "Bills", "sub_category": "Msp", "priority": 2, "match_type": "contains"},
    {"pattern": "abc*32130",           "category": "Bills", "sub_category": "Gym", "priority": 2, "match_type": "contains"},

    # ===================== Groceries =====================
    {"pattern": "walmart",           "category": "Groceries", "sub_category": "None",      "priority": 5, "match_type": "contains"},
    {"pattern": "real cdn supers",   "category": "Groceries", "sub_category": "None",      "priority": 5, "match_type": "contains"},
    {"pattern": "save on foods",     "category": "Groceries", "sub_category": "None",      "priority": 5, "match_type": "contains"},
    {"pattern": "t t supermarket",   "category": "Groceries", "sub_category": "None",      "priority": 5, "match_type": "contains"},
    {"pattern": "h-mart",            "category": "Groceries", "sub_category": "None",      "priority": 5, "match_type": "contains"},
    {"pattern": "costco wholesal",   "category": "Groceries", "sub_category": "None",      "priority": 5, "match_type": "contains"},
    {"pattern": "urban fare",        "category": "Groceries", "sub_category": "None",      "priority": 5, "match_type": "contains"},
    {"pattern": "dollarama",         "category": "Groceries", "sub_category": "Discount",  "priority": 6, "match_type": "contains"},
    {"pattern": "dollar tree",       "category": "Groceries", "sub_category": "Discount",  "priority": 6, "match_type": "contains"},
    {"pattern": "london drugs",      "category": "Groceries", "sub_category": "Pharmacy",  "priority": 30,"match_type": "contains"},
    {"pattern": "winners",           "category": "Groceries", "sub_category": "Apparel",   "priority": 30,"match_type": "contains"},
    {"pattern": "7-eleven",          "category": "Groceries", "sub_category": "None",      "priority": 20,"match_type": "contains"},
    {"pattern": "ins market",        "category": "Groceries", "sub_category": "None",      "priority": 20,"match_type": "contains"},
    {"pattern": "liquor",            "category": "Groceries", "sub_category": "Alcohol",   "priority": 21,"match_type": "contains"},

    # ===================== Eating Out =====================
    {"pattern": "mcdonald",          "category": "Eating Out", "sub_category": "Fast Food",      "priority": 10, "match_type": "contains"},
    {"pattern": "kfc",               "category": "Eating Out", "sub_category": "Fast Food",      "priority": 10, "match_type": "contains"},
    {"pattern": "popeyes",           "category": "Eating Out", "sub_category": "Fast Food",      "priority": 10, "match_type": "contains"},
    {"pattern": "subway",            "category": "Eating Out", "sub_category": "Fast Food",      "priority": 10, "match_type": "contains"},
    {"pattern": "fatburger",         "category": "Eating Out", "sub_category": "Burgers",        "priority": 11, "match_type": "contains"},
    {"pattern": "freshslice",        "category": "Eating Out", "sub_category": "Pizza",          "priority": 11, "match_type": "contains"},
    {"pattern": "ramen",             "category": "Eating Out", "sub_category": "Asian",          "priority": 12, "match_type": "contains"},
    {"pattern": "sushi",             "category": "Eating Out", "sub_category": "Asian",          "priority": 12, "match_type": "contains"},
    {"pattern": "shawarma",          "category": "Eating Out", "sub_category": "Middle Eastern", "priority": 12, "match_type": "contains"},
    {"pattern": "tim hortons",       "category": "Eating Out", "sub_category": "Cafe",           "priority": 13, "match_type": "contains"},
    {"pattern": "starbucks",         "category": "Eating Out", "sub_category": "Cafe",           "priority": 13, "match_type": "contains"},
    {"pattern": "lee's donut",       "category": "Eating Out", "sub_category": "Dessert",        "priority": 14, "match_type": "contains"},
    {"pattern": "rain or shi",       "category": "Eating Out", "sub_category": "Dessert",        "priority": 14, "match_type": "contains"},
    {"pattern": "charlatan",         "category": "Eating Out", "sub_category": "Bar",            "priority": 2,  "match_type": "contains"},
    {"pattern": "hungry guys",       "category": "Eating Out", "sub_category": "Bar",            "priority": 2,  "match_type": "contains"},
    {"pattern": "mangos kitchen",    "category": "Eating Out", "sub_category": "Asian",          "priority": 2,  "match_type": "contains"},
    {"pattern": "bbq chicken",       "category": "Eating Out", "sub_category": "Asian",          "priority": 2,  "match_type": "contains"},
    {"pattern": "menya raizo",       "category": "Eating Out", "sub_category": "Asian",          "priority": 2,  "match_type": "contains"},
    {"pattern": "big way hot pot",   "category": "Eating Out", "sub_category": "Asian",          "priority": 2,  "match_type": "contains"},
    {"pattern": "tst-jerusalem",     "category": "Eating Out", "sub_category": "Middle Eastern", "priority": 2,  "match_type": "contains"},
    {"pattern": "rio brazilian",     "category": "Eating Out", "sub_category": "Latin",          "priority": 2,  "match_type": "contains"},
    {"pattern": "boteco brasil",     "category": "Eating Out", "sub_category": "Latin",          "priority": 2,  "match_type": "contains"},
    {"pattern": "brazilliant",       "category": "Eating Out", "sub_category": "Latin",          "priority": 2,  "match_type": "contains"},
    {"pattern": "the old spaghet",   "category": "Eating Out", "sub_category": "Italian",        "priority": 2,  "match_type": "contains"},
    {"pattern": "trollers fish",     "category": "Eating Out", "sub_category": "Canadian",       "priority": 2,  "match_type": "contains"},
    {"pattern": "dip co. delight",   "category": "Eating Out", "sub_category": "Canadian",       "priority": 2,  "match_type": "contains"},
    {"pattern": "cactus club",       "category": "Eating Out", "sub_category": "Apparel",        "priority": 2,  "match_type": "contains"},

    # ===================== Shopping =====================
    {"pattern": "herschel",          "category": "Shopping", "sub_category": "Apparel",      "priority": 2, "match_type": "contains"},
    {"pattern": "tommy hilfiger",    "category": "Shopping", "sub_category": "Apparel",      "priority": 2, "match_type": "contains"},
    {"pattern": "roots mcarthur",    "category": "Shopping", "sub_category": "Apparel",      "priority": 2, "match_type": "contains"},
    {"pattern": "under armour",      "category": "Shopping", "sub_category": "Apparel",      "priority": 2, "match_type": "contains"},
    {"pattern": "best buy",          "category": "Shopping", "sub_category": "Electronics",  "priority": 30,"match_type": "contains"},
    {"pattern": "sephora",           "category": "Shopping", "sub_category": "Beauty",       "priority": 30,"match_type": "contains"},
    {"pattern": "decathlon",         "category": "Shopping", "sub_category": "Sports",       "priority": 30,"match_type": "contains"},
    {"pattern": "sport chek",        "category": "Shopping", "sub_category": "Sports",       "priority": 30,"match_type": "contains"},
    {"pattern": "uniqlo",            "category": "Shopping", "sub_category": "Apparel",      "priority": 30,"match_type": "contains"},
    {"pattern": "nike",              "category": "Shopping", "sub_category": "Apparel",      "priority": 30,"match_type": "contains"},
    {"pattern": "north face",        "category": "Shopping", "sub_category": "Apparel",      "priority": 30,"match_type": "contains"},
    {"pattern": "browns shoes",      "category": "Shopping", "sub_category": "Shoes",        "priority": 30,"match_type": "contains"},
    {"pattern": "shoppers drug",     "category": "Shopping", "sub_category": "Barber machine","priority": 2,"match_type": "contains"},
    {"pattern": "daiso",             "category": "Shopping", "sub_category": "Discount",     "priority": 6, "match_type": "contains"},
    {"pattern": "amzn mktp",         "category": "Shopping", "sub_category": "Amazon",       "priority": 40,"match_type": "contains"},
    {"pattern": "aliexpress",        "category": "Shopping", "sub_category": "AliExpress",   "priority": 40,"match_type": "contains"},
    {"pattern": "temu",              "category": "Shopping", "sub_category": "Temu",         "priority": 40,"match_type": "contains"},

    # ===================== Services =====================
    {"pattern": "driver serv.cen",   "category": "Services", "sub_category": "BCID",          "priority": 2, "match_type": "contains"},
    {"pattern": "jay hair salon",    "category": "Services", "sub_category": "Hair cut",      "priority": 2, "match_type": "contains"},
    {"pattern": "cfs-safecheck",     "category": "Services", "sub_category": "Certification", "priority": 2, "match_type": "contains"},
    {"pattern": "name-cheap",        "category": "Services", "sub_category": "Domains",       "priority": 41,"match_type": "contains"},
    {"pattern": "openai",            "category": "Services", "sub_category": "Salon",      "priority": 41,"match_type": "contains"},
    {"pattern": "hair salon",        "category": "Services",   "sub_category": "Salon",         "priority": 50,"match_type": "contains"},
]

def _load_sample_data() -> pd.DataFrame:
    acc1 = pd.read_csv('data/acc1.csv')
    acc2 = pd.read_csv('data/acc2.csv')
    acc1["Account"] = "Chequing"
    acc2["Account"] = "Savings"
    return pd.concat([acc1, acc2], ignore_index=True)


def _load_uploaded(files: List) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            try:
                f.seek(0)
            except Exception:
                pass
            df = pd.read_excel(f)
        account_label = getattr(f, 'name', 'Uploaded')
        df['Account'] = account_label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def preprocess_bank_df(bank_df_raw: pd.DataFrame) -> pd.DataFrame:
    if bank_df_raw.empty:
        return bank_df_raw

    cols_to_drop = [c for c in ["Filter", "Type of Transaction"] if c in bank_df_raw.columns]
    bank_df = bank_df_raw.drop(columns=cols_to_drop) if cols_to_drop else bank_df_raw.copy()

    if 'Description' in bank_df.columns:
        bank_df = bank_df[bank_df["Description"] != "customer transfer cr."]
        bank_df = bank_df[bank_df["Description"] != "customer transfer dr."]

    if 'Sub-description' in bank_df.columns:
        bank_df["Sub-description"] = bank_df["Sub-description"].fillna("none").replace("", "none")
        bank_df.loc[bank_df["Sub-description"].str.strip() == "", "Sub-description"] = "none"

    if {'Description', 'Sub-description'}.issubset(bank_df.columns):
        bank_df = bank_df[~((bank_df["Description"] == "deposit") & (bank_df["Sub-description"] == "none"))]

    if 'Description' in bank_df.columns:
        bank_df = bank_df[bank_df["Description"] != "abm deposit"]

    bank_df = bank_df.drop(index=[400, 397, 389, 349], errors='ignore')

    return bank_df


def make_pie_chart(series: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        series,
        labels=series.index,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    ax.set_title("Expenses by Category", fontsize=14)
    return fig


def main():
    st.set_page_config(page_title="Personal Finance Analysis", layout="wide")
    st.title("Personal Finance Analysis")

    with st.sidebar:
        st.header("Data Source")
        source = st.radio(
            "Choose data source",
            options=["Sample data", "Upload CSV/Excel"],
            index=0,
        )

        uploaded_files = []
        if source == "Upload CSV/Excel":
            uploaded_files = st.file_uploader(
                "Upload one or more statements",
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
            )

        st.header("Filters")
        show_table = st.checkbox("Show classified table", value=False)
        show_shopping_only = st.checkbox("Show Shopping subset", value=False)

    if source == "Sample data":
        try:
            bank_df_raw = _load_sample_data()
        except Exception as e:
            st.error(f"Failed to load sample data: {e}")
            return
    else:
        if not uploaded_files:
            st.info("Upload at least one file to proceed.")
            return
        bank_df_raw = _load_uploaded(uploaded_files)

    st.subheader("Raw Data Preview")
    st.write(bank_df_raw.head(10))

    bank_df = preprocess_bank_df(bank_df_raw)

    try:
        bank_df = classify_transactions(bank_df, pos_rules)
    except Exception as e:
        st.error(f"Classification failed: {e}")
        return

    if "Category" not in bank_df.columns:
        st.error("Category column not found after classification.")
        return

    expenses_only = bank_df[bank_df.get("Class", "Expenses") == "Expenses"].copy()
    if expenses_only.empty:
        st.warning("No expenses found to chart.")
    else:
        category_expenses = (
            expenses_only.groupby("Category")["Amount"].sum().abs().sort_values(ascending=False)
        )
        st.subheader("Expenses by Category")
        fig = make_pie_chart(category_expenses)
        st.pyplot(fig, use_container_width=True)

    if show_table:
        st.subheader("Classified Transactions")
        st.dataframe(bank_df, use_container_width=True)

    if show_shopping_only:
        st.subheader("Shopping Transactions")
        st.dataframe(bank_df[bank_df['Category'] == 'Shopping'], use_container_width=True)

    csv_bytes = bank_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Classified CSV",
        data=csv_bytes,
        file_name="classified_transactions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
    