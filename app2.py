# app.py
import os
import re
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- Page & layout ----------------
st.set_page_config(page_title="Personal Finance Dashboard", page_icon="💸", layout="wide")
st.title("💸 Personal Finance Dashboard")

# ---------------- Helpers: loading & normalization ----------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    """
    Read a CSV from path or file-like object and perform light normalization:
    - strip column names
    - coerce Date if present
    - build Amount from Debit/Credit if present, else coerce existing Amount
    """
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.strip()

    # Coerce Date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Build or coerce Amount
    if {"Debit", "Credit"}.issubset(df.columns):
        # Positive = inflow, Negative = outflow
        df["Amount"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0) - \
                       pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
    elif "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Clean text columns commonly used
    for col in ["Description", "Sub-description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    return df

def default_data_path():
    """
    Look for the first CSV inside ./data and return its path if present.
    """
    data_dir = Path(os.getcwd()) / "data"
    if data_dir.is_dir():
        for name in os.listdir(data_dir):
            if name.lower().endswith(".csv"):
                return str(data_dir / name)
    return None

# ---------------- Sidebar: data source ----------------
st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
use_default = st.sidebar.checkbox("Use first CSV in ./data", value=uploaded is None)

df = None
if uploaded is not None:
    df = load_csv(uploaded)
elif use_default:
    path = default_data_path()
    if path:
        df = load_csv(path)
        st.sidebar.success(f"Loaded: {os.path.basename(path)}")
    else:
        st.sidebar.warning("No CSV found in ./data")

if df is None:
    st.info("Upload a CSV in the sidebar or place one in the `data/` folder.")
    st.stop()

# ---------------- Optional: Date filter ----------------
if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
    # Drop NaT and sort
    dft = df.dropna(subset=["Date"]).sort_values("Date").copy()
    if not dft.empty:
        min_d = dft["Date"].min().date()
        max_d = dft["Date"].max().date()
        start, end = st.slider(
            "Time range",
            min_value=min_d,
            max_value=max_d,
            value=(min_d, max_d),
            format="YYYY-MM-DD",
        )
        mask = (dft["Date"].dt.date >= start) & (dft["Date"].dt.date <= end)
        dff = dft.loc[mask].copy()
    else:
        dff = df.copy()
else:
    dff = df.copy()

# ---------------- Classification logic (Description + Sub-description) ----------------
# We will:
# 1) Map transaction type from 'Description'
# 2) Canonicalize merchant from 'Sub-description'
# 3) Assign category from deterministic rules (txn_type + merchant)
# 4) Fallback with simple keyword heuristics on Sub-description

CITY_TOKENS = r"(vanco|burna|richm|nanai|n-van|north|edmond|robston|granv|ben|vict|calga)"
PREFIX_TOKENS = r"^(apos|opos|fpos|sq)\s+"

def normalize_text(s: str) -> str:
    t = (str(s) or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def map_txn_type(desc: str) -> str:
    d = normalize_text(desc)
    if d.startswith("payroll deposit"):
        return "payroll_deposit"
    if d.startswith("bill payment"):
        return "bill_payment"
    if d.startswith("customer transfer cr"):
        return "transfer_in"
    if d.startswith("customer transfer dr"):
        return "transfer_out"
    if d.startswith("withdrawal"):
        return "withdrawal"
    if d.startswith("deposit"):
        return "deposit"
    if d.startswith("pos purchase"):
        return "pos_purchase"
    if d.startswith("correction"):
        return "correction"
    return "other"

def canonicalize_merchant(sub_desc: str) -> str:
    """
    Clean noisy Sub-description to a merchant slug (very compact brand-like token).
    """
    t = normalize_text(sub_desc)
    # remove leading POS-like prefixes
    t = re.sub(PREFIX_TOKENS, "", t)
    # remove store numbers and mixed ids (#402, *32130, 1843, 1213, etc.)
    t = re.sub(r"[#*]?\d[\d\-]*", " ", t)
    # remove common city/abbr tokens
    t = re.sub(rf"\b{CITY_TOKENS}\b", " ", t)
    # keep letters, spaces and a few connectors
    t = re.sub(r"[^a-z\s&\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # normalize common brands
    t = t.replace("mcdonald s", "mcdonalds").replace("a w", "a&w").replace("real cdn supers", "real canadian superstore")
    t = t.replace("save on foods", "save on foods").replace("h mart", "h-mart").replace("t t", "t&t")

    # choose a short slug: first token is usually enough after cleaning
    tokens = t.split()
    if not tokens:
        return ""
    return tokens[0]

# Known merchants (slug -> category). Tune freely as you see real slugs.
KNOWN_MERCHANTS = {
    # Groceries
    "walmart": "Groceries",
    "save": "Groceries",                    # Save (on Foods) may reduce to 'save'
    "real": "Groceries",                    # may appear as 'real' (refine as needed)
    "t": "Groceries",                       # 't&t' can shrink to 't', adjust if needed
    "h": "Groceries",                       # 'h-mart' can shrink to 'h'
    "h-mart": "Groceries",
    "london": "Health",                     # Could also be Groceries; your call
    "shoppers": "Health",
    "urban": "Groceries",                   # Urban Fare
    "no": "Groceries",                      # No Frills can reduce to 'no'

    # Restaurants / Fast Food
    "kfc": "Restaurants",
    "mcdonalds": "Restaurants",
    "subway": "Restaurants",
    "freshslice": "Restaurants",
    "romano": "Restaurants",
    "ramen": "Restaurants",
    "edo": "Restaurants",
    "fatburger": "Restaurants",
    "popeyes": "Restaurants",
    "taco": "Restaurants",
    "cactus": "Restaurants",
    "jollibee": "Restaurants",
    "rain": "Restaurants",                  # Rain or Shine
    "lee": "Restaurants",                   # Lee's Donuts
    "kaisereck": "Restaurants",
    "butcher": "Restaurants",
    "pizza": "Restaurants",
    "donair": "Restaurants",
    "shawarma": "Restaurants",
    "coffee": "Restaurants",
    "breka": "Restaurants",
    "gelato": "Restaurants",
    "bbq": "Restaurants",
    "hungry": "Restaurants",                # Hungry Guys
    "mr": "Restaurants",                    # Mr. Shawarma / Mr. Supreme (coarse)
    "ak": "Restaurants",                    # AK Shawarma (coarse)
    "horin": "Restaurants",
    "danbo": "Restaurants",
    "fat": "Restaurants",                   # Fat Burger (coarse)

    # Transport
    "compass": "Transport",

    # Shopping / Retail
    "dollarama": "Shopping",
    "dollar": "Shopping",                   # Dollar Tree can reduce to 'dollar'
    "daiso": "Shopping",
    "best": "Shopping",                     # Best Buy
    "sport": "Shopping",                    # Sport Chek
    "decathlon": "Shopping",
    "winners": "Shopping",
    "sephora": "Shopping",
    "uniqlo": "Shopping",
    "roots": "Shopping",
    "north": "Shopping",                    # The North Face
    "browns": "Shopping",
    "tommy": "Shopping",
    "home": "Shopping",                     # Home Depot
    "konbiniya": "Shopping",

    # Online / Digital / Services
    "amzn": "Shopping",
    "amazon": "Shopping",
    "temu": "Shopping",
    "aliexpress": "Shopping",
    "google": "Digital",
    "namecheap": "Digital",
    "name": "Digital",                      # Sometimes reduced
    "openai": "Digital",
    "name-cheap": "Digital",

    # Utilities / Telecom
    "koodo": "Utilities",
    "bc": "Utilities",                      # bc hydro (coarse)
    "hydro": "Utilities",
    "fortis": "Utilities",
    "telus": "Utilities",
    "shaw": "Utilities",
    "rogers": "Utilities",
    "fido": "Utilities",

    # Fees / Taxes / Government
    "revenue": "Fees",
    "driver": "Fees",                       # Driver services center

    # Entertainment / Leisure
    "grouse": "Entertainment",
}

def categorize_row(txn_type: str, merchant_slug: str, sub_desc: str) -> str:
    """
    Category decision:
    1) by transaction type (strong signals)
    2) by known merchant slug
    3) by lightweight keyword fallback on Sub-description
    """
    # 1) by transaction type
    if txn_type == "payroll_deposit":
        return "Income"
    if txn_type in {"transfer_in", "deposit"}:
        return "Income"
    if txn_type in {"transfer_out", "withdrawal"}:
        return "Transfers"  # or "Other" if you prefer
    if txn_type == "bill_payment" and ("koodo" in merchant_slug):
        return "Utilities"

    # 2) by known merchant
    if merchant_slug in KNOWN_MERCHANTS:
        return KNOWN_MERCHANTS[merchant_slug]

    # 3) keyword fallback on sub-description
    t = normalize_text(sub_desc)
    if any(k in t for k in ["pizza","ramen","donair","shawarma","sushi","coffee","bakery","burger","chicken","bbq"]):
        return "Restaurants"
    if any(k in t for k in ["liquor"]):
        return "Shopping"
    if "compass" in t:
        return "Transport"

    return "Other"

# ---- Build combined text & derived fields ----
# We accept both columns; if Sub-description is missing, we still proceed.
if "Description" not in dff.columns:
    st.error("Column 'Description' is required in your CSV.")
    st.stop()

if "Sub-description" not in dff.columns:
    dff["Sub-description"] = ""

# Derive helper fields
dff["txn_type"] = dff["Description"].apply(map_txn_type)
dff["merchant_raw"] = dff["Sub-description"]
dff["merchant_slug"] = dff["merchant_raw"].apply(canonicalize_merchant)

# Final category
dff["Category"] = dff.apply(
    lambda r: categorize_row(r["txn_type"], r["merchant_slug"], r["Sub-description"]),
    axis=1
)

# ---------------- Preview ----------------
with st.expander("Preview data", expanded=False):
    preview_cols = [c for c in ["Date","Description","Sub-description","Amount","Debit","Credit","Category","merchant_slug","txn_type"] if c in dff.columns]
    st.write(dff[preview_cols].head(20))

# ---------------- KPIs (if Amount exists) ----------------
if "Amount" in dff.columns:
    total_in = float(dff.loc[dff["Amount"] > 0, "Amount"].sum())
    total_out = float(dff.loc[dff["Amount"] < 0, "Amount"].sum())
    net = total_in + total_out

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Inflows", f"${total_in:,.2f}")
    k2.metric("Total Outflows", f"${total_out:,.2f}")
    k3.metric("Net", f"${net:,.2f}")

# ---------------- Pie chart by category ----------------
st.subheader("Expenses by Category")

if "Amount" in dff.columns:
    # Use monetary amounts (sum of absolute negatives)
    expenses = dff.loc[dff["Amount"] < 0].copy()
    if expenses.empty:
        st.info("No expenses found in the selected period.")
    else:
        by_cat = (
            expenses.groupby("Category", as_index=False)["Amount"]
            .sum()
            .assign(Amount=lambda x: x["Amount"].abs())
            .sort_values("Amount", ascending=False)
        )
        c1, c2 = st.columns([2,1], gap="large")
        with c1:
            fig_pie = px.pie(by_cat, names="Category", values="Amount", title="Expenses by Category")
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.write("Totals by Category")
            st.dataframe(by_cat.rename(columns={"Amount": "Total Spent"}), use_container_width=True)

else:
    # Fallback: no Amount column → plot by transaction COUNT
    expenses = dff.copy()
    # If Description exists, consider excluding Income-like types from "expenses" count
    if "txn_type" in expenses.columns:
        expenses = expenses.loc[~expenses["txn_type"].isin(["payroll_deposit","transfer_in","deposit"])].copy()
    if expenses.empty:
        st.info("No expense-like transactions to display.")
    else:
        by_cat_count = (
            expenses.groupby("Category", as_index=False)
            .size()
            .sort_values("size", ascending=False)
        )
        c1, c2 = st.columns([2,1], gap="large")
        with c1:
            fig_pie = px.pie(by_cat_count, names="Category", values="size", title="Expenses by Category (count)")
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.write("Counts by Category")
            st.dataframe(by_cat_count.rename(columns={"size": "Transactions"}), use_container_width=True)

# ---------------- Extra: Top negative transactions (if Amount) ----------------
if "Amount" in dff.columns and not dff.empty:
    st.subheader("Top 15 Expenses (Most Negative Amounts)")
    top_exp = dff.loc[dff["Amount"] < 0].nsmallest(15, "Amount")
    cols = [c for c in ["Date","Description","Sub-description","Amount","Category"] if c in top_exp.columns]
    st.dataframe(top_exp[cols], use_container_width=True)
