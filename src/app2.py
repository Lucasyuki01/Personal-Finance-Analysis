# app.py
import os
import re
from pathlib import Path
from datetime import date
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events

# ---------------- Page config ----------------
st.set_page_config(page_title="Personal Finance Dashboard", page_icon="💸", layout="wide")
st.title("💸 Personal Finance Dashboard")

# ---------------- Helpers: loading & normalization ----------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    """
    Read a CSV (path or file-like object) and normalize common columns:
    - Strip column names
    - Coerce Date if present
    - Build Amount from Debit/Credit if needed
    - Clean text columns (Description, Sub-description variants)
    """
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.strip()

    # Unify "Sub-description" column naming (handle variants)
    sub_desc_aliases = ["Sub-description", "Sub Description", "Sub_Description", "Subdescription"]
    for col in sub_desc_aliases:
        if col in df.columns:
            if "Sub-description" not in df.columns:
                df.rename(columns={col: "Sub-description"}, inplace=True)
            break

    # Coerce Date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Amount logic
    if {"Debit", "Credit"}.issubset(df.columns):
        df["Amount"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0) - \
                       pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
    elif "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Clean text columns
    for col in ["Description", "Sub-description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    return df

def default_data_paths():
    """
    Return a list of CSV paths inside ./data (sorted by name).
    """
    data_dir = Path(os.getcwd()) / "data"
    if not data_dir.is_dir():
        return []
    paths = []
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(".csv"):
            paths.append(str(data_dir / name))
    return paths

# ---------------- Sidebar: multi-file upload with account labeling ----------------
st.sidebar.header("Data Source")
uploaded_files = st.sidebar.file_uploader("Upload one or more CSVs", type=["csv"], accept_multiple_files=True)

dfs = []
if uploaded_files:
    for f in uploaded_files:
        acc = st.sidebar.selectbox(
            f"Account for: {f.name}",
            options=["Chequing", "Savings", "Credit Card", "Other"],
            index=0,
            key=f"acc_{f.name}"
        )
        tmp = load_csv(f)
        tmp["Account"] = acc
        dfs.append(tmp)
else:
    # Fallback: auto-load all CSVs found in ./data (label guess by filename)
    paths = default_data_paths()
    for p in paths:
        tmp = load_csv(p)
        fname = os.path.basename(p).lower()
        if "sav" in fname:
            acc = "Savings"
        elif "cheq" in fname or "chk" in fname or "checking" in fname:
            acc = "Chequing"
        elif "card" in fname or "cc" in fname:
            acc = "Credit Card"
        else:
            acc = "Other"
        tmp["Account"] = acc
        dfs.append(tmp)

if not dfs:
    st.info("Upload at least one CSV or place CSV files in the `data/` folder.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# ---------------- Optional: Date filter ----------------
if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
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

# ---------------- Classification: txn type + merchant parsing ----------------
CITY_TOKENS = r"(vanco|burna|richm|nanai|n-van|north|edmond|granv|ben|vict|calga|robston)"
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
    Clean noisy Sub-description to a compact merchant slug.
    Heuristics:
      - remove POS prefixes (apos/opos/fpos/sq)
      - drop ids (#402, *32130, 1843...)
      - drop city tokens (vanco/burna/richm/...)
      - brand-specific fixes (nike/herschel/northface etc.)
      - choose a short, brand-like token
    """
    t = normalize_text(sub_desc)
    # remove leading POS-like prefixes
    t = re.sub(PREFIX_TOKENS, "", t)                      # remove POS prefixes
    # remove store numbers and mixed ids
    t = re.sub(r"[#*]?\d[\d\-]*", " ", t)
    # remove common city/abbr tokens
    t = re.sub(rf"\b{CITY_TOKENS}\b", " ", t)
    # keep letters/spaces/&/. only
    t = re.sub(r"[^a-z\s&\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # ---- brand-specific normalizations (ensure we don't pick generic tokens) ----
    low = t

    # THE NORTH FACE → northface
    if "the north face" in low or "north face" in low:
        return "northface"

    # HERSCHEL (e.g., "sp herschel sup vanco")
    if "herschel" in low:
        return "herschel"

    # NIKE
    if "nike" in low:
        return "nike"

    # KOODO MOBILE
    if "koodo" in low:
        return "koodo"

    # REVENUE SERVICES BC
    if "revenue" in low and "bc" in low:
        return "revenuebc"

    # ABC*... recurring charge (normalize to 'abc')
    if "abc" in low:
        return "abc"

    # Common brand fixes
    low = low.replace("mcdonald s", "mcdonalds").replace("a w", "a&w")
    low = low.replace("real cdn supers", "real canadian superstore")
    low = low.replace("save on foods", "save on foods")
    low = low.replace("h mart", "h-mart").replace("t t", "t&t")

    tokens = low.split()
    if not tokens:
        return ""
    # skip generic short tokens if the next looks more brand-like
    STOP_FIRST = {"sp", "mr", "ak", "the"}
    if tokens[0] in STOP_FIRST and len(tokens) > 1:
        return tokens[1]
    return tokens[0]

KNOWN_MERCHANTS = {
    # --- Groceries ---
    "walmart": "Groceries",
    "save": "Groceries",                    # Save On Foods
    "real": "Groceries",                    # Real Canadian Superstore (coarse)
    "h-mart": "Groceries",
    "urban": "Groceries",                   # Urban Fare
    "no": "Groceries",                      # No Frills
    "t": "Groceries",                       # t&t (very coarse; see also 't&t'/'tt' below)
    "t&t": "Groceries",                     # T&T Supermarket
    "tt": "Groceries",                      # T&T fallback
    "simon": "Groceries",                   # Simon's No Frills
    "ins": "Groceries",                     # INS Market
    "konbiniya": "Groceries",               # Konbiniya Japan Centre
    "santa": "Groceries",                   # Santa Barbara Market
    "costco": "Groceries",                  # or Shopping if you prefer
    "shoppers": "Health",                   # often Pharmacy/Health
    "london": "Health",                     # London Drugs (Health or Groceries, your call)

    # --- Restaurants / Fast Food / Cafés ---
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
    "butcher": "Restaurants",               # Ramen Butcher
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
    "charlatan": "Restaurants",
    "rio": "Restaurants",                   # Rio Brazilian Steakhouse
    "mangos": "Restaurants",                # Mangos Kitchen
    "menya": "Restaurants",                 # Menya Raizo
    "spaghet": "Restaurants",               # Old Spaghetti Factory (shortened)
    "oldspaghetti": "Restaurants",          # alternative normalization
    "dip": "Restaurants",                   # Dip Co. Delight
    "boteco": "Restaurants",
    "brazilliant": "Restaurants",
    "bigway": "Restaurants",                # Big Way Hot Pot
    "trollers": "Restaurants",
    "jerusalem": "Restaurants",             # Tst-Jerusalem S
    "whatafood": "Restaurants",
    "monpetitchoux": "Restaurants",         # Mon Petit Choux
    "mistericecream": "Restaurants",        # Mister Ice Cream
    "boba": "Restaurants",                  # 101 Boba
    "ricardo": "Restaurants",               # Ricardo's
    "honey": "Restaurants",                 # Honey Doughnuts
    "yogen": "Restaurants",                 # Yogen Fruz
    "javawocky": "Restaurants",
    "wendy": "Restaurants",                 # Wendy's
    "a&w": "Restaurants",                   # A&W (A W normalized)

    # --- Transport ---
    "compass": "Transport",
    "bcf": "Transport",                     # BC Ferries self-serve/dep

    # --- Shopping / Retail ---
    "dollarama": "Shopping",
    "dollar": "Shopping",                   # Dollar Tree / Dollarama variants
    "daiso": "Shopping",
    "best": "Shopping",                     # Best Buy
    "sport": "Shopping",                    # Sport Chek
    "decathlon": "Shopping",
    "winners": "Shopping",
    "sephora": "Shopping",
    "uniqlo": "Shopping",
    "roots": "Shopping",
    "north": "Shopping",                    # generic; keep but prefer 'northface'
    "northface": "Shopping",                # The North Face
    "browns": "Shopping",
    "tommy": "Shopping",                    # Tommy Hilfiger
    "home": "Shopping",                     # Home Depot
    "nike": "Shopping",
    "herschel": "Shopping",
    "underarmour": "Shopping",
    "under": "Shopping",                    # Under Armour (if slug reduced)
    "7eleven": "Shopping",                  # 7-Eleven (alt slug)
    "eleven": "Shopping",                   # if digits stripped
    "7": "Shopping",                        # very coarse fallback
    "print": "Shopping",                    # Print Print.ca
    "printprint": "Shopping",

    # --- Online / Digital / Services ---
    "amzn": "Shopping",
    "amazon": "Shopping",
    "temu": "Shopping",
    "aliexpress": "Shopping",
    "google": "Digital",
    "namecheap": "Digital",
    "name": "Digital",
    "openai": "Digital",

    # --- Utilities / Telecom ---
    "koodo": "Utilities",
    "bc": "Utilities",                      # BC Hydro (coarse)
    "hydro": "Utilities",
    "fortis": "Utilities",
    "telus": "Utilities",
    "shaw": "Utilities",
    "rogers": "Utilities",
    "fido": "Utilities",

    # --- Fees / Taxes / Government ---
    "revenue": "Fees",
    "revenuebc": "Fees",                    # Revenue Services BC
    "driver": "Fees",

    # --- Fixed expenses (as you asked, in English) ---
    "abc": "Fixed Expenses",

    # --- Entertainment / Leisure ---
    "grouse": "Entertainment",

    # --- Personal Care ---
    "jay": "Personal Care",                 # Jay Hair Salon

    # --- Other / Services / Unclear ---
    "vapes": "Other",
    "smoke": "Other",
    "nsh": "Other",
    "global": "Other",
    "nayax": "Other",
    "ann": "Other",
    "robot": "Other",
    "third": "Restaurants",                 # Third Beach Concession (food stand)
}

def categorize_row(txn_type: str, merchant_slug: str, sub_desc: str) -> str:
    """
    Category decision:
      1) by transaction type (strong signals)
      2) by known merchant slug
      3) by lightweight keyword fallback on Sub-description
    """
    # 1) Transaction type rules
    if txn_type == "payroll_deposit":
        return "Income"
    if txn_type in {"transfer_in", "deposit"}:
        return "Income"
    if txn_type in {"transfer_out", "withdrawal"}:
        return "Transfers"  # internal/external handled later
    if txn_type == "bill_payment" and ("koodo" in merchant_slug):
        return "Utilities"

    # 2) Known merchant
    if merchant_slug in KNOWN_MERCHANTS:
        return KNOWN_MERCHANTS[merchant_slug]

    # 3) Keyword fallback on sub-description
    t = normalize_text(sub_desc)
    if any(k in t for k in ["pizza","ramen","donair","shawarma","sushi","coffee","bakery","burger","chicken","bbq"]):
        return "Restaurants"
    if "liquor" in t:
        return "Shopping"
    if "compass" in t:
        return "Transport"

    return "Other"

# Ensure needed columns exist
if "Description" not in dff.columns:
    st.error("Column 'Description' is required in your CSV.")
    st.stop()
if "Sub-description" not in dff.columns:
    dff["Sub-description"] = ""

# Derived fields
dff["txn_type"] = dff["Description"].apply(map_txn_type)
dff["merchant_raw"] = dff["Sub-description"]
dff["merchant_slug"] = dff["merchant_raw"].apply(canonicalize_merchant)

# Final category (pre-internal-transfer cleanup)
dff["Category"] = dff.apply(
    lambda r: categorize_row(r["txn_type"], r["merchant_slug"], r["Sub-description"]),
    axis=1
)

# ---------------- Internal transfers detection & exclusion ----------------
TRANSFER_PATTERNS = [
    r"^mb\-transfer$",
    r"free interac e\-transfer",
    r"\be\-transfer\b",
]

def looks_like_transfer_text(s: str) -> bool:
    t = (str(s) or "").lower().strip()
    for pat in TRANSFER_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def mark_internal_transfers(dfin: pd.DataFrame) -> pd.DataFrame:
    """
    Mark internal transfers across your own accounts so they don't count as income/expense.
    Strategy:
      1) By txn_type in {transfer_in, transfer_out, withdrawal, deposit}
      2) By text (Mb-Transfer, Free Interac E-Transfer)
      3) Pair opposite amounts across different accounts on same date (if Date & Amount exist)
    Output: adds boolean column 'is_internal_transfer'
    """
    df2 = dfin.copy()

    # Base heuristics
    by_type = df2.get("txn_type", "").isin(["transfer_in", "transfer_out", "withdrawal", "deposit"])
    by_text = df2.get("Sub-description", "").apply(looks_like_transfer_text)

    df2["is_internal_transfer"] = by_type | by_text

    # Pairing: requires Date & Amount & Account
    if all(col in df2.columns for col in ["Amount", "Account"]) and "Date" in df2.columns and pd.api.types.is_datetime64_any_dtype(df2["Date"]):
        df2["_pair_key_amt"] = df2["Amount"].round(2).abs()
        df2["_pair_key_date"] = df2["Date"].dt.date

        grp_cols = ["_pair_key_date", "_pair_key_amt"]
        pair_idx = []
        for _, g in df2.groupby(grp_cols):
            if g["Amount"].gt(0).any() and g["Amount"].lt(0).any():
                if g["Account"].nunique() > 1:
                    pair_idx.extend(list(g.index))
        if pair_idx:
            df2.loc[pair_idx, "is_internal_transfer"] = True

        df2.drop(columns=["_pair_key_amt", "_pair_key_date"], inplace=True, errors="ignore")

    return df2

dff = mark_internal_transfers(dff)

# ---------------- Preview ----------------
with st.expander("Preview (first 25 rows)", expanded=False):
    show_cols = [c for c in ["Date","Account","Description","Sub-description","Amount","Debit","Credit","Category","txn_type","merchant_slug","is_internal_transfer"] if c in dff.columns]
    st.dataframe(dff[show_cols].head(25), use_container_width=True)

# ---------------- KPIs (exclude internal transfers) ----------------
if "Amount" in dff.columns:
    ext_in  = float(dff.loc[(dff["Amount"] > 0) & (~dff["is_internal_transfer"]), "Amount"].sum())
    ext_out = float(dff.loc[(dff["Amount"] < 0) & (~dff["is_internal_transfer"]), "Amount"].sum())
    net_ext = ext_in + ext_out

    k1, k2, k3 = st.columns(3)
    k1.metric("External Inflows", f"${ext_in:,.2f}")
    k2.metric("External Outflows", f"${ext_out:,.2f}")
    k3.metric("External Net", f"${net_ext:,.2f}")

# ---------------- Expenses pie (external only) + dropdown to drill ----------------
st.subheader("Expenses by Category (External Only)")

def show_category_table(df_expenses: pd.DataFrame, selected_cat: str):
    """Show a sortable table for the chosen category, with a CSV download."""
    st.markdown(f"**Selected category:** `{selected_cat}`")
    sub = df_expenses.loc[df_expenses["Category"] == selected_cat].copy()
    if sub.empty:
        st.info("No transactions in this category for the selected period.")
        return
    # Default sort: by Amount asc (most negative first) if Amount exists; else by Date desc if available
    if "Amount" in sub.columns:
        sub = sub.sort_values("Amount")
    elif "Date" in sub.columns:
        sub = sub.sort_values("Date", ascending=False)

    cols = [c for c in ["Date","Account","Description","Sub-description","Amount","Category"] if c in sub.columns]
    st.dataframe(sub[cols], use_container_width=True)

    csv_bytes = sub.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download this category as CSV",
        data=csv_bytes,
        file_name=f"expenses_{selected_cat}.csv",
        mime="text/csv",
        use_container_width=True
    )

if "Amount" in dff.columns:
    expenses = dff.loc[(dff["Amount"] < 0) & (~dff["is_internal_transfer"])].copy()
    if expenses.empty:
        st.info("No external expenses in the selected period.")
    else:
        by_cat = (
            expenses.groupby("Category", as_index=False)["Amount"]
            .sum()
            .assign(Amount=lambda x: x["Amount"].abs())
            .sort_values("Amount", ascending=False)
        )

        c1, c2 = st.columns([2,1], gap="large")
        with c1:
            # Classic pie (no click interaction)
            fig_pie = px.pie(by_cat, names="Category", values="Amount", title="External Expenses by Category")
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.write("Totals by Category")
            st.dataframe(by_cat.rename(columns={"Amount": "Total Spent"}), use_container_width=True)

        # Dropdown to drill into details
        selected_cat = st.selectbox(
            "Select a category to view transactions:",
            options=by_cat["Category"].tolist()
        )
        show_category_table(expenses, selected_cat)

else:
    # No Amount: count-based pie
    temp = dff.copy()
    if "txn_type" in temp.columns:
        temp = temp.loc[~temp["txn_type"].isin(["payroll_deposit","transfer_in","deposit"])]
    if temp.empty:
        st.info("No expense-like transactions to display.")
    else:
        by_cat_count = temp.groupby("Category", as_index=False).size().sort_values("size", ascending=False)

        c1, c2 = st.columns([2,1], gap="large")
        with c1:
            fig_pie = px.pie(by_cat_count, names="Category", values="size", title="Expenses by Category (count)")
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.write("Counts by Category")
            st.dataframe(by_cat_count.rename(columns={"size": "Transactions"}), use_container_width=True)

        selected_cat = st.selectbox(
            "Select a category to view transactions:",
            options=by_cat_count["Category"].tolist()
        )
        show_category_table(temp, selected_cat)

# ---------------- Top expenses table (external only) ----------------
if "Amount" in dff.columns:
    top_exp = dff.loc[(dff["Amount"] < 0) & (~dff["is_internal_transfer"])]
    if not top_exp.empty:
        st.subheader("Top 20 External Expenses (Most Negative Amounts)")
        cols = [c for c in ["Date","Account","Description","Sub-description","Amount","Category"] if c in top_exp.columns]
        st.dataframe(top_exp.nsmallest(20, "Amount")[cols], use_container_width=True)

# ---------------- Internal transfers overview (optional) ----------------
if "Amount" in dff.columns and "Account" in dff.columns:
    transfers = dff.loc[dff["is_internal_transfer"] & dff["Amount"].ne(0)].copy()
    if not transfers.empty:
        st.subheader("Internal Transfers (for audit, excluded from KPIs)")
        flows = transfers.groupby("Account", as_index=False)["Amount"].sum().sort_values("Amount")
        st.dataframe(flows.assign(Amount=lambda x: x["Amount"].round(2)), use_container_width=True)
