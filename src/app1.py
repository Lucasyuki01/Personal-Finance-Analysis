# app1.py  —  LLM-only categorization (no local dictionary)
# UI: English  •  Comments: English  •  Secrets: .env (never commit)

import os
import re
import csv
import time
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional
import json

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------------------
# Secure key loading from .env (do not commit your secrets)
# --------------------------------------------------------------------------------------
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(dotenv_path=find_dotenv(), override=False)
except Exception:
    pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    st.warning("OpenAI key not found. Create a local .env with OPENAI_API_KEY (do not commit it).")

# Create OpenAI client compatibly (supports both Responses and Chat Completions)
USE_RESPONSES_API = False
client = None
try:
    from openai import OpenAI  # SDK >= 1.x
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    # Detect Responses API availability
    USE_RESPONSES_API = hasattr(client, "responses")
except Exception:
    client = None
    USE_RESPONSES_API = False

# --------------------------------------------------------------------------------------
# App config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Personal Finance Dashboard (LLM-only)", page_icon="🤖💸", layout="wide")
st.title("🤖💸 Personal Finance Dashboard (LLM categorization)")

RULES_DIR = Path("config")
RULES_DIR.mkdir(parents=True, exist_ok=True)
LLM_CACHE_PATH = RULES_DIR / "llm_cache.csv"  # persistent cache: description_clean -> category

# --------------------------------------------------------------------------------------
# Sidebar toggles
# --------------------------------------------------------------------------------------
st.sidebar.header("Settings")
LLM_ONLY = st.sidebar.toggle("Use LLM for all rows (no local rules)", value=True)
SHORTCIRCUIT_TYPES = st.sidebar.toggle("Short-circuit obvious types (Income/Transfers/Utilities)", value=True)
st.sidebar.caption("Tip: Keep short-circuit ON to save tokens and keep KPIs clean.")

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    """Read CSV and normalize common columns."""
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.strip()

    # Unify "Sub-description" naming
    for alias in ["Sub-description", "Sub Description", "Sub_Description", "Subdescription"]:
        if alias in df.columns:
            if "Sub-description" not in df.columns:
                df.rename(columns={alias: "Sub-description"}, inplace=True)
            break

    # Coerce Date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Build/Coerce Amount
    if {"Debit", "Credit"}.issubset(df.columns):
        df["Amount"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0) - \
                       pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
    elif "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Clean texts
    for col in ["Description", "Sub-description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    return df

def default_data_paths() -> List[str]:
    data_dir = Path("data")
    if not data_dir.is_dir():
        return []
    return [str(data_dir / n) for n in sorted(os.listdir(data_dir)) if n.lower().endswith(".csv")]

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
    # Auto-load from ./data and infer account by filename
    for p in default_data_paths():
        tmp = load_csv(p)
        fname = os.path.basename(p).lower()
        if "sav" in fname:
            acc = "Savings"
        elif any(k in fname for k in ["cheq", "chk", "checking"]):
            acc = "Chequing"
        elif any(k in fname for k in ["card", "cc"]):
            acc = "Credit Card"
        else:
            acc = "Other"
        tmp["Account"] = acc
        dfs.append(tmp)

if not dfs:
    st.info("Upload at least one CSV or place CSV files in the `data/` folder.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# --------------------------------------------------------------------------------------
# Optional date filter
# --------------------------------------------------------------------------------------
if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
    dft = df.dropna(subset=["Date"]).sort_values("Date").copy()
    if not dft.empty:
        min_d = dft["Date"].min().date()
        max_d = dft["Date"].max().date()
        start, end = st.slider("Time range", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM-DD")
        mask = (dft["Date"].dt.date >= start) & (dft["Date"].dt.date <= end)
        dff = dft.loc[mask].copy()
    else:
        dff = df.copy()
else:
    dff = df.copy()

# --------------------------------------------------------------------------------------
# Canonicalization & txn type
# --------------------------------------------------------------------------------------
CITY_TOKENS = r"(vanco|burna|richm|nanai|n-van|north|edmond|granv|ben|vict|calga|howe|robston|new|w)"
PREFIX_TOKENS = r"^(apos|opos|fpos|sq)\s+"

def normalize_text(s: str) -> str:
    t = (str(s) or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def map_txn_type(desc: str) -> str:
    d = normalize_text(desc)
    if d.startswith("payroll deposit"):   return "payroll_deposit"
    if d.startswith("bill payment"):      return "bill_payment"
    if d.startswith("customer transfer cr"): return "transfer_in"
    if d.startswith("customer transfer dr"): return "transfer_out"
    if d.startswith("withdrawal"):        return "withdrawal"
    if d.startswith("deposit"):           return "deposit"
    if d.startswith("pos purchase"):      return "pos_purchase"
    if d.startswith("correction"):        return "correction"
    return "other"

def canonicalize_merchant(sub_desc: str) -> str:
    """
    Clean noisy Sub-description to a compact merchant slug (brand-like).
    Keeps it simple; LLM does the heavy lifting. We only normalize obvious noise.
    """
    t = normalize_text(sub_desc)
    t = re.sub(PREFIX_TOKENS, "", t)                      # remove POS prefixes
    t = re.sub(r"[#*]?\d[\d\-]*", " ", t)                 # store numbers/ids
    t = re.sub(rf"\b{CITY_TOKENS}\b", " ", t)             # city tokens
    t = re.sub(r"[^a-z\s&\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # a couple of safe brand compressions
    if "the north face" in t or "north face" in t: return "northface"
    if "under armour" in t or "underarmour" in t:  return "underarmour"
    if "rain or shi" in t: return "rain"
    if "a w" in t: return "a&w"

    tokens = t.split()
    if not tokens:
        return ""
    if tokens[0] in {"sp", "mr", "ak", "the"} and len(tokens) > 1:
        return tokens[1]
    return tokens[0]

# Derived fields
if "Description" not in dff.columns:
    st.error("Column 'Description' is required in your CSV.")
    st.stop()
if "Sub-description" not in dff.columns:
    dff["Sub-description"] = ""

dff["txn_type"] = dff["Description"].apply(map_txn_type)
dff["merchant_slug"] = dff["Sub-description"].apply(canonicalize_merchant)

# --------------------------------------------------------------------------------------
# Internal transfers detection (to keep KPIs clean)
# --------------------------------------------------------------------------------------
TRANSFER_PATTERNS = [r"^mb\-transfer$", r"free interac e\-transfer", r"\be\-transfer\b"]

def looks_like_transfer_text(s: str) -> bool:
    t = (str(s) or "").lower().strip()
    return any(re.search(p, t) for p in TRANSFER_PATTERNS)

def mark_internal_transfers(dfin: pd.DataFrame) -> pd.DataFrame:
    """
    Mark internal transfers so they don't count as income/expense.
    Strategy:
      1) txn_type in {transfer_in, transfer_out, withdrawal, deposit}
      2) text patterns (Mb-Transfer, Interac E-Transfer)
      3) pair +/- same amount across different accounts on same date (if Date+Amount exist)
    """
    df2 = dfin.copy()
    by_type = df2.get("txn_type", "").isin(["transfer_in", "transfer_out", "withdrawal", "deposit"])
    by_text = df2.get("Sub-description", "").apply(looks_like_transfer_text)
    df2["is_internal_transfer"] = by_type | by_text

    if all(col in df2.columns for col in ["Amount", "Account"]) and "Date" in df2.columns and pd.api.types.is_datetime64_any_dtype(df2["Date"]):
        df2["_pair_key_amt"] = df2["Amount"].round(2).abs()
        df2["_pair_key_date"] = df2["Date"].dt.date
        pair_idx = []
        for _, g in df2.groupby(["_pair_key_date", "_pair_key_amt"]):
            if g["Amount"].gt(0).any() and g["Amount"].lt(0).any():
                if g["Account"].nunique() > 1:
                    pair_idx.extend(list(g.index))
        if pair_idx:
            df2.loc[pair_idx, "is_internal_transfer"] = True
        df2.drop(columns=["_pair_key_amt", "_pair_key_date"], inplace=True, errors="ignore")
    return df2

# --------------------------------------------------------------------------------------
# LLM classification — cache + robust extraction
# --------------------------------------------------------------------------------------
ALLOWED_CATEGORIES = [
    "Groceries","Restaurants","Transport","Shopping","Health",
    "Utilities","Digital","Fees","Fixed Expenses","Entertainment",
    "Personal Care","Transfers","Income","Other"
]

def load_llm_cache() -> Dict[str, str]:
    cache = {}
    if LLM_CACHE_PATH.exists():
        try:
            with LLM_CACHE_PATH.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    cache[row.get("description_clean","")] = row.get("category","Other")
        except Exception:
            pass
    return {k: v for k, v in cache.items() if k}

def save_llm_cache(cache: Dict[str, str]) -> None:
    try:
        with LLM_CACHE_PATH.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["description_clean", "category", "model", "ts"])
            w.writeheader()
            now = pd.Timestamp.utcnow().isoformat()
            for k, v in cache.items():
                w.writerow({"description_clean": k, "category": v, "model": OPENAI_MODEL, "ts": now})
    except Exception:
        pass

LLM_CACHE = load_llm_cache()

def sanitize_for_llm(text: str) -> str:
    """Remove emails, numbers/ids, leave merchant-relevant tokens."""
    t = (str(text) or "")
    t = re.sub(r"\S+@\S+", " ", t)           # emails
    t = re.sub(r"[0-9#*_\-]+", " ", t)       # ids, numbers
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _extract_json_text(resp) -> Optional[str]:
    """
    Try multiple SDK shapes:
      - Responses API: resp.output[0].content[0].text or resp.output_text
      - Chat Completions: resp.choices[0].message.content
    """
    # Responses API convenience string
    try:
        t = getattr(resp, "output_text", None)
        if t:
            return t
    except Exception:
        pass
    # Responses API typical structure
    try:
        return resp.output[0].content[0].text
    except Exception:
        pass
    # Chat Completions structure
    try:
        return resp.choices[0].message.content
    except Exception:
        pass
    return None

def _safe_parse_category(s: str) -> str:
    """Parse a JSON string like {"category": "..."}; fall back to 'Other'."""
    try:
        data = json.loads(s)
        cat = data.get("category")
        if isinstance(cat, str) and cat:
            return cat
    except Exception:
        # Light regex fallback if the model returned text with JSON in it
        m = re.search(r'"category"\s*:\s*"([^"]+)"', s or "")
        if m:
            return m.group(1)
    return "Other"

def _prompt_text(desc_clean: str) -> str:
    return (
        "You are a bank transaction classifier.\n"
        "Given a sanitized bank transaction description (merchant and context), "
        "choose exactly ONE category from the allowed list. Reply ONLY with a compact JSON object.\n\n"
        f"Allowed categories: {', '.join(ALLOWED_CATEGORIES)}\n"
        f'Format: {{"category": "<one-of-allowed>"}}\n'
        f"Description: {desc_clean}"
    )

def llm_classify_one(description_clean: str) -> str:
    """
    Single call with robust extraction.
    No retries for parse/type errors; just return 'Other' on failure.
    """
    if client is None or not OPENAI_API_KEY:
        return "Other"

    try:
        if USE_RESPONSES_API:
            # Structured JSON schema (best)
            schema = {
                "name": "category_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ALLOWED_CATEGORIES},
                    },
                    "required": ["category"]
                },
                "strict": True
            }
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=_prompt_text(description_clean),
                temperature=0,
                response_format={"type": "json_schema", "json_schema": schema},
            )
        else:
            # Chat Completions fallback with json_object
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict classifier that always returns valid JSON."},
                    {"role": "user", "content": _prompt_text(description_clean)},
                ],
                temperature=0,
                response_format={"type": "json_object"},  # supported in modern SDKs; ignored in older
            )
        out_text = _extract_json_text(resp)
        return _safe_parse_category(out_text or "")
    except Exception:
        return "Other"

def llm_classify_series(descriptions: List[str]) -> List[str]:
    """Classify a list with per-item cache (no retries on parse/type errors)."""
    out: List[str] = []
    for d in descriptions:
        clean = sanitize_for_llm(d.lower())
        if clean in LLM_CACHE:
            out.append(LLM_CACHE[clean])
            continue
        cat = llm_classify_one(clean)
        LLM_CACHE[clean] = cat
        out.append(cat)
        save_llm_cache(LLM_CACHE)
        time.sleep(0.02)  # tiny pacing
    return out

# --------------------------------------------------------------------------------------
# Category from type (optional short-circuit to save tokens)
# --------------------------------------------------------------------------------------
def category_from_type(txn_type: str, slug: str) -> Optional[str]:
    if not SHORTCIRCUIT_TYPES:
        return None
    if txn_type == "payroll_deposit":
        return "Income"
    if txn_type in {"transfer_in", "deposit"}:
        return "Income"
    if txn_type in {"transfer_out", "withdrawal"}:
        return "Transfers"
    if txn_type == "bill_payment" and ("koodo" in slug):
        return "Utilities"
    return None

# Start with type-based category (optional)
dff["Category"] = dff.apply(lambda r: category_from_type(r["txn_type"], r["merchant_slug"]), axis=1)

# --------------------------------------------------------------------------------------
# Decide what goes to the LLM
# --------------------------------------------------------------------------------------
if LLM_ONLY:
    need_llm_mask = pd.Series(True, index=dff.index)  # send everything to LLM
else:
    # If not full LLM, only fill missing/Other and typically only purchases
    need_llm_mask = dff["Category"].isna() | dff["Category"].eq("Other")
    need_llm_mask &= dff["txn_type"].eq("pos_purchase")

texts_for_llm = (dff["Description"].astype(str) + " " + dff["Sub-description"].astype(str)).where(need_llm_mask)

if need_llm_mask.any():
    st.caption(f"LLM classifying {int(need_llm_mask.sum())} items…")
    cats = llm_classify_series(texts_for_llm[need_llm_mask].tolist())
    dff.loc[need_llm_mask, "Category_llm"] = cats
    dff["Category"] = dff["Category"].where(~need_llm_mask, dff["Category_llm"].fillna("Other"))

# --------------------------------------------------------------------------------------
# Mark internal transfers (for KPIs)
# --------------------------------------------------------------------------------------
dff = mark_internal_transfers(dff)

# --------------------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------------------
with st.expander("Preview (first 25 rows)", expanded=False):
    cols = [c for c in ["Date","Account","Description","Sub-description","Amount",
                        "Category","Category_llm","merchant_slug","txn_type","is_internal_transfer"]
            if c in dff.columns]
    st.dataframe(dff[cols].head(25), use_container_width=True)

# --------------------------------------------------------------------------------------
# KPIs (exclude internal transfers)
# --------------------------------------------------------------------------------------
if "Amount" in dff.columns:
    ext_in  = float(dff.loc[(dff["Amount"] > 0) & (~dff["is_internal_transfer"]), "Amount"].sum())
    ext_out = float(dff.loc[(dff["Amount"] < 0) & (~dff["is_internal_transfer"]), "Amount"].sum())
    net_ext = ext_in + ext_out

    k1, k2, k3 = st.columns(3)
    k1.metric("External Inflows", f"${ext_in:,.2f}")
    k2.metric("External Outflows", f"${ext_out:,.2f}")
    k3.metric("External Net", f"${net_ext:,.2f}")

# --------------------------------------------------------------------------------------
# Expenses pie (external only) + dropdown drilldown
# --------------------------------------------------------------------------------------
st.subheader("Expenses by Category (External Only)")

def show_category_table(df_expenses: pd.DataFrame, selected_cat: str):
    """Show a sortable table for the chosen category, with CSV download."""
    st.markdown(f"**Selected category:** `{selected_cat}`")
    sub = df_expenses.loc[df_expenses["Category"] == selected_cat].copy()
    if sub.empty:
        st.info("No transactions in this category for the selected period.")
        return
    if "Amount" in sub.columns:
        sub = sub.sort_values("Amount")  # negatives first
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
            fig_pie = px.pie(by_cat, names="Category", values="Amount", title="External Expenses by Category")
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.write("Totals by Category")
            st.dataframe(by_cat.rename(columns={"Amount": "Total Spent"}), use_container_width=True)
        chosen = st.selectbox("Select a category to view transactions:", options=by_cat["Category"].tolist())
        show_category_table(expenses, chosen)
else:
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
        chosen = st.selectbox("Select a category to view transactions:", options=by_cat_count["Category"].tolist())
        show_category_table(temp, chosen)

# --------------------------------------------------------------------------------------
# Top expenses (external only)
# --------------------------------------------------------------------------------------
if "Amount" in dff.columns:
    top_exp = dff.loc[(dff["Amount"] < 0) & (~dff["is_internal_transfer"])]
    if not top_exp.empty:
        st.subheader("Top 20 External Expenses (Most Negative Amounts)")
        cols = [c for c in ["Date","Account","Description","Sub-description","Amount","Category"] if c in top_exp.columns]
        st.dataframe(top_exp.nsmallest(20, "Amount")[cols], use_container_width=True)
