import os
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
# ---- Category rules persistence (load/save as JSON) ----
import json
from pathlib import Path

RULES_PATH = Path("config/category_rules.json")
RULES_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_RULES = {
    "Groceries":    ["walmart", "no frills", "save-on", "real canadian superstore", "loblaws", "costco", "safeway", "iga", "whole foods"],
    "Restaurants":  ["starbucks", "mcdonald", "tim hortons", "uber eats", "doordash", "skipthe dishes", "pizza", "sushi", "restaurant"],
    "Transport":    ["uber", "lyft", "translink", "compass", "gas", "petro", "shell", "esso", "bus", "skytrain", "taxi"],
    "Housing":      ["rent", "landlord", "bc hydro", "fortis", "internet", "shaw", "telus", "fido", "rogers"],
    "Health":       ["pharma", "drug", "shoppers", "london drugs", "clinic", "dent", "physio", "wellness"],
    "Entertainment":["netflix", "spotify", "youtube", "prime", "cinema", "steam", "playstation", "xbox"],
    "Shopping":     ["amazon", "best buy", "winners", "dollarama", "ikea", "mall", "sportchek", "sephora"],
    "Education":    ["tuition", "college", "course", "udemy", "coursera", "textbook"],
    "Fees":         ["fee", "charge", "interest", "nsf", "overdraft"],
    "Income":       ["payroll", "deposit", "transfer in", "refund", "interac e-transfer autodeposit"],
    "Other":        []
}

def load_rules() -> dict:
    if RULES_PATH.exists():
        with RULES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_RULES.copy()

def save_rules(rules: dict) -> None:
    with RULES_PATH.open("w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

# ---------- Page config ----------
st.set_page_config(page_title="Personal Finance Dashboard", page_icon="💸", layout="wide")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    """
    Reads a CSV from a path or a file-like object (e.g., Streamlit uploader).
    Returns a pandas DataFrame with normalized columns.
    Cached so it won't re-run on every widget change.
    """
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.strip()

    # Try to coerce common column names; adapt to your export if needed.
    # Example: unify an 'Amount' column (positive=inflow, negative=outflow)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # If your export has separate Debit/Credit columns, build Amount
    if {"Debit", "Credit"}.issubset(df.columns):
        df["Amount"] = df["Credit"].fillna(0) - df["Debit"].fillna(0)
    elif "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    return df

def default_data_path():
    """
    Looks for a CSV inside ./data and returns the first match if present.
    """
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            if name.lower().endswith(".csv"):
                return os.path.join(data_dir, name)
    return None

# ---------- Sidebar (controls) ----------
st.sidebar.header("Data Source")

uploaded = st.sidebar.file_uploader("Upload your Scotiabank CSV", type=["csv"])
use_default = st.sidebar.checkbox("Use CSV found in ./data", value=uploaded is None)

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

# ---------- Main content ----------
st.title("💸 Personal Finance Dashboard")

if df is None:
    st.info("Upload a CSV in the sidebar or place one in the `data/` folder.")
    st.stop()

# Basic data preview
with st.expander("Preview data", expanded=False):
    st.write(df.head(10))

# Simple KPIs
total_in = float(df.loc[df["Amount"] > 0, "Amount"].sum()) if "Amount" in df else 0.0
total_out = float(df.loc[df["Amount"] < 0, "Amount"].sum()) if "Amount" in df else 0.0
net = total_in + total_out

k1, k2, k3 = st.columns(3)
k1.metric("Total Inflows", f"${total_in:,.2f}")
k2.metric("Total Outflows", f"${total_out:,.2f}")
k3.metric("Net", f"${net:,.2f}")

# Time filter (optional)
# Ensure Date is datetime and cleaned
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date")

# Convert to Python date objects (NOT pandas.Timestamp)
min_d = df["Date"].min().date()
max_d = df["Date"].max().date()

start, end = st.slider(
    "Time range",
    min_value=min_d,
    max_value=max_d,
    value=(min_d, max_d),
    format="YYYY-MM-DD",
)

# Filter using .dt.date to compare date with date
mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
dff = df.loc[mask].copy()

# ---------- Categorization using only 'Description' + Pie chart ----------
import re

if "Description" not in dff.columns:
    st.error("Column 'Description' not found in your data.")
    st.stop()

# Normalize Description
dff["Description"] = dff["Description"].astype(str).fillna("").str.strip()

# Load rules (from JSON or defaults)
CATEGORY_RULES = load_rules()

def categorize_text(text: str) -> str:
    """Return a category based on keyword rules in Description; defaults to 'Other'."""
    t = text.lower()
    for cat, kws in CATEGORY_RULES.items():
        for kw in kws:
            if kw and re.search(rf"\b{re.escape(kw)}\b", t):
                return cat
    return "Other"

# Apply
dff["Category"] = dff["Description"].apply(categorize_text)

# Expenses only
expenses = pd.DataFrame()
if "Amount" not in dff.columns:
    st.warning("No 'Amount' column found to compute expenses.")
else:
    expenses = dff.loc[dff["Amount"] < 0].copy()
    if expenses.empty:
        st.warning("No expenses found in the selected period.")
    else:
        expenses["Amount_abs"] = expenses["Amount"].abs()
        by_cat = (
            expenses.groupby("Category", as_index=False)["Amount_abs"]
            .sum()
            .sort_values("Amount_abs", ascending=False)
        )
        total_expenses = by_cat["Amount_abs"].sum()
        by_cat["Percentage"] = (by_cat["Amount_abs"] / total_expenses * 100).round(2)

        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            fig_pie = px.pie(
                by_cat,
                names="Category",
                values="Amount_abs",
                title="Expenses by Category",
                hole=0.0  # change to 0.4 for a donut chart
            )
            fig_pie.update_traces(textinfo="percent+label+value")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("Totals by Category")
            st.metric("Total Expenses", f"${total_expenses:,.2f}")
            st.dataframe(
                by_cat.rename(columns={"Amount_abs": "Total Spent"}),
                use_container_width=True
            )
            
# ---------- Category Tuner (interactive rule builder) ----------
if "Amount" in dff.columns:
    other_mask = (dff["Amount"] < 0) & (dff["Category"] == "Other")
    other_desc_counts = (
        dff.loc[other_mask, "Description"]
        .value_counts()
        .head(30)
    )

    with st.expander("🔧 Category Tuner (map 'Other' to categories)"):
        if other_desc_counts.empty:
            st.write("No uncategorized ('Other') expenses in the current filter.")
        else:
            st.write("Top uncategorized descriptions:")
            st.table(other_desc_counts.rename("count"))

            # Form to add a new keyword rule
            with st.form("tuner_form", clear_on_submit=False):
                desc_choice = st.selectbox(
                    "Choose a description to categorize",
                    options=other_desc_counts.index.tolist()
                )
                # Pre-fill a reasonable keyword guess (you can edit)
                suggested_kw = desc_choice.lower()

                categories = list(CATEGORY_RULES.keys())
                # Avoid selecting 'Other' as a target category
                categories_no_other = [c for c in categories if c != "Other"]

                target_cat = st.selectbox(
                    "Target category",
                    options=categories_no_other,
                    index=0
                )
                keyword = st.text_input(
                    "Keyword to match (case-insensitive, word-boundary regex)",
                    value=suggested_kw
                )

                submitted = st.form_submit_button("Add rule and re-run")
                if submitted:
                    kw = keyword.strip().lower()
                    if not kw:
                        st.warning("Please enter a non-empty keyword.")
                    else:
                        # Append keyword to selected category
                        CATEGORY_RULES.setdefault(target_cat, [])
                        if kw not in CATEGORY_RULES[target_cat]:
                            CATEGORY_RULES[target_cat].append(kw)
                            save_rules(CATEGORY_RULES)
                            st.success(f"Added keyword '{kw}' to category '{target_cat}'.")
                            # Clear caches and rerun app so the new rule applies immediately
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.info("This keyword already exists in the chosen category.")

# Monthly overview
if {"Date", "Amount"}.issubset(dff.columns) and not dff.empty:
    monthly = dff.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").dt.to_timestamp()
    monthly["Inflows"] = monthly["Amount"].where(monthly["Amount"] > 0, 0)
    monthly["Outflows"] = monthly["Amount"].where(monthly["Amount"] < 0, 0)

    monthly_summary = (
        monthly.groupby("Month")
        .agg(
            Inflows=("Inflows", "sum"),
            Outflows=("Outflows", "sum"),
            Net=("Amount", "sum"),
        )
        .reset_index()
    )
    monthly_summary["Outflows"] = monthly_summary["Outflows"].abs()

    st.subheader("Monthly Cash Flow")
    fig_monthly = px.bar(
        monthly_summary.melt(id_vars="Month", value_vars=["Inflows", "Outflows", "Net"], var_name="Type", value_name="Amount"),
        x="Month",
        y="Amount",
        color="Type",
        barmode="group",
        title="Monthly Inflows vs Outflows",
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

# Overall inflow vs outflow comparison for selected range
if "Amount" in dff.columns:
    total_in_filtered = float(dff.loc[dff["Amount"] > 0, "Amount"].sum())
    total_out_filtered = float(dff.loc[dff["Amount"] < 0, "Amount"].sum())
    comparison_df = pd.DataFrame(
        {
            "Type": ["Inflows", "Outflows", "Net"],
            "Amount": [
                total_in_filtered,
                abs(total_out_filtered),
                total_in_filtered + total_out_filtered,
            ],
        }
    )
    st.subheader("Entradas vs Saídas")
    fig_compare = px.bar(
        comparison_df,
        x="Type",
        y="Amount",
        color="Type",
        text_auto=".2s",
        title="Comparison of Inflows and Outflows",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # Category drill-down
    if not expenses.empty:
        st.subheader("Maiores gastos por categoria")
        category_options = ["Todas"] + sorted(expenses["Category"].unique())
        selected_category = st.selectbox("Escolha uma categoria", category_options)

        if selected_category == "Todas":
            filtered_expenses = expenses.copy()
        else:
            filtered_expenses = expenses.loc[expenses["Category"] == selected_category].copy()

        if filtered_expenses.empty:
            st.info("Não há gastos para a categoria selecionada.")
        else:
            top_expenses = (
                filtered_expenses.assign(Amount_abs=lambda df_: df_["Amount"].abs())
                .nlargest(10, "Amount_abs")
                .drop(columns=["Amount_abs"])
            )
            st.dataframe(top_expenses)

# Daily totals line chart
if {"Date", "Amount"}.issubset(dff.columns):
    daily = dff.groupby("Date", as_index=False)["Amount"].sum()
    fig = px.line(daily, x="Date", y="Amount", title="Daily Net Flow")
    st.plotly_chart(fig, use_container_width=True)
