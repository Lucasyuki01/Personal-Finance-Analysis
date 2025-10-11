from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import re
import pandas as pd
import plotly.express as px
import streamlit as st


REQUIRED_COLUMNS: Sequence[str] = (
    "Date",
    "Description",
    "Sub-description",
    "Amount",
    "Class",
    "Category",
    "Sub-Category",
)


def normalize_text(s: str) -> str:
    t = (str(s) or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def classify_pos_purchase(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    """
    Apply POS rules only to rows with Description == 'pos purchase'.
    First match wins; rules are evaluated in ascending priority.
    """
    df = df.copy()

    is_pos = df["Description"].fillna("").map(normalize_text).eq("pos purchase")
    sub_norm = df["Sub-description"].fillna("").map(normalize_text)

    if "Category" not in df.columns:
        df["Category"] = "Others"
    if "Sub-Category" not in df.columns:
        df["Sub-Category"] = "None"

    rules_sorted = sorted(rules, key=lambda r: r.get("priority", 9999))
    for r in rules_sorted:
        if r.get("match_type") == "regex":
            r["_compiled"] = re.compile(r["pattern"], flags=re.I)

    matched = pd.Series(False, index=df.index)
    for r in rules_sorted:
        pattern = r["pattern"].lower()
        match_type = r.get("match_type", "contains")

        if match_type == "startswith":
            mask = is_pos & ~matched & sub_norm.str.startswith(pattern)
        elif match_type == "regex":
            mask = is_pos & ~matched & sub_norm.map(lambda x: bool(r["_compiled"].search(x)))
        else:
            mask = is_pos & ~matched & sub_norm.str.contains(re.escape(pattern), regex=True)

        if mask.any():
            df.loc[mask, "Category"] = r["category"]
            df.loc[mask, "Sub-Category"] = r.get("sub_category", "None")
            matched |= mask

    still_pos_unmatched = is_pos & ~matched
    df.loc[still_pos_unmatched, ["Category", "Sub-Category"]] = ["Others", "None"]

    return df


def classify_transactions(df: pd.DataFrame, pos_rules: List[Dict]) -> pd.DataFrame:
    """Classify transactions into Class, Category, Sub-Category, including POS."""
    df = df.copy()

    df["Class"] = "Expenses"
    df["Category"] = "Others"
    df["Sub-Category"] = "None"

    desc = df["Description"].fillna("").map(normalize_text)

    mask_payroll = desc.eq("payroll deposit")
    mask_correct = desc.eq("correction")
    mask_interest = desc.eq("interest")
    mask_deposit = desc.eq("deposit")

    df.loc[mask_payroll, ["Class", "Category"]] = ["Earnings", "Payment"]
    df.loc[mask_correct, ["Class", "Category"]] = ["Earnings", "earnings"]
    df.loc[mask_interest, ["Class", "Category"]] = ["Earnings", "earnings"]

    df.loc[mask_deposit, "Class"] = "Earnings"
    df.loc[mask_deposit, "Category"] = "shared bills"
    df.loc[
        mask_deposit & df["Amount"].between(300, 900, inclusive="neither"),
        "Category",
    ] = "Payment"

    mask_withdraw = desc.eq("withdrawal")
    df.loc[mask_withdraw & df.index.isin([399, 352, 289]), "Category"] = "Shopping"
    df.loc[mask_withdraw & (df["Amount"] == -1200), ["Category", "Sub-Category"]] = [
        "Bills",
        "Rent",
    ]
    df.loc[mask_withdraw & df["Category"].eq("Others"), "Category"] = "Money Sent"

    df.loc[desc.eq("bill payment"), ["Category", "Sub-Category"]] = [
        "Bills",
        "Cellphone",
    ]
    df.loc[desc.eq("service charge"), ["Category", "Sub-Category"]] = [
        "Bills",
        "Bank",
    ]

    df = classify_pos_purchase(df, pos_rules)
    return df


POS_RULES: List[Dict] = [
    {"pattern": "compass", "category": "Bills", "sub_category": "Transport", "priority": 1, "match_type": "startswith"},
    {"pattern": "compass vending", "category": "Bills", "sub_category": "Transport", "priority": 1, "match_type": "contains"},
    {"pattern": "bcf-dep self se", "category": "Services", "sub_category": "Nanaimo", "priority": 2, "match_type": "contains"},
    {"pattern": "bcf-dep self se", "category": "Services", "sub_category": "Nanaimo", "priority": 2, "match_type": "contains"},
    {"pattern": "revenue services bc", "category": "Bills", "sub_category": "Msp", "priority": 2, "match_type": "contains"},
    {"pattern": "abc*32130", "category": "Bills", "sub_category": "Gym", "priority": 2, "match_type": "contains"},
    {"pattern": "walmart", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "real cdn supers", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "save on foods", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "t t supermarket", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "h-mart", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "costco wholesal", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "urban fare", "category": "Groceries", "sub_category": "None", "priority": 5, "match_type": "contains"},
    {"pattern": "dollarama", "category": "Groceries", "sub_category": "Discount", "priority": 6, "match_type": "contains"},
    {"pattern": "dollar tree", "category": "Groceries", "sub_category": "Discount", "priority": 6, "match_type": "contains"},
    {"pattern": "london drugs", "category": "Groceries", "sub_category": "Pharmacy", "priority": 30, "match_type": "contains"},
    {"pattern": "winners", "category": "Groceries", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "7-eleven", "category": "Groceries", "sub_category": "None", "priority": 20, "match_type": "contains"},
    {"pattern": "ins market", "category": "Groceries", "sub_category": "None", "priority": 20, "match_type": "contains"},
    {"pattern": "liquor", "category": "Groceries", "sub_category": "Alcohol", "priority": 21, "match_type": "contains"},
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
    {"pattern": "charlatan", "category": "Eating Out", "sub_category": "Bar", "priority": 2, "match_type": "contains"},
    {"pattern": "hungry guys", "category": "Eating Out", "sub_category": "Bar", "priority": 2, "match_type": "contains"},
    {"pattern": "mangos kitchen", "category": "Eating Out", "sub_category": "Asian", "priority": 2, "match_type": "contains"},
    {"pattern": "bbq chicken", "category": "Eating Out", "sub_category": "Asian", "priority": 2, "match_type": "contains"},
    {"pattern": "menya raizo", "category": "Eating Out", "sub_category": "Asian", "priority": 2, "match_type": "contains"},
    {"pattern": "big way hot pot", "category": "Eating Out", "sub_category": "Asian", "priority": 2, "match_type": "contains"},
    {"pattern": "tst-jerusalem", "category": "Eating Out", "sub_category": "Middle Eastern", "priority": 2, "match_type": "contains"},
    {"pattern": "rio brazilian", "category": "Eating Out", "sub_category": "Latin", "priority": 2, "match_type": "contains"},
    {"pattern": "boteco brasil", "category": "Eating Out", "sub_category": "Latin", "priority": 2, "match_type": "contains"},
    {"pattern": "brazilliant", "category": "Eating Out", "sub_category": "Latin", "priority": 2, "match_type": "contains"},
    {"pattern": "the old spaghet", "category": "Eating Out", "sub_category": "Italian", "priority": 2, "match_type": "contains"},
    {"pattern": "trollers fish", "category": "Eating Out", "sub_category": "Canadian", "priority": 2, "match_type": "contains"},
    {"pattern": "dip co. delight", "category": "Eating Out", "sub_category": "Canadian", "priority": 2, "match_type": "contains"},
    {"pattern": "cactus club", "category": "Eating Out", "sub_category": "Apparel", "priority": 2, "match_type": "contains"},
    {"pattern": "herschel", "category": "Shopping", "sub_category": "Apparel", "priority": 2, "match_type": "contains"},
    {"pattern": "tommy hilfiger", "category": "Shopping", "sub_category": "Apparel", "priority": 2, "match_type": "contains"},
    {"pattern": "roots mcarthur", "category": "Shopping", "sub_category": "Apparel", "priority": 2, "match_type": "contains"},
    {"pattern": "under armour", "category": "Shopping", "sub_category": "Apparel", "priority": 2, "match_type": "contains"},
    {"pattern": "best buy", "category": "Shopping", "sub_category": "Electronics", "priority": 30, "match_type": "contains"},
    {"pattern": "sephora", "category": "Shopping", "sub_category": "Beauty", "priority": 30, "match_type": "contains"},
    {"pattern": "decathlon", "category": "Shopping", "sub_category": "Sports", "priority": 30, "match_type": "contains"},
    {"pattern": "sport chek", "category": "Shopping", "sub_category": "Sports", "priority": 30, "match_type": "contains"},
    {"pattern": "uniqlo", "category": "Shopping", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "nike", "category": "Shopping", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "north face", "category": "Shopping", "sub_category": "Apparel", "priority": 30, "match_type": "contains"},
    {"pattern": "browns shoes", "category": "Shopping", "sub_category": "Shoes", "priority": 30, "match_type": "contains"},
    {"pattern": "shoppers drug", "category": "Shopping", "sub_category": "Barber machine", "priority": 2, "match_type": "contains"},
    {"pattern": "daiso", "category": "Shopping", "sub_category": "Discount", "priority": 6, "match_type": "contains"},
    {"pattern": "amzn mktp", "category": "Shopping", "sub_category": "Amazon", "priority": 40, "match_type": "contains"},
    {"pattern": "aliexpress", "category": "Shopping", "sub_category": "AliExpress", "priority": 40, "match_type": "contains"},
    {"pattern": "temu", "category": "Shopping", "sub_category": "Temu", "priority": 40, "match_type": "contains"},
    {"pattern": "driver serv.cen", "category": "Services", "sub_category": "BCID", "priority": 2, "match_type": "contains"},
    {"pattern": "jay hair salon", "category": "Services", "sub_category": "Hair cut", "priority": 2, "match_type": "contains"},
    {"pattern": "cfs-safecheck", "category": "Services", "sub_category": "Certification", "priority": 2, "match_type": "contains"},
    {"pattern": "name-cheap", "category": "Services", "sub_category": "Domains", "priority": 41, "match_type": "contains"},
    {"pattern": "openai", "category": "Services", "sub_category": "Salon", "priority": 41, "match_type": "contains"},
    {"pattern": "hair salon", "category": "Services", "sub_category": "Salon", "priority": 50, "match_type": "contains"},
]


def load_bank_data() -> pd.DataFrame:
    """Load acc1.csv and acc2.csv from ../data, tagging their origin."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    acc1_path = data_dir / "acc1.csv"
    acc2_path = data_dir / "acc2.csv"

    missing = [p.name for p in (acc1_path, acc2_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required dataset(s) not found in {data_dir}: {', '.join(missing)}"
        )

    acc1 = pd.read_csv(acc1_path)
    acc2 = pd.read_csv(acc2_path)
    acc1["Account"] = "Chequing"
    acc2["Account"] = "Savings"

    return pd.concat([acc1, acc2], ignore_index=True)


def clean_bank_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the requested cleaning steps before classification."""
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    for col in ["Filter", "Type of Transaction"]:
        if col in df.columns:
            df = df.drop(columns=col)

    if "Description" in df.columns:
        df = df[df["Description"] != "customer transfer cr."]
        df = df[df["Description"] != "customer transfer dr."]

    if "Sub-description" in df.columns:
        df["Sub-description"] = (
            df["Sub-description"].fillna("none").replace("", "none")
        )
        mask_spaces = df["Sub-description"].astype(str).str.strip() == ""
        df.loc[mask_spaces, "Sub-description"] = "none"
    else:
        df["Sub-description"] = "none"

    if {"Description", "Sub-description"}.issubset(df.columns):
        mask_deposit_none = (df["Description"] == "deposit") & (
            df["Sub-description"] == "none"
        )
        df = df[~mask_deposit_none]

    if "Description" in df.columns:
        df = df[df["Description"] != "abm deposit"]

    df = df.drop(index=[400, 397, 389, 349], errors="ignore")

    return df


def classify_bank_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the classification and ensure required columns exist."""
    if df.empty:
        return df

    df["Amount"] = pd.to_numeric(df.get("Amount"), errors="coerce")
    df = df.dropna(subset=["Amount"])

    df = classify_transactions(df, POS_RULES)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown"

    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ("Class", "Category", "Sub-Category"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "Account" not in df.columns:
        df["Account"] = "Unknown"

    return df


def apply_filters(
    df: pd.DataFrame,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    classes: Sequence[str],
    categories: Sequence[str],
    subcats: Sequence[str],
) -> pd.DataFrame:
    """Apply date and categorical filters to the DataFrame."""
    if df.empty:
        return df

    start, end = date_range
    filtered = df[df["Date"].between(start, end)]

    if classes:
        filtered = filtered[filtered["Class"].isin(classes)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if subcats:
        filtered = filtered[filtered["Sub-Category"].isin(subcats)]
    return filtered


def compute_kpis(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Return total spent (abs), total earned, and their delta."""
    if df.empty:
        return 0.0, 0.0, 0.0
    spent = float(df.loc[df["Amount"] < 0, "Amount"].sum())
    earned = float(df.loc[df["Amount"] > 0, "Amount"].sum())
    spent_abs = abs(spent)
    delta = earned - spent_abs
    return spent_abs, earned, delta


def top10_expenses(df: pd.DataFrame, selected_category: str) -> pd.DataFrame:
    """Return the top 10 expenses sorted by absolute amount."""
    expenses = df[df["Amount"] < 0].copy()
    if selected_category and selected_category != "All categories":
        expenses = expenses[expenses["Category"] == selected_category]
    if expenses.empty:
        return expenses
    top10 = (
        expenses.assign(_abs=lambda s: s["Amount"].abs())
        .nlargest(10, "_abs")
        .drop(columns="_abs")
    )
    return top10


def monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate absolute expenses per Year-Month."""
    expenses = df[df["Amount"] < 0].copy()
    if expenses.empty:
        return pd.DataFrame(columns=["Month", "Total Spent"])
    expenses["Month"] = expenses["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        expenses.groupby("Month", as_index=False)["Amount"]
        .sum()
        .assign(Amount=lambda s: s["Amount"].abs())
        .rename(columns={"Amount": "Total Spent"})
    )
    return monthly


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate absolute expenses per category."""
    expenses = df[df["Amount"] < 0]
    if expenses.empty:
        return pd.DataFrame(columns=["Category", "Total Spent"])
    return (
        expenses.groupby("Category", as_index=False)["Amount"]
        .sum()
        .assign(Amount=lambda s: s["Amount"].abs())
        .rename(columns={"Amount": "Total Spent"})
    )


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    st.set_page_config(page_title="Personal Finance Analysis", layout="wide")
    st.title("Personal Finance Analysis")
    st.caption("Dados carregados automaticamente de acc1.csv e acc2.csv.")

    try:
        bank_df_raw = load_bank_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    bank_df_clean = clean_bank_df(bank_df_raw)
    if bank_df_clean.empty:
        st.error("Nenhuma transação disponível após a limpeza obrigatória.")
        return

    bank_df = classify_bank_df(bank_df_clean)
    if bank_df.empty:
        st.error("Nenhuma transação disponível após a classificação.")
        return

    bank_df = bank_df.sort_values("Date").reset_index(drop=True)

    min_date = bank_df["Date"].min().date()
    max_date = bank_df["Date"].max().date()

    with st.sidebar:
        st.header("Filtros")
        date_input = st.date_input(
            "Intervalo de datas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if not isinstance(date_input, (list, tuple)) or len(date_input) != 2:
            st.error("Selecione as datas inicial e final.")
            return
        start_date, end_date = date_input
        if start_date > end_date:
            st.error("A data inicial deve ser menor ou igual à data final.")
            return

        class_options = sorted(bank_df["Class"].dropna().unique().tolist())
        selected_classes = st.multiselect(
            "Class",
            options=class_options,
            default=class_options,
        )

        category_options = sorted(bank_df["Category"].dropna().unique().tolist())
        selected_categories = st.multiselect(
            "Categoria",
            options=category_options,
            default=category_options,
        )

        subcat_options = sorted(bank_df["Sub-Category"].dropna().unique().tolist())
        selected_subcats = st.multiselect(
            "Sub-Categoria",
            options=subcat_options,
            default=subcat_options,
        )

        category_selector_options = ["All categories"] + category_options
        selected_category = st.selectbox(
            "Categoria para Top 10",
            options=category_selector_options,
            index=0,
        )

    filtered_df = apply_filters(
        bank_df,
        (pd.Timestamp(start_date), pd.Timestamp(end_date)),
        selected_classes,
        selected_categories,
        selected_subcats,
    )

    total_spent, total_earned, delta = compute_kpis(filtered_df)
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Total Spent", format_currency(total_spent))
    kpi_cols[1].metric("Total Earned", format_currency(total_earned))
    kpi_cols[2].metric("Delta", format_currency(delta))

    chart_col_1, chart_col_2 = st.columns(2)

    monthly_df = monthly_spending(filtered_df)
    if monthly_df.empty:
        chart_col_1.info("Sem despesas no período selecionado.")
    else:
        fig_monthly = px.bar(
            monthly_df,
            x="Month",
            y="Total Spent",
            title="Despesas Mensais (apenas gastos)",
            labels={"Month": "Mês", "Total Spent": "Total gasto"},
        )
        chart_col_1.plotly_chart(fig_monthly, use_container_width=True)

    category_df = category_distribution(filtered_df)
    if category_df.empty:
        chart_col_2.info("Sem distribuição de categorias para exibir.")
    else:
        fig_category = px.pie(
            category_df,
            names="Category",
            values="Total Spent",
            title="Distribuição de Despesas por Categoria",
        )
        chart_col_2.plotly_chart(fig_category, use_container_width=True)

    st.subheader("Top 10 Maiores Despesas")
    top10_df = top10_expenses(filtered_df, selected_category)
    if top10_df.empty:
        st.info("Nenhuma despesa encontrada com os filtros atuais.")
    else:
        display_columns = [
            "Date",
            "Description",
            "Sub-description",
            "Amount",
            "Category",
            "Sub-Category",
        ]
        available_columns = [c for c in display_columns if c in top10_df.columns]
        top10_display = top10_df.copy()
        if "Date" in top10_display.columns:
            top10_display["Date"] = top10_display["Date"].dt.date
        st.dataframe(top10_display[available_columns], use_container_width=True)

    st.caption(f"{len(filtered_df):,} transações exibidas de {len(bank_df):,} disponíveis.")


if __name__ == "__main__":
    main()
