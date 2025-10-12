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

BASE_DIR = Path(__file__).resolve().parent.parent
POS_RULES_PATH = BASE_DIR / "config" / "pos_rules.csv"
MANUAL_OVERRIDES_PATH = BASE_DIR / "config" / "manual_overrides.csv"


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
    df.loc[df["Amount"] > 0, "Class"] = "Earnings"
    df.loc[df["Amount"] < 0, "Class"] = "Expenses"
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


def load_pos_rules() -> List[Dict]:
    """Load POS classification rules from CSV."""
    if not POS_RULES_PATH.exists():
        raise FileNotFoundError(
            f"POS rules file not found at {POS_RULES_PATH}. Please create it."
        )

    rules_df = pd.read_csv(POS_RULES_PATH)
    required_cols = {"pattern", "category", "sub_category", "priority", "match_type"}
    columns_map = {col.lower(): col for col in rules_df.columns}
    missing_cols = required_cols.difference(columns_map.keys())
    if missing_cols:
        raise ValueError(f"POS rules CSV is missing columns: {', '.join(sorted(missing_cols))}")

    rules_df = rules_df.rename(columns={columns_map[key]: key for key in required_cols})

    rules_df["priority"] = pd.to_numeric(rules_df["priority"], errors="coerce").fillna(9999).astype(int)
    rules_df["match_type"] = rules_df["match_type"].fillna("contains").str.lower()
    rules_df["sub_category"] = rules_df["sub_category"].fillna("None")

    return rules_df.to_dict(orient="records")


def load_manual_overrides() -> pd.DataFrame:
    """Load manual classification overrides from CSV."""
    if not MANUAL_OVERRIDES_PATH.exists():
        return pd.DataFrame(
            columns=["Date", "Description", "Sub-description", "Amount", "Category", "Sub-Category"]
        )

    overrides = pd.read_csv(MANUAL_OVERRIDES_PATH)
    expected_cols = {"Date", "Description", "Sub-description", "Amount", "Category", "Sub-Category"}
    missing = expected_cols.difference(overrides.columns)
    if missing:
        raise ValueError(
            f"Manual overrides CSV is missing columns: {', '.join(sorted(missing))}"
        )
    overrides["Date"] = pd.to_datetime(overrides["Date"], errors="coerce")
    overrides["Amount"] = pd.to_numeric(overrides["Amount"], errors="coerce")
    overrides = overrides.dropna(subset=["Date", "Description", "Amount", "Category"])
    overrides["Sub-Category"] = overrides["Sub-Category"].fillna("None")
    return overrides


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

    acc1 = pd.read_csv(acc1_path).copy()
    acc2 = pd.read_csv(acc2_path).copy()
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

    #if "Sub-description" in df.columns:
    #    df["Sub-description"] = (
    #        df["Sub-description"].fillna("none").replace("", "none")
    #    )
    #    mask_spaces = df["Sub-description"].astype(str).str.strip() == ""
    #    df.loc[mask_spaces, "Sub-description"] = "none"
    #else:
    #    df["Sub-description"] = "none"

    if "Sub-description" in df.columns:
        sub_desc = df["Sub-description"].fillna("").astype(str).str.strip()
        sub_desc = sub_desc.replace("", "none")
        df["Sub-description"] = sub_desc
    else:
        df["Sub-description"] = "none"

    # if {"Description", "Sub-description"}.issubset(df.columns):
    #     mask_deposit_none = (df["Description"] == "deposit") & (
    #         df["Sub-description"] == "none"
    #     )
    #     df = df[~mask_deposit_none]

    # if "Description" in df.columns:
    #     df = df[df["Description"] != "abm deposit"]

    # df = df.drop(index=[400, 397, 389, 349], errors="ignore")

    return df


def apply_manual_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Apply manual category overrides using Date/Description/Sub-description/Amount."""
    if overrides.empty:
        return df

    df = df.copy()
    key_cols = ["Date", "Description", "Sub-description", "Amount"]

    def normalize_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower()

    overrides = overrides.copy()
    overrides["Date_key"] = overrides["Date"].dt.normalize()
    overrides["Description_key"] = normalize_series(overrides["Description"])
    overrides["Sub-description_key"] = normalize_series(overrides["Sub-description"])

    overrides_map = {
        (
            row["Date_key"],
            row["Description_key"],
            row["Sub-description_key"],
            float(row["Amount"]),
        ): (row["Category"], row["Sub-Category"])
        for _, row in overrides.iterrows()
    }

    df["Date_key"] = df["Date"].dt.normalize()
    df["Description_key"] = normalize_series(df["Description"])
    df["Sub-description_key"] = normalize_series(df["Sub-description"])

    mask = []
    categories = []
    subcats = []
    for _, row in df.iterrows():
        key = (
            row["Date_key"],
            row["Description_key"],
            row["Sub-description_key"],
            float(row["Amount"]),
        )
        override = overrides_map.get(key)
        if override:
            mask.append(True)
            categories.append(override[0])
            subcats.append(override[1])
        else:
            mask.append(False)
            categories.append(row["Category"])
            subcats.append(row["Sub-Category"])

    mask = pd.Series(mask, index=df.index)
    if mask.any():
        df.loc[mask, "Category"] = pd.Series(categories, index=df.index)[mask]
        df.loc[mask, "Sub-Category"] = pd.Series(subcats, index=df.index)[mask]

    df = df.drop(columns=["Date_key", "Description_key", "Sub-description_key"])
    return df


def classify_bank_df(df: pd.DataFrame, pos_rules: List[Dict], overrides: pd.DataFrame) -> pd.DataFrame:
    """Run the classification and ensure required columns exist."""
    if df.empty:
        return df

    df["Amount"] = pd.to_numeric(df.get("Amount"), errors="coerce")
    df = df.dropna(subset=["Amount"])

    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df = df.dropna(subset=["Date"])

    df = classify_transactions(df, pos_rules)
    df = apply_manual_overrides(df, overrides)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown"

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


def monthly_earnings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings per Year-Month."""
    earnings = df[df["Amount"] > 0].copy()
    if earnings.empty:
        return pd.DataFrame(columns=["Month", "Total Earned"])
    earnings["Month"] = earnings["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        earnings.groupby("Month", as_index=False)["Amount"]
        .sum()
        .rename(columns={"Amount": "Total Earned"})
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


def earnings_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings totals per category."""
    earnings = df[df["Amount"] > 0]
    if earnings.empty:
        return pd.DataFrame(columns=["Category", "Total Earned"])
    return (
        earnings.groupby("Category", as_index=False)["Amount"]
        .sum()
        .rename(columns={"Amount": "Total Earned"})
    )


def subcategory_distribution(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Aggregate absolute totals per sub-category for a given category."""
    subset = df[df["Category"] == category].copy()
    if subset.empty:
        return pd.DataFrame(columns=["Sub-Category", "Total"])
    subset["Total"] = subset["Amount"].abs()
    grouped = (
        subset.groupby("Sub-Category", as_index=False)["Total"]
        .sum()
        .sort_values("Total", ascending=False)
    )
    return grouped


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def style_pie_with_values(fig) -> None:
    """Show absolute values and percentages on pie chart slices."""
    fig.update_traces(
        texttemplate="%{label}<br>$%{value:,.2f}<br>%{percent:.1%}",
        hovertemplate="%{label}<br>$%{value:,.2f}<br>%{percent:.1%}",
        textposition="inside",
    )
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode="hide")


def earnings_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return earnings rows sorted by date descending."""
    earnings = df[df["Amount"] > 0].copy()
    if earnings.empty:
        return earnings
    return earnings.sort_values("Date", ascending=False)


def expense_stats(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """Return median spending (absolute) and DataFrame prepared for boxplot."""
    expenses = df[df["Amount"] < 0].copy()
    if expenses.empty:
        return 0.0, pd.DataFrame(columns=["Spending"])
    expenses["Spending"] = expenses["Amount"].abs()
    median_val = float(expenses["Spending"].median())
    return median_val, expenses


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

    try:
        pos_rules = load_pos_rules()
        manual_overrides = load_manual_overrides()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    bank_df_classified = classify_bank_df(bank_df_clean, pos_rules, manual_overrides)
    if bank_df_classified.empty:
        st.error("Nenhuma transação disponível após a classificação.")
        return
    bank_df_classified = bank_df_classified[bank_df_classified["Amount"] != 0].copy()
    display_df = bank_df_classified.sort_values("Date").reset_index(drop=True)

    min_date = display_df["Date"].min().date()
    max_date = display_df["Date"].max().date()

    with st.sidebar:
        st.header("Filtros")
        default_range = (min_date, max_date)
        if "date_filter" not in st.session_state:
            st.session_state["date_filter"] = default_range
        date_input = st.date_input(
            "Intervalo de datas",
            value=st.session_state["date_filter"],
            min_value=min_date,
            max_value=max_date,
            key="date_filter",
        )
        if st.button("Mostrar todo o período"):
            st.session_state["date_filter"] = default_range
            st.experimental_rerun()
        if not isinstance(date_input, (list, tuple)) or len(date_input) != 2:
            st.error("Selecione as datas inicial e final.")
            return
        start_date, end_date = date_input
        if start_date > end_date:
            st.error("A data inicial deve ser menor ou igual à data final.")
            return

        class_options = sorted(display_df["Class"].dropna().unique().tolist())
        selected_classes = st.multiselect(
            "Class",
            options=class_options,
            default=class_options,
        )

        category_options = sorted(display_df["Category"].dropna().unique().tolist())
        selected_categories = st.multiselect(
            "Categoria",
            options=category_options,
            default=category_options,
        )

        subcat_options = sorted(display_df["Sub-Category"].dropna().unique().tolist())
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
        display_df,
        (pd.Timestamp(start_date), pd.Timestamp(end_date)),
        selected_classes,
        selected_categories,
        selected_subcats,
    )

    earnings_df = earnings_table(filtered_df)
    expense_median, expense_box_df = expense_stats(filtered_df)

    total_spent, total_earned, delta = compute_kpis(filtered_df)
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Spent", format_currency(total_spent))
    kpi_cols[1].metric("Total Earned", format_currency(total_earned))
    kpi_cols[2].metric("Delta", format_currency(delta))
    kpi_cols[3].metric("Mediana dos Gastos", format_currency(expense_median))

    chart_col_1, chart_col_2 = st.columns(2)

    monthly_df = monthly_spending(filtered_df)
    with chart_col_1:
        if monthly_df.empty:
            st.info("Sem despesas no período selecionado.")
        else:
            fig_monthly = px.bar(
                monthly_df,
                x="Month",
                y="Total Spent",
                title="Despesas Mensais (apenas gastos)",
                labels={"Month": "Mês", "Total Spent": "Total gasto"},
                color_discrete_sequence=px.colors.sequential.OrRd,
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        monthly_earnings_df = monthly_earnings(filtered_df)
        if monthly_earnings_df.empty:
            st.info("Sem ganhos no período selecionado.")
        else:
            fig_earnings = px.bar(
                monthly_earnings_df,
                x="Month",
                y="Total Earned",
                title="Ganhos Mensais",
                labels={"Month": "Mês", "Total Earned": "Total ganho"},
                color_discrete_sequence=px.colors.sequential.Blues,
            )
            st.plotly_chart(fig_earnings, use_container_width=True)

    category_df = category_distribution(filtered_df)
    earnings_category_df = earnings_category_distribution(filtered_df)
    with chart_col_2:
        if category_df.empty:
            st.info("Sem distribuição de categorias de gastos.")
        else:
            fig_category = px.pie(
                category_df,
                names="Category",
                values="Total Spent",
                title="Distribuição de Despesas por Categoria",
                color_discrete_sequence=px.colors.sequential.YlOrRd,
            )
            style_pie_with_values(fig_category)
            st.plotly_chart(fig_category, use_container_width=True)

        if earnings_category_df.empty:
            st.info("Sem distribuição de categorias de ganhos.")
        else:
            fig_category_earnings = px.pie(
                earnings_category_df,
                names="Category",
                values="Total Earned",
                title="Distribuição de Ganhos por Categoria",
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            style_pie_with_values(fig_category_earnings)
            st.plotly_chart(fig_category_earnings, use_container_width=True)

    st.subheader("Ganhos no Período")
    if earnings_df.empty:
        st.info("Não há ganhos no período selecionado.")
    else:
        earnings_display = earnings_df.copy()
        if "Date" in earnings_display.columns:
            earnings_display["Date"] = earnings_display["Date"].dt.date
        earnings_columns = [
            "Date",
            "Description",
            "Sub-description",
            "Amount",
            "Category",
            "Sub-Category",
            "Account",
        ]
        available_columns = [c for c in earnings_columns if c in earnings_display.columns]
        st.dataframe(earnings_display[available_columns], use_container_width=True)

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

    st.subheader("Distribuição de Subcategorias por Categoria")
    subcat_select_options = ["Selecione uma categoria"] + category_options
    selected_category_detail = st.selectbox(
        "Escolha uma categoria para visualizar as subcategorias",
        options=subcat_select_options,
        index=0,
        key="subcat_detail_category",
    )
    if selected_category_detail != "Selecione uma categoria":
        subcat_df = subcategory_distribution(filtered_df, selected_category_detail)
        if subcat_df.empty:
            st.info("Nenhuma subcategoria disponível para a categoria selecionada.")
        else:
            fig_subcat = px.pie(
                subcat_df,
                names="Sub-Category",
                values="Total",
                title=f"Subcategorias de {selected_category_detail}",
                color_discrete_sequence=px.colors.sequential.Sunset,
            )
            style_pie_with_values(fig_subcat)
            st.plotly_chart(fig_subcat, use_container_width=True)

    st.subheader("Histórico Completo de Transações")
    if display_df.empty:
        st.info("Nenhuma transação disponível para exibir.")
    else:
        history_df = display_df.copy()
        if "Date" in history_df.columns:
            history_df["Date"] = history_df["Date"].dt.date
        history_columns = [
            "Date",
            "Description",
            "Sub-description",
            "Amount",
            "Class",
            "Category",
            "Sub-Category",
            "Account",
        ]
        available_history_cols = [c for c in history_columns if c in history_df.columns]
        st.dataframe(history_df[available_history_cols], use_container_width=True)

    st.subheader("Histórico Completo de Ganhos")
    earnings_history = display_df[display_df["Amount"] > 0].copy()
    if earnings_history.empty:
        st.info("Nenhum ganho registrado.")
    else:
        if "Date" in earnings_history.columns:
            earnings_history["Date"] = earnings_history["Date"].dt.date
        st.dataframe(
            earnings_history[available_history_cols],
            use_container_width=True,
        )

    st.subheader("Histórico Completo de Gastos")
    expenses_history = display_df[display_df["Amount"] < 0].copy()
    if expenses_history.empty:
        st.info("Nenhum gasto registrado.")
    else:
        if "Date" in expenses_history.columns:
            expenses_history["Date"] = expenses_history["Date"].dt.date
        st.dataframe(
            expenses_history[available_history_cols],
            use_container_width=True,
        )

    st.subheader("Transações com Categoria 'Others'")
    others_history = display_df[display_df["Category"] == "Others"].copy()
    if others_history.empty:
        st.info("Nenhuma transação na categoria 'Others'.")
    else:
        if "Date" in others_history.columns:
            others_history["Date"] = others_history["Date"].dt.date
        st.dataframe(
            others_history[available_history_cols],
            use_container_width=True,
        )

    st.caption(f"{len(filtered_df):,} transações exibidas de {len(display_df):,} disponíveis.")

    csv_download = bank_df_classified.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Baixar CSV Limpo (Completo)",
        data=csv_download,
        file_name="transacoes_classificadas.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
