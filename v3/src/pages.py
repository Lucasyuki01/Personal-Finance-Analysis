"""Streamlit page components."""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple
import json

import pandas as pd
import streamlit as st

from . import charts
from .constants import CATEGORY_TO_SUBCATEGORIES, EXPENSE_CATEGORIES, INCOME_CATEGORIES
from .rules import rule_key, save_pos_rules, save_specific_rules, update_pos_rule, update_specific_rule_by_id, revert_rule


SEARCH_COLUMNS = ["ID", "Description", "Sub-description", "Category", "Sub-Category"]


def _ensure_date_range_state(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    if "date_start" not in st.session_state:
        st.session_state["date_start"] = min_date
    if "date_end" not in st.session_state:
        st.session_state["date_end"] = max_date

    if st.session_state["date_start"] < min_date:
        st.session_state["date_start"] = min_date
    if st.session_state["date_end"] > max_date:
        st.session_state["date_end"] = max_date

    return pd.Timestamp(min_date), pd.Timestamp(max_date)


def _date_filter(df: pd.DataFrame) -> pd.DataFrame:
    min_date, max_date = _ensure_date_range_state(df)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            key="date_start",
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
    with col2:
        end_date = st.date_input(
            "End",
            key="date_end",
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

    if end_date < start_date:
        st.warning("End date is earlier than start date. Adjusting to match start date.")
        end_date = start_date
        st.session_state["date_end"] = end_date

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    date_series = df["Date"].dt.normalize()
    mask = (date_series >= start_ts.normalize()) & (date_series <= end_ts.normalize())
    return df.loc[mask].copy()


def _search_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    query = query.strip()
    if not query:
        return df
    mask = pd.Series([False] * len(df), index=df.index)
    for col in SEARCH_COLUMNS:
        if col in df.columns:
            mask |= df[col].astype("string").fillna("").str.contains(query, case=False, na=False, regex=False)
    return df.loc[mask]


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in ["description_norm", "sub_description_norm"] if col in df.columns]
    return df.drop(columns=drop_cols)


def _empty_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().isin(["", "none"])


def _unique_months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start_month = pd.Timestamp(start.year, start.month, 1)
    end_month = pd.Timestamp(end.year, end.month, 1)
    if start_month > end_month:
        return []
    months = pd.date_range(start_month, end_month, freq="MS")
    return list(months)


def _apply_pos_save(df: pd.DataFrame, pos_rules: Dict, key_tuple: Tuple, category: str, sub_category: str) -> Dict:
    profit, desc_norm, sub_desc_norm = key_tuple
    key_str = rule_key(profit, desc_norm, sub_desc_norm)

    mask = (
        (df["Profit"] == profit)
        & (df["description_norm"] == desc_norm)
        & (df["sub_description_norm"] == sub_desc_norm)
    )

    cat_empty = _empty_mask(df["Category"])
    sub_empty = _empty_mask(df["Sub-Category"])
    to_update = mask & (cat_empty | sub_empty)

    previous_values = {
        row["ID"]: {"Category": row["Category"], "Sub-Category": row["Sub-Category"]}
        for _, row in df.loc[to_update, ["ID", "Category", "Sub-Category"]].iterrows()
    }

    df.loc[mask & cat_empty, "Category"] = category
    df.loc[mask & sub_empty, "Sub-Category"] = sub_category

    previous_rule = update_pos_rule(pos_rules, key_str, category, sub_category)
    save_pos_rules(pos_rules)

    return {
        "action_type": "pos_save",
        "affected_row_ids": list(previous_values.keys()),
        "previous_values": previous_values,
        "rule_key": key_str,
        "rule_prev": previous_rule,
    }


def _apply_specific_save(df: pd.DataFrame, specific_rules: Dict, rule_id: str, category: str, sub_category: str) -> Dict:
    mask = df["ID"] == rule_id
    previous_values = {
        row["ID"]: {"Category": row["Category"], "Sub-Category": row["Sub-Category"]}
        for _, row in df.loc[mask, ["ID", "Category", "Sub-Category"]].iterrows()
    }

    df.loc[mask, "Category"] = category
    df.loc[mask, "Sub-Category"] = sub_category

    previous_rule = update_specific_rule_by_id(specific_rules, rule_id, category, sub_category)
    save_specific_rules(specific_rules)

    return {
        "action_type": "specific_save",
        "affected_row_ids": list(previous_values.keys()),
        "previous_values": previous_values,
        "rule_key": rule_id,
        "rule_prev": previous_rule,
    }


def _undo_last_action(df: pd.DataFrame, pos_rules: Dict, specific_rules: Dict) -> str:
    undo_stack = st.session_state.get("undo_stack", [])
    if not undo_stack:
        return "Nothing to undo."

    action = undo_stack.pop()
    previous_values = action.get("previous_values", {})
    for row_id, values in previous_values.items():
        mask = df["ID"] == row_id
        df.loc[mask, "Category"] = values.get("Category", "none")
        df.loc[mask, "Sub-Category"] = values.get("Sub-Category", "none")

    revert_rule(pos_rules, specific_rules, action.get("action_type"), action.get("rule_key"), action.get("rule_prev"))
    save_pos_rules(pos_rules)
    save_specific_rules(specific_rules)

    st.session_state["undo_stack"] = undo_stack
    return "Undo completed successfully."


def render_home(df: pd.DataFrame) -> None:
    st.title("Personal Finance Analysis App")

    filtered = _date_filter(df)

    earnings_df = filtered[filtered["Profit"] == 1]
    spend_df = filtered[filtered["Profit"] == 0]

    earned = float(earnings_df["Amount"].sum()) if not earnings_df.empty else 0.0
    spent = float(spend_df["Amount"].abs().sum()) if not spend_df.empty else 0.0
    diff = earned - spent

    col1, col2, col3 = st.columns(3)
    col1.metric("Earned", f"{earned:,.2f}")
    col2.metric("Spent", f"{spent:,.2f}")
    col3.metric("Difference", f"{diff:,.2f}")

    st.subheader("EARNINGS")
    st.markdown(f"**Total:** {earned:,.2f}")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.earnings_over_time(earnings_df), use_container_width=True)
    with c2:
        st.plotly_chart(charts.earnings_by_category_pie(earnings_df), use_container_width=True)

    st.subheader("SPENDINGS")
    st.markdown(f"**Total:** {spent:,.2f}")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.spendings_over_time(spend_df), use_container_width=True)
    with c4:
        st.plotly_chart(charts.spendings_by_category_pie(spend_df), use_container_width=True)

    st.subheader("Comparison")
    st.plotly_chart(charts.comparison_bar(earnings_df, spend_df), use_container_width=True)

    st.subheader("Transactions")
    search = st.text_input("Search transactions", key="home_search")
    display_df = _search_filter(filtered, search)
    st.dataframe(_display_df(display_df), use_container_width=True)


def render_income(df: pd.DataFrame) -> None:
    st.title("Income")
    filtered = _date_filter(df)
    income_df = filtered[filtered["Profit"] == 1]

    total_income = float(income_df["Amount"].sum()) if not income_df.empty else 0.0

    start = pd.Timestamp(st.session_state.get("date_start"))
    end = pd.Timestamp(st.session_state.get("date_end"))
    months = _unique_months_in_range(start, end)
    months_count = max(len(months), 1)

    salary_df = income_df[income_df["Category"] == "Salary"]
    if not salary_df.empty:
        salary_df = salary_df.copy()
        salary_df["Month"] = salary_df["Date"].dt.to_period("M").dt.to_timestamp()
        salary_by_month = salary_df.groupby("Month")["Amount"].sum()
        salary_by_month = salary_by_month.reindex(months, fill_value=0)
        avg_salary = float(salary_by_month.mean())
    else:
        avg_salary = 0.0

    avg_earnings = total_income / months_count if months_count else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"{total_income:,.2f}")
    col2.metric("Average Salary", f"{avg_salary:,.2f}")
    col3.metric("Avg Earnings", f"{avg_earnings:,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.income_by_category_stacked(income_df), use_container_width=True)
    with c2:
        st.plotly_chart(charts.category_pie(income_df, "Income Distribution"), use_container_width=True)

    st.subheader("Income Transactions")
    search = st.text_input("Search income", key="income_search")
    display_df = _search_filter(income_df, search)
    st.dataframe(_display_df(display_df), use_container_width=True)

    st.subheader("Sub-Category")
    category = st.selectbox("Category", INCOME_CATEGORIES, key="income_category")

    category_df = income_df[income_df["Category"] == category]
    total_category = float(category_df["Amount"].sum()) if not category_df.empty else 0.0
    st.markdown(f"**Total for {category}:** {total_category:,.2f}")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.category_time_distribution(category_df, "Amount", f"{category} Over Time"), use_container_width=True)
    with c4:
        st.plotly_chart(charts.income_subcategory_pie(category_df), use_container_width=True)

    st.dataframe(_display_df(category_df), use_container_width=True)


def render_expenses(df: pd.DataFrame) -> None:
    st.title("Expenses")
    filtered = _date_filter(df)
    expense_df = filtered[filtered["Profit"] == 0]

    total_spent = float(expense_df["Amount"].abs().sum()) if not expense_df.empty else 0.0
    st.metric("Total Spent", f"{total_spent:,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.spend_by_weekday_category(expense_df), use_container_width=True)
    with c2:
        st.plotly_chart(charts.category_pie_spent(expense_df, "Expense Distribution"), use_container_width=True)

    st.plotly_chart(charts.spend_over_time(expense_df), use_container_width=True)

    st.subheader("Expense Transactions")
    search = st.text_input("Search expenses", key="expense_search")
    display_df = _search_filter(expense_df, search)
    st.dataframe(_display_df(display_df), use_container_width=True)

    st.subheader("Sub-Category")
    category = st.selectbox("Category", EXPENSE_CATEGORIES, key="expense_category")
    category_df = expense_df[expense_df["Category"] == category]
    total_category = float(category_df["Amount"].abs().sum()) if not category_df.empty else 0.0
    st.markdown(f"**Total for {category}:** {total_category:,.2f}")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.category_time_distribution(category_df.assign(Amount=category_df["Amount"].abs()), "Amount", f"{category} Over Time"), use_container_width=True)
    with c4:
        st.plotly_chart(charts.expense_subcategory_pie(category_df), use_container_width=True)

    st.dataframe(_display_df(category_df), use_container_width=True)


def render_uncategorized(df: pd.DataFrame, pos_rules: Dict, specific_rules: Dict) -> None:
    st.title("Uncategorized items & Edits")

    filtered = _date_filter(df)
    uncategorized = filtered[
        (filtered["Category"].astype("string").str.strip().str.lower() == "none")
        | (filtered["Sub-Category"].astype("string").str.strip().str.lower() == "none")
    ]

    st.subheader("Uncategorized")
    st.dataframe(_display_df(uncategorized), use_container_width=True)

    st.subheader("Unique classification (POS only)")
    pos_candidates = filtered[
        (filtered["description_norm"] == "pos purchase")
        & (
            (filtered["Category"].astype("string").str.strip().str.lower() == "none")
            | (filtered["Sub-Category"].astype("string").str.strip().str.lower() == "none")
        )
        & (filtered["description_norm"] != "withdrawal")
    ]

    if pos_candidates.empty:
        st.info("No POS purchases pending classification.")
    else:
        pos_candidates = pos_candidates.copy()
        pos_candidates["unique_key"] = list(
            zip(
                pos_candidates["Profit"],
                pos_candidates["description_norm"],
                pos_candidates["sub_description_norm"],
            )
        )
        unique_keys = pos_candidates["unique_key"].drop_duplicates().tolist()
        label_map = {}
        for key in unique_keys:
            profit, desc, sub_desc = key
            label = f"{'Income' if profit == 1 else 'Expense'} | {desc} | {sub_desc or '(empty)'}"
            label_map[label] = key

        selected_label = st.selectbox("Unique POS group", list(label_map.keys()), key="pos_unique")
        selected_key = label_map[selected_label]

        profit = selected_key[0]
        if profit == 1:
            categories = INCOME_CATEGORIES
        else:
            categories = EXPENSE_CATEGORIES

        selected_category = st.selectbox("Category", categories, key="pos_category")
        subcats = CATEGORY_TO_SUBCATEGORIES.get(selected_category, ["Other"])
        selected_subcategory = st.selectbox("Sub-Category", subcats, key="pos_subcategory")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save POS classification", key="pos_save"):
                action = _apply_pos_save(df, pos_rules, selected_key, selected_category, selected_subcategory)
                st.session_state.setdefault("undo_stack", []).append(action)
                st.success(
                    f"Unique: {selected_label} saved as {selected_category} - {selected_subcategory} successfully!"
                )
                st.rerun()
        with col2:
            if st.button("Undo POS", key="pos_undo"):
                message = _undo_last_action(df, pos_rules, specific_rules)
                st.success(message)

    st.subheader("Specific classification (by ID)")
    left, right = st.columns([1, 2])

    with left:
        search_id = st.text_input("Search by ID", key="specific_id")

        withdrawal_rows = filtered[filtered["description_norm"] == "withdrawal"]
        options = []
        option_map = {}

        if not withdrawal_rows.empty:
            for _, row in withdrawal_rows.iterrows():
                label = f"{row['ID']} | {row['Description']} | {row['Amount']:.2f}"
                options.append(label)
                option_map[label] = row["ID"]

        if search_id:
            match = filtered[filtered["ID"] == search_id]
            if not match.empty:
                row = match.iloc[0]
                label = f"{row['ID']} | {row['Description']} | {row['Amount']:.2f}"
                if label not in option_map:
                    options.insert(0, label)
                    option_map[label] = row["ID"]
                st.session_state["specific_select"] = label

        if not options:
            st.info("No withdrawal transactions found in this range.")
        else:
            selected_label = st.selectbox("Transaction", options, key="specific_select")
            selected_id = option_map[selected_label]

            row = filtered[filtered["ID"] == selected_id].iloc[0]
            profit = int(row["Profit"])
            categories = INCOME_CATEGORIES if profit == 1 else EXPENSE_CATEGORIES

            category = st.selectbox("Category", categories, key="specific_category")
            subcats = CATEGORY_TO_SUBCATEGORIES.get(category, ["Other"])
            sub_category = st.selectbox("Sub-Category", subcats, key="specific_subcategory")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Specific", key="specific_save"):
                    action = _apply_specific_save(df, specific_rules, selected_id, category, sub_category)
                    st.session_state.setdefault("undo_stack", []).append(action)
                    st.success("Specific rule saved successfully!")
            with col2:
                if st.button("Undo Specific", key="specific_undo"):
                    message = _undo_last_action(df, pos_rules, specific_rules)
                    st.success(message)

    with right:
        st.subheader("Transactions")
        search = st.text_input("Search transactions", key="specific_search")
        display_df = _search_filter(filtered, search)
        st.dataframe(_display_df(display_df), use_container_width=True)


def render_downloads(df: pd.DataFrame, pos_rules: Dict, specific_rules: Dict) -> None:
    st.title("Save/Download files")

    export_df = _display_df(df)
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download processed CSV", data=csv_data, file_name="processed_transactions.csv")

    pos_json = json.dumps(pos_rules, indent=2).encode("utf-8")
    st.download_button("Download pos_rules.json", data=pos_json, file_name="pos_rules.json")

    specific_json = json.dumps(specific_rules, indent=2).encode("utf-8")
    st.download_button("Download specific_rules.json", data=specific_json, file_name="specific_rules.json")

    zip_buffer = BytesIO()

    import zipfile

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("processed_transactions.csv", csv_data)
        zf.writestr("pos_rules.json", pos_json)
        zf.writestr("specific_rules.json", specific_json)

    st.download_button("Download all (ZIP)", data=zip_buffer.getvalue(), file_name="finance_outputs.zip")
