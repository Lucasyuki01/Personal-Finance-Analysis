"""Streamlit entrypoint for Personal Finance Analysis."""

from __future__ import annotations

import streamlit as st

from src.processing import process_uploaded_files
from src.rules import (
    ensure_rules_files,
    load_pos_rules,
    load_specific_rules,
    load_specific_rules_from_upload,
    save_specific_rules,
)
from src.pages import (
    render_home,
    render_income,
    render_expenses,
    render_uncategorized,
    render_custom_categories,
    render_downloads,
)


def _init_state() -> None:
    ensure_rules_files()
    if "pos_rules" not in st.session_state:
        st.session_state["pos_rules"] = load_pos_rules()
    if "specific_rules" not in st.session_state:
        st.session_state["specific_rules"] = load_specific_rules()
    if "undo_stack" not in st.session_state:
        st.session_state["undo_stack"] = []
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"


st.set_page_config(page_title="Personal Finance Analysis", layout="wide")
_init_state()

if "df_processed" not in st.session_state:
    st.title("Personal Finance Analysis App")
    uploaded_files = st.file_uploader(
        "Upload transaction files (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    specific_upload = st.file_uploader("Upload specific_rules (optional)", type=["json"], accept_multiple_files=False)

    if st.button("Process"):
        if not uploaded_files:
            st.error("Please upload at least one transactions file.")
            st.stop()

        try:
            pos_rules = load_pos_rules()
            if specific_upload is not None:
                specific_rules = load_specific_rules_from_upload(specific_upload)
                save_specific_rules(specific_rules)
            else:
                specific_rules = load_specific_rules()

            df_processed, meta = process_uploaded_files(uploaded_files, pos_rules, specific_rules)
            st.session_state["df_processed"] = df_processed
            st.session_state["pos_rules"] = pos_rules
            st.session_state["specific_rules"] = specific_rules
            st.session_state["processing_meta"] = meta
            st.success("Files processed successfully. Use the sidebar to navigate.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
            st.stop()
else:
    df = st.session_state["df_processed"]
    pos_rules = st.session_state["pos_rules"]
    specific_rules = st.session_state["specific_rules"]

    st.sidebar.title("Navigation")
    pages = ["Home", "Income", "Expenses", "Uncategorized", "Custom Categories"]
    prev_nav = st.session_state.get("nav_page", "Home")
    current_index = pages.index(prev_nav) if prev_nav in pages else 0

    selected_page = st.sidebar.radio("Pages", pages, index=current_index, key="nav_page")
    if selected_page != prev_nav:
        st.session_state["current_page"] = selected_page
    elif st.session_state["current_page"] != "Downloads":
        st.session_state["current_page"] = selected_page

    if st.sidebar.button("Save/Download files"):
        st.session_state["current_page"] = "Downloads"

    meta = st.session_state.get("processing_meta")
    if meta:
        invalid_rows = meta.get("invalid_rows_dropped", 0)
        if invalid_rows:
            st.warning(f"Dropped {invalid_rows} rows due to invalid Date or Amount.")
        per_file = meta.get("per_file", [])
        if per_file:
            with st.expander("Parsing details"):
                for item in per_file:
                    st.write(
                        f"{item.get('file')}: invalid Date={item.get('invalid_date', 0)}, invalid Amount={item.get('invalid_amount', 0)}"
                    )

    page = st.session_state["current_page"]
    if page == "Home":
        render_home(df)
    elif page == "Income":
        render_income(df, specific_rules)
    elif page == "Expenses":
        render_expenses(df, specific_rules)
    elif page == "Uncategorized":
        render_uncategorized(df, pos_rules, specific_rules)
    elif page == "Custom Categories":
        render_custom_categories(df, pos_rules, specific_rules)
    elif page == "Downloads":
        render_downloads(df, pos_rules, specific_rules)
