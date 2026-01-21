from pathlib import Path
import io
import sys
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st
import plotly.express as px


SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pfa.io import write_parquet  # noqa: E402
from pfa.pipeline import (
    run_pipeline_from_uploads,
    view_account_expenses,
    view_card_purchases,
    view_fixed_wastes,
    view_profits,
    view_wastes,
)  # noqa: E402
from rules.apply import apply_rules  # noqa: E402
from rules.store import load_rules  # noqa: E402


OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data" / "outputs" / "canonical_latest.parquet"


def main() -> None:
    st.title("Personal Finance Analysis - Canonical Viewer")

    uploads = _sidebar_controls()
    if st.session_state.get("run_pipeline_clicked"):
        st.session_state["run_pipeline_clicked"] = False
        _run_pipeline(uploads)
    if st.session_state.get("load_canonical_clicked"):
        st.session_state["load_canonical_clicked"] = False
        _load_canonical()

    canonical_df = st.session_state.get("canonical_df")
    if canonical_df is not None:
        _render_views(canonical_df)
    else:
        st.info("Upload files and run the pipeline to view data.")


def _sidebar_controls():
    with st.sidebar:
        st.header("Pipeline")
        uploads = st.file_uploader(
            "Upload CSV/XLSX files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        st.checkbox("Apply classification rules", key="apply_rules")
        if st.button("Run pipeline"):
            st.session_state["run_pipeline_clicked"] = True
        if st.button("Load saved canonical"):
            st.session_state["load_canonical_clicked"] = True
    return uploads


def _run_pipeline(uploads) -> None:
    if not uploads:
        st.error("Please upload at least one file before running the pipeline.")
        return

    with st.spinner("Running pipeline..."):
        try:
            canonical_df, _analysis_ready, _manifest = run_pipeline_from_uploads(uploads)
        except Exception as exc:  # noqa: BLE001 - surface pipeline errors
            st.error(f"Pipeline failed: {exc}")
            return

        if st.session_state.get("apply_rules"):
            rules_df = load_rules()
            if not rules_df.empty:
                canonical_df = apply_rules(canonical_df, rules_df)
            else:
                st.warning("No classification rules found; showing unclassified data.")

        write_parquet(canonical_df, OUTPUT_PATH)
        st.session_state["canonical_df"] = canonical_df
        st.success(f"Canonical dataset saved to {OUTPUT_PATH}")


def _load_canonical() -> None:
    if not OUTPUT_PATH.exists():
        st.error(f"No saved canonical dataset found at {OUTPUT_PATH}")
        return

    try:
        canonical_df = pd.read_parquet(OUTPUT_PATH)
    except Exception as exc:  # noqa: BLE001 - surface load errors
        st.error(f"Failed to load canonical dataset: {exc}")
        return

    st.session_state["canonical_df"] = canonical_df
    st.success(f"Loaded canonical dataset from {OUTPUT_PATH}")


def _render_views(canonical_df: pd.DataFrame) -> None:
    st.subheader("Summary")
    _render_summary(canonical_df)
    _render_download(canonical_df)

    st.subheader("Views")
    _render_flow_section("Profits", view_profits(canonical_df), is_waste=False)
    _render_flow_section("Wastes", view_wastes(canonical_df), is_waste=True)
    _render_simple_view("Card Purchases", view_card_purchases(canonical_df))
    _render_simple_view("Account Expenses", view_account_expenses(canonical_df))
    _render_simple_view("Fixed Wastes", view_fixed_wastes(canonical_df))


def _render_summary(canonical_df: pd.DataFrame) -> None:
    profits_df = view_profits(canonical_df)
    wastes_df = view_wastes(canonical_df)
    total_earned = _amount_series(profits_df).sum()
    total_spent = _amount_series(wastes_df).abs().sum()
    difference = total_earned - total_spent

    col_earned, col_spent, col_diff = st.columns(3)
    col_earned.metric("Total earned", _format_amount(total_earned))
    col_spent.metric("Total spent", _format_amount(total_spent))
    col_diff.metric("Difference", _format_amount(difference))


def _render_download(canonical_df: pd.DataFrame) -> None:
    file_name = OUTPUT_PATH.name
    data = None
    if OUTPUT_PATH.exists():
        try:
            data = OUTPUT_PATH.read_bytes()
        except OSError:
            data = None

    if data is None:
        buffer = io.BytesIO()
        canonical_df.to_parquet(buffer, index=False)
        buffer.seek(0)
        data = buffer.read()

    st.download_button(
        "Download canonical dataset",
        data=data,
        file_name=file_name,
        mime="application/octet-stream",
    )

def _render_flow_section(
    title: str,
    view_df: pd.DataFrame,
    is_waste: bool,
) -> None:
    st.markdown(f"### {title}")

    row_count = len(view_df)
    amounts = _amount_series(view_df)
    if is_waste:
        amounts = amounts.abs()
    total_amount = _format_amount(amounts.sum())
    st.write(f"Rows: {row_count:,} | Total amount: {total_amount}")

    if view_df.empty:
        st.info("No data available for this view.")
        return

    col_bar, col_pie = st.columns(2)
    with col_bar:
        _render_monthly_bar_chart(view_df, title, is_waste=is_waste)
    with col_pie:
        _render_category_pie_chart(view_df, title, is_waste=is_waste)

    table_df = _flow_table(view_df)
    st.dataframe(table_df, use_container_width=True)


def _render_simple_view(
    title: str,
    view_df: pd.DataFrame,
) -> None:
    st.markdown(f"### {title}")

    row_count = len(view_df)
    total_amount = _sum_amount(view_df)
    st.write(f"Rows: {row_count:,} | Total amount: {total_amount}")

    st.dataframe(view_df, use_container_width=True)


def _render_monthly_bar_chart(
    view_df: pd.DataFrame,
    title: str,
    is_waste: bool,
) -> None:
    monthly = _monthly_totals(view_df, is_waste=is_waste)
    if monthly.empty:
        st.info("No monthly data available for this view.")
        return

    palette = _flow_palette(is_waste)
    bar_color = palette[4] if len(palette) > 4 else palette[-1]
    fig = px.bar(
        monthly,
        x="Month",
        y="Total",
        labels={"Month": "Month", "Total": "Total"},
        title=f"{title} per month",
    )
    fig.update_traces(marker_color=bar_color)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_category_pie_chart(
    view_df: pd.DataFrame,
    title: str,
    is_waste: bool,
) -> None:
    categories = _category_totals(view_df, is_waste=is_waste)
    if categories.empty:
        st.info("No category data available for this view.")
        return

    fig = px.pie(
        categories,
        names="Category",
        values="Total",
        title=f"{title} by category",
        color_discrete_sequence=_flow_palette(is_waste),
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _monthly_totals(df: pd.DataFrame, is_waste: bool) -> pd.DataFrame:
    date_col = _pick_column(df, ["date", "transaction_date"])
    if date_col is None:
        return pd.DataFrame()

    dates = pd.to_datetime(df[date_col], errors="coerce")
    valid_mask = dates.notna()
    if not valid_mask.any():
        return pd.DataFrame()

    amounts = _amount_series(df).abs() if is_waste else _amount_series(df)
    amounts = amounts.loc[valid_mask]
    months = dates.loc[valid_mask].dt.to_period("M").dt.to_timestamp()
    grouped = amounts.groupby(months).sum().sort_index()
    return pd.DataFrame({"Month": grouped.index, "Total": grouped.values})


def _category_totals(df: pd.DataFrame, is_waste: bool) -> pd.DataFrame:
    category_col = _pick_column(df, ["category", "class", "sub_class"])
    if category_col is None:
        return pd.DataFrame()

    category_values = (
        df[category_col]
        .fillna("Unclassified")
        .astype(str)
        .str.strip()
        .replace("", "Unclassified")
    )
    amounts = _amount_series(df).abs() if is_waste else _amount_series(df)
    grouped = amounts.groupby(category_values).sum().sort_values(ascending=False)
    return pd.DataFrame({"Category": grouped.index, "Total": grouped.values})


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return None


def _flow_table(df: pd.DataFrame) -> pd.DataFrame:
    columns_map = [
        ("date", ["date", "transaction_date"]),
        ("sub-description", ["sub-description", "sub_description"]),
        ("class", ["class", "category"]),
        ("sub-class", ["sub-class", "sub_class", "sub_category"]),
        ("amount", ["amount"]),
    ]
    selected: List[str] = []
    rename_map: Dict[str, str] = {}
    for target, candidates in columns_map:
        source = _pick_column(df, candidates)
        if source is None:
            continue
        selected.append(source)
        rename_map[source] = target

    if not selected:
        return df

    table_df = df.loc[:, selected].copy()
    return table_df.rename(columns=rename_map)


def _flow_palette(is_waste: bool) -> List[str]:
    if is_waste:
        return px.colors.sequential.Reds
    return px.colors.sequential.Greens


def _amount_series(df: pd.DataFrame) -> pd.Series:
    if "Amount" not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df["Amount"], errors="coerce").fillna(0)


def _sum_amount(df: pd.DataFrame) -> str:
    return _format_amount(_amount_series(df).sum())


def _format_amount(value: float) -> str:
    return f"{float(value):,.2f}"


if __name__ == "__main__":
    main()
