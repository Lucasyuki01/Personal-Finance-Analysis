from pathlib import Path
import io
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
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
from pfa.constants import DEFAULT_CLASS, DEFAULT_SUB_CLASS  # noqa: E402
from pfa.pipeline.llm_classify import MANUAL_TAXONOMY  # noqa: E402
from rules.apply import apply_rules  # noqa: E402
from rules.store import load_rules  # noqa: E402


OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data" / "outputs" / "canonical_latest.parquet"
MANUAL_OVERRIDES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "outputs" / "manual_classifications.parquet"
)


def main() -> None:
    st.title("Personal Finance Analysis - Canonical Viewer")

    uploads, use_llm, apply_manual, page = _sidebar_controls()
    if st.session_state.get("run_pipeline_clicked"):
        st.session_state["run_pipeline_clicked"] = False
        _run_pipeline(uploads, use_llm=use_llm, apply_manual=apply_manual)
    if st.session_state.get("load_canonical_clicked"):
        st.session_state["load_canonical_clicked"] = False
        _load_canonical(apply_manual=apply_manual)

    canonical_df = st.session_state.get("canonical_df")
    if canonical_df is not None:
        _render_views(canonical_df, page)
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
        use_llm = st.checkbox("Use AI classification", value=True)
        apply_manual = st.checkbox(
            "Apply manual overrides",
            value=True,
            key="apply_manual_overrides",
        )
        st.divider()
        page = st.radio(
            "Page",
            options=["Overview", "Profits detail", "Wastes detail"],
            index=0,
        )
        if st.button("Run pipeline"):
            st.session_state["run_pipeline_clicked"] = True
        if st.button("Load saved canonical"):
            st.session_state["load_canonical_clicked"] = True
    return uploads, use_llm, apply_manual, page


def _run_pipeline(uploads, use_llm: bool, apply_manual: bool) -> None:
    if not uploads:
        st.error("Please upload at least one file before running the pipeline.")
        return

    with st.spinner("Running pipeline..."):
        os.environ["PFA_ENABLE_LLM"] = "1" if use_llm else "0"
        try:
            try:
                canonical_df, _analysis_ready, _manifest = run_pipeline_from_uploads(
                    uploads,
                    enable_llm=use_llm,
                )
            except TypeError as exc:
                if "enable_llm" not in str(exc):
                    raise
                canonical_df, _analysis_ready, _manifest = run_pipeline_from_uploads(
                    uploads
                )
        except Exception as exc:  # noqa: BLE001 - surface pipeline errors
            st.error(f"Pipeline failed: {exc}")
            return

        if st.session_state.get("apply_rules"):
            rules_df = load_rules()
            if not rules_df.empty:
                canonical_df = apply_rules(canonical_df, rules_df)
            else:
                st.warning("No classification rules found; showing unclassified data.")

        if apply_manual:
            canonical_df = _apply_manual_overrides(canonical_df)
        write_parquet(canonical_df, OUTPUT_PATH)
        st.session_state["canonical_df"] = canonical_df
        st.success(f"Canonical dataset saved to {OUTPUT_PATH}")


def _load_canonical(apply_manual: bool) -> None:
    if not OUTPUT_PATH.exists():
        st.error(f"No saved canonical dataset found at {OUTPUT_PATH}")
        return

    try:
        canonical_df = pd.read_parquet(OUTPUT_PATH)
    except Exception as exc:  # noqa: BLE001 - surface load errors
        st.error(f"Failed to load canonical dataset: {exc}")
        return

    if apply_manual:
        canonical_df = _apply_manual_overrides(canonical_df)
    st.session_state["canonical_df"] = canonical_df
    st.success(f"Loaded canonical dataset from {OUTPUT_PATH}")


def _render_views(canonical_df: pd.DataFrame, page: str) -> None:
    canonical_df = _ensure_short_id(canonical_df)
    if page == "Overview":
        _render_overview_page(canonical_df)
    elif page == "Profits detail":
        _render_profit_detail_page(canonical_df)
    else:
        _render_waste_detail_page(canonical_df)


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


def _ensure_short_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "transaction_id" not in df.columns:
        return df
    if "short_id" in df.columns and df["short_id"].notna().any():
        return df
    updated = df.copy()
    updated["short_id"] = (
        updated["transaction_id"].astype(str).str.slice(0, 4)
    )
    return updated


def _render_overview_page(canonical_df: pd.DataFrame) -> None:
    st.subheader("Summary")
    _render_summary(canonical_df)

    st.subheader("Profits")
    _render_flow_section(
        "Profits",
        view_profits(canonical_df),
        is_waste=False,
        show_tables=False,
        show_editor=False,
        render_drilldown=True,
    )

    st.subheader("Wastes")
    _render_flow_section(
        "Wastes",
        view_wastes(canonical_df),
        is_waste=True,
        show_tables=False,
        show_editor=False,
        render_drilldown=True,
    )

    st.subheader("Download")
    _render_download(canonical_df)


def _render_profit_detail_page(canonical_df: pd.DataFrame) -> None:
    st.subheader("Profits")
    profits_df = view_profits(canonical_df)
    filtered_df, selected_class = _render_filtered_flow(
        "Profits",
        profits_df,
        is_waste=False,
        render_drilldown=False,
    )
    if selected_class:
        _render_class_drilldown(filtered_df, selected_class, is_waste=False)
        _render_class_table(filtered_df, selected_class, is_waste=False)

    st.subheader("Transaction Lookup")
    _render_transaction_lookup(canonical_df)

    st.subheader("Unclassified")
    _render_unclassified_section(canonical_df)


def _render_waste_detail_page(canonical_df: pd.DataFrame) -> None:
    st.subheader("Wastes")
    wastes_df = view_wastes(canonical_df)
    filtered_df, selected_class = _render_filtered_flow(
        "Wastes",
        wastes_df,
        is_waste=True,
        render_drilldown=False,
    )
    if selected_class:
        _render_class_drilldown(filtered_df, selected_class, is_waste=True)
        _render_class_table(filtered_df, selected_class, is_waste=True)

    st.subheader("Transaction Lookup")
    _render_transaction_lookup(canonical_df)

    st.subheader("Unclassified")
    _render_unclassified_section(canonical_df)


def _render_filtered_flow(
    title: str,
    view_df: pd.DataFrame,
    is_waste: bool,
    render_drilldown: bool,
) -> Tuple[pd.DataFrame, Optional[str]]:
    filtered_df = view_df
    selected_class = _render_flow_section(
        title,
        filtered_df,
        is_waste=is_waste,
        show_tables=True,
        show_editor=False,
        render_drilldown=render_drilldown,
    )
    return filtered_df, selected_class


def _render_class_table(
    view_df: pd.DataFrame,
    selected_class: str,
    is_waste: bool,
) -> None:
    st.markdown("#### Selected class transactions")
    class_col = _pick_column(view_df, ["class", "category"])
    if class_col is None:
        st.info("Class data is unavailable.")
        return
    class_rows = view_df[
        view_df[class_col].fillna("").astype(str).str.strip().eq(selected_class)
    ]
    if class_rows.empty:
        st.info("No transactions for selected class.")
        return
    table_df = _flow_table(class_rows)
    st.dataframe(table_df, use_container_width=True)

def _render_flow_section(
    title: str,
    view_df: pd.DataFrame,
    is_waste: bool,
    show_tables: bool = True,
    show_editor: bool = True,
    render_drilldown: bool = True,
) -> Optional[str]:
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
        selected_class = _render_category_pie_chart(view_df, title, is_waste=is_waste)

    if not selected_class:
        class_options = _category_totals(view_df, is_waste=is_waste)["Category"].tolist()
        if class_options:
            picker = st.selectbox(
                "Select class to drill down",
                options=["(select)"] + class_options,
                index=0,
                key=f"{title.lower()}_drilldown_select",
            )
            if picker != "(select)":
                selected_class = picker

    if show_tables:
        table_df = _flow_table(view_df)
        st.dataframe(table_df, use_container_width=True)
    if show_editor:
        _render_manual_editor(title, view_df)
    if render_drilldown and selected_class:
        _render_class_drilldown(view_df, selected_class, is_waste=is_waste)
    return selected_class


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
) -> Optional[str]:
    categories = _category_totals(view_df, is_waste=is_waste)
    if categories.empty:
        st.info("No category data available for this view.")
        return None

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
    return None


def _render_class_drilldown(
    view_df: pd.DataFrame,
    selected_class: str,
    is_waste: bool,
) -> None:
    st.markdown(f"#### {selected_class} breakdown")
    class_rows = view_df.copy()
    class_col = _pick_column(class_rows, ["class", "category"])
    if class_col is None:
        st.info("Class data is unavailable for drilldown.")
        return

    class_rows = class_rows[
        class_rows[class_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(str(selected_class).strip())
    ]
    if class_rows.empty:
        st.info("No rows available for this class.")
        return

    amount_series = _amount_series(class_rows)
    if is_waste:
        amount_series = amount_series.abs()
    total_amount = float(amount_series.sum())
    st.caption(f"Total: {_format_amount(total_amount)}")

    sub_col = _pick_column(class_rows, ["sub_class", "sub-class", "sub_category"])
    if sub_col:
        sub_values = (
            class_rows[sub_col]
            .fillna("Unclassified")
            .astype(str)
            .str.strip()
            .replace("", "Unclassified")
        )
        sub_totals = amount_series.groupby(sub_values).sum().sort_values(ascending=False)
        sub_df = pd.DataFrame({"Sub-class": sub_totals.index, "Total": sub_totals.values})
    else:
        sub_df = pd.DataFrame()

    monthly = _monthly_totals(class_rows, is_waste=is_waste)

    col_sub, col_trend = st.columns(2)
    with col_sub:
        if sub_df.empty:
            st.info("No sub-class data available.")
        else:
            fig_sub = px.pie(
                sub_df,
                names="Sub-class",
                values="Total",
                title="Sub-class distribution",
                color_discrete_sequence=_flow_palette(is_waste),
            )
            fig_sub.update_traces(textposition="inside", textinfo="percent+label")
            fig_sub.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_sub, use_container_width=True)

    with col_trend:
        if monthly.empty:
            st.info("No monthly data available.")
        else:
            fig_trend = px.bar(
                monthly,
                x="Month",
                y="Total",
                title="Monthly totals",
                labels={"Month": "Month", "Total": "Total"},
            )
            fig_trend.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_trend, use_container_width=True)


def _monthly_totals(df: pd.DataFrame, is_waste: bool) -> pd.DataFrame:
    date_col = _pick_column(df, ["date", "Date", "transaction_date"])
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
    category_col = _pick_column(df, ["class", "category", "sub_class"])
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
        ("id", ["short_id", "transaction_id"]),
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


def _render_unclassified_section(canonical_df: pd.DataFrame) -> None:
    unclassified_df = _unclassified_manual_view(canonical_df)
    if unclassified_df.empty:
        st.info("No unclassified manual items found.")
        return

    table_df = _flow_table(unclassified_df)
    st.dataframe(table_df, use_container_width=True)
    _render_manual_editor(
        "Unclassified",
        unclassified_df,
        allow_unmodified_toggle=False,
    )

    if st.button("Undo last manual classification", key="undo_manual"):
        removed = _undo_last_manual_override()
        if removed:
            st.success(f"Undid {removed} manual classification entries.")
            st.rerun()
        else:
            st.info("No manual classifications to undo.")


def _unclassified_manual_view(canonical_df: pd.DataFrame) -> pd.DataFrame:
    combined = canonical_df.copy()
    if combined.empty:
        return combined
    if "transaction_id" not in combined.columns:
        return combined

    if "excluded_reason" in combined.columns:
        excluded = combined["excluded_reason"].fillna("").astype(str).str.strip().ne("")
        combined = combined.loc[~excluded].copy()

    class_series = (
        combined["class"] if "class" in combined.columns else pd.Series(DEFAULT_CLASS, index=combined.index)
    )
    sub_series = (
        combined["sub_class"]
        if "sub_class" in combined.columns
        else pd.Series(DEFAULT_SUB_CLASS, index=combined.index)
    )
    class_norm = class_series.fillna(DEFAULT_CLASS).astype(str).str.strip().str.lower()
    sub_norm = sub_series.fillna(DEFAULT_SUB_CLASS).astype(str).str.strip().str.lower()
    class_default = class_norm.isin({DEFAULT_CLASS.lower(), "", "none"})
    sub_default = sub_norm.isin({DEFAULT_SUB_CLASS.lower(), "", "none"})
    return combined.loc[class_default & sub_default].copy()


def _render_manual_editor(
    title: str,
    view_df: pd.DataFrame,
    allow_unmodified_toggle: bool = True,
) -> None:
    if view_df.empty or "transaction_id" not in view_df.columns:
        return

    with st.expander(f"Edit {title} classification", expanded=False):
        manual_ids = _manual_override_ids()
        show_only_unmodified = False
        if allow_unmodified_toggle:
            show_only_unmodified = st.checkbox(
                "Show only unmodified",
                value=True,
                key=f"{title.lower()}_show_unmodified",
            )
        order_by = st.selectbox(
            "Order by",
            options=[
                "Default",
                "Date (newest first)",
                "Date (oldest first)",
                "Amount (high to low)",
                "Amount (low to high)",
            ],
            index=0,
            key=f"{title.lower()}_order_by",
        )
        transaction_ids, labels = _build_transaction_labels(
            view_df,
            manual_ids=manual_ids,
            only_unmodified=show_only_unmodified,
            order_by=order_by,
        )
        if not transaction_ids:
            st.info("No unmodified transactions available.")
            return
        selected_tx = st.selectbox(
            "Select a transaction",
            options=transaction_ids,
            format_func=lambda value: labels.get(value, str(value)),
            key=f"{title.lower()}_tx_select",
        )

        row = view_df.loc[view_df["transaction_id"] == selected_tx]
        if row.empty:
            return
        row = row.iloc[0]

        class_options = sorted(MANUAL_TAXONOMY.keys())
        current_class = _safe_string(row.get("class"))
        class_index = class_options.index(current_class) if current_class in class_options else 0
        selected_class = st.selectbox(
            "Class",
            options=class_options,
            index=class_index,
            key=f"{title.lower()}_class_select",
        )

        sub_options = MANUAL_TAXONOMY.get(selected_class, [])
        current_sub = _safe_string(row.get("sub_class"))
        sub_index = sub_options.index(current_sub) if current_sub in sub_options else 0
        selected_sub = st.selectbox(
            "Sub-class",
            options=sub_options,
            index=sub_index,
            key=f"{title.lower()}_sub_class_select",
        )

        apply_to_all = st.checkbox(
            "Apply to all matching Sub-description",
            value=False,
            key=f"{title.lower()}_apply_all",
        )

        if st.button(f"Save {title} selection", key=f"{title.lower()}_manual_save"):
            saved_count = _persist_manual_override(
                selected_tx,
                selected_class,
                selected_sub,
                row,
                apply_to_all=apply_to_all,
            )
            if saved_count:
                st.success(
                    f"Manual classification saved for {saved_count} matching transactions."
                )
                st.rerun()
            else:
                st.info("No manual changes detected.")


def _build_transaction_labels(
    view_df: pd.DataFrame,
    manual_ids: Set[str],
    only_unmodified: bool,
    order_by: str,
) -> Tuple[List[str], Dict[str, str]]:
    short_id_col = _pick_column(view_df, ["short_id"])
    date_col = _pick_column(view_df, ["date", "transaction_date"])
    sub_desc_col = _pick_column(view_df, ["sub-description", "sub_description"])
    amount_col = _pick_column(view_df, ["amount"])

    labels: Dict[str, str] = {}
    items: List[Tuple[Optional[pd.Timestamp], float, str, bool]] = []
    for _, row in view_df.iterrows():
        transaction_id = row.get("transaction_id")
        if pd.isna(transaction_id):
            continue
        transaction_id = str(transaction_id)
        if transaction_id in labels:
            continue

        is_modified = _is_manually_classified(row, manual_ids, transaction_id)
        if only_unmodified and is_modified:
            continue

        date_value = row.get(date_col) if date_col else ""
        parsed_date = None
        if pd.notna(date_value):
            try:
                parsed_date = pd.to_datetime(date_value)
                date_value = parsed_date.date().isoformat()
            except Exception:  # noqa: BLE001 - fallback to string
                date_value = str(date_value)
        else:
            date_value = ""

        sub_desc_value = row.get(sub_desc_col) if sub_desc_col else ""
        sub_desc_value = _safe_string(sub_desc_value) or "n/a"

        amount_value = row.get(amount_col) if amount_col else ""
        amount_text = _safe_string(amount_value)
        try:
            amount_num = float(amount_value)
        except (TypeError, ValueError):
            amount_num = 0.0

        short_id = _safe_string(row.get(short_id_col)) if short_id_col else ""
        if short_id:
            label = f"{short_id} | {date_value} | {sub_desc_value} | {amount_text}"
        else:
            label = f"{date_value} | {sub_desc_value} | {amount_text}"
        labels[transaction_id] = label
        items.append((parsed_date, amount_num, transaction_id, is_modified))

    if order_by == "Date (newest first)":
        items.sort(key=lambda x: (x[0] is None, x[0]), reverse=True)
    elif order_by == "Date (oldest first)":
        items.sort(key=lambda x: (x[0] is None, x[0]))
    elif order_by == "Amount (high to low)":
        items.sort(key=lambda x: x[1], reverse=True)
    elif order_by == "Amount (low to high)":
        items.sort(key=lambda x: x[1])

    unmodified_ids = [item[2] for item in items if not item[3]]
    modified_ids = [item[2] for item in items if item[3]]

    if only_unmodified:
        return unmodified_ids, labels
    return unmodified_ids + modified_ids, labels


def _manual_override_ids() -> Set[str]:
    overrides_df = _load_manual_overrides()
    if overrides_df.empty or "transaction_id" not in overrides_df.columns:
        return set()
    return set(overrides_df["transaction_id"].dropna().astype(str))


def _is_manually_classified(
    row: pd.Series,
    manual_ids: Set[str],
    transaction_id: str,
) -> bool:
    if transaction_id in manual_ids:
        return True
    source = _safe_string(row.get("classification_source")).lower()
    return source == "manual"


def _persist_manual_override(
    transaction_id: str,
    new_class: str,
    new_sub_class: str,
    original_row: pd.Series,
    apply_to_all: bool,
) -> int:
    if not transaction_id or not new_class:
        return 0

    orig_class = _safe_string(original_row.get("class"))
    orig_sub = _safe_string(original_row.get("sub_class"))
    if new_class == orig_class and new_sub_class == orig_sub:
        return 0

    canonical_df = st.session_state.get("canonical_df")
    matched_ids = []
    if apply_to_all:
        matched_ids = _match_transaction_ids_by_sub_description(canonical_df, original_row)
    if not matched_ids:
        matched_ids = [transaction_id]

    overrides_df = _load_manual_overrides()
    apply_all_overrides = st.session_state.get("apply_manual_overrides", True)
    existing: Dict[str, Dict[str, str]] = {}
    if apply_all_overrides and not overrides_df.empty:
        existing = (
            overrides_df.set_index("transaction_id")
            .fillna("")
            .astype(str)
            .to_dict(orient="index")
        )

    canonical_lookup: Dict[str, Dict[str, str]] = {}
    if isinstance(canonical_df, pd.DataFrame) and not canonical_df.empty:
        lookup_df = canonical_df.copy()
        if "transaction_id" in lookup_df.columns:
            lookup_df = lookup_df.drop_duplicates(subset=["transaction_id"])
            lookup_df = lookup_df.set_index("transaction_id")
            available_cols = [
                col
                for col in ["class", "sub_class", "classification_source"]
                if col in lookup_df.columns
            ]
            if available_cols:
                canonical_lookup = (
                    lookup_df[available_cols].fillna("").astype(str).to_dict(orient="index")
                )

    updates: List[Dict[str, str]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for matched_id in matched_ids:
        if not matched_id:
            continue
        matched_id = str(matched_id)
        current = existing.get(matched_id)
        if current and current.get("class") == new_class and current.get("sub_class") == new_sub_class:
            continue
        base_values = canonical_lookup.get(matched_id, {})
        original_class = current.get("original_class") if current else ""
        original_sub = current.get("original_sub_class") if current else ""
        original_source = current.get("original_source") if current else ""
        if not original_class:
            original_class = _safe_string(base_values.get("class"))
        if not original_sub:
            original_sub = _safe_string(base_values.get("sub_class"))
        if not original_source:
            original_source = _safe_string(base_values.get("classification_source"))
        updates.append(
            {
                "transaction_id": matched_id,
                "class": new_class,
                "sub_class": new_sub_class,
                "original_class": original_class,
                "original_sub_class": original_sub,
                "original_source": original_source,
                "updated_at": timestamp,
            }
        )

    if not updates:
        return 0

    update_df = pd.DataFrame(updates)
    combined = (
        pd.concat([overrides_df, update_df], ignore_index=True)
        if not overrides_df.empty
        else update_df
    )
    combined = combined.drop_duplicates(subset=["transaction_id"], keep="last")
    _save_manual_overrides(combined)

    canonical_df = st.session_state.get("canonical_df")
    if isinstance(canonical_df, pd.DataFrame) and not canonical_df.empty:
        updated_canonical = _apply_manual_overrides(canonical_df, overrides_df=update_df)
        st.session_state["canonical_df"] = updated_canonical
        write_parquet(updated_canonical, OUTPUT_PATH)

    return len(updates)


def _render_transaction_lookup(canonical_df: pd.DataFrame) -> None:
    if canonical_df.empty or "transaction_id" not in canonical_df.columns:
        st.info("No transaction IDs available.")
        return

    id_col = "short_id" if "short_id" in canonical_df.columns else "transaction_id"
    tx_input = st.text_input(
        f"Search by {id_col}",
        value="",
        key="transaction_id_lookup",
    ).strip()
    if not tx_input:
        st.caption("Enter a transaction_id to edit its class/sub-class.")
        return

    matched = canonical_df[canonical_df[id_col].astype(str).eq(tx_input)]
    if matched.empty:
        st.warning("No transaction found for that ID.")
        return

    if len(matched) > 1:
        options = matched["transaction_id"].astype(str).tolist()
        labels = {
            tid: _safe_string(matched.loc[matched["transaction_id"].astype(str).eq(tid), "short_id"].iloc[0])
            for tid in options
        }
        selected_tid = st.selectbox(
            "Multiple matches found. Pick one transaction_id",
            options=options,
            format_func=lambda value: labels.get(value, value),
            key="transaction_id_lookup_select",
        )
        row = matched[matched["transaction_id"].astype(str).eq(selected_tid)].iloc[0]
    else:
        row = matched.iloc[0]
    st.write("Match:")
    st.dataframe(_flow_table(matched), use_container_width=True)

    class_options = sorted(MANUAL_TAXONOMY.keys())
    current_class = _safe_string(row.get("class"))
    class_index = class_options.index(current_class) if current_class in class_options else 0
    selected_class = st.selectbox(
        "Class",
        options=class_options,
        index=class_index,
        key="lookup_class_select",
    )

    sub_options = MANUAL_TAXONOMY.get(selected_class, [])
    current_sub = _safe_string(row.get("sub_class"))
    sub_index = sub_options.index(current_sub) if current_sub in sub_options else 0
    selected_sub = st.selectbox(
        "Sub-class",
        options=sub_options,
        index=sub_index,
        key="lookup_sub_class_select",
    )

    if st.button("Save transaction edit", key="lookup_save"):
        saved_count = _persist_manual_override(
            str(row.get("transaction_id")),
            selected_class,
            selected_sub,
            row,
            apply_to_all=False,
        )
        if saved_count:
            st.success("Manual classification saved.")
            st.rerun()
        else:
            st.info("No manual changes detected.")


def _match_transaction_ids_by_sub_description(
    canonical_df: Optional[pd.DataFrame],
    original_row: pd.Series,
) -> List[str]:
    if canonical_df is None or canonical_df.empty or "transaction_id" not in canonical_df.columns:
        return []

    sub_desc = _safe_string(_row_sub_description(original_row))
    if not sub_desc:
        return []

    sub_desc_col = _pick_column(canonical_df, ["sub-description", "sub_description"])
    if sub_desc_col is None:
        return []

    sub_norm = sub_desc.strip().lower()
    sub_norm_flat = sub_norm.replace("-", " ")
    if "free interac e-transfer" in sub_norm or "free interac e transfer" in sub_norm_flat:
        return []
    series = (
        canonical_df[sub_desc_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    mask = series.eq(sub_norm)
    matched = canonical_df.loc[mask, "transaction_id"].dropna().astype(str).unique().tolist()
    return matched


def _row_sub_description(row: pd.Series) -> str:
    for key in ("Sub-description", "sub-description", "sub_description"):
        if key in row:
            return _safe_string(row.get(key))
    return ""


def _apply_manual_overrides(
    canonical_df: pd.DataFrame,
    overrides_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if canonical_df.empty or "transaction_id" not in canonical_df.columns:
        return canonical_df

    overrides = overrides_df if overrides_df is not None else _load_manual_overrides()
    if overrides.empty:
        return canonical_df

    overrides = overrides.dropna(subset=["transaction_id"]).copy()
    overrides = overrides.drop_duplicates(subset=["transaction_id"], keep="last")

    merged = canonical_df.merge(
        overrides[["transaction_id", "class", "sub_class"]],
        on="transaction_id",
        how="left",
        suffixes=("", "_manual"),
    )
    manual_mask = merged["class_manual"].notna() | merged["sub_class_manual"].notna()
    if manual_mask.any():
        merged.loc[manual_mask, "class"] = merged.loc[manual_mask, "class_manual"].fillna(
            merged.loc[manual_mask, "class"]
        )
        merged.loc[manual_mask, "sub_class"] = merged.loc[manual_mask, "sub_class_manual"].fillna(
            merged.loc[manual_mask, "sub_class"]
        )
        merged.loc[manual_mask, "classification_source"] = "manual"

    merged = merged.drop(columns=["class_manual", "sub_class_manual"])
    return merged


def _load_manual_overrides() -> pd.DataFrame:
    if not MANUAL_OVERRIDES_PATH.exists():
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "class",
                "sub_class",
                "original_class",
                "original_sub_class",
                "original_source",
                "updated_at",
            ]
        )
    try:
        return pd.read_parquet(MANUAL_OVERRIDES_PATH)
    except Exception:  # noqa: BLE001 - return empty on load errors
        return pd.DataFrame(
            columns=[
                "transaction_id",
                "class",
                "sub_class",
                "original_class",
                "original_sub_class",
                "original_source",
                "updated_at",
            ]
        )


def _save_manual_overrides(df: pd.DataFrame) -> None:
    MANUAL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANUAL_OVERRIDES_PATH, index=False)


def _undo_last_manual_override() -> int:
    overrides_df = _load_manual_overrides()
    if overrides_df.empty:
        return 0

    if "updated_at" in overrides_df.columns and overrides_df["updated_at"].notna().any():
        parsed = pd.to_datetime(overrides_df["updated_at"], errors="coerce")
        latest = parsed.max()
        if pd.isna(latest):
            removed_rows = overrides_df.iloc[[-1]].copy()
            trimmed = overrides_df.iloc[:-1].copy()
        else:
            mask = parsed.eq(latest)
            removed_rows = overrides_df.loc[mask].copy()
            trimmed = overrides_df.loc[~mask].copy()
    else:
        removed_rows = overrides_df.iloc[[-1]].copy()
        trimmed = overrides_df.iloc[:-1].copy()

    _save_manual_overrides(trimmed)

    canonical_df = st.session_state.get("canonical_df")
    if isinstance(canonical_df, pd.DataFrame) and not canonical_df.empty:
        updated_canonical = _revert_manual_rows(canonical_df, removed_rows)
        st.session_state["canonical_df"] = updated_canonical
        write_parquet(updated_canonical, OUTPUT_PATH)

    return len(removed_rows)


def _revert_manual_rows(
    canonical_df: pd.DataFrame,
    removed_rows: pd.DataFrame,
) -> pd.DataFrame:
    if canonical_df.empty or removed_rows.empty:
        return canonical_df

    updated = canonical_df.copy()
    if "transaction_id" not in updated.columns:
        return updated

    fallback_class = DEFAULT_CLASS
    fallback_sub = DEFAULT_SUB_CLASS

    for _, row in removed_rows.iterrows():
        transaction_id = row.get("transaction_id")
        if pd.isna(transaction_id):
            continue
        transaction_id = str(transaction_id)
        mask = updated["transaction_id"].astype(str).eq(transaction_id)
        if not mask.any():
            continue

        original_class = _safe_string(row.get("original_class")) or fallback_class
        original_sub = _safe_string(row.get("original_sub_class")) or fallback_sub
        original_source = _safe_string(row.get("original_source"))

        updated.loc[mask, "class"] = original_class
        updated.loc[mask, "sub_class"] = original_sub
        if original_source:
            updated.loc[mask, "classification_source"] = original_source

    return updated


def _safe_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _flow_palette(is_waste: bool) -> List[str]:
    if is_waste:
        return px.colors.sequential.Reds
    return px.colors.sequential.Greens


def _amount_series(df: pd.DataFrame) -> pd.Series:
    if "Amount" not in df.columns:
        return pd.Series(0, index=df.index)
    cleaned = (
        df["Amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def _sum_amount(df: pd.DataFrame) -> str:
    return _format_amount(_amount_series(df).sum())


def _format_amount(value: float) -> str:
    return f"{float(value):,.2f}"


if __name__ == "__main__":
    main()
