from pathlib import Path
import io
import sys
from datetime import datetime, timezone
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
from pfa.pipeline.llm_classify import DEFAULT_TAXONOMY  # noqa: E402
from rules.apply import apply_rules  # noqa: E402
from rules.store import load_rules  # noqa: E402


OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data" / "outputs" / "canonical_latest.parquet"
MANUAL_OVERRIDES_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "outputs" / "manual_classifications.parquet"
)


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

        canonical_df = _apply_manual_overrides(canonical_df)
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

    canonical_df = _apply_manual_overrides(canonical_df)
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
    taxonomy_subs = {sub for subs in DEFAULT_TAXONOMY.values() for sub in subs}
    class_values = (
        table_df["class"].dropna().astype(str).tolist() if "class" in table_df.columns else []
    )
    sub_values = (
        table_df["sub-class"].dropna().astype(str).tolist()
        if "sub-class" in table_df.columns
        else []
    )
    class_options = sorted(set(DEFAULT_TAXONOMY.keys()) | set(class_values))
    sub_options = sorted(taxonomy_subs | set(sub_values))
    column_config: Dict[str, object] = {}
    if "date" in table_df.columns:
        column_config["date"] = st.column_config.TextColumn("date", disabled=True)
    if "sub-description" in table_df.columns:
        column_config["sub-description"] = st.column_config.TextColumn(
            "sub-description", disabled=True
        )
    if "amount" in table_df.columns:
        column_config["amount"] = st.column_config.NumberColumn(
            "amount", disabled=True, format="%.2f"
        )
    if "class" in table_df.columns:
        column_config["class"] = st.column_config.SelectboxColumn(
            "class", options=class_options
        )
    if "sub-class" in table_df.columns:
        column_config["sub-class"] = st.column_config.SelectboxColumn(
            "sub-class", options=sub_options
        )

    edited_df = st.data_editor(
        table_df,
        use_container_width=True,
        hide_index=True,
        key=f"{title.lower()}_editor",
        column_config=column_config,
    )

    if st.button(f"Save {title} changes", key=f"{title.lower()}_save"):
        changes = _persist_manual_changes(view_df, edited_df)
        if changes:
            st.success(f"Saved {changes} manual classification changes.")
            st.rerun()
        else:
            st.info("No manual changes detected.")


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


def _persist_manual_changes(
    original_view: pd.DataFrame,
    edited_df: pd.DataFrame,
) -> int:
    if edited_df.empty:
        return 0
    if "transaction_id" not in original_view.columns:
        return 0

    taxonomy = DEFAULT_TAXONOMY
    updates: List[Dict[str, str]] = []
    for idx in edited_df.index:
        try:
            transaction_id = original_view.at[idx, "transaction_id"]
        except KeyError:
            continue
        if pd.isna(transaction_id):
            continue

        new_class = _safe_string(edited_df.at[idx, "class"])
        new_sub_class = _safe_string(edited_df.at[idx, "sub-class"])
        if not new_class:
            continue
        allowed = taxonomy.get(new_class, [])
        if allowed:
            if new_sub_class not in allowed:
                new_sub_class = allowed[0]

        try:
            orig_class = _safe_string(original_view.at[idx, "class"])
        except KeyError:
            orig_class = ""
        try:
            orig_sub = _safe_string(original_view.at[idx, "sub_class"])
        except KeyError:
            orig_sub = ""

        if new_class == orig_class and new_sub_class == orig_sub:
            continue

        updates.append(
            {
                "transaction_id": transaction_id,
                "class": new_class,
                "sub_class": new_sub_class,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if not updates:
        return 0

    overrides_df = _load_manual_overrides()
    new_df = pd.DataFrame(updates)
    combined = pd.concat([overrides_df, new_df], ignore_index=True) if not overrides_df.empty else new_df
    combined = combined.drop_duplicates(subset=["transaction_id"], keep="last")
    _save_manual_overrides(combined)

    canonical_df = st.session_state.get("canonical_df")
    if isinstance(canonical_df, pd.DataFrame) and not canonical_df.empty:
        updated_canonical = _apply_manual_overrides(canonical_df, overrides_df=combined)
        st.session_state["canonical_df"] = updated_canonical
        write_parquet(updated_canonical, OUTPUT_PATH)

    return len(updates)


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
        return pd.DataFrame(columns=["transaction_id", "class", "sub_class", "updated_at"])
    try:
        return pd.read_parquet(MANUAL_OVERRIDES_PATH)
    except Exception:  # noqa: BLE001 - return empty on load errors
        return pd.DataFrame(columns=["transaction_id", "class", "sub_class", "updated_at"])


def _save_manual_overrides(df: pd.DataFrame) -> None:
    MANUAL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANUAL_OVERRIDES_PATH, index=False)


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
    return pd.to_numeric(df["Amount"], errors="coerce").fillna(0)


def _sum_amount(df: pd.DataFrame) -> str:
    return _format_amount(_amount_series(df).sum())


def _format_amount(value: float) -> str:
    return f"{float(value):,.2f}"


if __name__ == "__main__":
    main()
