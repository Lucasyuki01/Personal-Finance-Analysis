from pathlib import Path
import sys
import pandas as pd
import streamlit as st


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


def _render_views(canonical_df: pd.DataFrame) -> None:
    st.subheader("Views")

    _render_view("Profits", view_profits(canonical_df))
    _render_view("Wastes", view_wastes(canonical_df))
    _render_view("Card Purchases", view_card_purchases(canonical_df))
    _render_view("Account Expenses", view_account_expenses(canonical_df))
    _render_view("Fixed Wastes", view_fixed_wastes(canonical_df))


def _render_view(
    title: str,
    view_df: pd.DataFrame,
) -> None:
    st.markdown(f"### {title}")

    row_count = len(view_df)
    total_amount = _sum_amount(view_df)
    st.write(f"Rows: {row_count:,} | Total amount: {total_amount}")

    st.dataframe(view_df, use_container_width=True)


def _sum_amount(df: pd.DataFrame) -> str:
    if "Amount" not in df.columns:
        return "n/a"
    series = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    return f"{float(series.sum()):,.2f}"


if __name__ == "__main__":
    main()
