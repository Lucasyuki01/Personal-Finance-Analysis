import pandas as pd

from pfa.pipeline.clean import initial_clean
from pfa.pipeline.enrich import add_canonical_fields
from pfa.pipeline.exclusions import mark_inter_account_transfers


def test_initial_clean_drops_filter_column() -> None:
    df = pd.DataFrame(
        {
            "Filter": ["x", "y"],
            "Description": ["A", "B"],
        }
    )
    cleaned = initial_clean(df)
    assert "Filter" not in cleaned.columns


def test_description_norm_is_trimmed_lowercase() -> None:
    df = pd.DataFrame({"Description": ["  Hello World  ", None]})
    cleaned = initial_clean(df)
    assert cleaned.loc[0, "description_norm"] == "hello world"
    assert cleaned.loc[1, "description_norm"] == ""


def test_mark_inter_account_transfers_sets_excluded_reason() -> None:
    df = pd.DataFrame({"Description": ["customer transfer dr.", "pos purchase"]})
    cleaned = initial_clean(df)
    tagged = mark_inter_account_transfers(cleaned)
    assert tagged.loc[0, "excluded_reason"] == "inter_account_transfer"
    assert pd.isna(tagged.loc[1, "excluded_reason"])


def test_transaction_id_is_stable_for_same_inputs() -> None:
    df = pd.DataFrame(
        {
            "source_account": ["acc1", "acc1"],
            "Date": ["2024-01-01", "2024-01-02"],
            "Description": ["Deposit", "POS Purchase"],
            "Amount": [100.0, -50.0],
            "source_row_id": [0, 1],
        }
    )
    enriched_a = add_canonical_fields(df, processing_version="test")
    enriched_b = add_canonical_fields(df, processing_version="test")
    assert enriched_a["transaction_id"].tolist() == enriched_b["transaction_id"].tolist()
