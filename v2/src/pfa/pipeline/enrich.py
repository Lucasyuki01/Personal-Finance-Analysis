from typing import List

import pandas as pd

from pfa.constants import (
    DEFAULT_CLASS,
    DEFAULT_CLASSIFICATION_SOURCE,
    DEFAULT_FLOW_TYPE,
    DEFAULT_SUB_CLASS,
)
from pfa.utils.hashing import stable_hash


def _row_transaction_id(row: pd.Series) -> str:
    """Compute a stable transaction_id for a row."""
    return stable_hash(
        [
            row.get("source_account", ""),
            row.get("Date", ""),
            row.get("Description", ""),
            row.get("Amount", ""),
            row.get("source_row_id", row.name),
        ]
    )


def add_canonical_fields(df: pd.DataFrame, processing_version: str) -> pd.DataFrame:
    """Add canonical enrichment fields used across downstream views."""
    df = df.copy()

    if "description_norm" not in df.columns:
        df["description_norm"] = (
            df.get("Description", "")
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "source_row_id" not in df.columns:
        df["source_row_id"] = df.index.astype(int)

    if "transaction_id" not in df.columns:
        df["transaction_id"] = df.apply(_row_transaction_id, axis=1)
    else:
        missing_mask = df["transaction_id"].isna()
        if missing_mask.any():
            df.loc[missing_mask, "transaction_id"] = df.loc[missing_mask].apply(
                _row_transaction_id, axis=1
            )

    if "flow_type" not in df.columns:
        df["flow_type"] = _infer_flow_type_from_amount(df)
    else:
        flow_values = df["flow_type"].fillna("").astype(str).str.strip().str.lower()
        needs_infer = flow_values.eq("") | flow_values.eq(DEFAULT_FLOW_TYPE)
        if needs_infer.any():
            df.loc[needs_infer, "flow_type"] = _infer_flow_type_from_amount(
                df.loc[needs_infer]
            )

    if "channel" not in df.columns:
        df["channel"] = df["description_norm"].eq("pos purchase").map(
            lambda is_card: "card" if is_card else "account"
        )

    if "class" not in df.columns:
        df["class"] = DEFAULT_CLASS

    if "sub_class" not in df.columns:
        df["sub_class"] = DEFAULT_SUB_CLASS

    if "is_fixed_waste" not in df.columns:
        df["is_fixed_waste"] = False

    if "classification_source" not in df.columns:
        df["classification_source"] = DEFAULT_CLASSIFICATION_SOURCE

    if "processing_version" not in df.columns:
        df["processing_version"] = processing_version
    else:
        df["processing_version"] = df["processing_version"].fillna(processing_version)

    return df


def _infer_flow_type_from_amount(df: pd.DataFrame) -> pd.Series:
    if "Amount" not in df.columns:
        return pd.Series(DEFAULT_FLOW_TYPE, index=df.index)
    amounts = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    inferred = pd.Series(DEFAULT_FLOW_TYPE, index=df.index)
    inferred = inferred.mask(amounts.gt(0), "profit")
    inferred = inferred.mask(amounts.lt(0), "waste")
    return inferred
