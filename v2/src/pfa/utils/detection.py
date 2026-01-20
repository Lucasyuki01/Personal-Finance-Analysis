from typing import Dict

import pandas as pd


def processing_markers_status(df: pd.DataFrame) -> Dict[str, bool]:
    """Detect whether a dataframe already includes processing markers."""
    has_transaction_id = "transaction_id" in df.columns
    has_processing_version = "processing_version" in df.columns
    has_values = False
    if has_transaction_id and has_processing_version:
        has_values = df["transaction_id"].notna().any() and df["processing_version"].notna().any()

    return {
        "has_transaction_id_column": has_transaction_id,
        "has_processing_version_column": has_processing_version,
        "has_values": has_values,
    }

