import pandas as pd

from pfa.constants import TRANSFER_DESCRIPTIONS


def mark_inter_account_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Tag inter-account transfers with an excluded_reason."""
    df = df.copy()
    if "description_norm" not in df.columns:
        df["description_norm"] = (
            df.get("Description", "")
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "excluded_reason" not in df.columns:
        df["excluded_reason"] = pd.NA

    mask = df["description_norm"].isin(TRANSFER_DESCRIPTIONS)
    df.loc[mask, "excluded_reason"] = "inter_account_transfer"
    return df


def filter_analysis_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are not excluded."""
    if "excluded_reason" not in df.columns:
        return df.copy()

    excluded = df["excluded_reason"].fillna("").astype(str).str.strip()
    return df[excluded.eq("")].copy()

