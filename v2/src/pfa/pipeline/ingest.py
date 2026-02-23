from pathlib import Path
from typing import Iterable, List

import pandas as pd

from pfa.io.inputs import read_table, read_uploaded_table


def infer_source_account(path: Path) -> str:
    """Infer the source_account label from a file path."""
    return path.stem or "unknown"


def ingest_tables(paths: Iterable[Path]) -> pd.DataFrame:
    """Load input tables, add source_account and source_row_id, then union."""
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = read_table(path).copy(deep=True)
        df["source_account"] = infer_source_account(path)
        df["source_row_id"] = df.index.astype(int)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def ingest_uploads(uploads: Iterable[object]) -> pd.DataFrame:
    """Load uploaded tables, add source_account and source_row_id, then union."""
    frames: List[pd.DataFrame] = []
    for upload in uploads:
        name = getattr(upload, "name", None) or getattr(upload, "filename", None) or "upload"
        df = read_uploaded_table(upload, filename=name).copy(deep=True)
        df["source_account"] = infer_source_account(Path(name))
        df["source_row_id"] = df.index.astype(int)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
