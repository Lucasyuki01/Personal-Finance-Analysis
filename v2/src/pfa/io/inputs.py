import io
from pathlib import Path
from typing import List, Optional

import pandas as pd


def gather_input_files(sample_dir: Path) -> List[Path]:
    """Return CSV/XLSX files found in the provided directory."""
    if not sample_dir.exists():
        return []
    return sorted(
        [p for p in sample_dir.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    )


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/XLSX file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return df.copy()


def read_uploaded_table(upload: object, filename: Optional[str] = None) -> pd.DataFrame:
    """Read an uploaded CSV/XLSX object into a DataFrame."""
    name = filename or getattr(upload, "name", None) or getattr(upload, "filename", None)
    if not name:
        raise ValueError("Uploaded file is missing a filename.")

    suffix = Path(name).suffix.lower()
    data = _read_upload_bytes(upload)
    buffer = io.BytesIO(data)

    if suffix == ".csv":
        df = pd.read_csv(buffer)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(buffer)
    else:
        raise ValueError(f"Unsupported file type: {name}")

    return df.copy()


def _read_upload_bytes(upload: object) -> bytes:
    if hasattr(upload, "getvalue"):
        data = upload.getvalue()
    else:
        try:
            upload.seek(0)
        except Exception:
            pass
        data = upload.read()
    return data or b""
