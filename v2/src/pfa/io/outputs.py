import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def write_canonical_base(
    df: pd.DataFrame,
    output_dir: Path,
    basename: str = "canonical_base",
    write_parquet: bool = False,
) -> Dict[str, Optional[Path]]:
    """Write canonical dataset to CSV (and optional Parquet)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}.csv"
    df.to_csv(csv_path, index=False)

    parquet_path = None
    if write_parquet:
        parquet_path = output_dir / f"{basename}.parquet"
        df.to_parquet(parquet_path, index=False)

    return {"csv": csv_path, "parquet": parquet_path}


def write_processing_manifest(
    manifest: Dict[str, Any],
    output_dir: Path,
    name: str = "processing_manifest.json",
) -> Path:
    """Write a JSON processing manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
