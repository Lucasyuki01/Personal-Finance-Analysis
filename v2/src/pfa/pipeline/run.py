from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import pandas as pd

from pfa.constants import PIPELINE_VERSION
from pfa.pipeline.clean import compute_missing_stats, initial_clean
from pfa.pipeline.enrich import add_canonical_fields
from pfa.pipeline.exclusions import filter_analysis_ready, mark_inter_account_transfers
from pfa.pipeline.ingest import ingest_tables, ingest_uploads
from pfa.pipeline.llm_classify import classify_card_expenses
from pfa.utils.detection import processing_markers_status
from rules.apply import apply_rules


def build_manifest(
    input_files: Iterable[Union[Path, str]],
    total_rows: int,
    excluded_rows: int,
    missing_stats: Dict[str, Dict[str, float]],
    processing_version: str,
    processing_status: Dict[str, bool],
) -> Dict[str, Any]:
    """Create a processing manifest for the run."""
    input_files = [Path(path) for path in input_files]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "processing_version": processing_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": [path.name for path in input_files],
        "file_count": len(input_files),
        "total_rows": total_rows,
        "excluded_rows": excluded_rows,
        "analysis_ready_rows": total_rows - excluded_rows,
        "missing_stats": missing_stats,
        "processing_markers": processing_status,
    }


def run_pipeline(
    input_files: Iterable[Path],
    processing_version: str = PIPELINE_VERSION,
    rules_df: Optional[pd.DataFrame] = None,
    overwrite_manual: bool = False,
    enable_llm: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run the full pipeline and return canonical + analysis-ready data."""
    input_files = list(input_files)
    ingested = ingest_tables(input_files)
    return _run_pipeline_on_df(
        ingested,
        input_files,
        processing_version,
        rules_df=rules_df,
        overwrite_manual=overwrite_manual,
        enable_llm=enable_llm,
    )


def run_pipeline_from_uploads(
    uploads: Iterable[object],
    processing_version: str = PIPELINE_VERSION,
    rules_df: Optional[pd.DataFrame] = None,
    overwrite_manual: bool = False,
    enable_llm: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run the full pipeline from uploaded file objects."""
    upload_list = list(uploads)
    ingested = ingest_uploads(upload_list)
    upload_names = [
        getattr(upload, "name", None) or getattr(upload, "filename", None) or "upload"
        for upload in upload_list
    ]
    return _run_pipeline_on_df(
        ingested,
        upload_names,
        processing_version,
        rules_df=rules_df,
        overwrite_manual=overwrite_manual,
        enable_llm=enable_llm,
    )


def _run_pipeline_on_df(
    ingested: pd.DataFrame,
    input_files: Iterable[Union[Path, str]],
    processing_version: str,
    rules_df: Optional[pd.DataFrame],
    overwrite_manual: bool,
    enable_llm: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    processing_status = processing_markers_status(ingested) if not ingested.empty else {}

    if ingested.empty:
        manifest = build_manifest(
            input_files,
            total_rows=0,
            excluded_rows=0,
            missing_stats={},
            processing_version=processing_version,
            processing_status=processing_status,
        )
        return ingested, ingested, manifest

    cleaned = initial_clean(ingested)
    missing_stats = compute_missing_stats(cleaned)
    excluded_tagged = mark_inter_account_transfers(cleaned)
    canonical = add_canonical_fields(excluded_tagged, processing_version)
    if rules_df is not None and not rules_df.empty:
        canonical = apply_rules(
            canonical,
            rules_df,
            overwrite_manual=overwrite_manual,
        )
    if enable_llm:
        canonical = classify_card_expenses(canonical, rules_df=rules_df)
    analysis_ready = filter_analysis_ready(canonical)

    manifest = build_manifest(
        input_files,
        total_rows=len(canonical),
        excluded_rows=len(canonical) - len(analysis_ready),
        missing_stats=missing_stats,
        processing_version=processing_version,
        processing_status=processing_status,
    )

    return canonical, analysis_ready, manifest
