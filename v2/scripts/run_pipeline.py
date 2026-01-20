from pathlib import Path
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))

    from pfa.io import gather_input_files, write_canonical_base, write_processing_manifest
    from pfa.pipeline import run_pipeline

    sample_dir = repo_root / "data" / "samples"
    output_dir = repo_root / "data" / "outputs"

    input_files = gather_input_files(sample_dir)
    if not input_files:
        print(f"No sample input files found in {sample_dir}")
        return 1

    canonical, _analysis_ready, manifest = run_pipeline(input_files)

    write_canonical_base(canonical, output_dir, write_parquet=False)
    write_processing_manifest(manifest, output_dir)

    print(f"Wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
