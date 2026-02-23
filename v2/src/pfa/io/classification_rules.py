from pathlib import Path

import pandas as pd

from pfa.schemas.classification_rules import CLASSIFICATION_RULE_COLUMNS
from pfa.schemas.validators import validate_classification_rules_schema


def load_classification_rules(path: Path) -> pd.DataFrame:
    """Load classification rules from CSV, returning an empty template if missing."""
    if not path.exists():
        return pd.DataFrame(columns=CLASSIFICATION_RULE_COLUMNS)

    df = pd.read_csv(path)
    valid, missing = validate_classification_rules_schema(df)
    if not valid:
        raise ValueError(
            "Classification rules file missing columns: " + ", ".join(missing)
        )

    return df


def save_classification_rules(df: pd.DataFrame, path: Path) -> Path:
    """Save classification rules to CSV after schema validation."""
    valid, missing = validate_classification_rules_schema(df)
    if not valid:
        raise ValueError(
            "Classification rules file missing columns: " + ", ".join(missing)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

