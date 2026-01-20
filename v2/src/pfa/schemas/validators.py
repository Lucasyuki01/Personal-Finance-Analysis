from typing import List, Tuple

import pandas as pd

from pfa.schemas.classification_rules import CLASSIFICATION_RULE_COLUMNS


def validate_classification_rules_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate the placeholder schema for classification rules."""
    missing = [col for col in CLASSIFICATION_RULE_COLUMNS if col not in df.columns]
    return (len(missing) == 0), missing

