import pandas as pd

from rules.apply import apply_rules
from rules.schema import CLASSIFICATION_RULE_COLUMNS


def _base_canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "source_account": ["acc1", "acc2"],
            "description_norm": ["coffee shop", "coffee shop"],
            "excluded_reason": ["", ""],
            "flow_type": ["unclassified", "unclassified"],
            "class": ["none", "none"],
            "sub_class": ["none", "none"],
            "is_fixed_waste": [False, False],
            "classification_source": ["unclassified", "unclassified"],
        }
    )


def _rule(
    rule_id: str,
    priority: int,
    match_value: str,
    set_class: str = "",
    set_sub_class: str = "",
    set_flow_type: str = "",
    set_is_fixed_waste=None,
    scope_source_account: str = "",
    match_type: str = "equals",
) -> dict:
    return {
        "rule_id": rule_id,
        "priority": priority,
        "is_active": True,
        "scope_source_account": scope_source_account,
        "match_field": "description_norm",
        "match_type": match_type,
        "match_value": match_value,
        "set_flow_type": set_flow_type,
        "set_class": set_class,
        "set_sub_class": set_sub_class,
        "set_is_fixed_waste": set_is_fixed_waste,
        "created_at": pd.Timestamp("2024-01-01"),
        "updated_at": pd.Timestamp("2024-01-01"),
        "notes": "",
    }


def _rules_df(rules):
    return pd.DataFrame(rules, columns=CLASSIFICATION_RULE_COLUMNS)


def test_rule_priority_resolution() -> None:
    canonical = _base_canonical().head(1)
    rules = _rules_df(
        [
            _rule("r1", priority=1, match_value="coffee shop", set_class="Food"),
            _rule("r2", priority=10, match_value="coffee shop", set_class="Drinks"),
        ]
    )
    result = apply_rules(canonical, rules)
    assert result.loc[0, "class"] == "Food"


def test_scope_source_account() -> None:
    canonical = _base_canonical()
    rules = _rules_df(
        [
            _rule(
                "r1",
                priority=1,
                match_value="coffee",
                set_class="Food",
                match_type="contains",
                scope_source_account="acc1",
            )
        ]
    )
    result = apply_rules(canonical, rules)
    assert result.loc[0, "class"] == "Food"
    assert result.loc[1, "class"] == "none"


def test_match_type_contains() -> None:
    canonical = _base_canonical().head(1)
    rules = _rules_df(
        [_rule("r1", priority=1, match_value="coffee", set_sub_class="Cafe", match_type="contains")]
    )
    result = apply_rules(canonical, rules)
    assert result.loc[0, "sub_class"] == "Cafe"


def test_no_overwrite_manual_by_default() -> None:
    canonical = _base_canonical().head(1)
    canonical.loc[0, "class"] = "Bills"
    canonical.loc[0, "classification_source"] = "manual"
    rules = _rules_df(
        [_rule("r1", priority=1, match_value="coffee shop", set_class="Food")]
    )
    result = apply_rules(canonical, rules)
    assert result.loc[0, "class"] == "Bills"


def test_classification_source_set_to_rules() -> None:
    canonical = _base_canonical().head(1)
    rules = _rules_df(
        [_rule("r1", priority=1, match_value="coffee shop", set_flow_type="waste")]
    )
    result = apply_rules(canonical, rules)
    assert result.loc[0, "classification_source"] == "rules"
