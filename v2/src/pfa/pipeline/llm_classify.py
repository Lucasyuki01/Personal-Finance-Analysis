from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from pfa.constants import (
    DEFAULT_CLASS,
    DEFAULT_CLASSIFICATION_SOURCE,
    DEFAULT_SUB_CLASS,
)

try:  # Optional helper for loading .env files
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback if dependency missing
    load_dotenv = None  # type: ignore[assignment]

try:  # New SDK (>= 1.0)
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

try:  # Legacy SDK support
    import openai as openai_legacy
except ImportError:  # pragma: no cover - optional dependency
    openai_legacy = None


logger = logging.getLogger(__name__)

OPENAI_CARD_MODEL = os.getenv("OPENAI_CARD_MODEL", "gpt-4o")
try:
    OPENAI_CARD_MAX_ATTEMPTS = int(os.getenv("OPENAI_CARD_MAX_ATTEMPTS", "3"))
except ValueError:
    OPENAI_CARD_MAX_ATTEMPTS = 3

FALLBACK_CLASS = "Services"
FALLBACK_SUB_CLASS = "Other services"

DEFAULT_TAXONOMY: Dict[str, List[str]] = {
    "Fix Wastes": [
        "Transport",
        "Rent",
        "Cellphone Bill",
        "Cellphone Insurance",
        "Health Insurance",
        "Gym",
        "Other bills",
        "Services",
    ],
    "Groceries": ["Supermarket", "Convenience", "Liquor store"],
    "Shopping": ["Eletronics", "Dressing", "General"],
    "Eating Out": [
        "Fast food",
        "Bar",
        "Japanese",
        "Korean",
        "Arabic",
        "Latin",
        "Hamburguer",
        "Pizza",
        "Other food",
    ],
    "Entertainment": ["Cinema", "Tourism", "Lime", "Hotel"],
    "Services": ["Uber", "Lyft", "Other services"],
}

CARD_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a careful financial assistant. "
    "Classify card expenses using only the provided taxonomy. "
    "Respond with compact JSON containing the keys 'class' and 'sub_class'."
)

_OPENAI_CHAT_CLIENT: Optional[Any] = None

if load_dotenv:
    load_dotenv()


def classify_card_expenses(
    df: pd.DataFrame,
    rules_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if not _llm_is_available():
        return df.copy()

    df = df.copy()
    eligible = _eligible_card_expenses(df)
    if not eligible.any():
        return df

    catalog = _build_catalog(rules_df)
    classification_cache: Dict[str, Tuple[str, str]] = {}

    for idx in df.index[eligible]:
        row = df.loc[idx]
        descriptor = _normalize_text(row.get("Sub-description", ""))
        cache_key = descriptor

        classification = classification_cache.get(cache_key)
        if classification is None:
            classification = _classify_single_row(row, catalog)
            classification_cache[cache_key] = classification

        df.loc[idx, "class"] = classification[0]
        df.loc[idx, "sub_class"] = classification[1]
        df.loc[idx, "classification_source"] = "openai"

    return df


def _llm_is_available() -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    return (OpenAI is not None) or (openai_legacy is not None)


def _get_openai_chat_client() -> Optional[Any]:
    global _OPENAI_CHAT_CLIENT  # pylint: disable=global-statement
    if OpenAI is None:
        return None
    if _OPENAI_CHAT_CLIENT is None:
        _OPENAI_CHAT_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _OPENAI_CHAT_CLIENT


def _call_openai_chat(messages: Sequence[Dict[str, str]]) -> str:
    if not _llm_is_available():
        raise RuntimeError("OpenAI API is not configured for card classification.")

    last_error: Optional[Exception] = None
    for attempt in range(1, OPENAI_CARD_MAX_ATTEMPTS + 1):
        try:
            if OpenAI is not None:
                client = _get_openai_chat_client()
                if client is None:
                    raise RuntimeError("OpenAI SDK was not initialized.")
                response = client.chat.completions.create(
                    model=OPENAI_CARD_MODEL,
                    messages=messages,
                    temperature=0,
                )
                content = response.choices[0].message.content
                return _content_to_text(content)

            if openai_legacy is not None:
                openai_legacy.api_key = os.getenv("OPENAI_API_KEY")
                response = openai_legacy.ChatCompletion.create(
                    model=OPENAI_CARD_MODEL,
                    messages=messages,
                    temperature=0,
                )
                content = response["choices"][0]["message"]["content"]
                return str(content).strip()

            raise RuntimeError("OpenAI SDK is not installed.")
        except Exception as exc:  # noqa: BLE001 - retry on any API failure
            last_error = exc
            if attempt >= OPENAI_CARD_MAX_ATTEMPTS:
                break
            time.sleep(min(2**attempt, 10))

    if last_error:
        raise last_error
    raise RuntimeError("Unable to obtain a response from OpenAI.")


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            text = ""
            if isinstance(block, dict):
                text = block.get("text", "")
            else:
                text = getattr(block, "text", "")
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    if cleaned.count("{") > 1:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def _build_catalog(rules_df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    if rules_df is None or rules_df.empty:
        return DEFAULT_TAXONOMY.copy()

    class_col = _pick_column(rules_df, ["set_class", "class", "category"])
    sub_col = _pick_column(rules_df, ["set_sub_class", "sub_class", "sub_category"])
    if class_col is None:
        return DEFAULT_TAXONOMY.copy()

    if "is_active" in rules_df.columns:
        rules_df = rules_df[rules_df["is_active"].fillna(False).astype(bool)]

    catalog: Dict[str, set] = {}
    for _, row in rules_df.iterrows():
        raw_class = row.get(class_col)
        if pd.isna(raw_class):
            continue
        category = str(raw_class).strip()
        if not category:
            continue
        sub_value = row.get(sub_col) if sub_col else None
        sub_category = FALLBACK_SUB_CLASS
        if sub_value is not None and not pd.isna(sub_value):
            sub_candidate = str(sub_value).strip()
            if sub_candidate:
                sub_category = sub_candidate
        catalog.setdefault(category, set()).add(sub_category)

    if not catalog:
        return DEFAULT_TAXONOMY.copy()

    if FALLBACK_CLASS not in catalog:
        catalog[FALLBACK_CLASS] = {FALLBACK_SUB_CLASS}

    return {cat: sorted(subs) for cat, subs in catalog.items()}


def _catalog_to_text(catalog: Dict[str, List[str]]) -> str:
    lines = []
    for category in sorted(catalog.keys()):
        subcats = catalog[category]
        subcats_text = ", ".join(subcats) if subcats else FALLBACK_SUB_CLASS
        lines.append(f"- {category}: {subcats_text}")
    return "\n".join(lines)


def _normalize_category(candidate: str, catalog: Dict[str, List[str]]) -> str:
    if candidate:
        candidate_norm = candidate.strip().lower()
        for category in catalog.keys():
            if category.lower() == candidate_norm:
                return category
    return FALLBACK_CLASS if FALLBACK_CLASS in catalog else next(iter(catalog))


def _normalize_sub_category(
    category: str,
    candidate: str,
    catalog: Dict[str, List[str]],
) -> str:
    allowed = catalog.get(category, [])
    if candidate:
        candidate_norm = candidate.strip().lower()
        for subcat in allowed:
            if subcat.lower() == candidate_norm:
                return subcat
    if FALLBACK_SUB_CLASS in allowed:
        return FALLBACK_SUB_CLASS
    return allowed[0] if allowed else FALLBACK_SUB_CLASS


def _build_transaction_summary(row: pd.Series) -> str:
    merchant = str(row.get("Sub-description", "") or "").strip()
    if not merchant:
        merchant = "n/a"
    return f"Sub-description: {merchant}"


def _classify_single_row(
    row: pd.Series,
    catalog: Dict[str, List[str]],
) -> Tuple[str, str]:
    taxonomy_text = _catalog_to_text(catalog)
    transaction_text = _build_transaction_summary(row)
    user_prompt = (
        "Allowed classes and sub-classes:\n"
        f"{taxonomy_text}\n\n"
        "Classify the following card expense:\n"
        f"{transaction_text}\n\n"
        "Answer strictly in JSON with the keys 'class' and 'sub_class'."
    )

    try:
        response_text = _call_openai_chat(
            [
                {"role": "system", "content": CARD_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:  # noqa: BLE001 - propagate fallback
        logger.warning("OpenAI card classification failed: %s", exc)
        return FALLBACK_CLASS, FALLBACK_SUB_CLASS

    payload = _extract_payload(response_text)
    if not payload:
        logger.warning("Could not parse card classification response: %s", response_text)
        return FALLBACK_CLASS, FALLBACK_SUB_CLASS

    raw_class = str(payload.get("class", "") or "").strip()
    raw_sub_class = str(payload.get("sub_class", "") or "").strip()
    category = _normalize_category(raw_class, catalog)
    sub_category = _normalize_sub_category(category, raw_sub_class, catalog)
    return category, sub_category


def _normalize_text(value: object) -> str:
    text = str(value or "").lower().strip()
    return re.sub(r"\s+", " ", text)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _eligible_card_expenses(df: pd.DataFrame) -> pd.Series:
    excluded_series = (
        df["excluded_reason"] if "excluded_reason" in df.columns else pd.Series("", index=df.index)
    )
    excluded = excluded_series.fillna("").astype(str).str.strip().ne("")

    if "channel" in df.columns:
        channel = df["channel"].fillna("").astype(str).str.strip().str.lower()
        is_card = channel.eq("card")
    elif "description_norm" in df.columns:
        is_card = (
            df["description_norm"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("pos purchase")
        )
    else:
        is_card = pd.Series(False, index=df.index)

    amount_series = df["Amount"] if "Amount" in df.columns else pd.Series(0, index=df.index)
    amounts = pd.to_numeric(amount_series, errors="coerce").fillna(0)

    flow_series = df["flow_type"] if "flow_type" in df.columns else pd.Series("", index=df.index)
    flow_type = flow_series.fillna("").astype(str).str.strip().str.lower()
    is_waste = flow_type.eq("waste") | amounts.lt(0)

    class_series = df["class"] if "class" in df.columns else pd.Series(DEFAULT_CLASS, index=df.index)
    sub_series = df["sub_class"] if "sub_class" in df.columns else pd.Series(DEFAULT_SUB_CLASS, index=df.index)
    class_values = class_series.fillna(DEFAULT_CLASS).astype(str)
    sub_values = sub_series.fillna(DEFAULT_SUB_CLASS).astype(str)
    class_default = class_values.str.strip().str.lower().isin(
        {DEFAULT_CLASS.lower(), "", "none"}
    )
    sub_default = sub_values.str.strip().str.lower().isin(
        {DEFAULT_SUB_CLASS.lower(), "", "none"}
    )
    source_series = (
        df["classification_source"]
        if "classification_source" in df.columns
        else pd.Series(DEFAULT_CLASSIFICATION_SOURCE, index=df.index)
    )
    source = source_series.fillna(DEFAULT_CLASSIFICATION_SOURCE).astype(str).str.strip().str.lower()
    source_unclassified = source.eq(DEFAULT_CLASSIFICATION_SOURCE)

    return (~excluded) & is_card & is_waste & source_unclassified & class_default & sub_default


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    columns = {col.lower(): col for col in df.columns}
    for name in candidates:
        key = name.lower()
        if key in columns:
            return columns[key]
    return None
