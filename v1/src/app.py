from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

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


REQUIRED_COLUMNS: Sequence[str] = (
    "Date",
    "Description",
    "Sub-description",
    "Amount",
    "Class",
    "Category",
    "Sub-Category",
)

BASE_DIR = Path(__file__).resolve().parent.parent
POS_RULES_PATH = BASE_DIR / "config" / "pos_rules.csv"
MANUAL_OVERRIDES_PATH = BASE_DIR / "config" / "manual_overrides.csv"

if load_dotenv:
    load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_POS_MODEL = os.getenv("OPENAI_POS_MODEL", "gpt-4o-mini")
try:
    OPENAI_POS_MAX_ATTEMPTS = int(os.getenv("OPENAI_POS_MAX_ATTEMPTS", "3"))
except ValueError:
    OPENAI_POS_MAX_ATTEMPTS = 3

POS_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a meticulous financial assistant that tags Canadian debit point-of-sale "
    "transactions with a strict taxonomy. Only choose categories from the provided list "
    "and always answer with compact JSON containing the keys 'category' and 'sub_category'."
)
_OPENAI_CHAT_CLIENT: Optional[Any] = None


def normalize_text(s: str) -> str:
    t = (str(s) or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _llm_is_available() -> bool:
    """Return True if the OpenAI client can be used."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    return (OpenAI is not None) or (openai_legacy is not None)


def _get_openai_chat_client() -> Optional[Any]:
    """Instantiate the OpenAI client once to avoid re-creating sessions."""
    global _OPENAI_CHAT_CLIENT  # pylint: disable=global-statement
    if OpenAI is None:
        return None
    if _OPENAI_CHAT_CLIENT is None:
        _OPENAI_CHAT_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _OPENAI_CHAT_CLIENT


def _content_to_text(content: Any) -> str:
    """Convert SDK-dependent message content into plain text."""
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


def _call_openai_chat(messages: Sequence[Dict[str, str]]) -> str:
    """Call the OpenAI API with retries, returning the response text."""
    if not _llm_is_available():
        raise RuntimeError("OpenAI API is not configured for POS classification.")

    last_error: Optional[Exception] = None
    for attempt in range(1, OPENAI_POS_MAX_ATTEMPTS + 1):
        try:
            if OpenAI is not None:
                client = _get_openai_chat_client()
                if client is None:
                    raise RuntimeError("OpenAI SDK was not initialized.")
                response = client.chat.completions.create(
                    model=OPENAI_POS_MODEL,
                    messages=messages,
                    temperature=0,
                )
                content = response.choices[0].message.content
                return _content_to_text(content)

            if openai_legacy is not None:
                openai_legacy.api_key = os.getenv("OPENAI_API_KEY")
                response = openai_legacy.ChatCompletion.create(
                    model=OPENAI_POS_MODEL,
                    messages=messages,
                    temperature=0,
                )
                content = response["choices"][0]["message"]["content"]
                return str(content).strip()

            raise RuntimeError("OpenAI SDK is not installed.")
        except Exception as exc:  # noqa: BLE001 - retry on any API failure
            last_error = exc
            if attempt >= OPENAI_POS_MAX_ATTEMPTS:
                break
            time.sleep(min(2**attempt, 10))

    if last_error:
        raise last_error
    raise RuntimeError("Unable to obtain a response from OpenAI.")


def _extract_classification_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse the JSON payload returned by the LLM."""
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


def _build_category_catalog(rules: List[Dict]) -> Dict[str, List[str]]:
    """Create a dictionary of categories -> allowed sub-categories."""
    catalog: Dict[str, set] = {}
    for rule in rules:
        category = str(rule.get("category", "Others") or "Others").strip() or "Others"
        sub_category = str(rule.get("sub_category", "None") or "None").strip() or "None"
        catalog.setdefault(category, set()).add(sub_category)

    if "Others" not in catalog:
        catalog["Others"] = {"None"}

    return {cat: sorted(subs) for cat, subs in catalog.items()}


def _catalog_to_text(catalog: Dict[str, List[str]]) -> str:
    """Render the taxonomy as bullet points for the prompt."""
    lines = []
    for category in sorted(catalog.keys()):
        subcats = catalog[category]
        subcats_text = ", ".join(subcats) if subcats else "None"
        lines.append(f"- {category}: {subcats_text}")
    return "\n".join(lines)


def _normalize_category(candidate: str, catalog: Dict[str, List[str]]) -> str:
    """Map user-provided category to the closest allowed option."""
    if candidate:
        candidate_norm = candidate.strip().lower()
        for category in catalog.keys():
            if category.lower() == candidate_norm:
                return category
    return "Others" if "Others" in catalog else next(iter(catalog))


def _normalize_sub_category(category: str, candidate: str, catalog: Dict[str, List[str]]) -> str:
    """Ensure the selected sub-category exists inside the chosen category."""
    allowed = catalog.get(category, [])
    if candidate:
        candidate_norm = candidate.strip().lower()
        for subcat in allowed:
            if subcat.lower() == candidate_norm:
                return subcat
    if "None" in allowed:
        return "None"
    return allowed[0] if allowed else "None"


def _build_transaction_summary(row: pd.Series) -> str:
    """Structure the transaction information for the prompt."""
    details = []

    date_value = row.get("Date")
    if pd.notna(date_value):
        try:
            date_str = pd.to_datetime(date_value).date().isoformat()
        except Exception:  # noqa: BLE001 - fallback to string
            date_str = str(date_value)
        details.append(f"Date: {date_str}")

    account = str(row.get("Account", "") or "Unknown").strip()
    details.append(f"Account: {account}")

    amount = row.get("Amount", 0.0)
    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        amount_value = 0.0
    details.append(f"Amount (CAD): {amount_value:.2f}")

    merchant = str(row.get("Sub-description", "") or "").strip() or "n/a"
    details.append(f"Merchant descriptor: {merchant}")

    description = str(row.get("Description", "") or "").strip()
    if description:
        details.append(f"Bank description: {description}")

    extra = ""
    for key in ("Memo", "Details", "Notes"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            extra = value.strip()
            break
    if extra:
        details.append(f"Additional detail: {extra}")

    return "\n".join(details)


def _classify_single_pos_row(row: pd.Series, catalog: Dict[str, List[str]]) -> Tuple[str, str]:
    """Call the LLM and return (Category, Sub-Category) for a row."""
    taxonomy_text = _catalog_to_text(catalog)
    transaction_text = _build_transaction_summary(row)
    user_prompt = (
        "Allowed categories and sub-categories:\n"
        f"{taxonomy_text}\n\n"
        "Classify the following point-of-sale transaction:\n"
        f"{transaction_text}\n\n"
        "Answer strictly in JSON with the keys 'category' and 'sub_category'."
    )

    try:
        response_text = _call_openai_chat(
            [
                {"role": "system", "content": POS_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:  # noqa: BLE001 - propagate fallback
        logger.warning("OpenAI POS classification failed: %s", exc)
        return "Others", "None"

    payload = _extract_classification_payload(response_text)
    if not payload:
        logger.warning("Could not parse POS classification response: %s", response_text)
        return "Others", "None"

    raw_category = str(payload.get("category", "") or "").strip()
    raw_sub_category = str(payload.get("sub_category", "") or "None").strip()
    category = _normalize_category(raw_category, catalog)
    sub_category = _normalize_sub_category(category, raw_sub_category, catalog)
    return category, sub_category


def _classify_pos_purchase_with_rules(
    df: pd.DataFrame,
    rules_sorted: List[Dict],
    is_pos: pd.Series,
    sub_norm: pd.Series,
) -> pd.DataFrame:
    """Apply the legacy rule-based classification as a fallback."""

    matched = pd.Series(False, index=df.index)
    for r in rules_sorted:
        pattern = r["pattern"].lower()
        match_type = r.get("match_type", "contains")

        if match_type == "startswith":
            mask = is_pos & ~matched & sub_norm.str.startswith(pattern)
        elif match_type == "regex":
            mask = is_pos & ~matched & sub_norm.map(lambda x: bool(r["_compiled"].search(x)))
        else:
            mask = is_pos & ~matched & sub_norm.str.contains(re.escape(pattern), regex=True)

        if mask.any():
            df.loc[mask, "Category"] = r["category"]
            df.loc[mask, "Sub-Category"] = r.get("sub_category", "None")
            matched |= mask

    still_pos_unmatched = is_pos & ~matched
    df.loc[still_pos_unmatched, ["Category", "Sub-Category"]] = ["Others", "None"]
    return df


def _classify_pos_purchase_with_llm(
    df: pd.DataFrame,
    rules_sorted: List[Dict],
    is_pos: pd.Series,
) -> pd.DataFrame:
    """Classify POS rows using the OpenAI LLM."""

    catalog = _build_category_catalog(rules_sorted)
    classification_cache: Dict[Tuple[str, float, str], Tuple[str, str]] = {}
    pos_indices = df.index[is_pos]
    for idx in pos_indices:
        row = df.loc[idx]
        descriptor = normalize_text(row.get("Sub-description", ""))
        account = normalize_text(row.get("Account", ""))
        amount = row.get("Amount", 0.0)
        try:
            amount_value = float(amount)
        except (TypeError, ValueError):
            amount_value = 0.0
        cache_key = (descriptor, round(amount_value, 2), account)

        classification = classification_cache.get(cache_key)
        if classification is None:
            classification = _classify_single_pos_row(row, catalog)
            classification_cache[cache_key] = classification

        df.loc[idx, ["Category", "Sub-Category"]] = classification

    return df


def classify_pos_purchase(df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
    """Classify POS rows via OpenAI when possible, falling back to rules."""

    df = df.copy()

    is_pos = df["Description"].fillna("").map(normalize_text).eq("pos purchase")
    sub_norm = df["Sub-description"].fillna("").map(normalize_text)

    if "Category" not in df.columns:
        df["Category"] = "Others"
    if "Sub-Category" not in df.columns:
        df["Sub-Category"] = "None"

    if not is_pos.any():
        return df

    rules_sorted = sorted(rules, key=lambda r: r.get("priority", 9999))
    for r in rules_sorted:
        if r.get("match_type") == "regex":
            r["_compiled"] = re.compile(r["pattern"], flags=re.I)

    if _llm_is_available():
        try:
            return _classify_pos_purchase_with_llm(df, rules_sorted, is_pos)
        except Exception as exc:  # noqa: BLE001 - fallback to deterministic rules
            logger.warning("Falling back to rule-based POS classification: %s", exc)

    return _classify_pos_purchase_with_rules(df, rules_sorted, is_pos, sub_norm)


def classify_transactions(df: pd.DataFrame, pos_rules: List[Dict]) -> pd.DataFrame:
    """Classify transactions into Class, Category, Sub-Category, including POS."""
    # Trabalha numa cópia para evitar efeitos colaterais no DF original
    df = df.copy()

    # Inicializa colunas de saída
    # Convenção de sinal: Amount > 0 => Earnings; Amount < 0 => Expenses
    df["Class"] = "Expenses"
    df.loc[df["Amount"] > 0, "Class"] = "Earnings"
    df.loc[df["Amount"] < 0, "Class"] = "Expenses"
    df["Category"] = "Others"
    df["Sub-Category"] = "None"

    # Normaliza Description uma única vez para facilitar comparações
    desc = df["Description"].fillna("").map(normalize_text)

    # Masks para matches exatos em descrições normalizadas
    mask_payroll  = desc.eq("payroll deposit")
    mask_correct  = desc.eq("correction")
    mask_interest = desc.eq("interest")
    mask_deposit  = desc.eq("deposit")

    # Regras de earnings por tipo de crédito
    df.loc[mask_payroll,  ["Class", "Category"]] = ["Earnings", "Payment"]
    df.loc[mask_correct,  ["Class", "Category"]] = ["Earnings", "earnings"]
    df.loc[mask_interest, ["Class", "Category"]] = ["Earnings", "earnings"]

    # Depósitos genéricos: earnings, categoria base "shared bills";
    # com faixa específica, reclassifica como "Payment"
    df.loc[mask_deposit, "Class"] = "Earnings"
    df.loc[mask_deposit, "Category"] = "shared bills"
    df.loc[
        mask_deposit & df["Amount"].between(300, 900, inclusive="neither"),
        "Category",
    ] = "Payment"

    # Saques (withdrawal): regras específicas por valor/índice
    mask_withdraw = desc.eq("withdrawal")
    # Atenção: usar índices fixos é frágil se o DF muda (filtros/sorts)
    df.loc[mask_withdraw & df.index.isin([399, 352, 289]), "Category"] = "Shopping"
    df.loc[mask_withdraw & (df["Amount"] == -1200), ["Category", "Sub-Category"]] = ["Bills", "Rent"]
    df.loc[mask_withdraw & df["Category"].eq("Others"), "Category"] = "Money Sent"

    # Outros termos específicos de cobrança
    df.loc[desc.eq("bill payment"),    ["Category", "Sub-Category"]] = ["Bills", "Cellphone"]
    df.loc[desc.eq("service charge"),  ["Category", "Sub-Category"]] = ["Bills", "Bank"]

    # Classificação baseada em regras POS (pattern/match_type/priority)
    df = classify_pos_purchase(df, pos_rules)
    return df


def load_pos_rules() -> List[Dict]:
    """Load POS classification rules from CSV."""
    # Verifica se existe o path para o csv com as regras
    if not POS_RULES_PATH.exists():
        raise FileNotFoundError(
            f"POS rules file not found at {POS_RULES_PATH}. Please create it."
        )

    #salva o df nesta var
    rules_df = pd.read_csv(POS_RULES_PATH)

    # Define quais colunas são obrigatórias no arquivo de regras POS.
    # O programa não funcionará se faltar alguma delas.
    required_cols = {"pattern", "category", "sub_category", "priority", "match_type"}

    # Cria um dicionário que relaciona o nome das colunas (em minúsculas)
    # ao nome original encontrado no CSV. Isso permite reconhecer colunas
    # mesmo que o arquivo tenha letras maiúsculas/minúsculas diferentes.
    columns_map = {col.lower(): col for col in rules_df.columns}

    # Verifica se alguma das colunas obrigatórias está faltando.
    # 'difference' devolve o conjunto de colunas que deveriam existir,
    # mas não foram encontradas no CSV.
    missing_cols = required_cols.difference(columns_map.keys())

    # Se alguma coluna obrigatória estiver ausente, gera um erro informando
    # quais estão faltando para facilitar a correção do arquivo.
    if missing_cols:
        raise ValueError(f"POS rules CSV is missing columns: {', '.join(sorted(missing_cols))}")

    # Renomeia as colunas do DataFrame para padronizar os nomes em minúsculas.
    # Exemplo: 'Pattern' → 'pattern', 'Category' → 'category'
    rules_df = rules_df.rename(columns={columns_map[key]: key for key in required_cols})

    # Converte a coluna 'priority' para número inteiro.
    # Valores inválidos ou ausentes são substituídos por 9999 (baixa prioridade).
    rules_df["priority"] = pd.to_numeric(rules_df["priority"], errors="coerce").fillna(9999).astype(int)

    # Garante que a coluna 'match_type' nunca tenha valores nulos e esteja padronizada em minúsculas.
    # Exemplo: 'STARTSWITH' → 'startswith'
    rules_df["match_type"] = rules_df["match_type"].fillna("contains").str.lower()

    # Substitui valores ausentes na coluna 'sub_category' por "None"
    # para evitar problemas ao classificar as transações.
    rules_df["sub_category"] = rules_df["sub_category"].fillna("None")

    # Retorna o DataFrame convertido em uma lista de dicionários,
    # onde cada linha representa uma regra pronta para ser aplicada.
    return rules_df.to_dict(orient="records")


def load_manual_overrides() -> pd.DataFrame:
    """Load manual classification overrides from CSV."""
    
    # Verifica se o arquivo CSV com classificações manuais existe.
    # Caso não exista, retorna um DataFrame vazio já com as colunas esperadas.
    # Isso evita erros e permite que o programa continue rodando normalmente.
    if not MANUAL_OVERRIDES_PATH.exists():
        return pd.DataFrame(
            columns=["Date", "Description", "Sub-description", "Amount", "Category", "Sub-Category"]
        )

    # Carrega o arquivo CSV com as classificações manuais feitas pelo usuário.
    overrides = pd.read_csv(MANUAL_OVERRIDES_PATH)

    # Define as colunas que o CSV precisa obrigatoriamente conter.
    expected_cols = {"Date", "Description", "Sub-description", "Amount", "Category", "Sub-Category"}

    # Verifica se alguma das colunas obrigatórias está faltando no arquivo.
    missing = expected_cols.difference(overrides.columns)

    # Caso alguma coluna esteja ausente, gera um erro informando quais estão faltando.
    if missing:
        raise ValueError(
            f"Manual overrides CSV is missing columns: {', '.join(sorted(missing))}"
        )

    # Converte a coluna "Date" para o tipo datetime.
    # Valores inválidos viram NaT (nulo).
    overrides["Date"] = pd.to_datetime(overrides["Date"], errors="coerce")

    # Converte a coluna "Amount" para tipo numérico (float).
    # Valores não numéricos são convertidos em NaN.
    overrides["Amount"] = pd.to_numeric(overrides["Amount"], errors="coerce")

    # Remove linhas que não possuem informações essenciais:
    # data, descrição, valor ou categoria.
    overrides = overrides.dropna(subset=["Date", "Description", "Amount", "Category"])

    # Preenche valores nulos na coluna "Sub-Category" com "None"
    # para manter a consistência e evitar erros nas etapas seguintes.
    overrides["Sub-Category"] = overrides["Sub-Category"].fillna("None")

    # Retorna o DataFrame final, já limpo e padronizado,
    # pronto para ser usado nas classificações manuais.
    return overrides


def load_bank_data() -> pd.DataFrame:
    """Load acc1.csv and acc2.csv from ../data, tagging their origin."""
    #Dando a rota dos arquivos que serão usados
    data_dir = Path(__file__).resolve().parent.parent / "data"
    acc1_path = data_dir / "acc1.csv"
    acc2_path = data_dir / "acc2.csv"

    #Essa linha cria uma lista com o nome de cada arquivo (acc1.csv, acc2.csv) que não foi encontrado no diretório especificado.
    #Serve para verificar se os arquivos necessários estão lá antes de tentar carregá-los.
    missing = [p.name for p in (acc1_path, acc2_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required dataset(s) not found in {data_dir}: {', '.join(missing)}"
        )

    # Faz uma cópia COMPLETA (.copy) do DataFrame para evitar avisos (SettingWithCopyWarning)
    # e garantir que futuras modificações não afetem uma possível view retornada por read_csv().
    acc1 = pd.read_csv(acc1_path).copy()
    acc2 = pd.read_csv(acc2_path).copy()
    acc1["Account"] = "Chequing"
    acc2["Account"] = "Savings"

    return pd.concat([acc1, acc2], ignore_index=True)


def clean_bank_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the requested cleaning steps before classification."""
    # Checa se o df passado esta vazio
    if df_raw.empty:
        return df_raw

    # Faz outra cópia completa
    df = df_raw.copy()
    
    # Remove espaços em branco no início e no final dos nomes das colunas.
    # Isso evita erros causados por cabeçalhos vindos de CSVs com espaços extras,
    # garantindo que " Description " e "Description" sejam tratados como o mesmo nome.
    df.columns = df.columns.str.strip()

    # Dropando as colunas que não somam em minhas análises
    # Flag: Pode ser muito específico e melhor rever se for generalizar
    for col in ["Filter", "Type of Transaction"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Dropando as linhas que representam trasações internas pra evitar inconsistências
    # Flag: Mais um caso que pode ser muito específico, lembrar de checar para outros tipos de conta scotiabank se os nomes se mantem assim
    if "Description" in df.columns:
        df = df[df["Description"] != "customer transfer cr."]
        df = df[df["Description"] != "customer transfer dr."]

    # Garante que a coluna "Sub-description" exista e esteja limpa:
    # - Substitui valores nulos, vazios ou compostos apenas por espaços por "none"
    # - Caso a coluna não exista no DataFrame, cria uma nova preenchida com "none"
    if "Sub-description" in df.columns:
        sub_desc = df["Sub-description"].fillna("").astype(str).str.strip()
        sub_desc = sub_desc.replace("", "none")
        df["Sub-description"] = sub_desc
    else:
        df["Sub-description"] = "none"

    return df


def apply_manual_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Apply manual category overrides using Date/Description/Sub-description/Amount."""
    if overrides.empty:
        return df

    df = df.copy()
    key_cols = ["Date", "Description", "Sub-description", "Amount"]

    def normalize_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip().str.lower()

    overrides = overrides.copy()
    overrides["Date_key"] = overrides["Date"].dt.normalize()
    overrides["Description_key"] = normalize_series(overrides["Description"])
    overrides["Sub-description_key"] = normalize_series(overrides["Sub-description"])

    overrides_map = {
        (
            row["Date_key"],
            row["Description_key"],
            row["Sub-description_key"],
            float(row["Amount"]),
        ): (row["Category"], row["Sub-Category"])
        for _, row in overrides.iterrows()
    }

    df["Date_key"] = df["Date"].dt.normalize()
    df["Description_key"] = normalize_series(df["Description"])
    df["Sub-description_key"] = normalize_series(df["Sub-description"])

    mask = []
    categories = []
    subcats = []
    for _, row in df.iterrows():
        key = (
            row["Date_key"],
            row["Description_key"],
            row["Sub-description_key"],
            float(row["Amount"]),
        )
        override = overrides_map.get(key)
        if override:
            mask.append(True)
            categories.append(override[0])
            subcats.append(override[1])
        else:
            mask.append(False)
            categories.append(row["Category"])
            subcats.append(row["Sub-Category"])

    mask = pd.Series(mask, index=df.index)
    if mask.any():
        df.loc[mask, "Category"] = pd.Series(categories, index=df.index)[mask]
        df.loc[mask, "Sub-Category"] = pd.Series(subcats, index=df.index)[mask]

    df = df.drop(columns=["Date_key", "Description_key", "Sub-description_key"])
    return df


def classify_bank_df(df: pd.DataFrame, pos_rules: List[Dict], overrides: pd.DataFrame) -> pd.DataFrame:
    """Run the classification and ensure required columns exist."""
    # Se o DataFrame estiver vazio, não há nada a classificar.
    if df.empty:
        return df

    # Garante que a coluna "Amount" seja numérica.
    # Valores inválidos (como texto ou símbolos) são convertidos em NaN.
    df["Amount"] = pd.to_numeric(df.get("Amount"), errors="coerce")

    # Remove linhas que não possuem um valor válido na coluna "Amount".
    df = df.dropna(subset=["Amount"])

    # Converte a coluna "Date" para o tipo datetime.
    # Linhas com datas inválidas são removidas.
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df = df.dropna(subset=["Date"])

    # Aplica as regras automáticas de classificação (POS rules).
    df = classify_transactions(df, pos_rules)

    # Aplica as classificações manuais, substituindo as automáticas quando houver correspondência.
    df = apply_manual_overrides(df, overrides)

    # Garante que todas as colunas obrigatórias existam.
    # Se alguma estiver ausente, é criada e preenchida com "Unknown".
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown"

    # Substitui valores nulos em colunas-chave por "Unknown"
    # para manter a consistência e evitar erros em análises posteriores.
    for col in ("Class", "Category", "Sub-Category"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Garante que a coluna "Account" exista,
    # preenchendo com "Unknown" caso não tenha sido definida anteriormente.
    if "Account" not in df.columns:
        df["Account"] = "Unknown"

    # Retorna o DataFrame classificado e padronizado,
    # pronto para ser usado em visualizações e análises.
    return df


def apply_filters(
    df: pd.DataFrame,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    classes: Sequence[str],
    categories: Sequence[str],
    subcats: Sequence[str],
) -> pd.DataFrame:
    """Apply date and categorical filters to the DataFrame."""
    if df.empty:
        return df

    start, end = date_range
    filtered = df[df["Date"].between(start, end)]

    if classes:
        filtered = filtered[filtered["Class"].isin(classes)]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    if subcats:
        filtered = filtered[filtered["Sub-Category"].isin(subcats)]
    return filtered


def compute_kpis(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Return total spent (abs), total earned, and their delta."""
    # Se o DataFrame estiver vazio, retorna zeros para evitar erros
    if df.empty:
        return 0.0, 0.0, 0.0

    # Soma das despesas (valores negativos)
    spent = float(df.loc[df["Amount"] < 0, "Amount"].sum())

    # Soma dos ganhos (valores positivos)
    earned = float(df.loc[df["Amount"] > 0, "Amount"].sum())

    # Converte o total gasto para valor absoluto (positivo)
    spent_abs = abs(spent)

    # Calcula o delta líquido: ganhos - gastos
    delta = earned - spent_abs

    # Retorna: (total gasto, total ganho, diferença líquida)
    return spent_abs, earned, delta


def top10_expenses(df: pd.DataFrame, selected_category: str) -> pd.DataFrame:
    """Return the top 10 expenses sorted by absolute amount."""
    expenses = df[df["Amount"] < 0].copy()
    if selected_category and selected_category != "All categories":
        expenses = expenses[expenses["Category"] == selected_category]
    if expenses.empty:
        return expenses
    top10 = (
        expenses.assign(_abs=lambda s: s["Amount"].abs())
        .nlargest(10, "_abs")
        .drop(columns="_abs")
    )
    return top10


def monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate absolute expenses per Year-Month."""
    # Filtra apenas as despesas (valores negativos)
    expenses = df[df["Amount"] < 0].copy()

    # Retorna DataFrame vazio com colunas padronizadas se não houver despesas
    if expenses.empty:
        return pd.DataFrame(columns=["Month", "Total Spent"])

    # Extrai o mês/ano de cada transação e converte para timestamp (ex.: 2024-05 → 2024-05-01)
    expenses["Month"] = expenses["Date"].dt.to_period("M").dt.to_timestamp()

    # Agrupa por mês e soma os valores absolutos dos gastos
    monthly = (
        expenses.groupby("Month", as_index=False)["Amount"]
        .sum()
        .assign(Amount=lambda s: s["Amount"].abs())  # transforma em valor positivo
        .rename(columns={"Amount": "Total Spent"})   # renomeia a coluna
    )

    # Retorna a série mensal de despesas
    return monthly


def monthly_earnings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings per Year-Month."""
    # Filtra apenas as transações de ganhos (valores positivos)
    earnings = df[df["Amount"] > 0].copy()

    # Retorna DataFrame vazio com colunas padronizadas se não houver ganhos
    if earnings.empty:
        return pd.DataFrame(columns=["Month", "Total Earned"])

    # Extrai o mês/ano de cada transação e converte para timestamp (ex.: 2024-05 → 2024-05-01)
    earnings["Month"] = earnings["Date"].dt.to_period("M").dt.to_timestamp()

    # Agrupa por mês e soma os valores de ganhos
    monthly = (
        earnings.groupby("Month", as_index=False)["Amount"]
        .sum()
        .rename(columns={"Amount": "Total Earned"})  # renomeia a coluna
    )

    # Retorna a série mensal de ganhos
    return monthly


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:

    # Filtra apenas as despesas (valores negativos) e agrega por categoria.
    # - Se não houver despesas, retorna um DataFrame vazio com colunas padrão.
    # - Soma os valores por categoria e converte para valor absoluto (positivo).
    # - Renomeia a coluna "Amount" para "Total Spent" para clareza.

    """Aggregate absolute expenses per category."""
    expenses = df[df["Amount"] < 0]
    if expenses.empty:
        return pd.DataFrame(columns=["Category", "Total Spent"])
    return (
        expenses.groupby("Category", as_index=False)["Amount"]
        .sum()
        .assign(Amount=lambda s: s["Amount"].abs())
        .rename(columns={"Amount": "Total Spent"})
    )


def earnings_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate earnings totals per category."""
    
    # Filtra apenas as transações de ganhos (valores positivos)
    earnings = df[df["Amount"] > 0]

    # Caso não existam ganhos no DataFrame, retorna um DataFrame vazio
    # com as colunas padronizadas para evitar erros na visualização
    if earnings.empty:
        return pd.DataFrame(columns=["Category", "Total Earned"])

    # Agrupa os ganhos por categoria e soma o total recebido em cada uma
    # - as_index=False mantém "Category" como coluna normal, não índice
    # - rename muda o nome da coluna "Amount" para "Total Earned"
    return (
        earnings.groupby("Category", as_index=False)["Amount"]
        .sum()
        .rename(columns={"Amount": "Total Earned"})
    )


def subcategory_distribution(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Aggregate absolute totals per sub-category for a given category."""
    
    # Cria um subconjunto contendo apenas as transações da categoria escolhida
    subset = df[df["Category"] == category].copy()

    # Se a categoria não tiver registros, retorna DataFrame vazio com colunas padrão
    if subset.empty:
        return pd.DataFrame(columns=["Sub-Category", "Total"])

    # Cria uma nova coluna com o valor absoluto de cada transação
    # (remove o sinal negativo para facilitar somas e visualizações)
    subset["Total"] = subset["Amount"].abs()

    # Agrupa as transações por subcategoria e soma os valores totais
    # - as_index=False mantém a coluna "Sub-Category" visível
    # - sort_values ordena do maior para o menor total
    grouped = (
        subset.groupby("Sub-Category", as_index=False)["Total"]
        .sum()
        .sort_values("Total", ascending=False)
    )

    # Retorna o DataFrame final com totais por subcategoria
    return grouped


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def style_pie_with_values(fig) -> None:
    """Show absolute values and percentages on pie chart slices."""
    fig.update_traces(
        texttemplate="%{label}<br>$%{value:,.2f}<br>%{percent:.1%}",
        hovertemplate="%{label}<br>$%{value:,.2f}<br>%{percent:.1%}",
        textposition="inside",
    )
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode="hide")


def daily_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily cumulative balance."""
    if df.empty:
        return pd.DataFrame(columns=["Date", "Balance"])

    df_sorted = df.sort_values("Date")
    daily_net = (
        df_sorted.groupby("Date", as_index=False)["Amount"]
        .sum()
        .sort_values("Date")
    )
    daily_net["Balance"] = daily_net["Amount"].cumsum()
    return daily_net[["Date", "Balance"]]


def earnings_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return earnings rows sorted by date descending."""
    # Filtra apenas as transações de ganhos (valores positivos)
    earnings = df[df["Amount"] > 0].copy()

    # Se não houver ganhos no DataFrame, retorna vazio
    if earnings.empty:
        return earnings

    # Ordena os ganhos da data mais recente para a mais antiga
    return earnings.sort_values("Date", ascending=False)


def expense_stats(df: pd.DataFrame) -> Tuple[float, float]:
    """Return median and average spending (absolute)."""
    # Filtra apenas as despesas (valores negativos) e converte para valores absolutos
    expenses = df[df["Amount"] < 0]["Amount"].abs()

    # Se não houver despesas, retorna 0 para ambas as métricas
    if expenses.empty:
        return 0.0, 0.0

    # Calcula a mediana e a média de gastos
    median_val = float(expenses.median())
    mean_val = float(expenses.mean())

    # Retorna ambas as métricas como floats
    return median_val, mean_val


def main() -> None:
    # Apenas colocando o titulo e etc
    st.set_page_config(page_title="Personal Finance Analysis", layout="wide")
    st.title("Personal Finance Analysis")
    st.caption("Data automatically loaded from acc1.csv and acc2.csv.")

    # Chamando a função de carregar os CSVs
    try:
        bank_df_raw = load_bank_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    # Chamando a função de limpeza no df e salvando esta versão limpa em uma nova var
    bank_df_clean = clean_bank_df(bank_df_raw)
    if bank_df_clean.empty:
        st.error("No transactions available after mandatory cleaning.")
        return

    # Carregando as regras de pos (poderia ser substituido pela API de classificação)
    try:
        pos_rules = load_pos_rules()
        
        # Aqui é onde carregamos os casos especificos que indentifiquei em meus arquivos, como compras feitas no marketplace ou pagamento de alugueis por e-transfer
        manual_overrides = load_manual_overrides()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    # Aqui rodamos a função de classificação
    bank_df_classified = classify_bank_df(bank_df_clean, pos_rules, manual_overrides)
    if bank_df_classified.empty:
        st.error("No transactions available after classification.")
        return
    
    # Remove transações com valor igual a zero e cria uma cópia independente
    # para evitar avisos (SettingWithCopyWarning) e garantir segurança na manipulação.
    bank_df_classified = bank_df_classified[bank_df_classified["Amount"] != 0].copy()
    
    # Reordenando e resetando o index
    display_df = bank_df_classified.sort_values("Date").reset_index(drop=True)

    min_date = display_df["Date"].min().date()
    max_date = display_df["Date"].max().date()

    # Se ainda não existir um intervalo de datas salvo na sessão,
    # define o valor inicial como o intervalo completo disponível (min_date → max_date).
    # Isso garante que o filtro de datas tenha um valor padrão e persista entre interações.
    if "date_range" not in st.session_state:
        st.session_state["date_range"] = (min_date, max_date)

    
    # Cria a barra lateral com filtros interativos
    with st.sidebar:
        # Títulos de organização visual
        st.header("Filters")
        st.subheader("Date selection")

        # Filtro de intervalo de datas (início e fim)
        precise_range = st.date_input(
            "Precise range",
            value=st.session_state["date_range"],  # intervalo salvo na sessão
            min_value=min_date,                    # menor data disponível no dataset
            max_value=max_date,                    # maior data disponível no dataset
        )

        # Valida o intervalo de datas e atualiza a sessão
        if isinstance(precise_range, (list, tuple)) and len(precise_range) == 2:
            start_input, end_input = precise_range
            if start_input <= end_input:
                st.session_state["date_range"] = (start_input, end_input)
            else:
                st.error("Start date must be before or equal to end date.")

        # Botão para restaurar o período completo e recarregar o app
        if st.button("Show full period"):
            st.session_state["date_range"] = (min_date, max_date)
            st.rerun()

        # Filtro por "Class" (Earnings, Expenses)
        class_options = sorted(display_df["Class"].dropna().unique().tolist())
        selected_classes = st.multiselect(
            "Class",
            options=class_options,
            default=class_options,  # mostra todas por padrão
        )

        # Filtro por "Category" (Food, Transport, etc.)
        category_options = sorted(display_df["Category"].dropna().unique().tolist())
        selected_categories = st.multiselect(
            "Category",
            options=category_options,
            default=category_options,
        )

        # Filtro por "Sub-Category" (Restaurantes, Gasolina, etc.)
        subcat_options = sorted(display_df["Sub-Category"].dropna().unique().tolist())
        selected_subcats = st.multiselect(
            "Sub-Category",
            options=subcat_options,
            default=subcat_options,
        )

        # Seletor para o Top 10 de gastos por categoria
        category_selector_options = ["All categories"] + category_options
        selected_category = st.selectbox(
            "Category for Top 10 table",
            options=category_selector_options,
            index=0,  # padrão: "All categories"
        )

    # Slider para ajuste rápido do intervalo de datas.
    # Sincroniza com o valor salvo em st.session_state["date_range"]
    # e recarrega o app sempre que o usuário altera o intervalo,
    # garantindo que o filtro de datas permaneça consistente entre o slider e a sidebar.
    slider_value = st.slider(
        "Date range (quick adjust)",
        min_value=min_date,
        max_value=max_date,
        value=st.session_state["date_range"],
        format="YYYY-MM-DD",
    )
    if slider_value != st.session_state["date_range"]:
        st.session_state["date_range"] = slider_value
        st.rerun()

    # Desempacota o intervalo de datas salvo na sessão em duas variáveis
    # para facilitar o uso em filtros e análises.
    start_date, end_date = st.session_state["date_range"]

    # Aplica todos os filtros selecionados pelo usuário:
    # - intervalo de datas (convertido para Timestamp)
    # - classes (Earnings / Expenses)
    # - categorias e subcategorias escolhidas na barra lateral.
    # Retorna um novo DataFrame apenas com as transações que atendem a esses critérios.
    filtered_df = apply_filters(
        display_df,
        (pd.Timestamp(start_date), pd.Timestamp(end_date)),
        selected_classes,
        selected_categories,
        selected_subcats,
    )

    # KPIs e agregações para os gráficos/tabelas:
    # - earnings_table: tabela de ganhos (Amount > 0) já no período filtrado
    # - expense_stats: mediana e média de despesas (usando Amount < 0)
    earnings_df = earnings_table(filtered_df)
    expense_median, expense_mean = expense_stats(filtered_df)
    expenses_in_period = filtered_df[filtered_df["Amount"] < 0].copy()

    # - compute_kpis: total gasto, total ganho e delta (ganhos - gastos)
    total_spent, total_earned, delta = compute_kpis(filtered_df)

    # - Métricas exibidas em 5 colunas com st.metric e valores formatados
    # - monthly_spending / monthly_earnings: agregações mensais para gráficos
    # - category_distribution / earnings_category_distribution: distribuição por categoria (despesas/ganhos)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total spent", format_currency(total_spent))
    metric_cols[1].metric("Total earned", format_currency(total_earned))
    metric_cols[2].metric("Net delta", format_currency(delta))
    metric_cols[3].metric("Median spending", format_currency(expense_median))
    metric_cols[4].metric("Average spending", format_currency(expense_mean))

    # Gera os DataFrames agregados usados nos gráficos e tabelas do dashboard:
    # - monthly_df: soma mensal de despesas (Amount < 0)
    # - monthly_earnings_df: soma mensal de ganhos (Amount > 0)
    # - category_df: total de despesas por categoria (valores absolutos)
    # - earnings_category_df: total de ganhos por categoria
    monthly_df = monthly_spending(filtered_df)
    monthly_earnings_df = monthly_earnings(filtered_df)
    category_df = category_distribution(filtered_df)
    earnings_category_df = earnings_category_distribution(filtered_df)


    # Gráfico de barras (mensal) das despesas e estatísticas associadas
    expenses_chart_col, expenses_pie_col = st.columns(2)
    with expenses_chart_col:
        # Verifica se há dados disponíveis para o período selecionado
        if monthly_df.empty:
            st.info("No expenses available for the selected range.")
        else:
            # Cria gráfico de barras com Plotly (despesas por mês)
            fig_monthly = px.bar(
                monthly_df,
                x="Month",
                y="Total Spent",
                title="Monthly expenses",
                labels={"Month": "Month", "Total Spent": "Total spent"},
                color_discrete_sequence=px.colors.sequential.OrRd,
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

            # Calcula e exibe a média mensal de despesas
            avg_monthly_expense = float(monthly_df["Total Spent"].mean())
            st.caption(f"Average monthly expenses: {format_currency(avg_monthly_expense)}")

            # (Opcional) Calcula média excluindo outliers via método do IQR
            if not expenses_in_period.empty:
                exp_tmp = expenses_in_period.copy()
                exp_tmp["AbsAmount"] = exp_tmp["Amount"].abs()

                # Intervalo interquartil
                q1 = exp_tmp["AbsAmount"].quantile(0.25)
                q3 = exp_tmp["AbsAmount"].quantile(0.75)
                iqr = q3 - q1
                upper = q3 + 1.5 * iqr

                # Remove outliers, mas mantém "Rent"
                mask = (exp_tmp["AbsAmount"] <= upper) | exp_tmp["Sub-Category"].fillna("").str.lower().eq("rent")
                trimmed_series = exp_tmp.loc[mask, "AbsAmount"]

                # Média ajustada
                trimmed_mean = float(trimmed_series.mean()) if not trimmed_series.empty else float(exp_tmp["AbsAmount"].mean())
                # st.caption(f"Average monthly expenses (without outliers): {format_currency(trimmed_mean)}")

    # Gráfico de pizza exibindo a distribuição das despesas por categoria
    with expenses_pie_col:
        # Verifica se há dados de categorias para o período selecionado
        if category_df.empty:
            st.info("No expense category distribution to display.")
        else:
            # Cria o gráfico de pizza com Plotly
            fig_category = px.pie(
                category_df,
                names="Category",          # Categorias = nomes das fatias
                values="Total Spent",      # Tamanho das fatias = total gasto
                title="Expense distribution by category",
                color_discrete_sequence=px.colors.sequential.YlOrRd,  # Paleta de cores quentes
            )

            # Aplica formatação personalizada (exibe valores e porcentagens nas fatias)
            style_pie_with_values(fig_category)

            # Exibe o gráfico no dashboard, ajustando à largura da coluna
            st.plotly_chart(fig_category, use_container_width=True)


    # Exibe tabela detalhada das despesas do período selecionado
    if not expenses_in_period.empty:
        # Cria cópia para evitar modificar o DataFrame original
        expenses_display = expenses_in_period.copy()

        # Converte os valores para positivos (facilita leitura)
        expenses_display["Amount"] = expenses_display["Amount"].abs()

        # Formata a coluna de data para exibir apenas ano-mês-dia
        if "Date" in expenses_display.columns:
            expenses_display["Date"] = expenses_display["Date"].dt.date

        # Cria painel recolhível com a tabela de despesas
        with st.expander("Expenses in period", expanded=False):
            st.dataframe(
                # Exibe apenas as colunas principais, se existirem no DataFrame
                expenses_display[
                    [
                        col
                        for col in [
                            "Date",
                            "Description",
                            "Sub-description",
                            "Amount",
                            "Category",
                            "Sub-Category",
                            "Account",
                        ]
                        if col in expenses_display.columns
                    ]
                ],
                use_container_width=True,  # Ajusta tabela à largura da tela
            )

    # Caso não existam despesas, exibe aviso
    else:
        st.info("No expenses in the selected range.")


    # Gráfico de barras (mensal) dos ganhos e estatísticas associadas
    earnings_chart_col, earnings_pie_col = st.columns(2)
    with earnings_chart_col:
        # Verifica se há dados de ganhos no período selecionado
        if monthly_earnings_df.empty:
            st.info("No earnings available for the selected range.")
        else:
            # Cria gráfico de barras com Plotly (ganhos por mês)
            fig_earnings = px.bar(
                monthly_earnings_df,
                x="Month",
                y="Total Earned",
                title="Monthly earnings",
                labels={"Month": "Month", "Total Earned": "Total earned"},
                color_discrete_sequence=px.colors.sequential.Blues,  # Paleta azul (ganhos)
            )
            st.plotly_chart(fig_earnings, use_container_width=True)

            # Calcula e exibe a média mensal de ganhos
            avg_monthly_income = float(monthly_earnings_df["Total Earned"].mean())
            st.caption(f"Average monthly earnings: {format_currency(avg_monthly_income)}")


    # Gráfico de pizza mostrando a distribuição dos ganhos por categoria
    with earnings_pie_col:
        # Verifica se há dados disponíveis
        if earnings_category_df.empty:
            st.info("No earnings category distribution to display.")
        else:
            # Cria gráfico de pizza com Plotly (ganhos por categoria)
            fig_category_earnings = px.pie(
                earnings_category_df,
                names="Category",          # Nome das fatias = categorias
                values="Total Earned",     # Tamanho das fatias = valor total ganho
                title="Earnings distribution by category",
                color_discrete_sequence=px.colors.sequential.Blues_r,  # Tons de azul (invertidos)
            )

            # Aplica formatação visual (exibe valores e porcentagens)
            style_pie_with_values(fig_category_earnings)

            # Exibe o gráfico ajustado à largura da coluna
            st.plotly_chart(fig_category_earnings, use_container_width=True)


    # Exibe tabela detalhada dos ganhos no período selecionado
    if not earnings_df.empty:
        # Cria cópia para não alterar o DataFrame original
        earnings_display = earnings_df.copy()

        # Formata a coluna de data para exibir apenas ano-mês-dia
        if "Date" in earnings_display.columns:
            earnings_display["Date"] = earnings_display["Date"].dt.date

        # Cria painel recolhível com a tabela de ganhos
        with st.expander("Earnings in period", expanded=False):
            st.dataframe(
                # Exibe apenas as colunas principais, se existirem no DataFrame
                earnings_display[
                    [
                        col
                        for col in [
                            "Date",
                            "Description",
                            "Sub-description",
                            "Amount",
                            "Category",
                            "Sub-Category",
                            "Account",
                        ]
                        if col in earnings_display.columns
                    ]
                ],
                use_container_width=True,  # Ajusta tabela à largura total
            )
    # Caso não existam ganhos, exibe aviso
    else:
        st.info("No earnings in the selected range.")


    # Gráfico de série temporal mostrando a evolução do saldo (patrimônio)
    trend_df = daily_trends(filtered_df)

    # Verifica se há dados de saldo diário disponíveis
    if trend_df.empty:
        st.info("No time-series data available for the selected range.")
    else:
        # Cria gráfico de linha com Plotly (saldo ao longo do tempo)
        trend_fig = px.line(
            trend_df,
            x="Date",
            y="Balance",
            title="Account balance trend",
            labels={"Balance": "Balance ($)"},  # Rótulo do eixo Y
        )

        # Exibe o gráfico ajustado à largura da tela
        st.plotly_chart(trend_fig, use_container_width=True)


    # Exibe tabela com os 10 maiores gastos no período selecionado
    st.subheader("Top 10 largest expenses in the period")

    # Gera DataFrame com as 10 maiores despesas (filtradas por categoria, se selecionada)
    top10_df = top10_expenses(filtered_df, selected_category)

    # Caso não haja dados, exibe aviso
    if top10_df.empty:
        st.info("No expenses match the current filters.")
    else:
        # Define as colunas a exibir
        display_columns = [
            "Date",
            "Description",
            "Sub-description",
            "Amount",
            "Category",
            "Sub-Category",
        ]
        # Garante que apenas colunas existentes sejam mostradas
        available_columns = [c for c in display_columns if c in top10_df.columns]

        # Cria cópia e formata a data para exibição
        top10_display = top10_df.copy()
        if "Date" in top10_display.columns:
            top10_display["Date"] = top10_display["Date"].dt.date

        # Exibe a tabela formatada
        st.dataframe(top10_display[available_columns], use_container_width=True)

    # Seção para explorar subcategorias dentro de uma categoria específica
    st.subheader("Sub-category distribution by category")

    # Menu suspenso para o usuário escolher a categoria
    subcat_select_options = ["Select a category"] + category_options
    selected_category_detail = st.selectbox(
        "Choose a category to explore its sub-categories",
        options=subcat_select_options,
        index=0,
        key="subcat_detail_category",
    )

    # Executa apenas se uma categoria for selecionada
    if selected_category_detail != "Select a category":
        subcat_df = subcategory_distribution(filtered_df, selected_category_detail)

        # Caso não haja subcategorias registradas
        if subcat_df.empty:
            st.info("No sub-category data available for the selected category.")
        else:
            # Gráfico de pizza com a distribuição por subcategoria
            fig_subcat = px.pie(
                subcat_df,
                names="Sub-Category",
                values="Total",
                title=f"Sub-categories for {selected_category_detail}",
                color_discrete_sequence=px.colors.sequential.Sunset,
            )
            style_pie_with_values(fig_subcat)
            st.plotly_chart(fig_subcat, use_container_width=True)

            # Métricas e tabelas detalhadas da categoria selecionada
            category_rows = filtered_df[filtered_df["Category"] == selected_category_detail].copy()
            if category_rows.empty:
                st.info("No transactions recorded for this category within the current filters.")
            else:
                # Métricas de movimentação (total, mediana, média)
                movement_values = category_rows["Amount"].abs()
                movement_cols = st.columns(3)
                movement_cols[0].metric("Total movement", format_currency(float(movement_values.sum())))
                movement_cols[1].metric("Median movement", format_currency(float(movement_values.median())))
                movement_cols[2].metric("Average movement", format_currency(float(movement_values.mean())))

                # Separa despesas e ganhos dentro da categoria
                expenses_rows = category_rows[category_rows["Amount"] < 0].copy()
                earnings_rows = category_rows[category_rows["Amount"] > 0].copy()

                # Exibe tabela de despesas
                if not expenses_rows.empty:
                    expenses_display = expenses_rows.copy()
                    expenses_display["Amount"] = expenses_display["Amount"].abs()
                    expenses_display["Date"] = expenses_display["Date"].dt.date
                    st.markdown("#### Expense transactions")
                    st.dataframe(expenses_display[[...]], use_container_width=True)

                # Exibe tabela de ganhos
                if not earnings_rows.empty:
                    earnings_display = earnings_rows.copy()
                    earnings_display["Date"] = earnings_display["Date"].dt.date
                    st.markdown("#### Earning transactions")
                    st.dataframe(earnings_display[[...]], use_container_width=True)

                # Caso não haja nenhuma transação na categoria
                if expenses_rows.empty and earnings_rows.empty:
                    st.info("No transactions recorded for this category within the current filters.")

    # === Histórico completo de transações e download do dataset ===

    # Define as colunas padrão do histórico
    history_columns = [
        "Date", "Description", "Sub-description",
        "Amount", "Class", "Category", "Sub-Category", "Account",
    ]
    available_history_cols = [c for c in history_columns if c in display_df.columns]

    # Exibe todas as transações ou mensagem se não houver dados
    if display_df.empty:
        st.info("No transactions available to display.")
    else:
        history_df = display_df.copy()
        if "Date" in history_df.columns:
            history_df["Date"] = history_df["Date"].dt.date

        # Painel recolhível com o histórico completo
        st.subheader("Full transaction history")
        with st.expander("Show / hide full transaction history", expanded=False):
            st.dataframe(history_df[available_history_cols], use_container_width=True)

    # === Transações classificadas como "Others" ===
    st.subheader("Transactions categorized as 'Others'")
    others_history = display_df[display_df["Category"] == "Others"].copy()
    if others_history.empty:
        st.info("No transactions are currently categorized as 'Others'.")
    else:
        if "Date" in others_history.columns:
            others_history["Date"] = others_history["Date"].dt.date
        st.dataframe(others_history[available_history_cols], use_container_width=True)

    # Exibe quantidade de transações mostradas versus total
    st.caption(f"{len(filtered_df):,} transactions shown out of {len(display_df):,} available.")

    # === Botão para download do CSV classificado ===
    #csv_download = bank_df_classified.to_csv(index=False).encode("utf-8")
    #st.download_button(
    #    label="Download classified CSV",
    #    data=csv_download,
    #    file_name="classified_transactions.csv",
    #    mime="text/csv",
    #)



if __name__ == "__main__":
    main()




