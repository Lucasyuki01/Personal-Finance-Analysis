import hashlib
from typing import Any, Iterable


def stable_hash(values: Iterable[Any]) -> str:
    """Return a stable SHA-256 hash for the provided values."""
    normalized = ["" if v is None else str(v) for v in values]
    payload = "|".join(normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

