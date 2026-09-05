"""Pure safety checks for states produced by the vision pipeline."""
from __future__ import annotations

from typing import Any


CONFIDENCE_THRESHOLD = 0.7
CONFIDENCE_FIELDS = ("hole", "cards", "to_call", "position", "stage")


def confidence_block_reasons(
    state: dict[str, Any], *, threshold: float = CONFIDENCE_THRESHOLD
) -> list[str]:
    """Return deterministic reasons why a state must not reach poker engines.

    Missing confidence metadata is treated as legacy/unknown metadata, not a
    failed reading. Producers explicitly mark uncertain or estimated readings.
    """
    reasons: list[str] = []

    if state.get("is_uncertain"):
        reasons.append("state is uncertain")

    for reason in state.get("uncertainty_reasons") or []:
        if reason and reason not in reasons:
            reasons.append(str(reason))

    estimated_fields = state.get("estimated_fields") or []
    if state.get("to_call_source") == "estimated" or "to_call" in estimated_fields:
        reasons.append("to_call is estimated")

    confidence = state.get("confidence") or {}
    for field in CONFIDENCE_FIELDS:
        value = confidence.get(field)
        if value is not None and value < threshold:
            reasons.append(f"low confidence: {field}")

    return reasons
