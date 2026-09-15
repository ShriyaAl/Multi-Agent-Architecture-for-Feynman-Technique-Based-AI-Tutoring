from agents.gate_agent.schemas import GateResult

NON_ANSWERS = {"...", "idk", "i don't know", "n/a", "na", "?", "idk.", "dunno", ""}
MIN_LENGTH = 5

def cheap_prefilter(text) -> GateResult | None:
    """Returns a GateResult immediately if this is obviously rejectable,
    without calling the LLM. Returns None if it needs real classification."""
    if not isinstance(text, str):
        return GateResult(
            decision="reject",
            category="no_attempt",
            reasoning="Input is not a string (non-text value received).",
        )

    stripped = text.strip().lower()

    if len(stripped) < MIN_LENGTH:
        return GateResult(
            decision="reject",
            category="no_attempt",
            reasoning="Input is empty or too short to be a real attempt.",
        )

    if stripped in NON_ANSWERS:
        return GateResult(
            decision="reject",
            category="no_attempt",
            reasoning="Input matches a known non-answer phrase.",
        )

    return None