import warnings
from domain.misconception_taxonomy.loader import (
    load_taxonomy, get_nodes_by_dimension, format_nodes_for_prompt
)
from agents.diagnosis_agent.prompts import build_dimension_prompt
from agents.diagnosis_agent.scorer import score_dimension
from agents.diagnosis_agent.schemas import DiagnosisResult, DimensionResult

DIMENSIONS = ["accuracy", "completeness", "clarity", "coherence"]

# Known non-answer strings that should never be sent to the LLM.
_NON_ANSWERS = {"...", "idk", "i don't know", "n/a", "na", "?", "idk.", "dunno"}

_MIN_EXPLANATION_LEN = 5  # chars, after stripping whitespace


def is_diagnosable(explanation) -> bool:
    """Return True only if *explanation* is a real, scoreable student response.

    Checks (in order):
    1. Must be a str (rejects Ellipsis, None, int, …).
    2. Must be at least _MIN_EXPLANATION_LEN characters after stripping.
    3. Must not be a known non-answer placeholder (case-insensitive).
    """
    if not isinstance(explanation, str):
        return False
    stripped = explanation.strip()
    if len(stripped) < _MIN_EXPLANATION_LEN:
        return False
    if stripped.lower() in _NON_ANSWERS:
        return False
    return True


def filter_gap_ids(result: DimensionResult, valid_ids: set) -> DimensionResult:
    """Return a copy of *result* with matched_gap_ids restricted to *valid_ids*.

    Any ID returned by the LLM that is not in the taxonomy nodes passed to
    that dimension's prompt is silently dropped and a warning is emitted.
    This is applied to every dimension on every call as defense-in-depth
    against hallucinated taxonomy IDs.
    """
    if not result.matched_gap_ids:
        return result

    hallucinated = [gid for gid in result.matched_gap_ids if gid not in valid_ids]
    if hallucinated:
        warnings.warn(
            f"Dropping hallucinated gap IDs not present in taxonomy: {hallucinated}",
            stacklevel=2,
        )

    clean_ids = [gid for gid in result.matched_gap_ids if gid in valid_ids]
    return DimensionResult(
        score=result.score,
        matched_gap_ids=clean_ids,
        reasoning=result.reasoning,
    )


class DiagnosisAgent:
    def __init__(self, taxonomy_path: str = "domain/misconception_taxonomy/taxonomy.yaml"):
        self.taxonomy = load_taxonomy(taxonomy_path)

    def diagnose(self, question: str, student_explanation: str, reference_answer: str) -> DiagnosisResult:
        if not is_diagnosable(student_explanation):
            raise ValueError(
                f"student_explanation is not diagnosable: received {student_explanation!r}. "
                f"It must be a non-empty string of at least {_MIN_EXPLANATION_LEN} characters "
                f"and must not be a known non-answer placeholder."
            )

        results = {}
        for dimension in DIMENSIONS:
            nodes = get_nodes_by_dimension(self.taxonomy, dimension)
            valid_ids = {n["id"] for n in nodes}
            nodes_text = format_nodes_for_prompt(nodes)
            prompt = build_dimension_prompt(
                dimension=dimension,
                question=question,
                student_explanation=student_explanation,
                reference_answer=reference_answer,
                taxonomy_nodes_text=nodes_text,
            )
            raw_result = score_dimension(prompt)
            results[dimension] = filter_gap_ids(raw_result, valid_ids)

        return DiagnosisResult(**results)