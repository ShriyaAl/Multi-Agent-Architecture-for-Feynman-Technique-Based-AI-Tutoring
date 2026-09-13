import pytest
from unittest.mock import patch
from agents.diagnosis_agent.agent import DiagnosisAgent

MOCK_DIMENSION_RESULT = {
    "score": 1,
    "matched_gap_ids": ["needs_next_step"],
    "reasoning": "Student stopped before the final step.",
}

@patch("agents.diagnosis_agent.agent.score_dimension")
def test_diagnose_returns_all_dimensions(mock_score):
    mock_score.return_value.__class__ = dict  # simplified; adjust to actual DimensionResult mock
    from agents.diagnosis_agent.schemas import DimensionResult
    mock_score.return_value = DimensionResult(**MOCK_DIMENSION_RESULT)

    agent = DiagnosisAgent()
    result = agent.diagnose(
        question="Test question",
        student_explanation="Test explanation",
        reference_answer="Test answer",
    )

    assert result.completeness.score == 1
    assert result.accuracy.score == 1
    assert "needs_next_step" in result.completeness.matched_gap_ids