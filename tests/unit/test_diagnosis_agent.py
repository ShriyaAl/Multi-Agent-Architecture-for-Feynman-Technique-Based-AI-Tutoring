import pytest
from unittest.mock import patch
from agents.diagnosis_agent.agent import DiagnosisAgent, is_diagnosable, filter_gap_ids
from agents.diagnosis_agent.schemas import DimensionResult


MOCK_DIMENSION_RESULT = {
    "score": 1,
    "matched_gap_ids": ["needs_next_step"],
    "reasoning": "Student stopped before the final step.",
}

class TestIsDiagnosable:
    def test_valid_explanation_passes(self):
        assert is_diagnosable("189 divided by 27 is 7, so 10 times 7 is 70 dogs.") is True

    def test_ellipsis_object_rejected(self):
        assert is_diagnosable(...) is False

    def test_none_rejected(self):
        assert is_diagnosable(None) is False

    def test_int_rejected(self):
        assert is_diagnosable(42) is False

    def test_empty_string_rejected(self):
        assert is_diagnosable("") is False

    def test_whitespace_only_rejected(self):
        assert is_diagnosable("    ") is False

    def test_too_short_rejected(self):
        # "ok" is 2 chars — below 5-char minimum
        assert is_diagnosable("ok") is False

    def test_exactly_min_length_passes(self):
        # Exactly 5 chars and not a known non-answer
        assert is_diagnosable("hello") is True

    @pytest.mark.parametrize("non_answer", [
        "...", "idk", "IDK", "I don't know", "n/a", "NA", "?", "idk.", "dunno",
    ])
    def test_known_non_answers_rejected(self, non_answer):
        assert is_diagnosable(non_answer) is False


# ---------------------------------------------------------------------------
# filter_gap_ids — unit tests
# ---------------------------------------------------------------------------

class TestFilterGapIds:
    def _make_result(self, gap_ids):
        return DimensionResult(
            score=1,
            matched_gap_ids=gap_ids,
            reasoning="Test reasoning.",
        )

    def test_valid_ids_kept(self):
        result = self._make_result(["needs_next_step", "concept_not_understood"])
        valid = {"needs_next_step", "concept_not_understood", "incomplete_concept"}
        filtered = filter_gap_ids(result, valid)
        assert filtered.matched_gap_ids == ["needs_next_step", "concept_not_understood"]

    def test_hallucinated_ids_dropped(self):
        result = self._make_result(["needs_next_step", "clarity_vague_referent"])
        valid = {"needs_next_step"}
        filtered = filter_gap_ids(result, valid)
        assert filtered.matched_gap_ids == ["needs_next_step"]

    def test_all_hallucinated_returns_empty(self):
        result = self._make_result(["made_up_id_1", "made_up_id_2"])
        valid = {"needs_next_step"}
        filtered = filter_gap_ids(result, valid)
        assert filtered.matched_gap_ids == []

    def test_empty_valid_ids_clears_all(self):
        """Dimension with 0 nodes: every returned ID must be dropped."""
        result = self._make_result(["hallucinated_clarity_gap"])
        filtered = filter_gap_ids(result, valid_ids=set())
        assert filtered.matched_gap_ids == []

    def test_empty_gap_ids_no_change(self):
        result = self._make_result([])
        filtered = filter_gap_ids(result, valid_ids={"needs_next_step"})
        assert filtered.matched_gap_ids == []

    def test_score_and_reasoning_preserved(self):
        result = self._make_result(["hallucinated"])
        valid = set()
        filtered = filter_gap_ids(result, valid)
        assert filtered.score == result.score
        assert filtered.reasoning == result.reasoning

    def test_hallucinated_ids_emit_warning(self):
        result = self._make_result(["fake_id"])
        with pytest.warns(UserWarning, match="fake_id"):
            filter_gap_ids(result, valid_ids=set())


# ---------------------------------------------------------------------------
# DiagnosisAgent.diagnose — integration-ish tests (LLM mocked)
# ---------------------------------------------------------------------------

class TestDiagnoseGuard:
    """Verify that bad inputs are rejected before any LLM call."""

    @patch("agents.diagnosis_agent.agent.score_dimension")
    def test_ellipsis_raises_value_error(self, mock_score):
        agent = DiagnosisAgent()
        with pytest.raises(ValueError, match="not diagnosable"):
            agent.diagnose(
                question="Some question",
                student_explanation=...,
                reference_answer="Some answer",
            )
        mock_score.assert_not_called()

    @patch("agents.diagnosis_agent.agent.score_dimension")
    def test_empty_string_raises_value_error(self, mock_score):
        agent = DiagnosisAgent()
        with pytest.raises(ValueError, match="not diagnosable"):
            agent.diagnose(
                question="Some question",
                student_explanation="",
                reference_answer="Some answer",
            )
        mock_score.assert_not_called()

    @patch("agents.diagnosis_agent.agent.score_dimension")
    def test_idk_raises_value_error(self, mock_score):
        agent = DiagnosisAgent()
        with pytest.raises(ValueError, match="not diagnosable"):
            agent.diagnose(
                question="Some question",
                student_explanation="idk",
                reference_answer="Some answer",
            )
        mock_score.assert_not_called()


class TestDiagnoseReturnsAllDimensions:
    """Verify normal path still works and filter is applied."""

    @patch("agents.diagnosis_agent.agent.score_dimension")
    def test_returns_all_four_dimensions(self, mock_score):
        mock_score.return_value = DimensionResult(**MOCK_DIMENSION_RESULT)

        agent = DiagnosisAgent()
        result = agent.diagnose(
            question="Test question",
            student_explanation="Test explanation that is long enough",
            reference_answer="Test answer",
        )

        assert result.completeness.score == 1
        assert result.accuracy.score == 1
        assert result.clarity.score == 1
        assert result.coherence.score == 1

    @patch("agents.diagnosis_agent.agent.score_dimension")
    def test_hallucinated_id_filtered_on_normal_call(self, mock_score):
        """Any ID the LLM returns that isn't in taxonomy nodes is dropped."""
        mock_score.return_value = DimensionResult(
            score=1,
            matched_gap_ids=["needs_next_step", "totally_made_up_gap"],
            reasoning="Test reasoning.",
        )
        agent = DiagnosisAgent()
        result = agent.diagnose(
            question="Test question",
            student_explanation="Test explanation that is long enough",
            reference_answer="Test answer",
        )
        # "totally_made_up_gap" is not in taxonomy.yaml for any dimension
        for dim_name in ["completeness", "accuracy", "clarity", "coherence"]:
            dim = getattr(result, dim_name)
            assert "totally_made_up_gap" not in dim.matched_gap_ids