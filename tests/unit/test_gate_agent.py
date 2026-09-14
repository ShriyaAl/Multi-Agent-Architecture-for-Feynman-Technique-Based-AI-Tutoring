import pytest
from unittest.mock import patch
from agents.gate_agent.prefilter import cheap_prefilter
from agents.gate_agent.schemas import GateResult
from agents.gate_agent.agent import GateAgent


class TestCheapPrefilter:
    def test_none_input_rejected(self):
        result = cheap_prefilter(None)
        assert result.decision == "reject"
        assert result.category == "no_attempt"

    def test_ellipsis_rejected(self):
        result = cheap_prefilter(...)
        assert result.decision == "reject"

    def test_empty_string_rejected(self):
        result = cheap_prefilter("")
        assert result.decision == "reject"

    def test_short_string_rejected(self):
        result = cheap_prefilter("hi")
        assert result.decision == "reject"

    @pytest.mark.parametrize("phrase", ["idk", "IDK", "  idk  ", "n/a", "dunno", "?"])
    def test_known_non_answers_rejected(self, phrase):
        result = cheap_prefilter(phrase)
        assert result.decision == "reject"
        assert result.category == "no_attempt"

    def test_valid_text_passes_through(self):
        result = cheap_prefilter("The total ratio is 27, so each part is 7.")
        assert result is None  # needs real classification, not prefiltered


class TestGateAgent:
    @patch("agents.gate_agent.agent.score_with_ollama")
    def test_prefilter_short_circuits_llm(self, mock_ollama):
        agent = GateAgent()
        result = agent.check(student_text="idk")
        mock_ollama.assert_not_called()
        assert result.decision == "reject"

    @patch("agents.gate_agent.agent.score_with_ollama")
    def test_valid_input_calls_llm(self, mock_ollama):
        mock_ollama.return_value = {
            "decision": "pass",
            "category": "diagnosable",
            "reasoning": "Real attempt at reasoning.",
        }
        agent = GateAgent()
        result = agent.check(student_text="70 minus 10 is 60", question="How many dogs remain?")
        mock_ollama.assert_called_once()
        assert result.decision == "pass"
        assert result.category == "diagnosable"

    @patch("agents.gate_agent.agent.score_with_gemini")
    @patch("agents.gate_agent.agent.score_with_ollama")
    def test_escalates_to_gemini_on_ollama_failure(self, mock_ollama, mock_gemini):
        mock_ollama.side_effect = Exception("malformed JSON")
        mock_gemini.return_value = {
            "decision": "reject",
            "category": "off_topic",
            "reasoning": "Unrelated to the problem.",
        }
        agent = GateAgent()
        result = agent.check(student_text="I like pizza", question="How many dogs remain?")
        mock_gemini.assert_called_once()
        assert result.category == "off_topic"