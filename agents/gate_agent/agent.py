from agents.gate_agent.prefilter import cheap_prefilter
from agents.gate_agent.prompts import build_gate_prompt
from agents.gate_agent.schemas import GateResult
from agents.diagnosis_agent.scorer import score_with_ollama, score_with_gemini

class GateAgent:
    def check(self, student_text: str, question: str = "") -> GateResult:
        prefiltered = cheap_prefilter(student_text)
        if prefiltered is not None:
            return prefiltered

        prompt = build_gate_prompt(question, student_text)
        try:
            raw = score_with_ollama(prompt)
            return GateResult(**raw)
        except Exception as e:
            print(f"Gate: Tier 1 (Ollama) failed: {e} — escalating to Gemini")
            raw = score_with_gemini(prompt)
            return GateResult(**raw)