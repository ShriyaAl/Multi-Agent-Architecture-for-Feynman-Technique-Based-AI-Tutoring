from domain.misconception_taxonomy.loader import (
    load_taxonomy, get_nodes_by_dimension, format_nodes_for_prompt
)
from agents.diagnosis_agent.prompts import build_dimension_prompt
from agents.diagnosis_agent.scorer import score_dimension
from agents.diagnosis_agent.schemas import DiagnosisResult

DIMENSIONS = ["accuracy", "completeness", "clarity", "coherence"]

class DiagnosisAgent:
    def __init__(self, taxonomy_path: str = "domain/misconception_taxonomy/taxonomy.yaml"):
        self.taxonomy = load_taxonomy(taxonomy_path)

    def diagnose(self, question: str, student_explanation: str, reference_answer: str) -> DiagnosisResult:
        results = {}
        for dimension in DIMENSIONS:
            nodes = get_nodes_by_dimension(self.taxonomy, dimension)
            nodes_text = format_nodes_for_prompt(nodes)
            prompt = build_dimension_prompt(
                dimension=dimension,
                question=question,
                student_explanation=student_explanation,
                reference_answer=reference_answer,
                taxonomy_nodes_text=nodes_text,
            )
            results[dimension] = score_dimension(prompt)

        return DiagnosisResult(**results)