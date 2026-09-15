DIMENSION_DEFINITIONS = {
    "completeness": "Something the explanation needed is simply absent — not stated at all.",
    "accuracy": "Something the explanation states is factually or procedurally wrong.",
    "clarity": "The content may be correct, but it's stated ambiguously — unclear referents or vague wording.",
    "coherence": "Individual stated facts are correct and clear, but don't logically connect into a reasoning chain.",
}

DIMENSION_ORDER_NOTE = (
    "Apply dimensions in this priority: check accuracy first, then completeness, "
    "then clarity, then coherence. Only flag a dimension if fixing that one issue "
    "alone — leaving everything else as-is — would resolve it. Score 2 if there is "
    "no issue on this dimension."
)

def build_dimension_prompt(dimension: str, question: str, student_explanation: str,
                            reference_answer: str, taxonomy_nodes_text: str) -> str:
    if taxonomy_nodes_text.strip():
        gap_section = (
            "Known misconception patterns for this dimension (use these IDs if the student's\n"
            f"explanation matches one; otherwise leave matched_gap_ids empty):\n{taxonomy_nodes_text}"
        )
    else:
        gap_section = (
            "No known misconception patterns are defined for this dimension yet. "
            "You MUST return matched_gap_ids as an empty list []."
        )

    return f"""You are evaluating a middle-school student's explanation of a math problem,
specifically for the "{dimension}" dimension.

Definition: {DIMENSION_DEFINITIONS[dimension]}

{DIMENSION_ORDER_NOTE}

Problem: {question}
Reference answer/reasoning: {reference_answer}

Student's explanation: {student_explanation}

{gap_section}

Respond in this exact JSON structure:
{{
  "score": <0, 1, or 2>,
  "matched_gap_ids": [<zero or more IDs from the list above>],
  "reasoning": "<one sentence justifying the score>"
}}
"""