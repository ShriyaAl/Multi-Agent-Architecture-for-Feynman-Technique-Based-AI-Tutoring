def build_gate_prompt(question: str, student_text: str) -> str:
    return f"""Classify this student response to a math problem into exactly one category:

- diagnosable: a genuine attempt at explaining reasoning, even if wrong or incomplete
- no_attempt: blank, "I don't know", or just repeats the question with no reasoning
- off_topic: answers a different question, or is nonsensical/unrelated to the problem
- student_question: the student asks a clarifying question instead of attempting an answer

Problem: {question if question else "(not provided)"}
Student response: {student_text}

Respond in this exact JSON structure, with no extra text:
{{
  "decision": "pass" or "reject",
  "category": "<one of the four categories above>",
  "reasoning": "<one sentence>"
}}

Rule: decision must be "pass" if and only if category is "diagnosable". All other categories must have decision "reject".
"""