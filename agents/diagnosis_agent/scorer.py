import json
import ollama
from google import genai
from agents.diagnosis_agent.schemas import DimensionResult
from config.settings import GEMINI_API_KEY, OLLAMA_MODEL, GEMINI_MODEL

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def _try_parse(raw_text: str) -> dict:
    # strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)

def score_with_ollama(prompt: str) -> dict:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return _try_parse(response["message"]["content"])

def score_with_gemini(prompt: str) -> dict:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return _try_parse(response.text)

def score_dimension(prompt: str) -> DimensionResult:
    """Tier 1 (Ollama) first, escalate to Tier 2 (Gemini) on failure."""
    try:
        raw = score_with_ollama(prompt)
        return DimensionResult(**raw)
    except Exception as e:
        print(f"Tier 1 (Ollama) failed or produced invalid schema: {e} — escalating to Gemini")
        raw = score_with_gemini(prompt)
        return DimensionResult(**raw)