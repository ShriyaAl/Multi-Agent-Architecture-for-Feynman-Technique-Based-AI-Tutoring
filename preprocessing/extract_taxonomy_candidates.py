import json
import csv
from pathlib import Path

# --- Config: adjust per topic ---
TOPIC_KEYWORDS = ["fractions", "denominator", "numerator"]  # example, adjust to your topic
INPUT_PATH = "./data/SocraTeach_multi.json"
OUTPUT_PATH = "preprocessing/output/taxonomy_draft_fractions.csv"

def matches_topic(question: str, analysis: str) -> bool:
    text = (question + " " + analysis).lower()
    return any(kw in text for kw in TOPIC_KEYWORDS)

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for problem_id, problem_data in data.items():
        question = problem_data.get("question", "")
        analysis = problem_data.get("analysis", "")
        answer = problem_data.get("answer", "")
        steps = problem_data.get("steps", [])

        if not matches_topic(question, analysis):
            continue

        for i, step_text in enumerate(steps):
            rows.append({
                "problem_id": problem_id,
                "question": question[:200],
                "analysis": analysis[:300],
                "answer": answer,
                "step_index": i,
                "step_text": step_text,
                "inferred_gap": "",     # fill by hand
                "dimension": "",        # fill by hand: completeness / accuracy / clarity / coherence
            })

    if not rows:
        print("No matching problems found — check TOPIC_KEYWORDS against real data.")
        return

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} step candidates from "
          f"{len(set(r['problem_id'] for r in rows))} problems to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()