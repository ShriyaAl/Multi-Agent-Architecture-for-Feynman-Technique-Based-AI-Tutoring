import pandas as pd
from agents.diagnosis_agent.agent import DiagnosisAgent

GOLDEN_SET_PATH = "evaluation/golden_set.csv"
DIMENSIONS = ["completeness", "accuracy", "clarity", "coherence"]

def score_to_label(score: int) -> str:
    # collapse to binary "has issue" vs "no issue" for precision/recall
    return "issue" if score is not None and score < 2 else "no_issue"

def main():
    df = pd.read_csv(GOLDEN_SET_PATH)
    agent = DiagnosisAgent()

    results = []
    for idx, row in df.iterrows():
        diagnosis = agent.diagnose(
            question=row["question"],
            student_explanation=row["explanation_text"],
            reference_answer=row.get("reference_answer", ""),
        )
        for dim in DIMENSIONS:
            predicted_score = getattr(diagnosis, dim).score
            true_score_col = f"{dim}_score"
            true_score = row.get(true_score_col)
            results.append({
                "row_id": idx,
                "dimension": dim,
                "predicted_score": predicted_score,
                "true_score": true_score,
                "predicted_label": score_to_label(predicted_score),
                "true_label": score_to_label(true_score) if pd.notna(true_score) else None,
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=["true_label"])

    print("\n=== Per-dimension accuracy ===")
    for dim in DIMENSIONS:
        subset = results_df[results_df["dimension"] == dim]
        if len(subset) == 0:
            continue
        correct = (subset["predicted_label"] == subset["true_label"]).sum()
        total = len(subset)
        print(f"{dim}: {correct}/{total} = {correct/total:.2f}")

    results_df.to_csv("evaluation/eval_results.csv", index=False)
    print("\nFull results saved to evaluation/eval_results.csv")

if __name__ == "__main__":
    main()