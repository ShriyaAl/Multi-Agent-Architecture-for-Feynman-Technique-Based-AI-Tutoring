import pandas as pd
from agents.gate_agent.agent import GateAgent

GOLDEN_SET_PATH = "evaluation/gate_golden_set.csv"

def main():
    df = pd.read_csv(GOLDEN_SET_PATH)
    agent = GateAgent()

    results = []
    for _, row in df.iterrows():
        result = agent.check(student_text=row["explanation_text"], question=row.get("question", ""))
        results.append({
            "explanation_text": row["explanation_text"][:80],
            "true_label": row["true_label"],
            "predicted_category": result.category,
            "correct": result.category == row["true_label"],
        })

    results_df = pd.DataFrame(results)

    print("\n=== Overall accuracy ===")
    overall = results_df["correct"].mean()
    print(f"{results_df['correct'].sum()}/{len(results_df)} = {overall:.2f}")

    print("\n=== Per-category accuracy ===")
    for category in df["true_label"].unique():
        subset = results_df[results_df["true_label"] == category]
        acc = subset["correct"].mean()
        print(f"{category}: {subset['correct'].sum()}/{len(subset)} = {acc:.2f}")

    print("\n=== Misclassifications ===")
    misses = results_df[~results_df["correct"]]
    for _, row in misses.iterrows():
        print(f"true={row['true_label']}, predicted={row['predicted_category']}, text={row['explanation_text']}")

    results_df.to_csv("evaluation/gate_eval_results.csv", index=False)
    print("\nFull results saved to evaluation/gate_eval_results.csv")

if __name__ == "__main__":
    main()