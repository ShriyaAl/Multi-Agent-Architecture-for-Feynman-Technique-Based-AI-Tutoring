import pandas as pd
from pathlib import Path

# --- Config ---
SOCRATEACH_PATH = "preprocessing/output/taxonomy_final_50.csv"
MATHDIAL_PATH = "preprocessing/output/taxonomy_misconception_50.csv"
OUTPUT_PATH = "output/golden_set_candidates.csv"

TARGET_DIAGNOSABLE_PER_SOURCE = 10
TARGET_NEGATIVE_CONTROLS = 5
RANDOM_SEED = 42


def normalize(series):
    return series.astype(str).str.strip().str.lower()


def load_socrateach(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "source_dataset": "socrateach",
        "problem_id": df["problem_id"],
        "dialogue_id": df.get("dialogue_id", ""),
        "user_type_or_profile": df.get("user_type", ""),
        "question": df["question"],
        "reference_answer": df.get("answer", ""),
        "explanation_text": df["student_said"],
        "label_type": normalize(df["label_type"]),
        "inferred_gap_draft": df["inferred_gap"],
        "dimension_draft": normalize(df["dimension"]),
    })


def load_mathdial(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "source_dataset": "mathdial",
        "problem_id": df["problem_id"],
        "dialogue_id": "",
        "user_type_or_profile": df.get("student_profile", ""),
        "question": df["question"],
        "reference_answer": df.get("ground_truth", ""),
        "explanation_text": df["student_said"],
        "label_type": normalize(df["label_type"]),
        "inferred_gap_draft": df["inferred_gap"],
        "dimension_draft": normalize(df["dimension"]),
    })


def main():
    frames = []
    if Path(SOCRATEACH_PATH).exists():
        frames.append(load_socrateach(SOCRATEACH_PATH))
    if Path(MATHDIAL_PATH).exists():
        frames.append(load_mathdial(MATHDIAL_PATH))

    combined = pd.concat(frames, ignore_index=True)

    diagnosable = combined[combined["label_type"] == "diagnosable"]

    already_labeled = diagnosable[
        diagnosable["dimension_draft"].notna() &
        (diagnosable["dimension_draft"].astype(str).str.strip() != "") &
        (diagnosable["dimension_draft"] != "nan")
    ]
    not_yet_labeled = diagnosable[~diagnosable.index.isin(already_labeled.index)]

    selected_rows = []
    for source in combined["source_dataset"].unique():
        pool_labeled = already_labeled[already_labeled["source_dataset"] == source]
        pool_unlabeled = not_yet_labeled[not_yet_labeled["source_dataset"] == source]

        n_needed = TARGET_DIAGNOSABLE_PER_SOURCE
        take_labeled = pool_labeled.sample(n=min(n_needed, len(pool_labeled)), random_state=RANDOM_SEED)
        remaining = n_needed - len(take_labeled)
        take_unlabeled = (
            pool_unlabeled.sample(n=min(remaining, len(pool_unlabeled)), random_state=RANDOM_SEED)
            if remaining > 0 else pd.DataFrame()
        )
        selected_rows.append(pd.concat([take_labeled, take_unlabeled]))

    diagnosable_sample = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    clean_pool = diagnosable[
        diagnosable["inferred_gap_draft"].isna() |
        (diagnosable["inferred_gap_draft"].astype(str).str.strip() == "")
    ]
    negative_controls = (
        clean_pool.sample(n=min(TARGET_NEGATIVE_CONTROLS, len(clean_pool)), random_state=RANDOM_SEED)
        if len(clean_pool) > 0 else pd.DataFrame()
    )

    final = pd.concat([diagnosable_sample, negative_controls], ignore_index=True)

    final["completeness_score"] = ""
    final["accuracy_score"] = ""
    final["clarity_score"] = ""
    final["coherence_score"] = ""
    final["gap_tags"] = ""
    final["notes"] = ""

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(final)} golden-set candidates to {OUTPUT_PATH}")
    print(final["source_dataset"].value_counts())
    print(f"  of which {len(negative_controls)} are negative-control candidates")


if __name__ == "__main__":
    main()