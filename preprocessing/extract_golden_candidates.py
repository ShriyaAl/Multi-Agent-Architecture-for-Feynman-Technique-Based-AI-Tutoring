import pandas as pd
from pathlib import Path

# --- Config ---
SOCRATEACH_PATH = "preprocessing/output/taxonomy_final_50.csv"
MATHDIAL_PATH = "preprocessing/output/taxonomy_misconception_50.csv"
TAXONOMY_PATH = "domain/misconception_taxonomy/taxonomy.yaml"  # for reference while labeling
OUTPUT_PATH = "golden_set/output/golden_set_candidates.csv"

# How many diagnosable rows to pull per source, stratified roughly evenly
TARGET_DIAGNOSABLE_PER_SOURCE = 10
# How many clean/negative-control rows to include total (across both sources)
TARGET_NEGATIVE_CONTROLS = 5

RANDOM_SEED = 42


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
        "label_type": df["label_type"],
        "inferred_gap_draft": df["inferred_gap"],   # from Phase 1 labeling — becomes gap_tags reference
        "dimension_draft": df["dimension"],          # from Phase 1 labeling
    })


def load_mathdial(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "source_dataset": "mathdial",
        "problem_id": df["problem_id"],
        "dialogue_id": "",  # not applicable
        "user_type_or_profile": df.get("student_profile", ""),
        "question": df["question"],
        "reference_answer": df.get("ground_truth", ""),
        "explanation_text": df["student_said"],
        "label_type": df["label_type"],
        "inferred_gap_draft": df["inferred_gap"],
        "dimension_draft": df["dimension"],
    })


def main():
    frames = []
    if Path(SOCRATEACH_PATH).exists():
        frames.append(load_socrateach(SOCRATEACH_PATH))
    if Path(MATHDIAL_PATH).exists():
        frames.append(load_mathdial(MATHDIAL_PATH))

    if not frames:
        print("No input files found — check paths.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Split into diagnosable vs. everything else
    diagnosable = combined[combined["label_type"] == "diagnosable"]
    non_diagnosable = combined[combined["label_type"] != "diagnosable"]

    # Within diagnosable, separate rows that already have a dimension_draft
    # filled (i.e., you've already done Phase-1 labeling on them) from ones
    # that don't yet — prioritize already-labeled rows since they need less
    # re-work in Phase 2.
    already_labeled = diagnosable[diagnosable["dimension_draft"].notna() &
                                   (diagnosable["dimension_draft"].astype(str).str.strip() != "")]
    not_yet_labeled = diagnosable[~diagnosable.index.isin(already_labeled.index)]

    # Stratified sample per source, preferring already-labeled rows first
    selected_rows = []
    for source in combined["source_dataset"].unique():
        pool_labeled = already_labeled[already_labeled["source_dataset"] == source]
        pool_unlabeled = not_yet_labeled[not_yet_labeled["source_dataset"] == source]

        n_needed = TARGET_DIAGNOSABLE_PER_SOURCE
        take_labeled = pool_labeled.sample(n=min(n_needed, len(pool_labeled)), random_state=RANDOM_SEED)
        remaining = n_needed - len(take_labeled)
        take_unlabeled = pool_unlabeled.sample(n=min(remaining, len(pool_unlabeled)), random_state=RANDOM_SEED) if remaining > 0 else pd.DataFrame()

        selected_rows.append(pd.concat([take_labeled, take_unlabeled]))

    diagnosable_sample = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    # Negative controls: rows tagged diagnosable but with a blank/empty inferred_gap
    # (i.e., explanation looked fine, no real gap found) — if you don't have
    # explicit "clean" rows yet, this will just come up empty; note it.
    clean_pool = diagnosable[
        diagnosable["inferred_gap_draft"].isna() |
        (diagnosable["inferred_gap_draft"].astype(str).str.strip() == "")
    ]
    negative_controls = clean_pool.sample(
        n=min(TARGET_NEGATIVE_CONTROLS, len(clean_pool)), random_state=RANDOM_SEED
    ) if len(clean_pool) > 0 else pd.DataFrame()

    final = pd.concat([diagnosable_sample, negative_controls], ignore_index=True)

    if final.empty:
        print("WARNING: no candidates selected — check your input files have labeled rows.")
        return

    # Add blank Phase-2 labeling columns
    final["completeness_score"] = ""
    final["accuracy_score"] = ""
    final["clarity_score"] = ""
    final["coherence_score"] = ""
    final["gap_tags"] = ""   # to be filled with taxonomy.yaml node IDs, semicolon-separated
    final["notes"] = ""

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(final)} golden-set candidates to {OUTPUT_PATH}")
    print(final["source_dataset"].value_counts())
    print(f"  of which {len(negative_controls)} are negative-control candidates")

if __name__ == "__main__":
    main()