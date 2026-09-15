import pandas as pd
from pathlib import Path

SOCRATEACH_PATH = "preprocessing/output/taxonomy_final_50.csv"
MATHDIAL_PATH = "preprocessing/output/taxonomy_misconception_50.csv"
OUTPUT_PATH = "evaluation/gate_golden_set.csv"

TARGET_PER_CATEGORY = 10
RANDOM_SEED = 42

def normalize(series):
    return series.astype(str).str.strip().str.lower()

def main():
    frames = []
    for path, source in [(SOCRATEACH_PATH, "socrateach"), (MATHDIAL_PATH, "mathdial")]:
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        df["source_dataset"] = source
        df["label_type"] = normalize(df["label_type"])
        frames.append(df[["source_dataset", "question", "student_said", "label_type"]])

    if not frames:
        print("No input files found.")
        return

    combined = pd.concat(frames, ignore_index=True)

    samples = []
    for category in ["diagnosable", "no_attempt", "off_topic", "student_question"]:
        pool = combined[combined["label_type"] == category]
        take = pool.sample(n=min(TARGET_PER_CATEGORY, len(pool)), random_state=RANDOM_SEED)
        samples.append(take)
        print(f"{category}: {len(pool)} available, {len(take)} sampled")

    final = pd.concat(samples, ignore_index=True)
    final = final.rename(columns={"student_said": "explanation_text", "label_type": "true_label"})
    final["notes"] = ""

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(final)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()