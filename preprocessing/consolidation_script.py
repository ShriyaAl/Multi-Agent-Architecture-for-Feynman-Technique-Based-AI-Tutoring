import pandas as pd

SOCRATEACH_PATH = "preprocessing/output/taxonomy_draft.csv"
MATHDIAL_PATH = "preprocessing/output/mathdial_draft.csv"
OUTPUT_PATH = "preprocessing/output/taxonomy_consolidation_helper.csv"

def main():
    frames = []
    for path, source in [(SOCRATEACH_PATH, "socrateach"), (MATHDIAL_PATH, "mathdial")]:
        df = pd.read_csv(path)
        
        labeled = df[
            (df.get("label_type") == "diagnosable") &
            df["inferred_gap"].notna() &
            (df["inferred_gap"].astype(str).str.strip() != "") &
            df["dimension"].notna() &
            (df["dimension"].astype(str).str.strip() != "")
        ].copy()
        labeled["source_dataset"] = source
        frames.append(labeled[["source_dataset", "inferred_gap", "dimension"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("dimension")  # group by dimension for easy scanning
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(combined)} labeled gap entries, sorted by dimension, to {OUTPUT_PATH}")
    print(combined["dimension"].value_counts())

if __name__ == "__main__":
    main()