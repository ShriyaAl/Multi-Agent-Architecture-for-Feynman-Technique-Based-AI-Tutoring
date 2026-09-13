import pandas as pd
import yaml
from pathlib import Path

FINAL_NODES_PATH = "preprocessing/golden_set/taxonomy_final_nodes.csv"
OUTPUT_PATH = "domain/misconception_taxonomy/taxonomy.yaml"

def main():
    df = pd.read_csv(FINAL_NODES_PATH)

    # basic sanity checks before writing
    required_cols = {"id", "description", "dimension"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    valid_dimensions = {"completeness", "accuracy", "clarity", "coherence"}
    bad_dims = set(df["dimension"].str.strip().str.lower()) - valid_dimensions
    if bad_dims:
        raise ValueError(f"Found invalid dimension values: {bad_dims}")

    if df["id"].duplicated().any():
        dupes = df[df["id"].duplicated()]["id"].tolist()
        raise ValueError(f"Duplicate node ids found: {dupes}")

    nodes = df.to_dict(orient="records")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(nodes, f, sort_keys=False, allow_unicode=True)

    print(f"Wrote {len(nodes)} nodes to {OUTPUT_PATH}")
    print(df["dimension"].value_counts())

if __name__ == "__main__":
    main()