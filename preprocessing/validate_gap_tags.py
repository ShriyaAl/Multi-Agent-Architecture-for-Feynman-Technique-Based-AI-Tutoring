import pandas as pd
import yaml

import sys
from pathlib import Path

with open("domain/misconception_taxonomy/taxonomy.yaml") as f:
    taxonomy = yaml.safe_load(f)
valid_ids = {node["id"] for node in taxonomy}

golden_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/golden_set.csv"
golden = pd.read_csv(golden_path)

all_used_tags = set()
for tags in golden["gap_tags"].dropna():
    for t in str(tags).split(";"):
        t = t.strip()
        if t:
            all_used_tags.add(t)

unmatched = all_used_tags - valid_ids
print("Tags used in golden set but missing from taxonomy.yaml:", unmatched)
print("Taxonomy nodes never referenced by any golden-set row:", valid_ids - all_used_tags)