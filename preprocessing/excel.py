import pandas as pd

# csv_path = "preprocessing/output/taxonomy_draft_final.csv"
# excel_path = "preprocessing/output/taxonomy_draft_final.xlsx"

# csv_path = "data/train_failures.csv"
# excel_path = "data/train_failures.xlsx"

csv_path = "preprocessing/output/mathdial_draft.csv"
excel_path = "preprocessing/output/taxonomy_misconception_50.xlsx"

df = pd.read_csv(csv_path)
df.to_excel(excel_path, index=False, engine="openpyxl")

print(f"Excel file saved to: {excel_path}")
