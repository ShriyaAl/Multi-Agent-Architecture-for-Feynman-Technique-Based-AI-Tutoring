import pandas as pd

# Excel file path
excel_file = "preprocessing/output/taxonomy_misconception_50.xlsx"

# CSV output path
csv_file = "preprocessing/output/taxonomy_misconception_50.csv"

# Read Excel file
df = pd.read_excel(excel_file)

# Save as CSV
df.to_csv(csv_file, index=False)

print(f"Converted successfully: {csv_file}")
