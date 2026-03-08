import pandas as pd

# Load the CSV file
df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")

# ── Shape ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# ── Column Names ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>3}. {col}")

# ── Data Types ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA TYPES & NULL COUNTS")
print("=" * 60)
dtype_info = pd.DataFrame({
    "dtype"    : df.dtypes,
    "non_null" : df.notnull().sum(),
    "null"     : df.isnull().sum(),
    "null_%"   : (df.isnull().mean() * 100).round(2)
})
print(dtype_info.to_string())

# ── Sample Rows ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE ROWS (head 5)")
print("=" * 60)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.head(5).to_string())

# ── Summary Statistics ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY STATISTICS (numeric columns)")
print("=" * 60)
print(df.describe().to_string())

print("\n" + "=" * 60)
print("SUMMARY STATISTICS (object / categorical columns)")
print("=" * 60)
obj_cols = df.select_dtypes(include="object").columns
if len(obj_cols):
    print(df[obj_cols].describe().to_string())
else:
    print("No object columns found.")

print("\nData loading complete ✓")
