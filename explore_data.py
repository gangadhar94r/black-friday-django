import pandas as pd
import glob

# Find any CSV file in the current folder
csv_files = glob.glob("*.csv")
csv_file = csv_files[0]
df = pd.read_csv(csv_file)

# === SECTION 1: BASIC INFO ===
print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# === SECTION 2: FIRST 5 ROWS ===
print("`n" + "=" * 60)
print("2. FIRST 5 ROWS (what does a customer look like?)")
print("=" * 60)
print(df.head())

# === SECTION 3: DATA TYPES ===
print("`n" + "=" * 60)
print("3. DATA TYPES (number vs text)")
print("=" * 60)
print(df.dtypes)

# === SECTION 4: MISSING VALUES ===
print("`n" + "=" * 60)
print("4. MISSING VALUES (any blank cells?)")
print("=" * 60)
missing = df.isnull().sum()
for col, count in missing.items():
    if count > 0:
        pct = (count / len(df)) * 100
        print(f"  {col}: {count:,} missing ({pct:.1f}%)")
    else:
        print(f"  {col}: complete")

# === SECTION 5: PURCHASE STATISTICS ===
print("`n" + "=" * 60)
print("5. PURCHASE STATISTICS (our target)")
print("=" * 60)
print(f"Minimum purchase: ${df['Purchase'].min():,}")
print(f"Maximum purchase: ${df['Purchase'].max():,}")
print(f"Average purchase: ${df['Purchase'].mean():,.2f}")
print(f"Median purchase:  ${df['Purchase'].median():,.2f}")
