import pandas as pd

# =========================
# LOAD RAW DATA
# =========================
df = pd.read_csv("datasets/raw/converted_data.csv")

print("Original shape:", df.shape)

# =========================
# SELECT IMPORTANT FEATURES
# =========================
selected_cols = [
    'age',
    'gender',
    'arrivalhour_bin',
    'triage_vital_hr',
    'triage_vital_sbp',
    'triage_vital_temp',
    'n_admissions'
]

df = df[selected_cols]

# =========================
# CREATE TARGET (BED AVAILABILITY)
# =========================
df['available_beds'] = 100 - df['n_admissions']

# DROP OLD COLUMN
df = df.drop(columns=['n_admissions'])

# =========================
# CLEAN
# =========================
df = df.dropna()

print("Processed shape:", df.shape)

# =========================
# SAVE
# =========================
df.to_csv("datasets/processed/bed_availability_processed.csv", index=False)

print("Saved processed dataset")