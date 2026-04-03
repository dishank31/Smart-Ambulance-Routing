import pyreadr

print("🚀 Script started")

# Load file
result = pyreadr.read_r("datasets/raw/5v_cleandf.rdata")

print("📦 Objects inside file:", result.keys())

# Check if empty
if len(result.keys()) == 0:
    print("❌ No data found inside RData file")
else:
    df = list(result.values())[0]
    print("✅ Data loaded successfully")
    print(df.head())

    # ✅ Save in processed folder
    df.to_csv("datasets/processed/severity_processed.csv", index=False)

    print("💾 CSV saved successfully in processed folder!")
