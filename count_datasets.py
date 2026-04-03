import pandas as pd
import glob
import os

with open('dataset_counts.txt', 'w') as f:
    for file in glob.glob('datasets/**/*.csv', recursive=True):
        try:
            df = pd.read_csv(file, low_memory=False)
            f.write(f"{file}: Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
        except Exception as e:
            f.write(f"{file}: Error: {e}\n")
