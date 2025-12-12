import pandas as pd
import glob
import os

REQUIRED_COLS = {
    "subject_id", "set_id", "rep_id", "frame", "obj_id",
    "x", "y", "z", "v", "range_m", "azimuth_deg", "snr",
    "noise", "label"
}

def merge_all_csvs(input_pattern="*.csv", output="master_dataset.csv"):
    all_files = glob.glob(input_pattern)
    dfs = []

    for f in all_files:
        print(f"Reading {f}...")
        df = pd.read_csv(f)

        # Drop fully empty columns
        df = df.dropna(axis=1, how='all')

        # Normalize column names
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        # Validate expected columns
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"⚠ WARNING: {f} missing columns {missing}. Skipping file.")
            continue

        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)

    # Drop any stray blank/header rows
    master = master.dropna(subset=["subject_id"])

    # Drop any fully-empty columns again after merge
    master = master.dropna(axis=1, how='all')

    master.to_csv(output, index=False)

    print(f"\n✅ Saved merged dataset to {output}")
    print(f"Total rows: {len(master)}")
    print(f"Unique subjects: {master['subject_id'].unique()}")
    print(f"Label distribution:\n{master['label'].value_counts()}")

    return master


if __name__ == "__main__":
    merge_all_csvs()
