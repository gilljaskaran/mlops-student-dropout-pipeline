"""
Load the raw UCI dataset, clean it up, and split into train/test sets.

Usage: python src/prepare.py
Reads:  data/raw/data.csv, params.yaml (prepare:)
Writes: data/processed/train.csv, data/processed/test.csv, data/processed/label_map.json
"""
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/data.csv")
OUT_DIR = Path("data/processed")

# Dropout/Enrolled/Graduate -> 0/1/2. Fixed mapping (not alphabetical) so
# "worse outcome" gets a lower index -- easier to eyeball a confusion matrix.
LABEL_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """UCI's CSV has a stray tab in one header and a misspelling in another.
    Fix those without touching the raw file so the DVC-tracked source stays
    identical to what was downloaded."""
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns={"Nacionality": "Nationality"})
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # UCI says there are no missing values in this release, but we don't
    # want the pipeline to silently blow up if that ever changes (e.g. a
    # different export, or someone appends new rows later).
    n_missing = df.isna().sum().sum()
    if n_missing:
        print(f"[prepare] found {n_missing} missing values, imputing")
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        obj_cols = df.select_dtypes(exclude="number").columns
        for c in obj_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    else:
        print("[prepare] no missing values found")
    return df


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    df = pd.read_csv(RAW_PATH, sep=";", encoding="utf-8-sig")
    df = clean_columns(df)
    df = handle_missing(df)

    df["Target"] = df["Target"].map(LABEL_MAP)
    assert df["Target"].isna().sum() == 0, "unexpected label found in Target column"

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["Target"],  # keep the 50/32/18 class split roughly even in both sets
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"[prepare] train={len(train_df)} rows, test={len(test_df)} rows")


if __name__ == "__main__":
    main()
