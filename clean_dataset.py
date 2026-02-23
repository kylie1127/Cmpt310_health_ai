# usage: python3 {filepath}
# eg: python3 data/data.csv

import sys
from typing import List

import numpy as np
import pandas as pd

MISSING_TOKENS = [
    "",
    "na",
    "n/a",
    "null",
    "none",
    "nan",
    "nil",
    "unknown",
    "-",
    "?",
]


def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace(MISSING_TOKENS, np.nan, inplace=True)
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].replace(r"^\s*$", np.nan, regex=True)
    return df


def main() -> int:
    input_path = sys.argv[1]

    df = pd.read_csv(input_path)

    df = normalize_missing(df)
    missing_rows = df.isna().any(axis=1).sum()
    dupe_rows = df.duplicated().sum()
    print(f"Missing rows: {missing_rows}")
    print(f"Duplicate rows: {dupe_rows}")

    df = df.dropna(how="any")
    df = df.drop_duplicates()

    output_path = "data/cleaned_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned dataset to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
