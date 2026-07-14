"""
load_data.py
============
Phase 1 — Step 1: get the raw Lending Club CSV into a clean,
fast-to-reload DataFrame. Deliberately "dumb": no leakage
filtering, no feature engineering, no target derivation here —
that all lives in clean_data.py. This script's only jobs are:

    1. Read the raw CSV robustly (mixed dtypes, ~151 columns)
    2. Parse issue_d (and other date columns) into real datetimes
    3. Cache the result as parquet so we never re-parse the raw
       CSV on every run (it's large and slow)
    4. Print a quick shape / dtype / missingness snapshot

Run directly:
    python src/load_data.py
"""

import pandas as pd

from utils import load_config, resolve_path, get_logger

logger = get_logger("load_data")

# Date columns in the Lending Club schema that use the "Mon-YY" format
# (e.g. "Dec-18"). Not all of these will be present/used downstream,
# but parsing them now avoids repeated ad-hoc parsing later.
DATE_COLUMNS_MON_YY = [
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d",
]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Read the raw loan.csv with safe dtype handling."""
    logger.info(f"Reading raw CSV from {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded raw shape: {df.shape}")
    return df


def parse_dates(df: pd.DataFrame, date_format: str = "%b-%y") -> pd.DataFrame:
    """Parse Lending Club's 'Mon-YY' date columns into datetime64."""
    for col in DATE_COLUMNS_MON_YY:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")
    return df


def snapshot(df: pd.DataFrame) -> None:
    """Quick console snapshot: shape, dtypes, missingness, target balance."""
    logger.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    logger.info("Top 15 columns by missingness (%):")
    print(missing_pct.head(15).round(2).to_string())

    if "loan_status" in df.columns:
        logger.info("loan_status value counts:")
        print(df["loan_status"].value_counts(dropna=False).to_string())

    if "issue_d" in df.columns:
        logger.info(
            f"issue_d range: {df['issue_d'].min()} -> {df['issue_d'].max()}"
        )


def main():
    config = load_config()
    raw_path = resolve_path(config["paths"]["raw_csv"])
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(str(raw_path))
    df = parse_dates(df, date_format=config["split"]["date_format"])
    snapshot(df)

    out_path = processed_dir / "loan_raw_parsed.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Cached parsed raw data to {out_path}")


if __name__ == "__main__":
    main()