"""
clean_data.py
=============
Phase 1 — Step 2: leakage-safe cleaning.

Responsibilities:
    1. Derive the binary target from loan_status, drop unresolved loans
    2. Apply the leakage audit: drop post-origination / outcome columns,
       set aside pricing-only columns (grade/sub_grade/int_rate/installment)
    3. Drop columns above the missingness threshold
    4. Time-aware split into train / valid / test using issue_d
    5. Write reports/eda_findings.md documenting every decision above
       (this IS the "leakage audit document" deliverable)

Run directly (after load_data.py has produced the cached parquet):
    python src/clean_data.py
"""

import pandas as pd

from utils import load_config, resolve_path, get_logger

logger = get_logger("clean_data")


# ---------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------
def derive_target(df: pd.DataFrame, target_cfg: dict) -> pd.DataFrame:
    status_col = target_cfg["status_col"]
    target_col = target_cfg["target_col"]
    pos = set(target_cfg["positive_labels"])
    neg = set(target_cfg["negative_labels"])

    before = len(df)
    df = df[df[status_col].isin(pos | neg)].copy()
    dropped = before - len(df)
    logger.info(
        f"Dropped {dropped:,} rows with unresolved loan_status "
        f"(kept only {sorted(pos | neg)})"
    )

    df[target_col] = df[status_col].isin(pos).astype(int)
    logger.info(f"Target balance:\n{df[target_col].value_counts(normalize=True)}")
    return df


# ---------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------
def get_leakage_columns(df: pd.DataFrame, leakage_cfg: dict) -> list[str]:
    """Flatten explicit leakage lists + regex-matched prefix columns."""
    explicit = leakage_cfg["payment_outcome"] + leakage_cfg["identifiers_free_text"]

    prefixes = tuple(leakage_cfg["leakage_prefixes"])
    prefix_matched = [c for c in df.columns if c.startswith(prefixes)]

    all_leakage = sorted(set(explicit) | set(prefix_matched))
    return [c for c in all_leakage if c in df.columns]


def apply_leakage_audit(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Drop leakage + drop_columns, set aside pricing-only columns.

    Returns the cleaned df and an audit dict for the EDA report.
    """
    leakage_cfg = config["leakage_columns"]
    leakage_cols = get_leakage_columns(df, leakage_cfg)
    pricing_cols = [c for c in config["pricing_only_columns"] if c in df.columns]
    drop_cols = [c for c in config["drop_columns"] if c in df.columns]

    df_pricing = df[pricing_cols].copy() if pricing_cols else pd.DataFrame()

    to_drop = set(leakage_cols) | set(pricing_cols) | set(drop_cols)
    to_drop.discard(config["target"]["target_col"])
    to_drop.discard(config["target"]["status_col"])  # keep for now, drop later

    df_clean = df.drop(columns=[c for c in to_drop if c in df.columns])

    audit = {
        "leakage_cols": leakage_cols,
        "pricing_cols": pricing_cols,
        "drop_cols": drop_cols,
        "n_dropped": len(to_drop),
        "n_remaining": df_clean.shape[1],
    }
    logger.info(
        f"Leakage audit: dropped {len(leakage_cols)} leakage cols, "
        f"set aside {len(pricing_cols)} pricing-only cols, "
        f"dropped {len(drop_cols)} misc cols. "
        f"{df_clean.shape[1]} columns remain."
    )
    return df_clean, df_pricing, audit


def drop_high_missingness(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    missing_pct = df.isna().mean()
    to_drop = missing_pct[missing_pct > threshold].index.tolist()
    df_out = df.drop(columns=to_drop)
    logger.info(
        f"Dropped {len(to_drop)} columns with >{threshold:.0%} missingness"
    )
    return df_out, to_drop


# ---------------------------------------------------------------------
# Time-aware split
# ---------------------------------------------------------------------
def time_aware_split(df: pd.DataFrame, split_cfg: dict) -> dict[str, pd.DataFrame]:
    date_col = split_cfg["date_col"]
    train_end = pd.Timestamp(split_cfg["train_end"])
    valid_end = pd.Timestamp(split_cfg["valid_end"])

    df_sorted = df.sort_values(date_col)
    train = df_sorted[df_sorted[date_col] <= train_end]
    valid = df_sorted[(df_sorted[date_col] > train_end) & (df_sorted[date_col] <= valid_end)]
    test = df_sorted[df_sorted[date_col] > valid_end]

    logger.info(
        f"Split sizes -> train: {len(train):,} | valid: {len(valid):,} | "
        f"test (out-of-time): {len(test):,}"
    )
    return {"train": train, "valid": valid, "test": test}


# ---------------------------------------------------------------------
# EDA / leakage audit report
# ---------------------------------------------------------------------
def write_eda_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    audit: dict,
    missingness_dropped: list[str],
    splits: dict[str, pd.DataFrame],
    target_col: str,
    missing_threshold: float,
    out_path,
) -> None:
    lines = []
    lines.append("# EDA Findings & Leakage Audit\n")
    lines.append(f"- Raw columns: **{df_before.shape[1]}**")
    lines.append(f"- Columns after leakage/pricing/misc drop: **{df_before.shape[1] - audit['n_dropped']}**")
    lines.append(f"- Columns after missingness filter: **{df_after.shape[1]}**")
    lines.append(f"- Rows after target resolution filter: **{len(df_after):,}**\n")

    lines.append("## Leakage columns dropped (post-origination outcome data)\n")
    for c in audit["leakage_cols"]:
        lines.append(f"- `{c}`")

    lines.append("\n## Pricing-only columns (reserved for benchmarking, NOT model features)\n")
    for c in audit["pricing_cols"]:
        lines.append(f"- `{c}`")

    lines.append("\n## Misc columns dropped (constant / not useful)\n")
    for c in audit["drop_cols"]:
        lines.append(f"- `{c}`")

    lines.append(f"\n## Columns dropped for high missingness (> {missing_threshold:.0%})\n")
    for c in missingness_dropped:
        lines.append(f"- `{c}`")

    lines.append("\n## Target balance (post-filter)\n")
    balance = df_after[target_col].value_counts(normalize=True).round(4)
    lines.append(f"- 0 (Fully Paid): {balance.get(0, 0):.2%}")
    lines.append(f"- 1 (Charged Off / Default): {balance.get(1, 0):.2%}")

    lines.append("\n## Time-aware split sizes\n")
    for name, part in splits.items():
        lines.append(f"- {name}: {len(part):,} rows")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote EDA/leakage audit report to {out_path}")


# ---------------------------------------------------------------------
def main():
    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    raw_parsed_path = processed_dir / "loan_raw_parsed.parquet"

    logger.info(f"Loading {raw_parsed_path} ...")
    df = pd.read_parquet(raw_parsed_path)
    df_before = df.copy()

    df = derive_target(df, config["target"])
    df_clean, df_pricing, audit = apply_leakage_audit(df, config)
    df_clean, missingness_dropped = drop_high_missingness(
        df_clean, config["cleaning"]["drop_col_missing_threshold"]
    )

    # loan_status no longer needed once target is derived
    status_col = config["target"]["status_col"]
    if status_col in df_clean.columns:
        df_clean = df_clean.drop(columns=[status_col])

    splits = time_aware_split(df_clean, config["split"])

    # Persist
    df_clean.to_parquet(processed_dir / "loan_clean.parquet", index=False)
    df_pricing.to_parquet(processed_dir / "loan_pricing_only.parquet", index=False)
    for name, part in splits.items():
        part.to_parquet(processed_dir / f"loan_{name}.parquet", index=False)

    write_eda_report(
        df_before=df_before,
        df_after=df_clean,
        audit=audit,
        missingness_dropped=missingness_dropped,
        splits=splits,
        target_col=config["target"]["target_col"],
        missing_threshold=config["cleaning"]["drop_col_missing_threshold"],
        out_path=resolve_path(config["paths"]["eda_report"]),
    )

    logger.info("clean_data.py complete.")


if __name__ == "__main__":
    main()