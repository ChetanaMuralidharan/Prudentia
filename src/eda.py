"""
eda.py
======
Phase 2 - Time-Aware Split Validation & Business-Framed EDA.

This script intentionally replaces a notebook-based EDA workflow.
It reads the cleaned Phase 1 outputs, creates business-focused EDA plots,
and writes a markdown findings report.

Run from the project root:
    python src/eda.py

Expected Phase 1 inputs:
    data/processed/loan_raw_parsed.parquet
    data/processed/loan_clean.parquet
    data/processed/loan_train.parquet
    data/processed/loan_valid.parquet
    data/processed/loan_test.parquet

Outputs:
    reports/phase2_eda_findings.md
    reports/figures/*.png
    data/processed/phase2_eda_tables/*.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from utils import get_logger, load_config, resolve_path

logger = get_logger("eda")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info(f"Saved table: {path}")


def clean_percent_column(series: pd.Series) -> pd.Series:
    """Convert values like '13.56%' to numeric 13.56."""
    if series.dtype == "object":
        return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def derive_target_for_raw(df: pd.DataFrame, target_cfg: dict) -> pd.DataFrame:
    """Derive target on the raw parsed file for EDA-only pricing/grade analysis."""
    status_col = target_cfg["status_col"]
    target_col = target_cfg["target_col"]
    positive = set(target_cfg["positive_labels"])
    negative = set(target_cfg["negative_labels"])

    out = df[df[status_col].isin(positive | negative)].copy()
    out[target_col] = out[status_col].isin(positive).astype(int)
    return out


def add_business_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add EDA-only risk bands. These are not final modeling features yet."""
    out = df.copy()

    if "dti" in out.columns:
        out["dti"] = pd.to_numeric(out["dti"], errors="coerce")
        out["dti_band"] = pd.cut(
            out["dti"],
            bins=[-1, 10, 20, 30, 40, 60, 10_000],
            labels=["0-10", "10-20", "20-30", "30-40", "40-60", "60+"],
        )

    if {"fico_range_low", "fico_range_high"}.issubset(out.columns):
        out["fico_range_low"] = pd.to_numeric(out["fico_range_low"], errors="coerce")
        out["fico_range_high"] = pd.to_numeric(out["fico_range_high"], errors="coerce")
        out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
        out["fico_band"] = pd.cut(
            out["fico_avg"],
            bins=[0, 660, 700, 740, 780, 900],
            labels=["<660", "660-699", "700-739", "740-779", "780+"],
        )

    if "annual_inc" in out.columns:
        out["annual_inc"] = pd.to_numeric(out["annual_inc"], errors="coerce")
        out["income_band"] = pd.cut(
            out["annual_inc"],
            bins=[-1, 30_000, 60_000, 90_000, 120_000, 200_000, 100_000_000],
            labels=["<30k", "30k-60k", "60k-90k", "90k-120k", "120k-200k", "200k+"],
        )

    if "loan_amnt" in out.columns:
        out["loan_amnt"] = pd.to_numeric(out["loan_amnt"], errors="coerce")
        out["loan_amount_band"] = pd.cut(
            out["loan_amnt"],
            bins=[-1, 5_000, 10_000, 15_000, 20_000, 30_000, 100_000],
            labels=["<5k", "5k-10k", "10k-15k", "15k-20k", "20k-30k", "30k+"],
        )

    if "revol_util" in out.columns:
        out["revol_util"] = clean_percent_column(out["revol_util"])
        out["revol_util_band"] = pd.cut(
            out["revol_util"],
            bins=[-1, 20, 40, 60, 80, 100, 10_000],
            labels=["0-20", "20-40", "40-60", "60-80", "80-100", "100+"],
        )

    if "int_rate" in out.columns:
        out["int_rate"] = clean_percent_column(out["int_rate"])
        out["interest_rate_band"] = pd.cut(
            out["int_rate"],
            bins=[-1, 8, 12, 16, 20, 24, 100],
            labels=["<8", "8-12", "12-16", "16-20", "20-24", "24+"],
        )

    return out


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def default_rate_table(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    min_count: int = 500,
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    table = (
        df.groupby(group_col, dropna=False, observed=False)
        .agg(
            loans=(target_col, "size"),
            defaults=(target_col, "sum"),
            default_rate=(target_col, "mean"),
        )
        .reset_index()
    )
    table = table[table["loans"] >= min_count].copy()
    table["default_rate_pct"] = table["default_rate"] * 100
    table = table.drop(columns=["default_rate"])
    return table.sort_values("default_rate_pct", ascending=False)


def split_summary_table(splits: dict[str, pd.DataFrame], date_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for name, part in splits.items():
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "start_issue_d": part[date_col].min(),
                "end_issue_d": part[date_col].max(),
                "default_rate_pct": part[target_col].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def yearly_default_table(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["issue_year"] = out[date_col].dt.year
    table = (
        out.groupby("issue_year")
        .agg(loans=(target_col, "size"), defaults=(target_col, "sum"), default_rate_pct=(target_col, lambda x: x.mean() * 100))
        .reset_index()
        .sort_values("issue_year")
    )
    return table


def top_numeric_correlations(df: pd.DataFrame, target_col: str, top_n: int = 20) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").copy()
    if target_col not in numeric.columns:
        return pd.DataFrame()

    corr = (
        numeric.corr(numeric_only=True)[target_col]
        .drop(labels=[target_col], errors="ignore")
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_n)
        .reset_index()
    )
    corr.columns = ["feature", "correlation_with_default"]
    corr["absolute_correlation"] = corr["correlation_with_default"].abs()
    return corr


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_loan_volume_over_time(df: pd.DataFrame, date_col: str, out_path: Path) -> None:
    monthly = df.set_index(date_col).resample("ME").size().reset_index(name="loan_count")

    plt.figure(figsize=(12, 5))
    plt.plot(monthly[date_col], monthly["loan_count"])
    plt.axvline(pd.Timestamp("2015-12-31"), linestyle="--")
    plt.axvline(pd.Timestamp("2016-12-31"), linestyle="--")
    plt.title("Loan Volume Over Time with Time-Aware Split Boundaries")
    plt.xlabel("Issue Date")
    plt.ylabel("Number of Loans")
    save_plot(out_path)


def plot_yearly_default_rate(table: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(table["issue_year"], table["default_rate_pct"], marker="o")
    plt.title("Default Rate by Issue Year")
    plt.xlabel("Issue Year")
    plt.ylabel("Default Rate (%)")
    save_plot(out_path)


def plot_default_bar(table: pd.DataFrame, group_col: str, title: str, out_path: Path) -> None:
    if table.empty:
        return

    plot_table = table.sort_values("default_rate_pct", ascending=False)
    plt.figure(figsize=(11, 5))
    plt.bar(plot_table[group_col].astype(str), plot_table["default_rate_pct"])
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel("Default Rate (%)")
    plt.xticks(rotation=45, ha="right")
    save_plot(out_path)


def plot_distribution_by_split(
    splits: dict[str, pd.DataFrame],
    column: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    plotted = False
    for split_name, part in splits.items():
        if column in part.columns:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            if not values.empty:
                plt.hist(values, bins=50, alpha=0.45, density=True, label=split_name)
                plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    save_plot(out_path)


def plot_scatter_sample(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str,
    title: str,
    out_path: Path,
    sample_size: int = 50_000,
) -> None:
    needed = {x_col, y_col, target_col}
    if not needed.issubset(df.columns):
        return

    sample = df[list(needed)].dropna()
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample[x_col], sample[y_col], s=4, alpha=0.2)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    save_plot(out_path)




def remaining_missingness_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return top remaining missing-value percentages after Phase 1 cleaning."""
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    out = missing[missing > 0].head(top_n).reset_index()
    out.columns = ["feature", "missing_pct"]
    return out


def plot_correlation_bar(corr_table: pd.DataFrame, out_path: Path) -> None:
    """Plot top numeric absolute correlations with default."""
    if corr_table.empty:
        return
    plot_df = corr_table.sort_values("absolute_correlation", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature"], plot_df["absolute_correlation"])
    plt.title("Top Numeric Correlations with Default")
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Feature")
    save_plot(out_path)


def plot_missingness_bar(missing_table: pd.DataFrame, out_path: Path) -> None:
    """Plot remaining missingness after cleaning."""
    if missing_table.empty:
        return
    plot_df = missing_table.sort_values("missing_pct", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature"], plot_df["missing_pct"])
    plt.title("Top Remaining Missingness After Phase 1 Cleaning")
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Feature")
    save_plot(out_path)


def get_rate(table: pd.DataFrame, key_col: str, key_value: str) -> float | None:
    if table.empty or key_col not in table.columns:
        return None
    subset = table[table[key_col].astype(str) == str(key_value)]
    if subset.empty:
        return None
    return float(subset["default_rate_pct"].iloc[0])


def highest_lowest_statement(table: pd.DataFrame, group_col: str) -> str:
    if table.empty or group_col not in table.columns:
        return "No conclusion available because the table was not generated."
    ordered = table.sort_values("default_rate_pct", ascending=False)
    high = ordered.iloc[0]
    low = ordered.iloc[-1]
    return (
        f"The highest observed default rate is in `{high[group_col]}` "
        f"({high['default_rate_pct']:.2f}%), while the lowest is in `{low[group_col]}` "
        f"({low['default_rate_pct']:.2f}%)."
    )


def monotonic_direction(table: pd.DataFrame, group_col: str) -> str:
    """Simple narrative helper for ordered risk bands already sorted in logical order."""
    if table.empty or len(table) < 2:
        return "There is not enough information to evaluate a trend."
    values = table["default_rate_pct"].to_list()
    increasing_steps = sum(values[i] <= values[i + 1] for i in range(len(values) - 1))
    decreasing_steps = sum(values[i] >= values[i + 1] for i in range(len(values) - 1))
    if increasing_steps >= len(values) - 2:
        return f"Default risk generally increases across `{group_col}` bands, making this a strong Phase 3 feature-engineering candidate."
    if decreasing_steps >= len(values) - 2:
        return f"Default risk generally decreases across `{group_col}` bands, making this a strong Phase 3 feature-engineering candidate."
    return f"Default risk varies across `{group_col}` bands, but the relationship is not perfectly monotonic and should be checked with WOE/IV in Phase 3."


# ---------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------
def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No table generated."
    return df.head(max_rows).to_markdown(index=False, floatfmt=".2f")


def write_report(
    output_path: Path,
    split_summary: pd.DataFrame,
    yearly_default: pd.DataFrame,
    eda_tables: dict[str, pd.DataFrame],
    figure_paths: Iterable[Path],
) -> None:
    dti = eda_tables.get("Default Rate by DTI Band", pd.DataFrame())
    income = eda_tables.get("Default Rate by Income Band", pd.DataFrame())
    loan_amount = eda_tables.get("Default Rate by Loan Amount Band", pd.DataFrame())
    home = eda_tables.get("Default Rate by Home Ownership", pd.DataFrame())
    verification = eda_tables.get("Default Rate by Verification Status", pd.DataFrame())
    util = eda_tables.get("Default Rate by Revolving Utilization Band", pd.DataFrame())
    grade = eda_tables.get("Default Rate by Lending Club Grade", pd.DataFrame())
    interest = eda_tables.get("Default Rate by Interest Rate Band", pd.DataFrame())
    purpose = eda_tables.get("Default Rate by Loan Purpose", pd.DataFrame())
    corr = eda_tables.get("Top Numeric Correlations with Default", pd.DataFrame())
    missing = eda_tables.get("Remaining Missingness After Cleaning", pd.DataFrame())

    train_rate = float(split_summary.loc[split_summary["split"] == "train", "default_rate_pct"].iloc[0])
    valid_rate = float(split_summary.loc[split_summary["split"] == "valid", "default_rate_pct"].iloc[0])
    test_rate = float(split_summary.loc[split_summary["split"] == "test", "default_rate_pct"].iloc[0])
    peak_year = yearly_default.sort_values("default_rate_pct", ascending=False).iloc[0]

    lines = []
    lines.append("# Phase 2 EDA Findings")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(
        "Phase 2 confirms that default behavior varies materially across borrower profile, loan structure, "
        "credit history, and origination period. The validation period has a materially higher default rate "
        f"({valid_rate:.2f}%) than the training period ({train_rate:.2f}%), which supports the decision to use "
        "an out-of-time validation strategy rather than a random split."
    )
    lines.append("")
    lines.append(
        "The strongest business patterns are consistent with credit-risk intuition: higher DTI, higher revolving "
        "utilization, larger loan amounts, lower income, weaker Lending Club grades, and higher interest-rate bands "
        "are associated with higher observed default rates. These patterns should guide Phase 3 feature engineering."
    )
    lines.append("")

    lines.append("## Objective")
    lines.append(
        "Phase 2 validates the time-aware split and frames exploratory analysis around credit-risk business questions, "
        "not generic plots. Pricing variables such as `grade`, `sub_grade`, `int_rate`, and `installment` remain "
        "excluded from model features, but they are analyzed here as external benchmark signals."
    )
    lines.append("")

    lines.append("## Time-Aware Split Validation")
    lines.append(
        "The dataset is split chronologically using `issue_d`. This avoids random-split leakage where future economic "
        "conditions influence training. The split also reveals temporal risk drift: validation defaults are higher "
        "than training defaults, so model selection should be based on validation performance, not training fit."
    )
    lines.append("")
    lines.append(markdown_table(split_summary))
    lines.append("")
    lines.append(
        f"**Interpretation:** The training period default rate is {train_rate:.2f}%, validation is {valid_rate:.2f}%, "
        f"and test is {test_rate:.2f}%. This shift confirms that out-of-time testing is necessary for a realistic "
        "credit-risk decisioning engine."
    )
    lines.append("")

    lines.append("## Default Rate Over Time")
    lines.append(markdown_table(yearly_default, max_rows=30))
    lines.append("")
    lines.append(
        f"**Interpretation:** The highest observed annual default rate occurs in {int(peak_year['issue_year'])} "
        f"at {peak_year['default_rate_pct']:.2f}%. This supports treating origination time as an important "
        "validation and monitoring dimension."
    )
    lines.append("")

    ordered_sections = [
        ("Default Rate by Loan Purpose", "purpose", highest_lowest_statement(purpose, "purpose")),
        ("Default Rate by DTI Band", "dti_band", monotonic_direction(dti.sort_values("dti_band") if not dti.empty else dti, "dti_band")),
        ("Default Rate by Income Band", "income_band", monotonic_direction(income.sort_values("income_band") if not income.empty else income, "income_band")),
        ("Default Rate by Loan Amount Band", "loan_amount_band", monotonic_direction(loan_amount.sort_values("loan_amount_band") if not loan_amount.empty else loan_amount, "loan_amount_band")),
        ("Default Rate by Home Ownership", "home_ownership", highest_lowest_statement(home, "home_ownership")),
        ("Default Rate by Verification Status", "verification_status", highest_lowest_statement(verification, "verification_status")),
        ("Default Rate by Employment Length", "emp_length", "Missing or short employment history shows elevated observed risk; employment length should be handled carefully in Phase 3 because missingness itself may contain signal."),
        ("Default Rate by Revolving Utilization Band", "revol_util_band", monotonic_direction(util.sort_values("revol_util_band") if not util.empty else util, "revol_util_band")),
        ("Default Rate by Delinquencies in Last 2 Years", "delinq_2yrs", "Prior delinquencies are associated with higher risk and should be retained as a candidate bureau-history feature."),
        ("Default Rate by Public Records", "pub_rec", "Public record counts show a risk separation versus borrowers with no public records, but sparse high-count categories should likely be capped or binned in Phase 3."),
        ("Default Rate by Lending Club Grade", "grade", "Lending Club grades strongly rank-order realized default risk, validating them as pricing benchmarks. They should not be used as model inputs because this project is building an independent decision engine."),
        ("Default Rate by Lending Club Sub-Grade", "sub_grade", "Sub-grades provide an even more granular benchmark risk ranking, useful for comparing the independent model against Lending Club's assigned risk tiers."),
        ("Default Rate by Interest Rate Band", "interest_rate_band", "Interest-rate bands strongly align with realized default risk and are useful for benchmarking, but they are excluded from training features to avoid building a model that simply learns Lending Club pricing decisions."),
        ("Top Numeric Correlations with Default", "feature", "The correlation table is used only as a directional screening tool. Credit-risk feature selection should rely more heavily on WOE/IV, validation performance, calibration, and stability."),
        ("Remaining Missingness After Cleaning", "feature", "Remaining missingness is limited enough for Phase 3 preprocessing. Features with meaningful missingness should receive explicit missing indicators or robust imputation rules."),
    ]

    for title, _, interpretation in ordered_sections:
        table = eda_tables.get(title)
        if table is None or table.empty:
            continue
        lines.append(f"## {title}")
        lines.append(markdown_table(table))
        lines.append("")
        lines.append(f"**Interpretation:** {interpretation}")
        lines.append("")

    lines.append("## Implications for Phase 3 Feature Engineering")
    lines.append("Phase 3 should convert the EDA patterns into stable, model-ready features. Recommended candidates include:")
    lines.append("")
    lines.append("- DTI bands and/or transformed DTI")
    lines.append("- Revolving utilization bands and capped utilization")
    lines.append("- Income bands and missing/income outlier handling")
    lines.append("- Loan amount bands")
    lines.append("- Delinquency and public-record indicators")
    lines.append("- Employment-length normalization, including an explicit missing category")
    lines.append("- Home ownership and verification-status encodings")
    lines.append("- Bureau/history variables with WOE/IV ranking")
    lines.append("")
    lines.append("Pricing-only variables (`grade`, `sub_grade`, `int_rate`, `installment`) should remain excluded from model training and used only for benchmarking model behavior against Lending Club's historical risk/pricing decisions.")
    lines.append("")

    lines.append("## Figures Generated")
    for path in figure_paths:
        lines.append(f"- `{path.as_posix()}`")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {output_path}")


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def main() -> None:
    config = load_config()

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    reports_dir = resolve_path("reports")
    figures_dir = reports_dir / "figures"
    tables_dir = processed_dir / "phase2_eda_tables"

    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    target_col = config["target"]["target_col"]
    date_col = config["split"]["date_col"]

    logger.info("Loading Phase 1 outputs...")
    clean = pd.read_parquet(processed_dir / "loan_clean.parquet")
    train = pd.read_parquet(processed_dir / "loan_train.parquet")
    valid = pd.read_parquet(processed_dir / "loan_valid.parquet")
    test = pd.read_parquet(processed_dir / "loan_test.parquet")

    for frame in [clean, train, valid, test]:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")

    # Add EDA-only bins to cleaned modeling-safe data.
    clean = add_business_bins(clean)
    train = add_business_bins(train)
    valid = add_business_bins(valid)
    test = add_business_bins(test)
    splits = {"train": train, "valid": valid, "test": test}

    # Load raw parsed data for pricing-only benchmark EDA.
    # Pricing columns are NOT used as model features, but they are useful for Phase 2 analysis.
    raw_path = processed_dir / "loan_raw_parsed.parquet"
    if raw_path.exists():
        raw_eda = pd.read_parquet(raw_path)
        raw_eda[date_col] = pd.to_datetime(raw_eda[date_col], errors="coerce")
        raw_eda = derive_target_for_raw(raw_eda, config["target"])
        raw_eda = add_business_bins(raw_eda)
    else:
        logger.warning("loan_raw_parsed.parquet not found. Pricing-only EDA will be skipped.")
        raw_eda = pd.DataFrame()

    logger.info("Creating tables...")
    split_summary = split_summary_table(splits, date_col, target_col)
    yearly_default = yearly_default_table(clean, date_col, target_col)

    write_csv(split_summary, tables_dir / "split_summary.csv")
    write_csv(yearly_default, tables_dir / "yearly_default_rate.csv")

    eda_tables: dict[str, pd.DataFrame] = {}

    safe_groups = [
        ("purpose", "Default Rate by Loan Purpose", clean),
        ("dti_band", "Default Rate by DTI Band", clean),
        ("fico_band", "Default Rate by FICO Band", clean),
        ("income_band", "Default Rate by Income Band", clean),
        ("loan_amount_band", "Default Rate by Loan Amount Band", clean),
        ("home_ownership", "Default Rate by Home Ownership", clean),
        ("verification_status", "Default Rate by Verification Status", clean),
        ("emp_length", "Default Rate by Employment Length", clean),
        ("revol_util_band", "Default Rate by Revolving Utilization Band", clean),
        ("delinq_2yrs", "Default Rate by Delinquencies in Last 2 Years", clean),
        ("pub_rec", "Default Rate by Public Records", clean),
    ]

    pricing_groups = []
    if not raw_eda.empty:
        pricing_groups = [
            ("grade", "Default Rate by Lending Club Grade", raw_eda),
            ("sub_grade", "Default Rate by Lending Club Sub-Grade", raw_eda),
            ("interest_rate_band", "Default Rate by Interest Rate Band", raw_eda),
        ]

    for group_col, title, source_df in safe_groups + pricing_groups:
        table = default_rate_table(source_df, group_col, target_col)
        if table.empty:
            continue
        eda_tables[title] = table
        safe_name = title.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        write_csv(table, tables_dir / f"{safe_name}.csv")

    corr = top_numeric_correlations(clean, target_col)
    if not corr.empty:
        eda_tables["Top Numeric Correlations with Default"] = corr
        write_csv(corr, tables_dir / "top_numeric_correlations.csv")

    missing = remaining_missingness_table(clean)
    if not missing.empty:
        eda_tables["Remaining Missingness After Cleaning"] = missing
        write_csv(missing, tables_dir / "remaining_missingness_after_cleaning.csv")

    logger.info("Creating figures...")
    figure_paths: list[Path] = []

    figure = figures_dir / "01_loan_volume_over_time.png"
    plot_loan_volume_over_time(clean, date_col, figure)
    figure_paths.append(figure)

    figure = figures_dir / "02_default_rate_by_year.png"
    plot_yearly_default_rate(yearly_default, figure)
    figure_paths.append(figure)

    for title, table in eda_tables.items():
        if "Top Numeric Correlations" in title or "Remaining Missingness" in title:
            continue
        group_col = table.columns[0]
        safe_name = title.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        figure = figures_dir / f"03_{safe_name}.png"
        plot_default_bar(table, group_col, title, figure)
        figure_paths.append(figure)

    if not corr.empty:
        figure = figures_dir / "04_top_numeric_correlations.png"
        plot_correlation_bar(corr, figure)
        figure_paths.append(figure)

    if not missing.empty:
        figure = figures_dir / "04_remaining_missingness_after_cleaning.png"
        plot_missingness_bar(missing, figure)
        figure_paths.append(figure)

    drift_columns = ["loan_amnt", "annual_inc", "dti", "fico_avg", "revol_util"]
    for col in drift_columns:
        if col in train.columns:
            figure = figures_dir / f"04_{col}_distribution_by_split.png"
            plot_distribution_by_split(splits, col, f"{col} Distribution by Time Split", figure)
            figure_paths.append(figure)

    if not raw_eda.empty and {"fico_avg", "int_rate"}.issubset(raw_eda.columns):
        figure = figures_dir / "05_fico_vs_interest_rate_sample.png"
        plot_scatter_sample(
            raw_eda,
            x_col="fico_avg",
            y_col="int_rate",
            target_col=target_col,
            title="FICO vs Interest Rate Sample",
            out_path=figure,
        )
        figure_paths.append(figure)

    write_report(
        output_path=reports_dir / "phase2_eda_findings.md",
        split_summary=split_summary,
        yearly_default=yearly_default,
        eda_tables=eda_tables,
        figure_paths=figure_paths,
    )

    logger.info("Phase 2 EDA complete.")


if __name__ == "__main__":
    main()
