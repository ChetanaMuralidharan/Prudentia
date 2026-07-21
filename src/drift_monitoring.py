"""
Phase 8: Drift Monitoring

Reference population:
    2016 validation data

Monitoring population:
    Quarterly 2017-2018 out-of-time test data

Run:
    python src/drift_monitoring.py
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from utils import get_logger, load_config, resolve_path


logger = get_logger("drift_monitoring")


# =============================================================================
# Configuration
# =============================================================================

REFERENCE_LABEL = "2016 Reference"

TOP_MODEL_FEATURES = 250
NUMERIC_BINS = 10
MAX_CATEGORY_LEVELS = 20
MIN_QUARTER_SIZE = 100
EPSILON = 1e-6

PSI_WATCH = 0.10
PSI_ALERT = 0.25

DEFAULT_THRESHOLD = 0.08
APPROVAL_CHANGE_ALERT = 0.10
AUC_DROP_ALERT = 0.05
CALIBRATION_GAP_ALERT = 0.05
BRIER_INCREASE_ALERT = 0.20
FEATURE_ALERT_COUNT_THRESHOLD = 3

# Outcome-dependent metrics are reported only for loans issued on or before
# this date. Later quarters continue to receive feature, score, risk-tier,
# approval-rate and population monitoring.
PERFORMANCE_CUTOFF_DATE = pd.Timestamp("2018-06-30")

RISK_BINS = [
    -np.inf,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    np.inf,
]

RISK_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
]

CORE_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "fico_range_low",
    "fico_range_high",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "acc_open_past_24mths",
    "bc_open_to_buy",
    "num_tl_op_past_12m",
    "tot_hi_cred_lim",
    "total_bc_limit",
    "total_rev_hi_lim",
    "mths_since_recent_inq",
    "mths_since_recent_bc",
    "mths_since_last_delinq",
    "mo_sin_old_rev_tl_op",
    "loan_to_income",
]


# =============================================================================
# General helpers
# =============================================================================

def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Convert numeric columns and malformed strings such as '[1.84325E-1]'
    into floating-point values.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.replace(",", "", regex=False)
    )

    return pd.to_numeric(cleaned, errors="coerce")


def psi_status(value: float) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    if value >= PSI_ALERT:
        return "ALERT"
    if value >= PSI_WATCH:
        return "WATCH"
    return "STABLE"


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)


def markdown_table(df: pd.DataFrame, rows: int = 25) -> str:
    if df.empty:
        return "_No records generated._"

    display = df.head(rows).copy()

    numeric_columns = display.select_dtypes(include=[np.number]).columns
    display[numeric_columns] = display[numeric_columns].round(6)

    return display.to_markdown(index=False)


# =============================================================================
# Input loading
# =============================================================================

def load_probability_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "calibrated_pd" not in df.columns:
        candidates = [
            column
            for column in df.columns
            if "calibrated" in column.lower()
            and (
                "prob" in column.lower()
                or "pd" in column.lower()
            )
        ]

        if not candidates:
            raise ValueError(
                f"No calibrated probability column found in {path}"
            )

        df = df.rename(
            columns={candidates[0]: "calibrated_pd"}
        )

    df["calibrated_pd"] = (
        clean_numeric(df["calibrated_pd"])
        .clip(0, 1)
    )

    if "target" in df.columns:
        df["target"] = (
            pd.to_numeric(df["target"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return df.reset_index(drop=True)


def get_issue_date(df: pd.DataFrame) -> pd.Series:
    for column in [
        "issue_d",
        "issue_date",
        "loan_issue_date",
    ]:
        if column in df.columns:
            return pd.to_datetime(
                df[column],
                errors="coerce",
            )

    raise ValueError(
        "No issue-date column found. Expected issue_d, "
        "issue_date or loan_issue_date."
    )


def load_threshold(processed_dir: Path) -> float:
    path = (
        processed_dir
        / "phase6_decisioning"
        / "cost_threshold_optimization.csv"
    )

    if not path.exists():
        logger.warning(
            "Phase 6 threshold file not found. Using %.4f.",
            DEFAULT_THRESHOLD,
        )
        return DEFAULT_THRESHOLD

    table = pd.read_csv(path)

    if not {"threshold", "total_cost"}.issubset(table.columns):
        logger.warning(
            "Threshold table is missing threshold or total_cost. "
            "Using %.4f.",
            DEFAULT_THRESHOLD,
        )
        return DEFAULT_THRESHOLD

    table["threshold"] = pd.to_numeric(
        table["threshold"],
        errors="coerce",
    )

    table["total_cost"] = pd.to_numeric(
        table["total_cost"],
        errors="coerce",
    )

    table = table.dropna(
        subset=["threshold", "total_cost"]
    )

    if table.empty:
        return DEFAULT_THRESHOLD

    threshold = float(
        table.loc[
            table["total_cost"].idxmin(),
            "threshold",
        ]
    )

    logger.info(
        "Using Phase 6 cost-optimal PD threshold: %.4f",
        threshold,
    )

    return threshold


def build_population(
    context: pd.DataFrame,
    probabilities: pd.DataFrame,
    reference: bool,
) -> pd.DataFrame:
    if len(context) != len(probabilities):
        raise ValueError(
            "Context and probability rows do not align: "
            f"{len(context)} versus {len(probabilities)}"
        )

    df = remove_duplicate_columns(
        context.reset_index(drop=True)
    )

    df["calibrated_pd"] = probabilities[
        "calibrated_pd"
    ].reset_index(drop=True)

    if "target" in probabilities.columns:
        df["target"] = probabilities[
            "target"
        ].reset_index(drop=True)

    elif "target" in df.columns:
        df["target"] = (
            pd.to_numeric(df["target"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    else:
        raise ValueError(
            "Target was not found in context or probability data."
        )

    df["issue_date"] = get_issue_date(df)

    if reference:
        df["quarter"] = REFERENCE_LABEL
        df["performance_eligible"] = True
        df["maturity_status"] = "REFERENCE"

    else:
        df["quarter"] = (
            df["issue_date"]
            .dt.to_period("Q")
            .astype("string")
        )

        df["performance_eligible"] = (
            df["issue_date"]
            <= PERFORMANCE_CUTOFF_DATE
        )

        df["maturity_status"] = np.where(
            df["performance_eligible"],
            "MATURE",
            "IMMATURE",
        )

    valid_rows = (
        df["issue_date"].notna()
        & df["calibrated_pd"].notna()
        & df["target"].notna()
    )

    removed_rows = int((~valid_rows).sum())

    if removed_rows:
        logger.warning(
            "Removing %d rows with missing issue date, "
            "probability or target.",
            removed_rows,
        )

    return df.loc[valid_rows].reset_index(drop=True)


def load_inputs(processed_dir: Path) -> dict:
    required = {
        "valid_context": (
            processed_dir / "loan_valid.parquet"
        ),
        "test_context": (
            processed_dir / "loan_test.parquet"
        ),
        "x_valid": (
            processed_dir / "X_valid.parquet"
        ),
        "x_test": (
            processed_dir / "X_test.parquet"
        ),
        "valid_prob": (
            processed_dir
            / "phase5_calibration"
            / "validation_selected_calibrated_probabilities.csv"
        ),
        "test_prob": (
            processed_dir
            / "phase5_calibration"
            / "test_selected_calibrated_probabilities.csv"
        ),
    }

    missing = [
        str(path)
        for path in required.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n- "
            + "\n- ".join(missing)
        )

    valid_context = pd.read_parquet(
        required["valid_context"]
    ).reset_index(drop=True)

    test_context = pd.read_parquet(
        required["test_context"]
    ).reset_index(drop=True)

    x_valid = pd.read_parquet(
        required["x_valid"]
    ).reset_index(drop=True)

    x_test = pd.read_parquet(
        required["x_test"]
    ).reset_index(drop=True)

    valid_probabilities = load_probability_file(
        required["valid_prob"]
    )

    test_probabilities = load_probability_file(
        required["test_prob"]
    )

    validation_population = build_population(
        valid_context,
        valid_probabilities,
        reference=True,
    )

    test_population = build_population(
        test_context,
        test_probabilities,
        reference=False,
    )

    common_features = [
        column
        for column in x_valid.columns
        if column in x_test.columns
    ]

    x_valid = remove_duplicate_columns(
        x_valid[common_features]
    )

    x_test = remove_duplicate_columns(
        x_test[common_features]
    )

    if len(x_valid) != len(validation_population):
        raise ValueError(
            "X_valid and validation population differ in size: "
            f"{len(x_valid)} versus "
            f"{len(validation_population)}"
        )

    if len(x_test) != len(test_population):
        raise ValueError(
            "X_test and test population differ in size: "
            f"{len(x_test)} versus {len(test_population)}"
        )

    return {
        "validation_population": validation_population,
        "test_population": test_population,
        "x_valid": x_valid.reset_index(drop=True),
        "x_test": x_test.reset_index(drop=True),
        "threshold": load_threshold(processed_dir),
    }


# =============================================================================
# PSI functions
# =============================================================================

def safe_distribution(values: pd.Series) -> pd.Series:
    counts = values.value_counts(
        dropna=False
    ).astype(float)

    if counts.sum() == 0:
        return counts

    return (
        counts / counts.sum()
    ).clip(lower=EPSILON)


def calculate_distribution_psi(
    reference_distribution: pd.Series,
    current_distribution: pd.Series,
) -> float:
    categories = reference_distribution.index.union(
        current_distribution.index
    )

    reference = (
        reference_distribution
        .reindex(categories, fill_value=EPSILON)
        .clip(lower=EPSILON)
    )

    current = (
        current_distribution
        .reindex(categories, fill_value=EPSILON)
        .clip(lower=EPSILON)
    )

    return float(
        (
            (current - reference)
            * np.log(current / reference)
        ).sum()
    )


def numeric_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    reference = clean_numeric(reference)
    current = clean_numeric(current)

    clean_reference = reference.dropna()

    if (
        clean_reference.empty
        or clean_reference.nunique() <= 1
    ):
        reference_labels = (
            reference.astype("string")
            .fillna("__MISSING__")
        )

        current_labels = (
            current.astype("string")
            .fillna("__MISSING__")
        )

        return calculate_distribution_psi(
            safe_distribution(reference_labels),
            safe_distribution(current_labels),
        )

    quantiles = np.linspace(
        0,
        1,
        NUMERIC_BINS + 1,
    )

    edges = np.unique(
        clean_reference
        .quantile(quantiles)
        .to_numpy(dtype=float)
    )

    if len(edges) < 2:
        edges = np.array(
            [-np.inf, np.inf]
        )

    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_bins = (
        pd.cut(
            reference,
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )
        .astype("string")
        .fillna("__MISSING__")
    )

    current_bins = (
        pd.cut(
            current,
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )
        .astype("string")
        .fillna("__MISSING__")
    )

    return calculate_distribution_psi(
        safe_distribution(reference_bins),
        safe_distribution(current_bins),
    )


def categorical_psi(
    reference: pd.Series,
    current: pd.Series,
) -> float:
    reference = (
        reference.astype("string")
        .fillna("__MISSING__")
    )

    current = (
        current.astype("string")
        .fillna("__MISSING__")
    )

    common_levels = (
        reference.value_counts()
        .head(MAX_CATEGORY_LEVELS)
        .index
    )

    reference = reference.where(
        reference.isin(common_levels),
        "__OTHER__",
    )

    current = current.where(
        current.isin(common_levels),
        "__OTHER__",
    )

    return calculate_distribution_psi(
        safe_distribution(reference),
        safe_distribution(current),
    )


def is_numeric_feature(
    series: pd.Series,
    feature_name: str,
) -> bool:
    if feature_name.startswith("numeric__"):
        return True

    if feature_name.startswith("categorical__"):
        return False

    if pd.api.types.is_numeric_dtype(series):
        return True

    return clean_numeric(series).notna().mean() >= 0.90


def feature_psi(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
) -> float:
    if is_numeric_feature(
        reference,
        feature_name,
    ):
        return numeric_psi(reference, current)

    return categorical_psi(reference, current)


def original_feature_name(
    feature: str,
    raw_feature_names: list[str],
) -> str:
    """
    Convert transformed feature names back to their underlying raw feature.

    Examples:
        numeric__revol_util -> revol_util
        categorical__verification_status_Verified
            -> verification_status
    """
    cleaned = feature

    for prefix in [
        "numeric__",
        "categorical__",
    ]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    for raw_feature in sorted(
        raw_feature_names,
        key=len,
        reverse=True,
    ):
        if cleaned == raw_feature:
            return raw_feature

        if cleaned.startswith(
            f"{raw_feature}_"
        ):
            return raw_feature

    return cleaned


def calculate_feature_psi(
    reference: pd.DataFrame,
    monitoring: pd.DataFrame,
    quarters: pd.Series,
    feature_names: list[str],
    monitoring_level: str,
) -> pd.DataFrame:
    rows = []

    available_features = [
        feature
        for feature in feature_names
        if feature in reference.columns
        and feature in monitoring.columns
    ]

    quarter_values = sorted(
        quarters.dropna().astype(str).unique()
    )

    quarter_strings = quarters.astype(str)

    for feature_number, feature in enumerate(
        available_features,
        start=1,
    ):
        reference_series = reference[feature]

        feature_type = (
            "numeric"
            if is_numeric_feature(
                reference_series,
                feature,
            )
            else "categorical"
        )

        for quarter in quarter_values:
            mask = quarter_strings.eq(quarter)

            current_series = monitoring.loc[
                mask,
                feature,
            ]

            if len(current_series) < MIN_QUARTER_SIZE:
                continue

            value = feature_psi(
                reference_series,
                current_series,
                feature,
            )

            rows.append(
                {
                    "quarter": quarter,
                    "feature": feature,
                    "monitoring_level": monitoring_level,
                    "feature_type": feature_type,
                    "psi": value,
                    "status": psi_status(value),
                    "reference_missing_rate": (
                        reference_series.isna().mean()
                    ),
                    "current_missing_rate": (
                        current_series.isna().mean()
                    ),
                    "current_rows": len(current_series),
                }
            )

        if (
            feature_number % 25 == 0
            or feature_number
            == len(available_features)
        ):
            logger.info(
                "Processed %d/%d %s features.",
                feature_number,
                len(available_features),
                monitoring_level,
            )

    return pd.DataFrame(rows)


def summarize_feature_psi(
    table: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()

    quarter_order = {
        quarter: index
        for index, quarter in enumerate(
            sorted(
                table["quarter"]
                .astype(str)
                .unique()
            )
        )
    }

    ordered = table.copy()

    ordered["_quarter_order"] = (
        ordered["quarter"]
        .astype(str)
        .map(quarter_order)
    )

    latest_rows = (
        ordered.sort_values(
            [
                "original_feature",
                "_quarter_order",
                "psi",
            ]
        )
        .groupby(
            "original_feature",
            as_index=False,
        )
        .tail(1)
        [
            [
                "original_feature",
                "psi",
                "status",
            ]
        ]
        .rename(
            columns={
                "psi": "latest_psi",
                "status": "latest_status",
            }
        )
    )

    summary = (
        table.groupby(
            "original_feature",
            as_index=False,
        )
        .agg(
            maximum_psi=("psi", "max"),
            average_psi=("psi", "mean"),
            quarters_in_watch=(
                "status",
                lambda values: (
                    values == "WATCH"
                ).sum(),
            ),
            quarters_in_alert=(
                "status",
                lambda values: (
                    values == "ALERT"
                ).sum(),
            ),
            monitored_columns=(
                "feature",
                "nunique",
            ),
        )
        .merge(
            latest_rows,
            on="original_feature",
            how="left",
        )
        .sort_values(
            "maximum_psi",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return summary


# =============================================================================
# Model feature selection
# =============================================================================

def select_model_features(
    x_valid: pd.DataFrame,
    processed_dir: Path,
) -> list[str]:
    shap_path = (
        processed_dir
        / "phase7_explainability"
        / "global_shap_importance.csv"
    )

    selected = []

    if shap_path.exists():
        shap_table = pd.read_csv(shap_path)

        required_columns = {
            "feature",
            "mean_abs_shap",
        }

        if required_columns.issubset(
            shap_table.columns
        ):
            shap_table["mean_abs_shap"] = pd.to_numeric(
                shap_table["mean_abs_shap"],
                errors="coerce",
            )

            ranked_features = (
                shap_table
                .dropna(
                    subset=["mean_abs_shap"]
                )
                .sort_values(
                    "mean_abs_shap",
                    ascending=False,
                )["feature"]
            )

            selected = [
                feature
                for feature in ranked_features
                if feature in x_valid.columns
            ][:TOP_MODEL_FEATURES]

    if len(selected) < TOP_MODEL_FEATURES:
        logger.warning(
            "SHAP importance supplied only %d usable features. "
            "Using variance to fill the remaining positions.",
            len(selected),
        )

        remaining = [
            feature
            for feature in x_valid.columns
            if feature not in selected
        ]

        variances = {}

        for feature in remaining:
            variance = clean_numeric(
                x_valid[feature]
            ).var()

            variances[feature] = (
                float(variance)
                if pd.notna(variance)
                else 0.0
            )

        remaining = sorted(
            remaining,
            key=lambda feature: variances[feature],
            reverse=True,
        )

        remaining_slots = (
            TOP_MODEL_FEATURES
            - len(selected)
        )

        selected.extend(
            remaining[:remaining_slots]
        )

    logger.info(
        "Selected %d model features from %d encoded features.",
        len(selected),
        x_valid.shape[1],
    )

    return selected


# =============================================================================
# Portfolio and model metrics
# =============================================================================

def add_policy_fields(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    result = df.copy()

    result["approved"] = (
        result["calibrated_pd"] < threshold
    ).astype(int)

    result["risk_tier"] = pd.cut(
        result["calibrated_pd"],
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True,
    ).astype("string")

    return result


def safe_auc(
    target: pd.Series,
    scores: pd.Series,
) -> float:
    valid = pd.DataFrame(
        {
            "target": target,
            "score": scores,
        }
    ).dropna()

    if valid["target"].nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(
            valid["target"],
            valid["score"],
        )
    )


def safe_ks(
    target: pd.Series,
    scores: pd.Series,
) -> float:
    valid = pd.DataFrame(
        {
            "target": target,
            "score": scores,
        }
    ).dropna()

    if valid["target"].nunique() < 2:
        return np.nan

    false_positive_rate, true_positive_rate, _ = roc_curve(
        valid["target"],
        valid["score"],
    )

    return float(
        np.max(
            true_positive_rate
            - false_positive_rate
        )
    )


def calculate_period_metrics(
    df: pd.DataFrame,
    quarter: str,
    performance_eligible: bool,
) -> dict:
    approved = df["approved"].eq(1)

    average_pd = df[
        "calibrated_pd"
    ].mean()

    metrics = {
        "quarter": quarter,
        "loans": len(df),
        "performance_eligible": performance_eligible,
        "maturity_status": (
            "REFERENCE"
            if quarter == REFERENCE_LABEL
            else "MATURE"
            if performance_eligible
            else "IMMATURE"
        ),
        "approval_rate": df[
            "approved"
        ].mean(),
        "average_pd": average_pd,
        "high_risk_share": (
            df["risk_tier"]
            .isin(["E", "F"])
            .mean()
        ),
    }

    if not performance_eligible:
        metrics.update(
            {
                "observed_default_rate": np.nan,
                "approved_default_rate": np.nan,
                "rejected_default_rate": np.nan,
                "calibration_gap": np.nan,
                "roc_auc": np.nan,
                "gini": np.nan,
                "ks_statistic": np.nan,
                "brier_score": np.nan,
            }
        )

        return metrics

    observed_default_rate = df[
        "target"
    ].mean()

    auc = safe_auc(
        df["target"],
        df["calibrated_pd"],
    )

    metrics.update(
        {
            "observed_default_rate": (
                observed_default_rate
            ),
            "approved_default_rate": (
                df.loc[
                    approved,
                    "target",
                ].mean()
                if approved.any()
                else np.nan
            ),
            "rejected_default_rate": (
                df.loc[
                    ~approved,
                    "target",
                ].mean()
                if (~approved).any()
                else np.nan
            ),
            "calibration_gap": (
                average_pd
                - observed_default_rate
            ),
            "roc_auc": auc,
            "gini": (
                2 * auc - 1
                if pd.notna(auc)
                else np.nan
            ),
            "ks_statistic": safe_ks(
                df["target"],
                df["calibrated_pd"],
            ),
            "brier_score": brier_score_loss(
                df["target"],
                df["calibrated_pd"],
            ),
        }
    )

    return metrics


def calculate_quarterly_metrics(
    reference: pd.DataFrame,
    monitoring: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        calculate_period_metrics(
            reference,
            REFERENCE_LABEL,
            performance_eligible=True,
        )
    ]

    for quarter, batch in monitoring.groupby(
        "quarter",
        sort=True,
    ):
        if len(batch) < MIN_QUARTER_SIZE:
            continue

        performance_eligible = bool(
            batch["performance_eligible"].all()
        )

        rows.append(
            calculate_period_metrics(
                batch,
                str(quarter),
                performance_eligible,
            )
        )

    table = pd.DataFrame(rows)

    portfolio_columns = [
        "quarter",
        "loans",
        "performance_eligible",
        "maturity_status",
        "approval_rate",
        "observed_default_rate",
        "approved_default_rate",
        "rejected_default_rate",
        "average_pd",
        "calibration_gap",
        "high_risk_share",
    ]

    performance_columns = [
        "quarter",
        "loans",
        "performance_eligible",
        "maturity_status",
        "roc_auc",
        "gini",
        "ks_statistic",
        "brier_score",
    ]

    return (
        table[portfolio_columns],
        table[performance_columns],
    )


# =============================================================================
# Score and risk-tier monitoring
# =============================================================================

def calculate_score_psi(
    reference: pd.DataFrame,
    monitoring: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quarter, batch in monitoring.groupby(
        "quarter",
        sort=True,
    ):
        value = numeric_psi(
            reference["calibrated_pd"],
            batch["calibrated_pd"],
        )

        rows.append(
            {
                "quarter": str(quarter),
                "score_psi": value,
                "status": psi_status(value),
                "reference_average_pd": (
                    reference[
                        "calibrated_pd"
                    ].mean()
                ),
                "current_average_pd": (
                    batch[
                        "calibrated_pd"
                    ].mean()
                ),
                "current_rows": len(batch),
            }
        )

    return pd.DataFrame(rows)


def calculate_risk_tier_psi(
    reference: pd.DataFrame,
    monitoring: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quarter, batch in monitoring.groupby(
        "quarter",
        sort=True,
    ):
        value = categorical_psi(
            reference["risk_tier"],
            batch["risk_tier"],
        )

        rows.append(
            {
                "quarter": str(quarter),
                "risk_tier_psi": value,
                "status": psi_status(value),
            }
        )

    return pd.DataFrame(rows)


def calculate_risk_tier_distribution(
    reference: pd.DataFrame,
    monitoring: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    populations = [
        (REFERENCE_LABEL, reference)
    ]

    populations.extend(
        (
            str(quarter),
            batch,
        )
        for quarter, batch in monitoring.groupby(
            "quarter",
            sort=True,
        )
    )

    for quarter, population in populations:
        distribution = (
            population["risk_tier"]
            .value_counts(normalize=True)
            .reindex(
                RISK_LABELS,
                fill_value=0,
            )
        )

        for risk_tier, share in distribution.items():
            rows.append(
                {
                    "quarter": quarter,
                    "risk_tier": risk_tier,
                    "share": share,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Alerts and recommendations
# =============================================================================

def generate_alerts(
    feature_psi_table: pd.DataFrame,
    score_psi_table: pd.DataFrame,
    risk_psi_table: pd.DataFrame,
    portfolio: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    alerts = []

    reference_portfolio = portfolio[
        portfolio["quarter"].eq(
            REFERENCE_LABEL
        )
    ].iloc[0]

    reference_performance = performance[
        performance["quarter"].eq(
            REFERENCE_LABEL
        )
    ].iloc[0]

    for quarter in score_psi_table[
        "quarter"
    ]:
        score_row = score_psi_table[
            score_psi_table["quarter"].eq(
                quarter
            )
        ].iloc[0]

        if score_row["score_psi"] >= PSI_ALERT:
            alerts.append(
                [
                    quarter,
                    "Score drift",
                    "score_psi",
                    score_row["score_psi"],
                    PSI_ALERT,
                    "CRITICAL",
                    (
                        "Investigate model score "
                        "distribution."
                    ),
                ]
            )

        elif score_row["score_psi"] >= PSI_WATCH:
            alerts.append(
                [
                    quarter,
                    "Score drift",
                    "score_psi",
                    score_row["score_psi"],
                    PSI_WATCH,
                    "WATCH",
                    (
                        "Continue monitoring "
                        "score movement."
                    ),
                ]
            )

        quarter_features = feature_psi_table[
            feature_psi_table[
                "quarter"
            ].eq(quarter)
        ]

        distinct_feature_alerts = (
            quarter_features.loc[
                quarter_features[
                    "psi"
                ].ge(PSI_ALERT),
                "original_feature",
            ]
            .nunique()
        )

        if (
            distinct_feature_alerts
            >= FEATURE_ALERT_COUNT_THRESHOLD
        ):
            alerts.append(
                [
                    quarter,
                    "Feature drift",
                    "distinct_alert_feature_count",
                    distinct_feature_alerts,
                    FEATURE_ALERT_COUNT_THRESHOLD,
                    "HIGH",
                    (
                        "Review the most drifted "
                        "business features."
                    ),
                ]
            )

        risk_row = risk_psi_table[
            risk_psi_table["quarter"].eq(
                quarter
            )
        ].iloc[0]

        if (
            risk_row["risk_tier_psi"]
            >= PSI_ALERT
        ):
            alerts.append(
                [
                    quarter,
                    "Risk-tier drift",
                    "risk_tier_psi",
                    risk_row["risk_tier_psi"],
                    PSI_ALERT,
                    "HIGH",
                    (
                        "Review migration into "
                        "higher-risk tiers."
                    ),
                ]
            )

        portfolio_row = portfolio[
            portfolio["quarter"].eq(
                quarter
            )
        ].iloc[0]

        approval_change = abs(
            portfolio_row["approval_rate"]
            - reference_portfolio[
                "approval_rate"
            ]
        )

        if (
            approval_change
            >= APPROVAL_CHANGE_ALERT
        ):
            alerts.append(
                [
                    quarter,
                    "Policy impact",
                    "approval_rate_change",
                    approval_change,
                    APPROVAL_CHANGE_ALERT,
                    "HIGH",
                    (
                        "Review the approval threshold "
                        "and population mix."
                    ),
                ]
            )

        if not bool(
            portfolio_row[
                "performance_eligible"
            ]
        ):
            alerts.append(
                [
                    quarter,
                    "Outcome maturity",
                    "performance_eligible",
                    0,
                    1,
                    "PROVISIONAL",
                    (
                        "Population and score monitoring "
                        "remain valid, but outcome-based "
                        "performance metrics are deferred "
                        "until the vintage matures."
                    ),
                ]
            )

            continue

        calibration_gap = abs(
            portfolio_row[
                "calibration_gap"
            ]
        )

        if (
            calibration_gap
            >= CALIBRATION_GAP_ALERT
        ):
            alerts.append(
                [
                    quarter,
                    "Calibration drift",
                    "absolute_calibration_gap",
                    calibration_gap,
                    CALIBRATION_GAP_ALERT,
                    "HIGH",
                    (
                        "Review probability "
                        "calibration."
                    ),
                ]
            )

        performance_row = performance[
            performance["quarter"].eq(
                quarter
            )
        ].iloc[0]

        auc_drop = (
            reference_performance["roc_auc"]
            - performance_row["roc_auc"]
        )

        if (
            pd.notna(auc_drop)
            and auc_drop >= AUC_DROP_ALERT
        ):
            alerts.append(
                [
                    quarter,
                    "Performance degradation",
                    "roc_auc_drop",
                    auc_drop,
                    AUC_DROP_ALERT,
                    "CRITICAL",
                    (
                        "Review model "
                        "discrimination."
                    ),
                ]
            )

        reference_brier = (
            reference_performance[
                "brier_score"
            ]
        )

        current_brier = (
            performance_row[
                "brier_score"
            ]
        )

        if (
            pd.notna(reference_brier)
            and reference_brier > 0
            and pd.notna(current_brier)
        ):
            brier_increase = (
                current_brier
                - reference_brier
            ) / reference_brier

            if (
                brier_increase
                >= BRIER_INCREASE_ALERT
            ):
                alerts.append(
                    [
                        quarter,
                        "Performance degradation",
                        "brier_relative_increase",
                        brier_increase,
                        BRIER_INCREASE_ALERT,
                        "HIGH",
                        (
                            "Review model probability "
                            "accuracy."
                        ),
                    ]
                )

    columns = [
        "quarter",
        "alert_type",
        "metric",
        "observed_value",
        "threshold",
        "severity",
        "recommended_action",
    ]

    return pd.DataFrame(
        alerts,
        columns=columns,
    )


def build_recommendations(
    alerts: pd.DataFrame,
    score_psi_table: pd.DataFrame,
    feature_psi_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for quarter in score_psi_table[
        "quarter"
    ]:
        quarter_alerts = alerts[
            alerts["quarter"].eq(quarter)
        ]

        critical = (
            quarter_alerts["severity"]
            .eq("CRITICAL")
            .sum()
        )

        high = (
            quarter_alerts["severity"]
            .eq("HIGH")
            .sum()
        )

        watch = (
            quarter_alerts["severity"]
            .eq("WATCH")
            .sum()
        )

        provisional = (
            quarter_alerts["severity"]
            .eq("PROVISIONAL")
            .sum()
        )

        distinct_feature_alerts = (
            feature_psi_table.loc[
                feature_psi_table[
                    "quarter"
                ].eq(quarter)
                & feature_psi_table[
                    "psi"
                ].ge(PSI_ALERT),
                "original_feature",
            ]
            .nunique()
        )

        if critical > 0 or high >= 2:
            recommendation = "RETRAINING REVIEW"

            rationale = (
                "Material drift or degradation in a mature "
                "vintage requires model review."
            )

        elif high > 0 or watch > 0:
            recommendation = "INVESTIGATE"

            rationale = (
                "Moderate drift requires root-cause "
                "investigation."
            )

        elif provisional > 0:
            recommendation = "PROVISIONAL"

            rationale = (
                "Outcome-based performance is deferred because "
                "the loan vintage has not sufficiently matured."
            )

        else:
            recommendation = "STABLE"

            rationale = (
                "No material monitoring threshold "
                "was breached."
            )

        rows.append(
            {
                "quarter": quarter,
                "distinct_alert_feature_count": (
                    distinct_feature_alerts
                ),
                "critical_alerts": critical,
                "high_alerts": high,
                "watch_alerts": watch,
                "provisional_alerts": provisional,
                "recommendation": recommendation,
                "rationale": rationale,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def plot_score_psi(
    table: pd.DataFrame,
    path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.plot(
        table["quarter"],
        table["score_psi"],
        marker="o",
    )

    plt.axhline(
        PSI_WATCH,
        linestyle="--",
        label="Watch: 0.10",
    )

    plt.axhline(
        PSI_ALERT,
        linestyle="--",
        label="Alert: 0.25",
    )

    plt.title("Quarterly Model Score PSI")
    plt.xlabel("Quarter")
    plt.ylabel("PSI")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()

    save_plot(path)


def plot_feature_heatmap(
    table: pd.DataFrame,
    path: Path,
) -> None:
    concept_table = (
        table.groupby(
            [
                "quarter",
                "original_feature",
            ],
            as_index=False,
        )["psi"]
        .max()
    )

    top_features = (
        concept_table.groupby(
            "original_feature"
        )["psi"]
        .max()
        .nlargest(20)
        .index
    )

    pivot = (
        concept_table[
            concept_table[
                "original_feature"
            ].isin(top_features)
        ]
        .pivot(
            index="original_feature",
            columns="quarter",
            values="psi",
        )
        .fillna(0)
    )

    plt.figure(figsize=(12, 8))

    image = plt.imshow(
        pivot.values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="PSI",
    )

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
        rotation=45,
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index,
    )

    plt.title(
        "Top 20 Drifted Business Features"
    )

    plt.xlabel("Quarter")
    plt.ylabel("Original feature")

    save_plot(path)


def plot_portfolio_metrics(
    table: pd.DataFrame,
    path: Path,
) -> None:
    current = table[
        ~table["quarter"].eq(
            REFERENCE_LABEL
        )
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        current["quarter"],
        current["approval_rate"],
        marker="o",
        label="Approval rate",
    )

    matured = current[
        current["performance_eligible"]
    ]

    if not matured.empty:
        plt.plot(
            matured["quarter"],
            matured[
                "observed_default_rate"
            ],
            marker="o",
            label="Observed default rate",
        )

    plt.title(
        "Quarterly Approval and Matured Default Rates"
    )

    plt.xlabel("Quarter")
    plt.ylabel("Rate")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()

    save_plot(path)


def plot_calibration(
    table: pd.DataFrame,
    path: Path,
) -> None:
    current = table[
        ~table["quarter"].eq(
            REFERENCE_LABEL
        )
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        current["quarter"],
        current["average_pd"],
        marker="o",
        label="Average predicted PD",
    )

    matured = current[
        current["performance_eligible"]
    ]

    if not matured.empty:
        plt.plot(
            matured["quarter"],
            matured[
                "observed_default_rate"
            ],
            marker="o",
            label="Observed default rate",
        )

    plt.title(
        "Predicted PD versus Matured Observed Default Rate"
    )

    plt.xlabel("Quarter")
    plt.ylabel("Rate")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()

    save_plot(path)


def plot_model_performance(
    table: pd.DataFrame,
    path: Path,
) -> None:
    current = table[
        ~table["quarter"].eq(
            REFERENCE_LABEL
        )
        & table["performance_eligible"]
    ]

    if current.empty:
        logger.warning(
            "No mature monitoring quarters available "
            "for performance plot."
        )
        return

    plt.figure(figsize=(10, 6))

    for metric in [
        "roc_auc",
        "ks_statistic",
        "brier_score",
    ]:
        plt.plot(
            current["quarter"],
            current[metric],
            marker="o",
            label=metric.replace(
                "_",
                " ",
            ).title(),
        )

    plt.title(
        "Quarterly Model Performance "
        "for Mature Vintages"
    )

    plt.xlabel("Quarter")
    plt.ylabel("Metric value")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()

    save_plot(path)


def plot_risk_distribution(
    table: pd.DataFrame,
    path: Path,
) -> None:
    pivot = (
        table.pivot(
            index="quarter",
            columns="risk_tier",
            values="share",
        )
        .reindex(
            columns=RISK_LABELS,
            fill_value=0,
        )
    )

    pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 7),
    )

    plt.title(
        "Quarterly Risk-Tier Distribution"
    )

    plt.xlabel("Quarter")
    plt.ylabel("Population share")
    plt.xticks(rotation=45)

    plt.legend(
        title="Risk tier",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_plot(path)


# =============================================================================
# Report
# =============================================================================

def write_report(
    path: Path,
    reference: pd.DataFrame,
    monitoring: pd.DataFrame,
    threshold: float,
    feature_summary: pd.DataFrame,
    score_psi: pd.DataFrame,
    portfolio: pd.DataFrame,
    performance: pd.DataFrame,
    risk_psi: pd.DataFrame,
    alerts: pd.DataFrame,
    recommendations: pd.DataFrame,
    business_feature_count: int,
    model_feature_count: int,
    total_model_features: int,
) -> None:
    recommendation_values = set(
        recommendations["recommendation"]
    )

    if "RETRAINING REVIEW" in recommendation_values:
        overall = "RETRAINING REVIEW"

    elif "INVESTIGATE" in recommendation_values:
        overall = "INVESTIGATE"

    elif "PROVISIONAL" in recommendation_values:
        overall = (
            "STABLE WITH PROVISIONAL VINTAGES"
        )

    else:
        overall = "STABLE"

    top_features = feature_summary[
        [
            "original_feature",
            "maximum_psi",
            "average_psi",
            "quarters_in_alert",
            "monitored_columns",
            "latest_psi",
            "latest_status",
        ]
    ].head(20)

    monitoring_portfolio = portfolio[
        ~portfolio["quarter"].eq(
            REFERENCE_LABEL
        )
    ]

    monitoring_performance = performance[
        ~performance["quarter"].eq(
            REFERENCE_LABEL
        )
    ]

    mature_quarters = monitoring_performance[
        monitoring_performance[
            "performance_eligible"
        ]
    ]

    immature_quarters = monitoring_performance[
        ~monitoring_performance[
            "performance_eligible"
        ]
    ]

    report = f"""# Phase 8 Drift Monitoring Report

## Executive Summary

Phase 8 monitors applicant-population changes, model-score movement, lending
decisions, risk-tier migration, calibration and model discrimination.

- Overall recommendation: **{overall}**
- Decision threshold: **{threshold:.4f}**
- Reference population rows: **{len(reference):,}**
- Monitoring population rows: **{len(monitoring):,}**
- Monitoring quarters: **{monitoring["quarter"].nunique()}**
- Mature performance quarters: **{len(mature_quarters)}**
- Provisional performance quarters: **{len(immature_quarters)}**
- Core business features monitored: **{business_feature_count}**
- Encoded model features monitored: **{model_feature_count} of {total_model_features}**
- Alerts generated: **{len(alerts)}**

## Monitoring Windows

- Reference: **{reference["issue_date"].min().date()} to {reference["issue_date"].max().date()}**
- Monitoring: **{monitoring["issue_date"].min().date()} to {monitoring["issue_date"].max().date()}**
- Performance maturity cutoff: **{PERFORMANCE_CUTOFF_DATE.date()}**

## Outcome Maturity Policy

Population, feature, score, approval-rate and risk-tier monitoring are
performed for every quarter.

Observed default rate, calibration, ROC-AUC, Gini, KS and Brier score are
reported only for sufficiently matured loan vintages. Loans issued after
**{PERFORMANCE_CUTOFF_DATE.date()}** are marked as immature and excluded from
outcome-based performance alerts.

This prevents active or insufficiently observed loans from being incorrectly
treated as non-defaults.

## PSI Thresholds

| PSI | Status | Interpretation |
|---:|---|---|
| Below 0.10 | Stable | Little or no meaningful distribution change |
| 0.10 to 0.25 | Watch | Moderate population change |
| 0.25 or above | Alert | Significant population change |

## Model Score Drift

{markdown_table(score_psi)}

## Most Drifted Business Features

Raw and transformed versions of the same variable are grouped into one
underlying business feature for alert counting.

{markdown_table(top_features)}

## Quarterly Portfolio Metrics

{markdown_table(monitoring_portfolio)}

## Quarterly Model Performance

Performance metrics are shown only for sufficiently matured vintages.

{markdown_table(monitoring_performance)}

## Risk-Tier Drift

{markdown_table(risk_psi)}

## Monitoring Alerts

{markdown_table(alerts, 50)}

## Quarterly Recommendations

{markdown_table(recommendations)}

## Final Recommendation

**{overall}**

A PSI breach is a review trigger, not automatic proof that a model must be
retrained. Drift may arise from economic conditions, applicant acquisition,
credit-policy changes, seasonality, data-pipeline changes or genuine model
degradation.

Performance alerts are generated only for matured vintages. Recent immature
vintages retain population and policy monitoring but receive provisional
outcome status.

## Limitations

This is a retrospective monitoring simulation using Lending Club data. In a
production lending environment, the maturity cutoff should be derived from the
actual observation date, loan term, outcome definition and model-governance
policy.

## Saved Outputs

- `data/processed/phase8_monitoring/feature_psi_by_quarter.csv`
- `data/processed/phase8_monitoring/feature_psi_summary.csv`
- `data/processed/phase8_monitoring/score_psi_by_quarter.csv`
- `data/processed/phase8_monitoring/risk_tier_psi_by_quarter.csv`
- `data/processed/phase8_monitoring/risk_tier_distribution.csv`
- `data/processed/phase8_monitoring/quarterly_portfolio_metrics.csv`
- `data/processed/phase8_monitoring/quarterly_model_performance.csv`
- `data/processed/phase8_monitoring/drift_alerts.csv`
- `data/processed/phase8_monitoring/quarterly_monitoring_summary.csv`
- `data/processed/phase8_monitoring/monitoring_metadata.json`
"""

    path.write_text(
        report,
        encoding="utf-8",
    )

    logger.info(
        "Wrote report: %s",
        path,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    config = load_config()

    processed_dir = resolve_path(
        config["paths"]["processed_dir"]
    )

    monitoring_dir = (
        processed_dir
        / "phase8_monitoring"
    )

    reports_dir = resolve_path("reports")
    figures_dir = reports_dir / "figures"

    make_dir(monitoring_dir)
    make_dir(reports_dir)
    make_dir(figures_dir)

    logger.info(
        "Loading validation reference and "
        "out-of-time monitoring populations..."
    )

    inputs = load_inputs(processed_dir)

    reference = add_policy_fields(
        inputs["validation_population"],
        inputs["threshold"],
    )

    monitoring = add_policy_fields(
        inputs["test_population"],
        inputs["threshold"],
    )

    x_valid = inputs["x_valid"]
    x_test = inputs["x_test"]
    threshold = float(inputs["threshold"])

    business_features = [
        feature
        for feature in CORE_FEATURES
        if feature in reference.columns
        and feature in monitoring.columns
    ]

    logger.info(
        "Computing PSI for %d core business features...",
        len(business_features),
    )

    business_psi = calculate_feature_psi(
        reference=reference,
        monitoring=monitoring,
        quarters=monitoring["quarter"],
        feature_names=business_features,
        monitoring_level="business_feature",
    )

    model_features = select_model_features(
        x_valid,
        processed_dir,
    )

    logger.info(
        "Computing PSI for %d selected model features "
        "from %d encoded features...",
        len(model_features),
        x_valid.shape[1],
    )

    model_psi = calculate_feature_psi(
        reference=x_valid,
        monitoring=x_test,
        quarters=monitoring["quarter"],
        feature_names=model_features,
        monitoring_level="model_input",
    )

    feature_psi_table = pd.concat(
        [
            business_psi,
            model_psi,
        ],
        ignore_index=True,
    )

    raw_feature_names = list(
        reference.columns
    )

    feature_psi_table[
        "original_feature"
    ] = feature_psi_table[
        "feature"
    ].apply(
        lambda feature: original_feature_name(
            feature,
            raw_feature_names,
        )
    )

    feature_summary = summarize_feature_psi(
        feature_psi_table
    )

    logger.info(
        "Computing score and risk-tier PSI..."
    )

    score_psi = calculate_score_psi(
        reference,
        monitoring,
    )

    risk_psi = calculate_risk_tier_psi(
        reference,
        monitoring,
    )

    risk_distribution = (
        calculate_risk_tier_distribution(
            reference,
            monitoring,
        )
    )

    logger.info(
        "Computing maturity-aware quarterly "
        "portfolio and model metrics..."
    )

    portfolio, performance = (
        calculate_quarterly_metrics(
            reference,
            monitoring,
        )
    )

    logger.info(
        "Generating alerts and recommendations..."
    )

    alerts = generate_alerts(
        feature_psi_table,
        score_psi,
        risk_psi,
        portfolio,
        performance,
    )

    recommendations = build_recommendations(
        alerts,
        score_psi,
        feature_psi_table,
    )

    logger.info(
        "Saving Phase 8 tables..."
    )

    tables = {
        "feature_psi_by_quarter.csv": (
            feature_psi_table
        ),
        "feature_psi_summary.csv": (
            feature_summary
        ),
        "score_psi_by_quarter.csv": (
            score_psi
        ),
        "risk_tier_psi_by_quarter.csv": (
            risk_psi
        ),
        "risk_tier_distribution.csv": (
            risk_distribution
        ),
        "quarterly_portfolio_metrics.csv": (
            portfolio
        ),
        "quarterly_model_performance.csv": (
            performance
        ),
        "drift_alerts.csv": alerts,
        "quarterly_monitoring_summary.csv": (
            recommendations
        ),
    }

    for filename, table in tables.items():
        table.to_csv(
            monitoring_dir / filename,
            index=False,
        )

    metadata = {
        "reference_period": {
            "start": str(
                reference[
                    "issue_date"
                ].min()
            ),
            "end": str(
                reference[
                    "issue_date"
                ].max()
            ),
        },
        "monitoring_period": {
            "start": str(
                monitoring[
                    "issue_date"
                ].min()
            ),
            "end": str(
                monitoring[
                    "issue_date"
                ].max()
            ),
        },
        "performance_cutoff_date": str(
            PERFORMANCE_CUTOFF_DATE.date()
        ),
        "monitoring_quarters": sorted(
            monitoring["quarter"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "mature_monitoring_quarters": (
            performance.loc[
                ~performance["quarter"].eq(
                    REFERENCE_LABEL
                )
                & performance[
                    "performance_eligible"
                ],
                "quarter",
            ].tolist()
        ),
        "provisional_monitoring_quarters": (
            performance.loc[
                ~performance["quarter"].eq(
                    REFERENCE_LABEL
                )
                & ~performance[
                    "performance_eligible"
                ],
                "quarter",
            ].tolist()
        ),
        "decision_threshold": threshold,
        "psi_watch_threshold": PSI_WATCH,
        "psi_alert_threshold": PSI_ALERT,
        "business_features": business_features,
        "selected_model_features": model_features,
        "total_encoded_features": (
            x_valid.shape[1]
        ),
    }

    with open(
        monitoring_dir
        / "monitoring_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    logger.info(
        "Creating Phase 8 figures..."
    )

    plot_score_psi(
        score_psi,
        figures_dir
        / "phase8_score_psi_over_time.png",
    )

    plot_feature_heatmap(
        feature_psi_table,
        figures_dir
        / "phase8_top_feature_psi_heatmap.png",
    )

    plot_portfolio_metrics(
        portfolio,
        figures_dir
        / "phase8_approval_default_rate_over_time.png",
    )

    plot_calibration(
        portfolio,
        figures_dir
        / "phase8_average_pd_vs_observed_default.png",
    )

    plot_model_performance(
        performance,
        figures_dir
        / "phase8_model_performance_over_time.png",
    )

    plot_risk_distribution(
        risk_distribution,
        figures_dir
        / "phase8_risk_tier_distribution.png",
    )

    write_report(
        path=(
            reports_dir
            / "phase8_drift_monitoring_report.md"
        ),
        reference=reference,
        monitoring=monitoring,
        threshold=threshold,
        feature_summary=feature_summary,
        score_psi=score_psi,
        portfolio=portfolio,
        performance=performance,
        risk_psi=risk_psi,
        alerts=alerts,
        recommendations=recommendations,
        business_feature_count=len(
            business_features
        ),
        model_feature_count=len(
            model_features
        ),
        total_model_features=(
            x_valid.shape[1]
        ),
    )

    logger.info(
        "drift_monitoring.py complete."
    )


if __name__ == "__main__":
    main()