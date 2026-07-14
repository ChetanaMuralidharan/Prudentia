"""
src/woe_iv.py

Phase 3B — Credit Risk Feature Engineering

Adds:
- Risk bins for numeric variables
- WOE encoding
- Information Value calculation
- Feature selection report
- WOE-transformed model-ready datasets

Run:
    python src/woe_iv.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from utils import load_config, resolve_path, get_logger

logger = get_logger("woe_iv")


EPS = 1e-6


NUMERIC_BIN_FEATURES = {
    "dti": [-np.inf, 10, 20, 30, 40, np.inf],
    "annual_inc": [-np.inf, 30000, 60000, 90000, 120000, 200000, np.inf],
    "loan_amnt": [-np.inf, 5000, 10000, 15000, 20000, 30000, np.inf],
    "revol_util": [-np.inf, 20, 40, 60, 80, 100, np.inf],
    "delinq_2yrs": [-np.inf, 0, 1, 2, 3, 5, np.inf],
    "pub_rec": [-np.inf, 0, 1, 2, 3, np.inf],
    "inq_last_6mths": [-np.inf, 0, 1, 2, 3, np.inf],
    "open_acc": [-np.inf, 5, 10, 15, 20, 30, np.inf],
    "total_acc": [-np.inf, 10, 20, 30, 40, 60, np.inf],
    "revol_bal": [-np.inf, 5000, 10000, 20000, 50000, np.inf],
    "loan_to_income": [-np.inf, 0.05, 0.10, 0.20, 0.30, 0.50, np.inf],
    "revol_bal_to_income": [-np.inf, 0.05, 0.10, 0.20, 0.40, 0.80, np.inf],
    "open_account_ratio": [-np.inf, 0.25, 0.50, 0.75, 1.0, np.inf],
    "bankcard_available_ratio": [-np.inf, 0.10, 0.25, 0.50, 0.75, np.inf],
}


CATEGORICAL_FEATURES = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "disbursement_method",
]


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_credit_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

    if "revol_bal" in df.columns and "annual_inc" in df.columns:
        df["revol_bal_to_income"] = df["revol_bal"] / df["annual_inc"].replace(0, np.nan)

    if "open_acc" in df.columns and "total_acc" in df.columns:
        df["open_account_ratio"] = df["open_acc"] / df["total_acc"].replace(0, np.nan)

    if "bc_open_to_buy" in df.columns and "total_bc_limit" in df.columns:
        df["bankcard_available_ratio"] = df["bc_open_to_buy"] / df["total_bc_limit"].replace(0, np.nan)

    return df


def create_binned_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col, bins in NUMERIC_BIN_FEATURES.items():
        if col in df.columns:
            out[col + "_bin"] = pd.cut(df[col], bins=bins, include_lowest=True).astype(str)
            out[col + "_bin"] = out[col + "_bin"].replace("nan", "Missing")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            out[col] = df[col].astype("object").fillna("Missing").astype(str)

    binary_features = [
        "has_recent_delinquency",
        "has_public_record",
        "has_mortgage_account",
        "has_recent_inquiry",
        "high_dti_flag",
        "high_revol_util_flag",
    ]

    for col in binary_features:
        if col in df.columns:
            out[col] = df[col].fillna(0).astype(int).astype(str)

    return out


def calculate_woe_iv_for_feature(x: pd.Series, y: pd.Series) -> tuple[pd.DataFrame, dict, float]:
    temp = pd.DataFrame({"feature_value": x.fillna("Missing").astype(str), "target": y})
    grouped = (
        temp.groupby("feature_value")
        .agg(total=("target", "size"), bad=("target", "sum"))
        .reset_index()
    )

    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    grouped["good_dist"] = (grouped["good"] + EPS) / (total_good + EPS)
    grouped["bad_dist"] = (grouped["bad"] + EPS) / (total_bad + EPS)

    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
    grouped["iv_component"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    iv = grouped["iv_component"].sum()

    mapping = dict(zip(grouped["feature_value"], grouped["woe"]))

    grouped["default_rate"] = grouped["bad"] / grouped["total"]
    grouped["iv_total"] = iv

    return grouped, mapping, iv


def fit_woe_encoder(X_train_binned: pd.DataFrame, y_train: pd.Series):
    woe_maps = {}
    iv_rows = []
    detail_tables = {}

    for col in X_train_binned.columns:
        detail, mapping, iv = calculate_woe_iv_for_feature(X_train_binned[col], y_train)
        woe_maps[col] = mapping
        detail_tables[col] = detail

        iv_rows.append(
            {
                "feature": col,
                "iv": iv,
                "predictive_strength": classify_iv(iv),
            }
        )

    iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False)
    return woe_maps, iv_table, detail_tables


def transform_woe(X_binned: pd.DataFrame, woe_maps: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=X_binned.index)

    for col in X_binned.columns:
        mapping = woe_maps.get(col, {})
        out[col + "_woe"] = X_binned[col].fillna("Missing").astype(str).map(mapping).fillna(0.0)

    return out


def classify_iv(iv: float) -> str:
    if iv < 0.02:
        return "Not useful"
    if iv < 0.10:
        return "Weak"
    if iv < 0.30:
        return "Medium"
    if iv < 0.50:
        return "Strong"
    return "Very strong"


def select_features_by_iv(iv_table: pd.DataFrame, min_iv: float = 0.02, max_iv: float = 0.50) -> list[str]:
    selected = iv_table[
        (iv_table["iv"] >= min_iv) &
        (iv_table["iv"] <= max_iv)
    ]["feature"].tolist()

    return [feature + "_woe" for feature in selected]


def write_report(
    output_path: Path,
    iv_table: pd.DataFrame,
    selected_features: list[str],
    X_train_woe: pd.DataFrame,
    X_valid_woe: pd.DataFrame,
    X_test_woe: pd.DataFrame,
) -> None:
    top_iv = iv_table.head(25).copy()

    lines = []
    lines.append("# Phase 3B WOE / IV Feature Engineering Report\n")

    lines.append("## Objective\n")
    lines.append(
        "This phase extends generic preprocessing into credit-risk-specific feature engineering. "
        "Features are binned into interpretable risk groups, transformed using Weight of Evidence, "
        "and evaluated using Information Value.\n"
    )

    lines.append("## Why WOE and IV Matter\n")
    lines.append(
        "Weight of Evidence transforms feature groups into log-risk values based on the relationship "
        "between good and bad loans. Information Value measures how useful a feature is for separating "
        "fully paid loans from charged-off/default loans.\n"
    )

    lines.append("## Leakage Control\n")
    lines.append(
        "WOE mappings and IV values are fit only on the training period. The same mappings are then "
        "applied to validation and test sets, preserving the time-aware modeling design.\n"
    )

    lines.append("## Feature Matrix Summary\n")
    lines.append(f"- WOE train shape: **{X_train_woe.shape[0]} rows × {X_train_woe.shape[1]} features**")
    lines.append(f"- WOE validation shape: **{X_valid_woe.shape[0]} rows × {X_valid_woe.shape[1]} features**")
    lines.append(f"- WOE test shape: **{X_test_woe.shape[0]} rows × {X_test_woe.shape[1]} features**")
    lines.append(f"- Selected features by IV: **{len(selected_features)}**\n")

    lines.append("## Top Features by Information Value\n")
    lines.append(top_iv.to_markdown(index=False))
    lines.append("\n")

    lines.append("## IV Interpretation Rules\n")
    lines.append("| IV Range | Interpretation |")
    lines.append("|---:|:---|")
    lines.append("| < 0.02 | Not useful |")
    lines.append("| 0.02 - 0.10 | Weak |")
    lines.append("| 0.10 - 0.30 | Medium |")
    lines.append("| 0.30 - 0.50 | Strong |")
    lines.append("| > 0.50 | Very strong / investigate for leakage |\n")

    lines.append("## Selected WOE Features\n")
    for feature in selected_features:
        lines.append(f"- `{feature}`")

    lines.append("\n## Modeling Recommendation\n")
    lines.append(
        "Use the WOE-transformed feature matrices for the first scorecard-style logistic regression model. "
        "Tree-based models can later be trained on the generic encoded features from Phase 3A for comparison.\n"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {output_path}")


def main() -> None:
    config = load_config()

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    reports_dir = resolve_path("reports")
    artifacts_dir = resolve_path("artifacts")
    woe_dir = processed_dir / "woe_iv_tables"

    make_dir(processed_dir)
    make_dir(reports_dir)
    make_dir(artifacts_dir)
    make_dir(woe_dir)

    target_col = config["target"]["target_col"]

    logger.info("Loading Phase 1 split datasets...")

    train = pd.read_parquet(processed_dir / "loan_train.parquet")
    valid = pd.read_parquet(processed_dir / "loan_valid.parquet")
    test = pd.read_parquet(processed_dir / "loan_test.parquet")

    logger.info("Adding explainable credit-risk features...")

    train = add_credit_risk_features(train)
    valid = add_credit_risk_features(valid)
    test = add_credit_risk_features(test)

    if "delinq_2yrs" in train.columns:
        train["has_recent_delinquency"] = (train["delinq_2yrs"] > 0).astype(int)
        valid["has_recent_delinquency"] = (valid["delinq_2yrs"] > 0).astype(int)
        test["has_recent_delinquency"] = (test["delinq_2yrs"] > 0).astype(int)

    if "pub_rec" in train.columns:
        train["has_public_record"] = (train["pub_rec"] > 0).astype(int)
        valid["has_public_record"] = (valid["pub_rec"] > 0).astype(int)
        test["has_public_record"] = (test["pub_rec"] > 0).astype(int)

    if "mort_acc" in train.columns:
        train["has_mortgage_account"] = (train["mort_acc"] > 0).astype(int)
        valid["has_mortgage_account"] = (valid["mort_acc"] > 0).astype(int)
        test["has_mortgage_account"] = (test["mort_acc"] > 0).astype(int)

    if "inq_last_6mths" in train.columns:
        train["has_recent_inquiry"] = (train["inq_last_6mths"] > 0).astype(int)
        valid["has_recent_inquiry"] = (valid["inq_last_6mths"] > 0).astype(int)
        test["has_recent_inquiry"] = (test["inq_last_6mths"] > 0).astype(int)

    if "dti" in train.columns:
        train["high_dti_flag"] = (train["dti"] >= 30).astype(int)
        valid["high_dti_flag"] = (valid["dti"] >= 30).astype(int)
        test["high_dti_flag"] = (test["dti"] >= 30).astype(int)

    if "revol_util" in train.columns:
        train["high_revol_util_flag"] = (train["revol_util"] >= 80).astype(int)
        valid["high_revol_util_flag"] = (valid["revol_util"] >= 80).astype(int)
        test["high_revol_util_flag"] = (test["revol_util"] >= 80).astype(int)

    y_train = train[target_col].astype(int)
    y_valid = valid[target_col].astype(int)
    y_test = test[target_col].astype(int)

    logger.info("Creating binned feature frames...")

    X_train_binned = create_binned_frame(train)
    X_valid_binned = create_binned_frame(valid)
    X_test_binned = create_binned_frame(test)

    logger.info("Fitting WOE encoder and calculating IV on training data...")

    woe_maps, iv_table, detail_tables = fit_woe_encoder(X_train_binned, y_train)

    logger.info("Transforming train, validation, and test using training WOE maps...")

    X_train_woe = transform_woe(X_train_binned, woe_maps)
    X_valid_woe = transform_woe(X_valid_binned, woe_maps)
    X_test_woe = transform_woe(X_test_binned, woe_maps)

    selected_features = select_features_by_iv(iv_table, min_iv=0.02, max_iv=0.50)

    if selected_features:
        X_train_woe_selected = X_train_woe[selected_features]
        X_valid_woe_selected = X_valid_woe[selected_features]
        X_test_woe_selected = X_test_woe[selected_features]
    else:
        logger.warning("No features selected by IV thresholds. Saving all WOE features.")
        X_train_woe_selected = X_train_woe
        X_valid_woe_selected = X_valid_woe
        X_test_woe_selected = X_test_woe

    logger.info("Saving WOE datasets...")

    X_train_woe.to_parquet(processed_dir / "X_train_woe.parquet", index=False)
    X_valid_woe.to_parquet(processed_dir / "X_valid_woe.parquet", index=False)
    X_test_woe.to_parquet(processed_dir / "X_test_woe.parquet", index=False)

    X_train_woe_selected.to_parquet(processed_dir / "X_train_woe_selected.parquet", index=False)
    X_valid_woe_selected.to_parquet(processed_dir / "X_valid_woe_selected.parquet", index=False)
    X_test_woe_selected.to_parquet(processed_dir / "X_test_woe_selected.parquet", index=False)

    pd.DataFrame({"target": y_train}).to_parquet(processed_dir / "y_train_woe.parquet", index=False)
    pd.DataFrame({"target": y_valid}).to_parquet(processed_dir / "y_valid_woe.parquet", index=False)
    pd.DataFrame({"target": y_test}).to_parquet(processed_dir / "y_test_woe.parquet", index=False)

    logger.info("Saving IV and WOE artifacts...")

    iv_table.to_csv(woe_dir / "information_value_summary.csv", index=False)

    for feature, table in detail_tables.items():
        safe_name = feature.replace("/", "_").replace(" ", "_")
        table.to_csv(woe_dir / f"woe_detail_{safe_name}.csv", index=False)

    with (artifacts_dir / "woe_maps.json").open("w", encoding="utf-8") as f:
        json.dump(woe_maps, f, indent=2)

    with (artifacts_dir / "selected_woe_features.json").open("w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)

    joblib.dump(
        {
            "woe_maps": woe_maps,
            "selected_features": selected_features,
            "numeric_bins": NUMERIC_BIN_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        artifacts_dir / "woe_encoder.joblib",
    )

    write_report(
        output_path=reports_dir / "phase3b_woe_iv_report.md",
        iv_table=iv_table,
        selected_features=selected_features,
        X_train_woe=X_train_woe,
        X_valid_woe=X_valid_woe,
        X_test_woe=X_test_woe,
    )

    logger.info("Phase 3B WOE / IV feature engineering complete.")


if __name__ == "__main__":
    main()