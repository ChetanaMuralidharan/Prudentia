"""
src/feature_engineering.py

Phase 3 — Feature Engineering for Credit Risk Modeling

Purpose:
- Build model-ready feature matrices from Phase 1 cleaned data.
- Preserve time-aware train / validation / test splits.
- Encode categorical variables safely.
- Impute missing values using training-only statistics.
- Scale numeric features for logistic regression.
- Save reusable preprocessing artifacts.

Run from project root:
    python src/feature_engineering.py
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import load_config, resolve_path, get_logger

logger = get_logger("feature_engineering")


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def remove_non_model_columns(df: pd.DataFrame, target_col: str, date_col: str) -> pd.DataFrame:
    """
    Remove columns that should not enter the model matrix.
    """
    drop_cols = [
        target_col,
        date_col,
        "loan_status",
        "earliest_cr_line",
        "issue_d",
    ]

    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=existing_drop_cols)


def add_credit_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable borrower-risk features.

    These features are intentionally simple and explainable because this project
    is a credit risk decisioning engine, not only a predictive ML exercise.
    """
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

    if "delinq_2yrs" in df.columns:
        df["has_recent_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)

    if "pub_rec" in df.columns:
        df["has_public_record"] = (df["pub_rec"] > 0).astype(int)

    if "mort_acc" in df.columns:
        df["has_mortgage_account"] = (df["mort_acc"] > 0).astype(int)

    if "inq_last_6mths" in df.columns:
        df["has_recent_inquiry"] = (df["inq_last_6mths"] > 0).astype(int)

    if "dti" in df.columns:
        df["high_dti_flag"] = (df["dti"] >= 30).astype(int)

    if "revol_util" in df.columns:
        df["high_revol_util_flag"] = (df["revol_util"] >= 80).astype(int)

    return df


def cap_extreme_values(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    """
    Cap numeric outliers using training-only 1st and 99th percentiles.
    This avoids leakage from validation/test distributions.
    """
    train = train.copy()
    valid = valid.copy()
    test = test.copy()

    numeric_cols = train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    caps = {}

    for col in numeric_cols:
        lower = train[col].quantile(0.01)
        upper = train[col].quantile(0.99)

        if pd.notna(lower) and pd.notna(upper) and lower < upper:
            train[col] = train[col].clip(lower, upper)
            valid[col] = valid[col].clip(lower, upper)
            test[col] = test[col].clip(lower, upper)

            caps[col] = {
                "lower_p01": float(lower),
                "upper_p99": float(upper),
            }

    return train, valid, test, caps


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def save_feature_names(preprocessor: ColumnTransformer, output_path: Path) -> None:
    feature_names = preprocessor.get_feature_names_out().tolist()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(f"Saved feature names: {output_path}")


def write_feature_report(
    output_path: Path,
    X_train: pd.DataFrame,
    X_train_encoded,
    numeric_features: list[str],
    categorical_features: list[str],
    caps: dict,
) -> None:
    lines = []

    lines.append("# Phase 3 Feature Engineering Report\n")

    lines.append("## Objective\n")
    lines.append(
        "Phase 3 converts the cleaned Lending Club dataset into model-ready training, "
        "validation, and test matrices while preserving the out-of-time validation design.\n"
    )

    lines.append("## Feature Engineering Strategy\n")
    lines.append(
        "The feature engineering process focuses on explainable credit-risk variables. "
        "Rather than creating opaque transformations, the pipeline adds borrower and loan-level "
        "risk indicators such as loan-to-income ratio, revolving-balance-to-income ratio, "
        "high-DTI flag, high-utilization flag, delinquency flag, and public-record flag.\n"
    )

    lines.append("## Leakage Control\n")
    lines.append(
        "All preprocessing statistics are fit on the training set only. Median imputation values, "
        "categorical encodings, scaling parameters, and outlier caps are learned from the training "
        "period and then applied unchanged to validation and test data.\n"
    )

    lines.append("## Model Input Summary\n")
    lines.append(f"- Raw model input columns before encoding: **{X_train.shape[1]}**")
    lines.append(f"- Numeric input columns: **{len(numeric_features)}**")
    lines.append(f"- Categorical input columns: **{len(categorical_features)}**")
    lines.append(f"- Encoded model features: **{X_train_encoded.shape[1]}**\n")

    lines.append("## Engineered Features Added\n")
    engineered_features = [
        "fico_avg",
        "loan_to_income",
        "revol_bal_to_income",
        "open_account_ratio",
        "bankcard_available_ratio",
        "has_recent_delinquency",
        "has_public_record",
        "has_mortgage_account",
        "has_recent_inquiry",
        "high_dti_flag",
        "high_revol_util_flag",
    ]

    for feature in engineered_features:
        if feature in X_train.columns:
            lines.append(f"- `{feature}`")

    lines.append("\n## Outlier Treatment\n")
    lines.append(
        "Numeric variables are capped using the 1st and 99th percentile values from the training set. "
        "This reduces the impact of extreme values while avoiding validation/test leakage.\n"
    )
    lines.append(f"- Number of capped numeric features: **{len(caps)}**\n")

    lines.append("## Encoding and Imputation\n")
    lines.append("- Numeric features: median imputation + standard scaling")
    lines.append("- Categorical features: most-frequent imputation + one-hot encoding")
    lines.append("- Unknown validation/test categories: ignored safely during one-hot encoding\n")

    lines.append("## Saved Artifacts\n")
    lines.append("- `data/processed/X_train.parquet`")
    lines.append("- `data/processed/X_valid.parquet`")
    lines.append("- `data/processed/X_test.parquet`")
    lines.append("- `data/processed/y_train.parquet`")
    lines.append("- `data/processed/y_valid.parquet`")
    lines.append("- `data/processed/y_test.parquet`")
    lines.append("- `artifacts/preprocessor.joblib`")
    lines.append("- `artifacts/feature_names.json`")
    lines.append("- `artifacts/outlier_caps.json`\n")

    lines.append("## Phase 3 Completion Criteria\n")
    lines.append(
        "Phase 3 is complete when the processed feature matrices, target files, preprocessing "
        "pipeline, feature-name artifact, and feature-engineering report are all generated successfully.\n"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {output_path}")


def main() -> None:
    config = load_config()

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    reports_dir = resolve_path("reports")
    artifacts_dir = resolve_path("artifacts")

    make_dir(processed_dir)
    make_dir(reports_dir)
    make_dir(artifacts_dir)

    target_col = config["target"]["target_col"]
    date_col = config["split"]["date_col"]

    logger.info("Loading Phase 1 split datasets...")

    train = pd.read_parquet(processed_dir / "loan_train.parquet")
    valid = pd.read_parquet(processed_dir / "loan_valid.parquet")
    test = pd.read_parquet(processed_dir / "loan_test.parquet")

    logger.info("Adding credit-risk engineered features...")

    train = add_credit_risk_features(train)
    valid = add_credit_risk_features(valid)
    test = add_credit_risk_features(test)

    y_train = train[[target_col]].copy()
    y_valid = valid[[target_col]].copy()
    y_test = test[[target_col]].copy()

    X_train = remove_non_model_columns(train, target_col, date_col)
    X_valid = remove_non_model_columns(valid, target_col, date_col)
    X_test = remove_non_model_columns(test, target_col, date_col)

    logger.info("Applying training-only outlier caps...")

    X_train, X_valid, X_test, caps = cap_extreme_values(X_train, X_valid, X_test)

    logger.info("Building preprocessing pipeline...")

    preprocessor = build_preprocessor(X_train)

    numeric_features = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    logger.info("Fitting preprocessor on training data only...")

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_valid_encoded = preprocessor.transform(X_valid)
    X_test_encoded = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_valid_encoded = pd.DataFrame(X_valid_encoded, columns=feature_names, index=X_valid.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)

    logger.info("Saving model-ready datasets...")

    X_train_encoded.to_parquet(processed_dir / "X_train.parquet", index=False)
    X_valid_encoded.to_parquet(processed_dir / "X_valid.parquet", index=False)
    X_test_encoded.to_parquet(processed_dir / "X_test.parquet", index=False)

    y_train.to_parquet(processed_dir / "y_train.parquet", index=False)
    y_valid.to_parquet(processed_dir / "y_valid.parquet", index=False)
    y_test.to_parquet(processed_dir / "y_test.parquet", index=False)

    logger.info("Saving preprocessing artifacts...")

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

    with (artifacts_dir / "outlier_caps.json").open("w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2)

    save_feature_names(preprocessor, artifacts_dir / "feature_names.json")

    write_feature_report(
        output_path=reports_dir / "phase3_feature_engineering_report.md",
        X_train=X_train,
        X_train_encoded=X_train_encoded,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        caps=caps,
    )

    logger.info("Phase 3 feature engineering complete.")


if __name__ == "__main__":
    main()