"""
src/train_models.py

Phase 4 — Credit Risk Model Training & Evaluation

Models:
1. WOE Logistic Regression on full WOE-selected dataset
2. Encoded Logistic Regression on stratified sample
3. Random Forest benchmark on stratified sample
4. XGBoost benchmark on stratified sample

Run:
    python src/train_models.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from xgboost import XGBClassifier

from utils import load_config, resolve_path, get_logger

logger = get_logger("train_models")

RANDOM_STATE = 42
ENCODED_LOGREG_SAMPLE_SIZE = 100_000
RANDOM_FOREST_SAMPLE_SIZE = 150_000
XGBOOST_SAMPLE_SIZE = 200_000


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path}")


def load_target(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "target" in df.columns:
        return df["target"].astype(int)
    return df.iloc[:, 0].astype(int)


def stratified_sample(X: pd.DataFrame, y: pd.Series, sample_size: int):
    if len(X) <= sample_size:
        return X, y

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_sample, y_sample


def evaluate_model(model, X, y, threshold=0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold": threshold,
    }


def find_best_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    thresholds = np.arange(0.05, 0.95, 0.01)
    rows = []

    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )

    table = pd.DataFrame(rows)
    best_row = table.sort_values("f1", ascending=False).iloc[0]

    return float(best_row["threshold"]), table


def plot_roc_curve(model_results: dict, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))

    for model_name, values in model_results.items():
        y = values["y_valid"]
        proba = values["valid_proba"]

        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)

        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curves")
    plt.legend()

    save_plot(output_path)


def plot_pr_curve(model_results: dict, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))

    for model_name, values in model_results.items():
        y = values["y_valid"]
        proba = values["valid_proba"]

        precision, recall, _ = precision_recall_curve(y, proba)
        pr_auc = average_precision_score(y, proba)

        plt.plot(recall, precision, label=f"{model_name} PR-AUC={pr_auc:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curves")
    plt.legend()

    save_plot(output_path)


def plot_threshold_curve(threshold_table: pd.DataFrame, model_name: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision")
    plt.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall")
    plt.plot(threshold_table["threshold"], threshold_table["f1"], label="F1")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric")
    plt.title(f"Threshold Tradeoff - {model_name}")
    plt.legend()

    save_plot(output_path)


def get_logistic_coefficients(model, feature_names: list[str], top_n: int = 30) -> pd.DataFrame:
    coef = model.coef_[0]

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "absolute_coefficient": np.abs(coef),
        }
    )

    return table.sort_values("absolute_coefficient", ascending=False).head(top_n)


def get_feature_importance(model, feature_names: list[str], top_n: int = 30) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    return table.sort_values("importance", ascending=False).head(top_n)


def write_model_report(
    output_path: Path,
    summary_table: pd.DataFrame,
    best_model_name: str,
    best_threshold: float,
) -> None:
    lines = []

    lines.append("# Phase 4 Model Training and Evaluation Report\n")

    lines.append("## Objective\n")
    lines.append(
        "Phase 4 trains and evaluates credit-risk classification models using the time-aware "
        "train, validation, and test split created in earlier phases.\n"
    )

    lines.append("## Models Trained\n")
    lines.append("- **WOE Logistic Regression**: scorecard-style interpretable model trained on full WOE-selected training data.")
    lines.append("- **Encoded Logistic Regression**: generic encoded baseline trained on a stratified sample for computational efficiency.")
    lines.append("- **Random Forest**: nonlinear benchmark trained on a stratified sample for computational efficiency.")
    lines.append("- **XGBoost**: gradient-boosted tree benchmark trained on a stratified sample using CPU-friendly histogram training.\n")

    lines.append("## Why Sampling Was Used\n")
    lines.append(
        "The encoded feature matrix is large because one-hot encoding expands categorical variables into many columns. "
        "To keep training practical on a local machine, the generic encoded logistic regression, random forest, "
        "and XGBoost benchmarks were trained on stratified samples while preserving the original default/non-default "
        "class balance. The WOE scorecard model was trained on the full training population.\n"
    )

    lines.append("## Evaluation Strategy\n")
    lines.append(
        "Models are trained on the training period, threshold-tuned on the validation period, "
        "and finally evaluated on the held-out test period. This simulates a realistic out-of-time "
        "credit-risk deployment setting.\n"
    )

    lines.append("## Model Performance Summary\n")
    lines.append(summary_table.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Selected Model\n")
    lines.append(f"- Best validation ROC-AUC model: **{best_model_name}**")
    lines.append(f"- Selected operating threshold: **{best_threshold:.2f}**\n")

    lines.append("## Business Interpretation\n")
    lines.append(
        "ROC-AUC measures ranking quality, while precision and recall describe the approval/rejection tradeoff "
        "at a chosen threshold. In credit risk, threshold choice should not rely on accuracy alone because "
        "false approvals and false rejections have different financial consequences.\n"
    )

    lines.append("## Phase 4 Completion Criteria\n")
    lines.append("- Models trained successfully")
    lines.append("- Validation and test metrics generated")
    lines.append("- ROC and precision-recall curves saved")
    lines.append("- Threshold analysis completed")
    lines.append("- Best model metadata saved\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {output_path}")


def main() -> None:
    config = load_config()

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    reports_dir = resolve_path("reports")
    artifacts_dir = resolve_path("artifacts")
    figures_dir = reports_dir / "figures"
    model_dir = artifacts_dir / "models"
    metrics_dir = processed_dir / "phase4_model_metrics"

    make_dir(reports_dir)
    make_dir(figures_dir)
    make_dir(artifacts_dir)
    make_dir(model_dir)
    make_dir(metrics_dir)

    logger.info("Loading Phase 3 WOE-selected datasets...")

    X_train_woe = pd.read_parquet(processed_dir / "X_train_woe_selected.parquet")
    X_valid_woe = pd.read_parquet(processed_dir / "X_valid_woe_selected.parquet")
    X_test_woe = pd.read_parquet(processed_dir / "X_test_woe_selected.parquet")

    y_train_woe = load_target(processed_dir / "y_train_woe.parquet")
    y_valid_woe = load_target(processed_dir / "y_valid_woe.parquet")
    y_test_woe = load_target(processed_dir / "y_test_woe.parquet")

    logger.info("Loading Phase 3 generic encoded datasets...")

    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    X_valid = pd.read_parquet(processed_dir / "X_valid.parquet")
    X_test = pd.read_parquet(processed_dir / "X_test.parquet")

    y_train = load_target(processed_dir / "y_train.parquet")
    y_valid = load_target(processed_dir / "y_valid.parquet")
    y_test = load_target(processed_dir / "y_test.parquet")

    logger.info("Creating stratified training samples for heavier benchmark models...")

    X_train_encoded_lr, y_train_encoded_lr = stratified_sample(
        X_train,
        y_train,
        ENCODED_LOGREG_SAMPLE_SIZE,
    )

    X_train_rf, y_train_rf = stratified_sample(
        X_train,
        y_train,
        RANDOM_FOREST_SAMPLE_SIZE,
    )

    X_train_xgb, y_train_xgb = stratified_sample(
        X_train,
        y_train,
        XGBOOST_SAMPLE_SIZE,
    )

    X_train_xgb = sanitize_xgboost_columns(X_train_xgb)
    X_valid_xgb = sanitize_xgboost_columns(X_valid)
    X_test_xgb = sanitize_xgboost_columns(X_test)

    logger.info(f"Encoded Logistic Regression sample size: {len(X_train_encoded_lr):,}")
    logger.info(f"Random Forest sample size: {len(X_train_rf):,}")
    logger.info(f"XGBoost sample size: {len(X_train_xgb):,}")

    logger.info("Training WOE Logistic Regression on full training data...")

    woe_logreg = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    woe_logreg.fit(X_train_woe, y_train_woe)

    logger.info("Training encoded Logistic Regression on stratified sample...")

    encoded_logreg = LogisticRegression(
        max_iter=100,
        class_weight="balanced",
        solver="saga",
        penalty="l2",
        C=0.5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    encoded_logreg.fit(X_train_encoded_lr, y_train_encoded_lr)

    logger.info("Training Random Forest benchmark on stratified sample...")

    rf = RandomForestClassifier(
        n_estimators=70,
        max_depth=12,
        min_samples_leaf=100,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rf.fit(X_train_rf, y_train_rf)

    logger.info("Training XGBoost benchmark on stratified sample...")

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train_xgb, y_train_xgb)

    models = {
        "woe_logistic_regression": {
            "model": woe_logreg,
            "X_valid": X_valid_woe,
            "y_valid": y_valid_woe,
            "X_test": X_test_woe,
            "y_test": y_test_woe,
            "feature_names": X_train_woe.columns.tolist(),
            "model_type": "logistic",
        },
        "encoded_logistic_regression_sampled": {
            "model": encoded_logreg,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X_train.columns.tolist(),
            "model_type": "logistic",
        },
        "random_forest_sampled": {
            "model": rf,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": X_train.columns.tolist(),
            "model_type": "tree",
        },
        "xgboost_sampled": {
            "model": xgb,
            "X_valid": X_valid_xgb,
            "y_valid": y_valid,
            "X_test": X_test_xgb,
            "y_test": y_test,
            "feature_names": X_train_xgb.columns.tolist(),
            "model_type": "tree",
        },
    }

    logger.info("Evaluating models...")

    summary_rows = []
    model_results = {}

    for model_name, values in models.items():
        model = values["model"]

        valid_proba = model.predict_proba(values["X_valid"])[:, 1]
        best_threshold, threshold_table = find_best_threshold(values["y_valid"], valid_proba)

        valid_metrics = evaluate_model(
            model,
            values["X_valid"],
            values["y_valid"],
            threshold=best_threshold,
        )

        test_metrics = evaluate_model(
            model,
            values["X_test"],
            values["y_test"],
            threshold=best_threshold,
        )

        threshold_table.to_csv(
            metrics_dir / f"{model_name}_threshold_table.csv",
            index=False,
        )

        row = {"model": model_name}

        for key, value in valid_metrics.items():
            row[f"valid_{key}"] = value

        for key, value in test_metrics.items():
            row[f"test_{key}"] = value

        summary_rows.append(row)

        model_results[model_name] = {
            "y_valid": values["y_valid"],
            "valid_proba": valid_proba,
        }

        plot_threshold_curve(
            threshold_table,
            model_name,
            figures_dir / f"phase4_threshold_curve_{model_name}.png",
        )

        joblib.dump(model, model_dir / f"{model_name}.joblib")

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(metrics_dir / "model_performance_summary.csv", index=False)

    best_row = summary_table.sort_values("valid_roc_auc", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_threshold = float(best_row["valid_threshold"])

    logger.info(f"Best model by validation ROC-AUC: {best_model_name}")

    best_model = models[best_model_name]["model"]
    joblib.dump(best_model, model_dir / "best_model.joblib")

    with (model_dir / "best_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": best_model_name,
                "selection_metric": "validation_roc_auc",
                "best_threshold": best_threshold,
                "encoded_logistic_sample_size": ENCODED_LOGREG_SAMPLE_SIZE,
                "random_forest_sample_size": RANDOM_FOREST_SAMPLE_SIZE,
                "xgboost_sample_size": XGBOOST_SAMPLE_SIZE,
            },
            f,
            indent=2,
        )

    logger.info("Saving model interpretation artifacts...")

    woe_coef = get_logistic_coefficients(
        woe_logreg,
        X_train_woe.columns.tolist(),
    )
    woe_coef.to_csv(metrics_dir / "woe_logistic_coefficients.csv", index=False)

    encoded_coef = get_logistic_coefficients(
        encoded_logreg,
        X_train.columns.tolist(),
    )
    encoded_coef.to_csv(metrics_dir / "encoded_logistic_coefficients.csv", index=False)

    rf_importance = get_feature_importance(
        rf,
        X_train.columns.tolist(),
    )
    rf_importance.to_csv(metrics_dir / "random_forest_feature_importance.csv", index=False)

    xgb_importance = get_feature_importance(
        xgb,
        X_train_xgb.columns.tolist(),
    )
    xgb_importance.to_csv(metrics_dir / "xgboost_feature_importance.csv", index=False)

    plot_roc_curve(
        model_results,
        figures_dir / "phase4_validation_roc_curves.png",
    )

    plot_pr_curve(
        model_results,
        figures_dir / "phase4_validation_precision_recall_curves.png",
    )

    write_model_report(
        output_path=reports_dir / "phase4_model_training_report.md",
        summary_table=summary_table,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
    )

    logger.info("Phase 4 model training complete.")

def sanitize_xgboost_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X.columns = (
        X.columns.astype(str)
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "lt_", regex=False)
        .str.replace(">", "gt_", regex=False)
    )
    return X

if __name__ == "__main__":
    main()