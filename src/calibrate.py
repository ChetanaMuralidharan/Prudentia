"""
src/calibrate.py

Phase 5 — Probability Calibration

Purpose:
- Load the best Phase 4 model.
- Compare raw probabilities vs sigmoid calibration vs isotonic calibration.
- Evaluate calibration with Brier score.
- Save calibrated probabilities for Phase 6 decisioning.

Run:
    python src/calibrate.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

from utils import load_config, resolve_path, get_logger

logger = get_logger("calibrate")


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


def load_model_inputs(processed_dir: Path, best_model_name: str):
    if best_model_name == "woe_logistic_regression":
        X_valid = pd.read_parquet(processed_dir / "X_valid_woe_selected.parquet")
        X_test = pd.read_parquet(processed_dir / "X_test_woe_selected.parquet")
        y_valid = load_target(processed_dir / "y_valid_woe.parquet")
        y_test = load_target(processed_dir / "y_test_woe.parquet")
    else:
        X_valid = pd.read_parquet(processed_dir / "X_valid.parquet")
        X_test = pd.read_parquet(processed_dir / "X_test.parquet")
        y_valid = load_target(processed_dir / "y_valid.parquet")
        y_test = load_target(processed_dir / "y_test.parquet")

        if best_model_name == "xgboost_sampled":
            X_valid = sanitize_xgboost_columns(X_valid)
            X_test = sanitize_xgboost_columns(X_test)

    return X_valid, X_test, y_valid, y_test


def evaluate_probabilities(y_true: pd.Series, proba: np.ndarray) -> dict:
    return {
        "brier_score": brier_score_loss(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "mean_predicted_pd": float(np.mean(proba)),
        "observed_default_rate": float(np.mean(y_true)),
    }


def plot_reliability_curves(
    y_valid,
    raw_valid_proba,
    sigmoid_valid_proba,
    isotonic_valid_proba,
    output_path: Path,
):
    plt.figure(figsize=(8, 6))

    for name, proba in {
        "Raw": raw_valid_proba,
        "Sigmoid": sigmoid_valid_proba,
        "Isotonic": isotonic_valid_proba,
    }.items():
        frac_pos, mean_pred = calibration_curve(
            y_valid,
            proba,
            n_bins=10,
            strategy="quantile",
        )
        plt.plot(mean_pred, frac_pos, marker="o", label=name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Default Rate")
    plt.title("Validation Reliability Diagram")
    plt.legend()

    save_plot(output_path)


def save_probability_outputs(
    output_dir: Path,
    y_valid,
    y_test,
    raw_valid,
    raw_test,
    sigmoid_valid,
    sigmoid_test,
    isotonic_valid,
    isotonic_test,
):
    valid_out = pd.DataFrame(
        {
            "target": y_valid.values,
            "raw_pd": raw_valid,
            "sigmoid_calibrated_pd": sigmoid_valid,
            "isotonic_calibrated_pd": isotonic_valid,
        }
    )

    test_out = pd.DataFrame(
        {
            "target": y_test.values,
            "raw_pd": raw_test,
            "sigmoid_calibrated_pd": sigmoid_test,
            "isotonic_calibrated_pd": isotonic_test,
        }
    )

    valid_out.to_csv(output_dir / "validation_calibrated_probabilities.csv", index=False)
    test_out.to_csv(output_dir / "test_calibrated_probabilities.csv", index=False)

    logger.info(f"Saved calibrated probability outputs to {output_dir}")


def write_report(
    output_path: Path,
    best_model_name: str,
    metrics_table: pd.DataFrame,
    selected_method: str,
    brier_improvement_pct: float,
):
    lines = []

    lines.append("# Phase 5 Probability Calibration Report\n")

    lines.append("## Objective\n")
    lines.append(
        "Phase 5 evaluates whether the selected Phase 4 model produces reliable default probabilities, "
        "not just good risk rankings. This is necessary before using predicted probabilities for expected-loss "
        "and cost-sensitive decisioning in Phase 6.\n"
    )

    lines.append("## Base Model\n")
    lines.append(f"- Selected Phase 4 model: **{best_model_name}**\n")

    lines.append("## Calibration Methods Compared\n")
    lines.append("- **Raw model probabilities**: uncalibrated output from the selected model")
    lines.append("- **Sigmoid calibration / Platt scaling**: logistic calibration of predicted scores")
    lines.append("- **Isotonic calibration**: non-parametric monotonic calibration\n")

    lines.append("## Calibration Metrics\n")
    lines.append(metrics_table.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Selected Calibration Method\n")
    lines.append(f"- Selected method: **{selected_method}**")
    lines.append(f"- Test Brier score improvement vs raw probabilities: **{brier_improvement_pct:.2f}%**\n")

    lines.append("## Business Interpretation\n")
    lines.append(
        "Brier score measures the accuracy of predicted probabilities. Lower values are better. "
        "Calibration is important because Phase 6 will use predicted default probabilities to estimate "
        "expected loss, approval policy impact, and cost-sensitive thresholds.\n"
    )

    lines.append("## Saved Artifacts\n")
    lines.append("- `artifacts/models/sigmoid_calibrator.joblib`")
    lines.append("- `artifacts/models/isotonic_calibrator.joblib`")
    lines.append("- `artifacts/models/calibrated_model.joblib`")
    lines.append("- `artifacts/models/calibration_metadata.json`")
    lines.append("- `data/processed/phase5_calibration/validation_calibrated_probabilities.csv`")
    lines.append("- `data/processed/phase5_calibration/test_calibrated_probabilities.csv`\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {output_path}")


def main() -> None:
    config = load_config()

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    reports_dir = resolve_path("reports")
    artifacts_dir = resolve_path("artifacts")
    model_dir = artifacts_dir / "models"
    figures_dir = reports_dir / "figures"
    calibration_dir = processed_dir / "phase5_calibration"

    make_dir(reports_dir)
    make_dir(figures_dir)
    make_dir(model_dir)
    make_dir(calibration_dir)

    metadata_path = model_dir / "best_model_metadata.json"

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    best_model_name = metadata["best_model_name"]

    logger.info(f"Loading best Phase 4 model: {best_model_name}")

    base_model = joblib.load(model_dir / "best_model.joblib")

    X_valid, X_test, y_valid, y_test = load_model_inputs(
        processed_dir=processed_dir,
        best_model_name=best_model_name,
    )

    logger.info("Generating raw probabilities...")

    raw_valid_proba = base_model.predict_proba(X_valid)[:, 1]
    raw_test_proba = base_model.predict_proba(X_test)[:, 1]

    logger.info("Fitting sigmoid calibrator on validation set...")

    sigmoid_calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv="prefit",
    )
    sigmoid_calibrator.fit(X_valid, y_valid)

    logger.info("Fitting isotonic calibrator on validation set...")

    isotonic_calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv="prefit",
    )
    isotonic_calibrator.fit(X_valid, y_valid)

    logger.info("Generating calibrated probabilities...")

    sigmoid_valid_proba = sigmoid_calibrator.predict_proba(X_valid)[:, 1]
    sigmoid_test_proba = sigmoid_calibrator.predict_proba(X_test)[:, 1]

    isotonic_valid_proba = isotonic_calibrator.predict_proba(X_valid)[:, 1]
    isotonic_test_proba = isotonic_calibrator.predict_proba(X_test)[:, 1]

    logger.info("Evaluating calibration performance...")

    rows = []

    for split_name, y_true, raw_p, sig_p, iso_p in [
        ("validation", y_valid, raw_valid_proba, sigmoid_valid_proba, isotonic_valid_proba),
        ("test", y_test, raw_test_proba, sigmoid_test_proba, isotonic_test_proba),
    ]:
        for method_name, proba in [
            ("raw", raw_p),
            ("sigmoid", sig_p),
            ("isotonic", iso_p),
        ]:
            metrics = evaluate_probabilities(y_true, proba)
            metrics["split"] = split_name
            metrics["method"] = method_name
            rows.append(metrics)

    metrics_table = pd.DataFrame(rows)
    metrics_table = metrics_table[
        [
            "split",
            "method",
            "brier_score",
            "roc_auc",
            "pr_auc",
            "mean_predicted_pd",
            "observed_default_rate",
        ]
    ]

    metrics_table.to_csv(
        calibration_dir / "calibration_metrics.csv",
        index=False,
    )

    test_metrics = metrics_table[metrics_table["split"] == "test"].copy()
    selected_row = test_metrics.sort_values("brier_score", ascending=True).iloc[0]
    selected_method = selected_row["method"]

    raw_test_brier = float(
        test_metrics[test_metrics["method"] == "raw"]["brier_score"].iloc[0]
    )
    selected_test_brier = float(selected_row["brier_score"])

    brier_improvement_pct = (
        (raw_test_brier - selected_test_brier) / raw_test_brier
    ) * 100

    if selected_method == "sigmoid":
        selected_calibrator = sigmoid_calibrator
        selected_valid_proba = sigmoid_valid_proba
        selected_test_proba = sigmoid_test_proba
    elif selected_method == "isotonic":
        selected_calibrator = isotonic_calibrator
        selected_valid_proba = isotonic_valid_proba
        selected_test_proba = isotonic_test_proba
    else:
        selected_calibrator = base_model
        selected_valid_proba = raw_valid_proba
        selected_test_proba = raw_test_proba

    logger.info(f"Selected calibration method: {selected_method}")

    save_probability_outputs(
        output_dir=calibration_dir,
        y_valid=y_valid,
        y_test=y_test,
        raw_valid=raw_valid_proba,
        raw_test=raw_test_proba,
        sigmoid_valid=sigmoid_valid_proba,
        sigmoid_test=sigmoid_test_proba,
        isotonic_valid=isotonic_valid_proba,
        isotonic_test=isotonic_test_proba,
    )

    pd.DataFrame(
        {
            "target": y_valid.values,
            "calibrated_pd": selected_valid_proba,
        }
    ).to_csv(
        calibration_dir / "validation_selected_calibrated_probabilities.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "target": y_test.values,
            "calibrated_pd": selected_test_proba,
        }
    ).to_csv(
        calibration_dir / "test_selected_calibrated_probabilities.csv",
        index=False,
    )

    plot_reliability_curves(
        y_valid=y_valid,
        raw_valid_proba=raw_valid_proba,
        sigmoid_valid_proba=sigmoid_valid_proba,
        isotonic_valid_proba=isotonic_valid_proba,
        output_path=figures_dir / "phase5_reliability_diagram.png",
    )

    logger.info("Saving calibration artifacts...")

    joblib.dump(sigmoid_calibrator, model_dir / "sigmoid_calibrator.joblib")
    joblib.dump(isotonic_calibrator, model_dir / "isotonic_calibrator.joblib")
    joblib.dump(selected_calibrator, model_dir / "calibrated_model.joblib")

    with (model_dir / "calibration_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_name": best_model_name,
                "selected_calibration_method": selected_method,
                "raw_test_brier_score": raw_test_brier,
                "selected_test_brier_score": selected_test_brier,
                "brier_improvement_pct": brier_improvement_pct,
            },
            f,
            indent=2,
        )

    write_report(
        output_path=reports_dir / "phase5_calibration_report.md",
        best_model_name=best_model_name,
        metrics_table=metrics_table,
        selected_method=selected_method,
        brier_improvement_pct=brier_improvement_pct,
    )

    logger.info("Phase 5 calibration complete.")


if __name__ == "__main__":
    main()