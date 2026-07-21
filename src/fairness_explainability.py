"""
src/fairness_explainability.py

Phase 7 - Explainability and Fairness

Purpose:
- Generate global SHAP feature importance for the selected Phase 4/5 model.
- Generate local per-loan explanations for denied applicants.
- Convert model drivers into plain-English adverse action reason codes.
- Audit approval-rate fairness across protected-adjacent borrower segments:
  income band, geography/state, credit-history proxy, DTI band, and loan purpose.

Run after Phase 6:
    python src/fairness_explainability.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import get_logger, load_config, resolve_path

logger = get_logger("fairness_explainability")

SHAP_SAMPLE_SIZE = 1000
LOCAL_EXPLANATION_SIZE = 500
TOP_N_REASONS = 4
MIN_GROUP_SIZE = 500
DISPARATE_IMPACT_THRESHOLD = 0.80
DEFAULT_PD_THRESHOLD = 0.08
RANDOM_SEED = 42

FEATURE_REASON_MAP: Dict[str, str] = {
    "int_rate": "High interest-rate risk signal",
    "term": "Longer loan term increases repayment risk",
    "dti": "High debt-to-income ratio",
    "annual_inc": "Lower income relative to requested credit",
    "loan_amnt": "High requested loan amount",
    "funded_amnt": "High funded loan amount",
    "installment": "High monthly installment burden",
    "revol_util": "High revolving credit utilization",
    "revol_bal": "High revolving credit balance",
    "delinq_2yrs": "Recent delinquency history",
    "inq_last_6mths": "Recent credit inquiries",
    "open_acc": "Open account profile indicates elevated risk",
    "pub_rec": "Public record history",
    "total_acc": "Limited or risky credit account history",
    "mort_acc": "Mortgage account profile indicates elevated risk",
    "fico_avg": "Lower credit score range",
    "fico_range_low": "Lower credit score range",
    "fico_range_high": "Lower credit score range",
    "loan_to_income": "Loan amount is high relative to income",
    "revol_bal_to_income": "Revolving balance is high relative to income",
    "open_account_ratio": "Open account mix indicates elevated risk",
    "bankcard_available_ratio": "Limited available bankcard credit",
    "has_recent_delinquency": "Recent delinquency history",
    "has_public_record": "Public record history",
    "credit_history_years": "Limited credit history",
    "emp_length_num": "Shorter employment history",
    "purpose": "Loan purpose associated with elevated default risk",
    "home_ownership": "Housing status associated with elevated default risk",
    "verification_status": "Income verification status associated with elevated risk",
    "grade": "External lender grade indicates elevated risk",
    "sub_grade": "External lender sub-grade indicates elevated risk",
    "addr_state": "Geographic segment has elevated observed risk",
}


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)


def normalize_feature_name(name: str) -> str:
    name = str(name)
    name = name.replace("lt_", "<").replace("gt_", ">")
    name = re.sub(r"[\[\]\(\)]", "", name)
    return name


def plain_reason(feature_name: str) -> str:
    clean = normalize_feature_name(feature_name)
    for key, reason in FEATURE_REASON_MAP.items():
        if key in clean:
            return reason
    return f"Elevated risk associated with {clean.replace('_', ' ')}"


def load_target(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "target" in df.columns:
        return df["target"].astype(int).reset_index(drop=True)
    return df.iloc[:, 0].astype(int).reset_index(drop=True)


def clean_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Make feature matrix safe for sklearn/XGBoost/SHAP.

    Some parquet/csv round-trips can leave numeric values as strings like
    "[1.84325E-1]". TreeExplainer then fails with "could not convert
    string to float" and SHAP falls back to slow permutation explanations.
    This function keeps column names model-safe and coerces list-looking numeric
    strings back to floats.
    """
    X = X.copy()
    X.columns = (
        X.columns.astype(str)
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "lt_", regex=False)
        .str.replace(">", "gt_", regex=False)
    )

    for col in X.columns:
        if X[col].dtype == "object":
            cleaned = (
                X[col]
                .astype(str)
                .str.strip()
                .str.replace("[", "", regex=False)
                .str.replace("]", "", regex=False)
                .str.replace("'", "", regex=False)
                .str.replace('"', "", regex=False)
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() >= 0.95:
                X[col] = numeric.fillna(numeric.median())
            else:
                # Last-resort encoding for any true categorical column that reaches this stage.
                X[col] = pd.factorize(cleaned, sort=True)[0].astype(float)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

    return X


def sanitize_xgboost_columns(X: pd.DataFrame) -> pd.DataFrame:
    # Backward-compatible wrapper name used by older code.
    return clean_feature_matrix(X)


def load_model_and_inputs(processed_dir: Path, artifacts_dir: Path) -> Tuple[object, str, pd.DataFrame, pd.Series]:
    metadata_path = artifacts_dir / "models" / "best_model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_name = metadata.get("best_model_name", "xgboost_sampled")
    else:
        model_name = "xgboost_sampled"

    model_path = artifacts_dir / "models" / "best_model.joblib"
    if not model_path.exists():
        model_path = artifacts_dir / "models" / f"{model_name}.joblib"
    model = joblib.load(model_path)

    if model_name == "woe_logistic_regression":
        X_test = pd.read_parquet(processed_dir / "X_test_woe_selected.parquet")
        y_test = load_target(processed_dir / "y_test_woe.parquet")
    else:
        X_test = pd.read_parquet(processed_dir / "X_test.parquet")
        y_test = load_target(processed_dir / "y_test.parquet")

    X_test = clean_feature_matrix(X_test)
    return model, model_name, X_test.reset_index(drop=True), y_test


def load_probability_and_decision_data(processed_dir: Path) -> pd.DataFrame:
    decision_path = processed_dir / "phase6_decisioning" / "decisioning_test_scored_loans.csv"
    if decision_path.exists():
        df = pd.read_csv(decision_path)
    else:
        prob_path = processed_dir / "phase5_calibration" / "test_selected_calibrated_probabilities.csv"
        df = pd.read_csv(prob_path)

    if "calibrated_pd" not in df.columns:
        for col in ["isotonic_calibrated_pd", "sigmoid_calibrated_pd", "raw_pd"]:
            if col in df.columns:
                df["calibrated_pd"] = df[col]
                break
    if "calibrated_pd" not in df.columns:
        raise ValueError("Could not find calibrated probabilities from Phase 5/6.")

    return df.reset_index(drop=True)


def get_policy_threshold(processed_dir: Path) -> float:
    cost_path = processed_dir / "phase6_decisioning" / "cost_threshold_optimization.csv"
    if cost_path.exists():
        cost = pd.read_csv(cost_path)
        if {"threshold", "total_cost"}.issubset(cost.columns):
            return float(cost.sort_values("total_cost").iloc[0]["threshold"])
    return DEFAULT_PD_THRESHOLD


def compute_shap_values(model: object, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_n = min(SHAP_SAMPLE_SIZE, len(X))
    X_sample = X.sample(sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
    X_sample = clean_feature_matrix(X_sample)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception as exc:
        logger.warning("TreeExplainer failed (%s). Falling back to generic Explainer on a smaller sample.", exc)
        fallback_n = min(250, len(X_sample))
        X_sample = X_sample.sample(fallback_n, random_state=RANDOM_SEED).reset_index(drop=True)
        background = shap.sample(X_sample, min(100, len(X_sample)), random_state=RANDOM_SEED)
        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(X_sample).values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    return X_sample, shap_df


def build_global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    importance = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    importance.columns = ["feature", "mean_abs_shap"]
    importance["plain_english_reason"] = importance["feature"].map(plain_reason)
    return importance


def build_local_reason_codes(
    X_sample: pd.DataFrame,
    shap_df: pd.DataFrame,
    probabilities: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    n = min(len(X_sample), len(probabilities))
    local = probabilities.iloc[:n].copy().reset_index(drop=True)
    local["decision"] = np.where(local["calibrated_pd"] <= threshold, "Approve", "Deny")

    denied_idx = local.index[local["decision"] == "Deny"].tolist()[:LOCAL_EXPLANATION_SIZE]
    rows: List[dict] = []

    for idx in denied_idx:
        contrib = shap_df.iloc[idx].sort_values(ascending=False)
        top_features = [f for f in contrib.index if contrib.loc[f] > 0][:TOP_N_REASONS]
        reasons = [plain_reason(f) for f in top_features]

        row = {
            "sample_index": idx,
            "calibrated_pd": local.loc[idx, "calibrated_pd"],
            "decision_threshold": threshold,
            "decision": "Deny",
        }
        for rank in range(TOP_N_REASONS):
            feature = top_features[rank] if rank < len(top_features) else None
            row[f"reason_{rank + 1}"] = reasons[rank] if rank < len(reasons) else None
            row[f"feature_{rank + 1}"] = feature
            row[f"shap_contribution_{rank + 1}"] = float(contrib.loc[feature]) if feature else None
            row[f"feature_value_{rank + 1}"] = X_sample.loc[idx, feature] if feature else None
        rows.append(row)

    return pd.DataFrame(rows)


def safe_qcut(series: pd.Series, q: int, labels: Iterable[str]) -> pd.Series:
    try:
        return pd.qcut(series, q=q, labels=list(labels), duplicates="drop")
    except Exception:
        return pd.Series(["Unknown"] * len(series), index=series.index)


def add_fairness_segments(context: pd.DataFrame) -> pd.DataFrame:
    out = context.copy()

    if "annual_inc" in out.columns:
        out["income_band"] = safe_qcut(out["annual_inc"], 4, ["Low income", "Mid-low income", "Mid-high income", "High income"])
    if "dti" in out.columns:
        out["dti_band"] = pd.cut(
            out["dti"],
            bins=[-np.inf, 10, 20, 30, np.inf],
            labels=["DTI <=10", "DTI 10-20", "DTI 20-30", "DTI >30"],
        )
    if "fico_avg" in out.columns:
        out["fico_band"] = pd.cut(
            out["fico_avg"],
            bins=[-np.inf, 660, 700, 740, np.inf],
            labels=["FICO <660", "FICO 660-699", "FICO 700-739", "FICO 740+"],
        )
    elif {"fico_range_low", "fico_range_high"}.issubset(out.columns):
        out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
        out["fico_band"] = pd.cut(
            out["fico_avg"],
            bins=[-np.inf, 660, 700, 740, np.inf],
            labels=["FICO <660", "FICO 660-699", "FICO 700-739", "FICO 740+"],
        )
    if "credit_history_years" in out.columns:
        out["credit_history_band"] = pd.cut(
            out["credit_history_years"],
            bins=[-np.inf, 5, 10, 20, np.inf],
            labels=["<=5 years", "5-10 years", "10-20 years", "20+ years"],
        )
    if "addr_state" in out.columns:
        out["state"] = out["addr_state"].astype(str)
    if "purpose" in out.columns:
        out["purpose_segment"] = out["purpose"].astype(str)

    return out


def audit_disparate_impact(df: pd.DataFrame, segment_cols: List[str]) -> pd.DataFrame:
    rows: List[dict] = []
    overall_approval = df["approved"].mean()

    for segment_col in segment_cols:
        if segment_col not in df.columns:
            continue
        grouped = (
            df.groupby(segment_col, dropna=False, observed=False)
            .agg(
                loans=("approved", "size"),
                approval_rate=("approved", "mean"),
                observed_default_rate=("target", "mean"),
                avg_pd=("calibrated_pd", "mean"),
            )
            .reset_index()
            .rename(columns={segment_col: "group"})
        )
        grouped = grouped[grouped["loans"] >= MIN_GROUP_SIZE]
        if grouped.empty:
            continue

        reference_rate = grouped["approval_rate"].max()
        grouped["segment"] = segment_col
        grouped["reference_approval_rate"] = reference_rate
        grouped["disparate_impact_ratio"] = grouped["approval_rate"] / reference_rate if reference_rate > 0 else np.nan
        grouped["overall_approval_rate"] = overall_approval
        grouped["flag_80pct_rule"] = grouped["disparate_impact_ratio"] < DISPARATE_IMPACT_THRESHOLD
        rows.extend(grouped.to_dict("records"))

    cols = [
        "segment", "group", "loans", "approval_rate", "reference_approval_rate",
        "disparate_impact_ratio", "flag_80pct_rule", "observed_default_rate", "avg_pd",
        "overall_approval_rate",
    ]
    return pd.DataFrame(rows, columns=cols)


def plot_global_importance(importance: pd.DataFrame, output_path: Path) -> None:
    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Top Global SHAP Drivers")
    save_plot(output_path)


def plot_disparate_impact(fairness: pd.DataFrame, output_path: Path) -> None:
    if fairness.empty:
        return
    plot_df = fairness.sort_values("disparate_impact_ratio").head(25).iloc[::-1]
    labels = plot_df["segment"].astype(str) + ": " + plot_df["group"].astype(str)
    plt.figure(figsize=(10, 8))
    plt.barh(labels, plot_df["disparate_impact_ratio"])
    plt.axvline(DISPARATE_IMPACT_THRESHOLD, linestyle="--", label="80% rule threshold")
    plt.xlabel("Disparate impact ratio vs highest-approval group")
    plt.ylabel("Segment group")
    plt.title("Lowest Approval-Rate Fairness Ratios")
    plt.legend()
    save_plot(output_path)


def write_report(
    output_path: Path,
    model_name: str,
    threshold: float,
    importance: pd.DataFrame,
    reason_codes: pd.DataFrame,
    fairness: pd.DataFrame,
) -> None:
    top_features = importance.head(10)
    flagged = fairness[fairness["flag_80pct_rule"]] if not fairness.empty else pd.DataFrame()

    lines: List[str] = []
    lines.append("# Phase 7 Explainability and Fairness Report\n")
    lines.append("## Objective\n")
    lines.append(
        "Phase 7 explains the selected credit-risk model and audits the downstream approval policy. "
        "The goal is to make the model usable in a regulated lending workflow: global drivers, "
        "loan-level adverse action reason codes, and approval-rate fairness checks.\n"
    )
    lines.append("## Model and Policy Inputs\n")
    lines.append(f"- Base model explained: **{model_name}**")
    lines.append(f"- Decision threshold used for denial reason generation: **{threshold:.2f}**")
    lines.append(f"- Local denied-applicant explanations generated: **{len(reason_codes):,}**")
    lines.append(f"- Fairness minimum group size: **{MIN_GROUP_SIZE:,} loans**")
    lines.append(f"- Disparate impact alert threshold: **{DISPARATE_IMPACT_THRESHOLD:.0%}**\n")

    lines.append("## Global SHAP Feature Importance\n")
    lines.append(top_features.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Adverse Action Reason Code Logic\n")
    lines.append(
        "For denied applicants, the script selects the highest positive SHAP contributors, meaning "
        "the features that pushed the predicted probability of default upward. These are mapped to "
        "plain-English reason codes suitable for a portfolio project demonstration.\n"
    )
    if not reason_codes.empty:
        display_cols = ["sample_index", "calibrated_pd", "reason_1", "reason_2", "reason_3", "reason_4"]
        lines.append(reason_codes[display_cols].head(10).to_markdown(index=False))
        lines.append("\n")

    lines.append("## Fairness / Disparate Impact Audit\n")
    if fairness.empty:
        lines.append("No fairness segment table was produced because no eligible segment columns were available.\n")
    else:
        lines.append(f"- Segment groups evaluated: **{len(fairness):,}**")
        lines.append(f"- Groups flagged below the 80% rule: **{len(flagged):,}**\n")
        lines.append(fairness.sort_values(["flag_80pct_rule", "disparate_impact_ratio"], ascending=[False, True]).head(30).to_markdown(index=False))
        lines.append("\n")

    lines.append("## Important Interpretation Note\n")
    lines.append(
        "This is a protected-adjacent fairness audit, not a legal fair-lending certification. "
        "The Lending Club dataset does not provide protected classes such as race or sex. Therefore, "
        "the report uses available proxies and business segments such as income band, geography/state, "
        "credit-history band, DTI band, FICO band, and loan purpose. Any flagged group should be treated "
        "as a review trigger requiring business justification, documentation, and possible policy mitigation.\n"
    )

    lines.append("## Saved Artifacts\n")
    lines.append("- `data/processed/phase7_explainability/global_shap_importance.csv`")
    lines.append("- `data/processed/phase7_explainability/local_adverse_action_reason_codes.csv`")
    lines.append("- `data/processed/phase7_explainability/fairness_disparate_impact_audit.csv`")
    lines.append("- `reports/figures/phase7_global_shap_importance.png`")
    lines.append("- `reports/figures/phase7_disparate_impact.png`")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report: %s", output_path)


def main() -> None:
    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    artifacts_dir = resolve_path("artifacts")
    reports_dir = resolve_path("reports")
    figures_dir = reports_dir / "figures"
    phase7_dir = processed_dir / "phase7_explainability"

    make_dir(reports_dir)
    make_dir(figures_dir)
    make_dir(phase7_dir)

    logger.info("Loading selected model, test features, calibrated probabilities, and context...")
    model, model_name, X_test, y_test = load_model_and_inputs(processed_dir, artifacts_dir)
    probabilities = load_probability_and_decision_data(processed_dir)
    threshold = get_policy_threshold(processed_dir)

    logger.info("Computing SHAP explanations...")
    X_sample, shap_df = compute_shap_values(model, X_test)
    importance = build_global_importance(shap_df)

    logger.info("Generating local adverse action reason codes...")
    reason_codes = build_local_reason_codes(X_sample, shap_df, probabilities, threshold)

    logger.info("Running fairness audit...")
    context_path = processed_dir / "loan_test.parquet"
    context = pd.read_parquet(context_path).reset_index(drop=True)
    # Avoid duplicate column names from context/probability files. Duplicate names make
    # pandas groupby aggregation return a DataFrame where a Series is expected.
    context = context.loc[:, ~context.columns.duplicated()].copy()
    context = context.drop(columns=["target", "calibrated_pd", "approved"], errors="ignore")

    fairness_df = context.copy()
    fairness_df["target"] = y_test.reset_index(drop=True).astype(int)
    fairness_df["calibrated_pd"] = probabilities["calibrated_pd"].reset_index(drop=True).astype(float)
    fairness_df["approved"] = (fairness_df["calibrated_pd"] <= threshold).astype(int)
    fairness_df = add_fairness_segments(fairness_df)

    segment_cols = [
        "income_band",
        "dti_band",
        "fico_band",
        "credit_history_band",
        "state",
        "purpose_segment",
        "home_ownership",
        "verification_status",
    ]
    fairness = audit_disparate_impact(fairness_df, segment_cols)

    logger.info("Saving Phase 7 tables and figures...")
    importance.to_csv(phase7_dir / "global_shap_importance.csv", index=False)
    shap_df.to_csv(phase7_dir / "sample_shap_values.csv", index=False)
    X_sample.to_csv(phase7_dir / "sample_explained_features.csv", index=False)
    reason_codes.to_csv(phase7_dir / "local_adverse_action_reason_codes.csv", index=False)
    fairness.to_csv(phase7_dir / "fairness_disparate_impact_audit.csv", index=False)

    plot_global_importance(importance, figures_dir / "phase7_global_shap_importance.png")
    plot_disparate_impact(fairness, figures_dir / "phase7_disparate_impact.png")

    write_report(
        output_path=reports_dir / "phase7_explainability_fairness_report.md",
        model_name=model_name,
        threshold=threshold,
        importance=importance,
        reason_codes=reason_codes,
        fairness=fairness,
    )

    logger.info("fairness_explainability.py complete.")


if __name__ == "__main__":
    main()