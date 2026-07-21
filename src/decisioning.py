"""
src/decisioning.py

Phase 6 — Cost-Sensitive Decision Layer

Purpose:
- Use calibrated PDs from Phase 5.
- Produce credit-risk business metrics:
  - Decile analysis
  - Lift / gains
  - KS statistic
  - Approval-rate simulation
  - Expected loss calculation
  - Cost-sensitive threshold optimization
  - Business policy scenarios
  - Credit policy memo

Run:
    python src/decisioning.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

from utils import load_config, resolve_path, get_logger

logger = get_logger("decisioning")


LGD = 0.60
FALSE_REJECTION_OPPORTUNITY_COST_RATE = 0.05
DEFAULT_THRESHOLD_GRID = np.arange(0.01, 0.80, 0.01)


def make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path}")


def load_test_loan_context(processed_dir: Path) -> pd.DataFrame:
    test = pd.read_parquet(processed_dir / "loan_test.parquet")

    keep_cols = [
        "loan_amnt",
        "funded_amnt",
        "int_rate",
        "grade",
        "sub_grade",
        "term",
        "purpose",
        "annual_inc",
        "dti",
    ]

    existing_cols = [col for col in keep_cols if col in test.columns]
    return test[existing_cols].reset_index(drop=True)


def build_decision_frame(processed_dir: Path) -> pd.DataFrame:
    prob_path = processed_dir / "phase5_calibration" / "test_selected_calibrated_probabilities.csv"
    probs = pd.read_csv(prob_path)

    context = load_test_loan_context(processed_dir)
    df = pd.concat([context, probs], axis=1)

    if "loan_amnt" not in df.columns:
        raise ValueError("loan_amnt is required for expected loss calculation.")

    df["ead"] = df["loan_amnt"]
    df["lgd"] = LGD
    df["expected_loss"] = df["calibrated_pd"] * df["lgd"] * df["ead"]

    if "int_rate" in df.columns:
        df["int_rate_decimal"] = df["int_rate"] / 100
        df["opportunity_profit"] = (
            df["ead"] * df["int_rate_decimal"] * FALSE_REJECTION_OPPORTUNITY_COST_RATE
        )
    else:
        df["opportunity_profit"] = df["ead"] * FALSE_REJECTION_OPPORTUNITY_COST_RATE

    return df


def make_decile_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["risk_decile"] = pd.qcut(
        out["calibrated_pd"].rank(method="first", ascending=False),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)

    total_defaults = out["target"].sum()
    portfolio_default_rate = out["target"].mean()

    table = (
        out.groupby("risk_decile")
        .agg(
            loans=("target", "size"),
            defaults=("target", "sum"),
            avg_pd=("calibrated_pd", "mean"),
            realized_default_rate=("target", "mean"),
            avg_loan_amnt=("loan_amnt", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )

    table["default_capture_rate"] = table["defaults"] / total_defaults
    table["cumulative_defaults"] = table["defaults"].cumsum()
    table["cumulative_default_capture_rate"] = table["cumulative_defaults"] / total_defaults
    table["lift"] = table["realized_default_rate"] / portfolio_default_rate

    return table


def calculate_ks(y_true: pd.Series, scores: pd.Series) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def approval_simulation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for approval_rate in np.round(np.arange(0.10, 1.001, 0.05), 2):
        approval_rate = min(max(float(approval_rate), 0.0), 1.0)
        cutoff = df["calibrated_pd"].quantile(approval_rate)

        approved = df[df["calibrated_pd"] <= cutoff]
        rejected = df[df["calibrated_pd"] > cutoff]

        rows.append(
            {
                "target_approval_rate": approval_rate,
                "actual_approval_rate": len(approved) / len(df),
                "approved_loans": len(approved),
                "rejected_loans": len(rejected),
                "approved_default_rate": approved["target"].mean(),
                "rejected_default_rate": rejected["target"].mean() if len(rejected) > 0 else np.nan,
                "approved_expected_loss": approved["expected_loss"].sum(),
                "rejected_expected_loss_avoided": rejected["expected_loss"].sum(),
                "approved_loan_volume": approved["ead"].sum(),
            }
        )

    return pd.DataFrame(rows)


def threshold_cost_optimization(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for threshold in DEFAULT_THRESHOLD_GRID:
        approved = df[df["calibrated_pd"] < threshold]
        rejected = df[df["calibrated_pd"] >= threshold]

        false_approvals = approved[approved["target"] == 1]
        false_rejections = rejected[rejected["target"] == 0]

        false_approval_cost = (false_approvals["ead"] * LGD).sum()
        false_rejection_cost = false_rejections["opportunity_profit"].sum()
        total_cost = false_approval_cost + false_rejection_cost

        rows.append(
            {
                "threshold": threshold,
                "approval_rate": len(approved) / len(df),
                "approved_loans": len(approved),
                "rejected_loans": len(rejected),
                "approved_default_rate": approved["target"].mean() if len(approved) > 0 else np.nan,
                "rejected_default_rate": rejected["target"].mean() if len(rejected) > 0 else np.nan,
                "false_approval_cost": false_approval_cost,
                "false_rejection_cost": false_rejection_cost,
                "total_cost": total_cost,
                "expected_loss_approved": approved["expected_loss"].sum(),
                "expected_loss_avoided": rejected["expected_loss"].sum(),
            }
        )

    return pd.DataFrame(rows)


def business_policy_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "Conservative": 0.40,
        "Balanced": 0.60,
        "Growth": 0.80,
    }

    rows = []

    approve_all_expected_loss = df["expected_loss"].sum()
    portfolio_default_rate = df["target"].mean()

    for policy_name, approval_rate in scenarios.items():
        cutoff = df["calibrated_pd"].quantile(approval_rate)

        approved = df[df["calibrated_pd"] <= cutoff]
        rejected = df[df["calibrated_pd"] > cutoff]

        approved_expected_loss = approved["expected_loss"].sum()
        rejected_expected_loss_avoided = rejected["expected_loss"].sum()

        rows.append(
            {
                "policy": policy_name,
                "target_approval_rate": approval_rate,
                "actual_approval_rate": len(approved) / len(df),
                "approved_loans": len(approved),
                "rejected_loans": len(rejected),
                "approved_default_rate": approved["target"].mean(),
                "default_rate_reduction_pct": (
                    (portfolio_default_rate - approved["target"].mean())
                    / portfolio_default_rate
                ),
                "approved_loan_volume": approved["ead"].sum(),
                "approved_expected_loss": approved_expected_loss,
                "expected_loss_avoided": rejected_expected_loss_avoided,
                "expected_loss_reduction_pct": rejected_expected_loss_avoided / approve_all_expected_loss,
            }
        )

    return pd.DataFrame(rows)


def create_risk_tiers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    out["risk_tier"] = pd.cut(
        out["calibrated_pd"],
        bins=[0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.00],
        labels=["A", "B", "C", "D", "E", "F"],
        include_lowest=True,
    )

    table = (
        out.groupby("risk_tier", observed=False)
        .agg(
            loans=("target", "size"),
            avg_pd=("calibrated_pd", "mean"),
            realized_default_rate=("target", "mean"),
            avg_loan_amnt=("loan_amnt", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .reset_index()
    )

    table["model_price_rate"] = (table["avg_pd"] * LGD) + 0.05

    if "grade" in out.columns and "int_rate" in out.columns:
        grade_table = (
            out.groupby("grade")
            .agg(
                loans=("target", "size"),
                avg_actual_int_rate=("int_rate", "mean"),
                realized_default_rate=("target", "mean"),
                avg_pd=("calibrated_pd", "mean"),
            )
            .reset_index()
            .sort_values("grade")
        )
    else:
        grade_table = pd.DataFrame()

    return table, grade_table


def plot_decile_lift(decile_table: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(decile_table["risk_decile"].astype(str), decile_table["lift"])
    plt.xlabel("Risk Decile: 1 = Highest Risk")
    plt.ylabel("Lift vs Portfolio Average")
    plt.title("Default Lift by Risk Decile")
    save_plot(output_path)


def plot_cumulative_gains(decile_table: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(
        decile_table["risk_decile"],
        decile_table["cumulative_default_capture_rate"],
        marker="o",
        label="Model",
    )
    plt.plot(
        decile_table["risk_decile"],
        decile_table["risk_decile"] / 10,
        linestyle="--",
        label="Random",
    )
    plt.xlabel("Top Risk Deciles Included")
    plt.ylabel("Cumulative Default Capture Rate")
    plt.title("Cumulative Gains Chart")
    plt.legend()
    save_plot(output_path)


def plot_cost_curve(cost_table: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(cost_table["threshold"], cost_table["total_cost"])
    plt.xlabel("PD Threshold")
    plt.ylabel("Total Cost")
    plt.title("Cost-Sensitive Threshold Optimization")
    save_plot(output_path)


def plot_approval_simulation(approval_table: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(
        approval_table["actual_approval_rate"],
        approval_table["approved_default_rate"],
        marker="o",
    )
    plt.xlabel("Approval Rate")
    plt.ylabel("Approved Population Default Rate")
    plt.title("Approval Rate vs Approved Default Rate")
    save_plot(output_path)


def plot_policy_scenarios(policy_table: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(policy_table["policy"], policy_table["approved_default_rate"])
    plt.xlabel("Policy Scenario")
    plt.ylabel("Approved Default Rate")
    plt.title("Approved Default Rate by Business Policy")
    save_plot(output_path)


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_report(
    output_path: Path,
    df: pd.DataFrame,
    decile_table: pd.DataFrame,
    approval_table: pd.DataFrame,
    cost_table: pd.DataFrame,
    risk_tier_table: pd.DataFrame,
    grade_table: pd.DataFrame,
    policy_table: pd.DataFrame,
    ks_stat: float,
):
    portfolio_default_rate = df["target"].mean()
    portfolio_expected_loss = df["expected_loss"].sum()

    top_decile = decile_table.iloc[0]
    top_decile_lift = top_decile["lift"]
    top_decile_capture = top_decile["default_capture_rate"]

    optimal = cost_table.sort_values("total_cost").iloc[0]

    approve_all_cost = (df[df["target"] == 1]["ead"] * LGD).sum()
    savings_vs_approve_all = approve_all_cost - optimal["total_cost"]
    savings_pct = savings_vs_approve_all / approve_all_cost

    balanced_policy = policy_table[policy_table["policy"] == "Balanced"].iloc[0]

    lines = []

    lines.append("# Phase 6 Cost-Sensitive Decision Layer Report\n")

    lines.append("## Objective\n")
    lines.append(
        "Phase 6 converts calibrated probability-of-default scores into business decisions. "
        "The goal is to quantify risk concentration, approval policy tradeoffs, expected loss, "
        "and cost-sensitive threshold selection.\n"
    )

    lines.append("## Cost Assumptions\n")
    lines.append(f"- Loss Given Default (LGD): **{LGD:.0%}**")
    lines.append("- Exposure at Default (EAD): loan amount")
    lines.append(
        f"- False rejection opportunity cost proxy: **{FALSE_REJECTION_OPPORTUNITY_COST_RATE:.0%}** "
        "of interest-adjusted exposure\n"
    )

    lines.append("## Portfolio Summary\n")
    lines.append(f"- Test population loans: **{len(df):,}**")
    lines.append(f"- Observed default rate: **{pct(portfolio_default_rate)}**")
    lines.append(f"- Total calibrated expected loss: **{money(portfolio_expected_loss)}**")
    lines.append(f"- KS statistic: **{ks_stat:.3f}**\n")

    lines.append("## Decile and Lift Analysis\n")
    lines.append(
        f"The highest-risk decile has a default lift of **{top_decile_lift:.2f}x** "
        f"and captures **{pct(top_decile_capture)}** of all observed defaults.\n"
    )
    lines.append(decile_table.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Cost-Optimal Threshold\n")
    lines.append(f"- Optimal PD threshold: **{optimal['threshold']:.2f}**")
    lines.append(f"- Approval rate at optimal threshold: **{pct(optimal['approval_rate'])}**")
    lines.append(f"- Approved default rate: **{pct(optimal['approved_default_rate'])}**")
    lines.append(f"- Total cost at optimal threshold: **{money(optimal['total_cost'])}**")
    lines.append(
        f"- Estimated savings vs approve-all baseline: **{money(savings_vs_approve_all)} "
        f"({pct(savings_pct)})**\n"
    )

    lines.append("## Approval Rate Simulation\n")
    lines.append(approval_table.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Business Policy Scenarios\n")
    lines.append(
        "Instead of relying only on the mathematically cost-minimizing threshold, "
        "three practical underwriting strategies are simulated below. This allows a credit team "
        "to choose a policy based on risk appetite and growth goals.\n"
    )
    lines.append(policy_table.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Risk Tier Summary\n")
    lines.append(risk_tier_table.to_markdown(index=False))
    lines.append("\n")

    if not grade_table.empty:
        lines.append("## Lending Club Grade Comparison\n")
        lines.append(grade_table.to_markdown(index=False))
        lines.append("\n")

    lines.append("## Recommended Credit Policy\n")
    lines.append(
        f"The purely cost-optimal threshold is **{optimal['threshold']:.2f}**, but it approves only "
        f"**{pct(optimal['approval_rate'])}** of applicants. For a more realistic production policy, "
        "the **Balanced** strategy is recommended as an initial operating point.\n"
    )
    lines.append(
        f"The Balanced policy approves approximately **{pct(balanced_policy['actual_approval_rate'])}** "
        f"of applicants, reduces the approved-pool default rate to "
        f"**{pct(balanced_policy['approved_default_rate'])}**, and avoids approximately "
        f"**{money(balanced_policy['expected_loss_avoided'])}** in expected loss compared with approving all loans.\n"
    )

    lines.append("## Resume-Ready Quantifiers\n")
    lines.append(f"- Built a calibrated credit-risk decision layer over **{len(df):,}** out-of-time test loans.")
    lines.append(f"- Achieved **{ks_stat:.3f} KS statistic** on the held-out test period.")
    lines.append(f"- Highest-risk decile delivered **{top_decile_lift:.2f}x default lift**.")
    lines.append(
        f"- Cost-sensitive thresholding estimated **{money(savings_vs_approve_all)}** "
        f"loss reduction vs approve-all baseline under **{LGD:.0%} LGD** assumption."
    )
    lines.append(
        f"- Balanced policy approved **{pct(balanced_policy['actual_approval_rate'])}** of loans "
        f"with an approved-pool default rate of **{pct(balanced_policy['approved_default_rate'])}**."
    )
    lines.append(
        f"- Balanced policy avoided approximately **{money(balanced_policy['expected_loss_avoided'])}** "
        "in expected loss versus approving all loans."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote report: {output_path}")


def main() -> None:
    config = load_config()

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    reports_dir = resolve_path("reports")
    figures_dir = reports_dir / "figures"
    decision_dir = processed_dir / "phase6_decisioning"

    make_dir(reports_dir)
    make_dir(figures_dir)
    make_dir(decision_dir)

    logger.info("Loading calibrated Phase 5 probabilities and test loan context...")

    df = build_decision_frame(processed_dir)

    logger.info("Creating decile, lift, KS, approval, cost, and policy tables...")

    decile_table = make_decile_table(df)
    ks_stat = calculate_ks(df["target"], df["calibrated_pd"])
    approval_table = approval_simulation(df)
    cost_table = threshold_cost_optimization(df)
    risk_tier_table, grade_table = create_risk_tiers(df)
    policy_table = business_policy_scenarios(df)

    logger.info("Saving Phase 6 tables...")

    df.to_csv(decision_dir / "decisioning_test_scored_loans.csv", index=False)
    decile_table.to_csv(decision_dir / "decile_lift_table.csv", index=False)
    approval_table.to_csv(decision_dir / "approval_rate_simulation.csv", index=False)
    cost_table.to_csv(decision_dir / "cost_threshold_optimization.csv", index=False)
    risk_tier_table.to_csv(decision_dir / "risk_tier_summary.csv", index=False)
    policy_table.to_csv(decision_dir / "business_policy_scenarios.csv", index=False)

    if not grade_table.empty:
        grade_table.to_csv(decision_dir / "lending_club_grade_comparison.csv", index=False)

    logger.info("Creating Phase 6 figures...")

    plot_decile_lift(
        decile_table,
        figures_dir / "phase6_decile_lift.png",
    )

    plot_cumulative_gains(
        decile_table,
        figures_dir / "phase6_cumulative_gains.png",
    )

    plot_cost_curve(
        cost_table,
        figures_dir / "phase6_cost_threshold_curve.png",
    )

    plot_approval_simulation(
        approval_table,
        figures_dir / "phase6_approval_rate_simulation.png",
    )

    plot_policy_scenarios(
        policy_table,
        figures_dir / "phase6_policy_scenarios.png",
    )

    write_report(
        output_path=reports_dir / "phase6_decisioning_report.md",
        df=df,
        decile_table=decile_table,
        approval_table=approval_table,
        cost_table=cost_table,
        risk_tier_table=risk_tier_table,
        grade_table=grade_table,
        policy_table=policy_table,
        ks_stat=ks_stat,
    )

    logger.info("Phase 6 decisioning complete.")


if __name__ == "__main__":
    main()