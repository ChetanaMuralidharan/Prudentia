Prudentia — Explainable Credit Risk Decisioning Engine
An end-to-end credit risk system that goes beyond default prediction to deliver calibrated, cost-optimized, explainable, and fairness-audited lending decisions — combining traditional scorecard modeling, modern gradient-boosted machine learning, and production-style model risk monitoring.

Python scikit-learn XGBoost SHAP MLflow FastAPI Streamlit

## Overview

Prudentia is an end-to-end credit risk decisioning platform that predicts borrower default risk and converts that prediction into an actual, defensible lending decision — not just a probability score sitting in a notebook.

Most public credit risk projects stop at "train a model, report AUC." Prudentia instead treats risk modeling as what it actually is inside a regulated lender: a **decisioning lifecycle** — predict, calibrate, price, explain, audit for fairness, and monitor for drift after deployment. This is deliberately closer to how a bank's model risk management (MRM) function evaluates a model before it's approved for production than it is to a typical Kaggle submission.

The system is built on ~2.2M real Lending Club loans (2007–2018), using the loan's actual issue date to run genuine time-aware validation — training on earlier vintages and testing out-of-time on later ones, the way credit models are actually validated in practice, rather than a random shuffle that leaks future information into training.

**Positioning note:** Credit scoring itself is a mature, well-understood problem. Prudentia does not claim to invent a new prediction technique — its value is in demonstrating the *responsible deployment lifecycle* around a mature technique: calibration, cost-based decisioning, explainability, fairness auditing, and drift monitoring. This is the discipline that separates a model that gets approved for production at a bank from one that stays a notebook.

## System Architecture

```
Lending Club Loan Data (2007–2018, ~2.2M loans)
            │
            ▼
  Leakage-Safe Data Cleaning
            │
            ▼
  Time-Aware Train / Validation / Test Split
    (train on early vintages, test out-of-time)
            │
            ▼
  Feature Engineering (WOE binning + engineered features)
            │
   ┌────────┼─────────────┐
   ▼        ▼              ▼
 WOE      Random         XGBoost
Scorecard Forest       (tuned via
(Logistic (GridSearchCV/  GridSearchCV/
Regression) RandomizedSearchCV) RandomizedSearchCV)
   │        │              │
   └────────┴──────┬───────┘
                    ▼
        Model Comparison & Selection
           (KS-statistic, Gini, AUC)
                    ▼
        Probability Calibration
        (Isotonic Regression / Platt Scaling)
                    ▼
        Cost-Sensitive Decision Layer
   (expected-loss-minimizing threshold,
        risk-based pricing tiers)
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
     SHAP        Fairness      Drift
  Explainability   Audit     Monitoring
  + Reason Codes (disparate  (PSI across
                   impact)    vintages)
        │           │            │
        └───────────┴─────┬──────┘
                           ▼
              FastAPI Scoring Endpoint
                           ▼
              Streamlit Decision Dashboard
```

## Key Features

### Data Engineering
- Leakage audit distinguishing application-time features from post-origination outcome fields
- Time-aware (out-of-time) train/validation/test split based on loan issue date
- WOE (Weight of Evidence) binning with Information Value (IV) feature ranking

### Modeling
- WOE + Logistic Regression scorecard — the interpretable, regulator-approved baseline
- Random Forest — an interpretable non-linear challenger
- XGBoost — the top-performing gradient-boosted challenger
- Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
- Model comparison on KS-statistic, Gini coefficient, AUC-ROC, and PR-AUC — the metrics credit risk teams actually report

### Decisioning Layer
- Probability calibration (isotonic regression / Platt scaling) with reliability diagrams and Brier score
- Cost-sensitive threshold optimization based on explicit, documented dollar costs of false negatives vs. false positives
- Risk-based pricing tiers benchmarked against Lending Club's own assigned loan grades and interest rates

### Explainability & Fairness
- SHAP global and local (per-loan) explanations
- Automated adverse-action reason code generation (mirroring ECOA / Regulation B requirements for credit denial explanations)
- Disparate impact (80% rule) fairness audit across age, income, and geographic segments

### Monitoring
- Population Stability Index (PSI) tracking for features and model score across loan vintages
- Simulated drift alerting to flag when a retraining review would be warranted

### Application Layer
- FastAPI `/score` endpoint returning predicted default probability, calibrated probability, risk tier, decision, and top reason codes
- Streamlit decision dashboard for a loan-officer-style view of individual applicant decisions

### Evaluation & Observability
- MLflow experiment tracking across all model runs and hyperparameter configurations

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python |
| Data Source | Lending Club Loan Data (2007–2018) |
| Data Processing | Pandas, NumPy |
| Scorecard Modeling | WOE Binning, Logistic Regression |
| Machine Learning | scikit-learn (Random Forest), XGBoost |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV |
| Calibration | Isotonic Regression, Platt Scaling |
| Explainability | SHAP |
| Fairness Analysis | Disparate Impact Ratio (80% Rule) |
| Drift Monitoring | Population Stability Index (PSI) |
| Experiment Tracking | MLflow |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation Metrics | AUC-ROC, KS-Statistic, Gini, PR-AUC, Brier Score |

## Repository Structure

```
Prudentia/
│
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml              # paths, cost assumptions, thresholds
│
├── data/
│   ├── raw/                     # gitignored
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean.py
│   │   └── leakage_audit.py
│   ├── features/
│   │   ├── woe_binning.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_scorecard.py
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   ├── calibrate.py
│   │   └── evaluate.py
│   ├── decisioning/
│   │   ├── cost_matrix.py
│   │   ├── threshold_optimization.py
│   │   └── risk_pricing.py
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   └── reason_codes.py
│   ├── fairness/
│   │   └── disparate_impact.py
│   └── monitoring/
│       └── psi_drift.py
│
├── reports/
│   ├── eda_findings.md
│   ├── credit_policy_memo.md
│   ├── fairness_report.md
│   └── drift_report.md
│
├── app/
│   ├── api.py                   # FastAPI
│   └── dashboard.py              # Streamlit
│
├── artifacts/                    # saved models, calibrators
├── mlruns/                       # MLflow tracking
└── tests/
    ├── test_leakage_audit.py
    ├── test_woe_binning.py
    └── test_decisioning.py
```

## End-to-End Workflow

```
Loan Application Data
          │
          ▼
Leakage-Safe Cleaning
          │
          ▼
Time-Aware Split (train → validate → out-of-time test)
          │
          ▼
Feature Engineering (WOE bins + engineered features)
          │
 ┌────────┼────────────┐
 ▼        ▼             ▼
Scorecard Random Forest XGBoost
 │        │             │
 └────────┴──────┬──────┘
                  ▼
        Model Selection
                  ▼
        Calibration
                  ▼
        Cost-Sensitive Decision
                  ▼
   SHAP + Fairness + Drift Reports
                  ▼
        FastAPI → Streamlit
```

## Example Use Cases

**Risk Scoring**
- What is this applicant's calibrated probability of default?
- Which risk tier and interest rate should this applicant be offered?

**Decision Explainability**
- Why was this applicant denied? (adverse action reason codes)
- Which features most influenced this specific score? (local SHAP explanation)

**Portfolio-Level Analysis**
- How does model-driven risk-based pricing compare to Lending Club's actual assigned grades?
- Are outcomes systematically worse for any age, income, or geographic segment?
- Has the borrower population drifted enough since training to warrant model review?

## Current Status

| Component | Status |
|---|---|
| Data Ingestion & Leakage Audit | 🔲 Planned |
| Time-Aware Train/Test Split | 🔲 Planned |
| WOE Scorecard Model | 🔲 Planned |
| Random Forest & XGBoost Models | 🔲 Planned |
| Calibration | 🔲 Planned |
| Cost-Sensitive Decisioning | 🔲 Planned |
| SHAP Explainability & Reason Codes | 🔲 Planned |
| Fairness Audit | 🔲 Planned |
| Drift Monitoring (PSI) | 🔲 Planned |
| FastAPI Backend | 🔲 Planned |
| Streamlit Frontend | 🔲 Planned |

## Running the Project

Clone the repository
```
git clone https://github.com/<your-username>/Prudentia.git
cd Prudentia
```

Create a virtual environment
```
python -m venv venv
```
Windows
```
venv\Scripts\activate
```
Linux / macOS
```
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Run the pipeline (once implemented)
```
python src/data/load_data.py
python src/data/clean.py
python src/features/woe_binning.py
python src/models/train_scorecard.py
python src/models/train_random_forest.py
python src/models/train_xgboost.py
```

Start the FastAPI backend
```
uvicorn app.api:app --reload
```

Launch the Streamlit dashboard
```
streamlit run app/dashboard.py
```

## Engineering Highlights

One of the primary goals of Prudentia was to build a system that demonstrates production-inspired, model-risk-aware credit modeling practices rather than an isolated prediction notebook.

Some notable design decisions include:

- Time-aware, out-of-time validation instead of a random train/test split, to avoid leaking future information across economic cycles
- Explicit leakage audit distinguishing application-time features from post-origination outcome fields
- Benchmarking a fully interpretable scorecard against modern ML challengers rather than discarding the traditional approach
- Cost-based decision thresholding instead of a default 0.5 cutoff, grounded in documented dollar cost assumptions
- Explainability designed around a real regulatory requirement (ECOA/Reg B adverse action notices), not generic "nice to have" interpretability
- A dedicated fairness audit examining disparate impact across borrower segments
- Post-deployment thinking built in from the start via PSI-based drift monitoring, rather than treating deployment as the finish line

## Future Enhancements

Planned future improvements include:

- Authentication and role-based access for the dashboard
- Automated retraining triggers based on PSI drift thresholds
- Expanded fairness audit with intersectional segment analysis
- Cloud deployment
- CI/CD pipeline with automated model validation checks

## Disclaimer

Prudentia uses historical, publicly available Lending Club loan data and is intended for educational, research, and engineering demonstration purposes only.

This project is not a production credit decisioning system and should not be used for actual lending decisions.

## Author

*Your Name*

Data Scientist | Machine Learning Engineer | Applied AI

GitHub: *your-github-url*
