# Phase 8 Drift Monitoring Report

## Executive Summary

Phase 8 monitors applicant-population changes, model-score movement, lending
decisions, risk-tier migration, calibration and model discrimination.

- Overall recommendation: **INVESTIGATE**
- Decision threshold: **0.0800**
- Reference population rows: **274,711**
- Monitoring population rows: **206,101**
- Monitoring quarters: **8**
- Mature performance quarters: **6**
- Provisional performance quarters: **2**
- Core business features monitored: **28**
- Encoded model features monitored: **250 of 1101**
- Alerts generated: **4**

## Monitoring Windows

- Reference: **2016-01-01 to 2016-12-01**
- Monitoring: **2017-01-01 to 2018-12-01**
- Performance maturity cutoff: **2018-06-30**

## Outcome Maturity Policy

Population, feature, score, approval-rate and risk-tier monitoring are
performed for every quarter.

Observed default rate, calibration, ROC-AUC, Gini, KS and Brier score are
reported only for sufficiently matured loan vintages. Loans issued after
**2018-06-30** are marked as immature and excluded from
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

| quarter   |   score_psi | status   |   reference_average_pd |   current_average_pd |   current_rows |
|:----------|------------:|:---------|-----------------------:|---------------------:|---------------:|
| 2017Q1    |    0.010398 | STABLE   |               0.242725 |             0.228329 |          44701 |
| 2017Q2    |    0.004055 | STABLE   |               0.242725 |             0.233994 |          42045 |
| 2017Q3    |    0.001972 | STABLE   |               0.242725 |             0.240994 |          40870 |
| 2017Q4    |    0.011963 | STABLE   |               0.242725 |             0.235809 |          31294 |
| 2018Q1    |    0.020965 | STABLE   |               0.242725 |             0.233959 |          19997 |
| 2018Q2    |    0.023388 | STABLE   |               0.242725 |             0.233688 |          15658 |
| 2018Q3    |    0.044391 | STABLE   |               0.242725 |             0.219733 |           8065 |
| 2018Q4    |    0.063964 | STABLE   |               0.242725 |             0.212848 |           3471 |

## Most Drifted Business Features

Raw and transformed versions of the same variable are grouped into one
underlying business feature for alert counting.

| original_feature         |   maximum_psi |   average_psi |   quarters_in_alert |   monitored_columns |   latest_psi | latest_status   |
|:-------------------------|--------------:|--------------:|--------------------:|--------------------:|-------------:|:----------------|
| bc_util                  |      0.27547  |      0.12411  |                   1 |                   1 |     0.27547  | ALERT           |
| bankcard_available_ratio |      0.271379 |      0.124646 |                   1 |                   1 |     0.271379 | ALERT           |
| revol_util               |      0.262574 |      0.12231  |                   2 |                   2 |     0.262574 | ALERT           |
| percent_bc_gt_75         |      0.212209 |      0.093239 |                   0 |                   1 |     0.212209 | WATCH           |
| bc_open_to_buy           |      0.206941 |      0.085935 |                   0 |                   2 |     0.206941 | WATCH           |
| pub_rec                  |      0.184434 |      0.056647 |                   0 |                   2 |     0.184434 | WATCH           |
| all_util                 |      0.168585 |      0.084082 |                   0 |                   1 |     0.168585 | WATCH           |
| verification_status      |      0.132271 |      0.025022 |                   0 |                   4 |     0.132271 | WATCH           |
| loan_to_income           |      0.110964 |      0.045449 |                   0 |                   1 |     0.110964 | WATCH           |
| num_actv_rev_tl          |      0.100809 |      0.045113 |                   0 |                   1 |     0.100809 | WATCH           |
| num_rev_tl_bal_gt_0      |      0.100446 |      0.045935 |                   0 |                   1 |     0.100446 | WATCH           |
| revol_bal                |      0.078945 |      0.039696 |                   0 |                   3 |     0.078304 | STABLE          |
| total_bc_limit           |      0.077859 |      0.030016 |                   0 |                   2 |     0.077859 | STABLE          |
| purpose                  |      0.072028 |      0.006386 |                   0 |                   9 |     0.041387 | STABLE          |
| funded_amnt              |      0.068914 |      0.03212  |                   0 |                   2 |     0.068914 | STABLE          |
| loan_amnt                |      0.068914 |      0.03212  |                   0 |                   2 |     0.068914 | STABLE          |
| total_rev_hi_lim         |      0.062358 |      0.022072 |                   0 |                   2 |     0.062358 | STABLE          |
| num_actv_bc_tl           |      0.061875 |      0.02504  |                   0 |                   1 |     0.061875 | STABLE          |
| max_bal_bc               |      0.059065 |      0.028278 |                   0 |                   1 |     0.059065 | STABLE          |
| dti                      |      0.058243 |      0.029447 |                   0 |                   2 |     0.058243 | STABLE          |

## Quarterly Portfolio Metrics

| quarter   |   loans | performance_eligible   | maturity_status   |   approval_rate |   observed_default_rate |   approved_default_rate |   rejected_default_rate |   average_pd |   calibration_gap |   high_risk_share |
|:----------|--------:|:-----------------------|:------------------|----------------:|------------------------:|------------------------:|------------------------:|-------------:|------------------:|------------------:|
| 2017Q1    |   44701 | True                   | MATURE            |        0.124986 |                0.229122 |                0.054591 |                0.254052 |     0.228329 |         -0.000793 |          0.540726 |
| 2017Q2    |   42045 | True                   | MATURE            |        0.121085 |                0.236889 |                0.051267 |                0.262461 |     0.233994 |         -0.002895 |          0.560233 |
| 2017Q3    |   40870 | True                   | MATURE            |        0.117837 |                0.236946 |                0.058347 |                0.260803 |     0.240994 |          0.004048 |          0.568412 |
| 2017Q4    |   31294 | True                   | MATURE            |        0.13779  |                0.207803 |                0.049165 |                0.233155 |     0.235809 |          0.028005 |          0.547709 |
| 2018Q1    |   19997 | True                   | MATURE            |        0.148272 |                0.187028 |                0.044519 |                0.211837 |     0.233959 |          0.046931 |          0.537481 |
| 2018Q2    |   15658 | True                   | MATURE            |        0.150466 |                0.165091 |                0.041171 |                0.18704  |     0.233688 |          0.068596 |          0.534232 |
| 2018Q3    |    8065 | False                  | IMMATURE          |        0.167266 |              nan        |              nan        |              nan        |     0.219733 |        nan        |          0.493366 |
| 2018Q4    |    3471 | False                  | IMMATURE          |        0.16998  |              nan        |              nan        |              nan        |     0.212848 |        nan        |          0.469893 |

## Quarterly Model Performance

Performance metrics are shown only for sufficiently matured vintages.

| quarter   |   loans | performance_eligible   | maturity_status   |    roc_auc |       gini |   ks_statistic |   brier_score |
|:----------|--------:|:-----------------------|:------------------|-----------:|-----------:|---------------:|--------------:|
| 2017Q1    |   44701 | True                   | MATURE            |   0.699112 |   0.398224 |       0.288473 |      0.161182 |
| 2017Q2    |   42045 | True                   | MATURE            |   0.698691 |   0.397381 |       0.281594 |      0.164518 |
| 2017Q3    |   40870 | True                   | MATURE            |   0.70087  |   0.40174  |       0.290009 |      0.164177 |
| 2017Q4    |   31294 | True                   | MATURE            |   0.700904 |   0.401807 |       0.289292 |      0.151789 |
| 2018Q1    |   19997 | True                   | MATURE            |   0.697694 |   0.395388 |       0.286002 |      0.144294 |
| 2018Q2    |   15658 | True                   | MATURE            |   0.695028 |   0.390055 |       0.287717 |      0.135771 |
| 2018Q3    |    8065 | False                  | IMMATURE          | nan        | nan        |     nan        |    nan        |
| 2018Q4    |    3471 | False                  | IMMATURE          | nan        | nan        |     nan        |    nan        |

## Risk-Tier Drift

| quarter   |   risk_tier_psi | status   |
|:----------|----------------:|:---------|
| 2017Q1    |        0.008367 | STABLE   |
| 2017Q2    |        0.003369 | STABLE   |
| 2017Q3    |        0.001645 | STABLE   |
| 2017Q4    |        0.010392 | STABLE   |
| 2018Q1    |        0.019217 | STABLE   |
| 2018Q2    |        0.021283 | STABLE   |
| 2018Q3    |        0.044838 | STABLE   |
| 2018Q4    |        0.063832 | STABLE   |

## Monitoring Alerts

| quarter   | alert_type        | metric                       |   observed_value |   threshold | severity    | recommended_action                                                                                                          |
|:----------|:------------------|:-----------------------------|-----------------:|------------:|:------------|:----------------------------------------------------------------------------------------------------------------------------|
| 2018Q2    | Calibration drift | absolute_calibration_gap     |         0.068596 |        0.05 | HIGH        | Review probability calibration.                                                                                             |
| 2018Q3    | Outcome maturity  | performance_eligible         |         0        |        1    | PROVISIONAL | Population and score monitoring remain valid, but outcome-based performance metrics are deferred until the vintage matures. |
| 2018Q4    | Feature drift     | distinct_alert_feature_count |         3        |        3    | HIGH        | Review the most drifted business features.                                                                                  |
| 2018Q4    | Outcome maturity  | performance_eligible         |         0        |        1    | PROVISIONAL | Population and score monitoring remain valid, but outcome-based performance metrics are deferred until the vintage matures. |

## Quarterly Recommendations

| quarter   |   distinct_alert_feature_count |   critical_alerts |   high_alerts |   watch_alerts |   provisional_alerts | recommendation   | rationale                                                                                    |
|:----------|-------------------------------:|------------------:|--------------:|---------------:|---------------------:|:-----------------|:---------------------------------------------------------------------------------------------|
| 2017Q1    |                              0 |                 0 |             0 |              0 |                    0 | STABLE           | No material monitoring threshold was breached.                                               |
| 2017Q2    |                              0 |                 0 |             0 |              0 |                    0 | STABLE           | No material monitoring threshold was breached.                                               |
| 2017Q3    |                              0 |                 0 |             0 |              0 |                    0 | STABLE           | No material monitoring threshold was breached.                                               |
| 2017Q4    |                              0 |                 0 |             0 |              0 |                    0 | STABLE           | No material monitoring threshold was breached.                                               |
| 2018Q1    |                              0 |                 0 |             0 |              0 |                    0 | STABLE           | No material monitoring threshold was breached.                                               |
| 2018Q2    |                              0 |                 0 |             1 |              0 |                    0 | INVESTIGATE      | Moderate drift requires root-cause investigation.                                            |
| 2018Q3    |                              0 |                 0 |             0 |              0 |                    1 | PROVISIONAL      | Outcome-based performance is deferred because the loan vintage has not sufficiently matured. |
| 2018Q4    |                              3 |                 0 |             1 |              0 |                    1 | INVESTIGATE      | Moderate drift requires root-cause investigation.                                            |

## Final Recommendation

**INVESTIGATE**

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
