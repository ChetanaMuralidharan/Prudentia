# Phase 5 Probability Calibration Report

## Objective

Phase 5 evaluates whether the selected Phase 4 model produces reliable default probabilities, not just good risk rankings. This is necessary before using predicted probabilities for expected-loss and cost-sensitive decisioning in Phase 6.

## Base Model

- Selected Phase 4 model: **xgboost_sampled**

## Calibration Methods Compared

- **Raw model probabilities**: uncalibrated output from the selected model
- **Sigmoid calibration / Platt scaling**: logistic calibration of predicted scores
- **Isotonic calibration**: non-parametric monotonic calibration

## Calibration Metrics

| split      | method   |   brier_score |   roc_auc |   pr_auc |   mean_predicted_pd |   observed_default_rate |
|:-----------|:---------|--------------:|----------:|---------:|--------------------:|------------------------:|
| validation | raw      |      0.167641 |  0.708477 | 0.433743 |            0.194209 |                0.242724 |
| validation | sigmoid  |      0.166247 |  0.708477 | 0.433743 |            0.242725 |                0.242724 |
| validation | isotonic |      0.164699 |  0.708814 | 0.429721 |            0.242725 |                0.242724 |
| test       | raw      |      0.153681 |  0.697806 | 0.367565 |            0.186699 |                0.210285 |
| test       | sigmoid  |      0.155098 |  0.697806 | 0.367565 |            0.235878 |                0.210285 |
| test       | isotonic |      0.153517 |  0.697669 | 0.363633 |            0.233488 |                0.210285 |


## Selected Calibration Method

- Selected method: **isotonic**
- Test Brier score improvement vs raw probabilities: **0.11%**

## Business Interpretation

Brier score measures the accuracy of predicted probabilities. Lower values are better. Calibration is important because Phase 6 will use predicted default probabilities to estimate expected loss, approval policy impact, and cost-sensitive thresholds.

## Saved Artifacts

- `artifacts/models/sigmoid_calibrator.joblib`
- `artifacts/models/isotonic_calibrator.joblib`
- `artifacts/models/calibrated_model.joblib`
- `artifacts/models/calibration_metadata.json`
- `data/processed/phase5_calibration/validation_calibrated_probabilities.csv`
- `data/processed/phase5_calibration/test_calibrated_probabilities.csv`
