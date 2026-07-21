# Phase 4 Model Training and Evaluation Report

## Objective

Phase 4 trains and evaluates credit-risk classification models using the time-aware train, validation, and test split created in earlier phases.

## Models Trained

- **WOE Logistic Regression**: scorecard-style interpretable model trained on full WOE-selected training data.
- **Encoded Logistic Regression**: generic encoded baseline trained on a stratified sample for computational efficiency.
- **Random Forest**: nonlinear benchmark trained on a stratified sample for computational efficiency.
- **XGBoost**: gradient-boosted tree benchmark trained on a stratified sample using CPU-friendly histogram training.

## Why Sampling Was Used

The encoded feature matrix is large because one-hot encoding expands categorical variables into many columns. To keep training practical on a local machine, the generic encoded logistic regression, random forest, and XGBoost benchmarks were trained on stratified samples while preserving the original default/non-default class balance. The WOE scorecard model was trained on the full training population.

## Evaluation Strategy

Models are trained on the training period, threshold-tuned on the validation period, and finally evaluated on the held-out test period. This simulates a realistic out-of-time credit-risk deployment setting.

## Model Performance Summary

| model                               |   valid_roc_auc |   valid_pr_auc |   valid_accuracy |   valid_precision |   valid_recall |   valid_f1 |   valid_true_negatives |   valid_false_positives |   valid_false_negatives |   valid_true_positives |   valid_threshold |   test_roc_auc |   test_pr_auc |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_true_negatives |   test_false_positives |   test_false_negatives |   test_true_positives |   test_threshold |
|:------------------------------------|----------------:|---------------:|-----------------:|------------------:|---------------:|-----------:|-----------------------:|------------------------:|------------------------:|-----------------------:|------------------:|---------------:|--------------:|----------------:|-----------------:|--------------:|----------:|----------------------:|-----------------------:|-----------------------:|----------------------:|-----------------:|
| woe_logistic_regression             |        0.666952 |       0.38751  |         0.572387 |          0.324485 |       0.70412  |   0.444245 |                 110291 |                   97741 |                   19729 |                  46950 |              0.42 |       0.664121 |      0.331957 |        0.580148 |         0.28955  |      0.685579 |  0.407145 |                 89856 |                  72905 |                  13627 |                 29713 |             0.42 |
| encoded_logistic_regression_sampled |        0.697776 |       0.418726 |         0.6075   |          0.34788  |       0.705574 |   0.466001 |                 119840 |                   88192 |                   19632 |                  47047 |              0.46 |       0.685937 |      0.353224 |        0.609934 |         0.306724 |      0.67838  |  0.422443 |                 96307 |                  66454 |                  13939 |                 29401 |             0.46 |
| random_forest_sampled               |        0.69638  |       0.418884 |         0.608836 |          0.347841 |       0.699021 |   0.464528 |                 120644 |                   87388 |                   20069 |                  46610 |              0.48 |       0.686791 |      0.357101 |        0.624844 |         0.311941 |      0.650254 |  0.421621 |                100599 |                  62162 |                  15158 |                 28182 |             0.48 |
| xgboost_sampled                     |        0.708477 |       0.433743 |         0.629247 |          0.361578 |       0.688913 |   0.474247 |                 126925 |                   81107 |                   20743 |                  45936 |              0.17 |       0.697806 |      0.367565 |        0.631986 |         0.318646 |      0.658952 |  0.429568 |                101694 |                  61067 |                  14781 |                 28559 |             0.17 |


## Selected Model

- Best validation ROC-AUC model: **xgboost_sampled**
- Selected operating threshold: **0.17**

## Business Interpretation

ROC-AUC measures ranking quality, while precision and recall describe the approval/rejection tradeoff at a chosen threshold. In credit risk, threshold choice should not rely on accuracy alone because false approvals and false rejections have different financial consequences.

## Phase 4 Completion Criteria

- Models trained successfully
- Validation and test metrics generated
- ROC and precision-recall curves saved
- Threshold analysis completed
- Best model metadata saved
