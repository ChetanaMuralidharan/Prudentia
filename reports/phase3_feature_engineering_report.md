# Phase 3 Feature Engineering Report

## Objective

Phase 3 converts the cleaned Lending Club dataset into model-ready training, validation, and test matrices while preserving the out-of-time validation design.

## Feature Engineering Strategy

The feature engineering process focuses on explainable credit-risk variables. Rather than creating opaque transformations, the pipeline adds borrower and loan-level risk indicators such as loan-to-income ratio, revolving-balance-to-income ratio, high-DTI flag, high-utilization flag, delinquency flag, and public-record flag.

## Leakage Control

All preprocessing statistics are fit on the training set only. Median imputation values, categorical encodings, scaling parameters, and outlier caps are learned from the training period and then applied unchanged to validation and test data.

## Model Input Summary

- Raw model input columns before encoding: **88**
- Numeric input columns: **79**
- Categorical input columns: **9**
- Encoded model features: **1101**

## Engineered Features Added

- `loan_to_income`
- `revol_bal_to_income`
- `open_account_ratio`
- `bankcard_available_ratio`
- `has_recent_delinquency`
- `has_public_record`
- `has_mortgage_account`
- `has_recent_inquiry`
- `high_dti_flag`
- `high_revol_util_flag`

## Outlier Treatment

Numeric variables are capped using the 1st and 99th percentile values from the training set. This reduces the impact of extreme values while avoiding validation/test leakage.

- Number of capped numeric features: **74**

## Encoding and Imputation

- Numeric features: median imputation + standard scaling
- Categorical features: most-frequent imputation + one-hot encoding
- Unknown validation/test categories: ignored safely during one-hot encoding

## Saved Artifacts

- `data/processed/X_train.parquet`
- `data/processed/X_valid.parquet`
- `data/processed/X_test.parquet`
- `data/processed/y_train.parquet`
- `data/processed/y_valid.parquet`
- `data/processed/y_test.parquet`
- `artifacts/preprocessor.joblib`
- `artifacts/feature_names.json`
- `artifacts/outlier_caps.json`

## Phase 3 Completion Criteria

Phase 3 is complete when the processed feature matrices, target files, preprocessing pipeline, feature-name artifact, and feature-engineering report are all generated successfully.
