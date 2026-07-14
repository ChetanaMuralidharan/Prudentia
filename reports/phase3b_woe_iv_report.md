# Phase 3B WOE / IV Feature Engineering Report

## Objective

This phase extends generic preprocessing into credit-risk-specific feature engineering. Features are binned into interpretable risk groups, transformed using Weight of Evidence, and evaluated using Information Value.

## Why WOE and IV Matter

Weight of Evidence transforms feature groups into log-risk values based on the relationship between good and bad loans. Information Value measures how useful a feature is for separating fully paid loans from charged-off/default loans.

## Leakage Control

WOE mappings and IV values are fit only on the training period. The same mappings are then applied to validation and test sets, preserving the time-aware modeling design.

## Feature Matrix Summary

- WOE train shape: **822826 rows × 28 features**
- WOE validation shape: **274711 rows × 28 features**
- WOE test shape: **206101 rows × 28 features**
- Selected features by IV: **13**

## Top Features by Information Value

| feature                      |         iv | predictive_strength   |
|:-----------------------------|-----------:|:----------------------|
| term                         | 0.245383   | Medium                |
| loan_to_income_bin           | 0.11806    | Medium                |
| dti_bin                      | 0.0693706  | Weak                  |
| verification_status          | 0.0517434  | Weak                  |
| loan_amnt_bin                | 0.0358862  | Weak                  |
| high_dti_flag                | 0.0315521  | Weak                  |
| annual_inc_bin               | 0.0301064  | Weak                  |
| inq_last_6mths_bin           | 0.0277211  | Weak                  |
| bankcard_available_ratio_bin | 0.027511   | Weak                  |
| has_recent_inquiry           | 0.0226699  | Weak                  |
| revol_util_bin               | 0.0217161  | Weak                  |
| home_ownership               | 0.0208788  | Weak                  |
| purpose                      | 0.0207805  | Weak                  |
| addr_state                   | 0.0146496  | Not useful            |
| has_mortgage_account         | 0.0126637  | Not useful            |
| open_account_ratio_bin       | 0.0111414  | Not useful            |
| open_acc_bin                 | 0.00766155 | Not useful            |
| revol_bal_to_income_bin      | 0.00744033 | Not useful            |
| emp_length                   | 0.00676483 | Not useful            |
| high_revol_util_flag         | 0.00560882 | Not useful            |
| pub_rec_bin                  | 0.0051767  | Not useful            |
| has_public_record            | 0.00513992 | Not useful            |
| revol_bal_bin                | 0.00375008 | Not useful            |
| delinq_2yrs_bin              | 0.00175004 | Not useful            |
| has_recent_delinquency       | 0.00127156 | Not useful            |


## IV Interpretation Rules

| IV Range | Interpretation |
|---:|:---|
| < 0.02 | Not useful |
| 0.02 - 0.10 | Weak |
| 0.10 - 0.30 | Medium |
| 0.30 - 0.50 | Strong |
| > 0.50 | Very strong / investigate for leakage |

## Selected WOE Features

- `term_woe`
- `loan_to_income_bin_woe`
- `dti_bin_woe`
- `verification_status_woe`
- `loan_amnt_bin_woe`
- `high_dti_flag_woe`
- `annual_inc_bin_woe`
- `inq_last_6mths_bin_woe`
- `bankcard_available_ratio_bin_woe`
- `has_recent_inquiry_woe`
- `revol_util_bin_woe`
- `home_ownership_woe`
- `purpose_woe`

## Modeling Recommendation

Use the WOE-transformed feature matrices for the first scorecard-style logistic regression model. Tree-based models can later be trained on the generic encoded features from Phase 3A for comparison.
