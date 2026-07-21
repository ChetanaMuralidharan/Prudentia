# Phase 7 Explainability and Fairness Report

## Objective

Phase 7 explains the selected credit-risk model and audits the downstream approval policy. The goal is to make the model usable in a regulated lending workflow: global drivers, loan-level adverse action reason codes, and approval-rate fairness checks.

## Model and Policy Inputs

- Base model explained: **xgboost_sampled**
- Decision threshold used for denial reason generation: **0.08**
- Local denied-applicant explanations generated: **225**
- Fairness minimum group size: **500 loans**
- Disparate impact alert threshold: **80%**

## Global SHAP Feature Importance

| feature                       |   mean_abs_shap | plain_english_reason                                        |
|:------------------------------|----------------:|:------------------------------------------------------------|
| categorical__term_ 36 months  |      0.0576878  | Longer loan term increases repayment risk                   |
| numeric__loan_to_income       |      0.0266421  | Loan amount is high relative to income                      |
| numeric__acc_open_past_24mths |      0.0232111  | Elevated risk associated with numeric  acc open past 24mths |
| numeric__dti                  |      0.0196325  | High debt-to-income ratio                                   |
| numeric__bc_open_to_buy       |      0.014905   | Elevated risk associated with numeric  bc open to buy       |
| numeric__num_tl_op_past_12m   |      0.0136444  | Elevated risk associated with numeric  num tl op past 12m   |
| numeric__tot_hi_cred_lim      |      0.0121487  | Elevated risk associated with numeric  tot hi cred lim      |
| numeric__revol_util           |      0.0120873  | High revolving credit utilization                           |
| numeric__total_bc_limit       |      0.0104042  | Elevated risk associated with numeric  total bc limit       |
| categorical__term_ 60 months  |      0.00984122 | Longer loan term increases repayment risk                   |


## Adverse Action Reason Code Logic

For denied applicants, the script selects the highest positive SHAP contributors, meaning the features that pushed the predicted probability of default upward. These are mapped to plain-English reason codes suitable for a portfolio project demonstration.

|   sample_index |   calibrated_pd | reason_1                                                     | reason_2                                                    | reason_3                                                      | reason_4                                                    |
|---------------:|----------------:|:-------------------------------------------------------------|:------------------------------------------------------------|:--------------------------------------------------------------|:------------------------------------------------------------|
|              0 |        0.405383 | Longer loan term increases repayment risk                    | High funded loan amount                                     | Longer loan term increases repayment risk                     | High requested loan amount                                  |
|              1 |        0.500513 | Elevated risk associated with numeric  mths since recent inq | Elevated risk associated with numeric  tot hi cred lim      | Lower income relative to requested credit                     | Elevated risk associated with numeric  mths since recent bc |
|              2 |        0.116213 | Elevated risk associated with numeric  acc open past 24mths  | High debt-to-income ratio                                   | Elevated risk associated with categorical  emp length None    | Loan amount is high relative to income                      |
|              3 |        0.180587 | Longer loan term increases repayment risk                    | Elevated risk associated with numeric  mo sin old rev tl op | Elevated risk associated with numeric  mths since recent bc   | Longer loan term increases repayment risk                   |
|              4 |        0.202015 | Elevated risk associated with numeric  total bc limit        | High revolving credit balance                               | Elevated risk associated with numeric  open il 24m            | Elevated risk associated with numeric  total rev hi lim     |
|              5 |        0.37856  | Longer loan term increases repayment risk                    | High debt-to-income ratio                                   | Elevated risk associated with numeric  mths since recent inq  | Longer loan term increases repayment risk                   |
|              7 |        0.183754 | Longer loan term increases repayment risk                    | Elevated risk associated with numeric  acc open past 24mths | Elevated risk associated with numeric  mths since last delinq | Longer loan term increases repayment risk                   |
|              8 |        0.184326 | Elevated risk associated with numeric  num tl op past 12m    | Elevated risk associated with numeric  mths since recent bc | Elevated risk associated with numeric  acc open past 24mths   | Elevated risk associated with numeric  bc open to buy       |
|              9 |        0.313281 | Elevated risk associated with numeric  bc open to buy        | Elevated risk associated with numeric  mo sin old rev tl op | Elevated risk associated with numeric  total bc limit         | Elevated risk associated with numeric  total rev hi lim     |
|             10 |        0.313281 | Loan amount is high relative to income                       | Elevated risk associated with numeric  bc open to buy       | Loan purpose associated with elevated default risk            | Elevated risk associated with numeric  total bc limit       |


## Fairness / Disparate Impact Audit

- Segment groups evaluated: **68**
- Groups flagged below the 80% rule: **59**

| segment             | group              |   loans |   approval_rate |   reference_approval_rate |   disparate_impact_ratio | flag_80pct_rule   |   observed_default_rate |   avg_pd |   overall_approval_rate |
|:--------------------|:-------------------|--------:|----------------:|--------------------------:|-------------------------:|:------------------|------------------------:|---------:|------------------------:|
| dti_band            | DTI >30            |   21663 |       0.0207266 |                  0.235885 |                0.0878671 | True              |                0.287449 | 0.323309 |                0.131324 |
| purpose_segment     | small_business     |    2182 |       0.0293309 |                  0.236069 |                0.124247  | True              |                0.355637 | 0.319104 |                0.131324 |
| verification_status | Verified           |   53515 |       0.0526768 |                  0.23215  |                0.226908  | True              |                0.279492 | 0.287918 |                0.131324 |
| income_band         | Low income         |   53825 |       0.0493637 |                  0.207882 |                0.23746   | True              |                0.247747 | 0.26808  |                0.131324 |
| dti_band            | DTI 20-30          |   60121 |       0.0679962 |                  0.235885 |                0.288259  | True              |                0.243592 | 0.263045 |                0.131324 |
| state               | MS                 |    1201 |       0.0882598 |                  0.239183 |                0.369006  | True              |                0.285595 | 0.256794 |                0.131324 |
| state               | ID                 |     756 |       0.0899471 |                  0.239183 |                0.37606   | True              |                0.167989 | 0.254584 |                0.131324 |
| verification_status | Source Verified    |   81787 |       0.0955042 |                  0.23215  |                0.41139   | True              |                0.21276  | 0.23789  |                0.131324 |
| state               | OK                 |    1806 |       0.0991141 |                  0.239183 |                0.414386  | True              |                0.241971 | 0.259155 |                0.131324 |
| state               | MI                 |    5492 |       0.10142   |                  0.239183 |                0.424028  | True              |                0.219228 | 0.24805  |                0.131324 |
| state               | AR                 |    1473 |       0.101833  |                  0.239183 |                0.425754  | True              |                0.253225 | 0.251145 |                0.131324 |
| state               | AL                 |    2447 |       0.102575  |                  0.239183 |                0.428855  | True              |                0.236208 | 0.25108  |                0.131324 |
| purpose_segment     | debt_consolidation |  113234 |       0.102204  |                  0.236069 |                0.432942  | True              |                0.220384 | 0.248919 |                0.131324 |
| state               | NY                 |   15934 |       0.103678  |                  0.239183 |                0.433466  | True              |                0.252102 | 0.246419 |                0.131324 |
| purpose_segment     | moving             |    1832 |       0.103712  |                  0.236069 |                0.439328  | True              |                0.259825 | 0.230862 |                0.131324 |
| state               | NE                 |    1008 |       0.107143  |                  0.239183 |                0.447954  | True              |                0.24504  | 0.244318 |                0.131324 |
| state               | FL                 |   15292 |       0.108423  |                  0.239183 |                0.453305  | True              |                0.228289 | 0.243961 |                0.131324 |
| state               | OH                 |    6208 |       0.109858  |                  0.239183 |                0.459307  | True              |                0.194588 | 0.240723 |                0.131324 |
| home_ownership      | RENT               |   76637 |       0.0809139 |                  0.172749 |                0.46839   | True              |                0.261284 | 0.260035 |                0.131324 |
| state               | NV                 |    3627 |       0.113593  |                  0.239183 |                0.474919  | True              |                0.214227 | 0.241532 |                0.131324 |
| state               | WI                 |    2789 |       0.113661  |                  0.239183 |                0.475205  | True              |                0.221943 | 0.240702 |                0.131324 |
| state               | UT                 |    1778 |       0.114173  |                  0.239183 |                0.477347  | True              |                0.14342  | 0.232811 |                0.131324 |
| state               | ME                 |     630 |       0.114286  |                  0.239183 |                0.477818  | True              |                0.122222 | 0.229766 |                0.131324 |
| state               | KY                 |    1942 |       0.114315  |                  0.239183 |                0.477941  | True              |                0.205458 | 0.237157 |                0.131324 |
| state               | PA                 |    6608 |       0.114558  |                  0.239183 |                0.478957  | True              |                0.210654 | 0.238888 |                0.131324 |
| state               | IN                 |    3597 |       0.116486  |                  0.239183 |                0.487017  | True              |                0.217403 | 0.242468 |                0.131324 |
| state               | DE                 |     606 |       0.117162  |                  0.239183 |                0.489842  | True              |                0.224422 | 0.23883  |                0.131324 |
| state               | MO                 |    3206 |       0.117592  |                  0.239183 |                0.491641  | True              |                0.226138 | 0.234988 |                0.131324 |
| state               | NC                 |    5931 |       0.117687  |                  0.239183 |                0.492037  | True              |                0.218513 | 0.240342 |                0.131324 |
| state               | NM                 |    1080 |       0.118519  |                  0.239183 |                0.495515  | True              |                0.227778 | 0.233439 |                0.131324 |


## Important Interpretation Note

This is a protected-adjacent fairness audit, not a legal fair-lending certification. The Lending Club dataset does not provide protected classes such as race or sex. Therefore, the report uses available proxies and business segments such as income band, geography/state, credit-history band, DTI band, FICO band, and loan purpose. Any flagged group should be treated as a review trigger requiring business justification, documentation, and possible policy mitigation.

## Saved Artifacts

- `data/processed/phase7_explainability/global_shap_importance.csv`
- `data/processed/phase7_explainability/local_adverse_action_reason_codes.csv`
- `data/processed/phase7_explainability/fairness_disparate_impact_audit.csv`
- `reports/figures/phase7_global_shap_importance.png`
- `reports/figures/phase7_disparate_impact.png`