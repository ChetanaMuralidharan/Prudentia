# EDA Findings & Leakage Audit

- Raw columns: **145**
- Columns after leakage/pricing/misc drop: **98**
- Columns after missingness filter: **81**
- Rows after target resolution filter: **1,303,638**

## Leakage columns dropped (post-origination outcome data)

- `collection_recovery_fee`
- `collections_12_mths_ex_med`
- `debt_settlement_flag`
- `debt_settlement_flag_date`
- `desc`
- `emp_title`
- `funded_amnt_inv`
- `hardship_amount`
- `hardship_dpd`
- `hardship_end_date`
- `hardship_flag`
- `hardship_last_payment_amount`
- `hardship_length`
- `hardship_loan_status`
- `hardship_payoff_balance_amount`
- `hardship_reason`
- `hardship_start_date`
- `hardship_status`
- `hardship_type`
- `id`
- `last_credit_pull_d`
- `last_pymnt_amnt`
- `last_pymnt_d`
- `member_id`
- `next_pymnt_d`
- `out_prncp`
- `out_prncp_inv`
- `recoveries`
- `settlement_amount`
- `settlement_date`
- `settlement_percentage`
- `settlement_status`
- `settlement_term`
- `title`
- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_int`
- `total_rec_late_fee`
- `total_rec_prncp`
- `url`

## Pricing-only columns (reserved for benchmarking, NOT model features)

- `grade`
- `sub_grade`
- `int_rate`
- `installment`

## Misc columns dropped (constant / not useful)

- `policy_code`
- `pymnt_plan`
- `application_type`

## Columns dropped for high missingness (> 95%)

- `annual_inc_joint`
- `dti_joint`
- `verification_status_joint`
- `revol_bal_joint`
- `sec_app_earliest_cr_line`
- `sec_app_inq_last_6mths`
- `sec_app_mort_acc`
- `sec_app_open_acc`
- `sec_app_revol_util`
- `sec_app_open_act_il`
- `sec_app_num_rev_accts`
- `sec_app_chargeoff_within_12_mths`
- `sec_app_collections_12_mths_ex_med`
- `sec_app_mths_since_last_major_derog`
- `deferral_term`
- `payment_plan_start_date`
- `orig_projected_additional_accrued_interest`

## Target balance (post-filter)

- 0 (Fully Paid): 79.93%
- 1 (Charged Off / Default): 20.07%

## Time-aware split sizes

- train: 822,826 rows
- valid: 274,711 rows
- test: 206,101 rows