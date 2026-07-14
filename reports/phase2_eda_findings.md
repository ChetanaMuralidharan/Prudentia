# Phase 2 EDA Findings

## Executive Summary
Phase 2 confirms that default behavior varies materially across borrower profile, loan structure, credit history, and origination period. The validation period has a materially higher default rate (24.27%) than the training period (18.43%), which supports the decision to use an out-of-time validation strategy rather than a random split.

The strongest business patterns are consistent with credit-risk intuition: higher DTI, higher revolving utilization, larger loan amounts, lower income, weaker Lending Club grades, and higher interest-rate bands are associated with higher observed default rates. These patterns should guide Phase 3 feature engineering.

## Objective
Phase 2 validates the time-aware split and frames exploratory analysis around credit-risk business questions, not generic plots. Pricing variables such as `grade`, `sub_grade`, `int_rate`, and `installment` remain excluded from model features, but they are analyzed here as external benchmark signals.

## Time-Aware Split Validation
The dataset is split chronologically using `issue_d`. This avoids random-split leakage where future economic conditions influence training. The split also reveals temporal risk drift: validation defaults are higher than training defaults, so model selection should be based on validation performance, not training fit.

| split   |   rows | start_issue_d       | end_issue_d         |   default_rate_pct |
|:--------|-------:|:--------------------|:--------------------|-------------------:|
| train   | 822826 | 2007-06-01 00:00:00 | 2015-12-01 00:00:00 |              18.43 |
| valid   | 274711 | 2016-01-01 00:00:00 | 2016-12-01 00:00:00 |              24.27 |
| test    | 206101 | 2017-01-01 00:00:00 | 2018-12-01 00:00:00 |              21.03 |

**Interpretation:** The training period default rate is 18.43%, validation is 24.27%, and test is 21.03%. This shift confirms that out-of-time testing is necessary for a realistic credit-risk decisioning engine.

## Default Rate Over Time
|   issue_year |     loans |   defaults |   default_rate_pct |
|-------------:|----------:|-----------:|-------------------:|
|      2007.00 |    251.00 |      45.00 |              17.93 |
|      2008.00 |   1562.00 |     247.00 |              15.81 |
|      2009.00 |   4716.00 |     594.00 |              12.60 |
|      2010.00 |  11536.00 |    1487.00 |              12.89 |
|      2011.00 |  21721.00 |    3297.00 |              15.18 |
|      2012.00 |  53367.00 |    8644.00 |              16.20 |
|      2013.00 | 134793.00 |   21022.00 |              15.60 |
|      2014.00 | 221468.00 |   41056.00 |              18.54 |
|      2015.00 | 373412.00 |   75275.00 |              20.16 |
|      2016.00 | 274711.00 |   66679.00 |              24.27 |
|      2017.00 | 158910.00 |   36389.00 |              22.90 |
|      2018.00 |  47191.00 |    6951.00 |              14.73 |

**Interpretation:** The highest observed annual default rate occurs in 2016 at 24.27%. This supports treating origination time as an important validation and monitoring dimension.

## Default Rate by Loan Purpose
| purpose            |   loans |   defaults |   default_rate_pct |
|:-------------------|--------:|-----------:|-------------------:|
| small_business     |   15010 |       4465 |              29.75 |
| renewable_energy   |     911 |        216 |              23.71 |
| moving             |    9173 |       2151 |              23.45 |
| medical            |   15024 |       3292 |              21.91 |
| house              |    6967 |       1513 |              21.72 |
| debt_consolidation |  757610 |     161058 |              21.26 |
| other              |   74937 |      15867 |              21.17 |
| vacation           |    8732 |       1680 |              19.24 |
| major_purchase     |   28328 |       5304 |              18.72 |
| home_improvement   |   84497 |      15087 |              17.86 |
| credit_card        |  285708 |      48650 |              17.03 |
| car                |   14121 |       2068 |              14.64 |
| wedding            |    2294 |        279 |              12.16 |

**Interpretation:** The highest observed default rate is in `small_business` (29.75%), while the lowest is in `wedding` (12.16%).

## Default Rate by DTI Band
| dti_band   |   loans |   defaults |   default_rate_pct |
|:-----------|--------:|-----------:|-------------------:|
| 40-60      |    4559 |       1494 |              32.77 |
| 30-40      |  117389 |      34516 |              29.40 |
| 60+        |    1539 |        444 |              28.85 |
| 20-30      |  395995 |      91844 |              23.19 |
| 10-20      |  545202 |      97822 |              17.94 |
| 0-10       |  238640 |      35504 |              14.88 |

**Interpretation:** Default risk generally increases across `dti_band` bands, making this a strong Phase 3 feature-engineering candidate.

## Default Rate by Income Band
| income_band   |   loans |   defaults |   default_rate_pct |
|:--------------|--------:|-----------:|-------------------:|
| <30k          |   95861 |      23518 |              24.53 |
| 30k-60k       |  503193 |     111446 |              22.15 |
| 60k-90k       |  381468 |      74207 |              19.45 |
| 90k-120k      |  179844 |      30467 |              16.94 |
| 120k-200k     |  116550 |      18305 |              15.71 |
| 200k+         |   26722 |       3743 |              14.01 |

**Interpretation:** Default risk generally decreases across `income_band` bands, making this a strong Phase 3 feature-engineering candidate.

## Default Rate by Loan Amount Band
| loan_amount_band   |   loans |   defaults |   default_rate_pct |
|:-------------------|--------:|-----------:|-------------------:|
| 30k+               |   78351 |      19276 |              24.60 |
| 20k-30k            |  204194 |      47049 |              23.04 |
| 15k-20k            |  207879 |      47293 |              22.75 |
| 10k-15k            |  275714 |      57510 |              20.86 |
| 5k-10k             |  361909 |      62582 |              17.29 |
| <5k                |  175591 |      27976 |              15.93 |

**Interpretation:** Default risk generally increases across `loan_amount_band` bands, making this a strong Phase 3 feature-engineering candidate.

## Default Rate by Home Ownership
| home_ownership   |   loans |   defaults |   default_rate_pct |
|:-----------------|--------:|-----------:|-------------------:|
| RENT             |  517821 |     120909 |              23.35 |
| OWN              |  139849 |      29016 |              20.75 |
| MORTGAGE         |  645509 |     111675 |              17.30 |

**Interpretation:** The highest observed default rate is in `RENT` (23.35%), while the lowest is in `MORTGAGE` (17.30%).

## Default Rate by Verification Status
| verification_status   |   loans |   defaults |   default_rate_pct |
|:----------------------|--------:|-----------:|-------------------:|
| Verified              |  407689 |      97593 |              23.94 |
| Source Verified       |  503737 |     106388 |              21.12 |
| Not Verified          |  392212 |      57705 |              14.71 |

**Interpretation:** The highest observed default rate is in `Verified` (23.94%), while the lowest is in `Not Verified` (14.71%).

## Default Rate by Employment Length
| emp_length   |   loans |   defaults |   default_rate_pct |
|:-------------|--------:|-----------:|-------------------:|
| nan          |   75457 |      20397 |              27.03 |
| < 1 year     |  104552 |      21622 |              20.68 |
| 1 year       |   85678 |      17692 |              20.65 |
| 3 years      |  104204 |      20937 |              20.09 |
| 8 years      |   59127 |      11864 |              20.07 |
| 9 years      |   49504 |       9912 |              20.02 |
| 2 years      |  117825 |      23482 |              19.93 |
| 4 years      |   78033 |      15526 |              19.90 |
| 5 years      |   81623 |      16077 |              19.70 |
| 7 years      |   58148 |      11385 |              19.58 |
| 6 years      |   60934 |      11859 |              19.46 |
| 10+ years    |  428553 |      80933 |              18.89 |

**Interpretation:** Missing or short employment history shows elevated observed risk; employment length should be handled carefully in Phase 3 because missingness itself may contain signal.

## Default Rate by Revolving Utilization Band
| revol_util_band   |   loans |   defaults |   default_rate_pct |
|:------------------|--------:|-----------:|-------------------:|
| 100+              |    4537 |       1217 |              26.82 |
| 80-100            |  186589 |      42505 |              22.78 |
| 60-80             |  317552 |      69808 |              21.98 |
| nan               |     810 |        168 |              20.74 |
| 40-60             |  360967 |      74454 |              20.63 |
| 20-40             |  284863 |      51385 |              18.04 |
| 0-20              |  148320 |      22149 |              14.93 |

**Interpretation:** Default risk generally increases across `revol_util_band` bands, making this a strong Phase 3 feature-engineering candidate.

## Default Rate by Delinquencies in Last 2 Years
|   delinq_2yrs |      loans |   defaults |   default_rate_pct |
|--------------:|-----------:|-----------:|-------------------:|
|          7.00 |    1251.00 |     316.00 |              25.26 |
|          5.00 |    4062.00 |     969.00 |              23.86 |
|          4.00 |    7865.00 |    1856.00 |              23.60 |
|          3.00 |   17752.00 |    4130.00 |              23.26 |
|          8.00 |     737.00 |     170.00 |              23.07 |
|          6.00 |    2219.00 |     510.00 |              22.98 |
|          2.00 |   48863.00 |   10951.00 |              22.41 |
|          1.00 |  166919.00 |   35019.00 |              20.98 |
|          0.00 | 1052511.00 |  207404.00 |              19.71 |

**Interpretation:** Prior delinquencies are associated with higher risk and should be retained as a candidate bureau-history feature.

## Default Rate by Public Records
|   pub_rec |      loans |   defaults |   default_rate_pct |
|----------:|-----------:|-----------:|-------------------:|
|      4.00 |    2533.00 |     633.00 |              24.99 |
|      2.00 |   23745.00 |    5673.00 |              23.89 |
|      6.00 |     616.00 |     147.00 |              23.86 |
|      3.00 |    7167.00 |    1643.00 |              22.92 |
|      1.00 |  185109.00 |   42037.00 |              22.71 |
|      5.00 |    1224.00 |     275.00 |              22.47 |
|      0.00 | 1082535.00 |  211126.00 |              19.50 |

**Interpretation:** Public record counts show a risk separation versus borrowers with no public records, but sparse high-count categories should likely be capped or binned in Phase 3.

## Default Rate by Lending Club Grade
| grade   |   loans |   defaults |   default_rate_pct |
|:--------|--------:|-----------:|-------------------:|
| G       |    8951 |       4482 |              50.07 |
| F       |   31485 |      14265 |              45.31 |
| E       |   91574 |      35368 |              38.62 |
| D       |  195288 |      59449 |              30.44 |
| C       |  369937 |      83271 |              22.51 |
| B       |  380158 |      51083 |              13.44 |
| A       |  226245 |      13768 |               6.09 |

**Interpretation:** Lending Club grades strongly rank-order realized default risk, validating them as pricing benchmarks. They should not be used as model inputs because this project is building an independent decision engine.

## Default Rate by Lending Club Sub-Grade
| sub_grade   |   loans |   defaults |   default_rate_pct |
|:------------|--------:|-----------:|-------------------:|
| G5          |    1082 |        580 |              53.60 |
| G3          |    1593 |        820 |              51.48 |
| G4          |    1258 |        645 |              51.27 |
| F5          |    3880 |       1917 |              49.41 |
| G2          |    2099 |       1032 |              49.17 |
| G1          |    2919 |       1405 |              48.13 |
| F4          |    4783 |       2294 |              47.96 |
| F2          |    7064 |       3213 |              45.48 |
| F3          |    5992 |       2708 |              45.19 |
| F1          |    9766 |       4133 |              42.32 |
| E5          |   14058 |       5874 |              41.78 |
| E4          |   15374 |       6185 |              40.23 |
| E3          |   17958 |       6986 |              38.90 |
| E2          |   20961 |       7902 |              37.70 |
| E1          |   23223 |       8421 |              36.26 |
| D5          |   29105 |       9706 |              33.35 |
| D4          |   34567 |      11228 |              32.48 |
| D3          |   38183 |      11666 |              30.55 |
| D2          |   43483 |      12898 |              29.66 |
| D1          |   49950 |      13951 |              27.93 |

**Interpretation:** Sub-grades provide an even more granular benchmark risk ranking, useful for comparing the independent model against Lending Club's assigned risk tiers.

## Default Rate by Interest Rate Band
| interest_rate_band   |   loans |   defaults |   default_rate_pct |
|:---------------------|--------:|-----------:|-------------------:|
| 24+                  |   38232 |      17674 |              46.23 |
| 20-24                |   65774 |      24845 |              37.77 |
| 16-20                |  221336 |      69510 |              31.40 |
| 12-16                |  409511 |      88451 |              21.60 |
| 8-12                 |  368764 |      49478 |              13.42 |
| <8                   |  200021 |      11728 |               5.86 |

**Interpretation:** Interest-rate bands strongly align with realized default risk and are useful for benchmarking, but they are excluded from training features to avoid building a model that simply learns Lending Club pricing decisions.

## Top Numeric Correlations with Default
| feature              |   correlation_with_default |   absolute_correlation |
|:---------------------|---------------------------:|-----------------------:|
| acc_open_past_24mths |                       0.10 |                   0.10 |
| all_util             |                       0.09 |                   0.09 |
| dti                  |                       0.09 |                   0.09 |
| num_tl_op_past_12m   |                       0.09 |                   0.09 |
| bc_open_to_buy       |                      -0.08 |                   0.08 |
| open_rv_24m          |                       0.08 |                   0.08 |
| avg_cur_bal          |                      -0.08 |                   0.08 |
| tot_hi_cred_lim      |                      -0.08 |                   0.08 |
| mort_acc             |                      -0.08 |                   0.08 |
| total_bc_limit       |                      -0.07 |                   0.07 |
| tot_cur_bal          |                      -0.07 |                   0.07 |
| num_actv_rev_tl      |                       0.07 |                   0.07 |
| num_rev_tl_bal_gt_0  |                       0.07 |                   0.07 |
| percent_bc_gt_75     |                       0.07 |                   0.07 |
| bc_util              |                       0.07 |                   0.07 |
| funded_amnt          |                       0.07 |                   0.07 |
| loan_amnt            |                       0.07 |                   0.07 |
| inq_last_6mths       |                       0.07 |                   0.07 |
| open_rv_12m          |                       0.06 |                   0.06 |
| revol_util           |                       0.06 |                   0.06 |

**Interpretation:** The correlation table is used only as a directional screening tool. Credit-risk feature selection should rely more heavily on WOE/IV, validation performance, calibration, and stability.

## Remaining Missingness After Cleaning
| feature                        |   missing_pct |
|:-------------------------------|--------------:|
| mths_since_last_record         |         82.99 |
| mths_since_recent_bc_dlq       |         76.29 |
| mths_since_last_major_derog    |         73.74 |
| il_util                        |         66.81 |
| mths_since_recent_revol_delinq |         66.57 |
| mths_since_rcnt_il             |         62.68 |
| all_util                       |         61.68 |
| inq_last_12m                   |         61.68 |
| total_cu_tl                    |         61.68 |
| open_acc_6m                    |         61.68 |
| total_bal_il                   |         61.68 |
| open_rv_12m                    |         61.68 |
| open_rv_24m                    |         61.68 |
| open_il_12m                    |         61.68 |
| max_bal_bc                     |         61.68 |
| open_il_24m                    |         61.68 |
| inq_fi                         |         61.68 |
| open_act_il                    |         61.68 |
| mths_since_last_delinq         |         50.46 |
| mths_since_recent_inq          |         13.01 |

**Interpretation:** Remaining missingness is limited enough for Phase 3 preprocessing. Features with meaningful missingness should receive explicit missing indicators or robust imputation rules.

## Implications for Phase 3 Feature Engineering
Phase 3 should convert the EDA patterns into stable, model-ready features. Recommended candidates include:

- DTI bands and/or transformed DTI
- Revolving utilization bands and capped utilization
- Income bands and missing/income outlier handling
- Loan amount bands
- Delinquency and public-record indicators
- Employment-length normalization, including an explicit missing category
- Home ownership and verification-status encodings
- Bureau/history variables with WOE/IV ranking

Pricing-only variables (`grade`, `sub_grade`, `int_rate`, `installment`) should remain excluded from model training and used only for benchmarking model behavior against Lending Club's historical risk/pricing decisions.

## Figures Generated
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/01_loan_volume_over_time.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/02_default_rate_by_year.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_loan_purpose.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_dti_band.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_income_band.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_loan_amount_band.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_home_ownership.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_verification_status.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_employment_length.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_revolving_utilization_band.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_delinquencies_in_last_2_years.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_public_records.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_lending_club_grade.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_lending_club_sub_grade.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/03_default_rate_by_interest_rate_band.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/04_top_numeric_correlations.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/04_remaining_missingness_after_cleaning.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/04_loan_amnt_distribution_by_split.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/04_annual_inc_distribution_by_split.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/04_dti_distribution_by_split.png`
- `C:/Users/mchet/Documents/Work/Projects/Project 12 - Prudentia/reports/figures/04_revol_util_distribution_by_split.png`
