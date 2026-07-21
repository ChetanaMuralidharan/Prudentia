# Phase 6 Cost-Sensitive Decision Layer Report

## Objective

Phase 6 converts calibrated probability-of-default scores into business decisions. The goal is to quantify risk concentration, approval policy tradeoffs, expected loss, and cost-sensitive threshold selection.

## Cost Assumptions

- Loss Given Default (LGD): **60%**
- Exposure at Default (EAD): loan amount
- False rejection opportunity cost proxy: **5%** of interest-adjusted exposure

## Portfolio Summary

- Test population loans: **206,101**
- Observed default rate: **21.03%**
- Total calibrated expected loss: **$464,129,955**
- KS statistic: **0.284**

## Decile and Lift Analysis

The highest-risk decile has a default lift of **2.09x** and captures **20.94%** of all observed defaults.

|   risk_decile |   loans |   defaults |    avg_pd |   realized_default_rate |   avg_loan_amnt |   expected_loss |   default_capture_rate |   cumulative_defaults |   cumulative_default_capture_rate |     lift |
|--------------:|--------:|-----------:|----------:|------------------------:|----------------:|----------------:|-----------------------:|----------------------:|----------------------------------:|---------:|
|             1 |   20611 |       9076 | 0.517632  |               0.440347  |         20484.6 |     1.31704e+08 |              0.209414  |                  9076 |                          0.209414 | 2.09405  |
|             2 |   20610 |       6873 | 0.380681  |               0.333479  |         18218.6 |     8.59752e+07 |              0.158583  |                 15949 |                          0.367997 | 1.58584  |
|             3 |   20610 |       5988 | 0.315842  |               0.290539  |         16226.1 |     6.34682e+07 |              0.138163  |                 21937 |                          0.506161 | 1.38164  |
|             4 |   20610 |       5092 | 0.271029  |               0.247065  |         14630.9 |     4.90772e+07 |              0.11749   |                 27029 |                          0.62365  | 1.1749   |
|             5 |   20610 |       4361 | 0.23052   |               0.211596  |         13607.3 |     3.88427e+07 |              0.100623  |                 31390 |                          0.724273 | 1.00623  |
|             6 |   20610 |       3796 | 0.19614   |               0.184182  |         12907.2 |     3.13139e+07 |              0.0875865 |                 35186 |                          0.81186  | 0.87587  |
|             7 |   20610 |       3075 | 0.163041  |               0.149199  |         12329.1 |     2.48776e+07 |              0.0709506 |                 38261 |                          0.88281  | 0.70951  |
|             8 |   20610 |       2461 | 0.1251    |               0.119408  |         12218.2 |     1.89035e+07 |              0.0567836 |                 40722 |                          0.939594 | 0.567838 |
|             9 |   20610 |       1747 | 0.0907906 |               0.0847647 |         11974   |     1.34408e+07 |              0.0403092 |                 42469 |                          0.979903 | 0.403094 |
|            10 |   20610 |        871 | 0.044092  |               0.042261  |         11861.3 |     6.52654e+06 |              0.0200969 |                 43340 |                          1        | 0.20097  |


## Cost-Optimal Threshold

- Optimal PD threshold: **0.08**
- Approval rate at optimal threshold: **13.13%**
- Approved default rate: **4.88%**
- Total cost at optimal threshold: **$108,880,330**
- Estimated savings vs approve-all baseline: **$307,179,890 (73.83%)**

## Approval Rate Simulation

|   target_approval_rate |   actual_approval_rate |   approved_loans |   rejected_loans |   approved_default_rate |   rejected_default_rate |   approved_expected_loss |   rejected_expected_loss_avoided |   approved_loan_volume |
|-----------------------:|-----------------------:|-----------------:|-----------------:|------------------------:|------------------------:|-------------------------:|---------------------------------:|-----------------------:|
|                   0.1  |               0.100393 |            20691 |           185410 |               0.0424822 |                0.229011 |              6.56402e+06 |                      4.57566e+08 |            2.45412e+08 |
|                   0.15 |               0.151964 |            31320 |           174781 |               0.0532567 |                0.238424 |              1.26788e+07 |                      4.51451e+08 |            3.72534e+08 |
|                   0.2  |               0.203614 |            41965 |           164136 |               0.0644108 |                0.247581 |              2.06188e+07 |                      4.43511e+08 |            5.00739e+08 |
|                   0.25 |               0.261285 |            53851 |           152250 |               0.0752075 |                0.258062 |              3.10415e+07 |                      4.33088e+08 |            6.44817e+08 |
|                   0.3  |               0.306311 |            63131 |           142970 |               0.0835089 |                0.266266 |              4.02241e+07 |                      4.23906e+08 |            7.59038e+08 |
|                   0.35 |               0.357087 |            73596 |           132505 |               0.092274  |                0.275831 |              5.21736e+07 |                      4.11956e+08 |            8.87145e+08 |
|                   0.4  |               0.423889 |            87364 |           118737 |               0.10375   |                0.288672 |              7.0702e+07  |                      3.93428e+08 |            1.06024e+09 |
|                   0.45 |               0.452429 |            93246 |           112855 |               0.108605  |                0.294298 |              7.92776e+07 |                      3.84852e+08 |            1.13626e+09 |
|                   0.5  |               0.503792 |           103832 |           102269 |               0.116611  |                0.305391 |              9.63857e+07 |                      3.67744e+08 |            1.2737e+09  |
|                   0.55 |               0.572229 |           117937 |            88164 |               0.128085  |                0.320244 |              1.21761e+08 |                      3.42369e+08 |            1.46313e+09 |
|                   0.6  |               0.609206 |           125558 |            80543 |               0.133675  |                0.329712 |              1.37951e+08 |                      3.26179e+08 |            1.57044e+09 |
|                   0.65 |               0.653524 |           134692 |            71409 |               0.140914  |                0.341133 |              1.58879e+08 |                      3.05251e+08 |            1.70302e+09 |
|                   0.7  |               0.727454 |           149929 |            56172 |               0.153293  |                0.362405 |              1.98681e+08 |                      2.65449e+08 |            1.93271e+09 |
|                   0.75 |               0.775096 |           159748 |            46353 |               0.161492  |                0.378444 |              2.28748e+08 |                      2.35382e+08 |            2.09301e+09 |
|                   0.8  |               0.800709 |           165027 |            41074 |               0.166282  |                0.387082 |              2.46958e+08 |                      2.17172e+08 |            2.18202e+09 |
|                   0.85 |               0.850423 |           175273 |            30828 |               0.175281  |                0.409303 |              2.8592e+08  |                      1.7821e+08  |            2.36334e+09 |
|                   0.9  |               0.918957 |           189398 |            16703 |               0.18873   |                0.454709 |              3.52297e+08 |                      1.11833e+08 |            2.63252e+09 |
|                   0.95 |               0.956594 |           197155 |             8946 |               0.197297  |                0.496535 |              3.97416e+08 |                      6.67142e+07 |            2.78895e+09 |
|                   1    |               1        |           206101 |                0 |               0.210285  |              nan        |              4.6413e+08  |                      0           |            2.97729e+09 |


## Business Policy Scenarios

Instead of relying only on the mathematically cost-minimizing threshold, three practical underwriting strategies are simulated below. This allows a credit team to choose a policy based on risk appetite and growth goals.

| policy       |   target_approval_rate |   actual_approval_rate |   approved_loans |   rejected_loans |   approved_default_rate |   default_rate_reduction_pct |   approved_loan_volume |   approved_expected_loss |   expected_loss_avoided |   expected_loss_reduction_pct |
|:-------------|-----------------------:|-----------------------:|-----------------:|-----------------:|------------------------:|-----------------------------:|-----------------------:|-------------------------:|------------------------:|------------------------------:|
| Conservative |                    0.4 |               0.423889 |            87364 |           118737 |                0.10375  |                     0.506623 |             1060236425 |              7.0702e+07  |             3.93428e+08 |                      0.847668 |
| Balanced     |                    0.6 |               0.609206 |           125558 |            80543 |                0.133675 |                     0.364315 |             1570435450 |              1.37951e+08 |             3.26179e+08 |                      0.702775 |
| Growth       |                    0.8 |               0.800709 |           165027 |            41074 |                0.166282 |                     0.209256 |             2182023475 |              2.46958e+08 |             2.17172e+08 |                      0.467912 |


## Risk Tier Summary

| risk_tier   |   loans |    avg_pd |   realized_default_rate |   avg_loan_amnt |   expected_loss |   model_price_rate |
|:------------|--------:|----------:|------------------------:|----------------:|----------------:|-------------------:|
| A           |   12483 | 0.0325034 |               0.0334855 |         11824.9 |     2.93097e+06 |          0.069502  |
| B           |   21863 | 0.0752991 |               0.0700727 |         11952.3 |     1.18086e+07 |          0.0951795 |
| C           |   34906 | 0.12609   |               0.120065  |         12166   |     3.21445e+07 |          0.125654  |
| D           |   24029 | 0.178227  |               0.166174  |         12615.9 |     3.24494e+07 |          0.156936  |
| E           |   56648 | 0.248084  |               0.226822  |         14051.2 |     1.19347e+08 |          0.19885   |
| F           |   56172 | 0.415381  |               0.362405  |         18596.1 |     2.65449e+08 |          0.299229  |


## Recommended Credit Policy

The purely cost-optimal threshold is **0.08**, but it approves only **13.13%** of applicants. For a more realistic production policy, the **Balanced** strategy is recommended as an initial operating point.

The Balanced policy approves approximately **60.92%** of applicants, reduces the approved-pool default rate to **13.37%**, and avoids approximately **$326,178,830** in expected loss compared with approving all loans.

## Resume-Ready Quantifiers

- Built a calibrated credit-risk decision layer over **206,101** out-of-time test loans.
- Achieved **0.284 KS statistic** on the held-out test period.
- Highest-risk decile delivered **2.09x default lift**.
- Cost-sensitive thresholding estimated **$307,179,890** loss reduction vs approve-all baseline under **60% LGD** assumption.
- Balanced policy approved **60.92%** of loans with an approved-pool default rate of **13.37%**.
- Balanced policy avoided approximately **$326,178,830** in expected loss versus approving all loans.