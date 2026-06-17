# RESULTS.md — Mule Account Detection MVP
## Validated Metrics Table

> All results are from 5-fold stratified cross-validation (OOF), `random_state=42`.  
> No numbers have been fabricated or adjusted post-hoc.  
> PRD §4 integrity constraint: no F3924/F3912/F2230 used as model inputs.

---

## Model Comparison Table

| Model | PR-AUC | Precision @0.25 | Recall @0.25 | F1 @0.25 | Notes |
|---|---|---|---|---|---|
| **Rule-based (F3912 alone)** | 0.9398 | 0.9634 | 0.9753 | 0.9693 | ⚠️ LEAKAGE — F3912 is the bank's own flag, excluded from ML model |
| **Baseline (HistGradientBoosting)** | **0.5405** | **0.494** | **0.506** | **0.500** | ✅ Validated baseline — **headline result** |
| Extended Ensemble (XGB+LGBM+CAT) | 0.5128 | 0.134 | 0.864 | 0.231 | Lower PR-AUC than baseline; high recall at cost of precision |

> **Headline model: Baseline HistGB** (PR-AUC 0.5405 > Ensemble 0.5128 per PRD §2).  
> Random baseline PR-AUC at this imbalance ≈ **0.0089** → our model is **≈ 61× better than random**.

---

## Threshold Analysis (Baseline HistGB)

| Threshold | Precision | Recall | F1 | Flagged | True Positives |
|---|---|---|---|---|---|
| 0.20 | 0.489 | 0.543 | 0.515 | 90 | 44 / 81 |
| **0.25** | **0.494** | **0.506** | **0.500** | **83** | **41 / 81** |
| 0.30 | 0.539 | 0.506 | 0.522 | 76 | 41 / 81 |

---

## Extended Ensemble Details

| Base Model | OOF PR-AUC | Library Version |
|---|---|---|
| XGBoost | 0.5201 | xgboost 3.2.0 |
| LightGBM | 0.5246 | lightgbm 4.6.0 |
| CatBoost | 0.4960 | catboost 1.2.10 |
| **Meta-learner (LogReg stack)** | **0.5128** | sklearn |

The advanced ensemble path was taken (xgboost/lightgbm/catboost all installed).  
The meta-learner stacks OOF predictions from all 3 base models using LogisticRegression.

---

## Anomaly Engine Results (Engine 3)

| Account | Actual Label | F3912 | Anomaly Score | Anomaly Rank | Top-5% Cutoff (≤454) |
|---|---|---|---|---|---|
| **9046** | Fraud (Type C Stealth) | 0 (bank missed) | 46.0 | 948 / 9,082 | Outside top 5% |
| **9052** | Fraud (Type C Stealth) | 0 (bank missed) | 18.7 | 4,513 / 9,082 | Outside top 5% |

> **Honest reporting per PRD §4:** Both stealth mules missed the top-5% anomaly rank target.  
> Account 9046 reaches the **top 10.4%** (rank 948). Account 9052 is deeper at rank 4,513 (top 49.7%).  
> The anomaly engine uses IsolationForest on peer-group robust z-scores + behavioral rule-boosts  
> (magnitude-filtered step_down_flag, passive_receiver, z_max, z_abs_sum). The step-down pattern  
> that characterizes account 9052 is shared by ~2,052 legitimate accounts with similar exact-halving  
> flow patterns, making pure-unsupervised discrimination difficult at top-5% precision.

---

## Fraud Subtype Breakdown

| Subtype | Definition | Count |
|---|---|---|
| Type A — Classic | F3912=1 AND F3908=1 | 63 |
| Type B — Quiet | F3912=1 AND F3908=0 | 16 |
| Type C — Stealth | F3912=0 | 2 |
| **Total** | | **81** |

---

## Engine Outputs

| Engine | Output | Method |
|---|---|---|
| Engine 0 | 3,924 → 3,611 features | Leakage firewall (drop F3912/F2230/F3911/63 empty/246 dup-pairs) |
| Engine 1 | 47-feature model matrix | Feature engineering (20 engineered + 27 raw key features) |
| Engine 2 | OOF risk probability per account | HistGB baseline (headline); XGB+LGBM+CAT+stack (extended) |
| Engine 3 | Anomaly score 0–100 + rank | IsolationForest on peer-group robust z-scores + behavioral boosts |
| Engine 4 | Risk tier + subtype + explanations | SHAP TreeExplainer on baseline HistGB |

---

## Verification Shield

- **779 accounts (8.58%)** have F3905=1 OR F3906=1 OR F3907=1 OR F3915=1  
- **0 observed fraud** among these accounts  
- These accounts are pre-filtered to `risk_tier = LOW` without model inference

---

## Library Path

| Library | Available | Version | Used For |
|---|---|---|---|
| xgboost | ✅ | 3.2.0 | Extended ensemble base model |
| lightgbm | ✅ | 4.6.0 | Extended ensemble base model |
| catboost | ✅ | 1.2.10 | Extended ensemble base model |
| shap | ✅ | 0.52.0 | Per-account SHAP explanations |

**No fallback path taken** — all optional libraries were available.
