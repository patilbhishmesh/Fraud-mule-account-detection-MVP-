# Mule Account Detection MVP

## Problem Statement 2 — AI/ML Based Classification of Suspicious Mule Accounts  
**Hackathon Submission · 2026**

---

## Library Availability (for transparency)

| Library | Status | Notes |
|---|---|---|
| `xgboost` | ✅ Installed | Used in extended ensemble (3.2.0) |
| `lightgbm` | ✅ Installed | Used in extended ensemble (4.6.0) |
| `catboost` | ✅ Installed | Used in extended ensemble (1.2.10) |
| `shap` | ✅ Installed | Used for per-account explanations (0.52.0) |
| `fastapi` + `uvicorn` | ✅ Installed | REST API backend |
| `streamlit` + `plotly` | ✅ Installed | Dashboard |

**Ensemble path taken:** ADVANCED (XGBoost + LightGBM + CatBoost + LogisticRegression meta-learner).  
**Explanation method:** SHAP TreeExplainer on baseline HistGB model.

If any of the optional libraries had been unavailable, the fallback would have been:
- Ensemble: HistGradientBoosting + RandomForest + ExtraTrees (all sklearn)
- Explanations: Permutation importance + peer-group deviation labelled "Feature Deviation (Fallback)"

---

## Repository Structure

```
files/
├── DataSet.csv                    # Read-only ground truth (never modified)
├── model_pipeline.py              # Validated reference — M0 baseline
├── src/
│   ├── pipeline/
│   │   ├── leakage_firewall.py    # Engine 0
│   │   ├── feature_engineering.py # Engine 1
│   │   ├── train_ensemble.py      # Engine 2 (baseline + extended)
│   │   ├── anomaly_engine.py      # Engine 3 (unsupervised)
│   │   └── decision_engine.py     # Engine 4 (risk tier, SHAP)
│   ├── build_artifacts.py         # Full pipeline orchestrator
│   ├── api/
│   │   └── main.py                # FastAPI backend
│   └── dashboard/
│       └── app.py                 # Streamlit 6-page dashboard
├── models/                        # Generated artifacts (after build)
│   ├── baseline_model.pkl
│   ├── ensemble_base_models.pkl
│   ├── meta_model.pkl
│   ├── predictions.parquet/.csv
│   ├── anomaly_scores.parquet/.csv
│   ├── feature_importance.csv
│   ├── metrics.json
│   └── explanations.json
├── RESULTS.md                     # Metrics comparison table
├── requirements.txt
└── README.md
```

---

## How to Run — From Scratch

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Build all ML artifacts (runs Engines 0–4, ~10–20 min)
```bash
# From the files/ directory:
python -m src.build_artifacts
```
This saves all models, predictions, anomaly scores, and metrics to `models/`.

### Step 3: Start the FastAPI backend
```bash
python src/api/main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Step 4: Start the Streamlit dashboard
```bash
streamlit run src/dashboard/app.py
# Dashboard available at http://localhost:8501
```

---

## Validated Baseline (M0)

Run the original reference script to reproduce:
```bash
python model_pipeline.py
```

Expected output:
- **PR-AUC: 0.531** (±0.02 acceptable)
- Threshold 0.25: Recall 50.6% (41/81), Precision 49.4%, F1 0.500

---

## Integrity Constraints (PRD §4)

- ❌ `F3924`, `F3912`, `F2230` are **never** used as model inputs
- ❌ No hardcoded logic based on row index or account ID
- ✅ All scores from general functions applied identically to every account
- ✅ Actual numbers reported even if they miss targets (no fabrication)

---

## Non-Goals (out of scope)

- Graph Neural Networks, Kafka/Flink streaming, federated learning
- Authentication, CI/CD, production hardening
- Editing `DataSet.csv` (read-only ground truth)
