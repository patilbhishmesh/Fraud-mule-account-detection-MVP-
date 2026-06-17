"""
FastAPI Backend — Mule Account Detection MVP
=============================================
All endpoints backed by pre-computed artifacts from build_artifacts.py.
"""

import os
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Paths — main.py lives in src/api/; project root is two levels up
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_here))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


app = FastAPI(
    title="Mule Account Detection API",
    description="AI/ML Based Classification of Suspicious Mule Accounts",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# LOAD PRE-COMPUTED ARTIFACTS
# ------------------------------------------------------------------
def load_artifacts():
    """Load all pre-computed artifacts at startup."""
    global predictions_df, anomaly_df, metrics, feature_importance, explanations

    predictions_df = pd.read_csv(
        os.path.join(MODELS_DIR, 'predictions.csv'), index_col=0
    )
    anomaly_df = pd.read_csv(
        os.path.join(MODELS_DIR, 'anomaly_scores.csv'), index_col=0
    )

    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)

    feature_importance = pd.read_csv(
        os.path.join(MODELS_DIR, 'feature_importance.csv')
    )

    with open(os.path.join(MODELS_DIR, 'explanations.json'), 'r') as f:
        explanations = json.load(f)


# Load on module import
try:
    load_artifacts()
except Exception as e:
    print(f"Warning: Could not load artifacts: {e}")
    print("Run build_artifacts.py first!")
    predictions_df = pd.DataFrame()
    anomaly_df = pd.DataFrame()
    metrics = {}
    feature_importance = pd.DataFrame()
    explanations = {}


# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "artifacts_loaded": len(predictions_df) > 0,
        "total_accounts": len(predictions_df),
    }


@app.get("/dataset/stats")
def dataset_stats():
    if 'dataset_stats' not in metrics:
        raise HTTPException(404, "Metrics not loaded")
    return metrics['dataset_stats']


@app.get("/leakage-report")
def leakage_report():
    if 'firewall_report' not in metrics:
        raise HTTPException(404, "Firewall report not loaded")
    report = metrics['firewall_report']
    return {
        'leakage_details': report['leakage_details'],
        'features_before': report['total_features_raw'],
        'features_after': report['total_features_post_firewall'],
        'dropped_total': report['dropped_total'],
        'dropped_empty_count': report['dropped_empty_count'],
        'dropped_duplicate_count': report['dropped_duplicate_count'],
    }


@app.get("/model/metrics")
def model_metrics(model: str = Query("baseline", regex="^(baseline|ensemble)$")):
    if model not in metrics:
        raise HTTPException(404, f"Model '{model}' metrics not found")

    result = metrics[model].copy()
    result['random_baseline_pr_auc'] = metrics['baseline'].get(
        'random_baseline_pr_auc', 0.0089
    )
    return result


@app.get("/model/feature-importance")
def model_feature_importance(
    model: str = Query("baseline"),
    top: int = Query(10, ge=1, le=50)
):
    if feature_importance.empty:
        raise HTTPException(404, "Feature importance not loaded")
    return feature_importance.head(top).to_dict(orient='records')


@app.get("/accounts")
def list_accounts(
    risk_tier: str = Query(None),
    fraud_subtype: str = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    df = predictions_df.copy()
    df.index.name = 'account_id'
    df = df.reset_index()

    if risk_tier:
        df = df[df['risk_tier'] == risk_tier.upper()]
    if fraud_subtype:
        df = df[df['fraud_subtype'].str.contains(fraud_subtype, case=False, na=False)]

    total = len(df)
    df = df.sort_values('risk_score', ascending=False)
    page = df.iloc[offset:offset + limit]

    return {
        'total': total,
        'limit': limit,
        'offset': offset,
        'accounts': page.to_dict(orient='records'),
    }


@app.get("/accounts/{account_id}")
def get_account(account_id: int):
    if account_id not in predictions_df.index:
        raise HTTPException(404, f"Account {account_id} not found")

    row = predictions_df.loc[account_id]
    result = row.to_dict()
    result['account_id'] = account_id

    # Add explanations
    account_key = str(account_id)
    if account_key in explanations:
        result['top_factors'] = explanations[account_key]
    else:
        result['top_factors'] = []

    # Add anomaly details
    if account_id in anomaly_df.index:
        anom = anomaly_df.loc[account_id]
        result['step_down_flag'] = int(anom.get('step_down_flag', 0))
        result['passive_receiver_flag'] = int(anom.get('passive_receiver_flag', 0))

    return result


@app.get("/fraud-subtypes")
def fraud_subtypes():
    if 'fraud_subtypes' not in metrics:
        raise HTTPException(404, "Fraud subtypes not loaded")
    return metrics['fraud_subtypes']


@app.get("/stealth-mules")
def stealth_mules():
    if 'stealth_mules' not in metrics:
        raise HTTPException(404, "Stealth mule data not loaded")
    return metrics['stealth_mules']


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
