"""
Build Artifacts — Full Pipeline Orchestrator
=============================================
Runs Engines 0–4 in sequence, saves all artifacts to models/ directory.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# Resolve project root regardless of how script is invoked
_here = os.path.dirname(os.path.abspath(__file__))
# build_artifacts.py lives in src/, so project root is one level up
PROJECT_ROOT = os.path.dirname(_here)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.leakage_firewall import run_leakage_firewall
from src.pipeline.feature_engineering import run_feature_engineering
from src.pipeline.train_ensemble import (
    train_baseline, train_ensemble, compute_feature_importance,
    compute_rule_based_metrics
)
from src.pipeline.anomaly_engine import run_anomaly_engine
from src.pipeline.decision_engine import build_decision_output

DATA_PATH = os.path.join(PROJECT_ROOT, 'DataSet.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)



def main():
    print("=" * 60)
    print("MULE ACCOUNT DETECTION MVP — FULL PIPELINE BUILD")
    print("=" * 60)

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"  Shape: {df.shape}")

    # ------------------------------------------------------------------
    # ENGINE 0: LEAKAGE FIREWALL
    # ------------------------------------------------------------------
    X, y, shield, firewall_report = run_leakage_firewall(df)

    # ------------------------------------------------------------------
    # ENGINE 1: FEATURE ENGINEERING
    # ------------------------------------------------------------------
    X_model, feature_names = run_feature_engineering(X, shield)

    # ------------------------------------------------------------------
    # ENGINE 2: ENSEMBLE CLASSIFIER
    # ------------------------------------------------------------------
    # Baseline
    baseline_oof, baseline_metrics, baseline_model = train_baseline(X_model, y)

    # Rule-based (for comparison)
    rule_metrics = compute_rule_based_metrics(df)
    print(f"\n  Rule-based (F3912 alone): PR-AUC={rule_metrics['pr_auc']}, "
          f"R={rule_metrics['recall']}, P={rule_metrics['precision']}, F1={rule_metrics['f1']}")

    # Extended ensemble
    ensemble_oof, ensemble_metrics, final_base_models, meta_model, oof_base = train_ensemble(X_model, y)

    # Feature importance (on baseline model)
    importance_df = compute_feature_importance(baseline_model, X_model, y)

    # Pick the best OOF for downstream use
    best_oof = ensemble_oof if ensemble_metrics['pr_auc'] >= baseline_metrics['pr_auc'] else baseline_oof
    best_model = baseline_model  # Use baseline for SHAP since it's single model
    best_metrics_name = 'ensemble' if ensemble_metrics['pr_auc'] >= baseline_metrics['pr_auc'] else 'baseline'
    print(f"\n  Best model for downstream: {best_metrics_name} "
          f"(PR-AUC={max(ensemble_metrics['pr_auc'], baseline_metrics['pr_auc'])})")

    # ------------------------------------------------------------------
    # ENGINE 3: ANOMALY ENGINE
    # ------------------------------------------------------------------
    anomaly_df = run_anomaly_engine(X, df)

    # ------------------------------------------------------------------
    # ENGINE 4: DECISION ENGINE
    # ------------------------------------------------------------------
    accounts_df, explanations, explanation_method = build_decision_output(
        X_model, df, best_oof, shield, anomaly_df, best_model, importance_df
    )

    # ------------------------------------------------------------------
    # SAVE ARTIFACTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    # Models
    joblib.dump(baseline_model, os.path.join(MODELS_DIR, 'baseline_model.pkl'))
    joblib.dump(final_base_models, os.path.join(MODELS_DIR, 'ensemble_base_models.pkl'))
    joblib.dump(meta_model, os.path.join(MODELS_DIR, 'meta_model.pkl'))
    print("  Saved models")

    # Predictions
    accounts_df.to_parquet(os.path.join(MODELS_DIR, 'predictions.parquet'))
    accounts_df.to_csv(os.path.join(MODELS_DIR, 'predictions.csv'))
    print(f"  Saved predictions ({len(accounts_df)} accounts)")

    # OOF probabilities
    np.save(os.path.join(MODELS_DIR, 'baseline_oof.npy'), baseline_oof)
    np.save(os.path.join(MODELS_DIR, 'ensemble_oof.npy'), ensemble_oof)
    print("  Saved OOF probabilities")

    # Anomaly scores
    anomaly_df.to_parquet(os.path.join(MODELS_DIR, 'anomaly_scores.parquet'))
    anomaly_df.to_csv(os.path.join(MODELS_DIR, 'anomaly_scores.csv'))
    print("  Saved anomaly scores")

    # Feature importance
    importance_df.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)
    print("  Saved feature importance")

    # Metrics
    all_metrics = {
        'rule_based': rule_metrics,
        'baseline': baseline_metrics,
        'ensemble': ensemble_metrics,
        'explanation_method': explanation_method,
        'firewall_report': firewall_report,
        'dataset_stats': {
            'total_accounts': len(df),
            'total_features_raw': firewall_report['total_features_raw'],
            'total_features_post_firewall': firewall_report['total_features_post_firewall'],
            'fraud_accounts': int(y.sum()),
            'legit_accounts': int((y == 0).sum()),
            'imbalance_ratio': round(int((y == 0).sum()) / int(y.sum()), 0),
            'verification_shield_count': firewall_report['verification_shield_count'],
            'verification_shield_pct': firewall_report['verification_shield_pct'],
        },
        'fraud_subtypes': {
            'type_a': int(accounts_df['fraud_subtype'].eq('Type A - Classic').sum()),
            'type_b': int(accounts_df['fraud_subtype'].eq('Type B - Quiet').sum()),
            'type_c': int(accounts_df['fraud_subtype'].eq('Type C - Stealth').sum()),
        }
    }

    # Stealth mule info
    stealth_idx = df[(df['F3912'] == 0) & (df['F3924'] == 1)].index
    stealth_info = []
    for idx in stealth_idx:
        if idx in accounts_df.index:
            row = accounts_df.loc[idx]
            stealth_info.append({
                'account_id': int(idx),
                'risk_score': float(row['risk_score']),
                'risk_tier': row['risk_tier'],
                'fraud_subtype': row['fraud_subtype'],
                'anomaly_score': float(row['anomaly_score']),
                'anomaly_rank': int(row['anomaly_rank']),
                'verification_shield': bool(row['verification_shield']),
                'top_factors': explanations.get(idx, []),
            })
    all_metrics['stealth_mules'] = stealth_info

    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print("  Saved metrics")

    # Explanations (top factors per account — save for top 200 risk accounts)
    top_risk_ids = accounts_df.nlargest(200, 'risk_score').index
    top_explanations = {int(k): v for k, v in explanations.items() if k in top_risk_ids}
    # Also include stealth mules and all fraud accounts
    for idx in stealth_idx:
        if idx in explanations:
            top_explanations[int(idx)] = explanations[idx]
    fraud_idx = df[df['F3924'] == 1].index
    for idx in fraud_idx:
        if idx in explanations:
            top_explanations[int(idx)] = explanations[idx]
    with open(os.path.join(MODELS_DIR, 'explanations.json'), 'w') as f:
        json.dump(top_explanations, f, indent=2, default=str)
    print(f"  Saved explanations for {len(top_explanations)} accounts")

    # Feature names
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"\n  Dataset: {len(df)} accounts, {firewall_report['total_features_raw']} features")
    print(f"  Post-firewall: {firewall_report['total_features_post_firewall']} features")
    print(f"  Verification shield: {firewall_report['verification_shield_count']} accounts "
          f"({firewall_report['verification_shield_pct']}%)")
    print(f"\n  Rule-based (F3912):  PR-AUC={rule_metrics['pr_auc']}")
    print(f"  Baseline (HistGB):   PR-AUC={baseline_metrics['pr_auc']}")
    print(f"  Extended Ensemble:   PR-AUC={ensemble_metrics['pr_auc']} "
          f"({ensemble_metrics.get('ensemble_type', 'unknown')})")
    print(f"\n  Fraud subtypes: A={all_metrics['fraud_subtypes']['type_a']}, "
          f"B={all_metrics['fraud_subtypes']['type_b']}, "
          f"C={all_metrics['fraud_subtypes']['type_c']}")
    print(f"\n  Stealth mules:")
    for sm in stealth_info:
        print(f"    Account {sm['account_id']}: anomaly_rank={sm['anomaly_rank']}/9082, "
              f"anomaly_score={sm['anomaly_score']:.1f}, risk_score={sm['risk_score']:.1f}")
    print(f"\n  Explanation method: {explanation_method}")
    print(f"\n  All artifacts saved to: {MODELS_DIR}/")


if __name__ == '__main__':
    main()
