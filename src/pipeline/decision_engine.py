"""
Engine 4: Decision Engine + Explainability
============================================
Risk score, risk tier, fraud subtype, and per-account explanations.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Check for SHAP availability
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass


def compute_risk_scores(oof_proba, shield, X_model):
    """
    Compute risk_score = ensemble probability × 100 (0 for shield accounts).

    Parameters
    ----------
    oof_proba : np.ndarray
        Out-of-fold probabilities from Engine 2.
    shield : pd.Series
        Verification-shield boolean mask.
    X_model : pd.DataFrame
        Feature matrix (for index alignment).

    Returns
    -------
    risk_scores : pd.Series
        Risk score 0-100 for each account.
    """
    risk_scores = pd.Series(oof_proba * 100, index=X_model.index)
    risk_scores[shield.values] = 0.0
    return risk_scores


def assign_risk_tier(risk_score):
    """Assign risk tier: LOW (<30), MED (30-70), HIGH (>70)."""
    if risk_score < 30:
        return 'LOW'
    elif risk_score <= 70:
        return 'MED'
    else:
        return 'HIGH'


def assign_fraud_subtypes(df_raw):
    """
    Assign fraud subtypes for known fraud accounts (F3924=1).
    Type A: F3912=1 AND F3908=1 (Classic)
    Type B: F3912=1 AND F3908=0 (Quiet)
    Type C: F3912=0 (Stealth)
    """
    subtypes = pd.Series(None, index=df_raw.index, dtype=object)
    fraud_mask = df_raw['F3924'] == 1

    type_a = fraud_mask & (df_raw['F3912'] == 1) & (df_raw['F3908'] == 1)
    type_b = fraud_mask & (df_raw['F3912'] == 1) & (df_raw['F3908'] == 0)
    type_c = fraud_mask & (df_raw['F3912'] == 0)

    subtypes[type_a] = 'Type A - Classic'
    subtypes[type_b] = 'Type B - Quiet'
    subtypes[type_c] = 'Type C - Stealth'

    print(f"\n  Fraud subtypes: A={type_a.sum()}, B={type_b.sum()}, C={type_c.sum()}")
    return subtypes


def compute_shap_explanations(model, X_model, top_n=5):
    """
    Compute SHAP explanations if available, otherwise use permutation importance fallback.

    Returns
    -------
    explanations : dict
        {account_id: [{feature, value, contribution}, ...]}
    method : str
        'shap' or 'permutation_importance_fallback'
    """
    if SHAP_AVAILABLE:
        print("  Computing SHAP explanations (limited to top accounts for speed)...")
        try:
            explainer = shap.TreeExplainer(model)
            # Compute SHAP on full dataset — HistGB is fast
            shap_values = explainer.shap_values(X_model)
            # shap may return ndarray directly for binary classifiers
            if isinstance(shap_values, list):
                sv_arr = np.array(shap_values[1])
            else:
                sv_arr = np.array(shap_values)
            # If 3D (n_classes, n_samples, n_features), take class-1 slice
            if sv_arr.ndim == 3:
                sv_arr = sv_arr[1]

            explanations = {}
            for i, idx in enumerate(X_model.index):
                sv = sv_arr[i]
                top_idx = np.argsort(np.abs(sv))[-top_n:][::-1]
                explanations[idx] = [
                    {
                        'feature': X_model.columns[j],
                        'value': round(float(X_model.iloc[i, j]), 4) if pd.notna(X_model.iloc[i, j]) else None,
                        'contribution': round(float(sv[j]), 4)
                    }
                    for j in top_idx
                ]
            print(f"  SHAP computed for {len(explanations)} accounts")
            return explanations, 'shap'
        except Exception as e:
            print(f"  SHAP failed ({e}), falling back to permutation importance")

    # Fallback: global permutation importance + per-account deviation
    print("  Using fallback: permutation importance + peer-group deviation")
    from sklearn.inspection import permutation_importance

    y_dummy = np.zeros(len(X_model))  # We need to compute importance without target
    # Instead, use the model's own predictions as a proxy
    try:
        imp = permutation_importance(
            model, X_model, model.predict(X_model),
            n_repeats=3, random_state=42, n_jobs=-1
        )
        top_features = X_model.columns[np.argsort(imp.importances_mean)[-top_n*2:][::-1]]
    except Exception:
        # Last resort: use column variance
        top_features = X_model.columns[:top_n]

    # Per-account: deviation from median for the globally important features
    medians = X_model[top_features].median()
    stds = X_model[top_features].std() + 1e-9

    explanations = {}
    for idx in X_model.index:
        row = X_model.loc[idx, top_features]
        deviations = ((row - medians) / stds).abs()
        top_dev_idx = deviations.nlargest(top_n).index
        explanations[idx] = [
            {
                'feature': feat,
                'value': round(float(row[feat]), 4) if pd.notna(row[feat]) else None,
                'contribution': round(float(deviations[feat]), 4)
            }
            for feat in top_dev_idx
        ]

    return explanations, 'permutation_importance_fallback'


def build_decision_output(
    X_model, df_raw, oof_proba, shield, anomaly_df,
    model, feature_importance_df
):
    """
    Build the complete per-account decision output (PRD §10.5 schema).

    Returns
    -------
    accounts_df : pd.DataFrame
        Full account-level output.
    explanations : dict
        Per-account feature explanations.
    explanation_method : str
        Which method was used for explanations.
    """
    print("\n=== Engine 4: Decision Engine ===")

    # Risk scores
    risk_scores = compute_risk_scores(oof_proba, shield, X_model)

    # Risk tiers
    risk_tiers = risk_scores.apply(assign_risk_tier)

    # Fraud subtypes
    subtypes = assign_fraud_subtypes(df_raw)

    # Explanations
    explanations, explanation_method = compute_shap_explanations(model, X_model)
    print(f"  Explanation method: {explanation_method}")

    # Build output dataframe
    accounts_df = pd.DataFrame({
        'account_id': X_model.index,
        'risk_score': risk_scores.values.round(2),
        'risk_tier': risk_tiers.values,
        'fraud_subtype': subtypes.reindex(X_model.index).values,
        'anomaly_score': anomaly_df.reindex(X_model.index)['anomaly_score'].values,
        'anomaly_rank': anomaly_df.reindex(X_model.index)['anomaly_rank'].values,
        'verification_shield': shield.values,
        'actual_label': df_raw['F3924'].reindex(X_model.index).values,
    })
    accounts_df = accounts_df.set_index('account_id')

    # Summary
    tier_counts = accounts_df['risk_tier'].value_counts()
    print(f"\n  Risk tier distribution:")
    for tier in ['HIGH', 'MED', 'LOW']:
        print(f"    {tier}: {tier_counts.get(tier, 0)}")

    return accounts_df, explanations, explanation_method
