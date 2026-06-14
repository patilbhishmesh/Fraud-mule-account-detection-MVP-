"""
Engine 2: Ensemble Classifier
===============================
Baseline: HistGradientBoostingClassifier (validated, PR-AUC ≈ 0.531)
Extended: 3-model ensemble + LogisticRegression meta-learner

Fallback path: if xgboost/lightgbm/catboost unavailable, uses
HistGB + RandomForest + ExtraTrees as base learners.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Check which optional libraries are available ---
AVAILABLE_LIBS = {}
try:
    import xgboost as xgb
    AVAILABLE_LIBS['xgboost'] = True
except ImportError:
    AVAILABLE_LIBS['xgboost'] = False

try:
    import lightgbm as lgb
    AVAILABLE_LIBS['lightgbm'] = True
except ImportError:
    AVAILABLE_LIBS['lightgbm'] = False

try:
    import catboost as cb
    AVAILABLE_LIBS['catboost'] = True
except ImportError:
    AVAILABLE_LIBS['catboost'] = False


def _get_ensemble_models():
    """Return list of (name, model) tuples for ensemble, with fallback."""
    use_advanced = all(AVAILABLE_LIBS.get(k) for k in ['xgboost', 'lightgbm', 'catboost'])

    if use_advanced:
        print("  Using advanced ensemble: XGBoost + LightGBM + CatBoost")
        models = [
            ('xgboost', xgb.XGBClassifier(
                scale_pos_weight=111, max_depth=6, n_estimators=300,
                learning_rate=0.05, eval_metric='aucpr', random_state=42,
                verbosity=0, use_label_encoder=False
            )),
            ('lightgbm', lgb.LGBMClassifier(
                is_unbalance=True, num_leaves=63, n_estimators=300,
                learning_rate=0.05, random_state=42, verbose=-1
            )),
            ('catboost', cb.CatBoostClassifier(
                auto_class_weights='Balanced', iterations=300,
                depth=6, random_state=42, verbose=0
            )),
        ]
        ensemble_type = 'advanced'
    else:
        print(f"  Fallback ensemble: HistGB + RandomForest + ExtraTrees")
        print(f"  Library availability: {AVAILABLE_LIBS}")
        models = [
            ('histgb', HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
            )),
            ('random_forest', RandomForestClassifier(
                n_estimators=300, max_depth=12, class_weight={0: 1, 1: 111},
                random_state=42, n_jobs=-1
            )),
            ('extra_trees', ExtraTreesClassifier(
                n_estimators=300, max_depth=12, class_weight={0: 1, 1: 111},
                random_state=42, n_jobs=-1
            )),
        ]
        ensemble_type = 'fallback'

    return models, ensemble_type


def _compute_metrics(y_true, y_proba, thresholds=(0.20, 0.25, 0.30)):
    """Compute all metrics at given thresholds."""
    pr_auc = average_precision_score(y_true, y_proba)
    random_baseline = y_true.sum() / len(y_true)

    results = {
        'pr_auc': round(pr_auc, 4),
        'random_baseline_pr_auc': round(random_baseline, 4),
        'improvement_over_random': round(pr_auc / random_baseline, 1),
        'thresholds': {}
    }

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        results['thresholds'][str(t)] = {
            'precision': round(precision_score(y_true, pred, zero_division=0), 4),
            'recall': round(recall_score(y_true, pred, zero_division=0), 4),
            'f1': round(f1_score(y_true, pred, zero_division=0), 4),
            'flagged': int(pred.sum()),
            'true_positives': int((pred & y_true).sum()),
            'total_positives': int(y_true.sum()),
            'confusion_matrix': cm.tolist(),
        }

    # PR curve data
    prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_true, y_proba)
    results['pr_curve'] = {
        'precision': prec_curve.tolist(),
        'recall': rec_curve.tolist(),
        'thresholds': thresh_curve.tolist(),
    }

    return results


def train_baseline(X_model, y):
    """Train the validated baseline (HistGradientBoosting, 5-fold CV)."""
    print("\n=== Engine 2: Training BASELINE (HistGradientBoosting) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(y))

    for fold, (tr, te) in enumerate(skf.split(X_model, y)):
        sw = np.where(y[tr] == 1, 111, 1)
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
        )
        model.fit(X_model.iloc[tr], y[tr], sample_weight=sw)
        oof_proba[te] = model.predict_proba(X_model.iloc[te])[:, 1]
        print(f"  Fold {fold+1}/5 done")

    metrics = _compute_metrics(y, oof_proba)
    print(f"  Baseline PR-AUC: {metrics['pr_auc']}")
    for t in ['0.2', '0.25', '0.3']:
        m = metrics['thresholds'][t]
        print(f"    Threshold {t}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} ({m['true_positives']}/{m['total_positives']})")

    # Train final model on all data for feature importance and predictions
    sw_all = np.where(y == 1, 111, 1)
    final_model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
    )
    final_model.fit(X_model, y, sample_weight=sw_all)

    return oof_proba, metrics, final_model


def train_ensemble(X_model, y):
    """Train extended ensemble with meta-learner stacking."""
    print("\n=== Engine 2: Training EXTENDED ENSEMBLE ===")
    base_models, ensemble_type = _get_ensemble_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n = len(y)
    n_models = len(base_models)
    oof_base = np.zeros((n, n_models))

    # Train each base model with OOF predictions
    for i, (name, model_template) in enumerate(base_models):
        print(f"  Training {name}...")
        import copy
        for fold, (tr, te) in enumerate(skf.split(X_model, y)):
            model = copy.deepcopy(model_template)
            X_tr, X_te = X_model.iloc[tr], X_model.iloc[te]
            y_tr = y[tr]

            if name == 'histgb':
                sw = np.where(y_tr == 1, 111, 1)
                model.fit(X_tr, y_tr, sample_weight=sw)
            elif name in ('random_forest', 'extra_trees'):
                model.fit(X_tr, y_tr)
            elif name == 'xgboost':
                model.fit(X_tr, y_tr)
            elif name == 'lightgbm':
                model.fit(X_tr, y_tr)
            elif name == 'catboost':
                model.fit(X_tr, y_tr)

            oof_base[te, i] = model.predict_proba(X_te)[:, 1]
        print(f"    {name} OOF PR-AUC: {average_precision_score(y, oof_base[:, i]):.4f}")

    # Meta-learner: LogisticRegression on OOF predictions
    print("  Training meta-learner (LogisticRegression)...")
    meta_model = LogisticRegression(
        class_weight={0: 1, 1: 111}, random_state=42, max_iter=1000
    )

    # Use CV for the meta-learner too
    oof_ensemble = np.zeros(n)
    for fold, (tr, te) in enumerate(skf.split(oof_base, y)):
        meta_model_fold = LogisticRegression(
            class_weight={0: 1, 1: 111}, random_state=42, max_iter=1000
        )
        meta_model_fold.fit(oof_base[tr], y[tr])
        oof_ensemble[te] = meta_model_fold.predict_proba(oof_base[te])[:, 1]

    metrics = _compute_metrics(y, oof_ensemble)
    print(f"\n  Extended Ensemble PR-AUC: {metrics['pr_auc']}")
    for t in ['0.2', '0.25', '0.3']:
        m = metrics['thresholds'][t]
        print(f"    Threshold {t}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} ({m['true_positives']}/{m['total_positives']})")

    # Train final models on all data
    final_base_models = []
    for name, model_template in base_models:
        import copy
        model = copy.deepcopy(model_template)
        if name == 'histgb':
            sw = np.where(y == 1, 111, 1)
            model.fit(X_model, y, sample_weight=sw)
        else:
            model.fit(X_model, y)
        final_base_models.append((name, model))

    # Final meta-learner
    # Get full-data base predictions
    full_base_preds = np.column_stack([
        m.predict_proba(X_model)[:, 1] for _, m in final_base_models
    ])
    meta_model.fit(full_base_preds, y)

    metrics['ensemble_type'] = ensemble_type
    metrics['base_model_names'] = [name for name, _ in base_models]

    return oof_ensemble, metrics, final_base_models, meta_model, oof_base


def compute_feature_importance(model, X_model, y, top_n=20):
    """Compute permutation importance for a model."""
    print("\n  Computing permutation importance...")
    imp = permutation_importance(
        model, X_model, y, n_repeats=5, random_state=42,
        scoring='average_precision', n_jobs=-1
    )
    imp_df = pd.DataFrame({
        'feature': X_model.columns,
        'importance': imp.importances_mean
    }).sort_values('importance', ascending=False)

    print(f"  Top {min(top_n, len(imp_df))} features:")
    for _, row in imp_df.head(top_n).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    return imp_df


def compute_rule_based_metrics(df):
    """Compute metrics for the rule-based approach (F3912 alone)."""
    y_true = df['F3924'].values
    y_pred = df['F3912'].values

    pr_auc = average_precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'pr_auc': round(pr_auc, 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'confusion_matrix': cm.tolist(),
        'note': 'F3912 is the bank\'s own fraud flag — LEAKAGE, not a valid model'
    }
