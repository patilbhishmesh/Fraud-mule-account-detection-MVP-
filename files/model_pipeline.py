"""
Mule Account Detection — Validated Baseline Pipeline
======================================================
Reproduces the real numbers used in the revised slide deck:
  PR-AUC (5-fold OOF): 0.531
  At threshold 0.25 -> Recall 50.6% (41/81), Precision 49.4%, F1 0.500

Run: python3 model_pipeline.py
Requires: pandas, numpy, scikit-learn (all available offline)
Optional upgrade: pip install xgboost lightgbm catboost shap
"""

import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "DataSet.csv"  # adjust path as needed

# ------------------------------------------------------------------
# ENGINE 0: LEAKAGE FIREWALL
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, index_col=0)
y = df['F3924'].values

drop_cols = ['F3924', 'F3912', 'F2230', 'F3911']           # target + leakage + artifact + constant
empty_cols = df.columns[df.isna().all()].tolist()          # 63 fully-empty columns
drop_cols += empty_cols

# Duplicate-pair detection: gap=324, corr>0.999 (drops 246 columns)
numeric_cols = [c for c in df.columns if c.startswith('F') and c[1:].isdigit()
                and df[c].dtype in ['float64', 'int64']]
nums = sorted(int(c[1:]) for c in numeric_cols)
numset = set(nums)
dup_drop = []
for n in nums:
    m = n + 324
    if m in numset:
        a, b = f'F{n}', f'F{m}'
        if a in df.columns and b in df.columns and a not in drop_cols and b not in drop_cols:
            if df[a].corr(df[b]) > 0.999:
                dup_drop.append(b)
drop_cols += dup_drop

print(f"Leakage firewall: dropping {len(set(drop_cols))} columns "
      f"({len(empty_cols)} empty, {len(dup_drop)} duplicate-pair, 4 leakage/artifact/target)")

X = df.drop(columns=list(set(drop_cols)), errors='ignore')

# Pre-filter / verification shield (informational — does not change CV here)
shield = ((X.get('F3905', 0) == 1) | (X.get('F3906', 0) == 1) |
          (X.get('F3907', 0) == 1) | (X.get('F3915', 0) == 1))
print(f"Verification shield covers {shield.sum()} accounts "
      f"({shield.mean()*100:.2f}%), fraud among them = {y[shield.values].sum()}")

# ------------------------------------------------------------------
# ENGINE 1: SIGNAL ENGINEERING
# ------------------------------------------------------------------
fe = pd.DataFrame(index=X.index)
fe['receiver_amount_interaction'] = X['F2686'] * np.log1p(X['F2285'].clip(lower=0))
fe['spend_receive_ratio'] = np.log1p(X['F2481'].clip(lower=0)) / (np.log1p(X['F2482'].clip(lower=0)) + 1)

flow_cols = ['F3832', 'F3833', 'F3834', 'F3835', 'F3836', 'F3837']
flow = X[flow_cols].values
fe['flow_slope'] = [np.polyfit(range(6), row, 1)[0] for row in flow]
fe['all_flow_positive'] = (flow > 0).all(axis=1).astype(int)
fe['flow_consistency'] = flow.std(axis=1)

# Step-down detector: period1≈period2≈2×period3, period4≈period5≈2×period6
first_half, second_half = flow[:, :3].mean(axis=1), flow[:, 3:].mean(axis=1)
close12, close23 = np.abs(flow[:, 0]-flow[:, 1]) < 1, np.abs(flow[:, 1]-flow[:, 2]) < 1
close45, close56 = np.abs(flow[:, 3]-flow[:, 4]) < 1, np.abs(flow[:, 4]-flow[:, 5]) < 1
half_check = np.abs(second_half - first_half/2) / (first_half + 1) < 0.05
fe['step_down_flag'] = (close12 & close23 & close45 & close56 & half_check).astype(int)

fe['F515_present'] = (~X['F515'].isna()).astype(int)
fe['F518_present'] = (~X['F518'].isna()).astype(int)
fe['F448_group_absent'] = (X['F448'].isna() & X['F450'].isna() & X['F451'].isna()).astype(int)

# Risk encodings (derived from KS-stat differences in fraud vs legit)
fe['occupation_risk'] = X['F3891'].map({'student': 1.94, 'agriculture': 1.26, 'retired': 1.04,
                                          'housewife': 0.90, 'salaried': 0.73, 'selfemployed': 0.66}).fillna(1.0)
fe['location_risk'] = X['F3890'].map({'R': 1.44, 'SU': 0.88, 'U': 0.73, 'M': 0.62}).fillna(1.0)
fe['account_type_risk'] = X['F3886'].map({'Savings': 1.8, 'Current': 0.3}).fillna(1.0)

fe['f3923_danger'] = (X['F3923'] == 3).astype(int)              # alert count == 3 -> 13.0% fraud rate
fe['age_interaction_risk'] = ((X['F3887'] < 12) & (X['F3894'] < 30)).astype(int)
fe['verified_shield'] = shield.astype(int)

keep_known = ['F451', 'F448', 'F142', 'F453', 'F450', 'F2480', 'F261', 'F2284',
               'F2686', 'F2285', 'F2481', 'F2482', 'F2483', 'F2292',
               'F3832', 'F3833', 'F3834', 'F3835', 'F3836', 'F3837',
               'F3908', 'F3909', 'F3887', 'F3894', 'F3923',
               'F3905', 'F3906', 'F3907', 'F3913', 'F3914', 'F3915', 'F3922']
keep_known = [c for c in keep_known if c in X.columns]
X_model = pd.concat([fe, X[keep_known]], axis=1)
print(f"Final feature matrix: {X_model.shape}")

# ------------------------------------------------------------------
# ENGINE 2: ENSEMBLE CLASSIFIER (baseline)
# ------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(y))

for tr, te in skf.split(X_model, y):
    sw = np.where(y[tr] == 1, 111, 1)  # equivalent to scale_pos_weight=111
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_model.iloc[tr], y[tr], sample_weight=sw)
    oof_proba[te] = model.predict_proba(X_model.iloc[te])[:, 1]

pr_auc = average_precision_score(y, oof_proba)
print(f"\n=== VALIDATED RESULTS (5-fold OOF) ===")
print(f"PR-AUC: {pr_auc:.4f}")

for t in [0.20, 0.25, 0.30]:
    pred = (oof_proba >= t).astype(int)
    print(f"  Threshold {t}: Precision={precision_score(y,pred):.3f}, "
          f"Recall={recall_score(y,pred):.3f} ({(y[pred==1]==1).sum()}/{y.sum()}), "
          f"F1={f1_score(y,pred):.3f}, flagged={pred.sum()}")

# ------------------------------------------------------------------
# ENGINE 3: ANOMALY ENGINE — robust stealth-mule check (Phase-1 work-in-progress)
# ------------------------------------------------------------------
stealth_idx = df[(df['F3912'] == 0) & (df['F3924'] == 1)].index
print(f"\nKnown stealth mules (F3912=0, F3924=1): {list(stealth_idx)}")
for i in stealth_idx:
    pos = X_model.index.get_loc(i)
    print(f"  Account {i}: supervised model risk = {oof_proba[pos]:.3f} "
          f"(NOTE: not yet reliably flagged — see Phase-1 anomaly engine refinement)")

# ------------------------------------------------------------------
# Feature importance (permutation, real run — stand-in for SHAP)
# ------------------------------------------------------------------
sw_all = np.where(y == 1, 111, 1)
final_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
final_model.fit(X_model, y, sample_weight=sw_all)
imp = permutation_importance(final_model, X_model, y, n_repeats=5, random_state=42, scoring='average_precision', n_jobs=-1)
imp_df = pd.DataFrame({'feature': X_model.columns, 'importance': imp.importances_mean}).sort_values('importance', ascending=False)
print("\nTop 10 features by permutation importance (PR-AUC impact):")
print(imp_df.head(10).to_string(index=False))
