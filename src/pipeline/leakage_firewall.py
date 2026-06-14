"""
Engine 0: Leakage Firewall
===========================
Drops leaked, constant, empty, and duplicate-pair columns from the raw dataset.
Computes the verification-shield pre-filter.

This is extracted from the validated model_pipeline.py — formulas are identical.
"""

import pandas as pd
import numpy as np


def run_leakage_firewall(df: pd.DataFrame):
    """
    Engine 0: Remove leakage, constant, empty, and duplicate-pair columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from DataSet.csv (index_col=0).

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix (no target, no leakage).
    y : np.ndarray
        Binary target array.
    shield : pd.Series
        Boolean series — True for verification-shield accounts.
    firewall_report : dict
        Metadata about what was dropped (for API/dashboard).
    """
    y = df['F3924'].values

    # --- Must-drop columns ---
    drop_cols = ['F3924', 'F3912', 'F2230', 'F3911']

    # 63 fully-empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    drop_cols += empty_cols

    # Duplicate-pair detection: gap=324, corr>0.999 (drops ~246 columns)
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

    drop_set = list(set(drop_cols))
    X = df.drop(columns=drop_set, errors='ignore')

    # --- Verification shield ---
    shield = ((X.get('F3905', 0) == 1) | (X.get('F3906', 0) == 1) |
              (X.get('F3907', 0) == 1) | (X.get('F3915', 0) == 1))

    # --- Build firewall report ---
    firewall_report = {
        'total_features_raw': len(df.columns),
        'total_features_post_firewall': len(X.columns),
        'dropped_total': len(drop_set),
        'dropped_leakage': ['F3912', 'F2230'],
        'dropped_constant': ['F3911'],
        'dropped_empty_count': len(empty_cols),
        'dropped_empty_cols': empty_cols,
        'dropped_duplicate_count': len(dup_drop),
        'dropped_duplicate_cols': dup_drop,
        'leakage_details': [
            {
                'column': 'F3912',
                'reason': "Bank's own fraud flag — LEAKAGE",
                'detail': '79/81 fraud=1; only 3/9001 legit=1 (KS≈0.975, rank #1)',
                'severity': 'CRITICAL'
            },
            {
                'column': 'F2230',
                'reason': 'Investigation batch timestamp — 100% PERFECT SEPARATOR',
                'detail': 'Oct25 → 100% legit (9001/9001); Sep25/Nov25/Dec25 → 100% fraud (81/81). Most severe leakage found.',
                'severity': 'CRITICAL — MOST SEVERE'
            },
            {
                'column': 'F3911',
                'reason': 'Constant column (all values = 0)',
                'detail': 'Zero variance, no information',
                'severity': 'LOW'
            }
        ],
        'verification_shield_count': int(shield.sum()),
        'verification_shield_pct': round(float(shield.mean() * 100), 2),
        'fraud_in_shield': int(y[shield.values].sum()),
    }

    print(f"Engine 0: Dropped {len(drop_set)} columns "
          f"({len(empty_cols)} empty, {len(dup_drop)} duplicate-pair, 4 leakage/artifact/target)")
    print(f"  Post-firewall features: {len(X.columns)}")
    print(f"  Verification shield: {shield.sum()} accounts ({shield.mean()*100:.2f}%), "
          f"fraud among them = {y[shield.values].sum()}")

    return X, y, shield, firewall_report
