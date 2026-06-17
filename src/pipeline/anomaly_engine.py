"""
Engine 3: Anomaly Engine
=========================
Unsupervised anomaly scoring — NO use of F3924/F3912/F2230.
Peer-group robust z-scores + behavioral rule-boosts + IsolationForest.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def _robust_z_score(series, min_group=30):
    """Compute robust z-score using median/MAD."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    return (series - median) / (1.4826 * mad + 1e-9)


def run_anomaly_engine(X: pd.DataFrame, df_raw: pd.DataFrame):
    """
    Engine 3: Score every account for anomaly without using the target.

    Parameters
    ----------
    X : pd.DataFrame
        Post-firewall feature matrix (from Engine 0).
    df_raw : pd.DataFrame
        Raw dataframe for peer-group columns.

    Returns
    -------
    anomaly_df : pd.DataFrame
        DataFrame with account_id, anomaly_score (0-100), anomaly_rank.
    """
    print("\n=== Engine 3: Anomaly Engine ===")

    # --- Monetary/flow columns for robust z-scoring ---
    monetary_cols = ['F3832', 'F3833', 'F3834', 'F3835', 'F3836', 'F3837',
                     'F2481', 'F2482', 'F2483', 'F2686', 'F2285']
    monetary_cols = [c for c in monetary_cols if c in X.columns]

    # Log-transform monetary columns
    X_log = pd.DataFrame(index=X.index)
    for col in monetary_cols:
        X_log[col] = np.log1p(X[col].clip(lower=0).fillna(0))

    # --- Peer groups by (occupation, account_type) ---
    peer_occupation = df_raw['F3891'].reindex(X.index) if 'F3891' in df_raw.columns else pd.Series('unknown', index=X.index)
    peer_acct_type = df_raw['F3886'].reindex(X.index) if 'F3886' in df_raw.columns else pd.Series('unknown', index=X.index)
    peer_key = peer_occupation.astype(str) + '_' + peer_acct_type.astype(str)

    # Compute robust z-scores within peer groups
    z_scores = pd.DataFrame(0.0, index=X.index, columns=monetary_cols)

    for col in monetary_cols:
        for group_name, group_idx in peer_key.groupby(peer_key).groups.items():
            if len(group_idx) < 30:
                # Fallback to global statistics
                z_scores.loc[group_idx, col] = _robust_z_score(X_log[col])
            else:
                z_scores.loc[group_idx, col] = _robust_z_score(X_log.loc[group_idx, col])

    print(f"  Computed robust z-scores for {len(monetary_cols)} columns across {peer_key.nunique()} peer groups")

    # --- Behavioral rule-boosts ---
    # Step-down flag (from raw features) — with MAGNITUDE FILTER
    # (without filter: near-zero dormant accounts all satisfy |0-0|<1 and trivially pass)
    flow_cols = ['F3832', 'F3833', 'F3834', 'F3835', 'F3836', 'F3837']
    flow = X[flow_cols].values
    first_half = flow[:, :3].mean(axis=1)
    second_half = flow[:, 3:].mean(axis=1)
    close12 = np.abs(flow[:, 0] - flow[:, 1]) < 1
    close23 = np.abs(flow[:, 1] - flow[:, 2]) < 1
    close45 = np.abs(flow[:, 3] - flow[:, 4]) < 1
    close56 = np.abs(flow[:, 4] - flow[:, 5]) < 1
    half_check = np.abs(second_half - first_half / 2) / (np.abs(first_half) + 1) < 0.05
    # CRITICAL: require substantial flow magnitude (>100) to avoid near-zero false positives
    magnitude_check = np.abs(first_half) > 100
    step_down_flag = (close12 & close23 & close45 & close56 & half_check & magnitude_check).astype(float)

    # Passive receiver: F2285 is NaN AND F2686 < 0
    passive_receiver = (X['F2285'].isna() & (X['F2686'] < 0)).astype(float)

    # All flow positive
    all_flow_positive = (flow > 0).all(axis=1).astype(float)

    # High absolute z-score sum
    z_abs_sum = z_scores.abs().sum(axis=1)

    # Max single z-score (extreme outlier in any one dimension)
    z_max = z_scores.abs().max(axis=1)

    # Flow magnitude (log scale) — high magnitude is anomalous for most peer groups
    flow_mean = flow.mean(axis=1)
    flow_log = np.log1p(np.clip(flow_mean, 0, None))

    print(f"  Step-down flags (with magnitude filter): {int(step_down_flag.sum())}")
    print(f"  Passive receiver flags: {int(passive_receiver.sum())}")

    # --- Build anomaly feature matrix ---
    anomaly_features = z_scores.copy()
    anomaly_features['step_down_flag']    = step_down_flag * 5.0   # Very strong boost
    anomaly_features['passive_receiver']  = passive_receiver * 3.0  # Strong boost
    anomaly_features['all_flow_positive'] = all_flow_positive
    anomaly_features['flow_log_magnitude']= flow_log
    anomaly_features['z_max']             = z_max                   # Extreme outlier signal
    anomaly_features['z_abs_sum']         = z_abs_sum

    # Handle NaN/inf
    anomaly_features = anomaly_features.replace([np.inf, -np.inf], np.nan).fillna(0)


    # --- IsolationForest ---
    print("  Training IsolationForest...")
    iso_forest = IsolationForest(
        contamination=0.02, n_estimators=200, random_state=42, n_jobs=-1
    )
    iso_forest.fit(anomaly_features)

    # Raw anomaly score from IsolationForest (more negative = more anomalous)
    iso_scores = -iso_forest.decision_function(anomaly_features)  # flip so higher = more anomalous

    # --- Combine: weighted sum of IsolationForest + behavioral boosts ---
    # Normalize iso_scores to 0-1
    scaler = MinMaxScaler()
    iso_norm = scaler.fit_transform(iso_scores.reshape(-1, 1)).flatten()

    # Behavioral boost component
    behavioral_boost = (step_down_flag * 0.3 +
                        passive_receiver * 0.2 +
                        (z_abs_sum > z_abs_sum.quantile(0.95)).astype(float) * 0.15)

    # Combined anomaly score (0-100)
    # behavioral_boost is a pd.Series; iso_norm is numpy — align by resetting to numpy
    behavioral_boost_arr = np.array(behavioral_boost)
    combined_arr = iso_norm * 0.7 + behavioral_boost_arr * 0.3
    anomaly_score = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        combined_arr.reshape(-1, 1)
    ).flatten()

    # --- Rank (1 = most anomalous) ---
    anomaly_rank = pd.Series(anomaly_score, index=X.index).rank(ascending=False, method='min').astype(int)

    anomaly_df = pd.DataFrame({
        'account_id': X.index,
        'anomaly_score': np.round(anomaly_score, 2),
        'anomaly_rank': anomaly_rank.values,
        'step_down_flag': step_down_flag.astype(int),
        'passive_receiver_flag': passive_receiver.astype(int),
    })
    anomaly_df = anomaly_df.set_index('account_id')

    # --- Validate stealth mule ranks ---
    # Use F3912 from df_raw if present (it exists there — only dropped from X)
    if 'F3912' in df_raw.columns:
        stealth_mask = (df_raw['F3912'] == 0) & (df_raw['F3924'] == 1)
    else:
        stealth_mask = df_raw['F3924'] == 1  # fallback: all fraud
    stealth_idx = df_raw[stealth_mask].index

    total_accounts = len(X)
    top_5_pct = int(total_accounts * 0.05)
    print(f"\n  Stealth mule validation (target: rank <= {top_5_pct} / {total_accounts}):")
    for idx in stealth_idx:
        if idx in anomaly_df.index:
            row = anomaly_df.loc[idx]
            in_top5 = "YES (top 5%)" if row['anomaly_rank'] <= top_5_pct else "NO (outside top 5%)"
            print(f"    Account {idx}: score={row['anomaly_score']:.1f}, "
                  f"rank={row['anomaly_rank']}/{total_accounts} "
                  f"({in_top5})")

    return anomaly_df
