"""
Engine 1: Feature Engineering
==============================
All formulas extracted from validated model_pipeline.py — identical logic.
"""

import pandas as pd
import numpy as np


def run_feature_engineering(X: pd.DataFrame, shield: pd.Series):
    """
    Engine 1: Create engineered features from the cleaned feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Output of Engine 0 (leakage-free).
    shield : pd.Series
        Verification-shield boolean mask.

    Returns
    -------
    X_model : pd.DataFrame
        Final feature matrix for modeling.
    feature_names : list
        Names of all features in the model matrix.
    """
    fe = pd.DataFrame(index=X.index)

    # --- Behavioral interactions ---
    fe['receiver_amount_interaction'] = X['F2686'] * np.log1p(X['F2285'].clip(lower=0))
    fe['spend_receive_ratio'] = np.log1p(X['F2481'].clip(lower=0)) / (np.log1p(X['F2482'].clip(lower=0)) + 1)

    # --- Flow timeline features ---
    flow_cols = ['F3832', 'F3833', 'F3834', 'F3835', 'F3836', 'F3837']
    flow = X[flow_cols].values
    fe['flow_slope'] = [np.polyfit(range(6), row, 1)[0] for row in flow]
    fe['all_flow_positive'] = (flow > 0).all(axis=1).astype(int)
    fe['flow_consistency'] = flow.std(axis=1)

    # --- Step-down detector (catches mechanical redistribution) ---
    first_half = flow[:, :3].mean(axis=1)
    second_half = flow[:, 3:].mean(axis=1)
    close12 = np.abs(flow[:, 0] - flow[:, 1]) < 1
    close23 = np.abs(flow[:, 1] - flow[:, 2]) < 1
    close45 = np.abs(flow[:, 3] - flow[:, 4]) < 1
    close56 = np.abs(flow[:, 4] - flow[:, 5]) < 1
    half_check = np.abs(second_half - first_half / 2) / (first_half + 1) < 0.05
    fe['step_down_flag'] = (close12 & close23 & close45 & close56 & half_check).astype(int)

    # --- Missingness as signal ---
    fe['F515_present'] = (~X['F515'].isna()).astype(int)
    fe['F518_present'] = (~X['F518'].isna()).astype(int)
    fe['F448_group_absent'] = (X['F448'].isna() & X['F450'].isna() & X['F451'].isna()).astype(int)

    # --- Risk encodings ---
    fe['occupation_risk'] = X['F3891'].map({
        'student': 1.94, 'agriculture': 1.26, 'retired': 1.04,
        'housewife': 0.90, 'salaried': 0.73, 'selfemployed': 0.66
    }).fillna(1.0)
    fe['location_risk'] = X['F3890'].map({
        'R': 1.44, 'SU': 0.88, 'U': 0.73, 'M': 0.62
    }).fillna(1.0)
    fe['account_type_risk'] = X['F3886'].map({
        'Savings': 1.8, 'Current': 0.3
    }).fillna(1.0)

    # --- Danger flags ---
    fe['f3923_danger'] = (X['F3923'] == 3).astype(int)
    fe['age_interaction_risk'] = ((X['F3887'] < 12) & (X['F3894'] < 30)).astype(int)
    fe['verified_shield'] = shield.astype(int)

    # --- Keep known important raw features ---
    keep_known = [
        'F451', 'F448', 'F142', 'F453', 'F450', 'F2480', 'F261', 'F2284',
        'F2686', 'F2285', 'F2481', 'F2482', 'F2483', 'F2292',
        'F3832', 'F3833', 'F3834', 'F3835', 'F3836', 'F3837',
        'F3908', 'F3909', 'F3887', 'F3894', 'F3923',
        'F3905', 'F3906', 'F3907', 'F3913', 'F3914', 'F3915', 'F3922'
    ]
    keep_known = [c for c in keep_known if c in X.columns]

    X_model = pd.concat([fe, X[keep_known]], axis=1)
    print(f"Engine 1: Final feature matrix shape = {X_model.shape}")

    return X_model, list(X_model.columns)
