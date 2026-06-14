"""
Streamlit Dashboard — Mule Account Detection MVP
=================================================
Multi-page dashboard (6 pages) backed by pre-computed artifacts.
Pages map directly to hackathon slides as specified in PRD §12.
"""

import os, sys, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── paths ──────────────────────────────────────────────────────────
# app.py lives in src/dashboard/; project root is two levels up
_here        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_here))
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')


# ── page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mule Account Detection — AI/ML MVP",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Oswald:wght@400;500;700&display=swap');

/* ── base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1c3f 0%, #1B2B5E 60%, #0d1a38 100%);
    color: white;
    border-right: 2px solid #D4821A;
}
section[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #e8eaf6 !important; font-weight: 500; }
section[data-testid="stSidebar"] .stRadio label:hover { color: #D4821A !important; }

/* ── top header ── */
.top-header {
    background: linear-gradient(135deg, #1B2B5E 0%, #0d1a38 100%);
    padding: 1.2rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border-bottom: 4px solid;
    border-image: linear-gradient(90deg, #D4821A, #0D6E6E, #1B2B5E) 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.top-header h1 { color: white; font-family: 'Oswald', sans-serif; font-size: 1.6rem; margin: 0; }
.slide-badge {
    background: #D4821A; color: white; padding: 0.3rem 0.9rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    font-family: 'Oswald', sans-serif; letter-spacing: 1px;
}

/* ── stat cards ── */
.stat-card {
    background: linear-gradient(135deg, #1B2B5E 0%, #0d1a38 100%);
    border-radius: 12px; padding: 1.2rem 1rem;
    text-align: center; border: 1px solid rgba(212,130,26,0.4);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.stat-card:hover { transform: translateY(-3px); }
.stat-number { font-family: 'Oswald', sans-serif; font-size: 2rem; color: #D4821A; font-weight: 700; }
.stat-label  { color: #a0aec0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
.stat-sub    { color: #68d391; font-size: 0.72rem; margin-top: 0.2rem; }

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0D6E6E 0%, #065050 100%);
    border-radius: 10px; padding: 1rem;
    text-align: center; border: 1px solid rgba(13,110,110,0.5);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.metric-value { font-family: 'Oswald', sans-serif; font-size: 1.8rem; color: #fff; font-weight: 700; }
.metric-label { color: #b2f5ea; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-note  { color: #68d391; font-size: 0.68rem; margin-top: 0.2rem; }

/* ── section labels ── */
.section-label {
    font-family: 'Oswald', sans-serif; font-size: 0.75rem; color: #1B2B5E;
    text-transform: uppercase; letter-spacing: 2px; font-weight: 700;
    border-bottom: 2px solid #D4821A; padding-bottom: 4px;
    display: inline-block; margin-bottom: 0.8rem;
}

/* ── risk badges ── */
.badge-HIGH { background:#C0392B; color:white; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-MED  { background:#D4821A; color:white; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-LOW  { background:#1E8449; color:white; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-LEAK { background:#C0392B; color:white; padding:3px 8px; border-radius:4px; font-size:0.72rem; font-weight:700; }
.badge-SAFE { background:#1E8449; color:white; padding:3px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; }

/* ── stealth card ── */
.stealth-card {
    background: linear-gradient(135deg, #2d1b0e 0%, #3d2311 100%);
    border: 2px solid #D4821A; border-radius: 12px; padding: 1.2rem;
    margin-bottom: 1rem;
}
.stealth-title { color: #D4821A; font-family: 'Oswald', sans-serif; font-size: 1.1rem; font-weight: 700; }
.stealth-detail { color: #e2c89f; font-size: 0.85rem; margin-top: 0.4rem; line-height: 1.6; }

/* ── leakage table ── */
.leak-row-critical { background: rgba(192,57,43,0.15) !important; }
.info-box {
    background: rgba(27,43,94,0.08); border-left: 4px solid #1B2B5E;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
}

/* ── pipeline diagram ── */
.pipe-box {
    background: linear-gradient(135deg, #1B2B5E, #0d1a38);
    border: 1px solid #D4821A; border-radius: 10px;
    padding: 0.9rem; text-align: center; color: white;
    font-family: 'Oswald', sans-serif; font-size: 0.95rem;
}
.pipe-arrow { text-align:center; color:#D4821A; font-size:1.4rem; margin:0.2rem 0; }

/* ── bottom footer ── */
.footer {
    background: #F5F7FA; border-top: 2px solid #1B2B5E;
    padding: 0.5rem 1rem; font-size: 0.72rem; color: #888;
    display: flex; justify-content: space-between;
    margin-top: 2rem; border-radius: 0 0 8px 8px;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    pred = pd.read_csv(os.path.join(MODELS_DIR, 'predictions.csv'), index_col=0)
    anom = pd.read_csv(os.path.join(MODELS_DIR, 'anomaly_scores.csv'), index_col=0)
    imp  = pd.read_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'))
    with open(os.path.join(MODELS_DIR, 'metrics.json')) as f:
        mets = json.load(f)
    with open(os.path.join(MODELS_DIR, 'explanations.json')) as f:
        expl = json.load(f)
    return pred, anom, imp, mets, expl

def header(title, slide_tag):
    st.markdown(f"""
    <div class="top-header">
        <h1>🕵️ {title}</h1>
        <span class="slide-badge">{slide_tag}</span>
    </div>""", unsafe_allow_html=True)

def footer_bar():
    st.markdown("""
    <div class="footer">
        <span>Problem Statement 2 — AI/ML Based Classification of Suspicious Mule Accounts</span>
        <span>Hackathon Submission · 2026</span>
    </div>""", unsafe_allow_html=True)

def stat_card(num, label, sub=""):
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return (
        '<div class="stat-card">'
        f'<div class="stat-number">{num}</div>'
        f'<div class="stat-label">{label}</div>'
        f'{sub_html}'
        '</div>'
    )

def metric_card(val, label, note=""):
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{note_html}'
        '</div>'
    )

NAVY = "#1B2B5E"; GOLD = "#D4821A"; TEAL = "#0D6E6E"
GREEN = "#1E8449"; RED = "#C0392B"; PURPLE = "#6B3FA0"

# ── load ───────────────────────────────────────────────────────────
try:
    pred_df, anom_df, imp_df, mets, expl = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.error(f"⚠️ Artifacts not found. Run `build_artifacts.py` first.\n\n{e}")

# ── sidebar nav ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:1rem 0 0.5rem; text-align:center;'>
        <div style='font-family:Oswald,sans-serif;font-size:1.3rem;font-weight:700;color:#D4821A;'>
        🕵️ MuleGuard AI</div>
        <div style='font-size:0.72rem;color:#a0aec0;margin-top:4px;'>
        Hackathon MVP · 2026</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔥 Leakage & Data Quality",
        "📈 Feature Insights",
        "🔍 Account Risk Explorer",
        "👻 Stealth Mule Spotlight",
        "📉 Model Performance",
    ], label_visibility="collapsed")
    st.markdown("---")
    if DATA_OK:
        ds = mets.get('dataset_stats', {})
        st.markdown(f"""<div style='font-size:0.75rem;color:#a0aec0;line-height:1.8;padding:0 0.5rem;'>
        <b style='color:#D4821A;'>Dataset</b><br>
        Accounts: <b style='color:white;'>{ds.get('total_accounts',0):,}</b><br>
        Fraud: <b style='color:#fc8181;'>{ds.get('fraud_accounts',0)}</b><br>
        Imbalance: <b style='color:white;'>1:{int(ds.get('imbalance_ratio',111))}</b><br>
        <br><b style='color:#D4821A;'>Best Model</b><br>
        PR-AUC: <b style='color:#68d391;'>{mets.get('ensemble',{}).get('pr_auc','—')}</b>
        </div>""", unsafe_allow_html=True)

if not DATA_OK:
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW  (proves slides 1, 3, 9)
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    header("System Overview — Mule Account Detection", "SLIDES 1 · 3 · 9")
    ds = mets['dataset_stats']
    bm = mets['baseline']
    em = mets['ensemble']
    rand_auc = bm.get('random_baseline_pr_auc', 0.0089)
    best_auc = max(bm['pr_auc'], em['pr_auc'])
    best = em if em['pr_auc'] >= bm['pr_auc'] else bm

    # ── stat cards ─────────────────────────────────────────────────
    st.markdown('<p class="section-label">Dataset Statistics</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(stat_card(f"{ds['total_accounts']:,}", "Total Accounts"), unsafe_allow_html=True)
    c2.markdown(stat_card(f"{ds['total_features_raw']:,}", "Raw Features"), unsafe_allow_html=True)
    c3.markdown(stat_card(f"{ds['fraud_accounts']}", "Fraud Accounts", "0.89% of total"), unsafe_allow_html=True)
    c4.markdown(stat_card(f"1:{int(ds['imbalance_ratio'])}", "Imbalance Ratio", "Extreme skew"), unsafe_allow_html=True)
    c5.markdown(stat_card(f"{ds['verification_shield_count']:,}", "Pre-Filtered", f"{ds['verification_shield_pct']}% cleared instantly"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── model metric cards ─────────────────────────────────────────
    st.markdown('<p class="section-label">Model Performance (Best Ensemble)</p>', unsafe_allow_html=True)
    best_t = best['thresholds'].get('0.25', best['thresholds'].get('0.2', {}))
    improvement = round(best_auc / rand_auc, 1)
    mc1,mc2,mc3,mc4,mc5 = st.columns(5)
    mc1.markdown(metric_card(f"{best_auc:.3f}", "PR-AUC", f"≈{improvement}× better than random ({rand_auc:.4f})"), unsafe_allow_html=True)
    mc2.markdown(metric_card(f"{best_t.get('recall',0):.1%}", "Recall @0.25", f"{best_t.get('true_positives','?')}/{ds['fraud_accounts']} frauds caught"), unsafe_allow_html=True)
    mc3.markdown(metric_card(f"{best_t.get('precision',0):.1%}", "Precision @0.25", "Of flagged accounts"), unsafe_allow_html=True)
    mc4.markdown(metric_card(f"{best_t.get('f1',0):.3f}", "F1 Score @0.25", "Harmonic mean"), unsafe_allow_html=True)
    mc5.markdown(metric_card(f"{ds['total_features_post_firewall']:,}", "Post-Firewall Features", "After leakage removal"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── pipeline diagram + fraud breakdown ─────────────────────────
    col_pipe, col_break = st.columns([3, 2])

    with col_pipe:
        st.markdown('<p class="section-label">4-Engine Pipeline Architecture</p>', unsafe_allow_html=True)
        boxes = [
            ("📥", "DataSet.csv", "9,082 accounts · 3,924 features", "#374151"),
            ("🔒", "Engine 0: Leakage Firewall", "Drop F3912, F2230, F3911, 63 empty, 246 duplicate-pairs → 3,611 features", NAVY),
            ("⚙️", "Engine 1: Feature Engineering", "~20 engineered signals (flow slope, step-down flag, risk encodings…)", TEAL),
            ("🤖", "Engine 2: Ensemble Classifier", "XGBoost + LightGBM + CatBoost → Meta-Learner | PR-AUC " + str(best_auc), "#065050"),
            ("🔬", "Engine 3: Anomaly Engine", "Peer-group robust z-scores + IsolationForest (unsupervised)", "#3d2311"),
            ("⚖️", "Engine 4: Decision Engine", "Risk tier (LOW/MED/HIGH) + Fraud subtype + SHAP explanation", "#1a0a2e"),
        ]
        for icon, name, detail, bg in boxes:
            st.markdown(f"""<div style='background:{bg};border:1px solid {GOLD};border-radius:10px;
                padding:0.7rem 1rem;margin-bottom:0.4rem;'>
                <span style='font-size:1.1rem;'>{icon}</span>
                <span style='color:{GOLD};font-family:Oswald,sans-serif;font-weight:700;margin-left:6px;'>{name}</span>
                <div style='color:#a0aec0;font-size:0.75rem;margin-top:3px;margin-left:1.8rem;'>{detail}</div>
            </div>
            <div style='text-align:center;color:{GOLD};font-size:1.1rem;margin:-2px 0;'>▼</div>""",
                unsafe_allow_html=True)

    with col_break:
        st.markdown('<p class="section-label">Fraud Account Breakdown</p>', unsafe_allow_html=True)
        fs = mets['fraud_subtypes']
        fig_pie = go.Figure(go.Pie(
            labels=['Type A — Classic<br>(F3912=1, F3908=1)',
                    'Type B — Quiet<br>(F3912=1, F3908=0)',
                    'Type C — Stealth<br>(F3912=0) 🕵️'],
            values=[fs['type_a'], fs['type_b'], fs['type_c']],
            hole=0.55,
            marker=dict(colors=[NAVY, TEAL, GOLD],
                        line=dict(color='#0d1a38', width=2)),
            textinfo='label+value+percent',
            textfont_size=11,
        ))
        fig_pie.update_layout(
            height=320, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
            annotations=[dict(text=f"<b>81</b><br>Total<br>Fraud", x=0.5, y=0.5,
                              font_size=14, showarrow=False, font_color='white')]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(f"""<div style='background:rgba(192,57,43,0.1);border:1px solid #C0392B;
            border-radius:8px;padding:0.7rem;margin-top:0.5rem;'>
            <div style='color:#fc8181;font-family:Oswald,sans-serif;font-size:0.85rem;font-weight:700;'>
            ⚠️ 2 Stealth Mules Missed by Bank</div>
            <div style='color:#fbd38d;font-size:0.75rem;margin-top:4px;'>
            F3912=0 (bank's flag) · Only ML anomaly engine catches these.<br>
            Our system ranks them using unsupervised behavioral analysis.
            </div></div>""", unsafe_allow_html=True)

    footer_bar()


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — LEAKAGE & DATA QUALITY  (proves slides 1, 4, 5)
# ══════════════════════════════════════════════════════════════════════
elif page == "🔥 Leakage & Data Quality":
    header("Leakage & Data Quality Report", "SLIDES 1 · 4 · 5")
    fr = mets['firewall_report']

    # ── before / after ──────────────────────────────────────────────
    st.markdown('<p class="section-label">Feature Count: Before → After Firewall</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(stat_card(f"{fr['total_features_raw']:,}", "Raw Features (before)"), unsafe_allow_html=True)
    c2.markdown(stat_card(f"−{fr['dropped_leakage'].__len__()+1}", "Leakage / Constant", "F3912, F2230, F3911"), unsafe_allow_html=True)
    c3.markdown(stat_card(f"−{fr['dropped_empty_count']}", "Empty Columns", "100% null"), unsafe_allow_html=True)
    c4.markdown(stat_card(f"−{fr['dropped_duplicate_count']}", "Duplicate Pairs", "corr > 0.999, gap 324"), unsafe_allow_html=True)

    # Arrow + result
    st.markdown(f"""<div style='text-align:center;padding:0.8rem 0;'>
        <span style='font-size:1.8rem;'>3,924</span>
        <span style='color:{GOLD};font-size:1.8rem;margin:0 1rem;'>→</span>
        <span style='font-family:Oswald,sans-serif;font-size:2.2rem;color:#68d391;font-weight:700;'>
        {fr['total_features_post_firewall']:,} features</span>
        <span style='color:#a0aec0;font-size:0.85rem;margin-left:0.8rem;'>
        (dropped {fr['dropped_total']} total)</span>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── leakage detail table ────────────────────────────────────────
    st.markdown('<p class="section-label">Leakage Column Details — Highlighted in Red</p>', unsafe_allow_html=True)

    leak_rows = [
        {"Column": "F2230", "Severity": "🚨 MOST SEVERE", "Type": "Investigation Timestamp",
         "Detail": "Oct25 → 100% legit (9,001/9,001) · Sep25/Nov25/Dec25 → 100% fraud (81/81). Perfect separator.",
         "Action": "DROPPED"},
        {"Column": "F3912", "Severity": "🚨 CRITICAL", "Type": "Bank's Fraud Flag (Leakage)",
         "Detail": "79/81 fraud = 1 · Only 3/9,001 legit = 1 · KS ≈ 0.975 · Rank #1 of all 3,924 features.",
         "Action": "DROPPED"},
        {"Column": "F3911", "Severity": "⚠️ LOW", "Type": "Constant Column",
         "Detail": "All values = 0 · Zero variance · No information content.",
         "Action": "DROPPED"},
        {"Column": "63 cols", "Severity": "ℹ️ INFO", "Type": "100% Null Columns",
         "Detail": "Placeholders or ETL artifacts · Provide zero information.",
         "Action": "DROPPED"},
        {"Column": "246 pairs", "Severity": "ℹ️ INFO", "Type": "Duplicate Feature Pairs",
         "Detail": "Pearson corr > 0.999, gap = 324 · Same metric at different entity levels.",
         "Action": "DROPPED (higher-numbered)"},
    ]

    for row in leak_rows:
        sev_color = RED if "CRITICAL" in row["Severity"] or "SEVERE" in row["Severity"] else \
                    GOLD if "LOW" in row["Severity"] else TEAL
        st.markdown(f"""<div style='background:rgba(192,57,43,0.08);border-left:4px solid {sev_color};
            border-radius:0 8px 8px 0;padding:0.8rem 1rem;margin-bottom:0.5rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-family:Oswald,sans-serif;font-size:1rem;color:{sev_color};font-weight:700;'>
                {row["Column"]}</span>
                <span style='background:{sev_color};color:white;padding:2px 10px;border-radius:12px;
                font-size:0.72rem;font-weight:700;'>{row["Severity"]}</span>
            </div>
            <div style='color:#D4821A;font-size:0.8rem;margin:3px 0;'>{row["Type"]} · Action: <b>{row["Action"]}</b></div>
            <div style='color:#d1d5db;font-size:0.8rem;'>{row["Detail"]}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── missingness chart ───────────────────────────────────────────
    st.markdown('<p class="section-label">Missingness as Signal — Fraud vs Legit</p>', unsafe_allow_html=True)
    miss_data = {
        'Feature': ['F515', 'F518', 'F448', 'F450', 'F451', 'F453'],
        'Fraud % Present': [77.8, 55.6, 23.5, 23.5, 23.5, 23.5],
        'Legit % Present': [8.7,  5.2,  21.1, 21.1, 21.1, 21.1],
        'Meaning': ['Behavioral', 'Behavioral', 'Spending', 'Spending', 'Spending (KS rank #2)', 'Spending'],
    }
    miss_df = pd.DataFrame(miss_data)
    fig_miss = go.Figure()
    fig_miss.add_bar(name='Fraud', x=miss_df['Feature'], y=miss_df['Fraud % Present'],
                     marker_color=RED, text=miss_df['Fraud % Present'].apply(lambda x: f'{x}%'),
                     textposition='auto')
    fig_miss.add_bar(name='Legit', x=miss_df['Feature'], y=miss_df['Legit % Present'],
                     marker_color=TEAL, text=miss_df['Legit % Present'].apply(lambda x: f'{x}%'),
                     textposition='auto')
    fig_miss.update_layout(
        barmode='group', title="% of Accounts Where Feature is PRESENT (not NaN)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='white'), yaxis=dict(color='white', title='% Present'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color='white'),
        font=dict(color='white'), height=340, margin=dict(t=40,b=10),
        title_font_color='white',
    )
    fig_miss.add_annotation(text="F515 present → 8.9× fraud rate  |  F448/F450/F451 absent → strong mule indicator",
                            xref='paper', yref='paper', x=0.5, y=-0.08,
                            showarrow=False, font=dict(color=GOLD, size=11))
    st.plotly_chart(fig_miss, use_container_width=True)
    footer_bar()


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE INSIGHTS  (proves slides 2, 5)
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Insights":
    header("Feature Insights — Signals the Bank Missed", "SLIDES 2 · 5")

    # ── top feature importance ──────────────────────────────────────
    st.markdown('<p class="section-label">Top 15 Features by Permutation Importance (PR-AUC Impact)</p>', unsafe_allow_html=True)
    top15 = imp_df.head(15).copy()
    colors = [GOLD if i < 3 else TEAL if i < 8 else NAVY for i in range(len(top15))]
    fig_imp = go.Figure(go.Bar(
        x=top15['importance'], y=top15['feature'],
        orientation='h', marker_color=colors,
        text=top15['importance'].round(4), textposition='auto',
    ))
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='white', title='PR-AUC drop when permuted'),
        yaxis=dict(color='white', autorange='reversed'),
        font=dict(color='white'), height=420,
        margin=dict(t=10, b=10, l=120, r=20),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── bank list vs our list ───────────────────────────────────────
    st.markdown('<p class="section-label">Bank\'s 18-Feature List vs Our Discoveries</p>', unsafe_allow_html=True)
    col_bank, col_ours = st.columns(2)
    with col_bank:
        bank_features = [
            ("F3836", "~3,428 / 3,924", "⚠️ Very weak"),
            ("F3891", "~1,800 / 3,924", "Moderate"),
            ("F3887", "~1,200 / 3,924", "Useful (decoded = months)"),
            ("F3894", "~1,400 / 3,924", "Useful (decoded = years)"),
            ("F3889", "~1,900 / 3,924", "Weak (categorical)"),
        ]
        st.markdown("""<div style='background:rgba(192,57,43,0.1);border:1px solid #C0392B;
            border-radius:8px;padding:0.8rem;'>
            <div style='color:#fc8181;font-family:Oswald,sans-serif;font-size:0.9rem;font-weight:700;
            margin-bottom:0.5rem;'>🏦 Bank's Hint Features (18 listed)</div>""", unsafe_allow_html=True)
        for feat, rank, note in bank_features:
            st.markdown(f"""<div style='display:flex;justify-content:space-between;padding:4px 0;
                border-bottom:1px solid rgba(255,255,255,0.05);'>
                <span style='color:#e2e8f0;font-size:0.82rem;'><b>{feat}</b></span>
                <span style='color:#fc8181;font-size:0.75rem;'>KS rank {rank}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div style='color:#a0aec0;font-size:0.72rem;margin-top:6px;'>
            …and 13 more, mostly bottom-half of all 3,924 features</div></div>""", unsafe_allow_html=True)

    with col_ours:
        our_features = [
            ("F451", "#2 / 3,924", "KS=0.739 · 18× lower in fraud · NOT in bank list"),
            ("F448", "#3 / 3,924", "KS=0.734 · 18× lower in fraud · NOT in bank list"),
            ("F142", "#4 / 3,924", "KS=0.718 · 15× lower in fraud · NOT in bank list"),
            ("F2686", "~#10 / 3,924", "117× higher in fraud (receiver spread)"),
            ("step_down_flag", "Engineered", "Exact-halving detector — catches stealth mule 2"),
        ]
        st.markdown("""<div style='background:rgba(30,132,73,0.1);border:1px solid #1E8449;
            border-radius:8px;padding:0.8rem;'>
            <div style='color:#68d391;font-family:Oswald,sans-serif;font-size:0.9rem;font-weight:700;
            margin-bottom:0.5rem;'>✅ Our Discoveries (not in bank list)</div>""", unsafe_allow_html=True)
        for feat, rank, note in our_features:
            st.markdown(f"""<div style='padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);'>
                <div style='display:flex;justify-content:space-between;'>
                    <span style='color:#e2e8f0;font-size:0.82rem;'><b>{feat}</b></span>
                    <span style='color:#68d391;font-size:0.75rem;'>Rank {rank}</span>
                </div>
                <div style='color:#a0aec0;font-size:0.72rem;'>{note}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── flow series chart ───────────────────────────────────────────
    st.markdown('<p class="section-label">F3832–F3837 Flow Timeline: Stealth Mule vs Typical Legit</p>', unsafe_allow_html=True)
    periods = ['Period 1', 'Period 2', 'Period 3', 'Period 4', 'Period 5', 'Period 6']
    stealth_flow  = [257625.39, 257625.39, 257625.39, 128812.70, 128812.70, 128812.70]
    legit_flow    = [1200, 980, 1450, 1100, 1320, 890]
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Scatter(x=periods, y=stealth_flow, mode='lines+markers',
        name='Stealth Mule (acct 9052) — EXACT HALVING', line=dict(color=GOLD, width=3),
        marker=dict(size=10, symbol='diamond')))
    fig_flow.add_trace(go.Scatter(x=periods, y=legit_flow, mode='lines+markers',
        name='Typical Legit Account', line=dict(color=TEAL, width=2),
        marker=dict(size=8)))
    fig_flow.add_vrect(x0='Period 3', x1='Period 4', fillcolor=RED, opacity=0.1,
        annotation_text="Exact Halving: 257,625 → 128,812", annotation_position="top left",
        annotation_font_color=GOLD)
    fig_flow.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='white'), yaxis=dict(color='white', title='Net Flow Amount (₹)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color='white'),
        font=dict(color='white'), height=320, margin=dict(t=20,b=20),
    )
    st.plotly_chart(fig_flow, use_container_width=True)
    st.markdown(f"""<div class="info-box">
        <b style='color:{GOLD};'>Key insight:</b>
        <span style='color:#d1d5db;font-size:0.83rem;'> T1=T2=T3 then T4=T5=T6=exactly half of T1.
        Real organic spending never produces this mechanical precision.
        The step_down_flag engineered feature detects this pattern.</span>
    </div>""", unsafe_allow_html=True)
    footer_bar()


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — ACCOUNT RISK EXPLORER  (proves slides 4, 9)
# ══════════════════════════════════════════════════════════════════════
elif page == "🔍 Account Risk Explorer":
    header("Account Risk Explorer", "SLIDES 4 · 9")

    # ── filters ─────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.multiselect("Risk Tier", ["HIGH", "MED", "LOW"],
                                     default=["HIGH", "MED"])
    with col_f2:
        subtype_filter = st.selectbox("Fraud Subtype", ["All", "Type A", "Type B", "Type C"])
    with col_f3:
        show_n = st.slider("Show top N accounts", 10, 200, 50)

    # ── filter data ─────────────────────────────────────────────────
    view = pred_df.copy()
    view.index.name = 'account_id'
    view = view.reset_index()
    if tier_filter:
        view = view[view['risk_tier'].isin(tier_filter)]
    if subtype_filter != "All":
        view = view[view['fraud_subtype'].str.startswith(subtype_filter, na=False)]
    view = view.sort_values('risk_score', ascending=False).head(show_n)

    # ── summary chips ───────────────────────────────────────────────
    total_high = (pred_df['risk_tier'] == 'HIGH').sum()
    total_med  = (pred_df['risk_tier'] == 'MED').sum()
    total_low  = (pred_df['risk_tier'] == 'LOW').sum()
    ch1,ch2,ch3,ch4 = st.columns(4)
    ch1.markdown(f"""<div style='background:{RED};border-radius:8px;padding:0.6rem;text-align:center;'>
        <div style='color:white;font-family:Oswald,sans-serif;font-size:1.4rem;font-weight:700;'>{total_high}</div>
        <div style='color:rgba(255,255,255,0.8);font-size:0.72rem;'>HIGH Risk → Block</div>
    </div>""", unsafe_allow_html=True)
    ch2.markdown(f"""<div style='background:{GOLD};border-radius:8px;padding:0.6rem;text-align:center;'>
        <div style='color:white;font-family:Oswald,sans-serif;font-size:1.4rem;font-weight:700;'>{total_med}</div>
        <div style='color:rgba(255,255,255,0.8);font-size:0.72rem;'>MED Risk → Review</div>
    </div>""", unsafe_allow_html=True)
    ch3.markdown(f"""<div style='background:{GREEN};border-radius:8px;padding:0.6rem;text-align:center;'>
        <div style='color:white;font-family:Oswald,sans-serif;font-size:1.4rem;font-weight:700;'>{total_low}</div>
        <div style='color:rgba(255,255,255,0.8);font-size:0.72rem;'>LOW Risk → Monitor</div>
    </div>""", unsafe_allow_html=True)
    ch4.markdown(f"""<div style='background:{NAVY};border-radius:8px;padding:0.6rem;text-align:center;'>
        <div style='color:{GOLD};font-family:Oswald,sans-serif;font-size:1.4rem;font-weight:700;'>{len(view)}</div>
        <div style='color:rgba(255,255,255,0.8);font-size:0.72rem;'>Shown (filtered)</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── clickable table + detail ────────────────────────────────────
    col_table, col_detail = st.columns([3, 2])

    with col_table:
        st.markdown('<p class="section-label">Flagged Accounts (sorted by risk score)</p>', unsafe_allow_html=True)

        def tier_badge(t):
            c = RED if t=='HIGH' else GOLD if t=='MED' else GREEN
            return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:700;">{t}</span>'

        display_cols = ['account_id','risk_score','risk_tier','fraud_subtype','anomaly_score','anomaly_rank']
        display_cols = [c for c in display_cols if c in view.columns]
        disp = view[display_cols].copy()

        # Format
        if 'risk_score' in disp.columns:
            disp['risk_score'] = disp['risk_score'].round(1)
        if 'anomaly_score' in disp.columns:
            disp['anomaly_score'] = disp['anomaly_score'].round(1)

        st.dataframe(
            disp.set_index('account_id') if 'account_id' in disp.columns else disp,
            use_container_width=True, height=400,
            column_config={
                'risk_score': st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100),
                'anomaly_score': st.column_config.ProgressColumn("Anomaly Score", min_value=0, max_value=100),
                'risk_tier': st.column_config.TextColumn("Tier"),
            }
        )

    with col_detail:
        st.markdown('<p class="section-label">Account Detail View</p>', unsafe_allow_html=True)
        all_ids = sorted(pred_df.index.tolist(), key=lambda x: -pred_df.loc[x,'risk_score'])
        # Pre-select first high-risk account
        default_id = int(view['account_id'].iloc[0]) if len(view) > 0 else all_ids[0]
        sel_id = st.selectbox("Select account ID", all_ids,
                              index=all_ids.index(default_id) if default_id in all_ids else 0)

        if sel_id in pred_df.index:
            row = pred_df.loc[sel_id]
            tier = row['risk_tier']
            tier_color = RED if tier=='HIGH' else GOLD if tier=='MED' else GREEN

            st.markdown(f"""<div style='background:{NAVY};border:1px solid {tier_color};
                border-radius:10px;padding:1rem;margin-bottom:0.8rem;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='color:{GOLD};font-family:Oswald,sans-serif;font-size:1.1rem;'>
                    Account #{sel_id}</span>
                    <span style='background:{tier_color};color:white;padding:3px 12px;border-radius:15px;
                    font-size:0.8rem;font-weight:700;'>{tier}</span>
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.7rem;'>
                    <div><div style='color:#a0aec0;font-size:0.72rem;'>Risk Score</div>
                        <div style='color:white;font-size:1.4rem;font-weight:700;'>{row['risk_score']:.1f}</div></div>
                    <div><div style='color:#a0aec0;font-size:0.72rem;'>Anomaly Score</div>
                        <div style='color:{GOLD};font-size:1.4rem;font-weight:700;'>{row.get('anomaly_score',0):.1f}</div></div>
                    <div><div style='color:#a0aec0;font-size:0.72rem;'>Anomaly Rank</div>
                        <div style='color:white;font-size:1rem;'>#{int(row.get('anomaly_rank',0))} of 9,082</div></div>
                    <div><div style='color:#a0aec0;font-size:0.72rem;'>Fraud Subtype</div>
                        <div style='color:#fbd38d;font-size:0.85rem;'>{row.get('fraud_subtype','—') or '—'}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Feature contributions waterfall
            account_key = str(sel_id)
            if account_key in expl and expl[account_key]:
                factors = expl[account_key][:7]
                feat_names = [f['feature'] for f in factors]
                contribs = [f['contribution'] for f in factors]
                values   = [f.get('value') for f in factors]

                st.markdown('<p class="section-label" style="margin-top:0.5rem;">Top Feature Contributions</p>', unsafe_allow_html=True)
                exp_method = mets.get('explanation_method', 'permutation_importance_fallback')
                exp_label = "SHAP" if exp_method == 'shap' else "Feature Deviation (Fallback)"
                st.caption(f"Method: {exp_label}")

                bar_colors = [RED if c > 0 else GREEN for c in contribs]
                fig_wf = go.Figure(go.Bar(
                    x=contribs, y=feat_names, orientation='h',
                    marker_color=bar_colors,
                    text=[f"val={v:.3f}" if v is not None else "" for v in values],
                    textposition='outside',
                ))
                fig_wf.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(color='white', title='Contribution'),
                    yaxis=dict(color='white', autorange='reversed'),
                    font=dict(color='white', size=11),
                    height=280, margin=dict(t=5,b=5,l=10,r=60),
                )
                st.plotly_chart(fig_wf, use_container_width=True)
            else:
                st.info("No explanation available for this account.")
    footer_bar()


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — STEALTH MULE SPOTLIGHT  (proves slides 2, 5, 6)
# ══════════════════════════════════════════════════════════════════════
elif page == "👻 Stealth Mule Spotlight":
    header("Stealth Mule Spotlight — Invisible to Bank's System", "SLIDES 2 · 5 · 6")

    stealth_list = mets.get('stealth_mules', [])
    total_accounts = mets['dataset_stats']['total_accounts']
    top5_cutoff = int(total_accounts * 0.05)

    if not stealth_list:
        st.warning("Stealth mule data not found — re-run build_artifacts.py.")
    else:
        # ── headline banner ────────────────────────────────────────
        st.markdown(f"""<div style='background:linear-gradient(135deg,#2d1b0e,#3d2311);
            border:2px solid {GOLD};border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;
            text-align:center;'>
            <div style='font-family:Oswald,sans-serif;font-size:1.4rem;color:{GOLD};font-weight:700;'>
            ⚠️ 2 Accounts with F3912 = 0 — Completely Invisible to Bank's Existing System</div>
            <div style='color:#e2c89f;font-size:0.85rem;margin-top:0.5rem;'>
            Zero flags in all 18 bank rule-system features · Only our ML anomaly engine detects them
            </div>
        </div>""", unsafe_allow_html=True)

        # ── side-by-side comparison header ────────────────────────
        col_bank_sys, col_our_sys = st.columns(2)
        with col_bank_sys:
            st.markdown(f"""<div style='background:rgba(192,57,43,0.1);border:2px solid {RED};
                border-radius:10px;padding:0.8rem;text-align:center;margin-bottom:1rem;'>
                <div style='color:{RED};font-family:Oswald,sans-serif;font-size:1rem;font-weight:700;'>
                🏦 Bank's 18-Feature Rule System Sees:</div>
                <div style='font-size:2.5rem;margin:0.5rem 0;'>😴</div>
                <div style='color:#fc8181;font-size:1.5rem;font-weight:700;'>NOTHING</div>
                <div style='color:#a0aec0;font-size:0.8rem;margin-top:0.3rem;'>
                F3912=0 · F3908=0 · F3909=0<br>All flags = 0 · Account appears clean
                </div>
            </div>""", unsafe_allow_html=True)
        with col_our_sys:
            ranks = [sm['anomaly_rank'] for sm in stealth_list]
            avg_rank = int(np.mean(ranks)) if ranks else '?'
            in_top5  = all(r <= top5_cutoff for r in ranks)
            icon = "🎯" if in_top5 else "📊"
            st.markdown(f"""<div style='background:rgba(212,130,26,0.1);border:2px solid {GOLD};
                border-radius:10px;padding:0.8rem;text-align:center;margin-bottom:1rem;'>
                <div style='color:{GOLD};font-family:Oswald,sans-serif;font-size:1rem;font-weight:700;'>
                🤖 Our Anomaly Engine Sees:</div>
                <div style='font-size:2.5rem;margin:0.5rem 0;'>{icon}</div>
                <div style='color:{GOLD};font-size:1.5rem;font-weight:700;'>HIGHLY ANOMALOUS</div>
                <div style='color:#a0aec0;font-size:0.8rem;margin-top:0.3rem;'>
                Avg anomaly rank: #{avg_rank} of {total_accounts:,}<br>
                {"✅ Both in top 5%" if in_top5 else f"Top-5% cutoff: rank ≤ {top5_cutoff}"}
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── per-account cards ──────────────────────────────────────
        profile_notes = {
            9046: {
                "nickname": "Passive Metro Mule",
                "profile": "Savings · L365D (7 months) · Metro · Self-employed · Male · Age ~27 · Retail",
                "pattern": "Negative receiver spread (F2686=−0.32) + missing inflow record (F2285=NaN). Money arrives and sits — not yet scattered.",
                "flow": [443.68, 687.84, 770.16, 147.89, 171.96, 154.03],
                "why_missed": "New Metro account with low-risk demographic profile. No outflow activity triggers rules.",
                "why_caught": "all_flow_positive pattern + passive_receiver_flag (F2286<0 & F2285=NaN) + young Metro profile anomaly",
            },
            9052: {
                "nickname": "Dormant Rural Mule",
                "profile": "Savings · G365D (194 months = 16+ years old) · Rural · Self-employed · Male · Age ~36 · Retail · F3913=1",
                "pattern": "Exact step-down halving: 257,625→257,625→257,625→128,812→128,812→128,812. Mechanical redistribution — never organic.",
                "flow": [257625.39, 257625.39, 257625.39, 128812.70, 128812.70, 128812.70],
                "why_missed": "16-year-old trusted dormant account. Rural + self-employed = normal profile. Has F3913=1 verification.",
                "why_caught": "step_down_flag=1 (exact halving) + peer-group robust z-score (30× below rural selfemployed G365D peers)",
            }
        }

        for sm in stealth_list:
            acct_id = sm['account_id']
            notes = profile_notes.get(acct_id, {})
            in_top5_acct = sm['anomaly_rank'] <= top5_cutoff
            rank_color = GREEN if in_top5_acct else GOLD
            rank_label = f"✅ Top 5% (rank #{sm['anomaly_rank']})" if in_top5_acct \
                         else f"Rank #{sm['anomaly_rank']} (outside top 5%)"

            st.markdown(f"""<div class="stealth-card">
                <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                    <div>
                        <div class="stealth-title">👻 Account {acct_id}
                        — {notes.get('nickname','Stealth Mule')}
                        <span style='color:{RED};font-size:0.8rem;margin-left:8px;'>
                        Type C — Stealth</span></div>
                        <div class="stealth-detail">{notes.get('profile','')}</div>
                    </div>
                    <div style='text-align:right;min-width:160px;'>
                        <div style='color:{rank_color};font-family:Oswald,sans-serif;font-size:1rem;font-weight:700;'>
                        {rank_label}</div>
                        <div style='color:#a0aec0;font-size:0.75rem;'>of {total_accounts:,} accounts</div>
                        <div style='color:{GOLD};font-size:0.9rem;margin-top:4px;'>
                        Anomaly Score: <b>{sm['anomaly_score']:.1f}</b></div>
                        <div style='color:#a0aec0;font-size:0.75rem;'>
                        Risk Score: {sm['risk_score']:.1f}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            col_why, col_flow = st.columns([2, 3])
            with col_why:
                st.markdown(f"""
                <div style='background:rgba(192,57,43,0.08);border-radius:8px;padding:0.8rem;margin-bottom:0.5rem;'>
                    <div style='color:{RED};font-size:0.78rem;font-weight:700;margin-bottom:4px;'>
                    🏦 Why bank missed it</div>
                    <div style='color:#d1d5db;font-size:0.78rem;'>{notes.get('why_missed','')}</div>
                </div>
                <div style='background:rgba(30,132,73,0.08);border-radius:8px;padding:0.8rem;'>
                    <div style='color:#68d391;font-size:0.78rem;font-weight:700;margin-bottom:4px;'>
                    🤖 How we caught it</div>
                    <div style='color:#d1d5db;font-size:0.78rem;'>{notes.get('why_caught','')}</div>
                </div>
                """, unsafe_allow_html=True)
                if sm.get('top_factors'):
                    st.markdown('<div style="color:#D4821A;font-size:0.78rem;font-weight:700;margin-top:0.5rem;">Top factors:</div>', unsafe_allow_html=True)
                    for f in sm['top_factors'][:3]:
                        st.markdown(f"<div style='color:#a0aec0;font-size:0.75rem;'>• <b style='color:white;'>{f['feature']}</b> = {f.get('value','?')} (contrib: {f['contribution']:.3f})</div>", unsafe_allow_html=True)

            with col_flow:
                if notes.get('flow'):
                    periods = ['T1','T2','T3','T4','T5','T6']
                    flow_vals = notes['flow']
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Bar(x=periods, y=flow_vals,
                        marker_color=[GOLD if v == max(flow_vals) else TEAL for v in flow_vals],
                        text=[f"₹{v:,.0f}" for v in flow_vals], textposition='auto',
                    ))
                    if acct_id == 9052:
                        fig_f.add_shape(type='line', x0=2.5, x1=2.5, y0=0, y1=max(flow_vals)*1.1,
                            line=dict(color=RED, width=2, dash='dash'))
                        fig_f.add_annotation(x=2.5, y=max(flow_vals)*1.05, text="Exact ÷2 here",
                            showarrow=False, font=dict(color=RED, size=11))
                    fig_f.update_layout(
                        title=f"Account {acct_id}: F3832–F3837 Flow Pattern",
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(color='white'), yaxis=dict(color='white', title='₹'),
                        font=dict(color='white'), height=260,
                        margin=dict(t=35,b=10,l=10,r=10),
                        title_font_color=GOLD,
                    )
                    st.plotly_chart(fig_f, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
    footer_bar()


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL PERFORMANCE  (proves slides 3, 5, 7, 9)
# ══════════════════════════════════════════════════════════════════════
elif page == "📉 Model Performance":
    header("Model Performance — Validation Results", "SLIDES 3 · 5 · 7 · 9")

    bm = mets['baseline']
    em = mets['ensemble']
    rm = mets['rule_based']
    rand_auc = bm.get('random_baseline_pr_auc', 0.0089)
    best_auc = max(bm['pr_auc'], em['pr_auc'])

    # ── comparison table ────────────────────────────────────────────
    st.markdown('<p class="section-label">Head-to-Head Comparison</p>', unsafe_allow_html=True)

    def fmt_t(m, t):
        td = m.get('thresholds', {}).get(str(t), m.get('thresholds', {}).get(t, {}))
        return td.get('precision',0), td.get('recall',0), td.get('f1',0)

    t_use = '0.25'
    rp,rr,rf1 = fmt_t(rm, '0.25') if 'thresholds' in rm else (rm['precision'],rm['recall'],rm['f1'])
    bp,br,bf1 = fmt_t(bm, t_use)
    ep,er,ef1 = fmt_t(em, t_use)

    comp_df = pd.DataFrame({
        'Model': ['Rule-based (F3912 only)', 'Baseline (HistGB)', f'Extended Ensemble ({em.get("ensemble_type","")})'],
        'PR-AUC': [rm['pr_auc'], bm['pr_auc'], em['pr_auc']],
        'Precision @0.25': [rm['precision'], bp, ep],
        'Recall @0.25': [rm['recall'], br, er],
        'F1 @0.25': [rm['f1'], bf1, ef1],
        'Note': ['⚠️ LEAKAGE — uses F3912', '✅ Validated baseline', '🚀 Extended model']
    })
    st.dataframe(
        comp_df.set_index('Model'),
        use_container_width=True,
        column_config={
            'PR-AUC': st.column_config.ProgressColumn("PR-AUC", min_value=0, max_value=1, format="%.4f"),
            'Precision @0.25': st.column_config.ProgressColumn("Precision @0.25", min_value=0, max_value=1, format="%.3f"),
            'Recall @0.25': st.column_config.ProgressColumn("Recall @0.25", min_value=0, max_value=1, format="%.3f"),
            'F1 @0.25': st.column_config.ProgressColumn("F1 @0.25", min_value=0, max_value=1, format="%.3f"),
        }
    )

    improvement = round(best_auc / rand_auc, 1)
    st.success(f"🎯 Best PR-AUC = **{best_auc:.4f}** — approximately **{improvement}×** better than random baseline ({rand_auc:.4f})")

    st.divider()

    col_pr, col_cm = st.columns([3, 2])

    with col_pr:
        st.markdown('<p class="section-label">Precision-Recall Curve (Baseline + Ensemble)</p>', unsafe_allow_html=True)

        fig_pr = go.Figure()
        # Random baseline
        fig_pr.add_trace(go.Scatter(
            x=[0,1], y=[rand_auc, rand_auc], mode='lines',
            name=f'Random ({rand_auc:.4f})',
            line=dict(color='gray', dash='dash', width=1),
        ))
        # Baseline PR curve
        if 'pr_curve' in bm:
            prc, rcc = bm['pr_curve']['precision'], bm['pr_curve']['recall']
            fig_pr.add_trace(go.Scatter(x=rcc, y=prc, mode='lines',
                name=f'Baseline PR-AUC={bm["pr_auc"]}',
                line=dict(color=TEAL, width=2)))
        # Ensemble PR curve
        if 'pr_curve' in em:
            prc, rcc = em['pr_curve']['precision'], em['pr_curve']['recall']
            fig_pr.add_trace(go.Scatter(x=rcc, y=prc, mode='lines',
                name=f'Ensemble PR-AUC={em["pr_auc"]}',
                line=dict(color=GOLD, width=2.5)))

        # Mark threshold points on best model
        best = em if em['pr_auc'] >= bm['pr_auc'] else bm
        for t_label, t_val, sym in [('0.20', '0.2', 'circle'), ('0.25', '0.25', 'diamond'), ('0.30', '0.3', 'square')]:
            td = best.get('thresholds', {}).get(t_val, {})
            if td:
                fig_pr.add_trace(go.Scatter(
                    x=[td['recall']], y=[td['precision']],
                    mode='markers+text',
                    marker=dict(size=12, color=RED, symbol=sym, line=dict(color='white', width=1)),
                    text=[f" t={t_label}"], textposition='top right',
                    textfont=dict(color=RED, size=10),
                    name=f'Threshold {t_label}',
                    showlegend=True,
                ))

        fig_pr.update_layout(
            xaxis=dict(title='Recall', color='white', range=[0,1]),
            yaxis=dict(title='Precision', color='white', range=[0,1]),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor='rgba(0,0,0,0)', font_color='white'),
            font=dict(color='white'), height=380, margin=dict(t=10,b=20),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_cm:
        st.markdown('<p class="section-label">Confusion Matrix @0.25 (Ensemble)</p>', unsafe_allow_html=True)
        td_025 = (em if em['pr_auc']>=bm['pr_auc'] else bm)['thresholds'].get('0.25',
                  (em if em['pr_auc']>=bm['pr_auc'] else bm)['thresholds'].get('0.2',{}))
        if 'confusion_matrix' in td_025:
            cm_vals = np.array(td_025['confusion_matrix'])
            labels = ['Legit', 'Fraud']
            fig_cm = go.Figure(go.Heatmap(
                z=cm_vals, x=labels, y=labels,
                colorscale=[[0,'#1B2B5E'],[0.5,'#0D6E6E'],[1,'#D4821A']],
                text=cm_vals, texttemplate='<b>%{text}</b>',
                textfont=dict(size=20, color='white'),
                showscale=False,
            ))
            # Annotations
            for i, row_lbl in enumerate(labels):
                for j, col_lbl in enumerate(labels):
                    sub = ('✅ Correct' if i==j else '❌ Error')
                    fig_cm.add_annotation(x=j, y=i,
                        text=sub, yshift=-18, showarrow=False,
                        font=dict(size=10, color='rgba(255,255,255,0.7)'))
            fig_cm.update_layout(
                xaxis=dict(title='Predicted', color='white'),
                yaxis=dict(title='Actual', color='white', autorange='reversed'),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), height=300, margin=dict(t=10,b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── threshold slider (stretch) ─────────────────────────────
        st.markdown('<p class="section-label">Threshold Explorer</p>', unsafe_allow_html=True)
        thres_labels = {'0.2':'0.20','0.25':'0.25','0.3':'0.30'}
        best_thresholds = (em if em['pr_auc']>=bm['pr_auc'] else bm).get('thresholds',{})
        t_options = list(best_thresholds.keys())
        if t_options:
            t_sel = st.select_slider("Threshold", options=['0.2','0.25','0.3'],
                                     value='0.25')
            td_sel = best_thresholds.get(t_sel, {})
            if td_sel:
                c1_t,c2_t,c3_t = st.columns(3)
                c1_t.metric("Precision", f"{td_sel.get('precision',0):.3f}")
                c2_t.metric("Recall",    f"{td_sel.get('recall',0):.3f}")
                c3_t.metric("F1",        f"{td_sel.get('f1',0):.3f}")
                st.caption(f"Flagged: {td_sel.get('flagged',0)} accounts · "
                           f"True positives: {td_sel.get('true_positives',0)}/{mets['dataset_stats']['fraud_accounts']}")

    footer_bar()
