# =============================================================================
# PHASE 5 — INTERACTIVE STREAMLIT DASHBOARD
# Actuarial Risk Pipeline — Prudential Life Insurance Assessment
# Author: Priyanka Choudhury
# =============================================================================
# HOW TO RUN:
#   cd "C:\Users\Priyanka Choudhury\Downloads\FINAL YEAR PROJECT"
#   python -m streamlit run phase5_dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Actuarial Risk Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS — refined dark-navy & gold professional theme ──────────────────
st.markdown("""
<style>


html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background-color: #F7F8FC;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1f3d 0%, #1a3560 100%);
    border-right: 1px solid #2a4a7f;
}
[data-testid="stSidebar"] * {
    color: #e8edf5 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f0c040 !important;
    font-family: 'Playfair Display', serif !important;
}
[data-testid="stSidebar"] strong {
    color: #f0c040 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2a4a7f !important;
}

/* Main title area */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #0f1f3d;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #5a6a8a;
    font-weight: 300;
    margin-top: 4px;
    margin-bottom: 20px;
}

/* Section headers */
h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #0f1f3d !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e0e6f0;
    border-radius: 12px;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(15,31,61,0.06);
}
[data-testid="metric-container"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #5a6a8a !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    color: #0f1f3d !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a3560 100%) !important;
    color: #f0c040 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(15,31,61,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(15,31,61,0.35) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #e0e6f0 !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    color: #5a6a8a !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0f1f3d, #1a3560) !important;
    color: #f0c040 !important;
}

/* Sliders */
[data-testid="stSlider"] label {
    font-weight: 500 !important;
    color: #2c3e6b !important;
    font-size: 0.9rem !important;
}

/* Inputs */
.stSelectbox label, .stSlider label {
    font-weight: 500 !important;
    color: #2c3e6b !important;
}

/* Success / info boxes */
.stSuccess {
    background: #e8f5e9 !important;
    border-left: 4px solid #2e7d32 !important;
    border-radius: 8px !important;
}

/* Divider */
hr {
    border-color: #e0e6f0 !important;
    margin: 24px 0 !important;
}

/* Cards */
.info-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #e0e6f0;
    box-shadow: 0 2px 8px rgba(15,31,61,0.05);
    margin-bottom: 16px;
}

/* Risk score display */
.risk-display {
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}

/* Footer */
.footer-text {
    text-align: center;
    color: #9aa5be;
    font-size: 0.82rem;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.3px;
    padding: 20px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── PATH ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# LOAD MODEL AND METADATA
# =============================================================================
@st.cache_resource
def load_model():
    with open(os.path.join(PROJECT_DIR, "model.pkl"),      "rb") as f: model  = pickle.load(f)
    with open(os.path.join(PROJECT_DIR, "scaler.pkl"),     "rb") as f: scaler = pickle.load(f)
    with open(os.path.join(PROJECT_DIR, "model_meta.pkl"), "rb") as f: meta   = pickle.load(f)
    return model, scaler, meta

@st.cache_data
def load_simulation_results():
    return pd.read_csv(os.path.join(PROJECT_DIR, "simulation_results.csv"))

@st.cache_data
def load_original_data():
    return pd.read_csv(os.path.join(PROJECT_DIR, "master_view.csv"))

model, scaler, meta = load_model()
feature_cols = meta["feature_cols"]
keyword_cols = meta["keyword_cols"]
pca          = meta["pca"]
scaler_pca   = meta["scaler_pca"]

df_orig     = load_original_data()
sim_results = load_simulation_results()

# Pre-compute baseline scores array once
baseline_scores_arr = sim_results["scores_baseline"].values

# ── RISK LABELS ───────────────────────────────────────────────────────────────
RISK_LABELS = {
    1: ("Very Low Risk",    "#2ecc71", "#e8f8f0"),
    2: ("Low Risk",         "#27ae60", "#e3f5eb"),
    3: ("Low-Medium Risk",  "#f0b429", "#fef8e7"),
    4: ("Medium Risk",      "#e07b00", "#fdf0e0"),
    5: ("Medium-High Risk", "#d95f02", "#fdeee3"),
    6: ("High Risk",        "#c0392b", "#fce8e6"),
    7: ("Very High Risk",   "#96281b", "#f9e0de"),
    8: ("Extreme Risk",     "#641e16", "#f5d5d2"),
}

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🏥 Actuarial Risk\nPipeline")
    st.markdown("**Prudential Life Insurance**")
    st.markdown("*Priyanka Choudhury — Final Year Project*")
    st.markdown("---")
    st.markdown("### Pipeline Overview")
    st.markdown("""
- **Phase 1** — SQL data engineering
- **Phase 2** — Statistical analysis (R)
- **Phase 3** — XGBoost ML model
- **Phase 4** — Monte Carlo simulation
- **Phase 5** — This dashboard
    """)
    st.markdown("---")
    st.markdown("**Model:** XGBoost")
    st.markdown("**QWK Score:** 0.567")
    st.markdown("**Features:** 82 (incl. 5 PCA)")
    st.markdown("**Training data:** 41,566 applicants")
    st.markdown("**Dataset:** 59,381 applicants")

# =============================================================================
# MAIN TITLE
# =============================================================================
st.markdown('<p class="main-title">🏥 Actuarial Risk Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Predictive risk scoring and portfolio simulation for life insurance underwriting — Prudential Life Insurance Assessment</p>', unsafe_allow_html=True)
st.markdown("---")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2 = st.tabs(["🎯  Applicant Risk Scorer", "📊  Portfolio Simulator"])


# =============================================================================
# TAB 1 — APPLICANT RISK SCORER
# =============================================================================
with tab1:

    st.markdown("### Enter Applicant Details")
    st.markdown("Adjust the inputs below and click **Predict Risk Score** to classify the applicant.")
    st.markdown("")

    col_left, col_spacer, col_right = st.columns([1, 0.08, 1])

    with col_left:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**📋 Personal & Physical Details**")
        ins_age = st.slider("Age (normalised 0–1)",    0.0, 1.0, 0.40, 0.01,
                            help="0 = youngest, 1 = oldest applicant in dataset")
        bmi     = st.slider("BMI (normalised 0–1)",    0.0, 1.0, 0.47, 0.01,
                            help="Body Mass Index — higher values indicate higher weight")
        ht      = st.slider("Height (normalised 0–1)", 0.0, 1.0, 0.60, 0.01)
        wt      = st.slider("Weight (normalised 0–1)", 0.0, 1.0, 0.50, 0.01)
        st.markdown("**💼 Employment & Insurance**")
        emp_info_1 = st.slider("Employment Info 1", 0.0, 1.0, 0.0, 0.01,
                               help="Strongest predictor — odds ratio 3.29 (Phase 2 finding)")
        ins_hist_1 = st.selectbox("Insurance History 1", [0, 1, 2, 3], index=0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**🏥 Medical History**")
        mh2 = st.selectbox("Medical History 2", [0, 1, 2, 3], index=0,
                            help="Chi-Square X²=7,346, p<0.001 — strongest statistical predictor")
        mh4 = st.selectbox("Medical History 4",  [0, 1, 2, 3], index=0,
                            help="Odds ratio = 2.61 from Phase 2 ordinal logistic regression")
        st.markdown("**🔬 Medical Keywords**")
        st.caption("Select all conditions present in applicant's records")
        kw_cols_display = [f"Medical_Keyword_{i}" for i in [1, 3, 5, 8, 15, 23]]
        kw_values = {}
        kc1, kc2 = st.columns(2)
        for i, kw in enumerate(kw_cols_display):
            with (kc1 if i % 2 == 0 else kc2):
                kw_values[kw] = int(st.checkbox(
                    kw.replace("Medical_Keyword_", "Keyword "), value=False))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    btn_col, _, _ = st.columns([1, 3, 3])
    with btn_col:
        predict_btn = st.button("🔍  Predict Risk Score", use_container_width=True)

    # ── PREDICTION OUTPUT ─────────────────────────────────────────────────────
    if predict_btn:
        # Build input feature row
        input_row = {}
        for col in feature_cols:
            if col in df_orig.columns:
                if df_orig[col].dtype == object:
                    input_row[col] = 0
                else:
                    input_row[col] = df_orig[col].median()
            else:
                input_row[col] = 0

        # Override with user inputs
        input_row["Ins_Age"]             = ins_age
        input_row["BMI"]                 = bmi
        input_row["Ht"]                  = ht
        input_row["Wt"]                  = wt
        input_row["Employment_Info_1"]   = emp_info_1
        input_row["Insurance_History_1"] = ins_hist_1
        input_row["Medical_History_2"]   = mh2
        input_row["Medical_History_4"]   = mh4
        for kw, val in kw_values.items():
            if kw in input_row:
                input_row[kw] = val

        # PCA on keyword columns
        kw_input  = np.array([[input_row.get(c, 0) for c in keyword_cols]])
        kw_scaled = scaler_pca.transform(kw_input)
        pc_scores = pca.transform(kw_scaled)[0]
        for i in range(5):
            input_row[f"PC{i+1}"] = pc_scores[i]

        # Predict
        X_input  = np.array([[input_row.get(c, 0) for c in feature_cols]])
        X_scaled = scaler.transform(X_input)
        pred_score = int(model.predict(X_scaled)[0]) + 1
        pred_proba = model.predict_proba(X_scaled)[0]

        label, color, bg = RISK_LABELS[pred_score]

        st.markdown("---")

        r1, r2 = st.columns([1, 2])

        with r1:
            st.markdown(f"""
            <div class="risk-display" style="background:{bg}; border: 2px solid {color};">
                <p style="font-family:'DM Sans',sans-serif; font-size:0.78rem;
                           font-weight:600; text-transform:uppercase; letter-spacing:1px;
                           color:{color}; margin:0 0 4px 0;">PREDICTED RISK SCORE</p>
                <p style="font-family:'Playfair Display',serif; font-size:5rem;
                           font-weight:700; color:{color}; margin:0; line-height:1;">
                    {pred_score}
                </p>
                <p style="font-family:'DM Sans',sans-serif; font-size:1.05rem;
                           font-weight:600; color:{color}; margin:8px 0 0 0;">
                    {label}
                </p>
                <p style="font-family:'DM Sans',sans-serif; font-size:0.82rem;
                           color:#666; margin:4px 0 0 0;">
                    out of 8 risk classes
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Class Probabilities**")
            for i, prob in enumerate(pred_proba):
                _, sc, _ = RISK_LABELS[i+1]
                bar_w = int(prob * 100)
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin:3px 0; gap:8px;">
                    <span style="font-size:0.78rem; color:#5a6a8a;
                                 font-family:'DM Sans',sans-serif; width:52px;">
                        Score {i+1}
                    </span>
                    <div style="flex:1; background:#f0f2f8; border-radius:4px; height:10px;">
                        <div style="width:{bar_w}%; background:{sc};
                                    border-radius:4px; height:10px;"></div>
                    </div>
                    <span style="font-size:0.78rem; color:#2c3e6b; width:38px;
                                 text-align:right; font-family:'DM Sans',sans-serif;">
                        {prob*100:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            # Probability bar chart
            fig_prob = go.Figure(go.Bar(
                x=[f"Score {i+1}" for i in range(8)],
                y=pred_proba * 100,
                marker_color=[RISK_LABELS[i+1][1] for i in range(8)],
                marker_line_color="white",
                marker_line_width=1.5,
                text=[f"{p*100:.1f}%" for p in pred_proba],
                textposition="outside",
                textfont=dict(size=11, family="DM Sans")
            ))
            fig_prob.update_layout(
                title=dict(text="Predicted Probability by Risk Class",
                           font=dict(family="Playfair Display", size=16, color="#0f1f3d")),
                xaxis=dict(title="Risk Score", tickfont=dict(family="DM Sans", size=11)),
                yaxis=dict(title="Probability (%)", tickfont=dict(family="DM Sans", size=11)),
                height=340,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(t=50, b=40, l=40, r=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # SHAP explanation
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    sv = shap_vals[pred_score - 1][0]
                else:
                    sv = shap_vals[0, :, pred_score - 1]

                top_idx   = np.argsort(np.abs(sv))[::-1][:12]
                top_feats = [feature_cols[i] for i in top_idx]
                top_vals  = sv[top_idx]

                fig_shap = go.Figure(go.Bar(
                    x=top_vals[::-1],
                    y=top_feats[::-1],
                    orientation="h",
                    marker_color=["#c0392b" if v > 0 else "#2980b9" for v in top_vals[::-1]],
                    marker_line_color="white",
                    marker_line_width=0.5,
                    text=[f"{v:+.4f}" for v in top_vals[::-1]],
                    textposition="outside",
                    textfont=dict(size=10, family="DM Sans")
                ))
                fig_shap.update_layout(
                    title=dict(
                        text=f"SHAP Feature Importance — Why Score {pred_score}?",
                        font=dict(family="Playfair Display", size=16, color="#0f1f3d")),
                    xaxis=dict(title="SHAP Value (impact on prediction)",
                               tickfont=dict(family="DM Sans", size=10)),
                    yaxis=dict(tickfont=dict(family="DM Sans", size=10)),
                    height=400,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(t=50, b=40, l=160, r=60)
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption("🔴 Red = pushes risk score **higher**   🔵 Blue = pushes risk score **lower**")

            except Exception as e:
                st.info(f"SHAP chart unavailable: {e}")


# =============================================================================
# TAB 2 — PORTFOLIO SIMULATOR
# =============================================================================
with tab2:

    st.markdown("### Portfolio Monte Carlo Simulator")
    st.markdown("Adjust population parameters and run a live simulation to see how portfolio risk changes under different scenarios.")
    st.markdown("")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**👥 Population Size**")
        n_sim = st.slider("Number of applicants", 1000, 10000, 5000, 500)
        st.markdown('</div>', unsafe_allow_html=True)
    with ctrl2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**🕐 Age Adjustment**")
        age_shift = st.slider("Age shift (normalised)", -0.2, 0.2, 0.0, 0.01,
                              help="+0.125 ≈ ageing population by ~10 years")
        st.markdown('</div>', unsafe_allow_html=True)
    with ctrl3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**⚖️ BMI Adjustment**")
        bmi_shift_std = st.slider("BMI shift (std deviations)", -2.0, 2.0, 0.0, 0.1,
                                  help="+1 = one std dev higher BMI population")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    sim_col, _, _ = st.columns([1, 3, 3])
    with sim_col:
        run_sim = st.button("▶  Run Simulation", use_container_width=True)

    st.markdown("---")

    # ── PRE-COMPUTED BASELINE RESULTS ─────────────────────────────────────────
    if not run_sim:
        st.markdown("#### 📈 Phase 4 Pre-computed Baseline Results")
        st.caption("Results from 10,000 applicants simulated via Gaussian Copula in Phase 4. Press **Run Simulation** above for a custom scenario.")
        st.markdown("")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Risk Score", f"{baseline_scores_arr.mean():.3f}")
        m2.metric("VaR 95%",         f"{np.percentile(baseline_scores_arr,95):.1f}")
        m3.metric("VaR 99%",         f"{np.percentile(baseline_scores_arr,99):.1f}")
        m4.metric("High Risk (≥7)",  f"{(baseline_scores_arr>=7).mean()*100:.1f}%")

        st.markdown("")

        scenarios_saved = {
            "Baseline":    sim_results["scores_baseline"].values,
            "Ageing Pop.": sim_results["scores_aged"].values,
            "High BMI":    sim_results["scores_hibmi"].values,
        }
        colors_s = {"Baseline": "#1a3560", "Ageing Pop.": "#e07b00", "High BMI": "#c0392b"}

        fig_comp = go.Figure()
        for name, scores in scenarios_saved.items():
            counts = np.bincount(scores.astype(int), minlength=10)[1:9]
            pcts   = counts / len(scores) * 100
            fig_comp.add_trace(go.Scatter(
                x=list(range(1,9)), y=pcts,
                mode="lines+markers", name=name,
                line=dict(color=colors_s[name], width=3),
                marker=dict(size=9, line=dict(color="white", width=1.5))
            ))
        fig_comp.update_layout(
            title=dict(text="Scenario Comparison — Baseline vs Ageing vs High BMI",
                       font=dict(family="Playfair Display", size=17, color="#0f1f3d")),
            xaxis=dict(title="Risk Score", tickmode="linear", tick0=1, dtick=1,
                       tickfont=dict(family="DM Sans")),
            yaxis=dict(title="% of Portfolio", tickfont=dict(family="DM Sans")),
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(family="DM Sans")),
            margin=dict(t=70, b=50, l=50, r=20)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── LIVE SIMULATION ───────────────────────────────────────────────────────
    if run_sim:
        from scipy import stats as scipy_stats

        with st.spinner(f"Running Gaussian Copula simulation for {n_sim:,} applicants..."):
            bmi_data_c = df_orig["BMI"].dropna()
            bmi_data_c = bmi_data_c[bmi_data_c > 0]
            age_data_c = df_orig["Ins_Age"].dropna()

            bmi_shape, _, bmi_scale = scipy_stats.lognorm.fit(bmi_data_c, floc=0)
            age_mu, age_sigma       = scipy_stats.norm.fit(age_data_c)

            copula_vars = ["BMI", "Ins_Age", "Medical_History_2", "Medical_History_4"]
            corr_mat = df_orig[copula_vars].dropna().corr().values
            try:
                L = np.linalg.cholesky(corr_mat)
            except Exception:
                corr_mat += np.eye(4) * 1e-6
                L = np.linalg.cholesky(corr_mat)

            Z      = np.random.standard_normal((n_sim, 4))
            Z_corr = Z @ L.T
            U      = scipy_stats.norm.cdf(Z_corr)

            sim_bmi = scipy_stats.lognorm.ppf(U[:,0], s=bmi_shape, loc=0, scale=bmi_scale)
            sim_age = scipy_stats.norm.ppf(U[:,1], loc=age_mu, scale=age_sigma)
            sim_mh2 = (U[:,2] > 0.5).astype(int)
            sim_mh4 = (U[:,3] > 0.5).astype(int)

            bmi_std_v = df_orig["BMI"].std()
            sim_bmi = np.clip(sim_bmi + bmi_shift_std * bmi_std_v,
                              df_orig["BMI"].min(), df_orig["BMI"].max())
            sim_age = np.clip(sim_age + age_shift,
                              df_orig["Ins_Age"].min(), df_orig["Ins_Age"].max())

            df_ref_c = df_orig.copy()
            for col in df_ref_c.select_dtypes(include=["object"]).columns:
                df_ref_c[col] = df_ref_c[col].astype("category").cat.codes

            base_idx = np.random.choice(len(df_ref_c), size=n_sim, replace=True)
            synth_df = df_ref_c.iloc[base_idx].copy().reset_index(drop=True)
            synth_df["BMI"]               = sim_bmi
            synth_df["Ins_Age"]           = sim_age
            synth_df["Medical_History_2"] = sim_mh2
            synth_df["Medical_History_4"] = sim_mh4

            kw_present = [c for c in keyword_cols if c in synth_df.columns]
            if len(kw_present) == len(keyword_cols):
                kw_sc = scaler_pca.transform(synth_df[keyword_cols])
                pc_sc = pca.transform(kw_sc)
                for i in range(5):
                    synth_df[f"PC{i+1}"] = pc_sc[:,i]

            synth_df = synth_df.drop(columns=[c for c in keyword_cols
                                               if c in synth_df.columns], errors="ignore")
            if "Id" in synth_df.columns:
                synth_df = synth_df.drop(columns=["Id"])
            for c in feature_cols:
                if c not in synth_df.columns:
                    synth_df[c] = 0

            X_sim    = synth_df[feature_cols].fillna(0).values
            X_sc     = scaler.transform(X_sim)
            scores   = model.predict(X_sc) + 1

        st.success(f"✅  Simulation complete — {n_sim:,} applicants scored!")
        st.markdown("")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Risk Score", f"{scores.mean():.3f}",
                  delta=f"{scores.mean()-sim_results['scores_baseline'].mean():+.3f} vs baseline")
        m2.metric("VaR 95%", f"{np.percentile(scores,95):.1f}")
        m3.metric("VaR 99%", f"{np.percentile(scores,99):.1f}")
        m4.metric("High Risk (≥7)", f"{(scores>=7).mean()*100:.1f}%",
                  delta=f"{(scores>=7).mean()*100-(sim_results['scores_baseline']>=7).mean()*100:+.1f}pp vs baseline")

        st.markdown("")

        ch1, ch2 = st.columns(2)

        with ch1:
            counts = np.bincount(scores.astype(int), minlength=10)[1:9]
            pcts   = counts / len(scores) * 100
            fig_dist = go.Figure(go.Bar(
                x=list(range(1,9)), y=pcts,
                marker_color=["#c0392b" if s >= 7 else "#1a3560" for s in range(1,9)],
                marker_line_color="white", marker_line_width=1.5,
                text=[f"{p:.1f}%" for p in pcts], textposition="outside",
                textfont=dict(size=11, family="DM Sans")
            ))
            fig_dist.add_vline(x=scores.mean(), line_dash="dash", line_color="#e07b00",
                               annotation_text=f"Mean = {scores.mean():.2f}",
                               annotation_font=dict(family="DM Sans"))
            fig_dist.update_layout(
                title=dict(text=f"Risk Distribution — {n_sim:,} Applicants",
                           font=dict(family="Playfair Display", size=16, color="#0f1f3d")),
                xaxis=dict(title="Risk Score", tickmode="linear", tick0=1, dtick=1,
                           tickfont=dict(family="DM Sans")),
                yaxis=dict(title="% of Portfolio", tickfont=dict(family="DM Sans")),
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=50, b=50, l=50, r=20)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with ch2:
            sorted_sc = np.sort(scores)
            cum_pct   = np.arange(1, len(sorted_sc)+1) / len(sorted_sc) * 100
            var95     = np.percentile(scores, 95)
            var99     = np.percentile(scores, 99)

            fig_cdf = go.Figure()
            fig_cdf.add_trace(go.Scatter(
                x=sorted_sc, y=cum_pct, mode="lines",
                line=dict(color="#1a3560", width=2.5), name="Cumulative %",
                fill="tozeroy", fillcolor="rgba(26,53,96,0.08)"
            ))
            fig_cdf.add_hline(y=95, line_dash="dash", line_color="#e07b00",
                              annotation_text=f"VaR 95% = {var95:.0f}",
                              annotation_font=dict(family="DM Sans"))
            fig_cdf.add_hline(y=99, line_dash="dash", line_color="#c0392b",
                              annotation_text=f"VaR 99% = {var99:.0f}",
                              annotation_font=dict(family="DM Sans"))
            fig_cdf.update_layout(
                title=dict(text="Cumulative Distribution & Value at Risk",
                           font=dict(family="Playfair Display", size=16, color="#0f1f3d")),
                xaxis=dict(title="Risk Score", tickmode="linear", tick0=1, dtick=1,
                           tickfont=dict(family="DM Sans")),
                yaxis=dict(title="Cumulative % of Portfolio",
                           tickfont=dict(family="DM Sans")),
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=50, b=50, l=50, r=20)
            )
            st.plotly_chart(fig_cdf, use_container_width=True)

        # Comparison vs baseline
        st.markdown("#### 📊 Your Scenario vs Baseline")
        base_cnt  = np.bincount(baseline_scores_arr.astype(int), minlength=10)[1:9]
        base_pcts = base_cnt / len(baseline_scores_arr) * 100
        sim_pcts  = counts / len(scores) * 100

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Bar(name="Baseline", x=list(range(1,9)), y=base_pcts,
                                marker_color="#9aa5be", marker_line_color="white",
                                marker_line_width=1))
        fig_vs.add_trace(go.Bar(name="Your Scenario", x=list(range(1,9)), y=sim_pcts,
                                marker_color="#1a3560", marker_line_color="white",
                                marker_line_width=1))
        fig_vs.update_layout(
            barmode="group",
            title=dict(text="Your Simulation vs Baseline — Side by Side",
                       font=dict(family="Playfair Display", size=16, color="#0f1f3d")),
            xaxis=dict(title="Risk Score", tickmode="linear", tick0=1, dtick=1,
                       tickfont=dict(family="DM Sans")),
            yaxis=dict(title="% of Portfolio", tickfont=dict(family="DM Sans")),
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(family="DM Sans")),
            margin=dict(t=70, b=50, l=50, r=20)
        )
        st.plotly_chart(fig_vs, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<p class="footer-text">
    Actuarial Risk Pipeline — Priyanka Choudhury | Final Year Project<br>
    XGBoost · Gaussian Copula · Monte Carlo Simulation · SHAP Explainability · Built with Streamlit
</p>
""", unsafe_allow_html=True)
