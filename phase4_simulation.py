# =============================================================================
# PHASE 4 — STOCHASTIC RISK SIMULATION (MONTE CARLO)
# Actuarial Risk Pipeline — Prudential Life Insurance Assessment
# Author: Priyanka Choudhury
# =============================================================================
# WHAT THIS SCRIPT DOES (in order):
#   4.1  Load saved model, scaler and metadata from Phase 3
#   4.2  Gaussian Copula — generate 10,000 correlated synthetic applicants
#        (BMI and Age are correlated, not sampled independently)
#   4.3  Pass all 10,000 applicants through the saved XGBoost model
#   4.4  Portfolio risk score distribution
#   4.5  Tail Risk — Value at Risk (VaR) at 95th and 99th percentiles
#   4.6  Scenario Stress Testing — Baseline vs Ageing vs High-BMI
#   4.7  Save all plots and simulation results
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── PATH — update if your folder is different ─────────────────────────────────
PROJECT_DIR = r"C:\Users\Priyanka Choudhury\Downloads\FINAL YEAR PROJECT"
# ─────────────────────────────────────────────────────────────────────────────

PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

np.random.seed(42)  # for reproducibility


# =============================================================================
# 4.1  LOAD SAVED MODEL, SCALER AND METADATA FROM PHASE 3
# =============================================================================
print("=" * 60)
print("4.1  LOADING PHASE 3 OUTPUTS")
print("=" * 60)

with open(os.path.join(PROJECT_DIR, "model.pkl"),      "rb") as f: model     = pickle.load(f)
with open(os.path.join(PROJECT_DIR, "scaler.pkl"),     "rb") as f: scaler    = pickle.load(f)
with open(os.path.join(PROJECT_DIR, "model_meta.pkl"), "rb") as f: meta      = pickle.load(f)

feature_cols = meta["feature_cols"]
keyword_cols = meta["keyword_cols"]
pca          = meta["pca"]
scaler_pca   = meta["scaler_pca"]

print(f"  Model loaded     : XGBoost ({model.n_estimators} estimators)")
print(f"  Features         : {len(feature_cols)}")
print(f"  PCA components   : {meta['n_pca_components']}")

# Load original data to fit distribution parameters and correlation structure
df_orig = pd.read_csv(os.path.join(PROJECT_DIR, "master_view.csv"))
print(f"  Original data    : {df_orig.shape[0]:,} rows loaded for calibration")


# =============================================================================
# 4.2  GAUSSIAN COPULA — CORRELATED SYNTHETIC POPULATION
#
# WHY A COPULA?
# If we sampled BMI and Age independently, we would ignore the real-world
# correlation between them (older people tend to have higher BMI on average).
# Independent sampling understates tail risk.
# A Gaussian Copula preserves the observed correlation structure between
# ALL continuous variables while still using the correct marginal
# distributions (Log-Normal for BMI, Normal for Age) fitted in Phase 2.3.
#
# HOW IT WORKS:
# 1. Compute the correlation matrix from the real data
# 2. Use Cholesky decomposition to generate correlated standard normals
# 3. Transform via CDF to uniform [0,1] (the "copula" step)
# 4. Apply inverse CDF of each marginal distribution to get final values
# =============================================================================
print("\n" + "=" * 60)
print("4.2  GAUSSIAN COPULA — GENERATING 10,000 SYNTHETIC APPLICANTS")
print("=" * 60)

from scipy import stats

N = 10_000   # synthetic portfolio size

# ── Step 1: Fit marginal distributions (calibrated from Phase 2.3) ────────────
# BMI  — Log-Normal (best fit from Phase 2.3, AIC = -88,823)
# Age  — Normal     (best fit from Phase 2.3, AIC = -25,124)

bmi_data = df_orig["BMI"].dropna()
age_data = df_orig["Ins_Age"].dropna()

# Shift BMI slightly above 0 to satisfy Log-Normal requirement
bmi_data = bmi_data[bmi_data > 0]  # remove zeros
bmi_shape, bmi_loc, bmi_scale = stats.lognorm.fit(bmi_data, floc=0)
print(f"  BMI  — Log-Normal: shape={bmi_shape:.4f}, scale={bmi_scale:.4f}")

# Fit Normal to Age
age_mu, age_sigma = stats.norm.fit(age_data)
print(f"  Age  — Normal:     mu={age_mu:.4f}, sigma={age_sigma:.4f}")

# ── Step 2: Compute correlation matrix from real data ─────────────────────────
# Include BMI, Age, and key Medical_History variables
copula_vars = ["BMI", "Ins_Age", "Medical_History_2", "Medical_History_4"]
copula_data = df_orig[copula_vars].dropna()
corr_matrix = copula_data.corr().values

print(f"\n  Correlation matrix (real data):")
corr_df = pd.DataFrame(corr_matrix, index=copula_vars, columns=copula_vars)
print(corr_df.round(4).to_string())

# ── Step 3: Cholesky decomposition ───────────────────────────────────────────
# Cholesky decomposes the correlation matrix so we can generate correlated
# standard normal samples that respect the observed correlation structure.
try:
    L = np.linalg.cholesky(corr_matrix)
except np.linalg.LinAlgError:
    # If matrix is not positive definite, add small regularisation
    corr_matrix += np.eye(len(copula_vars)) * 1e-6
    L = np.linalg.cholesky(corr_matrix)

# ── Step 4: Generate correlated standard normals ──────────────────────────────
Z = np.random.standard_normal((N, len(copula_vars)))
Z_corr = Z @ L.T   # apply correlation structure

# ── Step 5: Transform to uniform via standard normal CDF ──────────────────────
U = stats.norm.cdf(Z_corr)   # now U is correlated uniform [0,1]

# ── Step 6: Apply inverse CDF of each marginal ────────────────────────────────
synth_bmi  = stats.lognorm.ppf(U[:, 0], s=bmi_shape, loc=0,     scale=bmi_scale)
synth_age  = stats.norm.ppf(   U[:, 1], loc=age_mu,             scale=age_sigma)
synth_mh2  = (U[:, 2] > 0.5).astype(int)   # binary Medical_History_2
synth_mh4  = (U[:, 3] > 0.5).astype(int)   # binary Medical_History_4

# Clip to realistic ranges (BMI 0.01-1.0 normalised, Age 0.0-1.0 normalised)
synth_bmi = np.clip(synth_bmi, df_orig["BMI"].min(),     df_orig["BMI"].max())
synth_age = np.clip(synth_age, df_orig["Ins_Age"].min(), df_orig["Ins_Age"].max())

print(f"\n  Synthetic population generated: {N:,} applicants")
print(f"  BMI  — mean: {synth_bmi.mean():.4f}, std: {synth_bmi.std():.4f}")
print(f"  Age  — mean: {synth_age.mean():.4f}, std: {synth_age.std():.4f}")
print(f"  Corr(BMI, Age) observed : {corr_matrix[0,1]:.4f}")
print(f"  Corr(BMI, Age) synthetic: {np.corrcoef(synth_bmi, synth_age)[0,1]:.4f}")


# =============================================================================
# 4.3  BUILD FULL FEATURE MATRIX FOR SYNTHETIC POPULATION
# We need to construct ALL 82 features the model expects.
# For features we haven't explicitly simulated, use the observed median
# from the real data (a standard actuarial assumption).
# =============================================================================
print("\n" + "=" * 60)
print("4.3  BUILDING FEATURE MATRIX FOR SYNTHETIC POPULATION")
print("=" * 60)

def build_synthetic_df(n, bmi_vals, age_vals, mh2_vals, mh4_vals, df_ref):
    """
    Build a synthetic dataframe with all required features.
    Simulated: BMI, Ins_Age, Medical_History_2, Medical_History_4
    All others: sampled from real data distribution (with replacement)
    """
    # Start with random rows from real data as a base
    base_idx = np.random.choice(len(df_ref), size=n, replace=True)
    synth = df_ref.iloc[base_idx].copy().reset_index(drop=True)

    # Override with our correlated simulated values
    synth["BMI"]               = bmi_vals
    synth["Ins_Age"]           = age_vals
    synth["Medical_History_2"] = mh2_vals
    synth["Medical_History_4"] = mh4_vals

    # Handle categorical Product_Info_2 if present
    for col in synth.select_dtypes(include=["object"]).columns:
        synth[col] = synth[col].astype("category").cat.codes

    return synth

def prepare_for_model(synth_df, feature_cols, keyword_cols, pca, scaler_pca, scaler):
    """
    Apply PCA to keyword columns, assemble feature matrix, scale.
    Mirrors exactly what was done in Phase 3.
    """
    df_work = synth_df.copy()

    # Compute PCA scores for Medical_Keyword columns
    kw_cols_present = [c for c in keyword_cols if c in df_work.columns]
    if len(kw_cols_present) == len(keyword_cols):
        kw_scaled = scaler_pca.transform(df_work[keyword_cols])
        pc_scores = pca.transform(kw_scaled)
        for i in range(5):
            df_work[f"PC{i+1}"] = pc_scores[:, i]

    # Drop keyword columns (replaced by PCA)
    df_work = df_work.drop(columns=[c for c in keyword_cols if c in df_work.columns], errors="ignore")

    # Drop ID if present
    if "Id" in df_work.columns:
        df_work = df_work.drop(columns=["Id"])

    # Select exactly the features the model expects
    missing = [c for c in feature_cols if c not in df_work.columns]
    if missing:
        for c in missing:
            df_work[c] = 0  # fill missing features with 0

    X = df_work[feature_cols].fillna(0).values

    # Scale using Phase 3 scaler
    X_scaled = scaler.transform(X)
    return X_scaled

# Prepare baseline synthetic population
df_ref_clean = df_orig.copy()
for col in df_ref_clean.select_dtypes(include=["object"]).columns:
    df_ref_clean[col] = df_ref_clean[col].astype("category").cat.codes

synth_df_baseline = build_synthetic_df(N, synth_bmi, synth_age, synth_mh2, synth_mh4, df_ref_clean)
X_synth_baseline  = prepare_for_model(synth_df_baseline, feature_cols, keyword_cols, pca, scaler_pca, scaler)

print(f"  Feature matrix shape: {X_synth_baseline.shape}")
print(f"  Ready to score through XGBoost model")


# =============================================================================
# 4.4  PORTFOLIO RISK SCORING
# Pass all 10,000 synthetic applicants through the saved Phase 3 model.
# Generate the portfolio-wide distribution of predicted risk scores (1-8).
# =============================================================================
print("\n" + "=" * 60)
print("4.4  PORTFOLIO RISK SCORING")
print("=" * 60)

# Predict risk scores (model outputs 0-7, add 1 to get 1-8)
baseline_scores = model.predict(X_synth_baseline) + 1

# Distribution summary
score_counts = pd.Series(baseline_scores).value_counts().sort_index()
print(f"\n  Portfolio risk score distribution (N={N:,}):")
print(f"  {'Score':<8} {'Count':>8} {'Pct':>8}")
print(f"  {'-'*26}")
for score, count in score_counts.items():
    bar = "█" * int(count / N * 40)
    print(f"  {int(score):<8} {count:>8,} {count/N*100:>7.1f}%  {bar}")

high_risk_pct = (baseline_scores >= 7).sum() / N * 100
print(f"\n  High risk (score >= 7): {high_risk_pct:.1f}% of portfolio")


# =============================================================================
# 4.5  TAIL RISK — VALUE AT RISK (VaR)
# VaR tells us: "What is the worst risk score at a given confidence level?"
# VaR 95%: 95% of applicants score AT OR BELOW this level
# VaR 99%: 99% of applicants score AT OR BELOW this level
# The remaining 1-5% are the extreme tail — the highest-risk applicants.
# =============================================================================
print("\n" + "=" * 60)
print("4.5  TAIL RISK — VALUE AT RISK (VaR)")
print("=" * 60)

var_95 = np.percentile(baseline_scores, 95)
var_99 = np.percentile(baseline_scores, 99)
mean_score = baseline_scores.mean()
median_score = np.median(baseline_scores)

print(f"  Mean risk score     : {mean_score:.3f}")
print(f"  Median risk score   : {median_score:.3f}")
print(f"  VaR 95%             : {var_95:.1f}  (95% of portfolio scores at or below this)")
print(f"  VaR 99%             : {var_99:.1f}  (99% of portfolio scores at or below this)")
print(f"  Tail (score = 8)    : {(baseline_scores == 8).sum():,} applicants ({(baseline_scores == 8).mean()*100:.1f}%)")

# ── Plot: Portfolio distribution with VaR annotations ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
bins = np.arange(0.5, 9.5, 1)
n_hist, _, patches = ax.hist(baseline_scores, bins=bins, color="steelblue",
                              edgecolor="white", linewidth=0.5, alpha=0.85)

# Colour tail region red
for i, patch in enumerate(patches):
    score = i + 1
    if score >= var_95:
        patch.set_facecolor("#d62728")
        patch.set_alpha(0.85)

ax.axvline(var_95, color="orange", linewidth=2, linestyle="--",
           label=f"VaR 95% = {var_95:.0f}")
ax.axvline(var_99, color="red", linewidth=2, linestyle="-",
           label=f"VaR 99% = {var_99:.0f}")
ax.axvline(mean_score, color="green", linewidth=2, linestyle=":",
           label=f"Mean = {mean_score:.2f}")

ax.set_xlabel("Predicted Risk Score", fontsize=13)
ax.set_ylabel("Number of Applicants", fontsize=13)
ax.set_title(f"Portfolio Risk Distribution — {N:,} Synthetic Applicants\n"
             f"VaR 95% = {var_95:.0f}  |  VaR 99% = {var_99:.0f}  |  Mean = {mean_score:.2f}",
             fontsize=13)
ax.set_xticks(range(1, 9))
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path_var = os.path.join(PLOTS_DIR, "portfolio_var.png")
plt.savefig(path_var, dpi=150)
plt.show()
print(f"\n  Saved: {path_var}")


# =============================================================================
# 4.6  SCENARIO STRESS TESTING
# Compare three portfolio scenarios on a single chart:
#   Baseline     — current observed population parameters
#   Ageing       — mean Age shifted up by 10 years (normalised equivalent)
#   High-BMI     — mean BMI shifted up by 1 standard deviation
#
# This directly mirrors real actuarial stress-testing practice.
# =============================================================================
print("\n" + "=" * 60)
print("4.6  SCENARIO STRESS TESTING")
print("=" * 60)

bmi_std = df_orig["BMI"].std()
age_range = df_orig["Ins_Age"].max() - df_orig["Ins_Age"].min()

# ── Scenario 1: Baseline (already computed) ───────────────────────────────────
scores_baseline = baseline_scores.copy()

# ── Scenario 2: Ageing population (shift Age up by 10 years normalised) ───────
# In the dataset Age is normalised 0-1. 10 years ~ 10/80 = 0.125 shift.
age_shift = 0.125
synth_age_aged = np.clip(synth_age + age_shift,
                         df_orig["Ins_Age"].min(),
                         df_orig["Ins_Age"].max())

synth_df_aged = build_synthetic_df(N, synth_bmi, synth_age_aged, synth_mh2, synth_mh4, df_ref_clean)
X_synth_aged  = prepare_for_model(synth_df_aged, feature_cols, keyword_cols, pca, scaler_pca, scaler)
scores_aged   = model.predict(X_synth_aged) + 1

# ── Scenario 3: High-BMI population (shift BMI up by 1 std dev) ───────────────
synth_bmi_high = np.clip(synth_bmi + bmi_std,
                         df_orig["BMI"].min(),
                         df_orig["BMI"].max())

synth_df_hibmi = build_synthetic_df(N, synth_bmi_high, synth_age, synth_mh2, synth_mh4, df_ref_clean)
X_synth_hibmi  = prepare_for_model(synth_df_hibmi, feature_cols, keyword_cols, pca, scaler_pca, scaler)
scores_hibmi   = model.predict(X_synth_hibmi) + 1

# ── Summary table ─────────────────────────────────────────────────────────────
scenarios = {
    "Baseline":     scores_baseline,
    "Ageing Pop.":  scores_aged,
    "High BMI":     scores_hibmi,
}

print(f"\n  {'Scenario':<15} {'Mean':>7} {'Median':>8} {'VaR 95%':>9} {'VaR 99%':>9} {'High Risk%':>12}")
print(f"  {'-'*58}")
for name, scores in scenarios.items():
    print(f"  {name:<15} {scores.mean():>7.3f} {np.median(scores):>8.3f} "
          f"{np.percentile(scores,95):>9.1f} {np.percentile(scores,99):>9.1f} "
          f"{(scores>=7).mean()*100:>11.1f}%")

# ── Plot: Three scenarios on one chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

colors = ["steelblue", "darkorange", "crimson"]
alphas = [0.6, 0.6, 0.6]
bins   = np.arange(0.5, 9.5, 1)

for (name, scores), color, alpha in zip(scenarios.items(), colors, alphas):
    counts, _ = np.histogram(scores, bins=bins)
    pcts = counts / N * 100
    ax.plot(range(1, 9), pcts, "o-", color=color, linewidth=2.5,
            markersize=8, alpha=0.9, label=name)
    ax.fill_between(range(1, 9), pcts, alpha=0.15, color=color)

ax.set_xlabel("Risk Score", fontsize=13)
ax.set_ylabel("% of Portfolio", fontsize=13)
ax.set_title("Scenario Stress Test — Portfolio Risk Distribution\n"
             "Baseline vs Ageing Population vs High BMI",
             fontsize=13)
ax.set_xticks(range(1, 9))
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
path_stress = os.path.join(PLOTS_DIR, "stress_test_scenarios.png")
plt.savefig(path_stress, dpi=150)
plt.show()
print(f"\n  Saved: {path_stress}")

# ── Plot: Side-by-side bar chart ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
scenario_titles = ["Baseline", "Ageing Population\n(+10 years)", "High BMI\n(+1 std dev)"]

for ax, (name, scores), color, title in zip(axes, scenarios.items(), colors, scenario_titles):
    counts, _ = np.histogram(scores, bins=bins)
    pcts = counts / N * 100
    bars = ax.bar(range(1, 9), pcts, color=color, alpha=0.75, edgecolor="white")

    # Highlight high risk bars
    for i, bar in enumerate(bars):
        if i + 1 >= 7:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.5)

    ax.set_title(f"{title}\nMean={scores.mean():.2f}  VaR99={np.percentile(scores,99):.0f}",
                 fontsize=11)
    ax.set_xlabel("Risk Score", fontsize=11)
    ax.set_xticks(range(1, 9))
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("% of Portfolio", fontsize=12)
plt.suptitle("Stress Test — Side-by-Side Comparison", fontsize=13, y=1.02)
plt.tight_layout()
path_bars = os.path.join(PLOTS_DIR, "stress_test_bars.png")
plt.savefig(path_bars, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {path_bars}")


# =============================================================================
# 4.7  SAVE SIMULATION RESULTS AS CSV
# Save the scores for all three scenarios so they can be loaded
# directly into the Phase 5 Streamlit dashboard without re-running.
# =============================================================================
print("\n" + "=" * 60)
print("4.7  SAVING SIMULATION RESULTS")
print("=" * 60)

results_df = pd.DataFrame({
    "synth_bmi":        synth_bmi,
    "synth_age":        synth_age,
    "scores_baseline":  scores_baseline,
    "scores_aged":      scores_aged,
    "scores_hibmi":     scores_hibmi,
})

results_path = os.path.join(PROJECT_DIR, "simulation_results.csv")
results_df.to_csv(results_path, index=False)
print(f"  Simulation results saved: {results_path}")
print(f"  Rows: {len(results_df):,}  |  Columns: {list(results_df.columns)}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"\n  Synthetic population  : {N:,} applicants (Gaussian Copula)")
print(f"  Correlation preserved : BMI-Age r = {np.corrcoef(synth_bmi, synth_age)[0,1]:.4f}")
print(f"  Marginals             : Log-Normal (BMI), Normal (Age)")
print(f"\n  BASELINE PORTFOLIO:")
print(f"    Mean risk score     : {scores_baseline.mean():.3f}")
print(f"    VaR 95%             : {np.percentile(scores_baseline, 95):.1f}")
print(f"    VaR 99%             : {np.percentile(scores_baseline, 99):.1f}")
print(f"    High risk (>=7)     : {(scores_baseline>=7).mean()*100:.1f}%")
print(f"\n  STRESS TEST IMPACT:")
print(f"    Ageing scenario     : mean score {scores_aged.mean():.3f}  "
      f"(vs baseline {scores_baseline.mean():.3f})")
print(f"    High-BMI scenario   : mean score {scores_hibmi.mean():.3f}  "
      f"(vs baseline {scores_baseline.mean():.3f})")
print(f"\n  Files saved:")
print(f"    simulation_results.csv")
print(f"    plots/portfolio_var.png")
print(f"    plots/stress_test_scenarios.png")
print(f"    plots/stress_test_bars.png")
print(f"\n  Next step: Phase 5 — Streamlit Dashboard")
print("=" * 60)
