# =============================================================================
# PHASE 3 — PREDICTIVE MODELLING (MACHINE LEARNING)
# Actuarial Risk Pipeline — Prudential Life Insurance Assessment
# Author: Priyanka Choudhury
# =============================================================================
# WHAT THIS SCRIPT DOES (in order):
#   3.1  Load master_view.csv
#   3.2  Recompute PCA on 48 Medical_Keyword columns (5 components)
#   3.3  Preprocessing — scaling, encoding, train/val/test split (70/15/15)
#   3.4  Train XGBoost (primary) + Random Forest (benchmark)
#   3.5  Hyperparameter tuning with Optuna (50 trials)
#   3.6  Evaluate — QWK, confusion matrix, calibration plot, lift curve
#   3.7  SHAP explainability — summary plot + waterfall chart
#   3.8  Save model.pkl, scaler.pkl, model_meta.pkl
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, pickle
warnings.filterwarnings("ignore")

# ── UPDATE THIS PATH IF NEEDED ────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Priyanka Choudhury\Downloads\FINAL YEAR PROJECT"
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# 3.1  LOAD DATA
# =============================================================================
print("=" * 60)
print("3.1  LOADING DATA")
print("=" * 60)

df = pd.read_csv(os.path.join(PROJECT_DIR, "master_view.csv"))

print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
print(f"Response range: {df['Response'].min()} to {df['Response'].max()}")
print(f"Nulls in Response: {df['Response'].isnull().sum()}")
print(f"\nResponse distribution:\n{df['Response'].value_counts().sort_index()}")


# =============================================================================
# 3.2  PCA ON 48 MEDICAL_KEYWORD COLUMNS
# Replicates your Phase 2 R analysis in Python.
# 5 PCA components REPLACE the 48 raw binary keyword columns as features.
# =============================================================================
print("\n" + "=" * 60)
print("3.2  PCA ON MEDICAL_KEYWORD COLUMNS")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Find the 48 Medical_Keyword columns
keyword_cols = [c for c in df.columns if c.startswith("Medical_Keyword_")]
print(f"Medical_Keyword columns found: {len(keyword_cols)}")

# Standardise before PCA
scaler_pca = StandardScaler()
keywords_scaled = scaler_pca.fit_transform(df[keyword_cols])

# Fit PCA — 5 components matches your Phase 2 R result
pca = PCA(n_components=5, random_state=42)
pca_scores = pca.fit_transform(keywords_scaled)

# Add PCA scores to dataframe
for i in range(5):
    df[f"PC{i+1}"] = pca_scores[:, i]

explained = pca.explained_variance_ratio_ * 100
print(f"\nVariance explained per component:")
for i, v in enumerate(explained):
    print(f"  PC{i+1}: {v:.2f}%")
print(f"  Total (5 components): {explained.sum():.2f}%")


# =============================================================================
# 3.3  PREPROCESSING
# Drop the 48 raw keyword cols (replaced by PCA scores).
# Scale all numeric features. Split 70 / 15 / 15.
# =============================================================================
print("\n" + "=" * 60)
print("3.3  PREPROCESSING")
print("=" * 60)

from sklearn.model_selection import train_test_split

# Drop original keyword columns and ID
df_model = df.drop(columns=keyword_cols)
if "Id" in df_model.columns:
    df_model = df_model.drop(columns=["Id"])

# Target and features
y = df_model["Response"].values
feature_cols = [c for c in df_model.columns if c != "Response"]
# Encode any text/categorical columns as numeric codes
df_features = df_model[feature_cols].copy()
for col in df_features.select_dtypes(include=['object']).columns:
    print(f"  Encoding categorical column: {col}")
    df_features[col] = df_features[col].astype('category').cat.codes

# Fill remaining nulls with median
X = df_features.fillna(df_features.median(numeric_only=True)).values
print(f"Features used: {len(feature_cols)}")
print(f"Feature columns: {feature_cols}")

# Train/Val/Test split — 70 / 15 / 15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nSplit sizes:")
print(f"  Train : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Val   : {X_val.shape[0]:,} rows ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test  : {X_test.shape[0]:,} rows ({X_test.shape[0]/len(X)*100:.1f}%)")

# Scale — fit on train only to prevent data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
print("\nScaling complete (fitted on train only — no data leakage).")

# XGBoost needs 0-indexed classes: 1->0, 2->1, ..., 8->7
y_train_xgb = y_train - 1
y_val_xgb   = y_val   - 1
y_test_xgb  = y_test  - 1


# =============================================================================
# 3.4  TRAIN MODELS
# XGBoost  — primary model (gradient boosted trees, best for tabular data)
# Random Forest — benchmark model (ensemble of decision trees)
# =============================================================================
print("\n" + "=" * 60)
print("3.4  TRAINING MODELS")
print("=" * 60)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

def qwk(y_true, y_pred):
    """Quadratic Weighted Kappa — main metric for ordered classification."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

# XGBoost with sensible defaults
print("\nTraining XGBoost (default params)...")
xgb_model = XGBClassifier(
    objective="multi:softmax", num_class=8,
    n_estimators=300, learning_rate=0.1, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train_xgb, eval_set=[(X_val, y_val_xgb)], verbose=False)

xgb_val_pred  = xgb_model.predict(X_val)  + 1
xgb_test_pred = xgb_model.predict(X_test) + 1
print(f"  XGBoost  — Val QWK: {qwk(y_val, xgb_val_pred):.4f}  |  Test QWK: {qwk(y_test, xgb_test_pred):.4f}")

# Random Forest benchmark
print("\nTraining Random Forest (benchmark)...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_val_pred  = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)
rf_test_qwk = qwk(y_test, rf_test_pred)
print(f"  Random Forest — Val QWK: {qwk(y_val, rf_val_pred):.4f}  |  Test QWK: {rf_test_qwk:.4f}")

print(f"\n  {'Model':<20} {'Val QWK':>10} {'Test QWK':>10}")
print(f"  {'-'*42}")
print(f"  {'XGBoost':<20} {qwk(y_val, xgb_val_pred):>10.4f} {qwk(y_test, xgb_test_pred):>10.4f}")
print(f"  {'Random Forest':<20} {qwk(y_val, rf_val_pred):>10.4f} {rf_test_qwk:>10.4f}")
print(f"  {'Target':<20} {'> 0.60':>10}")


# =============================================================================
# 3.5  HYPERPARAMETER TUNING WITH OPTUNA
# Optuna automatically searches for the best XGBoost hyperparameters.
# 50 trials — each trial tries a different combination and scores it by QWK.
# Takes approximately 5-10 minutes.
# =============================================================================
print("\n" + "=" * 60)
print("3.5  HYPERPARAMETER TUNING WITH OPTUNA (50 trials)")
print("     Please wait ~5-10 minutes...")
print("=" * 60)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "objective": "multi:softmax", "num_class": 8,
        "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1,
        "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3),
    }
    m = XGBClassifier(**params)
    m.fit(X_train, y_train_xgb, verbose=False)
    return qwk(y_val, m.predict(X_val) + 1)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = {**study.best_params,
               "objective": "multi:softmax", "num_class": 8,
               "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1}

print(f"\nBest Val QWK after tuning: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

print("\nRetraining with best parameters...")
best_xgb = XGBClassifier(**best_params)
best_xgb.fit(X_train, y_train_xgb, verbose=False)

tuned_val_pred  = best_xgb.predict(X_val)  + 1
tuned_test_pred = best_xgb.predict(X_test) + 1
tuned_val_qwk   = qwk(y_val,  tuned_val_pred)
tuned_test_qwk  = qwk(y_test, tuned_test_pred)

print(f"\n  Tuned XGBoost — Val QWK: {tuned_val_qwk:.4f}  |  Test QWK: {tuned_test_qwk:.4f}")
if tuned_test_qwk > 0.60:
    print("  Blueprint target (> 0.60) ACHIEVED!")
else:
    print("  Below 0.60 — paste your score and I will help you improve it.")


# =============================================================================
# 3.6  EVALUATION PLOTS — saved to plots/ subfolder
# (a) Confusion Matrix
# (b) Calibration Plot
# (c) Lift Curve
# (d) Optuna Optimisation History
# =============================================================================
print("\n" + "=" * 60)
print("3.6  GENERATING EVALUATION PLOTS")
print("=" * 60)

from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# (a) Confusion Matrix
print("  Saving confusion_matrix.png...")
cm = confusion_matrix(y_test, tuned_test_pred, labels=list(range(1, 9)))
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(1,9), yticklabels=range(1,9), ax=ax)
ax.set_xlabel("Predicted Risk Score", fontsize=12)
ax.set_ylabel("Actual Risk Score", fontsize=12)
ax.set_title(f"Confusion Matrix — Tuned XGBoost  |  Test QWK = {tuned_test_qwk:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# (b) Calibration Plot
print("  Saving calibration_plot.png...")
y_prob = best_xgb.predict_proba(X_test)
frac_pos, mean_pred_prob = calibration_curve((y_test == 8).astype(int), y_prob[:, 7], n_bins=10)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.plot(mean_pred_prob, frac_pos, "o-", color="steelblue", label="XGBoost (Risk=8)")
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives", fontsize=12)
ax.set_title("Calibration Plot — Risk Score 8", fontsize=13)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "calibration_plot.png"), dpi=150)
plt.show()

# (c) Lift Curve
print("  Saving lift_curve.png...")
y_lift    = (y_test >= 7).astype(int)
prob_high = y_prob[:, 6] + y_prob[:, 7]
order     = np.argsort(-prob_high)
y_sorted  = y_lift[order]
n         = len(y_sorted)
lift      = (np.cumsum(y_sorted) / y_lift.sum()) / (np.arange(1, n+1) / n)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(1, n+1) / n * 100, lift, color="steelblue", linewidth=2, label="XGBoost")
ax.axhline(y=1, color="red", linestyle="--", label="Random baseline")
ax.set_xlabel("% of Population (ranked by predicted risk)", fontsize=12)
ax.set_ylabel("Lift", fontsize=12)
ax.set_title("Lift Curve — High Risk Applicants (Score >= 7)", fontsize=13)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "lift_curve.png"), dpi=150)
plt.show()

# (d) Optuna History
print("  Saving optuna_history.png...")
vals = [t.value for t in study.trials]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(vals, alpha=0.5, color="grey", label="Trial QWK")
ax.plot(np.maximum.accumulate(vals), color="steelblue", linewidth=2, label="Best so far")
ax.axhline(y=0.60, color="red", linestyle="--", label="Target = 0.60")
ax.set_xlabel("Trial Number", fontsize=12)
ax.set_ylabel("Validation QWK", fontsize=12)
ax.set_title("Optuna Hyperparameter Search History", fontsize=13)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "optuna_history.png"), dpi=150)
plt.show()

print(f"\n  All 4 plots saved to: {PLOTS_DIR}")


# =============================================================================
# 3.7  SHAP EXPLAINABILITY
# (a) Summary plot  — which features matter most globally
# (b) Waterfall plot — why the riskiest applicant got a high score
# =============================================================================
print("\n" + "=" * 60)
print("3.7  SHAP EXPLAINABILITY  (1-2 minutes)")
print("=" * 60)

import shap

# Sample 2000 rows to keep it fast
rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
X_sample = X_test[idx]

explainer  = shap.TreeExplainer(best_xgb)
shap_vals  = explainer.shap_values(X_sample)

# Handle both old and new SHAP output formats
if isinstance(shap_vals, list):
    sv_class8 = shap_vals[7]   # old format — list of arrays
else:
    sv_class8 = shap_vals[:, :, 7]  # new format — 3D array

# (a) Summary plot
print("  Saving shap_summary.png...")
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_class8, X_sample, feature_names=feature_cols, show=False, max_display=20)
plt.title("SHAP Summary — Risk Score 8 (Top 20 Features)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.show()

# (b) Waterfall — riskiest applicant
print("  Saving shap_waterfall.png...")
riskiest = np.argmax(best_xgb.predict_proba(X_sample)[:, 7])
exp = shap.Explanation(
    values=sv_class8[riskiest],
    base_values=explainer.expected_value[7],
    data=X_sample[riskiest],
    feature_names=feature_cols
)
plt.figure(figsize=(10, 8))
shap.waterfall_plot(exp, max_display=15, show=False)
plt.title("SHAP Waterfall — Riskiest Applicant", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.show()

# Top 10 features by importance
mean_shap = np.abs(sv_class8).mean(axis=0)
top10 = np.argsort(mean_shap)[::-1][:10]
print("\n  Top 10 features by mean |SHAP| for Risk Score 8:")
for rank, i in enumerate(top10, 1):
    print(f"    {rank:2d}. {feature_cols[i]:<35} {mean_shap[i]:.4f}")


# =============================================================================
# 3.8  SAVE MODEL, SCALER, METADATA
# These three files are loaded by Phase 4 (simulation) and Phase 5 (dashboard)
# =============================================================================
print("\n" + "=" * 60)
print("3.8  SAVING FILES")
print("=" * 60)

with open(os.path.join(PROJECT_DIR, "model.pkl"),  "wb") as f: pickle.dump(best_xgb, f)
with open(os.path.join(PROJECT_DIR, "scaler.pkl"), "wb") as f: pickle.dump(scaler,   f)

meta = {
    "feature_cols":     feature_cols,
    "keyword_cols":     keyword_cols,
    "pca":              pca,
    "scaler_pca":       scaler_pca,
    "n_pca_components": 5,
    "target_classes":   list(range(1, 9)),
}
with open(os.path.join(PROJECT_DIR, "model_meta.pkl"), "wb") as f: pickle.dump(meta, f)

print(f"  model.pkl       saved")
print(f"  scaler.pkl      saved")
print(f"  model_meta.pkl  saved")
print(f"  Location: {PROJECT_DIR}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print(f"  Model      : XGBoost tuned via Optuna (50 trials)")
print(f"  Features   : {len(feature_cols)}  (including 5 PCA components)")
print(f"  Train rows : {X_train.shape[0]:,}")
print(f"  Val QWK    : {tuned_val_qwk:.4f}")
print(f"  Test QWK   : {tuned_test_qwk:.4f}   (target > 0.60)")
print(f"  RF bench   : {rf_test_qwk:.4f}")
print(f"\n  Plots saved : {PLOTS_DIR}")
print(f"    confusion_matrix.png")
print(f"    calibration_plot.png")
print(f"    lift_curve.png")
print(f"    optuna_history.png")
print(f"    shap_summary.png")
print(f"    shap_waterfall.png")
print(f"\n  Ready for Phase 4 — Monte Carlo Simulation")
print("=" * 60)
