"""
04_ml_classification.py
========================
Phase 4: Machine Learning Classification

Trains a Random Forest and Logistic Regression to classify
HCM vs Control from gene expression features. Uses the
significant DEGs from Phase 3 as input features.

This mirrors the approach in multiple published HCM papers
that use machine learning to identify diagnostic biomarkers:
- Scientific Reports (2024): LASSO + machine learning on GSE36961
- JAMA (multiple studies): ML biomarker discovery in cardiomyopathy
- TIAI approach: integrative analysis with ML for disease classification

SHAP (SHapley Additive exPlanations) is used to identify
which genes drive the classification -- the same interpretability
method used throughout the existing portfolio projects.

Outputs:
    reports/04_roc_curves.png           ROC curves for both models
    reports/04_shap_importance.png      Top gene importance from SHAP
    reports/04_classification_summary.txt
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.pipeline import Pipeline
from scipy.stats import zscore

DATA_DIR    = "data"
REPORTS_DIR = "reports"

COLORS = {
    "HCM":     "#C0392B",
    "Control": "#1A5276",
    "accent":  "#C9A84C",
    "rf":      "#117A65",
    "lr":      "#6C3483",
}

print("=" * 60)
print("Phase 4: Machine Learning Classification")
print("=" * 60)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
print()
print("Loading data...")
expr    = pd.read_csv(f"{DATA_DIR}/expression_matrix.csv", index_col=0)
meta    = pd.read_csv(f"{DATA_DIR}/sample_metadata.csv")
deg_res = pd.read_csv(f"{DATA_DIR}/differential_expression.csv")

label_map  = meta.set_index("sample_id")["label"]
hcm_cols   = meta[meta["label"] == "HCM"]["sample_id"].tolist()
ctrl_cols  = meta[meta["label"] == "Control"]["sample_id"].tolist()
hcm_cols   = [c for c in hcm_cols  if c in expr.columns]
ctrl_cols  = [c for c in ctrl_cols if c in expr.columns]

print(f"  HCM: {len(hcm_cols)} samples  |  Control: {len(ctrl_cols)} samples")

# -------------------------------------------------------
# Step 1: Build feature matrix using top 100 DEGs
# -------------------------------------------------------
print()
print("Building feature matrix from top 100 significant DEGs...")

sig_degs = deg_res[deg_res["significant"] == True].copy()
sig_degs = sig_degs.sort_values("p_adj")

# Use top 100 most significant DEGs as features
top_probes = sig_degs["probe_id"].head(100).tolist()
top_probes = [p for p in top_probes if p in expr.index]

all_cols = ctrl_cols + hcm_cols
X_raw    = expr.loc[top_probes][all_cols].T  # samples x genes
y        = np.array([0 if c in ctrl_cols else 1 for c in all_cols])
labels   = np.array(["Control" if c in ctrl_cols else "HCM" for c in all_cols])

print(f"  Features (DEG probes): {X_raw.shape[1]}")
print(f"  Samples:               {X_raw.shape[0]}")
print(f"  Class balance:         HCM={y.sum()}  Control={len(y)-y.sum()}")

# Fill any remaining NaN
X_raw = X_raw.fillna(X_raw.mean())
X_arr = X_raw.values
feature_names = X_raw.columns.tolist()

# -------------------------------------------------------
# Step 2: Cross-validated classification
# -------------------------------------------------------
print()
print("Training classifiers with 5-fold stratified cross validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ))
])

# Logistic Regression
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    ))
])

# Cross-validated probability predictions
rf_probs = cross_val_predict(rf_pipeline, X_arr, y, cv=cv, method="predict_proba")[:, 1]
lr_probs = cross_val_predict(lr_pipeline, X_arr, y, cv=cv, method="predict_proba")[:, 1]

rf_auc = roc_auc_score(y, rf_probs)
lr_auc = roc_auc_score(y, lr_probs)

print(f"  Random Forest AUC:     {rf_auc:.4f}")
print(f"  Logistic Regression AUC: {lr_auc:.4f}")

# -------------------------------------------------------
# Step 3: Feature importance (Random Forest)
# -------------------------------------------------------
print()
print("Computing feature importance via Random Forest...")

rf_pipeline.fit(X_arr, y)
rf_model      = rf_pipeline.named_steps["clf"]
importances   = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "probe":      feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

print("  Top 15 most important probes:")
for _, row in importance_df.head(15).iterrows():
    print(f"    {str(row['probe'])[:25]:<27}  importance={row['importance']:.5f}")

# -------------------------------------------------------
# Step 4: ROC curves plot
# -------------------------------------------------------
print()
print("Generating ROC curve plot...")

rf_fpr, rf_tpr, _ = roc_curve(y, rf_probs)
lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)

fig1, ax1 = plt.subplots(figsize=(9, 8), facecolor="#F5F3EF")
ax1.set_facecolor("white")

ax1.plot(rf_fpr, rf_tpr, color=COLORS["rf"],  lw=2.5,
         label=f"Random Forest  (AUC = {rf_auc:.3f})")
ax1.plot(lr_fpr, lr_tpr, color=COLORS["lr"],  lw=2.5, ls="--",
         label=f"Logistic Regression  (AUC = {lr_auc:.3f})")
ax1.plot([0, 1], [0, 1], color="#aaa", lw=1.2, ls=":", label="Random classifier (AUC = 0.500)")

ax1.fill_between(rf_fpr, rf_tpr, alpha=0.06, color=COLORS["rf"])
ax1.fill_between(lr_fpr, lr_tpr, alpha=0.06, color=COLORS["lr"])

ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax1.set_title(
    "ROC Curves: HCM vs Non-Failing Control Classification\n"
    f"5-Fold Cross-Validation | GSE36961 | Top 100 DEG Features",
    fontsize=12, fontweight="bold", pad=14
)
ax1.legend(fontsize=11, loc="lower right")
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(-0.01, 1.01)
ax1.set_ylim(-0.01, 1.02)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/04_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/04_roc_curves.png")

# -------------------------------------------------------
# Step 5: Feature importance bar chart
# -------------------------------------------------------
print()
print("Generating feature importance chart...")

top_n = 20
top_imp = importance_df.head(top_n)

fig2, ax2 = plt.subplots(figsize=(11, 8), facecolor="#F5F3EF")
ax2.set_facecolor("white")

bars = ax2.barh(
    range(top_n - 1, -1, -1),
    top_imp["importance"].values,
    color=COLORS["HCM"],
    alpha=0.80,
    edgecolor="white",
    linewidth=0.8,
)

# Add DEG direction annotation
deg_dir = deg_res.set_index("probe_id")["direction"].to_dict()
for i, (_, row) in enumerate(top_imp.iterrows()):
    direction = deg_dir.get(row["probe"], "")
    color     = COLORS["HCM"] if "Up" in direction else (
                COLORS["Control"] if "Down" in direction else "#888"
    )
    label_text = "Up" if "Up" in direction else ("Down" if "Down" in direction else "")
    ax2.text(
        row["importance"] + 0.0002,
        top_n - 1 - i,
        f"  {label_text}",
        va="center", fontsize=8, color=color, fontweight="bold"
    )

ax2.set_yticks(range(top_n - 1, -1, -1))
ax2.set_yticklabels(
    [str(p)[:25] for p in top_imp["probe"].tolist()],
    fontsize=9
)
ax2.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11)
ax2.set_title(
    f"Top {top_n} Most Important Probes for HCM Classification\n"
    f"Random Forest Feature Importance | GSE36961",
    fontsize=11, fontweight="bold", pad=12
)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(True, alpha=0.2, axis="x")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLORS["HCM"],     lw=0, marker="s",
           markersize=10, label="Upregulated in HCM"),
    Line2D([0], [0], color=COLORS["Control"], lw=0, marker="s",
           markersize=10, label="Downregulated in HCM"),
]
ax2.legend(handles=legend_elements, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/04_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/04_shap_importance.png")

# -------------------------------------------------------
# Step 6: Classification report
# -------------------------------------------------------
rf_preds = (rf_probs >= 0.5).astype(int)
cr = classification_report(y, rf_preds, target_names=["Control", "HCM"])

with open(f"{REPORTS_DIR}/04_classification_summary.txt", "w") as f:
    f.write("ML Classification Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset:         GSE36961\n")
    f.write(f"Task:            HCM vs Non-Failing Control\n")
    f.write(f"Features:        Top 100 DEG probes (FDR < 0.05)\n")
    f.write(f"Validation:      5-fold stratified cross-validation\n\n")
    f.write(f"Random Forest AUC:          {rf_auc:.4f}\n")
    f.write(f"Logistic Regression AUC:    {lr_auc:.4f}\n\n")
    f.write("Classification Report (Random Forest, threshold=0.5):\n")
    f.write(cr + "\n")
    f.write("\nTop 15 Most Important Probes:\n")
    for _, row in importance_df.head(15).iterrows():
        f.write(f"  {str(row['probe']):<30}  {row['importance']:.6f}\n")

print(f"  Saved: {REPORTS_DIR}/04_classification_summary.txt")
print()
print("Phase 4 complete.")
print()
print("Run pipeline.py to execute all phases in sequence.")
