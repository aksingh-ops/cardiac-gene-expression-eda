"""
02_exploratory_analysis.py
==========================
Phase 2: Exploratory Data Analysis

Inputs:
    data/expression_matrix.csv
    data/sample_metadata.csv

Outputs:
    reports/02_pca.png              PCA plot colored by HCM vs Control
    reports/02_top_variable_genes.png  Heatmap of 50 most variable genes
    reports/02_expression_boxplot.png  Distribution comparison HCM vs Control
    reports/02_correlation_heatmap.png Sample-to-sample correlation

Biological context:
    Before running any statistical test, exploratory analysis tells us
    whether the data has a detectable disease signal. In HCM research,
    PCA consistently separates disease from control samples due to
    the strong transcriptional reprogramming that occurs in the failing
    myocardium (ECM remodeling, metabolic shifts, sarcomere gene changes).
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR    = "data"
REPORTS_DIR = "reports"

COLORS = {
    "HCM":     "#C0392B",
    "Control": "#1A5276",
    "accent":  "#C9A84C",
}

print("=" * 60)
print("Phase 2: Exploratory Data Analysis")
print("=" * 60)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
print()
print("Loading data...")
expr  = pd.read_csv(f"{DATA_DIR}/expression_matrix.csv", index_col=0)
meta  = pd.read_csv(f"{DATA_DIR}/sample_metadata.csv")

# Align samples
common = [c for c in expr.columns if c in meta["sample_id"].values]
expr   = expr[common]
meta   = meta[meta["sample_id"].isin(common)].set_index("sample_id").loc[common].reset_index()

label_map = meta.set_index("sample_id")["label"]
colors_per_sample = [COLORS[label_map[s]] for s in expr.columns]

print(f"  Expression matrix: {expr.shape[0]:,} probes x {expr.shape[1]} samples")
print(f"  HCM: {(meta['label']=='HCM').sum()}  |  Control: {(meta['label']=='Control').sum()}")

# -------------------------------------------------------
# Step 1: PCA on all samples
# -------------------------------------------------------
print()
print("Running PCA...")

# Use top 5,000 most variable probes for PCA (standard practice)
gene_var   = expr.var(axis=1)
top5k      = gene_var.nlargest(5000).index
expr_top   = expr.loc[top5k]

scaler     = StandardScaler()
expr_scaled = scaler.fit_transform(expr_top.T)  # samples x probes

pca      = PCA(n_components=10, random_state=42)
pcs      = pca.fit_transform(expr_scaled)
var_exp  = pca.explained_variance_ratio_ * 100

pc_df = pd.DataFrame({
    "PC1":   pcs[:, 0],
    "PC2":   pcs[:, 1],
    "PC3":   pcs[:, 2],
    "label": meta["label"].values,
})

print(f"  PC1 explains {var_exp[0]:.1f}% of variance")
print(f"  PC2 explains {var_exp[1]:.1f}% of variance")
print(f"  PC3 explains {var_exp[2]:.1f}% of variance")
print(f"  Top 10 PCs explain {sum(var_exp[:10]):.1f}% of variance")

# PCA plot
fig1, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#F5F3EF")
fig1.suptitle(
    "Principal Component Analysis of GSE36961 Gene Expression\n"
    "HCM (n=106) vs Non-Failing Control (n=39) Cardiac Tissue",
    fontsize=12, fontweight="bold", y=0.98
)

for ax, (xpc, ypc, xcol, ycol) in zip(
    axes,
    [("PC1", "PC2", var_exp[0], var_exp[1]),
     ("PC1", "PC3", var_exp[0], var_exp[2])]
):
    ax.set_facecolor("white")
    for label, color in [("HCM", COLORS["HCM"]), ("Control", COLORS["Control"])]:
        sub = pc_df[pc_df["label"] == label]
        ax.scatter(
            sub[xpc], sub[ypc],
            c=color, alpha=0.80, s=55, edgecolors="white",
            linewidths=0.5, label=f"{label} (n={len(sub)})", zorder=3
        )
    ax.set_xlabel(f"{xpc} ({xcol:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"{ypc} ({ycol:.1f}% variance)", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.axvline(0, color="#ccc", lw=0.8)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/02_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/02_pca.png")

# -------------------------------------------------------
# Step 2: Top variable genes heatmap
# -------------------------------------------------------
print()
print("Building top variable genes heatmap...")

top50 = gene_var.nlargest(50).index
expr_top50 = expr.loc[top50]

# Sort columns by label for clear visual grouping
col_order = (
    meta[meta["label"] == "Control"]["sample_id"].tolist() +
    meta[meta["label"] == "HCM"]["sample_id"].tolist()
)
col_order = [c for c in col_order if c in expr_top50.columns]
expr_heatmap = expr_top50[col_order]

# Color bar for sample labels
label_colors = [
    COLORS["Control"] if label_map[s] == "Control" else COLORS["HCM"]
    for s in col_order
]

fig2, ax2 = plt.subplots(figsize=(18, 12), facecolor="#F5F3EF")
ax2.set_facecolor("white")

# Z-score normalize each probe for visual clarity
from scipy.stats import zscore as _zscore
import numpy as np
expr_z = pd.DataFrame(
    np.apply_along_axis(_zscore, 1, expr_heatmap.values.astype(float)),
    index=expr_heatmap.index,
    columns=expr_heatmap.columns
)

im = ax2.imshow(
    expr_z.values.astype(float),
    aspect="auto",
    cmap="RdBu_r",
    vmin=-3, vmax=3
)

ax2.set_yticks(range(len(top50)))
ax2.set_yticklabels(
    [str(g)[:20] for g in top50],
    fontsize=6.5
)
ax2.set_xticks([])
ax2.set_xlabel(
    f"Samples (Control n={len([c for c in col_order if label_map[c]=='Control'])}  |  "
    f"HCM n={len([c for c in col_order if label_map[c]=='HCM'])})",
    fontsize=10
)
ax2.set_title(
    "Top 50 Most Variable Probes Across All Samples\n"
    "Z-score normalized expression | GSE36961",
    fontsize=12, fontweight="bold", pad=12
)

# Add color band for sample groups at top
# Add color band as colored rectangles at bottom of heatmap
n_ctrl_in_order = sum(1 for c in col_order if label_map.get(c) == "Control")
ax2.axvline(n_ctrl_in_order - 0.5, color="white", lw=2, zorder=4)

plt.colorbar(im, ax=ax2, label="Z-score", fraction=0.015, pad=0.01)

legend_patches = [
    mpatches.Patch(color=COLORS["Control"], label="Control"),
    mpatches.Patch(color=COLORS["HCM"],     label="HCM"),
]
ax2.legend(
    handles=legend_patches, loc="lower right",
    fontsize=9, framealpha=0.9
)

plt.savefig(f"{REPORTS_DIR}/02_top_variable_genes.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/02_top_variable_genes.png")

# -------------------------------------------------------
# Step 3: Expression distribution comparison
# -------------------------------------------------------
print()
print("Building expression distribution comparison...")

# Sample 8 random samples per group for readability
rng = np.random.default_rng(42)
ctrl_samps = meta[meta["label"] == "Control"]["sample_id"].tolist()
hcm_samps  = meta[meta["label"] == "HCM"]["sample_id"].tolist()
sample8_ctrl = rng.choice(ctrl_samps, min(8, len(ctrl_samps)), replace=False)
sample8_hcm  = rng.choice(hcm_samps,  min(8, len(hcm_samps)),  replace=False)

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6), sharey=True, facecolor="#F5F3EF")
fig3.suptitle(
    "Expression Distribution Comparison\n"
    "Representative Samples per Group | GSE36961",
    fontsize=12, fontweight="bold"
)

for ax, samples, label, color in [
    (axes3[0], sample8_ctrl, "Control", COLORS["Control"]),
    (axes3[1], sample8_hcm,  "HCM",    COLORS["HCM"]),
]:
    ax.set_facecolor("white")
    plot_data = []
    labels_list = []
    for s in samples:
        vals = expr[s].dropna().values
        plot_data.append(vals)
        labels_list.append(s[:8])

    ax.violinplot(plot_data, showmedians=True)
    ax.set_xticks(range(1, len(samples) + 1))
    ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{label} Samples", fontsize=11, fontweight="bold", color=color)
    ax.set_ylabel("Log2 Expression" if ax == axes3[0] else "", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.15, axis="y")

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/02_expression_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/02_expression_boxplot.png")

# -------------------------------------------------------
# Step 4: Sample-to-sample correlation heatmap
# -------------------------------------------------------
print()
print("Building sample correlation heatmap...")

# Use top 1000 variable probes for correlation
top1k   = gene_var.nlargest(1000).index
corr_mat = expr.loc[top1k][col_order].corr()

fig4, ax4 = plt.subplots(figsize=(14, 12), facecolor="#F5F3EF")
ax4.set_facecolor("white")

im4 = ax4.imshow(
    corr_mat.values.astype(float),
    cmap="coolwarm",
    vmin=0.85, vmax=1.0,
    aspect="auto"
)

n_ctrl = len([c for c in col_order if label_map[c] == "Control"])
n_hcm  = len([c for c in col_order if label_map[c] == "HCM"])

ax4.axhline(n_ctrl - 0.5, color="white", lw=2)
ax4.axvline(n_ctrl - 0.5, color="white", lw=2)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_title(
    "Sample-to-Sample Correlation (Top 1,000 Variable Probes)\n"
    f"GSE36961 | Control (n={n_ctrl}) then HCM (n={n_hcm})",
    fontsize=12, fontweight="bold", pad=12
)

plt.colorbar(im4, ax=ax4, label="Pearson r", fraction=0.02, pad=0.02)

# Add labels via text on the axes directly
ax4.text(n_ctrl / 2, corr_mat.shape[0] + 3, "Control",
         ha="center", va="bottom", color=COLORS["Control"],
         fontsize=9, fontweight="bold")
ax4.text(n_ctrl + n_hcm / 2, corr_mat.shape[0] + 3, "HCM",
         ha="center", va="bottom", color=COLORS["HCM"],
         fontsize=9, fontweight="bold")
ax4.axhline(n_ctrl - 0.5, color="white", lw=2)
ax4.axvline(n_ctrl - 0.5, color="white", lw=2)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/02_correlation_heatmap.png")

print()
print("Phase 2 complete. Run 03_differential_expression.py next.")
