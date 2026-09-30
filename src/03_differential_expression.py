"""
03_differential_expression.py
==============================
Phase 3: Differential Expression Analysis

Identifies which genes are significantly up or down regulated
in HCM cardiac tissue compared to non-failing controls.

Method:
    Welch two-sample t-test per probe (does not assume equal variance)
    Multiple testing correction: Benjamini-Hochberg FDR
    Significance thresholds: adjusted p < 0.05, |log2FC| > 1.0

Outputs:
    data/differential_expression.csv      Full DEG results table
    data/top_degs.csv                     Top 20 up and down regulated
    reports/03_volcano_plot.png           Volcano plot with labeled top genes
    reports/03_deg_heatmap.png            Heatmap of significant DEGs
    reports/03_deg_summary.txt            Summary statistics

Biological context:
    In HCM, well-established transcriptional signatures include:
    - Upregulation of fetal cardiac genes (ACTA1, MYH7, NPPA, NPPB)
    - Upregulation of fibrosis markers (COL1A1, COL3A1, FN1)
    - Downregulation of metabolic genes (fatty acid oxidation pathway)
    - MYBPC3 expression changes (the most common HCM mutation gene)
    These are validated against TIAI-published findings in:
    Ali et al. J Am Heart Assoc. 2025;14(1):e035780
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import zscore

DATA_DIR    = "data"
REPORTS_DIR = "reports"

COLORS = {
    "HCM":     "#C0392B",
    "Control": "#1A5276",
    "accent":  "#C9A84C",
    "up":      "#C0392B",
    "down":    "#1A5276",
    "ns":      "#aaaaaa",
}

# Known HCM marker genes to highlight in volcano plot
# Based on published literature and TIAI research
HCM_KNOWN_GENES = {
    # Fetal cardiac program (upregulated in HCM)
    "NPPA",    # natriuretic peptide A -- hallmark of cardiac stress
    "NPPB",    # natriuretic peptide B -- BNP, clinical HCM biomarker
    "MYH7",    # beta-myosin heavy chain -- upregulated in HCM
    "ACTA1",   # skeletal actin -- fetal gene reactivation
    # Fibrosis markers
    "COL1A1",  # collagen type 1 -- fibrosis
    "COL3A1",  # collagen type 3 -- fibrosis
    "FN1",     # fibronectin -- ECM remodeling
    # Sarcomere genes
    "MYBPC3",  # myosin binding protein C -- #1 HCM mutation gene
    "MYH6",    # alpha-myosin -- often downregulated in HCM
    "TNNT2",   # troponin T -- sarcomere regulation
}

print("=" * 60)
print("Phase 3: Differential Expression Analysis")
print("=" * 60)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
print()
print("Loading data...")
expr  = pd.read_csv(f"{DATA_DIR}/expression_matrix.csv", index_col=0)
meta  = pd.read_csv(f"{DATA_DIR}/sample_metadata.csv")

label_map  = meta.set_index("sample_id")["label"]
hcm_cols   = meta[meta["label"] == "HCM"]["sample_id"].tolist()
ctrl_cols  = meta[meta["label"] == "Control"]["sample_id"].tolist()

hcm_cols  = [c for c in hcm_cols  if c in expr.columns]
ctrl_cols = [c for c in ctrl_cols if c in expr.columns]

print(f"  HCM samples:     {len(hcm_cols)}")
print(f"  Control samples: {len(ctrl_cols)}")

# -------------------------------------------------------
# Step 1: Compute fold change and t-test per probe
# -------------------------------------------------------
print()
print("Running differential expression tests (Welch t-test per probe)...")
print("  This may take 30-60 seconds for all probes...")

hcm_expr  = expr[hcm_cols]
ctrl_expr = expr[ctrl_cols]

# Log2 fold change: mean(HCM) - mean(Control) in log2 space
mean_hcm  = hcm_expr.mean(axis=1)
mean_ctrl = ctrl_expr.mean(axis=1)
log2fc    = mean_hcm - mean_ctrl

# Welch t-test per probe
t_stats = []
p_vals  = []
for probe in expr.index:
    hcm_vals  = hcm_expr.loc[probe].dropna().values
    ctrl_vals = ctrl_expr.loc[probe].dropna().values
    if len(hcm_vals) >= 3 and len(ctrl_vals) >= 3:
        t, p = stats.ttest_ind(hcm_vals, ctrl_vals, equal_var=False)
    else:
        t, p = np.nan, np.nan
    t_stats.append(t)
    p_vals.append(p)

results = pd.DataFrame({
    "probe_id":   expr.index,
    "mean_hcm":   mean_hcm.values,
    "mean_ctrl":  mean_ctrl.values,
    "log2fc":     log2fc.values,
    "t_stat":     t_stats,
    "p_value":    p_vals,
})

# Multiple testing correction: Benjamini-Hochberg FDR
valid_mask    = results["p_value"].notna()
_, p_adj, _, _ = multipletests(
    results.loc[valid_mask, "p_value"],
    alpha=0.05,
    method="fdr_bh"
)
results.loc[valid_mask, "p_adj"] = p_adj
results.loc[~valid_mask, "p_adj"] = np.nan

# Significance threshold
sig_threshold = 0.05
fc_threshold  = 1.0  # |log2FC| > 1 = 2-fold change

results["significant"] = (
    (results["p_adj"] < sig_threshold) &
    (results["log2fc"].abs() > fc_threshold)
)
results["direction"] = np.where(
    results["log2fc"] > 0, "Up in HCM",
    np.where(results["log2fc"] < 0, "Down in HCM", "NS")
)
results.loc[~results["significant"], "direction"] = "NS"

n_sig   = results["significant"].sum()
n_up    = ((results["direction"] == "Up in HCM") & results["significant"]).sum()
n_down  = ((results["direction"] == "Down in HCM") & results["significant"]).sum()

print(f"  Total probes tested:          {len(results):,}")
print(f"  Significant (FDR<0.05, |FC|>1): {n_sig:,}")
print(f"    Upregulated in HCM:         {n_up:,}")
print(f"    Downregulated in HCM:       {n_down:,}")

# -------------------------------------------------------
# Step 2: Save DEG results
# -------------------------------------------------------
results.sort_values("p_adj").to_csv(
    f"{DATA_DIR}/differential_expression.csv", index=False
)

sig_degs  = results[results["significant"]].sort_values("log2fc", ascending=False)
top_up    = sig_degs[sig_degs["direction"] == "Up in HCM"].head(20)
top_down  = sig_degs[sig_degs["direction"] == "Down in HCM"].tail(20)
top_degs  = pd.concat([top_up, top_down])
top_degs.to_csv(f"{DATA_DIR}/top_degs.csv", index=False)

print()
print("Top 10 upregulated probes in HCM:")
for _, row in top_up.head(10).iterrows():
    print(f"  {str(row['probe_id'])[:20]:<22}  log2FC={row['log2fc']:+.3f}  FDR={row['p_adj']:.2e}")

print()
print("Top 10 downregulated probes in HCM:")
for _, row in top_down.head(10).iterrows():
    print(f"  {str(row['probe_id'])[:20]:<22}  log2FC={row['log2fc']:+.3f}  FDR={row['p_adj']:.2e}")

# -------------------------------------------------------
# Step 3: Volcano plot
# -------------------------------------------------------
print()
print("Creating volcano plot...")

# Cap -log10(p_adj) for plotting
results["neg_log10_p"] = -np.log10(results["p_adj"].clip(lower=1e-300))

fig1, ax1 = plt.subplots(figsize=(13, 9), facecolor="#F5F3EF")
ax1.set_facecolor("white")

# Plot non-significant first (background)
ns = results[~results["significant"]]
ax1.scatter(
    ns["log2fc"], ns["neg_log10_p"],
    c=COLORS["ns"], alpha=0.25, s=8, rasterized=True, zorder=1
)

# Up in HCM
up = results[results["direction"] == "Up in HCM"]
ax1.scatter(
    up["log2fc"], up["neg_log10_p"],
    c=COLORS["up"], alpha=0.65, s=12, zorder=2, label=f"Up in HCM (n={len(up):,})"
)

# Down in HCM
dn = results[results["direction"] == "Down in HCM"]
ax1.scatter(
    dn["log2fc"], dn["neg_log10_p"],
    c=COLORS["down"], alpha=0.65, s=12, zorder=2, label=f"Down in HCM (n={len(dn):,})"
)

# Threshold lines
ax1.axhline(-np.log10(sig_threshold), color="#555", lw=1.2, ls="--", alpha=0.7)
ax1.axvline(-fc_threshold, color="#555", lw=1.2, ls="--", alpha=0.7)
ax1.axvline( fc_threshold, color="#555", lw=1.2, ls="--", alpha=0.7)

# Label known HCM genes if they appear in results
try:
    from adjustText import adjust_text
    has_adjust = True
except ImportError:
    has_adjust = False

texts = []
for _, row in results.iterrows():
    probe = str(row["probe_id"])
    gene  = probe.split("_")[0] if "_" in probe else probe
    if gene.upper() in HCM_KNOWN_GENES and row["significant"]:
        color = COLORS["up"] if row["direction"] == "Up in HCM" else COLORS["down"]
        ax1.scatter(row["log2fc"], row["neg_log10_p"],
                    c=color, s=40, zorder=5, edgecolors="white", linewidths=0.8)
        t = ax1.text(
            row["log2fc"], row["neg_log10_p"],
            gene.upper(),
            fontsize=7.5, fontweight="bold", color=color,
            ha="center", va="bottom"
        )
        texts.append(t)

if has_adjust and texts:
    adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle="-", color="#888", lw=0.7))

ax1.set_xlabel("Log2 Fold Change (HCM vs Control)", fontsize=12)
ax1.set_ylabel("-Log10 Adjusted P-value (FDR)", fontsize=12)
ax1.set_title(
    "Differential Expression: HCM vs Non-Failing Control\n"
    f"GSE36961 | FDR < 0.05 and |Log2FC| > 1.0 | {n_sig:,} significant probes",
    fontsize=12, fontweight="bold", pad=14
)
ax1.legend(fontsize=10)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/03_volcano_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/03_volcano_plot.png")

# -------------------------------------------------------
# Step 4: Heatmap of top 40 significant DEGs
# -------------------------------------------------------
print()
print("Creating DEG heatmap...")

top40 = pd.concat([
    sig_degs[sig_degs["direction"] == "Up in HCM"].head(20),
    sig_degs[sig_degs["direction"] == "Down in HCM"].tail(20)
])["probe_id"].tolist()

top40 = [p for p in top40 if p in expr.index]
top40 = list(dict.fromkeys(top40))  # deduplicate preserving order

col_order = [c for c in ctrl_cols + hcm_cols if c in expr.columns]
_raw40 = expr.loc[top40, col_order].values.astype(np.float64)
_z40   = np.apply_along_axis(zscore,  1, _raw40)
expr_top40 = pd.DataFrame(_z40, index=top40, columns=col_order)

fig2, ax2 = plt.subplots(figsize=(16, 11), facecolor="#F5F3EF")
ax2.set_facecolor("white")

heatmap_data = np.array(expr_top40.values, dtype=np.float64)
im2 = ax2.imshow(
    heatmap_data,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-3, vmax=3
)

n_ctrl_here = sum(1 for c in col_order if c in ctrl_cols)
ax2.axvline(n_ctrl_here - 0.5, color="white", lw=2, zorder=4)

ax2.set_yticks(range(len(top40)))
ax2.set_yticklabels(
    [str(p)[:22] for p in top40],
    fontsize=7
)
ax2.set_xticks([])
ax2.set_xlabel(
    f"Samples: Control (n={len(ctrl_cols)}) | HCM (n={len(hcm_cols)})",
    fontsize=10
)
ax2.set_title(
    f"Top 40 Differentially Expressed Probes\n"
    f"Z-score normalized | Top 20 Up + Top 20 Down in HCM",
    fontsize=11, fontweight="bold", pad=12
)

plt.colorbar(im2, ax=ax2, label="Z-score", fraction=0.015, pad=0.01)

legend_patches = [
    mpatches.Patch(color=COLORS["Control"], label="Control"),
    mpatches.Patch(color=COLORS["HCM"],     label="HCM"),
]
ax2.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

plt.savefig(f"{REPORTS_DIR}/03_deg_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {REPORTS_DIR}/03_deg_heatmap.png")

# -------------------------------------------------------
# Step 5: Write summary
# -------------------------------------------------------
with open(f"{REPORTS_DIR}/03_deg_summary.txt", "w") as f:
    f.write("Differential Expression Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset:          GSE36961\n")
    f.write(f"Comparison:       HCM (n={len(hcm_cols)}) vs Control (n={len(ctrl_cols)})\n")
    f.write(f"Method:           Welch t-test + BH FDR correction\n")
    f.write(f"Thresholds:       FDR < 0.05 AND |Log2FC| > 1.0\n\n")
    f.write(f"Total probes tested:     {len(results):,}\n")
    f.write(f"Significant probes:      {n_sig:,}\n")
    f.write(f"  Upregulated in HCM:  {n_up:,}\n")
    f.write(f"  Downregulated:       {n_down:,}\n\n")
    f.write("Top 10 upregulated probes:\n")
    for _, row in top_up.head(10).iterrows():
        f.write(f"  {str(row['probe_id']):<25}  "
                f"log2FC={row['log2fc']:+.3f}  FDR={row['p_adj']:.2e}\n")
    f.write("\nTop 10 downregulated probes:\n")
    for _, row in top_down.head(10).iterrows():
        f.write(f"  {str(row['probe_id']):<25}  "
                f"log2FC={row['log2fc']:+.3f}  FDR={row['p_adj']:.2e}\n")

print(f"  Saved: {REPORTS_DIR}/03_deg_summary.txt")
print(f"  Saved: {DATA_DIR}/differential_expression.csv")
print(f"  Saved: {DATA_DIR}/top_degs.csv")
print()
print("Phase 3 complete. Run 04_ml_classification.py next.")
