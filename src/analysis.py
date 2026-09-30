"""
analysis.py
-----------
Cardiac Gene Expression Exploratory Analysis
Hypertrophic Cardiomyopathy vs Non-Failing Controls

This script runs all four analytical phases in sequence:

  Phase 1  Data quality control and validation
  Phase 2  Exploratory data analysis and visualization
  Phase 3  Differential expression analysis with multiple testing correction
  Phase 4  Machine learning classification with SHAP interpretability

Data source modeled on:
  GSE36961 (NCBI GEO)
  Hebl VB, et al. Targeting HCM-Causing Sarcomere Mutations.
  Circulation: Cardiovascular Genetics, 2012.

Biological context:
  Hypertrophic Cardiomyopathy (HCM) is the most common inherited
  cardiovascular disorder, affecting 1 in 500 people. It is caused
  primarily by mutations in sarcomere genes, most commonly MYBPC3 and
  MYH7. The Tufts Institute for AI (TIAI) actively researches HCM
  at the single-cell and spatial transcriptomic level. This project
  analyzes bulk RNA-seq expression patterns as a foundation for
  understanding the transcriptional landscape of HCM.

  Published TIAI work on HCM:
    Ali SA, Perera G, Laird J, Batorsky R et al.
    Single Cell Transcriptomic Profiling of MYBPC3-Associated HCM
    Across Species. J Am Heart Assoc. 2025 Jan 7;14(1):e035780.

Usage:
  python src/analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
)
from statsmodels.stats.multitest import multipletests
import shap

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)


# -------------------------------------------------------
# Color palette (publication quality)
# -------------------------------------------------------
C_HCM     = "#C0392B"   # deep red for HCM
C_CTRL    = "#1A5276"   # deep blue for control
C_GOLD    = "#C9A84C"   # gold accent
C_LIGHT   = "#F4F6F7"   # background
C_DARK    = "#1C2833"   # text
PALETTE   = {"HCM": C_HCM, "Control": C_CTRL}


def load_data():
    """Load expression matrix and metadata. Run generate_data.py first."""
    if not os.path.exists("data/expression_matrix.csv"):
        print("Data not found. Running generate_data.py first...")
        import subprocess
        subprocess.run(["python3", "src/generate_data.py"], check=True)

    expr = pd.read_csv("data/expression_matrix.csv", index_col=0)
    meta = pd.read_csv("data/sample_metadata.csv")
    meta = meta.set_index("sample_id")

    # Align order
    meta = meta.loc[expr.index]
    return expr, meta


# -------------------------------------------------------
# Phase 1: Quality control
# -------------------------------------------------------

def phase1_qc(expr, meta):
    print("=" * 65)
    print("PHASE 1  DATA QUALITY CONTROL")
    print("=" * 65)

    # QC metrics per sample
    qc = pd.DataFrame(index=expr.index)
    qc["n_genes_detected"]   = (expr > 0).sum(axis=1)
    qc["mean_expression"]    = expr.mean(axis=1)
    qc["total_counts"]       = expr.sum(axis=1)
    qc["pct_low_expr_genes"] = ((expr > 0) & (expr < 1)).sum(axis=1) / expr.shape[1] * 100
    qc["condition"]          = meta["condition"]

    print(f"\n  Samples:    {len(expr):,}")
    print(f"  Genes:      {expr.shape[1]:,}")
    print(f"  HCM:        {(qc.condition == 'HCM').sum():,}")
    print(f"  Controls:   {(qc.condition == 'Control').sum():,}")
    print(f"  Nulls:      {expr.isnull().sum().sum()}")

    print("\n  QC summary by condition:")
    summary = qc.groupby("condition")[["n_genes_detected","mean_expression","total_counts"]].mean().round(2)
    print(summary.to_string())

    # Flag potential outliers (>3 SD from group mean on total counts)
    outliers = []
    for cond in ["HCM", "Control"]:
        grp = qc[qc.condition == cond]["total_counts"]
        z = np.abs((grp - grp.mean()) / grp.std())
        flagged = z[z > 3].index.tolist()
        outliers.extend(flagged)

    print(f"\n  Outlier samples (>3 SD total counts): {len(outliers)}")
    for s in outliers:
        print(f"    {s}")

    # QC figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=C_LIGHT)
    fig.suptitle(
        "Phase 1: Sample-Level Quality Control\nGSE36961 (HCM vs Non-Failing Controls)",
        fontsize=12, fontweight="bold", color=C_DARK,
    )

    for ax, metric, label in zip(
        axes,
        ["n_genes_detected", "mean_expression", "total_counts"],
        ["Genes Detected per Sample", "Mean Log2 Expression", "Total Expression Count"],
    ):
        for cond, color in [("HCM", C_HCM), ("Control", C_CTRL)]:
            vals = qc[qc.condition == cond][metric]
            ax.hist(vals, bins=20, alpha=0.65, color=color, label=cond, edgecolor="white", lw=0.5)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Number of Samples", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("reports/01_quality_control.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: reports/01_quality_control.png")

    return qc


# -------------------------------------------------------
# Phase 2: Exploratory analysis
# -------------------------------------------------------

def phase2_eda(expr, meta):
    print()
    print("=" * 65)
    print("PHASE 2  EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    labels = meta["condition"]

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expr.values)
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    print(f"\n  PCA explained variance:")
    print(f"    PC1: {explained[0]:.1f}%")
    print(f"    PC2: {explained[1]:.1f}%")
    print(f"    PC3: {explained[2]:.1f}%")
    print(f"    Top 3 combined: {explained[:3].sum():.1f}%")

    # Top variable genes
    gene_var = expr.var(axis=0).sort_values(ascending=False)
    top_var_genes = gene_var.head(30).index.tolist()
    print(f"\n  Top 5 most variable genes:")
    for g in top_var_genes[:5]:
        print(f"    {g}: variance {gene_var[g]:.3f}")

    # Figure: PCA + variance explained + heatmap
    fig = plt.figure(figsize=(18, 12), facecolor=C_LIGHT)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                             left=0.06, right=0.97, top=0.90, bottom=0.08)
    fig.suptitle(
        "Phase 2: Exploratory Data Analysis\nPCA, Variance Explained, and Top Variable Gene Expression",
        fontsize=12, fontweight="bold", color=C_DARK,
    )

    # PCA scatter
    ax1 = fig.add_subplot(gs[0, 0:2])
    for cond, color in [("HCM", C_HCM), ("Control", C_CTRL)]:
        mask = labels == cond
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color, label=cond, alpha=0.65, s=45, edgecolors="white", lw=0.3,
        )
    ax1.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=10)
    ax1.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontsize=10)
    ax1.set_title("PCA: Sample Separation by Condition", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor("white")

    # Scree plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(range(1, 11), explained, color=C_GOLD, alpha=0.85, edgecolor="white")
    ax2.plot(range(1, 11), np.cumsum(explained), "o-", color=C_HCM, lw=2, ms=5, label="Cumulative")
    ax2.set_xlabel("Principal Component", fontsize=10)
    ax2.set_ylabel("Variance Explained (%)", fontsize=10)
    ax2.set_title("Scree Plot", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor("white")

    # Heatmap of top variable genes
    ax3 = fig.add_subplot(gs[1, :])
    top15 = top_var_genes[:15]
    heatmap_data = expr[top15]

    # Sort by condition for cleaner visualization
    sort_idx = labels.sort_values().index
    heatmap_sorted = heatmap_data.loc[sort_idx].T
    cond_sorted    = labels.loc[sort_idx]

    # Add condition color bar above heatmap
    cond_colors = cond_sorted.map({"HCM": C_HCM, "Control": C_CTRL})
    col_colors  = pd.Series(cond_colors.values, index=heatmap_sorted.columns)

    import matplotlib.patches as mpatches
    im = ax3.imshow(
        heatmap_sorted.values,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )
    ax3.set_yticks(range(len(top15)))
    ax3.set_yticklabels(top15, fontsize=8.5)
    ax3.set_xticks([])
    ax3.set_xlabel("Samples (sorted by condition)", fontsize=10)
    ax3.set_title("Top 15 Most Variable Genes  |  Blue = Control   Red = HCM (approx)", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax3, shrink=0.5, label="Log2 Expression")

    plt.savefig("reports/02_exploratory_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: reports/02_exploratory_analysis.png")

    return X_pca, explained, top_var_genes


# -------------------------------------------------------
# Phase 3: Differential expression
# -------------------------------------------------------

def phase3_differential_expression(expr, meta):
    print()
    print("=" * 65)
    print("PHASE 3  DIFFERENTIAL EXPRESSION ANALYSIS")
    print("=" * 65)

    hcm_samples  = meta[meta["condition"] == "HCM"].index
    ctrl_samples = meta[meta["condition"] == "Control"].index

    results = []
    for gene in expr.columns:
        hcm_vals  = expr.loc[hcm_samples,  gene].values
        ctrl_vals = expr.loc[ctrl_samples, gene].values

        t_stat, p_val = ttest_ind(hcm_vals, ctrl_vals, equal_var=False)
        log2fc = hcm_vals.mean() - ctrl_vals.mean()
        mean_hcm  = hcm_vals.mean()
        mean_ctrl = ctrl_vals.mean()

        results.append({
            "gene":       gene,
            "log2fc":     round(log2fc, 4),
            "mean_hcm":   round(mean_hcm, 4),
            "mean_ctrl":  round(mean_ctrl, 4),
            "t_stat":     round(t_stat, 4),
            "p_value":    p_val,
        })

    de = pd.DataFrame(results)

    # Multiple testing correction (Benjamini Hochberg)
    reject, p_adj, _, _ = multipletests(de["p_value"], method="fdr_bh")
    de["p_adj"]       = p_adj.round(6)
    de["significant"] = reject

    de = de.sort_values("p_adj")
    de.to_csv("reports/differential_expression_results.csv", index=False)

    sig = de[de["significant"]]
    up  = sig[sig["log2fc"] > 0].sort_values("log2fc", ascending=False)
    dn  = sig[sig["log2fc"] < 0].sort_values("log2fc")

    print(f"\n  Total genes tested:            {len(de):,}")
    print(f"  Significant (FDR < 0.05):      {len(sig):,}")
    print(f"    Upregulated in HCM:          {len(up):,}")
    print(f"    Downregulated in HCM:        {len(dn):,}")
    print()
    print("  Top 5 upregulated in HCM:")
    for _, row in up.head(5).iterrows():
        print(f"    {row.gene:<16} log2FC={row.log2fc:+.3f}  padj={row.p_adj:.2e}")
    print()
    print("  Top 5 downregulated in HCM:")
    for _, row in dn.head(5).iterrows():
        print(f"    {row.gene:<16} log2FC={row.log2fc:+.3f}  padj={row.p_adj:.2e}")

    # Known HCM genes check
    known = ["MYBPC3", "MYH7", "TNNT2", "TNNI3", "SERCA2A"]
    print()
    print("  Known HCM gene results:")
    for g in known:
        row = de[de.gene == g]
        if not row.empty:
            r = row.iloc[0]
            sig_label = "significant" if r.significant else "not significant"
            print(f"    {g:<12} log2FC={r.log2fc:+.3f}  padj={r.p_adj:.2e}  ({sig_label})")

    # Figure: Volcano plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_LIGHT)
    fig.suptitle(
        "Phase 3: Differential Expression Analysis\nHCM vs Non-Failing Controls (Welch t-test, Benjamini-Hochberg FDR)",
        fontsize=12, fontweight="bold", color=C_DARK,
    )

    # Volcano
    ax_v = axes[0]
    neg_log_p = -np.log10(de["p_adj"].clip(lower=1e-30))

    colors_v = de.apply(lambda r:
        C_HCM   if (r.significant and r.log2fc > 0) else
        C_CTRL  if (r.significant and r.log2fc < 0) else
        "#BDC3C7", axis=1
    )

    ax_v.scatter(de["log2fc"], neg_log_p, c=colors_v, alpha=0.7, s=30, edgecolors="none")
    ax_v.axhline(-np.log10(0.05), color="gray", ls="--", lw=1, alpha=0.6, label="FDR 0.05")
    ax_v.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.4)

    # Label top genes
    label_genes = up.head(6)["gene"].tolist() + dn.head(4)["gene"].tolist()
    for g in label_genes:
        row = de[de.gene == g].iloc[0]
        ax_v.annotate(
            g,
            xy=(row.log2fc, -np.log10(max(row.p_adj, 1e-30))),
            fontsize=7.5, fontweight="bold", color=C_DARK,
            xytext=(4, 4), textcoords="offset points",
        )

    ax_v.set_xlabel("Log2 Fold Change (HCM vs Control)", fontsize=10)
    ax_v.set_ylabel("-Log10 Adjusted p-value", fontsize=10)
    ax_v.set_title("Volcano Plot", fontsize=11, fontweight="bold")
    ax_v.legend(fontsize=9)
    ax_v.grid(True, alpha=0.2)
    ax_v.set_facecolor("white")

    # Bar chart: top DE genes by log2FC
    ax_b = axes[1]
    top_show = pd.concat([up.head(8), dn.head(8)]).sort_values("log2fc")
    bar_colors = [C_HCM if fc > 0 else C_CTRL for fc in top_show["log2fc"]]
    bars = ax_b.barh(top_show["gene"], top_show["log2fc"], color=bar_colors, alpha=0.85)
    ax_b.axvline(0, color=C_DARK, lw=1)
    ax_b.set_xlabel("Log2 Fold Change (HCM vs Control)", fontsize=10)
    ax_b.set_title("Top Differentially Expressed Genes", fontsize=11, fontweight="bold")
    ax_b.grid(True, alpha=0.2, axis="x")
    ax_b.set_facecolor("white")

    import matplotlib.patches as mpatches
    ax_b.legend(
        handles=[
            mpatches.Patch(color=C_HCM,  label="Higher in HCM"),
            mpatches.Patch(color=C_CTRL, label="Higher in Control"),
        ], fontsize=9, loc="lower right",
    )

    plt.tight_layout()
    plt.savefig("reports/03_differential_expression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: reports/03_differential_expression.png")
    print("  Saved: reports/differential_expression_results.csv")

    return de


# -------------------------------------------------------
# Phase 4: Machine learning classification
# -------------------------------------------------------

def phase4_ml_classification(expr, meta, de):
    print()
    print("=" * 65)
    print("PHASE 4  MACHINE LEARNING CLASSIFICATION WITH SHAP")
    print("=" * 65)

    # Use top 40 differentially expressed genes as features
    top_genes = de[de["significant"]].head(40)["gene"].tolist()
    if len(top_genes) < 10:
        top_genes = de.head(40)["gene"].tolist()

    X = expr[top_genes].values
    y = (meta["condition"] == "HCM").astype(int).values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    print(f"\n  Feature set: {len(top_genes)} differentially expressed genes")
    print(f"  Samples:     {len(y):,}  (HCM={y.sum():,}, Control={(1-y).sum():,})")

    # Cross-validated AUC: Logistic Regression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    lr_aucs = cross_val_score(lr, X_sc, y, cv=cv, scoring="roc_auc")

    # Cross-validated AUC: Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf_aucs = cross_val_score(rf, X_sc, y, cv=cv, scoring="roc_auc")

    print()
    print("  5-fold cross-validated AUC:")
    print(f"    Logistic Regression:  {lr_aucs.mean():.4f} (+/- {lr_aucs.std():.4f})")
    print(f"    Random Forest:        {rf_aucs.mean():.4f} (+/- {rf_aucs.std():.4f})")

    # Fit full model for SHAP
    rf.fit(X_sc, y)
    lr.fit(X_sc, y)

    # SHAP values (Random Forest)
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sc)

    # shap_values may be list (one per class) or 3D array
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class 1 = HCM
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_shap = np.abs(sv).mean(axis=0)
    shap_df   = pd.DataFrame({
        "gene":       top_genes,
        "mean_shap":  mean_shap,
    }).sort_values("mean_shap", ascending=False)

    print()
    print("  Top 10 most important genes (SHAP, Random Forest):")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {row.gene:<16} SHAP={row.mean_shap:.4f}")

    # ROC curve (full dataset fit)
    y_prob_lr = lr.predict_proba(X_sc)[:, 1]
    y_prob_rf = rf.predict_proba(X_sc)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y, y_prob_rf)
    auc_lr = roc_auc_score(y, y_prob_lr)
    auc_rf = roc_auc_score(y, y_prob_rf)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=C_LIGHT)
    fig.suptitle(
        "Phase 4: Machine Learning Classification and SHAP Interpretability\n"
        "Predicting HCM vs Control from Gene Expression Features",
        fontsize=12, fontweight="bold", color=C_DARK,
    )

    # ROC curves
    ax_r = axes[0]
    ax_r.plot(fpr_lr, tpr_lr, color=C_CTRL, lw=2,
              label=f"Logistic Regression (AUC={auc_lr:.3f})")
    ax_r.plot(fpr_rf, tpr_rf, color=C_HCM, lw=2,
              label=f"Random Forest (AUC={auc_rf:.3f})")
    ax_r.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    ax_r.set_xlabel("False Positive Rate", fontsize=10)
    ax_r.set_ylabel("True Positive Rate", fontsize=10)
    ax_r.set_title("ROC Curves", fontsize=11, fontweight="bold")
    ax_r.legend(fontsize=8.5)
    ax_r.grid(True, alpha=0.2)
    ax_r.set_facecolor("white")

    # SHAP bar chart
    ax_s = axes[1]
    top_shap = shap_df.head(12)
    ax_s.barh(
        top_shap["gene"][::-1],
        top_shap["mean_shap"][::-1],
        color=C_GOLD, alpha=0.85, edgecolor="white",
    )
    ax_s.set_xlabel("Mean |SHAP Value| (feature importance)", fontsize=10)
    ax_s.set_title("Top Genes by SHAP Importance\n(Random Forest)", fontsize=11, fontweight="bold")
    ax_s.grid(True, alpha=0.2, axis="x")
    ax_s.set_facecolor("white")

    # CV AUC comparison
    ax_c = axes[2]
    bp_data = [lr_aucs, rf_aucs]
    bp = ax_c.boxplot(
        bp_data,
        labels=["Logistic\nRegression", "Random\nForest"],
        patch_artist=True,
        widths=0.4,
    )
    colors_bp = [C_CTRL, C_HCM]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax_c.set_ylabel("Cross-Validated AUC (5-fold)", fontsize=10)
    ax_c.set_title("Model Comparison\n5-Fold Cross-Validation", fontsize=11, fontweight="bold")
    ax_c.set_ylim(0.5, 1.05)
    ax_c.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="Random")
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.2, axis="y")
    ax_c.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("reports/04_ml_classification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: reports/04_ml_classification.png")

    # Save SHAP ranking
    shap_df.to_csv("reports/shap_gene_importance.csv", index=False)
    print("  Saved: reports/shap_gene_importance.csv")

    return rf, shap_df, lr_aucs, rf_aucs


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    print()
    print("=" * 65)
    print("Cardiac Gene Expression EDA")
    print("Hypertrophic Cardiomyopathy vs Non-Failing Controls")
    print("Modeled on GSE36961 (NCBI GEO)")
    print("=" * 65)

    expr, meta = load_data()

    qc           = phase1_qc(expr, meta)
    pca_out      = phase2_eda(expr, meta)
    de           = phase3_differential_expression(expr, meta)
    ml_out       = phase4_ml_classification(expr, meta, de)

    print()
    print("=" * 65)
    print("Pipeline complete")
    print("=" * 65)
    print("Output files:")
    for f in sorted(os.listdir("reports")):
        size = os.path.getsize(f"reports/{f}")
        print(f"  reports/{f:<50}  {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
