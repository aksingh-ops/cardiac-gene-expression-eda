"""
01_download_and_qc.py
=====================
Phase 1: Data acquisition and quality control

Dataset: GSE36961 from NCBI Gene Expression Omnibus
  Bos JM et al. Mayo Clin Proc. 2020.
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36961

  106 HCM patients, 39 non-failing controls
  Platform: GPL15389 (Illumina HumanHT-12 V3.0 expression beadchip)
  Left ventricular cardiac tissue samples

When NCBI GEO is not accessible, a synthetic dataset mirroring
the real GSE36961 structure and biological signal is used instead.
See src/generate_synthetic_data.py for details.

Outputs:
  data/expression_matrix.csv
  data/sample_metadata.csv
  data/qc_metrics.csv
  reports/01_sample_distribution.png
  reports/01_library_size.png
  reports/01_qc_summary.txt
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")

DATA_DIR    = "data"
REPORTS_DIR = "reports"
GEO_ID      = "GSE36961"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

COLORS = {"HCM": "#C0392B", "Control": "#1A5276", "accent": "#C9A84C"}

print("=" * 60)
print("Phase 1: Data Acquisition and Quality Control")
print(f"Dataset: {GEO_ID} -- Hypertrophic Cardiomyopathy (HCM)")
print("=" * 60)
print()

# -------------------------------------------------------
# Step 1: Load data (GEO download or synthetic fallback)
# -------------------------------------------------------
try:
    import GEOparse
    print("Attempting download from NCBI GEO...")
    gse = GEOparse.get_GEO(geo=GEO_ID, destdir=DATA_DIR, silent=True)
    print(f"Download complete. Samples: {len(gse.gsms)}")

    # Extract expression matrix
    expr_data = {}
    for gsm_name, gsm in gse.gsms.items():
        tbl = gsm.table
        if tbl is not None and not tbl.empty and "ID_REF" in tbl.columns:
            expr_data[gsm_name] = pd.to_numeric(
                tbl.set_index("ID_REF")["VALUE"], errors="coerce"
            )
    expr_matrix = pd.DataFrame(expr_data)

    # Extract labels
    meta_rows = []
    for gsm_name, gsm in gse.gsms.items():
        title  = gsm.metadata.get("title", [""])[0].lower()
        source = gsm.metadata.get("source_name_ch1", [""])[0].lower()
        if any(k in title or k in source for k in ["hcm","hypertrophic","obstructive"]):
            label = "HCM"
        elif any(k in title or k in source for k in ["control","normal","donor","non-failing"]):
            label = "Control"
        else:
            label = "Unknown"
        meta_rows.append({"sample_id": gsm_name, "label": label,
                           "title": gsm.metadata.get("title",[""])[0],
                           "source": gsm.metadata.get("source_name_ch1",[""])[0]})
    metadata = pd.DataFrame(meta_rows)
    metadata = metadata[metadata["label"].isin(["HCM","Control"])].reset_index(drop=True)

    # Filter and normalize
    missing_pct  = expr_matrix.isna().mean(axis=1)
    expr_matrix  = expr_matrix[missing_pct <= 0.20]
    expr_matrix  = expr_matrix.apply(lambda r: r.fillna(r.mean()), axis=1)
    if expr_matrix.max().max() > 30:
        expr_matrix = np.log2(expr_matrix + 1)

    USE_SYNTHETIC = False

except Exception as e:
    print(f"GEO download not available: {type(e).__name__}")
    print("Using synthetic dataset that mirrors GSE36961 structure.")
    print("(Set up with real GEO data by running with network access to ncbi.nlm.nih.gov)")
    print()
    from generate_synthetic_data import generate_synthetic_gse36961
    expr_matrix, metadata = generate_synthetic_gse36961()
    USE_SYNTHETIC = True

print()
print(f"Dataset loaded ({'synthetic' if USE_SYNTHETIC else 'real GEO'}):")
print(f"  HCM samples:     {(metadata['label']=='HCM').sum()}")
print(f"  Control samples: {(metadata['label']=='Control').sum()}")
print(f"  Probes:          {expr_matrix.shape[0]:,}")
val_min = float(expr_matrix.min().min())
val_max = float(expr_matrix.max().max())
print(f"  Value range:     {val_min:.2f} to {val_max:.2f} (log2)")

# -------------------------------------------------------
# Step 2: QC metrics per sample
# -------------------------------------------------------
print()
print("Computing per-sample QC metrics...")

labeled = metadata["sample_id"].tolist()
expr_matrix = expr_matrix[[c for c in expr_matrix.columns if c in labeled]]

qc_rows = []
for col in expr_matrix.columns:
    vals = expr_matrix[col].dropna()
    qc_rows.append({
        "sample_id":   col,
        "mean_expr":   float(vals.mean()),
        "median_expr": float(vals.median()),
        "std_expr":    float(vals.std()),
        "n_detected":  int((vals > 5).sum()),
        "pct_low":     float((vals < 5).mean() * 100),
    })
qc_df = pd.DataFrame(qc_rows)
cohort_mean = qc_df["mean_expr"].mean()
cohort_std  = qc_df["mean_expr"].std()
qc_df["outlier"] = np.abs(qc_df["mean_expr"] - cohort_mean) > 3 * cohort_std
n_outliers = qc_df["outlier"].sum()
print(f"  Outlier samples (3 SD from mean): {n_outliers}")

# -------------------------------------------------------
# Step 3: Save outputs
# -------------------------------------------------------
expr_matrix.to_csv(f"{DATA_DIR}/expression_matrix.csv")
metadata.to_csv(f"{DATA_DIR}/sample_metadata.csv", index=False)
qc_df.to_csv(f"{DATA_DIR}/qc_metrics.csv", index=False)

# -------------------------------------------------------
# Step 4: QC plots
# -------------------------------------------------------
# Plot 1: Sample distribution
fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor="#F5F3EF")
ax1.set_facecolor("white")
counts = metadata["label"].value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[COLORS["HCM"], COLORS["Control"]],
               alpha=0.88, width=0.5, edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha="center", fontsize=13, fontweight="bold")
ax1.set_title("GSE36961 Sample Distribution\nHCM vs Non-Failing Controls",
              fontsize=12, fontweight="bold", pad=12)
ax1.set_ylabel("Number of Samples", fontsize=11)
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/01_sample_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Per-sample mean expression
meta_qc = metadata.merge(qc_df, on="sample_id")
meta_qc = meta_qc.sort_values(["label","mean_expr"])
meta_qc["x_pos"] = range(len(meta_qc))
label_map = meta_qc.set_index("sample_id")["label"]

fig2, ax2 = plt.subplots(figsize=(14, 5), facecolor="#F5F3EF")
ax2.set_facecolor("white")
for label, color in [("HCM", COLORS["HCM"]), ("Control", COLORS["Control"])]:
    sub = meta_qc[meta_qc["label"] == label]
    ax2.bar(sub["x_pos"], sub["mean_expr"], color=color, alpha=0.80,
            label=f"{label} (n={len(sub)})", width=0.8)
ax2.axhline(cohort_mean, color=COLORS["accent"], lw=1.8, ls="--",
            label=f"Cohort mean ({cohort_mean:.2f})")
ax2.set_title("Per-Sample Mean Expression (QC)\nGSE36961 Sorted by Group",
              fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Sample Index", fontsize=10)
ax2.set_ylabel("Mean Log2 Expression", fontsize=10)
ax2.legend(fontsize=9)
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/01_library_size.png", dpi=150, bbox_inches="tight")
plt.close()

# QC report
with open(f"{REPORTS_DIR}/01_qc_summary.txt", "w") as f:
    f.write(f"GSE36961 Quality Control Report\n{'='*50}\n\n")
    f.write(f"Data source:     {'Synthetic (GEO-mirrored)' if USE_SYNTHETIC else 'Real NCBI GEO'}\n")
    f.write(f"Dataset:         {GEO_ID}\n")
    f.write(f"HCM samples:     {(metadata['label']=='HCM').sum()}\n")
    f.write(f"Control samples: {(metadata['label']=='Control').sum()}\n")
    f.write(f"Probes:          {expr_matrix.shape[0]:,}\n")
    f.write(f"Value range:     {val_min:.2f} to {val_max:.2f} (log2)\n")
    f.write(f"Outlier samples: {n_outliers}\n\n")
    f.write("Per-group summary:\n")
    for lbl in ["HCM", "Control"]:
        samps = meta_qc[meta_qc["label"] == lbl]
        f.write(f"  {lbl}: mean={samps['mean_expr'].mean():.4f}  "
                f"std={samps['mean_expr'].std():.4f}  "
                f"detected={samps['n_detected'].mean():.0f}\n")

print(f"Saved: {DATA_DIR}/expression_matrix.csv")
print(f"Saved: {DATA_DIR}/sample_metadata.csv")
print(f"Saved: {DATA_DIR}/qc_metrics.csv")
print(f"Saved: {REPORTS_DIR}/01_sample_distribution.png")
print(f"Saved: {REPORTS_DIR}/01_library_size.png")
print(f"Saved: {REPORTS_DIR}/01_qc_summary.txt")
print()
print("Phase 1 complete. Run 02_exploratory_analysis.py next.")
