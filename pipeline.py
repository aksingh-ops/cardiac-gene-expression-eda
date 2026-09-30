"""
pipeline.py
===========
Run the complete cardiac gene expression analysis pipeline.

Phases:
    01  Data acquisition from NCBI GEO and quality control
    02  Exploratory analysis: PCA, heatmaps, distributions
    03  Differential expression: Welch t-test with BH FDR correction
    04  ML classification: Random Forest and Logistic Regression

Usage:
    python pipeline.py

All outputs are saved to data/ and reports/
"""

import subprocess
import sys
import time

phases = [
    ("src/01_download_and_qc.py",       "Phase 1: Data Acquisition and QC"),
    ("src/02_exploratory_analysis.py",  "Phase 2: Exploratory Analysis"),
    ("src/03_differential_expression.py", "Phase 3: Differential Expression"),
    ("src/04_ml_classification.py",     "Phase 4: ML Classification"),
]

print()
print("=" * 65)
print("Cardiac Gene Expression Analysis Pipeline")
print("Dataset: GSE36961 -- Hypertrophic Cardiomyopathy")
print("NCBI GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36961")
print("=" * 65)
print()

for script, label in phases:
    print(f"Running {label}...")
    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"ERROR in {script}. Check output above.")
        sys.exit(1)
    print(f"Completed in {elapsed:.0f}s")
    print()

print("=" * 65)
print("Pipeline complete. Outputs:")
print()

import os
for folder in ["data", "reports"]:
    for f in sorted(os.listdir(folder)):
        size = os.path.getsize(f"{folder}/{f}")
        print(f"  {folder}/{f:<45} {size/1024:.1f} KB")

print()
print("Key files:")
print("  reports/02_pca.png                 -- Disease separation visualization")
print("  reports/03_volcano_plot.png        -- DEG significance vs fold change")
print("  reports/04_roc_curves.png          -- ML classification performance")
print("  data/differential_expression.csv  -- Full DEG results table")
