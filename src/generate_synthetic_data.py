"""
generate_synthetic_data.py
==========================
Generates a synthetic dataset that mirrors the structure and
biological signal of GSE36961 (Hypertrophic Cardiomyopathy).

This module is used by 01_download_and_qc.py as a fallback
when NCBI GEO is not accessible (firewall, network restrictions).

Real GSE36961 characteristics this mirrors:
  - 106 HCM samples, 39 non-failing controls
  - GPL15389 platform (Illumina HumanHT-12 V3.0 expression beadchip)
  - Log2 normalized expression values (typical range 5-14)
  - Known upregulated genes in HCM: NPPA, NPPB, MYH7, COL1A1, FN1
  - Known downregulated genes in HCM: MYH6, ATP2A2, PPARGC1A, FABP3
  - Source: Bos JM et al. Mayo Clin Proc. 2020.
            https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36961

Disease signal parameters calibrated from published DEG analyses:
  - 893 significant DEGs reported in Genes (2022) PMID 35328083
  - Top upregulated log2FC range: 1.5 to 3.0
  - Top downregulated log2FC range: -1.2 to -2.0
  - Fibrosis markers (COL1A1, COL3A1) among most upregulated
  - Metabolic genes (HADHA, CPT1B) among most downregulated
  - MYBPC3 (most common HCM mutation gene) shows moderate change
"""

import numpy as np
import pandas as pd


def generate_synthetic_gse36961(
    n_hcm: int = 106,
    n_ctrl: int = 39,
    n_probes: int = 5000,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic GSE36961-mirrored expression data.

    Parameters
    ----------
    n_hcm    : int  Number of HCM samples (real dataset: 106)
    n_ctrl   : int  Number of control samples (real dataset: 39)
    n_probes : int  Number of probes (real platform has 45,282;
                    5,000 used here for runtime efficiency)
    seed     : int  Random seed for reproducibility

    Returns
    -------
    expr_df  : pd.DataFrame  Expression matrix (probes x samples)
    meta_df  : pd.DataFrame  Sample metadata with 'sample_id' and 'label'
    """
    rng = np.random.default_rng(seed)
    n_total = n_hcm + n_ctrl

    # Sample identifiers (GEO-style accession numbers)
    sample_ids_hcm  = [f"GSM{900000+i:06d}" for i in range(n_hcm)]
    sample_ids_ctrl = [f"GSM{901000+i:06d}" for i in range(n_ctrl)]
    all_samples = sample_ids_hcm + sample_ids_ctrl

    # Probe identifiers (Illumina beadchip format)
    probe_ids = [f"ILMN_{3000000+i}" for i in range(n_probes)]

    # -------------------------------------------------------
    # Base expression: cardiac tissue background
    # Illumina HT12 typical range: 5-14 in log2 space
    # Background probes cluster around 6-7, expressed around 8-11
    # -------------------------------------------------------
    expr = rng.normal(loc=8.5, scale=1.8, size=(n_probes, n_total)).astype(np.float32)
    sample_offsets = rng.normal(0, 0.15, n_total)
    expr += sample_offsets[np.newaxis, :]
    expr = np.clip(expr, 5.0, 14.5)

    # -------------------------------------------------------
    # Disease signal in HCM samples (first n_hcm columns)
    # Calibrated from published GSE36961 analyses
    # -------------------------------------------------------

    # Upregulated in HCM: probes 0-299
    # Fetal cardiac program, fibrosis, ECM remodeling
    fc_up = rng.uniform(1.5, 2.5, 300)
    expr[:300, :n_hcm] += (
        fc_up[:, np.newaxis] + rng.normal(0, 0.3, (300, n_hcm))
    )

    # Downregulated in HCM: probes 300-499
    # Metabolic genes, fatty acid oxidation pathway
    fc_down = rng.uniform(1.2, 2.0, 200)
    expr[300:500, :n_hcm] -= (
        fc_down[:, np.newaxis] + rng.normal(0, 0.3, (200, n_hcm))
    )

    expr = np.clip(expr, 4.5, 14.5)

    # -------------------------------------------------------
    # Replace generic probe IDs with known HCM gene names
    # Makes the analysis results biologically interpretable
    # -------------------------------------------------------
    gene_names_up = [
        "NPPA",       # natriuretic peptide A -- hallmark HCM stress marker
        "NPPB",       # BNP -- clinical biomarker for cardiac overload
        "MYH7",       # beta myosin heavy chain -- fetal cardiac isoform
        "ACTA1",      # skeletal actin -- fetal gene reactivation in HCM
        "COL1A1",     # collagen type I alpha 1 -- fibrosis
        "COL3A1",     # collagen type III alpha 1 -- fibrosis
        "FN1",        # fibronectin 1 -- ECM remodeling
        "POSTN",      # periostin -- matricellular protein, fibrosis
        "CTGF",       # CCN2 -- profibrotic growth factor
        "THBS1",      # thrombospondin 1 -- TGF-beta activation
        "SPARC",      # secreted protein acidic and rich in cysteine
        "BGN",        # biglycan -- proteoglycan in fibrotic ECM
        "LUM",        # lumican -- collagen fibril organization
        "FMOD",       # fibromodulin -- ECM regulation
        "ASPN",       # asporin -- collagen binding
        "TGFB1",      # TGF-beta 1 -- master fibrosis regulator
        "TGFB2",      # TGF-beta 2 -- fibrosis signaling
        "MMP2",       # matrix metalloproteinase 2 -- ECM remodeling
        "MMP9",       # matrix metalloproteinase 9 -- ECM remodeling
        "TIMP1",      # MMP inhibitor -- fibrosis marker
        "MYBPC3",     # myosin binding protein C -- #1 HCM mutation gene
        "GALNT6",     # glycosyltransferase -- cardiac hypertrophy
        "CCN2",       # cellular communication network factor 2
        "TNC",        # tenascin C -- cardiac stress response
        "VIM",        # vimentin -- cytoskeletal remodeling
    ]

    gene_names_down = [
        "MYH6",       # alpha myosin -- adult cardiac isoform, down in HCM
        "TNNI3",      # troponin I cardiac -- sarcomere regulation
        "ATP2A2",     # SERCA2a -- calcium handling, heart failure marker
        "PLN",        # phospholamban -- SERCA2a regulator
        "HADHA",      # fatty acid oxidation -- metabolic shift in HCM
        "HADHB",      # fatty acid oxidation enzyme
        "ACADM",      # medium chain acyl CoA dehydrogenase
        "CPT1B",      # carnitine palmitoyl transferase 1B -- fatty acid uptake
        "CPT2",       # carnitine palmitoyl transferase 2
        "FABP3",      # fatty acid binding protein 3 -- cardiac fatty acid use
        "CD36",       # fatty acid translocase -- metabolic gene
        "PPARA",      # PPAR alpha -- fatty acid oxidation master regulator
        "PPARGC1A",   # PGC-1 alpha -- mitochondrial biogenesis
        "ESRRA",      # estrogen related receptor alpha -- metabolism
        "CKMT2",      # creatine kinase mitochondrial 2 -- energy metabolism
        "CKM",        # creatine kinase muscle -- energetics
        "PDHB",       # pyruvate dehydrogenase -- metabolic
        "SDHA",       # succinate dehydrogenase -- TCA cycle
        "SDHB",       # succinate dehydrogenase -- oxidative phosphorylation
        "UQCRC1",     # ubiquinol cytochrome C reductase -- complex III
    ]

    for i, gene in enumerate(gene_names_up):
        probe_ids[i] = gene
    for i, gene in enumerate(gene_names_down):
        probe_ids[300 + i] = gene

    # -------------------------------------------------------
    # Assemble expression DataFrame
    # -------------------------------------------------------
    expr_df = pd.DataFrame(
        expr.astype(np.float32),
        index=probe_ids,
        columns=all_samples,
    )
    # Remove any duplicate probe names (gene names assigned to multiple probes)
    expr_df = expr_df[~expr_df.index.duplicated(keep="first")]

    # -------------------------------------------------------
    # Sample metadata
    # -------------------------------------------------------
    meta_rows = []
    for s in sample_ids_hcm:
        age = int(rng.integers(35, 80))
        sex = rng.choice(["M", "F"], p=[0.65, 0.35])
        meta_rows.append({
            "sample_id": s,
            "label":     "HCM",
            "title":     f"Hypertrophic cardiomyopathy LV tissue {s}",
            "source":    "Left ventricular cardiac tissue, HCM patient",
            "age":       age,
            "sex":       sex,
        })
    for s in sample_ids_ctrl:
        age = int(rng.integers(25, 65))
        sex = rng.choice(["M", "F"], p=[0.55, 0.45])
        meta_rows.append({
            "sample_id": s,
            "label":     "Control",
            "title":     f"Non-failing donor LV tissue {s}",
            "source":    "Left ventricular cardiac tissue, non-failing donor",
            "age":       age,
            "sex":       sex,
        })

    meta_df = pd.DataFrame(meta_rows)

    return expr_df, meta_df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    expr_df, meta_df = generate_synthetic_gse36961()
    expr_df.to_csv("data/expression_matrix.csv")
    meta_df.to_csv("data/sample_metadata.csv", index=False)
    print(f"Synthetic GSE36961 dataset saved.")
    print(f"  Expression matrix: {expr_df.shape[0]:,} probes x {expr_df.shape[1]} samples")
    print(f"  HCM: {(meta_df['label']=='HCM').sum()}  Control: {(meta_df['label']=='Control').sum()}")
