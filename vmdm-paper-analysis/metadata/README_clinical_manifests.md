# Clinical Cohort Manifests

This directory contains the final sample manifests used for manuscript
inspection and accession-level traceability. It does not include raw sequencing
reads or local build caches.

## Included Files

| File | Contents |
|---|---|
| `PRJEB88154_tNGS_samples_pDST_QC.csv` | Portugal ONT tNGS cohort with sample accessions, pDST calls and tNGS QC fields. |
| `PRJNA1160005_ONT_tNGS_samples_pDST_QC.csv` | Seq&Treat ONT / AmPORE-TB samples with BioSample, SRA run, pDST and workflow QC fields. |
| `PRJNA1160005_GS_tNGS_samples_pDST_QC.csv` | Seq&Treat GS / Deeplex samples with BioSample, SRA run, pDST and workflow QC fields. |
| `PRJNA1160005_ABL_tNGS_samples_pDST_QC.csv` | Seq&Treat ABL / DeepChek samples with BioSample, SRA run and pDST fields. |
| `PRJNA1160005_all_tNGS_samples_pDST_QC.csv` | Combined PRJNA1160005 tNGS manifest across ONT, GS and ABL. |
| `PRJNA486713_sputum_WGS_samples_pDST_QC.csv` | Direct sputum Illumina WGS / SureSelect manifest for PRJNA486713. |
| `PRJEB56100_sputum_Illumina_WGS_samples_pDST_QC.csv` | Direct sputum Illumina WGS manifest for PRJEB56100 with Table 2 pDST fields where available. |
| `PRJEB56100_culture_Illumina_WGS_samples_reference_QC.csv` | Culture-derived Illumina WGS manifest for PRJEB56100 with culture-WGS reference calls and Table 2 pDST fields where available. |

## Field Notes

- `biosample_accession`, `run_accession` and related accession fields provide
  traceability to public archives.
- `pDST_*` columns encode phenotypic drug-susceptibility testing calls where
  available.
- `sequencing_raw_qc`, `tngs_interpretation_qc` and `tngs_workflow_qc` retain
  source-study QC annotations used for cohort inspection.

The manifests are supporting metadata for the paper-analysis repository. The
figure-generation scripts use the derived source data stored under `figures/`,
and the numerical validation scripts use the derived tables stored under
`analysis_checks/`.
