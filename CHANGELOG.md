# Changelog

## Version 1.0.0 (2026-07-02)

Initial public release for manuscript submission.

### Features

- Core VMDM workflow for low-coverage TB drug-resistance prediction
- Packaged database for four first-line anti-TB drugs: rifampicin, isoniazid, pyrazinamide and ethambutol
- PRESS feature selection for missing-data-tolerant variant discovery
- Bayesian genotype QC for sparse sequencing data
- Sample-specific LightGBM modelling with optional XGBoost backend
- WHO catalogue-based resistance evidence extraction
- Multi-sample parallel execution

### Dependencies

- Python 3.11
- BWA 0.7.17, Samtools 1.18, Bcftools 1.17, Bedtools 2.31.1, SnpEff 5.0
- LightGBM 4.3.0, XGBoost 2.0.3, Pandas, NumPy and Scikit-learn
