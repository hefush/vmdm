from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE_TABLE = Path("sTable5_base.xlsx")
FEATURE_TABLE = Path("PRJNA1160005_tNGS_result_d10_with_feature_num.tsv")
OUT = ROOT / "supplementary_tables" / "Supplementary_Table_5_rebuilt.xlsx"

QC_ORDER = ["pass", "partial", "fail"]
DRUG_ORDER = ["Rifampicin", "Isoniazid", "Ethambutol", "Pyrazinamide"]


def iqr_low(x):
    return x.quantile(0.25)


def iqr_high(x):
    return x.quantile(0.75)


def main():
    sheets = pd.read_excel(BASE_TABLE, sheet_name=None)
    feature = pd.read_csv(FEATURE_TABLE, sep="\t")
    feature = feature[(feature["Method"] == "vmdm") & (feature["QC"] != "NOT_RUN")].copy()

    feature_summary = (
        feature.groupby("QC", sort=False)["FeatureNum"]
        .agg(
            n_specimen_drug_pairs="count",
            median="median",
            IQR_low=iqr_low,
            IQR_high=iqr_high,
            mean="mean",
            min="min",
            max="max",
        )
        .reindex(QC_ORDER)
        .reset_index()
    )
    feature_summary.insert(
        1,
        "SeqTreat_sequence_outcome",
        feature_summary["QC"].map(
            {
                "pass": "Complete sequence",
                "partial": "Partial sequence",
                "fail": "Sequence failure",
            }
        ),
    )

    feature_by_drug = (
        feature.groupby(["QC", "Drug"], sort=False)["FeatureNum"]
        .agg(
            n_specimen_drug_pairs="count",
            median="median",
            IQR_low=iqr_low,
            IQR_high=iqr_high,
            mean="mean",
            min="min",
            max="max",
        )
        .reset_index()
    )
    feature_by_drug["QC"] = pd.Categorical(feature_by_drug["QC"], QC_ORDER, ordered=True)
    feature_by_drug["Drug"] = pd.Categorical(feature_by_drug["Drug"], DRUG_ORDER, ordered=True)
    feature_by_drug = feature_by_drug.sort_values(["QC", "Drug"]).reset_index(drop=True)

    raw_feature = feature[
        ["Drug", "Name", "laboratory_method", "y_pred", "Evidence", "QC", "FeatureNum"]
    ].copy()
    raw_feature["QC"] = pd.Categorical(raw_feature["QC"], QC_ORDER, ordered=True)
    raw_feature["Drug"] = pd.Categorical(raw_feature["Drug"], DRUG_ORDER, ordered=True)
    raw_feature = raw_feature.sort_values(["QC", "Drug", "Name"]).reset_index(drop=True)

    note = pd.DataFrame(
        {
            "Field": [
                "FeatureNum",
                "feature_availability_by_QC",
                "feature_availability_by_QC_drug",
                "raw_vmdm_feature_num",
            ],
            "Description": [
                (
                    "Number of model-eligible PRESS features retained in the VMDM input matrix "
                    "for a specimen-drug pair after the 10x depth filter and Bayesian genotype QC."
                ),
                "Summary of FeatureNum across Seq&Treat target-sequence recovery strata.",
                "Drug-stratified summary of FeatureNum across Seq&Treat target-sequence recovery strata.",
                "VMDM specimen-drug rows used to derive the FeatureNum summaries.",
            ],
        }
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        feature_summary.to_excel(writer, sheet_name="feature_availability_by_QC", index=False)
        feature_by_drug.to_excel(writer, sheet_name="feature_by_QC_drug", index=False)
        raw_feature.to_excel(writer, sheet_name="raw_vmdm_feature_num", index=False)
        note.to_excel(writer, sheet_name="notes", index=False)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
