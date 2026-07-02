import os
import sys
import pandas as pd
import gzip

def read_feature_list(feature_list, drug_label):
    """
    Read feature list file and store data in dictionaries.
    """
    df = pd.read_csv(feature_list, delimiter='\t')

    sites = {}
    evidence = {}

    for _, row in df.iterrows():
        feature = row['ID']
        if '-' in feature:
            chr, pos, alt = feature.split('-')[0:3]
            sites.setdefault(f"{chr}-{pos}", {})[alt] = True
        #record evidence.
        label = row['Label']
        who = str(row['WHOs'])
        if drug_label.lower() in who.lower():
            evidence[feature] = f"{label}\t{who}"

    return sites, evidence

def read_vcf_file(vcf_file, sites, min_cov=0):
    """
    Read VCF file and store sample genotype data in dictionaries.
    """
    genos = {}

    def process_vcf_line(line):
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split('\t')
            chr, pos, ref, alt = parts[0], parts[1], parts[3], parts[4]
            key = f"{chr}-{pos}"
            if key in sites:
                # Parse DP (depth) from INFO field for min_cov filtering
                if min_cov > 0 and len(parts) >= 8:
                    info = parts[7]
                    dp = 0
                    for field in info.split(';'):
                        if field.startswith('DP='):
                            try:
                                dp = int(field.split('=')[1])
                            except (ValueError, IndexError):
                                dp = 0
                            break
                    if dp < min_cov:
                        return
                alt = alt.split(',')[0] if ',' in alt else alt
                if alt in sites[key]:
                    genos[f"{chr}-{pos}-{alt}"] = 1
                else:
                    for oth in sites[key]:
                        genos[f"{chr}-{pos}-{oth}"] = 0

    if vcf_file.endswith('.gz'):
        with gzip.open(vcf_file, 'rt') as f:
            for line in f:
                process_vcf_line(line)
    else:
        with open(vcf_file, 'r') as f:
            for line in f:
                process_vcf_line(line)

    return genos

def add_other_feature(genos, other_feature_file):
    """
    Add other feature information.
    """
    with open(other_feature_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('Name'):
                name, gene, count = line.split("\t")[0:3]
                genos[gene] = int(count)
    return genos

def output_evidence(genos, evidence, out_dir, name, drug_label):
    """
    Output evidence information.
    """
    positive_evidence_list = {}
    negative_confidence_list = {}

    for feature in evidence:
        if feature in genos:
            label, who = evidence[feature].split('\t')
            if genos[feature] > 0:
                positive_evidence_list[label] = True
            else:
                negative_confidence_list[who] = True

    positive_evidence = 'NULL' if not positive_evidence_list else ','.join(sorted(positive_evidence_list.keys()))
    negative_confidence = min(sum(float(who.split('(')[1].split('%')[0]) for who in negative_confidence_list if '(' in who and '%' in who), 100)

    with open(os.path.join(out_dir, f"{name}.evidence.xls"), 'w') as f:
        f.write("Drug\tPositiveEvidence\tNegativeConfidence\n")
        f.write(f"{drug_label}\t{positive_evidence}\t{negative_confidence}\n")

def output_train_data_old(genos, train_data, out_dir, name, top_max):
    """
    Output training data.
    """
    df = pd.read_csv(train_data, delimiter='\t')
    # Filter columns
    snps = [col for col in df.columns[2:] if col in genos]
    snps = snps[:top_max] if len(snps) > top_max else snps

    # Prepare data for output
    train_data_out = df.loc[:, ['Name', 'Drug'] + snps]
    train_data_out = train_data_out.rename(columns=lambda x: x if x in ['Name', 'Drug'] + snps else None)
    train_data_out = train_data_out.dropna(axis=1, how='all')

    # Write to file in one go
    output_file = os.path.join(out_dir, f"{name}.train.xls")
    train_data_out.to_csv(output_file, sep='\t', index=False)

    return snps

def output_train_data(genos, train_data, out_dir, name, top_max):
    """
    Output training data.
    """
    # Read entire file to get column names
    df = pd.read_csv(train_data, delimiter='\t', nrows=0)  # Only read column names
    all_columns = df.columns.tolist()
    # Filter SNP columns
    snps = [col for col in all_columns[2:] if col in genos]
    snps = snps[:top_max] if len(snps) > top_max else snps
    # Read necessary columns
    df = pd.read_csv(train_data, delimiter='\t', usecols=['Name', 'Drug'] + snps)
    # Prepare output data
    train_data_out = df.loc[:, ['Name', 'Drug'] + snps]
    # Write to file
    output_file = os.path.join(out_dir, f"{name}.train.xls")
    train_data_out.to_csv(output_file, sep='\t', index=False)

    return snps

def output_test_data(genos, snps, out_dir, name):
    """
    Output test data.
    """
    with open(os.path.join(out_dir, f"{name}.test.xls"), 'w') as f:
        f.write("Name\t" + "\t".join(snps) + "\n")
        f.write(name + "\t" + "\t".join(str(genos.get(snp, 'NA')) for snp in snps) + "\n")

def main(vcf_file, drug_label, feature_list, train_data, name, out_dir, top_max=50000, min_cov=0, other_feature_file=''):
    os.makedirs(out_dir, exist_ok=True)

    sites, evidence = read_feature_list(feature_list, drug_label)
    genos = read_vcf_file(vcf_file, sites, min_cov)
    # add other feature to genos.
    if os.path.isfile(other_feature_file):
        genos = add_other_feature(genos, other_feature_file)
    # output evidence.
    output_evidence(genos, evidence, out_dir, name, drug_label)
    # output train and test data.
    snps = output_train_data(genos, train_data, out_dir, name, top_max)
    output_test_data(genos, snps, out_dir, name)

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python script.py <input.vcf> <drug:rifampin> <key_features.xls> <train_data.xls> <name> <outdir> [top_max:50000] [min_cov:0] [other_features_data.xls]")
        sys.exit(1)

    vcf_file = sys.argv[1]
    drug_label = sys.argv[2]
    feature_list = sys.argv[3]
    train_data = sys.argv[4]
    name = sys.argv[5]
    out_dir = sys.argv[6]
    top_max = int(sys.argv[7]) if len(sys.argv) > 7 else 50000
    min_cov = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    other_feature_file = sys.argv[9] if len(sys.argv) > 9 else ''

    main(vcf_file, drug_label, feature_list, train_data, name, out_dir, top_max, min_cov, other_feature_file)
