import os
import sys
import csv

def read_known_file(known_file):
    evidence = {}
    drug_mapping = {
        'rifampin': 'Rifampicin',
        'isoniazid': 'Isoniazid',
        'ethambutol': 'Ethambutol',
        'pyrazinamide': 'Pyrazinamide'
    }

    with open(known_file, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            name = row['Name']
            label = row['Label']
            whos = row['WHOs']
            drugs = whos.split(',')
            for drug_info in drugs:
                drug, confidence = drug_info.split('(')[0], drug_info.split('(')[1].rstrip(')')
                drug = drug_mapping.get(drug, drug)
                if name not in evidence:
                    evidence[name] = {}
                if drug not in evidence[name]:
                    evidence[name][drug] = []
                evidence[name][drug].append(f"{label}({confidence})")
    return evidence

def read_predict_list(predict_list, evidence):
    drug_mapping = {
        'rifampin': 'Rifampicin',
        'isoniazid': 'Isoniazid',
        'ethambutol': 'Ethambutol',
        'pyrazinamide': 'Pyrazinamide'
    }

    with open(predict_list, 'r') as file:
        for line in file:
            drug, file_path = line.strip().split()
            # Skip rows with insufficient data that did not generate prediction files.
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found,skipping...", file=sys.stderr)
                continue
            drug = drug_mapping.get(drug, drug)
            with open(file_path, 'r') as pred_file:
                reader = csv.DictReader(pred_file, delimiter='\t')
                for row in reader:
                    name = row['Name']
                    prob = f"{float(row['y_pred_prob']):.2f}"
                    pred = int(row['y_pred'])
                    if pred > 0:
                        if name not in evidence:
                            evidence[name] = {}
                        if drug not in evidence[name]:
                            evidence[name][drug] = []
                        evidence[name][drug].append(f"model({prob})")
    return evidence

def write_output_file(out_file, evidence):
    with open(out_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')
        writer.writerow(['Name', 'Drug', 'Evidence'])
        for name in sorted(evidence.keys()):
            for drug in sorted(evidence[name].keys()):
                writer.writerow([name, drug, ','.join(evidence[name][drug])])

def main(known_file, predict_list, out_file):
    evidence = {}
    if known_file:
        evidence = read_known_file(known_file)
    evidence = read_predict_list(predict_list, evidence)
    write_output_file(out_file, evidence)

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print(f"Usage: python {sys.argv[0]} [<SAMN02586075.known_feature.xls>] <SAMN02586075.predict.list> <SAMN02586075.report.xls>")
        sys.exit(1)

    known_file = sys.argv[1] if len(sys.argv) == 4 else None
    predict_list, out_file = sys.argv[-2], sys.argv[-1]
    main(known_file, predict_list, out_file)
