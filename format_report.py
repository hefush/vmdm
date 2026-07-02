#!/usr/bin/env python3
"""Script for processing drug resistance analysis reports"""

import sys
from collections import defaultdict
from typing import Dict, Set, DefaultDict

def load_annotations(anno_file: str) -> Dict[str, str]:
    """Load annotation file"""
    annotations = {}
    try:
        with open(anno_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    label, annotation = parts[0], parts[1]
                    annotations[label] = annotation
    except IOError as e:
        sys.exit(f"Error: Unable to read annotation file {anno_file}: {e}")
    
    return annotations

def parse_report_file(report_file: str, annotations: Dict[str, str]) -> tuple:
    """Parse report file"""
    evidences = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    models = defaultdict(lambda: defaultdict(set))
    
    try:
        with open(report_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or ('Evidence' in line):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    print(f"Warning: Line {line_num} has insufficient columns, skipping")
                    continue
                
                sample_name, drug, evidence_str = parts[0], parts[1], parts[2]
                drug = annotations.get(drug, drug)
                
                for evidence in evidence_str.split(','):
                    evidence = evidence.strip()
                    if not evidence:
                        continue
                    
                    if 'model' in evidence:
                        models[sample_name][evidence].add(drug)
                    else:
                        variant = evidence.split('(')[0].strip()
                        gene = variant.split('_')[0]
                        evidences[sample_name][gene][variant].add(drug)
    except IOError as e:
        sys.exit(f"Error: Unable to read report file {report_file}: {e}")
    
    return evidences, models

def write_annotated_report(
    output_file: str, 
    evidences: DefaultDict, 
    models: DefaultDict, 
    species: str, 
    annotations: Dict[str, str]
) -> None:
    """Write annotated report"""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            # Write header
            header = [
                "SampleID", "Name", "Depth", "MutationRate[%]", "MutationConfidenceScore",
                "AssociatedSpecies", "AssociationConfidenceScore", "ResistanceCategory", "ResistanceEvidence",
                "SpeciesOfInterest", "SpeciesList", "ResistanceMechanismAnnotation"
            ]
            file.write("\t".join(header) + "\n")
            
            processed_drugs = defaultdict(set)
            
            # Process variants with evidence
            for sample in sorted(evidences.keys()):
                for gene in sorted(evidences[sample].keys()):
                    gene_annotation = annotations.get(gene, "Insufficient research on this gene")
                    
                    for variant in sorted(evidences[sample][gene].keys()):
                        drugs = ";".join(sorted(evidences[sample][gene][variant]))
                        
                        row_data = [
                            sample, variant, "1", "90%", "Resistant",
                            species, "95%", drugs, drugs,
                            f"{species}:{variant}", f"{species}:{variant}",
                            gene_annotation
                        ]
                        file.write("\t".join(row_data) + "\n")
                        
                        # Record processed drugs
                        processed_drugs[sample].update(evidences[sample][gene][variant])
            
            # Process model predictions
            for sample in sorted(models.keys()):
                for model in sorted(models[sample].keys()):
                    for drug in sorted(models[sample][model]):
                        if drug in processed_drugs[sample]:
                            continue
                            
                        row_data = [
                            sample, "Predicted", "0", "Unknown", "Probable Resistance",
                            species, "90%", drug, drug,
                            f"{species}:{model}", f"{species}:{model}",
                            "Model prediction without direct mutation evidence coverage."
                        ]
                        file.write("\t".join(row_data) + "\n")
    except IOError as e:
        sys.exit(f"Error: Unable to write output file {output_file}: {e}")

def main():
    """Main function"""
    if len(sys.argv) != 5:
        script_name = sys.argv[0] if sys.argv else "script.py"
        print(f"Usage: {script_name} <annotation_file> <input_report> <species> <output_report>")
        print("Example: script.py drug.anno SAMEA1016078.report.xls \"Mycobacterium tuberculosis complex\" SAMEA1016078.report_anno.xls")
        sys.exit(1)
    
    anno_file, input_file, species, output_file = sys.argv[1:5]
    
    # Load annotations
    annotations = load_annotations(anno_file)
    
    # Parse report
    evidences, models = parse_report_file(input_file, annotations)
    
    # Generate annotated report
    write_annotated_report(output_file, evidences, models, species, annotations)

if __name__ == "__main__":
    main()
