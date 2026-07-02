import os
import sys

def parse_known_file(known_file):
    """
    Parses the known file to get SINGLES and REGIONS dictionaries.
    """
    singles = {}
    regions = {}
    rares = {}
    with open(known_file, 'r') as k:
        for line in k:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            id, chr, begin, end, func, label, who, freq = parts[0], parts[1], int(parts[2]), int(parts[3]), parts[4], parts[5], parts[6], parts[7]
            if begin == end:
                singles[id] = f"{label}\t{func}\t{who}"
                if freq == 'rare':
                    rares[id] = 1
            else:
                if begin > end:
                    begin, end = end, begin
                for pos in range(begin, end + 1):
                    regions[f"{chr}-{pos}"] = f"{label}\t{func}\t{who}"
    return singles, regions, rares

def process_vcf_file(vcf_file, singles, regions, rares, name, min_cov=0):
    """
    Processes the VCF file to populate the KNOWN dictionary based on the conditions.
    """
    known = {}
    with open(vcf_file, 'r') as v:
        for line in v:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            chr, pos, alt, info, geno = parts[0], parts[1], parts[4].split(',')[0], parts[7], parts[9]
            # Skip non-variant positions
            if geno.startswith("0/0") or geno.startswith("./."):
                continue
            # Skip variants not in known list or region
            id = f"{chr}-{pos}-{alt}"
            region_id = f"{chr}-{pos}"
            if id not in singles and region_id not in regions:
                continue
            # Get variants information for QC
            dp = None
            func = None
            fields = info.split(';')
            for field in fields:
                if field.startswith('DP='):
                    dp = int(field.split('=')[1])
                elif field.startswith('ANN='):
                    func = field.split('|')[1]
                    break
            # Initial QC: filter by min_cov threshold (applied before other QC checks)
            if min_cov > 0 and dp is not None and dp < min_cov:
                continue
            # Skip variants with 0/1 genotype but 1<dp<4.
            if geno.startswith("0/1") and dp > 1 and dp < 4:
                continue
            # Known position point mutations
            if id in singles:
                # Quality control for rare variants: depth>=4 or hom with depth>=2
                if id in rares:
                    if (not geno.startswith("1/1") and dp < 4) or (geno.startswith("1/1") and dp < 2):
                        continue
                known[singles[id]] = f"DP={dp},func={func},geno={geno}"
                continue
            # Region-specific LoF or frameshift mutations
            if region_id in regions:
                label, s_func, who = regions[region_id].split('\t')
                # Quality control for stop_gained mutations( stop_gained is not position fixed, will acculate error)
                if 'stop_gained' in func.lower():
                    if not geno.startswith("1/1") or dp < 2:
                        continue
                else:
                #frameshift and start_lost is rare variants
                    if not geno.startswith("1/1") and dp < 4:
                        continue
                # Check if the mutation type is required
                if any(f.lower() in func.lower() for f in s_func.split(',')):
                    known[regions[region_id]] = f"DP={dp},func={func},geno={geno}"
    return known

def write_output_file(out_file, known, name):
    """
    Writes the output file with the required format.
    """
    with open(out_file, 'w') as o:
        o.write("Name\tLabel\tCount\tFunc\tWHOs\tInfo\n")
        for key in sorted(known.keys()):
            parts = key.split('\t')
            label, func, whos = parts[0], parts[1], parts[2]
            o.write(f"{name}\t{label}\t1\t{func}\t{whos}\t{known[key]}\n")

def main():
    """
    Main function to parse command-line arguments and process the files.
    """
    if len(sys.argv) < 5:
        print(f"Usage: python {sys.argv[0]} <SAMN02585979.variant.anno.vcf> <WHO_known.info.xls> <name:SAMN02585979> <SAMN02585979.known_feature.xls> [min_cov:0]")
        sys.exit(1)

    vcf_file, known_file, name, out_file = sys.argv[1:5]
    min_cov = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    singles, regions, rares = parse_known_file(known_file)
    known = process_vcf_file(vcf_file, singles, regions, rares, name, min_cov)
    write_output_file(out_file, known, name)

if __name__ == "__main__":
    main()
