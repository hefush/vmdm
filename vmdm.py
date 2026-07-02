import os
import shlex
import subprocess
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DRUGS = 'Rifampicin,Isoniazid,Pyrazinamide,Ethambutol'
DEFAULT_MIN_PPV = '0.95,0.97,0.85,0.70'
REQUIRED_SCRIPTS = [
    'multi-process.py',
    'features_for_model.py',
    'collect_evidence.py',
    'format_report.py',
    'single_lightgbm_model_sparse.py',
    'single_xgboost_model_sparse.py',
]


def q(value):
    return shlex.quote(str(value))


def qjoin(values):
    return ' '.join(q(value) for value in values)


def resolve_train_data_path(traindb, drug):
    """Return the training table path, accepting either plain text or gzip."""
    candidates = [
        os.path.join(traindb, f'{drug}.data.xls'),
        os.path.join(traindb, f'{drug}.data.xls.gz'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def detect_tool_path(tool_name, venv_path=None):
    """
    Auto-detect tool path with priority: env var > venv path > system PATH.
    
    Args:
        tool_name: Name of the tool (e.g., 'python', 'bcftools')
        venv_path: Path to venv directory (optional)
    
    Returns:
        Path to the tool executable
    """
    import shutil
    
    # 1. Check environment variable (e.g., VMDM_BCFTOOLS)
    env_var = f"VMDM_{tool_name.upper()}"
    env_path = os.environ.get(env_var)
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    
    # 2. Check venv path
    if venv_path:
        venv_tool = os.path.join(venv_path, 'bin', tool_name)
        if os.path.isfile(venv_tool) and os.access(venv_tool, os.X_OK):
            return venv_tool
    
    # 3. Check system PATH
    system_path = shutil.which(tool_name)
    if system_path:
        return system_path
    
    # 4. Return default venv path (may not exist, will error later)
    return os.path.join(SCRIPT_DIR, 'venv', 'bin', tool_name)


def build_config(args):
    drugs = list(OrderedDict.fromkeys(
        drug.strip() for drug in args.drugs.split(',') if drug.strip()
    ))
    min_ppv = [ppv.strip() for ppv in args.min_ppv.split(',') if ppv.strip()]
    configs = {
        'bed': args.bed,
        'traindb': args.traindb,
        'refdb': args.refdb,
        'known': args.known,
        'anno': args.anno,
        'drugs': ','.join(drugs),
        'method': args.method,
        'max_snps': args.max_snps,
        'min_ppv': ','.join(min_ppv),
        'jobs': args.jobs,
        'threads': args.threads,
        'min_cov': args.min_cov,
    }

    for tool in ['python', 'bcftools', 'snpEff', 'bwa', 'samtools', 'bedtools']:
        configs[tool] = detect_tool_path(tool, os.path.join(SCRIPT_DIR, 'venv'))

    return configs


def validate_config(configs):
    if not configs['drugs']:
        raise ValueError('--drugs must contain at least one drug name')
    if configs['jobs'] <= 0:
        raise ValueError('--jobs must be a positive integer')
    if configs['threads'] <= 0:
        raise ValueError('--threads must be a positive integer')
    if configs['max_snps'] <= 0:
        raise ValueError('--max_snps must be a positive integer')
    if configs['min_cov'] < 0:
        raise ValueError('--min_cov must be a non-negative integer')
    if not configs['min_ppv']:
        raise ValueError('--min_ppv must contain at least one value')
    drugs = configs['drugs'].split(',')
    min_ppv = configs['min_ppv'].split(',')
    if len(min_ppv) != len(drugs):
        raise ValueError('--min_ppv must contain one value for each drug in --drugs')

    for value in min_ppv:
        try:
            float(value)
        except ValueError as exc:
            raise ValueError(f'--min_ppv contains a non-numeric value: {value}') from exc

    required_files = {
        'refdb': configs['refdb'],
        'anno': configs['anno'],
    }
    for drug in drugs:
        required_files[f'{drug}.features.xls'] = os.path.join(configs['traindb'], f'{drug}.features.xls')
        required_files[f'{drug}.data.xls(.gz)'] = resolve_train_data_path(configs['traindb'], drug)
    for script in REQUIRED_SCRIPTS:
        required_files[script] = os.path.join(SCRIPT_DIR, script)
    if configs['known']:
        required_files['known'] = configs['known']
        required_files['get_known_feature.py'] = os.path.join(SCRIPT_DIR, 'get_known_feature.py')

    missing = [f'{label}: {path}' for label, path in required_files.items() if not os.path.isfile(path)]
    if missing:
        raise ValueError('missing required file(s): ' + '; '.join(missing))


def read_fq_list(fq_list):
    fqs = {}
    names = set()
    with open(fq_list) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                parts = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"warn: invalid line {line_no} in {fq_list}, ignore!")
                continue
            name, fq_file = parts[0].strip(), parts[1].strip()
            if not name or not fq_file:
                print(f"warn: invalid line {line_no} in {fq_list}, ignore!")
                continue
            if os.path.isfile(fq_file):
                fqs.setdefault(name, {})[fq_file] = True
                names.add(name)
            else:
                print(f"warn: not find {name} {fq_file}, ignore!")
    return fqs, names

def timed_step(name, step, command, log_file):
    return (
        f"(echo {q(f'Starting {step} for {name}')} >> {q(log_file)}; "
        f"bash -o pipefail -c {q(f'time {{ {command}; }}')} && "
        f"echo {q(f'Finished {step} for {name}')} >> {q(log_file)}) "
        f"2>> {q(log_file)}"
    )


def generate_sample_shell(name, files, out_dir, configs):
    sample_dir = os.path.join(out_dir, name)
    os.makedirs(sample_dir, exist_ok=True)

    log_file = os.path.join(sample_dir, f'{name}.time.log')
    raw_vcf = os.path.join(sample_dir, f'{name}.vcf.gz')
    selected_vcf = os.path.join(sample_dir, f'{name}.select.vcf.gz')
    variant_vcf = os.path.join(sample_dir, f'{name}.variant.vcf.gz')
    variant_anno_vcf = os.path.join(sample_dir, f'{name}.variant.anno.vcf')
    known_file = os.path.join(sample_dir, f'{name}.known_feature.xls')
    threads = configs['threads']

    def vcf_call_cmd(input_fq):
        rg = f'"@RG\\tID:MTB\\tSM:{name}\\tPL:illumina\\tLB:{name}"'
        return (
            f"{q(configs['bwa'])} mem -a -k 19 -t {threads} -Y -h 10000 -R {rg} "
            f"{q(configs['refdb'])} {q(input_fq)} | "
            f"{q(configs['samtools'])} view -@ {threads} -F 4 -h -bS - | "
            f"{q(configs['samtools'])} sort -@ {threads} | "
            f"{q(configs['bcftools'])} mpileup -Ou -f {q(configs['refdb'])} - | "
            f"{q(configs['bcftools'])} call -c -Oz -o {q(raw_vcf)}"
        )

    shell_content = []
    cleanup_files = []

    if len(files) == 1:
        vcf_cmd = vcf_call_cmd(files[0])
    else:
        gz_files = [f for f in files if f.endswith('.gz')]
        ungz_files = [f for f in files if not f.endswith('.gz')]
        if gz_files and ungz_files:
            p1_fq = os.path.join(sample_dir, f'{name}.p1.fq')
            p1_gz = os.path.join(sample_dir, f'{name}.p1.fq.gz')
            p2_gz = os.path.join(sample_dir, f'{name}.p2.fq.gz')
            merged_fq = os.path.join(sample_dir, f'{name}.fq.gz')
            merge_cmd = (
                f"cat {qjoin(ungz_files)} > {q(p1_fq)} && gzip {q(p1_fq)} && "
                f"cat {qjoin(gz_files)} > {q(p2_gz)} && "
                f"cat {q(p1_gz)} {q(p2_gz)} > {q(merged_fq)}"
            )
            cleanup_files = [p1_gz, p2_gz, merged_fq]
        elif gz_files:
            merged_fq = os.path.join(sample_dir, f'{name}.fq.gz')
            merge_cmd = f"cat {qjoin(gz_files)} > {q(merged_fq)}"
            cleanup_files = [merged_fq]
        else:
            merged_fq = os.path.join(sample_dir, f'{name}.fq')
            merge_cmd = f"cat {qjoin(ungz_files)} > {q(merged_fq)}"
            cleanup_files = [merged_fq]

        cleanup_cmd = f" && rm {qjoin(cleanup_files)}" if cleanup_files else ""
        vcf_cmd = f"{merge_cmd} && {vcf_call_cmd(merged_fq)}{cleanup_cmd}"

    if configs['bed'] and os.path.isfile(configs['bed']):
        bed_cmd = (
            f"{q(configs['bedtools'])} intersect -a {q(raw_vcf)} -b {q(configs['bed'])} "
            f"-header | gzip - > {q(selected_vcf)}"
        )
        vcf_cmd = f"{vcf_cmd} && {bed_cmd}"
        vcf_file = selected_vcf
    else:
        vcf_file = raw_vcf
    shell_content.append(timed_step(name, 'VCF calling', vcf_cmd, log_file))

    has_known = configs['known'] and os.path.isfile(configs['known'])
    if has_known:
        ref_dir = os.path.dirname(configs['refdb'])
        known_cmd = (
            f"{q(configs['bcftools'])} view -v snps,indels,mnps,ref,bnd,other "
            f"{q(vcf_file)} -o {q(variant_vcf)} && "
            f"{q(configs['snpEff'])} ann -noLog -noStats -no-downstream -no-upstream "
            f"-no-utr -c {q(os.path.join(ref_dir, 'snpeff.config'))} "
            f"-dataDir {q(ref_dir)} ref {q(variant_vcf)} > {q(variant_anno_vcf)} && "
            f"{q(configs['python'])} {q(os.path.join(SCRIPT_DIR, 'get_known_feature.py'))} "
            f"{q(variant_anno_vcf)} {q(configs['known'])} {q(name)} {q(known_file)} {configs['min_cov']}"
        )
        shell_content.append(timed_step(name, 'known feature extraction', known_cmd, log_file))

    drugs = configs['drugs'].split(',')
    for drug in drugs:
        train_data = resolve_train_data_path(configs['traindb'], drug)
        feature_cmd = (
            f"{q(configs['python'])} {q(os.path.join(SCRIPT_DIR, 'features_for_model.py'))} "
            f"{q(vcf_file)} {q(drug)} "
            f"{q(os.path.join(configs['traindb'], f'{drug}.features.xls'))} "
            f"{q(train_data)} "
            f"{q(name)} {q(os.path.join(sample_dir, drug))} "
            f"{configs['max_snps']} {configs['min_cov']}"
        )
        if has_known:
            feature_cmd += f" {q(known_file)}"
        shell_content.append(timed_step(name, f'feature extraction for {drug}', feature_cmd, log_file))

    min_ppv_by_drug = dict(zip(drugs, configs['min_ppv'].split(',')))
    predict_method = os.path.join(SCRIPT_DIR, 'single_lightgbm_model_sparse.py')
    if 'xgboost' in configs['method'].lower():
        predict_method = os.path.join(SCRIPT_DIR, 'single_xgboost_model_sparse.py')
    for drug in drugs:
        min_ppv = min_ppv_by_drug[drug]
        predict_cmd = (
            f"{q(configs['python'])} {q(predict_method)} "
            f"{q(os.path.join(sample_dir, drug, f'{name}.train.xls'))} "
            f"{q(os.path.join(sample_dir, drug, f'{name}.test.xls'))} "
            f"{q(os.path.join(sample_dir, drug, f'{name}.predict.xls'))} {min_ppv}"
        )
        shell_content.append(timed_step(name, f'prediction for {drug}', predict_cmd, log_file))

    predict_list = os.path.join(sample_dir, f'{name}.predict.list')
    with open(predict_list, 'w') as p:
        for drug in drugs:
            p.write(f"{drug}\t{os.path.join(sample_dir, drug, f'{name}.predict.xls')}\n")

    out_file = os.path.join(sample_dir, f'{name}.report.xls')
    report_args = [known_file, predict_list, out_file] if has_known else [predict_list, out_file]
    report_cmd = (
        f"{q(configs['python'])} {q(os.path.join(SCRIPT_DIR, 'collect_evidence.py'))} "
        f"{qjoin(report_args)}"
    )
    shell_content.append(timed_step(name, 'report', report_cmd, log_file))

    report_file = os.path.join(sample_dir, f'{name}.report_anno.xls')
    anno_cmd = (
        f"{q(configs['python'])} {q(os.path.join(SCRIPT_DIR, 'format_report.py'))} "
        f"{q(configs['anno'])} {q(out_file)} {q('Mycobacterium tuberculosis complex')} {q(report_file)}"
    )
    shell_content.append(timed_step(name, 'annotation', anno_cmd, log_file))

    return "\n".join(shell_content)

def main(fq_list, out_dir, configs):
    os.makedirs(out_dir, exist_ok=True)
    fqs, names = read_fq_list(fq_list)
    if not names:
        raise ValueError(f'no valid FASTQ files found in {fq_list}')
    
    # Run prediction for each sample in parallel.
    run_sh = os.path.join(out_dir, 'run.sh')
    with open(run_sh, 'w') as r:
        for name in sorted(names):
            files = list(fqs[name].keys())
            shell_content = generate_sample_shell(name, files, out_dir, configs)
            sample_sh = os.path.join(out_dir, name, f'{name}.sh')
            with open(sample_sh, 'w') as s:
                s.write(shell_content + "\n")
            r.write(f"sh {q(sample_sh)}\n")
    subprocess.run(
        [configs['python'], os.path.join(SCRIPT_DIR, 'multi-process.py'), '-cpu', str(configs['jobs']), run_sh],
        check=True,
    )

if __name__ == "__main__":
    import argparse

    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        usage='%(prog)s [-h] [options] fq_list out_dir',
        description='VMDM: Variant-aware, Missing-tolerant Dynamic Modelling for TB drug resistance prediction',
        formatter_class=HelpFormatter,
        epilog='''
Examples:
  # Run with default configuration (mNGS mode, min_cov=0)
  python vmdm.py sample.list output_dir

  # Run tNGS mode with min_cov=10
  python vmdm.py sample.list output_dir --min_cov 10

  # Show this help message
  python vmdm.py -h

        '''
    )
    parser.add_argument('fq_list', help='Input FASTQ list file (tab-separated: sample_name<TAB>fastq_path)')
    parser.add_argument('out_dir', help='Output directory for results')

    data_group = parser.add_argument_group('Data files')
    data_group.add_argument('--refdb', default=os.path.join(SCRIPT_DIR, 'MTBdb/reference/ref.fa'),
                            help='Reference database path')
    data_group.add_argument('--known', default=os.path.join(SCRIPT_DIR, 'MTBdb/WHO_known.info.xls'),
                            help='WHO known resistance variants file')
    data_group.add_argument('--anno', default=os.path.join(SCRIPT_DIR, 'MTBdb/drug.anno'),
                            help='Drug annotation file')
    data_group.add_argument('--bed', default=os.path.join(SCRIPT_DIR, 'MTBdb/features.bed'),
                            help='Optional BED file for targeted regions; use "" to disable')
    data_group.add_argument('--traindb', default=os.path.join(SCRIPT_DIR, 'MTBdb'),
                            help='Training database directory')

    analysis_group = parser.add_argument_group('Analysis parameters')
    analysis_group.add_argument('--drugs', default=DEFAULT_DRUGS,
                                help='Comma-separated drug names')
    analysis_group.add_argument('--min_ppv', default=DEFAULT_MIN_PPV,
                                help='Comma-separated PPV thresholds matched to --drugs order')
    analysis_group.add_argument('--method', default='LightGBM', choices=['LightGBM', 'XGBoost'],
                                help='Machine learning method')
    analysis_group.add_argument('--max_snps', default=1000, type=int,
                                help='Maximum number of SNPs to consider')
    analysis_group.add_argument('--min_cov', default=0, type=int,
                                help='Minimum coverage depth; recommended 10 for tNGS')
    analysis_group.add_argument('--jobs', default=5, type=int,
                                help='Number of samples to run in parallel')
    analysis_group.add_argument('--threads', default=20, type=int,
                                help='Threads per sample for bwa/samtools steps')

    args = parser.parse_args()
    configs = build_config(args)
    try:
        validate_config(configs)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        main(args.fq_list, args.out_dir, configs)
    except ValueError as exc:
        parser.error(str(exc))
