#!/usr/bin/env python3
"""Validate paired tuning results and save a machine-readable comparison."""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from harness.model_serving.benchmark import summarize
from harness.model_serving.lifecycle import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args = parser.parse_args()
    results = json.loads((args.root/'results.json').read_text())
    report = summarize(results)
    report['slurm_job_id'] = results.get('job_id')
    for line in (args.root/'baseline/serve.log').read_text().splitlines():
        if line.startswith('{') and '"gpu_preflight"' in line:
            report['hardware_preflight'] = json.loads(line)
            break
    for variant in report['variants']:
        directory = args.root/variant['name']
        endpoint = json.loads((directory/'endpoint.json').read_text())
        runtime = endpoint.get('runtime_env')
        if runtime is None:
            matches = re.findall(r'OMP_NUM_THREADS=(\d+) is set', (directory/'serve.log').read_text())
            runtime = {'OMP_NUM_THREADS':matches[-1]} if matches else {}
        variant['runtime_env'] = runtime
    report['sampling'] = {'temperature':1.0,'top_p':0.95,'thinking':'adaptive','effort':'max'}
    report['measurement_scope'] = 'Fixed 512-output-token short/long image-code requests; 256-token concurrency sweeps; not full DexJoCo episode timing'
    write_json(args.root/'comparison.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
