#!/usr/bin/env python3
"""Compare serving variants sequentially within one existing GPU allocation."""
import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness.model_serving.benchmark import measure, workloads
from harness.model_serving.configuration import ServeConfig
from harness.model_serving.lifecycle import write_json
from harness.model_serving.probes import MessagesClient, smoke


def stop(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def metrics(client):
    with client.request('/metrics') as response:
        lines = response.read().decode().splitlines()
    return [line for line in lines if line.startswith('vllm:') and any(
        name in line for name in ('spec_decode','prefix_cache','request_success','generation_tokens_total',
                                  'prompt_tokens_cached','request_decode_time_seconds_sum'))
        and '_bucket' not in line and '_created' not in line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['root']).resolve()
    prepared = json.loads(Path(plan['prepared']).read_text())
    root.mkdir(parents=True, exist_ok=True)
    results = []
    overall = {'status':'running', 'job_id':os.environ.get('SLURM_JOB_ID'),
               'started_at':time.time(), 'variants':results}
    write_json(root/'results.json', overall)
    active = None
    def terminate(number, _):
        if active is not None:
            stop(active)
        raise SystemExit(128+number)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, terminate)
    for variant in plan['variants']:
        config = ServeConfig.load(variant['config'])
        directory = root/variant['name']
        directory.mkdir(exist_ok=True)
        if (directory/'benchmark.json').exists():
            result = json.loads((directory/'benchmark.json').read_text())
            if result.get('status') == 'passed':
                results.append(result)
                continue
        write_json(directory/'prepared.json', prepared)
        shared_cache = root/'cache'
        shared_cache.mkdir(exist_ok=True)
        if not (directory/'cache').exists():
            (directory/'cache').symlink_to(shared_cache, target_is_directory=True)
        result = {'name':variant['name'], 'config':asdict(config), 'status':'starting',
                  'started_at':time.time(), 'cases':[]}
        results.append(result)
        write_json(root/'results.json', overall)
        command = [sys.executable, '-m', 'harness.model_serving', 'serve', '--config', variant['config'],
                   '--root', str(directory), '--key-file', plan['key_file']]
        env = dict(os.environ)
        env.update(variant.get('environment', {}))
        try:
            with (directory/'serve.log').open('a') as log:
                active = subprocess.Popen(command, cwd=PROJECT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic()+plan.get('ready_timeout', 3600)
                while time.monotonic() < deadline:
                    if active.poll() is not None:
                        raise RuntimeError(f'Server startup exited {active.returncode}')
                    endpoint = directory/'endpoint.json'
                    if endpoint.exists() and json.loads(endpoint.read_text()).get('status') == 'ready':
                        break
                    time.sleep(5)
                else:
                    raise TimeoutError('Variant readiness timed out')
                result.update(status='testing', ready_at=time.time())
                write_json(root/'results.json', overall)
                client = MessagesClient(f'http://127.0.0.1:{config.port}', plan['key_file'],timeout=900)
                result['smoke'] = smoke(client, config.served_model, image=plan.get('image'))
                cases = workloads(config.served_model)
                if plan.get('image'):
                    encoded = base64.b64encode(Path(plan['image']).read_bytes()).decode()
                    text = cases['long_code']['messages'][0]['content']
                    cases['long_code']['messages'][0]['content'] = [
                        {'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':encoded}},
                        {'type':'text','text':text}]
                # Small warmup; every measured request retains maximum reasoning.
                warmup = {**cases['short_code'], 'max_tokens':64}
                measure(client,warmup)
                result['metrics_before'] = metrics(client)
                for name in ('short_code','long_code'):
                    for repeat in range(plan.get('repeats',2)):
                        row = {'workload':name, 'repeat':repeat,
                               **measure(client,cases[name])}
                        result['cases'].append(row)
                        write_json(root/'results.json',overall)
                        print(json.dumps({'variant':variant['name'], **row}),flush=True)
                for concurrency in plan.get('concurrencies',[1,2,4,8]):
                    row = {'workload':'batch_code', **measure(client,cases['batch_code'],concurrency=concurrency)}
                    result['cases'].append(row)
                    write_json(root/'results.json',overall)
                    print(json.dumps({'variant':variant['name'], **row}),flush=True)
                result['metrics_after'] = metrics(client)
                result['status'] = 'passed'
        except Exception as error:
            result.update(status='failed', error=str(error))
            print(json.dumps({'variant':variant['name'], 'error':str(error)}),flush=True)
        finally:
            if active is not None:
                stop(active)
                active = None
            result['finished_at'] = time.time()
            write_json(directory/'benchmark.json',result)
            write_json(root/'results.json',overall)
    overall.update(status='finished',finished_at=time.time())
    write_json(root/'results.json',overall)
    if not any(result['status']=='passed' for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
