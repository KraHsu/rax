"""Reproducible Anthropic streaming measurements using server token usage."""
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor


def stream(client, payload):
    started = time.perf_counter()
    first = None
    usage = {}
    stop = None
    blocks = []
    with client.request('/v1/messages', {**payload, 'stream': True}) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            kind = event.get('type')
            if kind == 'error':
                raise RuntimeError('Streaming benchmark returned an error')
            if kind == 'message_start':
                usage.update(event.get('message', {}).get('usage', {}))
            elif kind == 'content_block_start':
                blocks.append(event['content_block']['type'])
            elif kind == 'content_block_delta':
                delta = event.get('delta', {})
                if first is None and any(delta.get(name) for name in ('text', 'thinking', 'partial_json')):
                    first = time.perf_counter()
            elif kind == 'message_delta':
                usage.update(event.get('usage', {}))
                stop = event.get('delta', {}).get('stop_reason', stop)
            elif kind == 'message_stop':
                break
    finished = time.perf_counter()
    tokens = usage.get('output_tokens')
    if first is None or not isinstance(tokens, int) or tokens <= 0 or stop is None:
        raise RuntimeError('Incomplete stream or missing server token usage')
    elapsed = finished-started
    return {'output_tokens': tokens, 'input_tokens': usage.get('input_tokens'),
            'elapsed_seconds': elapsed, 'ttft_seconds': first-started,
            'end_to_end_tps': tokens/elapsed,
            'post_first_token_tps_approx': tokens/(finished-first) if finished-first > 0.01 else None,
            'stop_reason': stop, 'content_blocks': blocks}


def measure(client, payload, *, concurrency=1):
    """Wall time and aggregate tokens remain valid with multi-token draft chunks."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: stream(client, payload), range(concurrency)))
    elapsed = time.perf_counter()-started
    total = sum(row['output_tokens'] for row in results)
    return {'concurrency': concurrency, 'payload_sha256': hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            'total_output_tokens': total, 'wall_seconds': elapsed,
            'aggregate_end_to_end_tps': total/elapsed, 'requests': results}


def workloads(model):
    """Same payloads, maximum reasoning, short/long robotics code and batching."""
    base = {'model': model, 'max_tokens': 512, 'temperature': 1.0, 'top_p': 0.95,
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'max'}}
    question = ('Write a Python function for a robot to interpolate position and quaternion poses. '
                'Handle quaternion sign, normalization and the nearly identical case. '
                'Explain numerical edge cases and include tests.')
    context = '\n'.join(f'Calibration waypoint {i}: position=[0.40,0.10,0.25]; '
                        'quaternion_wxyz=[1,0,0,0]; control_hz=30; units=meters.' for i in range(1600))
    return {
        'short_code': {**base, 'messages': [{'role': 'user', 'content': question}]},
        'long_code': {**base, 'messages': [{'role': 'user', 'content': context+'\n'+question}]},
        'batch_code': {**base, 'max_tokens': 256,
                       'messages': [{'role': 'user', 'content': question}]},
    }


def summarize(results):
    """Choose by single-request latency, requiring identical paired payloads."""
    passed = [row for row in results['variants'] if row['status'] == 'passed']
    baseline = next(row for row in passed if row['name'] == 'baseline')
    fixed = ('model','revision','image','tensor_parallel_size','max_model_len','reasoning_effort')
    def key(case):
        return case['workload'],case.get('repeat'),case['concurrency']
    reference = {key(case):case for case in baseline['cases']}
    expected_single = sum(case['concurrency']==1 and case['workload']!='batch_code'
                          for case in baseline['cases'])
    summaries = []
    for row in passed:
        if any(row['config'][name] != baseline['config'][name] for name in fixed):
            raise ValueError('Benchmark changed model, hardware or reasoning settings')
        if not all(row['smoke'].get(name) for name in ('tool_call','tool_result','streaming','thinking_blocks','vision')):
            continue
        if {key(case) for case in row['cases']} != set(reference):
            raise ValueError('Benchmark is missing paired cases')
        single = []
        for case in row['cases']:
            original = reference[key(case)]
            if case['payload_sha256'] != original['payload_sha256']:
                raise ValueError('Benchmark payloads differ between variants')
            if case['concurrency'] == 1 and case['workload'] != 'batch_code':
                single.append(case['aggregate_end_to_end_tps'])
        if not single or len(single) != expected_single:
            raise ValueError('Expected all paired single-request repetitions')
        score = math.exp(sum(math.log(value) for value in single)/len(single))
        item = {'name':row['name'],'single_request_score_tps':score,
                'cases':[{'workload':case['workload'],'repeat':case.get('repeat'),
                          'concurrency':case['concurrency'],
                          'aggregate_end_to_end_tps':case['aggregate_end_to_end_tps'],
                          'input_tokens':[r.get('input_tokens') for r in case['requests']],
                          'output_tokens':[r.get('output_tokens') for r in case['requests']],
                          'mean_ttft_seconds':sum(r['ttft_seconds'] for r in case['requests'])/len(case['requests']),
                          'mean_post_first_token_tps_approx':sum(r['post_first_token_tps_approx'] or 0
                              for r in case['requests'])/len(case['requests'])} for case in row['cases']]}
        def counters(lines):
            totals = {}
            for line in lines:
                match = re.match(r'([^ {]+)(?:\{.*\})?\s+([^ ]+)',line)
                if match:
                    name,value = match.groups()
                    totals[name] = totals.get(name,0)+float(value)
            return totals
        before,after = counters(row.get('metrics_before',[])),counters(row.get('metrics_after',[]))
        deltas = {name:after[name]-before.get(name,0) for name in after if 'spec_decode' in name}
        item['speculative_counter_deltas'] = deltas
        summaries.append(item)
    original = next(row for row in summaries if row['name']=='baseline')
    for row in summaries:
        row['single_request_speedup'] = row['single_request_score_tps']/original['single_request_score_tps']
    winner = max(summaries,key=lambda row:row['single_request_score_tps'])
    return {'selection_metric':'geometric mean of paired single-request end-to-end throughput',
            'fixed_configuration':{name:baseline['config'][name] for name in fixed},
            'winner':winner['name'],'single_request_speedup':winner['single_request_speedup'],
            'variants':summaries}
