#!/usr/bin/env python3
"""Continuously instantiate reviewed task templates directly on the data host."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.robotwin.annotation import digest
from services.robotwin.annotation_templates import match, refresh
from services.robotwin.collection import atomic_json
from scripts.robotwin_annotation_queue import complete


def run_once(root):
    root = Path(root); results = []
    for task in json.loads((root/'batch_manifest.json').read_text())['tasks']:
        refresh(root, task)
    for path in sorted(root.glob('*/aloha-agilex_clean_50/annotation_queue/episode*.json')):
        start = time.perf_counter()
        with (root/'.annotation_queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = json.loads(path.read_text())
            if receipt['annotation_status'] != 'pending' or receipt.get('programmatic_error'): continue
            episode = Path(receipt['path'])
            try: matched = match(root, receipt['task'], episode)
            except (ValueError, KeyError) as exc:
                receipt['programmatic_error'] = f'{type(exc).__name__}: {exc}'
                atomic_json(path, receipt)
                continue
            if matched is None: continue
            entry, constraints, audit = matched
            job = uuid.uuid4().hex
            receipt.update(annotation_status='processing', claimed_by='programmatic', claim_id=job, claimed_unix=time.time())
            atomic_json(path, receipt)
        try:
            # Numeric work has finished before hashing/copying images; keep the
            # same checked, atomic publication path as the agent workflow.
            proposal = {'source_sha256': digest(episode), 'constraints': constraints}
            result = complete(path, job, proposal, 'template:'+audit['template_sha256'])
            result.update(task=receipt['task'], index=receipt['index'], seconds=time.perf_counter()-start,
                          template_sha256=audit['template_sha256'])
            with (root/'programmatic_publications.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            results.append(result)
        except Exception as exc:
            with (root/'.annotation_queue.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                current = json.loads(path.read_text())
                if current.get('claim_id') == job and current['annotation_status'] == 'processing':
                    current.update(annotation_status='pending', programmatic_error=f'{type(exc).__name__}: {exc}')
                    atomic_json(path, current)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--once', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    lock = (root/'.programmatic.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    count = 0
    while True:
        results = run_once(root); count += len(results)
        for result in results: print(json.dumps(result), flush=True)
        total = len(list(root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5')))
        target = json.loads((root/'batch_manifest.json').read_text())['total_target']
        atomic_json(root/'programmatic_status.json', {'status': 'complete' if total == target else 'running',
                    'published_this_run': count, 'published_total': total, 'target': target, 'updated_unix': time.time()})
        if args.once or total == target: return
        time.sleep(10)


if __name__ == '__main__': main()
