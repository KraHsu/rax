#!/usr/bin/env python3
"""Atomically upgrade published demonstrations to complete direction-aware coverage."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from services.robotwin.annotation import digest, embed_review
from services.robotwin.collection import atomic_json
from services.robotwin.constraint_coverage import coverage_report
from services.robotwin.constraint_schema import read_embedded
from services.robotwin.full_trajectory import POLICY


def upgrade(path):
    with h5py.File(path, 'r') as h:
        bundle = read_embedded(h)
        report = coverage_report(bundle['constraints'], len(h['endpose/left_endpose']))
        stored = json.loads(h['trajectory_constraints/coverage'][()]) if 'trajectory_constraints/coverage' in h else {}
        if report['complete'] and stored.get('policy') == POLICY:
            return None
    return embed_review(path, {'source_sha256': digest(path), 'constraints': bundle['constraints']},
                        agent_run='full-motion-coverage-v1')


def run(root):
    root = Path(root)
    lock = (root/'.coverage_backfill.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = sorted(root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5'))
    status = {'status': 'running', 'snapshot_published': len(paths), 'checked': 0, 'updated': 0,
              'already_complete': 0, 'failures': [], 'started_unix': time.time()}
    for path in paths:
        try:
            result = upgrade(path)
            if result is None: status['already_complete'] += 1
            else:
                receipt_path = path.parent.parent/'annotation_queue'/f'{path.stem}.json'
                if receipt_path.exists():
                    with (root/'.annotation_queue.lock').open('a') as queue_lock:
                        fcntl.flock(queue_lock, fcntl.LOCK_EX)
                        receipt = json.loads(receipt_path.read_text())
                        if receipt['annotation_status'] == 'done':
                            receipt.update(annotation_result=result, coverage_updated_unix=time.time())
                            atomic_json(receipt_path, receipt)
                status['updated'] += 1
        except Exception as exc:
            status['failures'].append({'episode': str(path), 'error': f'{type(exc).__name__}: {exc}'})
        status.update(checked=status['checked']+1, updated_unix=time.time())
        atomic_json(root/'coverage_backfill_status.json', status)
        if status['checked'] % 25 == 0: print(json.dumps(status), flush=True)
    status.update(status='failed' if status['failures'] else 'complete', finished_unix=time.time())
    atomic_json(root/'coverage_backfill_status.json', status)
    print(json.dumps(status), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); result = run(args.root.resolve())
    raise SystemExit(bool(result['failures']))
