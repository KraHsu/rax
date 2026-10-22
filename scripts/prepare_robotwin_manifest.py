#!/usr/bin/env python3
"""Select native-feasible clean/formal scenes on an existing GPU environment."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.robotwin.catalog import COLLECTIONS, EpisodeManifest
from services.robotwin.runtime import initialize_runtime, make_environment, source_commit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clean-episodes', type=int, default=5)
    p.add_argument('--formal-episodes', type=int, default=100)
    p.add_argument('--clean-start', type=int, default=0)
    p.add_argument('--formal-start', type=int, default=100000)
    p.add_argument('--max-attempts', type=int, default=2000)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; immutable manifests are never overwritten')
    if min(args.clean_episodes, args.formal_episodes, args.max_attempts) < 1:
        p.error('Episode counts and attempt limit must be positive')
    initialize_runtime()
    from generate_episode_instructions import generate_episode_descriptions
    result = {'schema_version': 1, 'task': args.task, 'selection': 'native_expert_success',
              'source_commit': source_commit(), 'collections': {}}
    for collection, count, start in ((0, args.clean_episodes, args.clean_start),
                                      (1, args.formal_episodes, args.formal_start)):
        selected = []
        for seed in range(start, start+args.max_attempts):
            native = None
            try:
                entry = {'seed': seed, 'instruction': 'manifest preparation', 'task_config': COLLECTIONS[collection]}
                native = make_environment(args.task, collection, entry, episode_index=len(selected))
                info = native.play_once()
                if not (native.plan_success and native.check_success()):
                    continue
                random.seed(seed)
                descriptions = generate_episode_descriptions(args.task, [info['info']], 1)[0]
                pool = descriptions['seen' if collection == 0 else 'unseen']
                entry['instruction'] = random.Random(seed).choice(pool)
                selected.append(entry)
                print(json.dumps({'collection': collection, 'accepted': len(selected), 'seed': seed}), flush=True)
                if len(selected) == count:
                    break
            except Exception as exc:
                print(json.dumps({'seed': seed, 'skipped': str(exc)}), flush=True)
            finally:
                if native is not None:
                    native.close_env()
        if len(selected) != count:
            raise RuntimeError(f'Not enough feasible scenes in collection {collection}; no manifest published')
        result['collections'][str(collection)] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    EpisodeManifest(args.output, args.task)


if __name__ == '__main__':
    main()
