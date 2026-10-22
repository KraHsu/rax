"""GPU interpreter entry point. Never install dependencies automatically."""
import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from services.robotwin.catalog import EpisodeManifest, step_limit
from services.robotwin.runtime import initialize_runtime, make_environment, native_config, source_commit
from services.robotwin.session import RoboTwinSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--collection', type=int, choices=(0, 1), required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--watch-stdin', action='store_true')
    args = parser.parse_args()
    if args.watch_stdin:
        try:
            os.setsid()
        except PermissionError:
            if os.getpgrp() != os.getpid():
                raise
        # The SSH owner holds stdin open. Disconnects also end a server still
        # warming up, before its RPC socket exists. Never signal unrelated jobs.
        def owner_disconnected():
            while os.read(sys.stdin.fileno(), 1024):
                pass
            os.killpg(os.getpid(), signal.SIGTERM)
        threading.Thread(target=owner_disconnected, daemon=True).start()
    manifest = EpisodeManifest(args.manifest, args.task)
    if args.manifest_sha256 and manifest.sha256 != args.manifest_sha256:
        raise ValueError('Local and remote episode manifests differ')
    if source_commit() != manifest.source_commit:
        raise ValueError('RoboTwin source changed since manifest preparation')
    entry = manifest.entry(args.collection, args.index)
    initialize_runtime()
    recording = None
    if args.collection == 0:
        types = native_config(args.task, args.collection)['embodiment']
        embodiment = types[0] if len(types) == 1 else '+'.join(types[:2])
        root = Path(os.environ.get('ROBOTWIN_DATASET_DIR', str(Path(args.output).parents[2] / 'dataset')))
        recording = root / args.task / f'{embodiment}_clean_50' / 'data' / f'episode_{Path(args.output).parent.name}.hdf5'
    session = RoboTwinSession(lambda: make_environment(args.task, args.collection, entry, episode_index=args.index),
                             task=args.task, instruction=entry['instruction'], output=args.output,
                             native_step_limit=step_limit(args.task), recording_path=recording, provenance={
                                 'manifest_sha256': manifest.sha256, 'source_commit': manifest.source_commit,
                                 'collection': args.collection, 'index': args.index, 'seed': entry['seed']})
    from services.simulator.rpc import serve
    try:
        serve(session, args.port)
    finally:
        session.close()


if __name__ == '__main__':
    main()
