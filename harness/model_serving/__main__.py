"""python -m harness.model_serving {prepare,serve,test} --config ... --root ..."""
import argparse
import json
import time
from pathlib import Path

from .configuration import ServeConfig
from .lifecycle import prepare, prepare_image, serve, write_json
from .probes import MessagesClient, smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'image', 'serve', 'test'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--image', type=Path, help='JPEG for the vision smoke')
    parser.add_argument('--wait-seconds', type=int, default=10800)
    args = parser.parse_args()
    config = ServeConfig.load(args.config)
    if args.action == 'prepare':
        prepare(config, args.root)
    elif args.action == 'image':
        prepare_image(config, args.root)
    elif args.action == 'serve':
        if not args.key_file:
            parser.error('serve requires --key-file')
        serve(config, args.root, args.key_file)
    else:
        if not args.key_file:
            parser.error('test requires --key-file')
        deadline = time.monotonic()+args.wait_seconds
        while True:
            try:
                endpoint = json.loads((args.root/'endpoint.json').read_text())
            except (OSError, ValueError):
                endpoint = {}
            if endpoint.get('status') == 'ready':
                break
            if endpoint.get('status') == 'stopped':
                raise RuntimeError('Model service stopped before protocol validation')
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for the model endpoint')
            time.sleep(10)
        report = smoke(MessagesClient(endpoint['base_url'], args.key_file), config.served_model, image=args.image)
        write_json(args.root/'smoke.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
