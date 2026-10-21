#!/usr/bin/env python3
"""Render a portable Slurm serving pipeline; --submit explicitly starts its jobs."""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness.model_serving.configuration import ServeConfig
from harness.model_serving.slurm import submit_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/serving/deepseek-v41-flash-1017.json')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--openrouter-key-file', type=Path, help='Optional; the private-model experiment needs no cloud key')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    ServeConfig.load(args.config)
    from harness.claude_cli.login import private_text
    for key in filter(None, (args.key_file, args.openrouter_key_file)):
        if not private_text(key).strip():
            parser.error('Key files must contain a credential')
    print(json.dumps(submit_pipeline(project=PROJECT, root=args.root, config=args.config,
        profile=args.profile, key_file=args.key_file, openrouter_key_file=args.openrouter_key_file,
        image=args.image, submit=args.submit), indent=2))


if __name__ == '__main__':
    main()
