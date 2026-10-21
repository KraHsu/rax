#!/usr/bin/env python3
"""Install the pinned native DeepSeek Harness outside the source checkout."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.deepseek_cli.installation import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.prefix, args.cache), indent=2))


if __name__ == '__main__':
    main()
