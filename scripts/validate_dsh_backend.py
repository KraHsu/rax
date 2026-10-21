#!/usr/bin/env python3
"""Check DeepSeek Harness with the benchmark-independent MCP fixture."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_native_cli import main


if __name__ == '__main__':
    sys.argv[1:1] = ['--backend', 'dsh', '--fixture-workspace']
    main()
