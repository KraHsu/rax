#!/usr/bin/env python3
"""Export a compact offline review or embed checked RAX annotations on the data host."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.robotwin.annotation import export_review, embed_review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--episode', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    embed = commands.add_parser('embed')
    embed.add_argument('--episode', type=Path, required=True)
    embed.add_argument('--proposal', type=Path, required=True)
    embed.add_argument('--agent-run', required=True)
    a = p.parse_args()
    if a.command == 'export':
        result = export_review(a.episode, a.output)
        result = {k: result[k] for k in ('source_sha256','recording_id','frame_count','gripper_events')}
    else:
        result = embed_review(a.episode, json.loads(a.proposal.read_text()), agent_run=a.agent_run)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
