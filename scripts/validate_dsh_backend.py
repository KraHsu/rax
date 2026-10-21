#!/usr/bin/env python3
"""Exercise the real native DSH loop, provider, MCP images, files, and resume.

The MCP fixture supplies a real robot image, never a simulator or benchmark score.
This check consumes model API tokens but does not consume exploration episodes.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import socketserver
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness import agent_host
from harness.deepseek_cli import auto_research as dsh
from harness.model_serving.lifecycle import write_json
from services.controller.config import Configuration, Limits


class Fixture(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            request = json.loads(line)
            if 'id' not in request:
                continue
            method, params = request['method'], request.get('params', {})
            result = {}
            if method == 'initialize':
                result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'rax-dsh-validation', 'version': '1'}}
            elif method == 'tools/list':
                result = {'tools': [
                    {'name': 'report_number', 'description': 'Accept integer 7 and return a validation marker.',
                     'inputSchema': {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']}},
                    {'name': 'get_robot_image', 'description': 'Return the real robot scene image for this protocol test.',
                     'inputSchema': {'type': 'object', 'properties': {}}}]}
            elif method == 'tools/call':
                name = params['name']
                if name == 'report_number':
                    assert params['arguments']['value'] == 7
                    result = {'content': [{'type': 'text', 'text': 'Accepted. Validation marker: '+self.server.marker}]}
                elif name == 'get_robot_image':
                    result = {'content': [{'type': 'image', 'mimeType': 'image/jpeg', 'data': self.server.image}]}
                else:
                    raise ValueError('Unknown fixture tool')
                self.server.calls.append(name)
            self.wfile.write((json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result})+'\n').encode())
            self.wfile.flush()


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def run(config, deployed, prompt, label, timeout, *, require_thinking=True):
    home = deployed[dsh.HOME_KEY]
    socket = agent_host.socket_path(config.root)
    command = dsh.agent_command(config, deployed, prompt=prompt)
    env = dsh.agent_environment(config, deployed, socket)
    events = []
    with (config.root/f'{label}.stderr.log').open('w') as err, (config.root/f'{label}.stdout.jsonl').open('w') as out:
        child = agent_host.start(command, cwd=deployed['workspace'], env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, text=True)
        try:
            code = child.wait(timeout=timeout)
        finally:
            agent_host.stop(child)
    for line in (config.root/f'{label}.stdout.jsonl').read_text().splitlines():
        dsh.capture_event(home, line)
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    if code:
        raise RuntimeError(f'{label}: DSH exited {code}; inspect {label}.stderr.log and stdout.jsonl')
    if not any(e.get('type') == 'status' and e.get('phase') == 'turn_end'
               and (e.get('reason') or {}).get('kind') == 'completed' for e in events):
        raise RuntimeError(f'{label}: no successful native turn/end event')
    if require_thinking and not any(e.get('type') == 'thinking' for e in events):
        raise RuntimeError(f'{label}: no native reasoning event')
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--provider', choices=('openrouter','deepseek-api','deepseek-account','custom'), default='openrouter')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--auth-home', type=Path)
    parser.add_argument('--base-url')
    parser.add_argument('--model')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    limits = Limits(600, 8192, 4, 256, 1024, 10*1024**3)
    config = Configuration(root, 'click_mouse', tuple(range(16)), 0, 0, 'GPU-aaaa', limits, limits, limits,
        agent_cli='dsh', agent_provider=args.provider, agent_effort='max', benchmark='dexjoco',
        formal_episodes=150, formal_eval_seed=1, formal_workers=4, external_perception=False,
        observation_profile='rgbd', demonstration_context='terminal_state')
    deployed = dsh.deploy(config, binary=args.binary, model=args.model, key_file=args.key_file,
                          auth_home=args.auth_home, base_url=args.base_url)
    path = agent_host.socket_path(root)
    path.unlink(missing_ok=True)
    server = Server(str(path), Fixture)
    server.marker = 'RAX-'+secrets.token_hex(10)
    server.image = base64.b64encode(args.image.read_bytes()).decode()
    server.calls = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    report = {'status': 'running', 'provider': args.provider, 'model': json.loads(deployed['login'].read_text())['model'],
              'effort': 'max', 'started_at': time.time(), 'synthetic_mcp_fixture': True,
              'real_robot_image': str(args.image), 'exploration_episodes_used': 0}
    write_json(root/'validation.json', report)
    try:
        from harness.deepseek_cli.probes import boundary
        report['boundary'] = boundary(config, deployed)
        prompt = ('This is an infrastructure validation, not a robot task. Do not start any episodes. '
            'Use the report_number MCP tool with value 7. Use get_robot_image and inspect the returned image. '
            'Use your native file tools to read AGENTS.md and create output/native-file.txt containing NATIVE_OK. '
            'Use your native bash tool to run Python and write the exact marker returned by report_number '
            'to output/native-shell.txt. Finish with READY, the exact marker, and a brief description of the image. '
            'These two MCP tools are validation fixtures; no robot lifecycle tools exist here.')
        events = run(config, deployed, prompt, 'first', args.timeout)
        marker_file = deployed['workspace']/'output/native-shell.txt'
        assert marker_file.read_text().strip() == server.marker, 'Native shell did not write the real tool-result marker'
        assert (deployed['workspace']/'output/native-file.txt').read_text().strip() == 'NATIVE_OK'
        assert {'report_number','get_robot_image'} <= set(server.calls)
        report.update(tool_call=True, tool_result=True, native_file=True, native_shell=True,
                      thinking=True, session_id=dsh.latest_session(deployed[dsh.HOME_KEY]))
        deployed['resume_session'] = report['session_id']
        resumed = run(config, deployed,
            'Without using any tools or reading files, repeat the exact validation marker from our previous conversation, then say RESUMED.',
            'resume', args.timeout, require_thinking=False)
        text = ' '.join(e.get('text','') for e in resumed if e.get('type') in ('text','final'))
        assert server.marker in text and 'RESUMED' in text, 'Native persisted conversation did not resume'
        assert dsh.latest_session(deployed[dsh.HOME_KEY]) == report['session_id']
        audit = [json.loads(line) for line in (deployed[dsh.HOME_KEY]/'request-audit.jsonl').read_text().splitlines()]
        if args.provider == 'openrouter':
            assert audit and all((r.get('reasoning') or {}).get('effort') == 'max' or r.get('reasoning_effort') == 'max' for r in audit)
        else:
            assert audit and all((r.get('thinking') or {}).get('type') == 'enabled'
                                 and (r.get('output_config') or {}).get('effort') == 'max' for r in audit)
        assert any(r.get('image_blocks',0) > 0 for r in audit), 'No image reached the real provider request'
        report.update(status='passed', resume=True, vision_wire=True, wire_max_effort=True,
                      requests=len(audit), tool_calls=len(server.calls))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        server.shutdown()
        server.server_close()
        report['finished_at'] = time.time()
        write_json(root/'validation.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
