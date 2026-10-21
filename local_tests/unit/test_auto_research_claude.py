"""Actual native Claude Code, its sandbox and the robot MCP tools against a local fake Messages endpoint."""
import json
import shlex
import shutil
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_auto_research_codex import secret_probes
from test_bundle_sandbox import sandbox_runtime  # noqa: F401 (fixture)
from test_controller_backend import configuration

from harness import agent_host
from harness.claude_cli import auto_research as claude
from harness.claude_cli.login import PROVIDERS, Credential
from harness.codex_cli.auto_research import PROJECT


@pytest.mark.parametrize('model', ['deepseek/deepseek-v4.1-flash-1017',
                                 'qwen/qwen3-coder'])
def test_non_claude_model_routes_auxiliary_requests_to_same_model(tmp_path, model):
    key = tmp_path/'key'
    key.write_text('fixture-private-secret')
    key.chmod(0o600)
    environment = Credential('openrouter', key_file=key).environment(model=model)
    assert environment['ANTHROPIC_BASE_URL'] == 'https://openrouter.ai/api'
    assert environment['ANTHROPIC_AUTH_TOKEN'] == 'fixture-private-secret'
    assert environment['ANTHROPIC_API_KEY'] == ''
    for name in ('ANTHROPIC_DEFAULT_OPUS_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL',
                 'ANTHROPIC_DEFAULT_HAIKU_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL'):
        assert environment[name] == model


@pytest.mark.parametrize('model', [None, 'anthropic/claude-opus-5.5'])
def test_claude_model_keeps_provider_auxiliary_models(tmp_path, model):
    key = tmp_path/'key'
    key.write_text('fixture-private-secret')
    key.chmod(0o600)
    environment = Credential('openrouter', key_file=key).environment(model=model)
    assert environment['ANTHROPIC_DEFAULT_SONNET_MODEL'] == PROVIDERS['openrouter']['auditor']
    assert environment['ANTHROPIC_DEFAULT_HAIKU_MODEL'] == PROVIDERS['openrouter']['fast']
    assert 'CLAUDE_CODE_SUBAGENT_MODEL' not in environment

# Run by the agent's Bash tool, inside Claude Code's sandbox: what its commands can reach.
PROBE = r'''
import importlib.util, json, os, socket
from pathlib import Path
WORKSPACE = Path(WORKSPACE_PATH)
seen = {'cwd': str(Path.cwd()), 'pythonpath': os.environ.get('PYTHONPATH')}
def readable(path):
    try:
        Path(path).read_bytes()
        return True
    except OSError:
        return False
def writable(path):
    try:
        Path(path).write_text('x')
        return True
    except OSError:
        return False
seen['readable'] = [path for path in HIDDEN if readable(path)]
seen['writable'] = [str(path) for path in (WORKSPACE/'.claude/settings.json', WORKSPACE/'.mcp.json')
                    if writable(path)]
# Hidden directories are empty in-memory mounts: a write there lands in the sandbox, never on the host.
seen['hidden_write'] = writable(WORKSPACE.parent/'escape.txt')
seen['hidden_listing'] = sorted(os.listdir(WORKSPACE.parent))
seen['secrets'] = sorted(name for name in os.environ if any(word in name for word in ('KEY', 'TOKEN', 'SECRET')))
seen['gpus'] = [str(device) for device in Path('/dev').glob('nvidia*') if readable(device)]
try:
    socket.create_connection(('1.1.1.1', 443), timeout=.5).close()
    seen['internet'] = True
except OSError:
    seen['internet'] = False
from api.runtime import Context
with Context() as ctx:
    seen['status'] = 'manual_success_confirmed' in ctx.call('status')['content'][0]['text']
seen['api'] = importlib.util.find_spec('api.runtime').origin
Path('probe.json').write_text(json.dumps(seen))
print('PROBE-DONE')
'''


def sse(blocks, stop, ident):
    """One streamed Messages response (Claude Code merges assistant messages that share an id)."""
    events = [('message_start', {'type': 'message_start', 'message': {
        'id': f'msg_local_{ident}', 'type': 'message', 'role': 'assistant', 'model': 'local', 'content': [],
        'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 1, 'output_tokens': 1}}})]
    for index, block in enumerate(blocks):
        if block['type'] == 'tool_use':
            start, delta = {**block, 'input': {}}, {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
        else:
            start, delta = {'type': 'text', 'text': ''}, {'type': 'text_delta', 'text': block['text']}
        events += [('content_block_start', {'type': 'content_block_start', 'index': index, 'content_block': start}),
                   ('content_block_delta', {'type': 'content_block_delta', 'index': index, 'delta': delta}),
                   ('content_block_stop', {'type': 'content_block_stop', 'index': index})]
    events += [('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': stop, 'stop_sequence': None},
                                  'usage': {'output_tokens': 1}}),
               ('message_stop', {'type': 'message_stop'})]
    return ''.join(f'event: {name}\ndata: {json.dumps(data)}\n\n' for name, data in events).encode()


@pytest.mark.parametrize('provider,model,effort', [
    ('openrouter', None, 'xhigh'),
    ('openrouter', 'deepseek/deepseek-v4.1-flash-1017', 'max'),
    ('custom', 'deepseek-flash', 'max')])
def test_real_native_claude_code_sandbox_and_robot_tools(sandbox_runtime, monkeypatch, provider, model, effort):
    binary = shutil.which('claude')
    if not binary:
        pytest.skip('Claude Code is not installed')
    if not (shutil.which('bwrap') and shutil.which('socat')):
        pytest.skip("Claude Code's sandbox needs bubblewrap and socat")
    monkeypatch.setattr(claude, 'CALL_SECONDS', 120)
    root = sandbox_runtime
    config = replace(configuration(root), agent_effort=effort)
    workspace = config.root/'agent-workspace'
    secret = root/'test-key'
    secret.write_text('fixture-private-secret')
    secret.chmod(0o600)
    probe = PROBE.replace('WORKSPACE_PATH', repr(str(workspace))).replace('HIDDEN', repr([
        str(PROJECT/'AGENTS.md'), str(secret), str(config.root/'claude-home/settings.json'),
        str(config.root/'configuration.json'), *secret_probes()]))
    agent_requests, other_requests = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body, kind='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            self.reply(200, b'')

        def do_GET(self):
            self.reply(200, b'{}')

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
            if not self.path.startswith('/v1/messages') or self.path.startswith('/v1/messages/count_tokens'):
                other_requests.append(self.path)
                return self.reply(200, json.dumps({'input_tokens': 1}).encode())
            tools = {tool.get('name') for tool in request.get('tools', [])}
            if 'Bash' not in tools:  # Claude Code's own side requests (titles, summaries).
                other_requests.append(self.path)
                return self.reply(200, sse([{'type': 'text', 'text': 'ok'}], 'end_turn', f'side{len(other_requests)}'),
                                  'text/event-stream')
            assert self.headers.get('Authorization') == 'Bearer fixture-private-secret'
            agent_requests.append(request)
            turns = sum(message['role'] == 'assistant' for message in request['messages'])
            if turns == 0:
                blocks = [{'type': 'tool_use', 'id': 'toolu_probe', 'name': 'Bash',
                           'input': {'command': 'python -c ' + shlex.quote(probe), 'description': 'probe'}}]
            elif turns == 1:
                blocks = [{'type': 'tool_use', 'id': 'toolu_status', 'name': f'mcp__{claude.MCP_SERVER}__status',
                           'input': {}}]
            else:
                blocks = [{'type': 'text', 'text': 'Local check complete.'}]
            self.reply(200, sse(blocks, 'tool_use' if turns < 2 else 'end_turn', len(agent_requests)), 'text/event-stream')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setitem(PROVIDERS, 'openrouter', {**PROVIDERS['openrouter'],
                                                  'base_url': f'http://127.0.0.1:{server.server_port}'})
    try:
        deployed = claude.deploy(config, provider=provider, key_file=secret, claude=binary, model=model,
                                base_url=f'http://127.0.0.1:{server.server_port}' if provider == 'custom' else None)
        if provider == 'custom':
            assert json.loads(deployed['login'].read_text())['base_url'] == f'http://127.0.0.1:{server.server_port}'
        assert deployed['workspace'] == workspace
        assert 'fixture-private-secret' not in (deployed['claude_home']/'settings.json').read_text()
        from services.controller.sandbox import deadline
        with deadline(180):  # Sessions have no wall clock: the fixture bounds its own run.
            result = claude.launch(config, deployed, prompt='Local infrastructure check. Do not start a robot episode.',
                                   capture=True)
    finally:
        server.shutdown()
        server.server_close()
    (root/'claude-stdout.jsonl').write_text(result.stdout)
    (root/'claude-stderr.log').write_text(result.stderr)
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    final = [event for event in events if event.get('type') == 'result']
    assert result.returncode == 0, (final or result.stdout[-3000:], result.stderr[-3000:])
    assert len(agent_requests) == 3, (result.stdout[-3000:], other_requests)
    if model:
        assert all(req['model'] == model for req in agent_requests)
        assert all(req.get('output_config', {}).get('effort') == 'max' for req in agent_requests), [
            {'model': req['model'], 'thinking': req.get('thinking'), 'output_config': req.get('output_config')}
            for req in agent_requests]
    results = [block for message in agent_requests[-1]['messages'] if message['role'] == 'user'
               and isinstance(message['content'], list) for block in message['content']
               if block.get('type') == 'tool_result']
    assert [block['tool_use_id'] for block in results] == ['toolu_probe', 'toolu_status']
    assert 'PROBE-DONE' in json.dumps(results[0]), results[0]
    assert 'manual_success_confirmed' in json.dumps(results[1]) and not results[1].get('is_error')
    seen = json.loads((workspace/'probe.json').read_text())
    (root/'probe.json').write_text(json.dumps(seen, indent=1))
    assert seen['cwd'] == str(workspace.resolve()) and seen['pythonpath'] == str(workspace.resolve())
    assert seen['api'] == str(workspace.resolve()/'api/runtime.py') and seen['status']
    # Claude Code's sandbox: the repository, the session's private files and the operator's
    # credentials are unreadable, writes stay in the workspace (not its Claude settings), no
    # internet, no GPU.
    assert seen['readable'] == [] and seen['writable'] == []
    assert set(seen['hidden_listing']) <= {'agent-workspace', 'published', 'escape.txt'}  # Only what it may see.
    assert not (config.root/'escape.txt').exists() and not (workspace/'.mcp.json').exists()
    assert not seen['internet'] and seen['gpus'] == []
    # As with native Claude Code, its own login reaches the agent's commands; nothing of the
    # session's (the Gemini key, the operator's variables) does.
    assert {'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_API_KEY'} <= set(seen['secrets']), seen['secrets']
    assert not [name for name in seen['secrets'] if name.startswith(('RAX', 'OPENROUTER', 'GEMINI', 'GOOGLE'))]
    # The robot tools the model saw: the session's, no web tools.
    tools = {tool['name'] for tool in agent_requests[0]['tools']}
    # Claude may advertise deferred discovery before individual MCP definitions.
    # The successful MCP status round trip above verifies the live server.
    assert 'Bash' in tools
    assert (f'mcp__{claude.MCP_SERVER}__start_episode' in tools or
            {'ToolSearch', 'DeferredToolPlaceholder'} & tools)
    assert not {'WebFetch', 'WebSearch'} & tools
    state = json.loads((config.root/'state.json').read_text())
    assert state['exploration_started'] == 0
