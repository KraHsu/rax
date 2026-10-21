"""Run Claude Code as the auto-research agent in Docker, like the Codex launcher.

The container is the same (offline, read-only root, capped workspace, no GPU); the
host keeps the robot MCP frontend and a Messages relay that holds the provider
credential (messages_relay.py). Selected by ``agent_cli: claude`` in the operator
configuration; everything else about a session is shared with the Codex launcher.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from harness.claude_cli.messages_relay import Credential, MessagesRelay, PROVIDERS
from harness.codex_cli import workspace_storage
from harness.codex_cli.auto_research import (CALL_SECONDS, PROJECT, frontend_failure, native_binary,
                                             open_frontend_log, prepare_workspace, published_mounts, refresh_reviews,
                                             relay_slots, start_frontend, stop_services)
from services.controller.sandbox import DockerSandbox, thread_env

HOME_NAME = 'claude-home'
DENIED_TOOLS = ('WebFetch', 'WebSearch')
SUBAGENT_TOOLS = ('Agent', 'Task')  # 'Task' is the tool's former name.
# Default reasoning effort when a deployment predates agent_effort (Claude Code's own
# default is high); sessions select xhigh or max like Codex's model_reasoning_effort.
EFFORT = 'xhigh'
RELAY_TOKEN = 'host-relay'  # Placeholder the relay replaces; not a credential.


def claude_binary(path=None):
    return native_binary(path, 'claude')


def claude_workspace(workspace):
    """Claude Code reads CLAUDE.md and .claude/skills; the template keeps AGENTS.md and
    .agents/skills, shared with Codex."""
    (workspace/'CLAUDE.md').write_text('@AGENTS.md\n')
    (workspace/'.claude').mkdir(exist_ok=True)
    (workspace/'.claude/skills').symlink_to('../.agents/skills', target_is_directory=True)


def preflight_login():
    """claude-login: before anything starts, check that the experiment token is accepted.
    A network failure is reported and does not block: the relay passes any real provider
    error to the agent."""
    import http.client
    from harness.claude_cli import experiment_login
    try:
        return experiment_login.check()
    except (OSError, http.client.HTTPException) as exc:
        print(f'Could not verify the Claude experiment token ({type(exc).__name__}: {exc}); continuing',
              file=sys.stderr)
        return None


COMPACT_WINDOW = 250_000


def deploy(config, *, provider='openrouter', key_file=None, model=None, claude=None, base_url=None):
    """Prepare a new SSD workspace for Claude Code without starting a model or simulator."""
    binary = claude_binary(claude)
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root/'state.json').exists() or (root/'agent-workspace').exists():
        raise FileExistsError('Use a fresh auto-research trial root')
    model = model or PROVIDERS[provider]['model']
    auditor, fast = PROVIDERS[provider]['auditor'], PROVIDERS[provider]['fast']
    if not model.lstrip('~').startswith(('anthropic/', 'claude-')):
        auditor = fast = model
    account = None
    if provider == 'claude-login' and key_file is None:
        # The dedicated experiment token (never the personal ~/.claude login), which the relay
        # re-reads per request, so rotating it applies at once.
        from harness.claude_cli import experiment_login
        experiment_login.status()  # The token exists and is usable.
        key_file, account = experiment_login.token_file(), experiment_login.identity()
    Credential(provider, key_file=key_file, base_url=base_url)  # Validate before provisioning.
    # Only paths and names: the relay reads the credential itself at each request.
    relay_file = root/'model-relay.json'
    relay_file.write_text(json.dumps({
        'agent_cli': 'claude', 'provider': provider, 'base_url': base_url or PROVIDERS[provider]['base_url'], 'model': model,
        'models': sorted({model, auditor, fast}),
        'key_file': str(key_file) if key_file else None,
        # Identity only (no token): which dedicated account this session bills.
        'account': {k: account.get(k) for k in ('email', 'organization')} if account else None}))
    relay_file.chmod(0o600)
    prepared = prepare_workspace(config, HOME_NAME)
    workspace, home = prepared['workspace'], prepared['home']
    claude_workspace(workspace)
    timeout_ms = (CALL_SECONDS+300)*1000
    settings = {
        'permissions': {'defaultMode': 'bypassPermissions', 'deny': list(DENIED_TOOLS)},
        'skipDangerousModePermissionPrompt': True,
        'autoUpdates': False, 'includeCoAuthoredBy': False, 'cleanupPeriodDays': 3650,
        'env': {
            'ANTHROPIC_BASE_URL': 'http://127.0.0.1:17777', 'ANTHROPIC_AUTH_TOKEN': RELAY_TOKEN,
            # Subagents inherit the agent's model unless spawned with model "sonnet"
            # (a cheaper model for review-style subagents) or "haiku".
            'ANTHROPIC_MODEL': model, 'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
            'ANTHROPIC_DEFAULT_SONNET_MODEL': auditor, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': fast,
            # A formal batch pauses the container; the host relay keeps provider I/O alive.
            'API_TIMEOUT_MS': str(timeout_ms), 'MCP_TOOL_TIMEOUT': str(timeout_ms), 'MCP_TIMEOUT': '60000',
            'BASH_DEFAULT_TIMEOUT_MS': '600000', 'BASH_MAX_TIMEOUT_MS': str(CALL_SECONDS*1000),
            'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY': '1',
            # Every MCP tool stays a local tool definition (the relay refuses server tools).
            'ENABLE_TOOL_SEARCH': 'false'}}
    (home/'settings.json').write_text(json.dumps(settings, indent=2))
    (home/'.claude.json').write_text(json.dumps({
        'hasCompletedOnboarding': True, 'bypassPermissionsModeAccepted': True,
        'projects': {'/workspace': {'hasTrustDialogAccepted': True, 'hasCompletedProjectOnboarding': True}}}))
    (home/'mcp.json').write_text(json.dumps({'mcpServers': {'auto_research': {
        'type': 'stdio', 'command': '/usr/local/bin/python', 'args': ['/opt/agent/container_bridge.py', 'mcp']}}}))
    resume_session = None
    if config.adaptation_source:
        from services.adaptation import import_source
        resume_session = import_source(config, workspace, home)
    return {'workspace': workspace, 'claude_home': home, 'claude': binary, 'model': model,
            'configuration': prepared['configuration'], 'relay_configuration': relay_file,
            'bridge': prepared['bridge'], 'subagents': subagent_cap(config), 'resume_session': resume_session}


def latest_claude_session(home):
    """Id of the newest top-level Claude Code conversation (subagent transcripts live below it)."""
    transcripts = sorted(Path(home, 'projects', '-workspace').glob('*.jsonl'), key=lambda p: p.stat().st_mtime)
    if not transcripts:
        raise RuntimeError('No Claude Code conversation to resume in '+str(home))
    return transcripts[-1].stem


latest_session = latest_claude_session


def last_error(home):
    """The newest API error Claude Code recorded in its newest conversation, whole, or None."""
    transcripts = sorted(Path(home, 'projects', '-workspace').glob('*.jsonl'), key=lambda p: p.stat().st_mtime)
    if not transcripts:
        return None
    error = None
    with transcripts[-1].open(errors='replace') as stream:
        for line in stream:
            if '"isApiErrorMessage":true' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            content = (row.get('message') or {}).get('content')
            text = ' '.join(block.get('text', '') for block in content if isinstance(block, dict)) \
                if isinstance(content, list) else str(content or '')
            error = text or json.dumps({k: row.get(k) for k in ('error', 'apiErrorStatus')})
    return error


def redeploy(config, claude=None):
    """Reattach an existing Claude Code session workspace for a resumed launch."""
    root = config.root
    workspace_storage.remount(root)
    relay_file = root/'model-relay.json'
    deployed = {'workspace': root/'agent-workspace', 'claude_home': root/HOME_NAME, 'claude': claude_binary(claude),
                'configuration': root/'configuration.json', 'relay_configuration': relay_file,
                'bridge': root/'container_bridge.py'}
    for key in ('workspace', 'claude_home', 'configuration', 'relay_configuration', 'bridge'):
        if not deployed[key].exists():
            raise FileNotFoundError(f'Cannot resume: {deployed[key]} is missing')
    deployed['model'] = json.loads(relay_file.read_text())['model']
    deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
    deployed['subagents'] = subagent_cap(config)
    refresh_reviews(config, deployed['workspace'])
    return deployed


def subagent_cap(config):
    from services.mcp_contract import Contract
    return Contract.from_config(config).subagent_cap


def claude_arguments(deployed, *, prompt, interactive=False, json_output=False):
    denied = DENIED_TOOLS
    if deployed.get('subagents', 1) == 0:
        # Claude Code has no concurrency setting; the cap is stated in the workspace
        # and measured afterwards, but a zero cap removes the subagent tool entirely.
        denied += SUBAGENT_TOOLS
    args = ['--model', deployed['model'], '--effort', deployed.get('effort', EFFORT), '--mcp-config', '/claude-home/mcp.json', '--strict-mcp-config',
            '--permission-mode', 'bypassPermissions', '--disallowedTools', *denied]
    if deployed.get('resume_session'):
        args += ['--resume', deployed['resume_session']]  # Continue the session's own conversation.
    if not interactive:
        args = ['-p', *args, '--output-format', 'stream-json' if json_output else 'text', '--verbose']
    return [*args, '--', prompt]


def container_command(config, deployed, relay_directory, name, *, prompt, interactive=False, json_output=False):
    if os.getuid() == 0:
        raise RuntimeError('Run the launcher as a non-root operator')
    lim = config.training
    command = ['docker', 'run', '--pull=never', '--name', name, '--interactive',
        '--network=none', '--read-only', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cap-drop=ALL', '--security-opt=no-new-privileges', '--ipc=private', '--init',
        '--pids-limit', str(lim.pids), '--memory', f'{lim.memory_mb}m', '--memory-swap', f'{lim.memory_mb}m',
        '--cpus', str(lim.cpus), '--ulimit', 'core=0', '--log-driver=none',
        '--tmpfs', f'/tmp:rw,nosuid,nodev,size={lim.scratch_mb}m,mode=1777',
        # HOME must differ from the project directory: Claude Code skips project skills
        # (.claude/skills) when the working directory is the home directory.
        '--workdir', '/workspace', '--env', 'HOME=/claude-home', '--env', 'CLAUDE_CONFIG_DIR=/claude-home',
        '--env', 'AGENT_BINARY=/opt/claude/claude',
        '--env', 'PIP_NO_INDEX=1', '--env', 'HF_HUB_OFFLINE=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'PYTHONPATH=/workspace',
        '--env', 'PATH=/opt/claude:/usr/local/bin:/usr/bin:/bin', '--env', 'TERM=xterm-256color',
        # Set per launch (not in settings.json, written once at deploy) so resumed sessions get them too.
        # Print mode otherwise stops background subagents 600 s after the main turn ends.
        '--env', 'CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0',
        # Compact near COMPACT_WINDOW tokens instead of the model's 1M window: every call
        # re-reads the whole context, and an expired cache re-writes all of it.
        '--env', f'CLAUDE_CODE_AUTO_COMPACT_WINDOW={COMPACT_WINDOW}',
        # One-hour prompt cache (writes cost 2x base instead of 1.25x): agents often wait 5-30
        # minutes on rehearsals and episodes, and a lapsed 5-minute cache re-writes the context.
        '--env', 'ENABLE_PROMPT_CACHING_1H=1', '--env', 'CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL=1h',
        # The agent CLI shares --pids-limit with everything the agent runs.
        *thread_env(lim.cpus)]
    if interactive:
        command.append('--tty')
    mounts = [(deployed['workspace'], '/workspace', False), (deployed['claude_home'], '/claude-home', False),
        (deployed['claude_home']/'settings.json', '/claude-home/settings.json', True),
        (deployed['claude_home']/'mcp.json', '/claude-home/mcp.json', True),
        *published_mounts(config, deployed),
        (relay_directory, '/run/agent-relay', True), (deployed['bridge'], '/opt/agent/container_bridge.py', True),
        (deployed['claude'], '/opt/claude/claude', True)]
    for source, destination, readonly in mounts:
        if any(c in str(source) for c in (',', '\n')):
            raise ValueError('Invalid mount path')
        command += ['--mount', f'type=bind,src={source},dst={destination}'+(',readonly' if readonly else '')]
    for variable in deployed.get('container_env', ()):
        command += ['--env', variable]
    command += ['--entrypoint', '/usr/local/bin/python', config.image, '/opt/agent/container_bridge.py',
                *claude_arguments({'effort': getattr(config, 'agent_effort', EFFORT), **deployed}, prompt=prompt, interactive=interactive, json_output=json_output)]
    return command


def relay_credential(deployed):
    relay = json.loads(deployed['relay_configuration'].read_text())
    if relay.get('credentials_file'):
        raise RuntimeError('This session was prepared with the personal Claude login, which experiments no '
                           'longer use; prepare a new session (claude-login uses the experiment token)')
    return relay, Credential(relay['provider'], key_file=relay['key_file'], base_url=relay.get('base_url'))


def model_relay(config, deployed, relay_dir, *, calls_file=None, seconds, usage_log=None):
    """Start the host Messages relay whose socket the container bridge forwards to."""
    relay, credential = relay_credential(deployed)
    model = MessagesRelay(relay_dir/'model.sock', credential, base_url=relay['base_url'],
                          models=relay.get('models') or [relay['model']],
                          calls_file=calls_file, client_write_timeout=seconds, usage_log=usage_log,
                          slots=relay_slots(config), allow_private_http=relay['provider'] == 'custom')
    model.start()
    return model


def session_files(home, session_id):
    """Transcript files that resume a Claude conversation in a fresh home, relative to it."""
    transcript = Path(home, 'projects', '-workspace', session_id+'.jsonl')
    if not transcript.is_file():
        raise RuntimeError(f'Claude conversation {session_id} not found in {home}')
    return [transcript.relative_to(home)]


HOME_KEY = 'claude_home'
# Trusted launcher-written configuration a copied session needs (never credentials).
HOME_FILES = ('settings.json', 'mcp.json', '.claude.json')


USAGE_PROMPT = ("The model usage limit interrupted this session; it has now been resumed and any background "
                "workers were stopped. Call status, reread memory/PROGRESS.md, and continue from the live state.")
MAX_USAGE_WAITS = 10
USAGE_STOP_SECONDS = 600  # Let Claude Code end its own turn first; then stop the container.


def watch(agent, model, name, *, stop=True):
    """Wait for the agent; stop it once the usage window has been exhausted for a while."""
    limited = None
    while agent.poll() is None:
        if stop and model.limited_until > time.time():
            limited = limited or time.monotonic()
            if time.monotonic()-limited > USAGE_STOP_SECONDS:
                subprocess.run(['docker', 'stop', '--time', '30', name], capture_output=True, timeout=90)
        time.sleep(1)


def until_usage_resets(run, model, *, deployed, prompt, interactive, on_wait=None, sleep=time.sleep):
    """Run the agent. If the subscription usage window ran out during a run, wait for the reset
    (the host frontend, relay and live episodes stay up), then continue the agent's own conversation."""
    waited, waits = 0.0, 0
    while True:
        run_started = time.time()
        returncode = run(prompt)
        if interactive or model.limited_until <= run_started or waits >= MAX_USAGE_WAITS:
            return returncode
        pause = max(0.0, model.limited_until+60-time.time())
        if on_wait and pause:
            on_wait(model.limited_until, waited)
        sleep(pause)
        waited, waits = waited+pause, waits+1
        if on_wait:
            on_wait(None, waited)
        deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
        prompt = USAGE_PROMPT


def launch(config, deployed, *, gemini_key_file=None, prompt, interactive=False, capture=False, calls_file=None,
           on_wait=None, agent_runs=None):
    """Host owns both relays and cleanup; Docker owns every agent-executed command.

    The relays and the frontend (the sole owner of the session's episodes) stay up for every
    agent run of this launch: `agent_runs(run)` may call `run(prompt)` again after a run ends (a
    finished turn, an API error), and the open episodes keep running in between."""
    DockerSandbox(config.image, config.training).preflight()
    workspace_storage.validate(config.root, config.workspace_mb, deployed['workspace'], deployed['claude_home'])
    relay_credential(deployed)  # Fail before starting anything.
    name = 'rax-agent-'+uuid.uuid4().hex
    server = model = agent = None
    with tempfile.TemporaryDirectory(prefix='rax-relay-') as temporary:
        relay_dir = Path(temporary)
        with open_frontend_log(config.root) as log:
            try:
                model = model_relay(config, deployed, relay_dir, calls_file=calls_file, seconds=CALL_SECONDS+300,
                                    usage_log=config.root/'model-usage.jsonl')
                args = [sys.executable, '-m', 'services.controller.frontend', '--config', str(deployed['configuration']),
                        '--workspace', str(deployed['workspace']), '--socket', str(relay_dir/'mcp.sock'),
                        '--agent-container', name]
                if gemini_key_file:  # gemini_generate stays brokered by the host frontend.
                    args += ['--key-file', str(gemini_key_file)]
                server = start_frontend(args, log)
                deadline = time.monotonic()+30
                while not (relay_dir/'mcp.sock').exists():
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise frontend_failure(server, log, 30)
                    time.sleep(.05)
                if capture:
                    command = container_command(config, deployed, relay_dir, name, prompt=prompt,
                                                interactive=interactive, json_output=True)
                    agent = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    stdout, stderr = agent.communicate()
                    return subprocess.CompletedProcess(command, agent.returncode, stdout, stderr)
                def run(prompt):
                    nonlocal agent
                    subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
                    command = container_command(config, deployed, relay_dir, name, prompt=prompt, interactive=interactive)
                    agent = subprocess.Popen(command)
                    watch(agent, model, name, stop=not interactive)
                    return agent.returncode
                def waited(prompt):
                    return until_usage_resets(run, model, deployed=deployed, prompt=prompt, interactive=interactive,
                                              on_wait=on_wait)
                returncode = agent_runs(waited) if agent_runs else waited(prompt)
                return subprocess.CompletedProcess(agent.args, returncode, None, None)
            finally:
                stop_services(name, agent, server, model)
