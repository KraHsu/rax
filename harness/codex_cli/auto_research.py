"""Run the complete agent (Codex, or Claude Code via harness/claude_cli) in Docker; keep
robot/API services on the host."""
from contextlib import contextmanager
from dataclasses import asdict
import argparse
import fcntl
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from urllib.parse import urlsplit
import uuid

from harness.codex_cli.configuration import _write_config, ROBOT_STATUS_LINE, subagent_settings
from harness.codex_cli.model_relay import ModelRelay, MODEL, chatgpt_headers, slot_count
from harness.codex_cli import workspace_storage
from services.controller.config import Configuration
from services.controller.demonstrations import provision_demonstration, task_demonstration_context
from services.controller.gemini import load_key
from services.controller.sandbox import DockerSandbox, thread_env
from services.controller.storage import persist
from services.benchmarks import for_config

PROJECT = Path(__file__).resolve().parents[2]
# Sessions have no wall-clock or model-call budget; only each episode's own limits end it.
# One agent tool or model call times out after this, below the 2**31-1 ms (24.8-day) limit
# of the Claude CLI's Node timers.
CALL_SECONDS = 20*24*3600
DISABLED_FEATURES = ('apps', 'plugins', 'multi_agent', 'multi_agent_v2', 'browser_use',
    'browser_use_external', 'computer_use', 'image_generation', 'artifact',
    'skill_mcp_dependency_install', 'hooks', 'enable_request_compression')


def populate_workspace(config, workspace, contract=None, *, formal_episode=False):
    """Template, skills, TASK.md and demonstrations for a new agent workspace.

    formal_episode: the workspace of an agent formal copy (services/agent_formal), whose
    instructions describe its one live episode instead of exploration. With agent formal
    on (live-copy mode) no text frames the work as an evaluation.
    """
    for name in ('code', 'memory', 'output', 'runtime/autonomous_controller'):
        (workspace/name).mkdir(parents=True, exist_ok=True)
    from services.mcp_contract import Contract
    from services.workspace import Setup, build, recipe_for
    if formal_episode:
        options = frozenset({'formal-episode'})
    else:
        options = frozenset(name for name, enabled in (('handoff', config.write_handoff),
                                                       ('agent-formal', config.agent_formal)) if enabled)
    # Demonstrations first: the instructions name the kind and number of images supplied.
    demonstration = provision_demonstration(config, workspace, PROJECT/'runtime/reference-demos',
                                            record=not formal_episode)
    benchmark = for_config(config)
    built = build(recipe_for(options),
                  Setup(contract or Contract.from_config(config), options, config.task, config.agent_cli,
                        demonstrations=config.demonstration_context != 'none',
                        demonstration_kind=demonstration['kind'],
                        demonstration_images=demonstration.get('image_count', 0),
                        carries_handoff='handoff' in config.transfer_parts,
                        expert_demonstrations=config.expert_demonstrations,
                        expert_content=config.expert_content), workspace)
    rubric = benchmark.task_context(config, demonstration, live_formal=formal_episode)
    text = rubric + task_demonstration_context(demonstration, live_formal=formal_episode, bundles=not config.agent_formal)
    (workspace/'TASK.md').write_text(text)
    if not formal_episode:
        refresh_reviews(config, workspace)
    return built


def refresh_reviews(config, workspace):
    """An exploring agent's instructions follow its benchmark's current settings, such as
    its episode reviews (at deploy and on every resume)."""
    for_config(config).refresh_workspace(config, workspace)


def prepare_workspace(config, home_name):
    """Capped storage, the agent workspace (template, skills, TASK.md), the private
    configuration and the container bridge; shared by the Codex and Claude launchers."""
    root = config.root
    storage = workspace_storage.provision(root, config.image, config.workspace_mb)
    workspace, home = root/'agent-workspace', root/home_name
    for link, name in ((workspace, 'workspace'), (home, home_name)):
        (storage/name).mkdir(mode=0o700)
        link.symlink_to(storage/name, target_is_directory=True)
    (root/'published').mkdir(mode=0o700)
    persist(root/'workspace-manifest.json', populate_workspace(config, workspace))
    private_config = root/'configuration.json'
    with private_config.open('x') as stream:
        json.dump(asdict(config), stream, default=str)
    private_config.chmod(0o600)
    bridge = root/'container_bridge.py'
    shutil.copyfile(PROJECT/'harness/codex_cli/container_bridge.py', bridge)
    return {'workspace': workspace, 'home': home, 'configuration': private_config, 'bridge': bridge}


DEFAULT_OPENROUTER_KEY = Path('.codex/secrets/openrouter_api_key')  # Under the operator's home.
# The default 'openrouter' profile when no provider file defines one: OpenRouter's Responses API.
OPENROUTER_PROVIDER = {'name': 'OpenRouter', 'base_url': 'https://openrouter.ai/api/v1', 'wire_api': 'responses'}


def openrouter_key_file(path=None):
    """The OpenRouter key file (Gemini, and the model on the OpenRouter profile): the operator's
    --key-file, else ~/.codex/secrets/openrouter_api_key when it exists. None leaves the
    OPENROUTER_API_KEY environment variable as the only source (load_key(None))."""
    if path is not None:
        return Path(path)
    default = Path.home()/DEFAULT_OPENROUTER_KEY
    return default if default.is_file() else None


def openrouter_endpoint(name, provider):
    """Whether a Codex provider profile is OpenRouter, the only provider the OpenRouter key is sent to."""
    return name == 'openrouter' or urlsplit(str(provider.get('base_url', ''))).hostname == 'openrouter.ai'


def native_binary(path, name):
    """A native executable for the offline container, which has no Node: never an npm wrapper script."""
    binary = Path(path or shutil.which(name) or '').resolve()
    if not binary.is_file():
        raise RuntimeError(f'The {name} binary is required on the host PATH')
    with binary.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            raise RuntimeError(f'{binary} is not a native executable (an npm wrapper script?). The agent container '
                               f'has no Node.js: install the standalone {name} binary and put it first on PATH')
    return binary


def deploy(config, *, provider_home, profile='openrouter', key_file=None, codex=None):
    """Prepare a new SSD workspace without starting a model or simulator.

    key_file: the OpenRouter key file, or None for OPENROUTER_API_KEY. It is sent only to an
    OpenRouter profile; another provider uses its profile's [rax] key_file or env_key."""
    if not profile or Path(profile).name != profile:
        raise ValueError('Invalid provider profile')
    codex = native_binary(codex, 'codex')
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root/'state.json').exists() or (root/'agent-workspace').exists():
        raise FileExistsError('Use a fresh auto-research trial root')
    provider, relay = {}, {}
    for path in (provider_home/'config.toml', provider_home/f'{profile}.config.toml'):
        if path.exists():
            data = tomllib.loads(path.read_text())
            # Harness-only profile settings; Codex never reads these profile files.
            relay.update(data.get('rax', {}))
            for key in ('model_provider', 'model_providers'):
                if key in data:
                    if key == 'model_providers':
                        provider.setdefault(key, {}).update(data[key])
                    else:
                        provider[key] = data[key]
    name = provider.get('model_provider')
    selected = provider.get('model_providers', {}).get(name)
    if not isinstance(selected, dict) and profile == 'openrouter':
        name, selected = 'openrouter', dict(OPENROUTER_PROVIDER)
    if not isinstance(selected, dict):
        raise ValueError(f'Provider profile {profile!r} needs model_provider and its [model_providers.<name>] table '
                         f'in {provider_home/"config.toml"} or {provider_home/(profile+".config.toml")}')
    if selected.get('wire_api', 'responses') != 'responses':
        raise ValueError('Responses provider required')
    headers = dict(selected.get('http_headers', {}))
    chatgpt_auth = str((provider_home/relay['chatgpt_auth']).resolve()) if relay.get('chatgpt_auth') else None
    if chatgpt_auth:
        chatgpt_headers(chatgpt_auth)  # Only the login's path is stored; the relay reads it per call.
    elif relay.get('key_file'):
        # The profile's own model key; the OpenRouter key still serves Gemini.
        headers['Authorization'] = 'Bearer '+load_key(provider_home/relay['key_file'])
    elif openrouter_endpoint(name, selected):
        key = load_key(key_file)  # None: OPENROUTER_API_KEY.
        if not key:
            raise ValueError('The OpenRouter profile needs a key: --key-file, ~/.codex/secrets/openrouter_api_key '
                             'or OPENROUTER_API_KEY')
        headers['Authorization'] = 'Bearer '+key
    elif selected.get('env_key'):
        value = os.environ.get(selected['env_key'])
        if not value:
            raise ValueError(f'Provider profile {profile!r} reads its key from ${selected["env_key"]}, which is not set')
        headers['Authorization'] = 'Bearer '+value
    elif selected.get('auth'):
        raise ValueError('Auth commands are not copied or executed; set [rax] key_file in the provider profile')
    # A profile may name its model ([rax] model); the relay admits only that one.
    model = relay.get('model', MODEL)
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}', model):
        raise ValueError(f'Invalid model in provider profile {profile!r}')
    # Provider/auth material stays in a private host file, never in a mounted config.
    relay_file = root/'model-relay.json'
    relay_file.write_text(json.dumps({'base_url': selected['base_url'], 'headers': headers, 'model': model,
                                      **({'native_models': True} if relay.get('native_models') else {}),
                                      **({'chatgpt_auth': chatgpt_auth} if chatgpt_auth else {})}))
    relay_file.chmod(0o600)
    prepared = prepare_workspace(config, 'codex-home')
    workspace, home = prepared['workspace'], prepared['home']
    private_config, bridge = prepared['configuration'], prepared['bridge']
    settings = {'model_provider': 'host_relay', 'model': model, 'model_reasoning_effort': config.agent_effort,
        'model_providers': {'host_relay': {'name': 'Host model relay',
            'base_url': 'http://127.0.0.1:17777', 'wire_api': 'responses', 'requires_openai_auth': False,
            # Codex owns model recovery and retains completed tool results. Do
            # not wrap launch(), MCP calls, or robot episodes in retry loops.
            'request_max_retries': 2, 'stream_max_retries': 2,
            # Docker pause freezes Codex's stream reader, not its wall clock.
            # A formal batch can pause it for most of the session. The host
            # still enforces provider I/O.
            'stream_idle_timeout_ms': (CALL_SECONDS+300)*1000}},
        'model_auto_compact_token_limit': 256000, 'model_auto_compact_token_limit_scope': 'total',
        'approval_policy': 'never', 'sandbox_mode': 'danger-full-access', 'web_search': 'disabled',
        'features': {**{name: False for name in DISABLED_FEATURES}, 'shell_tool': True,
                     'unified_exec': True, 'code_mode_host': True, 'skip_host_skill_discovery': True},
        'projects': {'/workspace': {'trust_level': 'trusted'}}, 'tui': {'status_line': ROBOT_STATUS_LINE},
        'mcp_servers': {'auto_research': {'command': '/usr/local/bin/python',
            'args': ['/opt/agent/container_bridge.py', 'mcp'], 'required': True,
            'tool_timeout_sec': CALL_SECONDS+300,
            'default_tools_approval_mode': 'approve'}}}
    from services.mcp_contract import Contract
    if cap := Contract.from_config(config).subagent_cap:
        agents = subagent_settings(cap)
        settings['features'].update(agents['features'])
        settings['agents'] = agents['agents']
    _write_config(home/'config.toml', settings)
    resume_session = None
    if config.adaptation_source:
        from services.adaptation import import_source
        resume_session = import_source(config, workspace, home)
    return {'workspace': workspace, 'codex_home': home, 'codex': codex, 'resume_session': resume_session,
            'configuration': private_config, 'relay_configuration': relay_file, 'bridge': bridge}


def published_mounts(config, deployed):
    """The session's run tree, or (agent formal) only the current episode's frames, and the
    benchmark's read-only reference data (such as expert episodes)."""
    from services.benchmarks import for_config
    runs = (list(deployed['published_mounts']) if 'published_mounts' in deployed
            else [(config.root/'published', '/workspace/runtime/autonomous_controller', True)])
    return runs + [(source, destination, True) for source, destination in for_config(config).reference_mounts(config)]


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
        '--workdir', '/workspace', '--env', 'HOME=/workspace', '--env', 'CODEX_HOME=/codex-home',
        '--env', 'PIP_NO_INDEX=1', '--env', 'HF_HUB_OFFLINE=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', 'PYTHONPATH=/workspace',
        '--env', 'PATH=/opt/codex:/usr/local/bin:/usr/bin:/bin', '--env', 'TERM=xterm-256color',
        # The agent CLI shares --pids-limit with everything the agent runs.
        *thread_env(lim.cpus)]
    if interactive:
        command.append('--tty')
    mounts = [(deployed['workspace'], '/workspace', False), (deployed['codex_home'], '/codex-home', False),
        (deployed['codex_home']/'config.toml', '/codex-home/config.toml', True),
        *published_mounts(config, deployed),
        (relay_directory, '/run/agent-relay', True), (deployed['bridge'], '/opt/agent/container_bridge.py', True),
        (deployed['codex'], '/opt/codex/codex', True)]
    helper = deployed['codex'].with_name('codex-code-mode-host')
    if helper.is_file():
        mounts.append((helper, '/opt/codex/codex-code-mode-host', True))
    for source, destination, readonly in mounts:
        if any(c in str(source) for c in (',', '\n')):
            raise ValueError('Invalid mount path')
        command += ['--mount', f'type=bind,src={source},dst={destination}'+(',readonly' if readonly else '')]
    for variable in deployed.get('container_env', ()):
        command += ['--env', variable]
    command += ['--entrypoint', '/usr/local/bin/python', config.image,
                '/opt/agent/container_bridge.py', '--no-daemon', '-C', '/workspace',
                # One-call configuration overrides (session compaction); never permissions.
                *deployed.get('codex_overrides', ())]
    if not interactive:
        command += ['exec']
        if deployed.get('resume_session'):
            command += ['resume']
        command += ['--skip-git-repo-check']
        if json_output:
            command.append('--json')
    elif deployed.get('resume_session'):
        command.append('resume')
    if deployed.get('resume_session'):
        command.append(deployed['resume_session'])
    command.append(prompt)
    return command


def relay_slots(config):
    """Concurrent model calls a session's relay serves: the agent and its subagents, plus spare."""
    from services.mcp_contract import Contract
    return slot_count(Contract.from_config(config).subagent_cap)


def model_relay(config, deployed, relay_dir, *, calls_file=None, seconds):
    """Start the host model relay whose socket the container bridge forwards to."""
    provider = json.loads(deployed['relay_configuration'].read_text())
    model = ModelRelay(relay_dir/'model.sock', **provider, calls_file=calls_file, client_write_timeout=seconds,
                       slots=relay_slots(config))
    model.start()
    return model


def session_files(home, session_id):
    """Transcript files that resume a Codex thread in a fresh home, relative to it."""
    for path in Path(home, 'sessions').rglob('rollout-*.jsonl'):
        try:
            with path.open() as stream:
                if (json.loads(stream.readline()).get('payload') or {}).get('id') == session_id:
                    return [path.relative_to(home)]
        except (OSError, ValueError):
            continue
    raise RuntimeError(f'Codex thread {session_id} not found in {home}')


HOME_KEY, HOME_NAME = 'codex_home', 'codex-home'
# Trusted launcher-written configuration a copied session needs (never credentials).
HOME_FILES = ('config.toml',)
# The frontend closes its simulator before exiting; a scene teardown alone may take over a minute.
SERVER_STOP_SECONDS = 120


STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


@contextmanager
def deferred_sigterm():
    """Hold stop signals (SIGTERM, SIGHUP) until the block ends, then deliver them. A repeated
    stop request must not abort cleanup half way and orphan the frontend and its simulator."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending = []
    previous = {number: signal.signal(number, lambda number, _: pending.append(number)) for number in STOP_SIGNALS}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    for number in dict.fromkeys(pending):
        signal.raise_signal(number)


def end_process(process, seconds):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def stop_services(name, agent, server, model):
    """Launcher cleanup: every step runs even if an earlier one fails."""
    steps = [lambda: subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)]
    steps += [lambda: end_process(agent, 10)] if agent else []
    steps += [lambda: end_process(server, SERVER_STOP_SECONDS)] if server else []
    steps += [model.close] if model else []
    errors = []
    with deferred_sigterm():
        for step in steps:
            try:
                step()
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]


def pass_stderr(stream, errors):
    """Pass the agent CLI's stderr through to the launcher's, keeping its ERROR lines whole
    (Codex prints the provider's error there before it exits)."""
    for line in stream:
        try:
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
        except (AttributeError, OSError, ValueError):
            pass  # Keep draining: a full pipe would stall the agent.
        if line.startswith(b'ERROR'):
            errors.append(line.decode('utf-8', 'replace').strip())


def open_frontend_log(root):
    """The session's frontend.log, appended to with a header per launch: an earlier attempt's
    tracebacks and simulator errors stay readable after a resume."""
    log = (Path(root)/'frontend.log').open('ab')
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    log.write(f'\n===== launch {stamp} (launcher pid {os.getpid()}) =====\n'.encode())
    log.flush()
    log.launch_start = log.tell()  # Where this launch's output begins.
    return log


def start_frontend(args, log):
    """The host MCP frontend, in its own session: a terminal's Ctrl+C reaches only the launcher,
    whose cleanup then stops the frontend (and through it the simulators) in order."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith('RAX_')}
    environment['PYTHONPATH'] = str(PROJECT)
    return subprocess.Popen(args, cwd=PROJECT, env=environment, stdout=log, stderr=log, start_new_session=True)


def frontend_failure(server, log, waited):
    """The frontend's own startup error, whole: how it ended (its exit code, or still starting after
    ``waited`` seconds) and the end of what it wrote to ``log`` (an open frontend.log) in this
    launch, where it wrote its traceback."""
    code = server.poll()
    ended = f'exited with code {code}' if code is not None else f'opened no socket within {waited:g} s'
    log.flush()
    try:
        with open(log.name, 'rb') as stream:
            stream.seek(log.launch_start)
            tail = stream.read()[-4000:].decode('utf-8', 'replace').strip() or '(empty)'
    except OSError as exc:
        tail = f'(unreadable: {type(exc).__name__}: {exc})'
    return RuntimeError(f'Host MCP startup failed: the frontend {ended}; frontend.log ends:\n{tail}')


def launch(config, deployed, *, key_file=None, prompt, interactive=False, capture=False, calls_file=None,
           agent_runs=None):
    """Host owns both relays and cleanup; Docker owns every agent-executed command.

    The relays and the frontend (the sole owner of the session's episodes) stay up for every
    agent run of this launch: `agent_runs(run)` may call `run(prompt)` again after a run ends (a
    finished turn, a provider error), and the open episodes keep running in between."""
    DockerSandbox(config.image, config.training).preflight()
    workspace_storage.validate(config.root, config.workspace_mb,
                               deployed['workspace'], deployed['codex_home'])
    provider = json.loads(deployed['relay_configuration'].read_text())
    if provider.get('chatgpt_auth'):
        try:
            chatgpt_headers(provider['chatgpt_auth'])
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f'ChatGPT login unusable: {exc}') from None
    name = 'rax-agent-'+uuid.uuid4().hex
    server = model = agent = None
    # Only tiny socket files use /tmp, to stay below Linux's Unix-socket path limit.
    with tempfile.TemporaryDirectory(prefix='rax-relay-') as temporary:
        relay_dir = Path(temporary)
        with open_frontend_log(config.root) as log:
            try:
                model = model_relay(config, deployed, relay_dir, calls_file=calls_file, seconds=CALL_SECONDS+300)
                args = [sys.executable, '-m', 'services.controller.frontend', '--config', str(deployed['configuration']),
                        '--workspace', str(deployed['workspace']), '--socket', str(relay_dir/'mcp.sock'),
                        '--agent-container', name]
                if key_file:
                    args += ['--key-file', str(key_file)]
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
                    """One agent run against this launch's frontend; its last ERROR line is kept."""
                    nonlocal agent
                    subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
                    command = container_command(config, deployed, relay_dir, name, prompt=prompt,
                                                interactive=interactive)
                    errors = []
                    if interactive:
                        agent = subprocess.Popen(command)
                    else:
                        # No stdin: Codex exec would wait to read more prompt from an open one.
                        agent = subprocess.Popen(command, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        copier = threading.Thread(target=pass_stderr, args=(agent.stderr, errors), daemon=True)
                        copier.start()
                    agent.wait()
                    if not interactive:
                        copier.join(timeout=10)
                    deployed['agent_error'] = errors[-1] if errors else None
                    return agent.returncode
                returncode = agent_runs(run) if agent_runs else run(prompt)
                return subprocess.CompletedProcess(agent.args, returncode, None, None)
            finally:
                # Exact owned identity; never remove containers from other sessions.
                stop_services(name, agent, server, model)


RESUME_PROMPT = ("This session was interrupted and has now been resumed in the same workspace. Reread "
                 "memory/PROGRESS.md, call status, and continue from the live state. Do not repeat finished "
                 "work, and do not restart an episode merely because the conversation resumed.")
CALLS_FILE = 'model-calls.json'
CONTINUE_PROMPT = ("Your previous turn ended while exploration is still open; any background workers it left "
                   "running were stopped, and your open episodes kept running. Call status, reread "
                   "memory/PROGRESS.md, and continue from the live state until exploration is closed.")
ERROR_PROMPT = ("Your previous run ended on an error while exploration is still open ({error}); any background "
                "workers it left running were stopped, and your open episodes kept running. Call status, reread "
                "memory/PROGRESS.md, and continue from the live state.")
MAX_RELAUNCHES = 3
# A run that fails (a provider overload, a dropped stream) is resumed after a pause; a run
# that keeps failing quickly is given up after this many tries in a row.
MAX_ERROR_RELAUNCHES = 5
ERROR_PAUSE_SECONDS = 300
STEADY_SECONDS = 1800  # A run this long resets the count of failures in a row.


def run_agent(start, *, deployed, prompt, interactive, open_exploration, latest, lifecycle, save,
              last_error=lambda: None, sleep=time.sleep):
    """Run the agent; while exploration is open, resume the same conversation when a
    non-interactive run ends its turn (a bounded number of times) or fails (after a pause)."""
    failures = 0
    while True:
        started = time.monotonic()
        returncode = start(prompt)
        if interactive or not open_exploration():
            return returncode
        if returncode != 0:
            failures = 1 if time.monotonic()-started >= STEADY_SECONDS else failures+1
            error = last_error() or f'exit code {returncode}'
            lifecycle.setdefault('agent_errors', []).append({'time': time.time(), 'returncode': returncode,
                                                             'error': error})
            save()
            if failures > MAX_ERROR_RELAUNCHES:
                return returncode
            next_prompt = ERROR_PROMPT.format(error=error)
        else:
            failures = 0  # Failures count only in a row.
            relaunches = lifecycle.get('relaunches', 0)
            if relaunches >= MAX_RELAUNCHES:
                return returncode
            next_prompt = CONTINUE_PROMPT
        try:
            deployed['resume_session'] = latest()
        except (RuntimeError, OSError, TypeError):
            return returncode  # No conversation to continue.
        if returncode != 0:
            sleep(ERROR_PAUSE_SECONDS)
        else:
            lifecycle['relaunches'] = lifecycle.get('relaunches', 0)+1
            save()
        prompt = next_prompt


def latest_codex_thread(home):
    """Id of the newest top-level (user) Codex thread in a session's Codex home."""
    newest = None
    for path in Path(home, 'sessions').rglob('rollout-*.jsonl'):
        try:
            with path.open() as stream:
                meta = json.loads(stream.readline()).get('payload') or {}
        except (OSError, ValueError):
            continue
        if meta.get('thread_source', 'user') != 'user' or not meta.get('id'):
            continue  # Subagent threads are not resumed on their own.
        if newest is None or path.stat().st_mtime > newest[0]:
            newest = (path.stat().st_mtime, meta['id'])
    if newest is None:
        raise RuntimeError('No Codex conversation to resume in '+str(home))
    return newest[1]


latest_session = latest_codex_thread


def resumable_lifecycle(record):
    """Earlier attempts of a session, or a reason it cannot be resumed."""
    if not record.is_file():
        raise ValueError('Nothing to resume: this session was never launched')
    previous = json.loads(record.read_text())
    if previous.get('status') == 'prepared-only':
        raise ValueError('The session was only prepared; launch it instead')
    if not previous.get('finished_at'):
        # The caller holds launch.lock, so no launcher is alive: the previous one died
        # without cleanup (kill -9, host reboot). Its end is the session's last activity.
        logs = [path.stat().st_mtime for path in (record.parent/'frontend.log', record.parent/'state.json')
                if path.exists()]
        previous.update(status='crashed', finished_at=max(logs, default=previous.get('started_at', time.time())))
    return [*previous.pop('attempts', []), previous]


def initial_calls(config, agent_cli):
    """Model calls already spent, for sessions launched before calls were persisted."""
    path = config.root/CALLS_FILE
    if path.is_file():
        return int(json.loads(path.read_text()).get('calls', 0))
    if agent_cli == 'claude':
        usage = config.root/'model-usage.jsonl'
        return sum(1 for _ in usage.open()) if usage.is_file() else 0
    return sum(sum(1 for line in path.open() if '"token_count"' in line)
               for path in (config.root/'codex-home/sessions').rglob('rollout-*.jsonl'))


def redeploy(config, codex=None):
    """Reattach an existing Codex session workspace for a resumed launch."""
    codex = native_binary(codex, 'codex')
    root = config.root
    workspace_storage.remount(root)
    deployed = {'workspace': root/'agent-workspace', 'codex_home': root/'codex-home', 'codex': codex,
                'configuration': root/'configuration.json', 'relay_configuration': root/'model-relay.json',
                'bridge': root/'container_bridge.py'}
    for key in ('workspace', 'codex_home', 'configuration', 'relay_configuration', 'bridge'):
        if not deployed[key].exists():
            raise FileNotFoundError(f'Cannot resume: {deployed[key]} is missing')
    deployed['resume_session'] = latest_codex_thread(deployed['codex_home'])
    refresh_reviews(config, deployed['workspace'])
    return deployed


def startup_prompt(workspace, override=None):
    """Keep the default kickoff editable alongside the standing instructions."""
    if override:
        return override
    if (workspace/'ADAPTATION.md').is_file():
        from services.adaptation import PROMPT
        return PROMPT
    return (workspace/'START_PROMPT.md').read_text().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--provider-home', type=Path, default=Path(os.environ.get('CODEX_HOME', Path.home()/'.codex')))
    parser.add_argument('--profile', help='Provider profile; default: <provider-home>/experiment-profile, else openrouter')
    parser.add_argument('--key-file', type=Path,
                        help='OpenRouter key file for Gemini and the OpenRouter model profile; default: '
                             '~/.codex/secrets/openrouter_api_key when it exists, else OPENROUTER_API_KEY')
    parser.add_argument('--prepare-only', action='store_true', help='Disposable deployment check; starts no model/episode')
    parser.add_argument('--exec-prompt', help='Run noninteractively instead of opening the terminal UI')
    parser.add_argument('--non-interactive', action='store_true', help='Run the default research prompt without a terminal UI')
    parser.add_argument('--formal-only', action='store_true',
                        help='Post-hoc agent formal of an imported agent (adaptation_source, agent_formal on): '
                             'import, freeze and run agent formal without an exploring agent')
    parser.add_argument('--agent-formal', action='store_true',
                        help='With --resume of a session whose exploration has closed: play its formal layouts '
                             'with live copies of the frozen agent (agent formal), beside any bundle formal it ran')
    parser.add_argument('--resume', action='store_true',
                        help="Relaunch a stopped/exited/failed session: same workspace and state, "
                             "continuing the agent's own conversation")
    # agent_cli: claude (operator configuration) runs Claude Code instead; see harness/claude_cli.
    parser.add_argument('--claude-key-file', type=Path,
                        help='anthropic: an API key file; claude-login: override the experiment setup-token '
                             'file (default: harness/claude_cli/experiment_login.py). The personal '
                             '~/.claude login is always refused.')
    parser.add_argument('--claude-model', help='Override the provider default model')
    parser.add_argument('--claude-base-url', help='Anthropic-compatible endpoint for the custom provider')
    parser.add_argument('--agent-binary', type=Path, help='Native third-party agent CLI executable')
    parser.add_argument('--agent-model', help='Model override for the native agent backend')
    parser.add_argument('--agent-key-file', type=Path, help='Private model key file for the native agent backend')
    parser.add_argument('--agent-base-url', help='Private model API base URL')
    parser.add_argument('--agent-auth-home', type=Path, help='Dedicated native agent account login directory')
    args = parser.parse_args()
    if args.prepare_only and (args.resume or args.formal_only or args.agent_formal):
        parser.error('--prepare-only checks a fresh deployment; it cannot be combined with --resume, '
                     '--formal-only or --agent-formal')
    if not args.profile:
        chosen = args.provider_home/'experiment-profile'
        args.profile = chosen.read_text().strip() if chosen.is_file() else 'openrouter'
    args.key_file = openrouter_key_file(args.key_file)
    config = Configuration.from_dict(json.loads(args.config.read_text()))
    from services.controller.config import with_task_timeouts
    config = with_task_timeouts(config)
    def stop(*_):
        raise KeyboardInterrupt
    # A closed terminal (SIGHUP) stops the session like Stop does: the frontend runs in its own
    # session and would otherwise outlive the launcher with its simulators. A launcher started
    # under nohup keeps ignoring it.
    signal.signal(signal.SIGTERM, stop)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, stop)
    config.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.root/'launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'Another launcher is running this session ({config.root}); stop it with '
                             'scripts/stop_auto_research_agent.sh and wait for its cleanup before launching again')
        lifecycle = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'preparing'}
        record = config.root/'agent-lifecycle.json'
        from services.agent_formal.checkpoint import exploration_closed
        closed = args.resume and exploration_closed(config)  # Only the frozen formal remains to run.
        if args.formal_only:
            if args.resume or not (config.agent_formal and config.adaptation_source):
                raise SystemExit('--formal-only needs a fresh configuration with agent formal and an adaptation source')
            closed = True
        if args.agent_formal and not (args.resume and closed):
            raise SystemExit('--agent-formal needs --resume of a session whose exploration has closed')
        if args.resume:
            try:
                attempts = resumable_lifecycle(record)
            except ValueError as exc:
                raise SystemExit(str(exc))
            if not (config.root/'configuration.json').is_file():
                failed = attempts[-1].get('error_type') if attempts else None
                raise SystemExit('Nothing to resume: the first launch failed before the agent workspace was '
                                 f'prepared{f" ({failed})" if failed else ""}; read its error in the launch output '
                                 f'({config.root/"agent.log"} for dashboard launches), fix the cause and prepare a '
                                 'fresh configuration')
            calls = config.root/CALLS_FILE
            if not closed and not calls.exists():  # Launched before call counts were persisted.
                persist(calls, {'calls': initial_calls(config, config.agent_cli)})
            lifecycle.update(resumed=True, attempts=attempts)
        elif record.exists():
            # A repeated launch must not overwrite the original run's lifecycle.
            raise FileExistsError('This session was already launched; use --resume or a fresh configuration')
        persist(record, lifecycle)
        deployed = None
        try:
            if config.agent_cli == 'claude':
                from harness.claude_cli import auto_research as claude
            elif config.agent_cli not in ('codex', 'claude'):
                from harness.backends import get
                external_cli = get(config.agent_cli)
                if args.agent_formal or args.formal_only:
                    raise ValueError(f'{config.agent_cli} supports code formal; live agent formal is not implemented')
            if args.resume:
                if config.agent_cli == 'claude' and config.agent_provider == 'claude-login' and not args.claude_key_file:
                    claude.preflight_login()  # The token may have been revoked since the first launch.
                deployed = (external_cli.redeploy(config, binary=args.agent_binary) if config.agent_cli not in ('codex', 'claude') else
                            claude.redeploy(config) if config.agent_cli == 'claude' else redeploy(config))
            elif config.agent_cli not in ('codex', 'claude'):
                deployed = external_cli.deploy(config, binary=args.agent_binary, model=args.agent_model,
                    key_file=args.agent_key_file or args.key_file, base_url=args.agent_base_url,
                    auth_home=args.agent_auth_home)
            elif config.agent_cli == 'claude':
                provider = config.agent_provider
                key_file = {'openrouter': args.key_file, 'anthropic': args.claude_key_file,
                            'claude-login': args.claude_key_file, 'custom': args.claude_key_file}[provider]
                if provider == 'claude-login' and not args.claude_key_file:
                    # Fail before provisioning if the session's experiment token is missing or rejected.
                    claude.preflight_login()
                deployed = claude.deploy(config, provider=provider, model=args.claude_model, key_file=key_file,
                                        base_url=args.claude_base_url)
            else:
                deployed = deploy(config, provider_home=args.provider_home, profile=args.profile, key_file=args.key_file)
            if args.prepare_only:
                print(json.dumps({k: str(v) for k, v in deployed.items()}, indent=2))
                lifecycle['status'] = 'prepared-only'
                return
            prompt = args.exec_prompt or RESUME_PROMPT if args.resume else startup_prompt(deployed['workspace'], args.exec_prompt)
            if (not args.resume and config.adaptation_source and config.adaptation_context == 'compact'
                    and deployed.get('resume_session')):
                # Compact the carried conversation once, before its first turn here.
                from harness.session_context import compact
                persist(config.root/'adaptation-compaction.json', compact(
                    config, deployed, deployed['resume_session'], log=config.root/'compaction.jsonl'))
            if args.formal_only:
                # No exploring agent: the imported one is frozen as it arrived.
                persist(config.root/'exploration-complete.json', {'finished_at': time.time(), 'formal_only': True})
            if closed:
                returncode = 0  # Exploration already ended; only the frozen formal remains.
            else:
                lifecycle['status'] = 'running'
                persist(record, lifecycle)
                interactive = not (args.non_interactive or args.exec_prompt)
                base_wait = 0.0
                def usage_wait(until, waited):
                    nonlocal base_wait
                    if until is None:  # One wait finished; record the total time waited.
                        lifecycle.update(status='running', usage_wait_seconds=base_wait+waited)
                        lifecycle.pop('usage_limited_until', None)
                    else:
                        lifecycle.update(status='usage_wait', usage_limited_until=until)
                    persist(record, lifecycle)
                from services.exploration.pool import recover_interrupted, report_refunds
                recovered = recover_interrupted(config.root/'exploration')  # Left live by a previous launcher.
                if recovered:
                    lifecycle.setdefault('interrupted_episodes', []).extend(recovered)
                    persist(record, lifecycle)
                from harness.backends import get
                cli = get(config.agent_cli)
                home = deployed.get(cli.HOME_KEY)
                def agent_runs(run):
                    """Every agent run of this launcher, against one frontend: open episodes outlive a run."""
                    def start(prompt):
                        nonlocal base_wait
                        base_wait = lifecycle.get('usage_wait_seconds', 0.0)
                        note = report_refunds(config.root/'exploration')  # Told once, with the next prompt.
                        return run(f'{prompt}\n\n{note}' if note else prompt)
                    return run_agent(
                        start, deployed=deployed, prompt=prompt, interactive=interactive,
                        open_exploration=lambda: not exploration_closed(config),
                        latest=lambda: cli.latest_session(home),
                        last_error=lambda: (cli.last_error(home) if config.agent_cli != 'codex'
                                            else deployed.get('agent_error')),
                        lifecycle=lifecycle, save=lambda: persist(record, lifecycle))
                if config.agent_cli not in ('codex', 'claude'):
                    returncode = external_cli.launch(config, deployed, gemini_key_file=args.key_file, prompt=prompt,
                                           interactive=interactive, agent_runs=agent_runs).returncode
                elif config.agent_cli == 'claude':
                    returncode = claude.launch(config, deployed, gemini_key_file=args.key_file, prompt=prompt,
                                               interactive=interactive, calls_file=config.root/CALLS_FILE,
                                               on_wait=usage_wait, agent_runs=agent_runs).returncode
                else:
                    returncode = launch(config, deployed, key_file=args.key_file, prompt=prompt, interactive=interactive,
                                        calls_file=config.root/CALLS_FILE, agent_runs=agent_runs).returncode
            lifecycle.update(status='exited', returncode=returncode)
            # The exploring agent has ended: freeze it, then run formal in isolation.
            lifecycle['status'] = 'formal'
            persist(record, lifecycle)
            from services.agent_formal.checkpoint import after_exploration
            summary = after_exploration(config, deployed, gemini_key_file=args.key_file, live_copies=args.agent_formal)
            lifecycle.update(status='exited', formal=summary['status'],
                             **({'agent_formal': summary['agent_formal'].get('status')}
                                if isinstance(summary.get('agent_formal'), dict) else {}))
            raise SystemExit(returncode)
        except KeyboardInterrupt:
            lifecycle['status'] = 'stopped'
            raise SystemExit(130)
        except Exception as exc:
            lifecycle.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            lifecycle['finished_at'] = time.time()
            persist(record, lifecycle)
            if deployed and lifecycle['status'] != 'prepared-only':
                record_subagents(config, deployed)


def record_subagents(config, deployed):
    """Subagent spawn/concurrency record for the multi-agent baseline (best effort)."""
    from services import subagent_metrics
    from services.mcp_contract import Contract
    if config.agent_cli not in ('codex', 'claude'):
        from harness.backends import get
        return get(config.agent_cli).record_subagents(config, deployed)
    claude = config.agent_cli == 'claude'
    home = deployed.get('claude_home' if claude else 'codex_home')
    if home:
        subagent_metrics.write(home, config.root/'subagents.json', cli='claude' if claude else 'codex',
                               cap=Contract.from_config(config).subagent_cap)


if __name__ == '__main__':
    main()
