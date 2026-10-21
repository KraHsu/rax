"""One live agent per held-out formal episode, behind the supervisor's normal episode.

Supervisor._run_episode reserves the layout, starts the native episode, enforces the
wall limit and evaluates natively; this runner replaces the frozen controller with a
copy of the exploring agent (Codex or Claude Code). The copy starts in a fresh capped
workspace holding only the transfer package and, unless the context is 'clear', the
conversation. It sees its own episode's frames, never the exploration frame history.
Robot tools and development Python reach the gateway through the usual container
socket; lifecycle tools do not exist here.
"""
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import socket
import subprocess
import tempfile
import time

from harness.codex_cli import workspace_storage
from services.controller.errors import public_failure
from services.controller.gemini import MAX_REQUEST_BYTES
from services.controller.sandbox import DockerSandbox, ExecutionDeadline, deadline, finalize_within, time_left, wall_end
from services.transfer import copy_regular, unpack

PROJECT = Path(__file__).resolve().parents[2]
FORMAL_PROMPT = (
    'An episode of this task is already running, on a layout you have not seen. Read AGENTS.md first: it '
    'replaces your earlier instructions. Carried over: {transferred}. Complete the task in this episode with '
    'your carried code as the controller, and control the robot yourself where it needs help.')
# A code-mode explorer built its controller to run unattended (agent formal after its bundle formal).
UNATTENDED = (' Your carried code was built to run a whole episode end to end on its own; now you are here '
              'with it, so run it, watch it, and step in or control the robot yourself wherever that helps.')
# Its code-mode runs were closed once their recording filled the sandbox's artifact limit; a copy's frames
# and tool replies are kept on the host instead (deployed: published_mounts), so nothing caps them.
RECORDING = (' This episode has no recording cap: its camera frames and tool replies are stored outside your '
             'workspace and never end it. Its only limits are its step limit and any wall limit `status` reports.')


def episode_prompt(config, transferred):
    """A copy's first message; a code-mode explorer's copy also learns it may now join its code, and
    that the recording limit of its code-mode runs does not apply."""
    prompt = FORMAL_PROMPT.format(transferred=', '.join(transferred) or 'nothing')
    if config.agent_formal:
        return prompt
    return prompt + UNATTENDED + RECORDING
# A copy whose run fails while its episode is live (a provider overload, a dropped stream) resumes its
# conversation after a pause, as an exploring agent does; one that keeps failing quickly is given up after
# this many tries in a row.
ERROR_PROMPT = ('Your previous run ended on an error while this episode is still running ({error}); anything it '
                'left running in the background was stopped, and the episode kept its state. Call status and '
                'continue from the live state.')
MAX_ERROR_RELAUNCHES = 5
ERROR_PAUSE_SECONDS = 300
STEADY_SECONDS = 1800  # A run this long resets the count of failures in a row.
# How much of the exploring conversation a copy has, in the words its status uses.
CONVERSATION = {'clear': 'none', 'compact': 'summary', 'resume': 'full'}


def harness(cli):
    from harness.backends import get
    return get(cli)


def status_tool():
    return {'name': 'status', 'description': 'This episode: task, wall limit, elapsed seconds, whether the '
            'episode has ended, the folders you carried and how much of your earlier conversation you have.',
            'inputSchema': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}


class FormalMCP:
    """The container's only endpoint: robot tools via the gateway, plus a local status."""
    def __init__(self, gateway, token, *, info):
        self.gateway, self.token, self.info = gateway, token, info
        self.started = time.monotonic()

    def handle(self, request):
        if not isinstance(request, dict):
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
        params = request.get('params') if isinstance(request.get('params'), dict) else {}
        if request.get('method') == 'tools/call' and params.get('name') == 'status':
            value = {**self.info, 'elapsed_seconds': round(time.monotonic() - self.started, 1),
                     'episode_ended': bool(self.gateway.terminal)}
            return {'jsonrpc': '2.0', 'id': request.get('id'),
                    'result': {'content': [{'type': 'text', 'text': json.dumps(value)}]}}
        response = self.gateway.handle(self.token, request)
        if request.get('method') == 'tools/list' and isinstance(response, dict) and 'result' in response:
            response['result']['tools'] = [status_tool(), *response['result']['tools']]
        return response


class AgentRunner(DockerSandbox):
    """Runs a copied agent instead of controller.py; same run() contract as DockerSandbox."""
    # The agent may finish its notes after the episode ends; nothing it does then counts.
    finalization_seconds = 120

    def __init__(self, image, limits, *, config, layout_id=None):
        """``config`` is the session's (the parent of the formal workers); ``layout_id`` is the
        held-out layout this copy plays."""
        super().__init__(image, limits)
        self.config = config
        self.layout_id = layout_id
        self.cli = harness(config.agent_cli)

    def preflight(self):
        if os.getuid() == 0:
            raise RuntimeError('Run agent formal as a non-root operator')
        super().preflight()
        binary = shutil.which('claude' if self.config.agent_cli == 'claude' else 'codex')
        if not binary:
            raise RuntimeError(f'{self.config.agent_cli} executable is required for agent formal')

    def deployed(self, package, output, frames):
        """A fresh capped workspace and agent home, holding only the package."""
        record = json.loads((package/'package.json').read_text())
        storage = workspace_storage.provision(output, self.image, self.config.workspace_mb)
        workspace, home = storage/'workspace', storage/self.cli.HOME_NAME
        workspace.mkdir(mode=0o700)
        home.mkdir(mode=0o700)
        from harness.codex_cli.auto_research import populate_workspace
        from services.mcp_contract import Contract
        # The formal-episode workspace: its own AGENTS.md, the isolated tool contract.
        built = populate_workspace(self.config, workspace, contract=Contract.from_config(self.config, isolated=True),
                                   formal_episode=True)
        (output/'workspace-manifest.json').write_text(json.dumps(built, indent=1) + '\n')
        if self.config.agent_cli == 'claude':
            self.cli.claude_workspace(workspace)
        context = self.config.formal_context
        transferred = unpack(package, workspace=workspace, home=home, context=context)
        run = 'runs/'+Path(frames).parent.name+'/frames'
        # Pre-create the mount point so Docker never creates root-owned directories.
        (workspace/'runtime/autonomous_controller'/run).mkdir(parents=True, exist_ok=True)
        bridge = output/'container_bridge.py'
        shutil.copyfile(PROJECT/'harness/codex_cli/container_bridge.py', bridge)
        relay = self.config.root/'model-relay.json'
        deployed = {'workspace': workspace, self.cli.HOME_KEY: home, 'bridge': bridge,
                    'relay_configuration': relay, 'subagents': 0,
                    'published_mounts': [(frames, '/workspace/runtime/autonomous_controller/'+run, True)],
                    'resume_session': record['session_id'] if context != 'clear' else None,
                    'baseline_usage': record.get('baseline_usage')}
        if self.config.agent_cli == 'claude':
            deployed.update(claude=self.cli.claude_binary(), model=json.loads(relay.read_text())['model'])
        else:
            deployed['codex'] = Path(shutil.which('codex')).resolve()
        return deployed, transferred, record

    def run(self, bundle, *, mode, entrypoint, gateway=None, token=None, output, name=None,
            frames=None, wall_seconds=None, workspace_path=None):
        if mode != 'formal' or gateway is None or frames is None:
            raise ValueError('Agent formal runs one formal episode with its frames')
        output = Path(output)
        output.mkdir(mode=0o700, exist_ok=True)
        package = Path(bundle)
        end_at = wall_end(self.limits.wall_seconds, wall_seconds)
        reason, returncode, finalizing, process, relay = 'exit', None, False, None, None
        deployed = None
        errors, unrecovered = [], None  # A copy's failed runs; the error that ended its last one, if it was given up.
        started = None
        try:
            deployed, transferred, record = self.deployed(package, output, frames)
            info = {'task': self.config.task, 'episode_timeout_seconds': wall_seconds,
                    'carried': transferred, 'conversation': CONVERSATION[self.config.formal_context]}
            endpoint = FormalMCP(gateway, token, info=info)
            prompt = episode_prompt(self.config, transferred)
            with tempfile.TemporaryDirectory(prefix='rax-formal-') as temporary:
                relay_dir = Path(temporary)
                extra = {'usage_log': output/'model-usage.jsonl'} if self.config.agent_cli == 'claude' else {}
                relay = self.cli.model_relay(self.config, deployed, relay_dir,
                                             calls_file=output/'model-calls.json',
                                             seconds=int(time_left(end_at) or self.cli.CALL_SECONDS)+300, **extra)
                with socket.socket(socket.AF_UNIX) as listener:
                    listener.bind(str(relay_dir/'mcp.sock'))
                    os.chmod(relay_dir/'mcp.sock', 0o600)
                    listener.listen(16)
                    command = self.cli.container_command(self.config, deployed, relay_dir, name,
                                                         prompt=prompt, json_output=True)
                    remaining = time_left(end_at)
                    if remaining is not None and remaining <= 0:
                        raise ExecutionDeadline()
                    # Owns SIGALRM for the wall limit and serve()'s finalization timer;
                    # without a handler the alarm would kill this worker unrecorded.
                    with deadline(remaining):
                        started = time.monotonic()
                        failures = 0
                        while True:
                            # The agent's JSON event stream is this run's stdout log; a resumed run appends.
                            with (output/'stdout.log').open('ab' if errors else 'wb') as log:
                                offset, began = log.tell(), time.monotonic()
                                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                                           stderr=subprocess.STDOUT, start_new_session=True)
                                finalizing = self.serve(listener, endpoint, process, gateway)
                            returncode = process.wait(timeout=30)
                            # Its own exit (giving up, ending its turn) or the episode's end is final.
                            if finalizing or returncode == 0 or gateway.terminal:
                                break
                            failures = 1 if time.monotonic()-began >= STEADY_SECONDS else failures+1
                            home = deployed[self.cli.HOME_KEY]
                            error = run_error(self.config.agent_cli, output/'stdout.log', offset, home) \
                                or f'exit code {returncode}'
                            errors.append({'time': time.time(), 'returncode': returncode, 'error': error,
                                           'run_seconds': round(time.monotonic()-began, 1)})
                            unrecovered = error
                            if failures > MAX_ERROR_RELAUNCHES:
                                break
                            try:
                                deployed['resume_session'] = self.cli.latest_session(home)
                            except (RuntimeError, OSError, TypeError):
                                break  # No conversation to continue.
                            time.sleep(ERROR_PAUSE_SECONDS)
                            if gateway.terminal:
                                break
                            subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=20)
                            command = self.cli.container_command(self.config, deployed, relay_dir, name,
                                                                 prompt=ERROR_PROMPT.format(error=error),
                                                                 json_output=True)
                            unrecovered = None
                    if finalizing:
                        reason = 'episode_ended'
        except ExecutionDeadline:
            reason = 'finalization_timeout' if finalizing or (gateway and gateway.terminal) else 'timeout'
        finally:
            if started is not None:
                self.execution_seconds = time.monotonic() - started
            subprocess.run(['docker', 'kill', name], capture_output=True, timeout=20)
            if process and process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=20)
            requests = relay.calls if relay else 0
            if relay:
                relay.close()
            evidence = self.keep_evidence(output, deployed) if deployed else {}
            (output/'stdout.log').touch()
            (output/'stderr.log').touch()
            size = (output/'stdout.log').stat().st_size
            (output/'logs.json').write_text(json.dumps({'stdout': {'bytes_received': size, 'bytes_saved': size,
                                                                   'truncated': False}}))
        if reason != 'exit':
            unrecovered = None
        # A copy given up on an error ended on it, not by its own choice: the episode is an error.
        return {'reason': reason, 'returncode': returncode if unrecovered else episode_returncode(requests, returncode),
                **({'error': unrecovered} if unrecovered else {}),
                'agent_returncode': returncode, 'logs': {},
                'simulator_exited': gateway and gateway.simulator_exited, 'output_error': None,
                'agent_cli': self.config.agent_cli, 'formal_context': self.config.formal_context,
                'agent_model_requests': requests, 'agent_errors': errors, **evidence}

    def serve(self, listener, endpoint, process, gateway):
        """Dispatch container clients on this (deadline-owning) thread until the agent exits."""
        finalizing = False
        listener.setblocking(False)
        buffers = {}
        with selectors.DefaultSelector() as poll:
            poll.register(listener, selectors.EVENT_READ)
            try:
                while process.poll() is None:
                    for key, _ in poll.select(timeout=1):
                        connection = key.fileobj
                        if connection is listener:
                            try:
                                client, _ = listener.accept()
                            except BlockingIOError:
                                continue
                            client.setblocking(False)
                            buffers[client] = bytearray()
                            poll.register(client, selectors.EVENT_READ)
                            continue
                        try:
                            block = connection.recv(65536)
                            if not block:
                                raise ConnectionError('closed')
                            pending = buffers[connection]
                            pending.extend(block)
                            if len(pending) > MAX_REQUEST_BYTES:
                                raise ConnectionError('Request exceeds wire limit')
                            while b'\n' in pending:
                                line, _, tail = pending.partition(b'\n')
                                pending[:] = tail
                                try:
                                    response = endpoint.handle(json.loads(line))
                                except (ValueError, UnicodeError):
                                    response = {'jsonrpc': '2.0', 'id': None,
                                                'error': {'code': -32700, 'message': 'Invalid request'}}
                                if response is not None:
                                    connection.settimeout(5)
                                    connection.sendall(json.dumps(response, allow_nan=False).encode()+b'\n')
                                    connection.setblocking(False)
                        except (ConnectionError, OSError):
                            poll.unregister(connection)
                            buffers.pop(connection, None)
                            connection.close()
                    if gateway.terminal and not finalizing:
                        finalizing = True
                        finalize_within(self.finalization_seconds)
            finally:
                for client in buffers:
                    client.close()
        return finalizing or bool(gateway.terminal)

    def keep_evidence(self, output, deployed):
        """Keep the episode's transcript and notes, then drop its private storage. A failure to keep
        them is recorded whole beside the episode's task outcome, which it does not change."""
        result = {}
        home = deployed[self.cli.HOME_KEY]
        try:
            copy_regular(home, output/'files/home', budget={'files': 2000, 'bytes': 512 * 1024**2})
            copy_regular(deployed['workspace']/'memory', output/'files/memory',
                         budget={'files': 2000, 'bytes': 64 * 1024**2})
            baseline = None if self.config.formal_context == 'clear' else deployed.get('baseline_usage')
            # From the live home: a resumed conversation's rollout (gigabytes) is over the kept copy's budget.
            result['agent_token_usage'] = agent_usage(self.config.agent_cli, home, output, baseline)
        except Exception as exc:
            result['infrastructure_errors'] = [public_failure(exc, operation='keep_evidence', stage='evidence_copy')]
        finally:
            try:
                workspace_storage.unmount(output)
            finally:
                if not workspace_storage.mounted(output/'agent-storage'):
                    (output/'agent-storage.ext4').unlink(missing_ok=True)
        return result


def run_error(cli, log, offset, home):
    """The error a copy's failed run ended on, whole: Claude Code's newest API error in its conversation;
    Codex's last error event in the run's part of its event stream (from offset), read from its tail."""
    if cli == 'claude':
        from harness.claude_cli.auto_research import last_error
        return last_error(home)
    error = None
    with open(log, 'rb') as stream:
        stream.seek(max(offset, stream.seek(0, 2) - 65536))
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue  # The tail's cut first line.
            if not isinstance(event, dict):
                continue
            if event.get('type') == 'error':
                error = event.get('message') or error
            elif event.get('type') == 'turn.failed':
                error = (event.get('error') or {}).get('message') or error
    return error


def episode_returncode(requests, returncode):
    """The agent's own exit (giving up, ending its turn) is its result, not an infrastructure
    fault: once it has reached the model, the native evaluation decides the outcome. A copy
    that never made a model call keeps its exit code and counts as an error."""
    return 0 if requests and returncode is not None else returncode


def agent_usage(cli, home, output, baseline=None):
    """New tokens spent in this episode, excluding a resumed transcript's earlier usage."""
    if cli == 'claude':
        # The episode's own relay log holds only this episode's calls.
        from services.dashboard.sessions import relay_usage
        usage = relay_usage(output/'model-usage.jsonl')
        return usage and usage['token_usage']
    from services.codex_usage import CodexUsageReader, KEYS
    usage = None
    for rollout in sorted(Path(home, 'sessions').rglob('rollout-*.jsonl'), key=lambda p: p.stat().st_mtime):
        try:
            usage = CodexUsageReader().read(rollout)['total_session_usage']
        except (OSError, ValueError, TypeError, KeyError):
            continue
    if usage is None:
        return None
    baseline = baseline or {}
    return {key: max(0, usage.get(key, 0) - baseline.get(key, 0)) for key in KEYS if key in usage}
