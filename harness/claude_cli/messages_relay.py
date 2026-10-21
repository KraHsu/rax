"""Host-only Anthropic Messages endpoint for Claude Code. Credentials never enter Docker.

Claude Code in the container talks to ANTHROPIC_BASE_URL=http://127.0.0.1:17777,
which container_bridge.py forwards to this relay's Unix socket. The relay adds the
provider credential and enforces the same controls as the Codex Responses relay:
a model allowlist, a model-call count, local (client-executed) tools only, inline
media only, bounded bodies, and provider throttling retried before any byte reaches
the client. Token usage is appended to a host-side JSONL for the operator.

Providers:
- ``openrouter``: https://openrouter.ai/api/v1/messages with the OpenRouter key.
- ``anthropic``: https://api.anthropic.com/v1/messages with an API key (x-api-key).
- ``claude-login``: https://api.anthropic.com with the dedicated experiment account's
  long-lived ``claude setup-token`` token (harness/claude_cli/experiment_login.py),
  re-read for every request so a rotated token takes effect at once. The operator's
  personal Claude login (~/.claude) is refused.
"""
import gzip
from http.server import BaseHTTPRequestHandler
import http.client
import io
import json
import os
import re
import stat
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from harness.codex_cli.model_relay import (MAX_ERROR_BODY, MIN_SLOTS, RESPONSE_OWN, CallCounter, UnixHTTPServer,
                                           passed_headers, refusal, retry_delay)

MAX_BODY = 256*1024*1024
REQUEST_TIMEOUT_SECONDS = 120
RETRY_STATUSES = frozenset({429, 502, 503, 504, 529})
RETRY_WINDOW_SECONDS = 600
MAX_OUTPUT_TOKENS = 128000
OAUTH_BETA = 'oauth-2025-04-20'
# model: the agent; auditor: Claude Code's `sonnet` alias, for cheaper review-style
# subagents; fast: the `haiku` alias (background tasks). Sonnet 5.5 is not served
# to the Claude login (checked 2026-09-29), so first-party providers use Haiku 4.5 there.
PROVIDERS = {
    'openrouter': {'base_url': 'https://openrouter.ai/api', 'model': 'anthropic/claude-opus-5.5',
                   'auditor': 'anthropic/claude-sonnet-5.5', 'fast': 'anthropic/claude-haiku-4.5'},
    'anthropic': {'base_url': 'https://api.anthropic.com', 'model': 'claude-opus-5-5',
                  'auditor': 'claude-haiku-4-5-20251001', 'fast': 'claude-haiku-4-5-20251001'},
    'claude-login': {'base_url': 'https://api.anthropic.com', 'model': 'claude-opus-5-5',
                     'auditor': 'claude-haiku-4-5-20251001', 'fast': 'claude-haiku-4-5-20251001'},
}
PROVIDERS['custom'] = {'base_url': None, 'model': 'deepseek-flash', 'auditor': 'deepseek-flash', 'fast': 'deepseek-flash'}
ROUTES = {'/v1/messages': True, '/v1/messages/count_tokens': False}  # path -> counts as a model call
# Requests and responses pass through whole, as between Claude Code and its provider natively
# (see the Codex relay). The relay replaces only the client's placeholder credential.
CLIENT_CREDENTIALS = ('authorization', 'x-api-key')
MODEL_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/@-]{0,199}')


def _private_text(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'r') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError(f'{path} must be a private, owner-readable regular file')
        return stream.read(1 << 20)


def personal_login(path):
    """True for the operator's own Claude login, which experiments must never use."""
    path = Path(path).expanduser()
    personal = (Path.home() / '.claude').resolve()
    resolved = path.resolve()
    return (path.name == '.credentials.json' or resolved == personal
            or personal in resolved.parents or resolved == (Path.home() / '.claude.json').resolve())


class Credential:
    """Provider auth headers, resolved per request so a rotated token applies at once."""

    def __init__(self, provider, *, key_file=None, base_url=None):
        if provider not in PROVIDERS:
            raise ValueError(f'Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}')
        self.provider, self.key_file = provider, key_file
        if provider == 'custom':
            from harness.claude_cli.login import Credential as NativeCredential
            self.native = NativeCredential(provider, key_file=key_file, base_url=base_url)
        if provider == 'claude-login':
            if not key_file:
                raise ValueError('claude-login needs the experiment setup-token file; '
                                 'run scripts/setup_claude_experiment_login.sh')
            if personal_login(key_file):
                raise ValueError('claude-login refuses the personal Claude login; '
                                 'use the dedicated experiment token')
        self.headers()  # Fail at startup, not on the first model call.

    def _token(self):
        token = _private_text(self.key_file).strip()
        if token.startswith('{') and 'claudeAiOauth' in token:
            raise ValueError('claude-login refuses a Claude login credentials file; '
                             'use the dedicated experiment token')
        if not token or len(token) > 16384 or any(c.isspace() for c in token):
            raise ValueError('Invalid provider credential')
        return token

    def headers(self):
        if self.provider == 'custom':
            return self.native.headers()
        if self.provider == 'anthropic':
            key = _private_text(self.key_file).strip() if self.key_file else os.environ.get('ANTHROPIC_API_KEY', '')
            if not key:
                raise ValueError('The anthropic provider needs --claude-key-file or ANTHROPIC_API_KEY')
            return {'x-api-key': key}
        if self.provider == 'openrouter':
            key = _private_text(self.key_file).strip() if self.key_file else os.environ.get('OPENROUTER_API_KEY', '')
            if not key:
                raise ValueError('The openrouter provider needs --key-file or OPENROUTER_API_KEY')
            return {'Authorization': 'Bearer ' + key}
        return {'Authorization': 'Bearer ' + self._token()}


def _media_is_inline(value):
    """No URL or Files-API sources: the provider must never fetch on the agent's behalf."""
    if isinstance(value, list):
        return all(_media_is_inline(item) for item in value)
    if isinstance(value, dict):
        source = value.get('source')
        if isinstance(source, dict) and source.get('type') not in ('base64', 'text', 'content'):
            return False
        return all(_media_is_inline(item) for item in value.values())
    return True


def local_tools(tools):
    """Only client-executed tools; server tools (web search/fetch, code execution) are refused."""
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') not in (None, 'custom'):
            return False
        if not isinstance(tool.get('name'), str):
            return False
    return True


def refused(payload, allowed):
    """Why the relay refuses a request body, or None when it may go to the provider."""
    if not isinstance(payload, dict):
        return 'Request body must be a JSON object'
    if payload.get('model') not in allowed:
        return f'Model {payload.get("model")!r} is not served to this session; allowed: {", ".join(sorted(allowed))}'
    if any(key in payload for key in ('mcp_servers', 'container')):
        return 'Provider-side MCP servers and containers are not allowed'
    if not local_tools(payload.get('tools', [])):
        return 'Only client-executed tools are allowed (no server tools such as web search or code execution)'
    if not (_media_is_inline(payload.get('messages', [])) and _media_is_inline(payload.get('system', []))):
        return 'Only inline (base64) media is allowed; no URL or Files API sources'
    return None


def acceptable(payload, allowed):
    return refused(payload, allowed) is None


def merged_beta(headers, beta):
    """``headers`` with ``beta`` added to its anthropic-beta list (header names in any case)."""
    names = [name for name in headers if name.lower() == 'anthropic-beta']
    values = [part.strip() for name in names for part in headers.pop(name).split(',') if part.strip()]
    headers['anthropic-beta'] = ','.join(dict.fromkeys([*values, beta]))
    return headers


class UsageLog:
    """Parses usage from a Messages response (SSE or JSON) and appends one JSONL row."""

    def __init__(self, path):
        self.path, self.lock = (Path(path) if path else None), threading.Lock()

    def record(self, model, usage):
        if self.path is None or not usage:
            return
        row = {'time': time.time(), 'model': model}
        for key in ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'):
            if isinstance(usage.get(key), int):
                row[key] = usage[key]
        split = usage.get('cache_creation')  # Cache writes by lifetime; 1h writes cost more.
        if isinstance(split, dict):
            for key in ('ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens'):
                if isinstance(split.get(key), int):
                    row['cache_creation_' + key.split('_')[1]] = split[key]
        if isinstance(usage.get('cost'), (int, float)):
            row['cost_usd'] = usage['cost']
        with self.lock, open(self.path, 'a') as stream:
            stream.write(json.dumps(row) + '\n')


class StreamUsage:
    """Incremental SSE scan: message_start carries input usage, message_delta the output."""

    def __init__(self):
        self.buffer, self.usage = b'', {}

    def feed(self, block):
        self.buffer += block
        *lines, self.buffer = self.buffer.split(b'\n')
        for line in lines:
            if not line.startswith(b'data:') or b'usage' not in line:
                continue
            try:
                event = json.loads(line[5:])
            except ValueError:
                continue
            usage = (event.get('message') or {}).get('usage') if event.get('type') == 'message_start' \
                else event.get('usage')
            if isinstance(usage, dict):
                self.usage.update({k: v for k, v in usage.items() if v is not None})


def usage_reset(response):
    """Epoch at which an exhausted subscription usage window reopens; None for an ordinary 429."""
    if response.status != 429 or response.getheader('anthropic-ratelimit-unified-status') != 'rejected':
        return None
    try:
        return float(response.getheader('anthropic-ratelimit-unified-reset'))
    except (TypeError, ValueError):
        return None


class MessagesRelay:
    def __init__(self, path, credential, *, base_url, models, client_write_timeout=120,
                 usage_log=None, calls_file=None, slots=MIN_SLOTS, allow_private_http=False):
        allowed = set()
        for model in models:
            if not isinstance(model, str) or not MODEL_ID.fullmatch(model):
                raise ValueError('Invalid operator model identifier')
            allowed.add(model)
        url = urlsplit(base_url)
        if url.scheme not in {'https', 'http'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Invalid operator model endpoint')
        if url.scheme == 'http' and not allow_private_http and url.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('Non-TLS endpoint is only allowed for local tests')
        self.url, self.credential = url, credential
        self.counter, self.lock = CallCounter(calls_file), threading.Lock()
        self.usage = UsageLog(usage_log)
        self.limited_until = 0.0  # Subscription usage window exhausted until this epoch.
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass  # Never log headers, prompts, provider failures or credentials.

            def reject(self, status, reason, kind='permission_error'):
                """The relay's own answer, with an Anthropic error body naming the reason."""
                body = refusal(reason, kind)
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_POST(self):
                upstream, started = None, False
                try:
                    self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)
                    parts = urlsplit(self.path)
                    if parts.path not in ROUTES or parts.query not in ('', 'beta=true'):
                        self.reject(403, f'The relay serves only POST {" and ".join(ROUTES)}, not {self.path}')
                        return
                    if self.headers.get('Transfer-Encoding'):
                        self.reject(411, 'Send the request body with Content-Length, not Transfer-Encoding',
                                    'invalid_request_error')
                        return
                    length = int(self.headers.get('Content-Length', '-1'))
                    if not 0 < length <= MAX_BODY:
                        self.reject(413, f'Request body must be 1..{MAX_BODY} bytes', 'request_too_large')
                        return
                    body = self.rfile.read(length)
                    if self.headers.get('Content-Encoding') == 'gzip':
                        with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                            body = stream.read(MAX_BODY+1)
                    elif self.headers.get('Content-Encoding'):
                        self.reject(415, 'Only gzip request encoding is accepted', 'invalid_request_error')
                        return
                    if len(body) > MAX_BODY:
                        self.reject(413, f'Decoded request body exceeds {MAX_BODY} bytes', 'request_too_large')
                        return
                    payload = json.loads(body)
                    if reason := refused(payload, allowed):
                        self.reject(403, reason)
                        return
                    counted = ROUTES[parts.path]
                    if relay.limited_until > time.time():
                        # Retrying would only wait out the window; let the launcher pause instead.
                        error = json.dumps({'type': 'error', 'error': {'type': 'rate_limit_error',
                                            'message': 'Usage limit reached; the session resumes after it resets'}}).encode()
                        self.send_response(429)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(error)))
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.wfile.write(error)
                        self.close_connection = True
                        return
                    if counted:
                        maximum = payload.get('max_tokens')
                        if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT_TOKENS:
                            self.reject(400, f'max_tokens must be an integer 1..{MAX_OUTPUT_TOKENS}',
                                        'invalid_request_error')
                            return
                    try:
                        auth = relay.credential.headers()
                    except (OSError, ValueError, RuntimeError) as exc:
                        self.reject(401, f'Relay provider credential unavailable: {type(exc).__name__}: {exc}',
                                    'authentication_error')
                        return
                    own = {**auth, 'Content-Type': 'application/json', 'Accept-Encoding': 'identity'}
                    headers = {**dict(passed_headers(self.headers.items(), {*own, *CLIENT_CREDENTIALS})), **own}
                    if not any(name.lower() == 'accept' for name in headers):
                        headers['Accept'] = 'application/json'
                    if relay.credential.provider == 'claude-login':
                        merged_beta(headers, OAUTH_BETA)
                    if not any(name.lower() == 'anthropic-version' for name in headers):
                        headers['anthropic-version'] = '2023-06-01'
                    connection = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
                    target = url.path.rstrip('/') + parts.path + (f'?{parts.query}' if parts.query else '')
                    deadline, attempt = time.monotonic()+RETRY_WINDOW_SECONDS, 0
                    while True:
                        upstream = connection(url.hostname, url.port, timeout=600)
                        upstream.request('POST', target, body, headers)
                        response = upstream.getresponse()
                        if response.status not in RETRY_STATUSES:
                            break
                        if (reset := usage_reset(response)) is not None:
                            relay.limited_until = max(relay.limited_until, reset)
                            break
                        delay = retry_delay(attempt, response.getheader('Retry-After'))
                        if time.monotonic()+delay >= deadline:
                            break
                        upstream.close()
                        time.sleep(delay)
                        attempt += 1
                    if counted:
                        with relay.lock:  # One model call the provider answered.
                            relay.counter.increment()
                    if not 400 <= response.status <= 599 and response.status != 200:
                        # Redirects are never followed.
                        self.reject(502, f'The provider answered {response.status} {response.reason}; '
                                         'the relay never follows redirects', 'api_error')
                        return
                    if response.status != 200:
                        # The provider's own status, headers and error message: Claude Code needs
                        # the error type (e.g. context too long) and the rate-limit headers to react.
                        error = response.read(MAX_ERROR_BODY)
                        started = True
                        self.send_response(response.status)
                        for name, value in passed_headers(response.getheaders(), RESPONSE_OWN):
                            self.send_header(name, value)
                        self.send_header('Content-Length', str(len(error)))
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.wfile.write(error)
                        self.close_connection = True
                        return
                    # A paused container must not truncate a valid response (see the Codex relay).
                    self.connection.settimeout(client_write_timeout)
                    started = True
                    self.send_response(200)
                    for name, value in passed_headers(response.getheaders(), RESPONSE_OWN):
                        self.send_header(name, value)
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    streaming = 'text/event-stream' in (response.getheader('Content-Type') or '')
                    scan, whole, total = StreamUsage(), [], 0
                    while block := response.read1(65536):
                        total += len(block)
                        if total > MAX_BODY:
                            break
                        if streaming:
                            scan.feed(block)
                        elif total <= 1 << 20:
                            whole.append(block)
                        self.wfile.write(block)
                        self.wfile.flush()
                    if counted:
                        usage = scan.usage
                        if not streaming and whole:
                            try:
                                usage = json.loads(b''.join(whole)).get('usage') or {}
                            except ValueError:
                                usage = {}
                        relay.usage.record(payload.get('model'), usage)
                except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException) as exc:
                    self.close_connection = True
                    if not started:
                        # Before any response byte: answer with the cause (see the Codex relay).
                        provider = upstream is not None
                        reason = (f'The relay could not complete the provider call: {type(exc).__name__}: {exc}'
                                  if provider else f'Invalid request: {type(exc).__name__}: {exc}')
                        try:
                            self.reject(502 if provider else 400, reason,
                                        'api_error' if provider else 'invalid_request_error')
                        except OSError:
                            pass  # The client is gone.
                finally:
                    if upstream:
                        upstream.close()

            def do_GET(self):
                self.reject(403, f'The relay serves only POST {" and ".join(ROUTES)}')
            do_CONNECT = do_GET

        self.server = UnixHTTPServer(str(path), Handler, slots=slots)
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def calls(self):
        return self.counter.value

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
