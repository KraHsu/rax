"""Claude Code's login for a session: provider defaults and the credential Claude Code uses natively.

Providers:
- ``claude-login``: the dedicated experiment account's long-lived ``claude setup-token``
  token (harness/claude_cli/experiment_login.py), given to Claude Code as
  CLAUDE_CODE_OAUTH_TOKEN. The operator's personal Claude login (~/.claude) is refused.
- ``anthropic``: an Anthropic API key (ANTHROPIC_API_KEY).
- ``openrouter``: OpenRouter's Anthropic-compatible endpoint with the OpenRouter key.
- ``custom``: a private Anthropic-compatible endpoint with its own bearer credential.
"""
import os
import stat
from pathlib import Path
from urllib.parse import urlsplit

OAUTH_BETA = 'oauth-2025-04-20'
# model: the agent; auditor and fast: what Claude Code's `sonnet` and `haiku` aliases name on
# a provider that does not know Anthropic's model ids (OpenRouter), and the model of the
# experiment login's one-token usage probe.
PROVIDERS = {
    'openrouter': {'base_url': 'https://openrouter.ai/api', 'model': 'anthropic/claude-opus-5.5',
                   'auditor': 'anthropic/claude-sonnet-5.5', 'fast': 'anthropic/claude-haiku-4.5'},
    'anthropic': {'base_url': 'https://api.anthropic.com', 'model': 'claude-opus-5-5',
                  'auditor': 'claude-haiku-4-5-20251001', 'fast': 'claude-haiku-4-5-20251001'},
    'claude-login': {'base_url': 'https://api.anthropic.com', 'model': 'claude-opus-5-5',
                     'auditor': 'claude-haiku-4-5-20251001', 'fast': 'claude-haiku-4-5-20251001'},
    'custom': {'base_url': None, 'model': 'deepseek-flash',
               'auditor': 'deepseek-flash', 'fast': 'deepseek-flash'},
}


def private_text(path):
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
    """A provider credential, read from its file whenever it is used (a rotated token applies)."""

    def __init__(self, provider, *, key_file=None, base_url=None):
        if provider not in PROVIDERS:
            raise ValueError(f'Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}')
        self.provider, self.key_file = provider, key_file
        self.base_url = base_url or PROVIDERS[provider]['base_url']
        if provider == 'custom':
            parsed = urlsplit(self.base_url or '')
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError('custom needs --claude-base-url with an HTTP(S) endpoint and no embedded credentials')
        if provider == 'claude-login':
            if not key_file:
                raise ValueError('claude-login needs the experiment setup-token file; '
                                 'run scripts/setup_claude_experiment_login.sh')
            if personal_login(key_file):
                raise ValueError('claude-login refuses the personal Claude login; '
                                 'use the dedicated experiment token')
        self.secret()  # Fail at startup, not on the first model call.

    def secret(self):
        if self.provider == 'claude-login':
            token = private_text(self.key_file).strip()
            if token.startswith('{') and 'claudeAiOauth' in token:
                raise ValueError('claude-login refuses a Claude login credentials file; '
                                 'use the dedicated experiment token')
            if not token or len(token) > 16384 or any(c.isspace() for c in token):
                raise ValueError('Invalid provider credential')
            return token
        variable, flag = {'anthropic': ('ANTHROPIC_API_KEY', '--claude-key-file'),
                          'custom': ('ANTHROPIC_AUTH_TOKEN', '--claude-key-file')}.get(
                              self.provider, ('OPENROUTER_API_KEY', '--key-file'))
        key = private_text(self.key_file).strip() if self.key_file else os.environ.get(variable, '')
        if not key:
            raise ValueError(f'The {self.provider} provider needs {flag} or {variable}')
        return key

    def headers(self):
        """HTTP auth headers, for the harness's own account checks (experiment_login)."""
        if self.provider == 'anthropic':
            return {'x-api-key': self.secret()}
        return {'Authorization': 'Bearer ' + self.secret()}

    def environment(self, *, model=None):
        """The variables that sign Claude Code in natively."""
        if self.provider == 'claude-login':
            return {'CLAUDE_CODE_OAUTH_TOKEN': self.secret()}
        if self.provider == 'anthropic':
            return {'ANTHROPIC_API_KEY': self.secret()}
        provider = PROVIDERS[self.provider]
        environment = {'ANTHROPIC_BASE_URL': self.base_url.rstrip('/'), 'ANTHROPIC_AUTH_TOKEN': self.secret(),
                'ANTHROPIC_API_KEY': '',
                # OpenRouter names Anthropic's models its own way; Claude Code's aliases must too.
                'ANTHROPIC_DEFAULT_OPUS_MODEL': provider['model'],
                'ANTHROPIC_DEFAULT_SONNET_MODEL': provider['auditor'],
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': provider['fast']}
        if model and not model.lstrip('~').startswith('anthropic/'):
            # A non-Claude OpenRouter model must also handle side requests and subagents;
            # otherwise a DeepSeek session silently keeps calling paid Claude models.
            environment.update({name: model for name in (
                'ANTHROPIC_DEFAULT_OPUS_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL', 'CLAUDE_CODE_SUBAGENT_MODEL')})
        return environment
