"""Private-file credentials; vendor account resolution stays outside the agent."""
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from harness.claude_cli.login import private_text


class Credential:
    def __init__(self, spec, provider, *, key_file=None, auth_home=None, base_url=None, binary=None):
        if provider not in spec.PROVIDERS:
            raise ValueError(f'{spec.NAME}: choose a provider from {spec.PROVIDERS}')
        self.spec, self.provider, self.binary = spec, provider, binary
        self.account = provider in getattr(spec, 'ACCOUNTS', (spec.ACCOUNT,))
        self.key_file = Path(key_file).resolve() if key_file and not self.account else None
        self.auth_home = Path(auth_home).resolve() if auth_home else None
        if self.account:
            if not self.auth_home or self.auth_home == Path.home():
                raise ValueError('Account mode requires a dedicated experiment --agent-auth-home')
            if not self.auth_home.is_dir() or self.auth_home.stat().st_uid != os.getuid():
                raise ValueError('Experiment auth home must exist and belong to the current user')
            if self.auth_home.stat().st_mode & 0o077:
                raise ValueError('Experiment auth home must have mode 0700')
        elif not self.key_file or not private_text(self.key_file).strip():
            raise ValueError('API mode requires a nonempty private --agent-key-file')
        self.endpoint = (base_url if provider == 'custom' else
                         'https://openrouter.ai/api/v1' if provider == 'openrouter' else
                         spec.endpoint(provider, self.auth_home) if hasattr(spec, 'endpoint') else
                         spec.ACCOUNT_ENDPOINT if self.account else spec.API_ENDPOINT)
        parsed = urlsplit(self.endpoint or '')
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Expected an HTTP(S) API base URL without embedded credentials')

    def record(self):
        return {'provider': self.provider, 'key_file': str(self.key_file) if self.key_file else None,
                'auth_home': str(self.auth_home) if self.auth_home else None, 'base_url': self.endpoint}

    def headers(self):
        if not self.account:
            return {'Authorization': 'Bearer '+private_text(self.key_file).strip()}
        # The vendor helper uses its native credential store and refresh logic.
        # Its stdout is consumed here, never logged or archived in session state.
        result = subprocess.run(self.spec.account_command(self.binary, self.auth_home, self.provider),
            capture_output=True, text=True, timeout=120, env=self.spec.account_environment(self.auth_home))
        if result.returncode:
            raise RuntimeError('Vendor account credential resolution failed; run the dedicated account login')
        value = json.loads(result.stdout)
        return value['headers']
