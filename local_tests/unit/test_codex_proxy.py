"""The native model client retains the operator's network route, not arbitrary env."""
import json
from types import SimpleNamespace

from harness.codex_cli.auto_research import agent_environment


def test_model_client_inherits_proxy_and_bypass_list_only(tmp_path, monkeypatch):
    for name, value in {
        'HTTPS_PROXY': 'http://127.0.0.1:40933',
        'https_proxy': 'http://127.0.0.1:40933',
        'ALL_PROXY': 'socks5h://127.0.0.1:40933',
        'NO_PROXY': 'localhost,127.0.0.1',
        'UNRELATED_TOKEN': 'must-not-leak',
    }.items():
        monkeypatch.setenv(name, value)
    login = tmp_path/'login.json'
    login.write_text(json.dumps({'chatgpt_auth': '/operator/auth.json'}))
    deployed = {'workspace': tmp_path/'workspace', 'codex_home': tmp_path/'codex-home',
                'codex': '/operator/node/bin/codex', 'login': login}
    env = agent_environment(SimpleNamespace(training=SimpleNamespace(cpus=2)),
                            deployed, tmp_path/'mcp.sock')
    assert env['HTTPS_PROXY'] == env['https_proxy'] == 'http://127.0.0.1:40933'
    assert env['ALL_PROXY'] == 'socks5h://127.0.0.1:40933'
    assert env['NO_PROXY'] == 'localhost,127.0.0.1'
    assert 'UNRELATED_TOKEN' not in env
    assert '/operator/node/bin' in env['PATH'].split(':')
