"""Optional container backends preserve the credential and execution boundaries."""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from harness.cli_runtime import CONTAINER

if not CONTAINER:
    pytest.skip('Container branch contracts', allow_module_level=True)

from harness.backends import get
from test_controller_backend import configuration


@pytest.mark.parametrize('name,provider', [
    ('dsh', 'openrouter'), ('dsh', 'deepseek-api'), ('dsh', 'deepseek-account'), ('dsh', 'custom'),
    ('kimi', 'openrouter'), ('kimi', 'moonshot-api'), ('kimi', 'kimi-account'), ('kimi', 'custom')])
def test_launch_keeps_credentials_on_host_and_agent_inside_container(tmp_path, monkeypatch, name, provider):
    cli = get(name)
    native = importlib.import_module('harness.'+{'dsh': 'deepseek', 'kimi': 'kimi'}[name]+'_cli.auto_research')
    root = tmp_path/'trial'
    root.mkdir()
    config = replace(configuration(root), agent_cli=name, agent_provider=provider, agent_effort='max')
    key = tmp_path/'private-model.key'
    key.write_text('fixture-private-provider-key')
    key.chmod(0o600)
    auth = tmp_path/'private-account'
    auth.mkdir(mode=0o700)
    (auth/'.credentials.yaml').write_text('version: 1\nrefs: {}\nrecords: {}\n')
    (auth/'.credentials.yaml').chmod(0o600)
    install = tmp_path/'install'
    package = install/('node_modules/@deepseek-ai/dsh' if name == 'dsh'
                       else 'node_modules/@moonshot-ai/kimi-code')
    package.mkdir(parents=True)
    (package/'package.json').write_text('{}')
    binary = package/'cli.mjs'
    binary.write_text('// fixture executable\n')
    def prepare(config, home_name):
        config.root.mkdir(parents=True, exist_ok=True)
        workspace, home = config.root/'agent-workspace', config.root/home_name
        workspace.mkdir(); home.mkdir(); (config.root/'published').mkdir()
        return {'workspace': workspace, 'home': home, 'configuration': config.root/'configuration.json'}
    module = native if name == 'dsh' else importlib.import_module('harness.isolated_cli.adapter')
    monkeypatch.setattr(module, 'prepare_workspace', prepare)
    deployed = cli.deploy(config, binary=binary, key_file=key, auth_home=auth,
                          base_url='http://model-host:8000' if provider == 'custom' else None)
    command = cli.agent_command(config, deployed, prompt='Validate transport')
    text = '\n'.join(map(str, command))
    assert command[:2] == ['docker', 'run']
    assert '--network=none' in command and '--read-only' in command and '--cap-drop=ALL' in command
    assert '--memory' in command and '--cpus' in command and '--pids-limit' in command
    assert '--user' in command and '--gpus' not in command
    assert key.read_text() not in text and str(key) not in text and str(auth) not in text
    assert '--entrypoint' in command and config.image in command
    assert any('dst=/workspace/runtime/autonomous_controller,readonly' in str(arg) for arg in command)
    assert any('dst='+str(deployed['workspace']/'runtime/autonomous_controller')+',readonly' in str(arg)
               for arg in command)
    login = json.loads(deployed['login'].read_text())
    assert key.read_text() not in json.dumps(login)
    home_text = '\n'.join(p.read_text() for p in Path(deployed[cli.HOME_KEY]).rglob('*') if p.is_file())
    assert key.read_text() not in home_text and str(auth) not in home_text
    if name == 'dsh':
        assert 'rax-model.invalid' in home_text
        assert 'container_cli/runner.py' in home_text
        assert 'account_headers' not in text
    else:
        assert '/usr/local/bin/python' in home_text


def test_private_messages_endpoint_is_explicit_and_does_not_change_defaults(tmp_path):
    from harness.claude_cli.messages_relay import MessagesRelay
    credential = type('Credential', (), {'provider': 'custom'})()
    with pytest.raises(ValueError, match='Non-TLS'):
        MessagesRelay(tmp_path/'default.sock', credential, base_url='http://model-host:8000', models=['deepseek-flash'])
    relay = MessagesRelay(tmp_path/'local.sock', credential, base_url='http://model-host:8000',
                          models=['deepseek-flash'], allow_private_http=True)
    relay.start()
    relay.close()
