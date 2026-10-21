"""Provider-specific submission and backward compatibility for existing DSH trials."""
import json
from types import SimpleNamespace

import pytest

from scripts import submit_dsh_experiment as launcher


@pytest.mark.parametrize('provider', ['openrouter', 'deepseek-account', None])
def test_worker_uses_only_the_selected_provider_credential(provider):
    record = {'model': 'deepseek-flash', 'task': 'click_mouse', 'binary': '/tools/dsh',
              'exploration_budget': 16, 'formal_episodes': 150, 'experiment_record': '/run/experiment.json',
              'key_file': '/private/openrouter.key', 'auth_home': '/private/account'}
    if provider is not None:
        record['provider'] = provider
    command = launcher.experiment_command(record, 'GPU-fixture')
    assert command[command.index('--provider')+1] == (provider or 'openrouter')
    if provider == 'deepseek-account':
        assert command[command.index('--agent-auth-home')+1] == '/private/account'
        assert '--key-file' not in command and '/private/openrouter.key' not in command
    else:
        assert command[command.index('--key-file')+1] == '/private/openrouter.key'
        assert '--agent-auth-home' not in command and '/private/account' not in command
    assert command[command.index('--episodes')+1] == '16'
    assert command[command.index('--formal-episodes')+1] == '150'


@pytest.mark.parametrize('provider', ['openrouter', 'deepseek-account'])
def test_render_matches_validation_and_preserves_non_requeue_settings(tmp_path, monkeypatch, provider):
    key, auth = tmp_path/'api.key', tmp_path/'account'
    key.write_text('fixture-api-credential')
    key.chmod(0o600)
    auth.mkdir()
    store = auth/'.credentials.yaml'
    store.write_text('fixture-native-grant')
    store.chmod(0o600)
    model = launcher.MODEL if provider == 'openrouter' else 'deepseek-flash'
    validation = tmp_path/'validation.json'
    validation.write_text(json.dumps({'status':'passed', 'provider':provider, 'model':model,
                                    'wire_max_effort':True, 'vision_wire':True, 'resume':True}))
    profile = tmp_path/'slurm.json'
    profile.write_text(json.dumps({'account':'fixture', 'experiment':{
        'partition':'regular_gpu', 'cpus':12, 'memory':'96G', 'walltime':'24:00:00', 'gpus':'fixture:1'}}))
    monkeypatch.setattr(launcher, 'dsh_binary', lambda path: tmp_path/'dsh')
    monkeypatch.setattr(launcher.shutil, 'which', lambda name: '/usr/bin/'+name)
    args = SimpleNamespace(provider=provider, key_file=key if provider == 'openrouter' else None,
        auth_home=auth if provider == 'deepseek-account' else None, validation=validation,
        binary=None, profile=profile, root=tmp_path/'trial', task='click_mouse', submit=False)
    launcher.submit(args)
    record = json.loads((args.root/'trial.json').read_text())
    assert record['provider'] == provider and record['model'] == model
    assert record['effort'] == 'max' and record['evaluation_mode'] == 'code_formal'
    assert (record['key_file'] is None) == (provider == 'deepseek-account')
    assert '#SBATCH --no-requeue' in (args.root/'experiment.sbatch').read_text()
    assert 'fixture-api-credential' not in json.dumps(record)
    assert 'fixture-native-grant' not in json.dumps(record)
    args.root = tmp_path/'wrong-validation'
    args.provider = 'deepseek-account' if provider == 'openrouter' else 'openrouter'
    args.key_file, args.auth_home = key, auth
    with pytest.raises(ValueError, match='matching native DSH'):
        launcher.submit(args)
    assert not args.root.exists()
