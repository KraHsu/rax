from types import SimpleNamespace

import pytest

from services.robotwin.native_compat import configure_assets, adapt_reward_constructor


def test_asset_root_is_parent_of_assets_and_explicit_override_is_preserved(tmp_path, monkeypatch):
    monkeypatch.delenv('ASSETS_PATH', raising=False)
    (tmp_path/'assets/objects').mkdir(parents=True)
    assert configure_assets(tmp_path) == str(tmp_path)
    monkeypatch.setenv('ASSETS_PATH', '/external/robotwin')
    assert configure_assets(tmp_path) == '/external/robotwin'


def test_serial_constructor_preserves_native_factory_and_rejects_unknown_arguments():
    class Factory:
        @staticmethod
        def build_top(config):
            return SimpleNamespace(config=config)
    module = SimpleNamespace(Reward=Factory)
    tasks = [object(), object(), object()]
    assert adapt_reward_constructor(module)
    result = module.Reward(subtasks=tasks, transition_rewards=[1, 3])
    assert result.config == {'type': 'Serial', 'subtasks': tasks, 'transition_rewards': [1, 3]}
    assert result.config['subtasks'] is tasks
    assert module.Reward.build_top({'type': 'Parallel'}).config == {'type': 'Parallel'}
    assert module.Reward(subtasks=tasks).config == {'type': 'Serial', 'subtasks': tasks}
    with pytest.raises(TypeError):
        module.Reward(subtasks=tasks, unsupported=True)
    adapted = module.Reward
    assert adapt_reward_constructor(module)
    assert module.Reward is adapted


def test_real_constructor_or_missing_factory_is_not_replaced():
    class Existing:
        def __init__(self, value): self.value = value
        @staticmethod
        def build_top(config): return config
    module = SimpleNamespace(Reward=Existing)
    assert not adapt_reward_constructor(module)
    assert module.Reward(5).value == 5
    module.Reward = type('NoFactory', (), {})
    assert not adapt_reward_constructor(module)
