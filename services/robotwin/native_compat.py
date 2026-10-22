"""Narrow adapters for mixed RoboTwin task/config interfaces.

These do not change expert actions, object sampling, or check_success().
"""
import os


def configure_assets(root):
    """Native task loaders expect the directory containing assets/, not assets/."""
    os.environ.setdefault('ASSETS_PATH', str(root))
    return os.environ['ASSETS_PATH']


def adapt_reward_constructor(module):
    """Map the legacy Serial constructor to the native reward-tree factory.

    place_fan still calls Reward(subtasks=..., transition_rewards=...), while
    the installed Reward exposes build_top(config). Keep the native subtasks,
    transition rewards, normalization, and public factory methods intact.
    A version with its own constructor is deliberately left alone.
    """
    factory = module.Reward
    if getattr(factory, '_rax_serial_constructor', False):
        return True
    if (factory.__init__ is not object.__init__ or factory.__new__ is not object.__new__
            or not callable(getattr(factory, 'build_top', None))):
        return False

    class CompatibleReward(factory):
        _rax_serial_constructor = True

        def __new__(cls, *, subtasks, transition_rewards=None):
            config = {'type': 'Serial', 'subtasks': subtasks}
            if transition_rewards is not None:
                config['transition_rewards'] = transition_rewards
            return factory.build_top(config)

    module.Reward = CompatibleReward
    return True


_report = {}


def compatibility_report():
    return dict(_report)


def install(root, reward_module):
    _report.update(assets_path=configure_assets(root),
                   reward_serial_constructor=adapt_reward_constructor(reward_module))
