"""Hooks for the existing RAX exploration and frozen-program evaluation flow."""
from services.robotwin.catalog import check_task, configured_manifest, step_limit, task_names


def tool_definitions():
    from services.robotwin.tools import tool_definitions
    return tool_definitions()


def backend(config):
    from services.robotwin.backend import RoboTwinBackend
    return RoboTwinBackend(config)


def task_timeouts(task):
    return {'native_step_limit': step_limit(task), 'episode_seconds': None, 'startup_seconds': 300}


def validate_config(config):
    check_task(config.task)
    if config.observation_profile != 'rgb-only':
        raise ValueError('Initial RoboTwin adapter uses observation_profile=rgb-only')
    if (config.eval_seed, config.formal_collection) != (0, 1):
        raise ValueError('RoboTwin C2R uses clean collection 0 and randomized formal collection 1')
    if config.demonstration_context != 'none' or config.task_rules != 'public':
        raise ValueError('RoboTwin uses manifest instructions and demonstration_context=none')
    if config.agent_formal or config.adaptation_source or config.write_handoff:
        raise ValueError('Initial C2R adapter evaluates frozen code; use code mode without adaptation/handoff')


def validate_settings(values):
    if values.get('agent_formal') not in (None, False, 'off'):
        raise ValueError('RoboTwin C2R uses frozen-code evaluation')


def validate_layouts(config):
    manifest = configured_manifest(config.task)
    for index in config.exploration_seeds:
        manifest.entry(0, index)
    for index in range(config.formal_seed, config.formal_seed + config.formal_episodes):
        manifest.entry(1, index)


def exploration_seeds(task, collection, start, episodes, **_):
    if collection != 0 or start < 0 or episodes < 1:
        raise ValueError('Exploration requires clean collection 0 and valid episode indices')
    return tuple(range(start, start + episodes))


def task_context(config, demonstration, *, live_formal):
    return (f'# RoboTwin task: {config.task}\n\n'
            'Read the episode instruction and measured observations. The native task success check is the evaluator.\n'
            'Explore clean episodes and compile a controller for the held-out randomized scenes.\n'
            'Write composable geometric constraints through robotwin_constraints while recording clean data. '
            'Use observation frame indices, justified feasible bounds and explicit coordinate frames. '
            'No operation category or simulator boolean is used as a training loss.\n')


def refresh_workspace(config, workspace):
    pass


def fetch_demonstration(demos_root, task, kind):
    if kind != 'none':
        raise ValueError('RoboTwin demonstration provisioning is not configured')


def provision_demonstration(config, target, demos_root):
    fetch_demonstration(demos_root, config.task, config.demonstration_context)
    return {'kind': 'none', 'image_count': 0}


def episode_reviewer(root):
    return None
