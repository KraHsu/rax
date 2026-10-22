"""Simulator backends behind the shared harness: one registry entry per benchmark.

Pure configuration, like ``services.mcp_contract``: no simulator or heavy imports at module
load. Each benchmark is a peer: an entry here holds its data (robot tool names, robot facts,
setup defaults and fixed fields, capabilities), and its adapter module
(``services/<benchmark>/benchmark.py``) implements the hooks the shared harness calls. Shared
code dispatches through the entry and never compares benchmark names. The benchmark is fixed
at session creation; configurations saved before the field existed are RoboDojo (DEFAULT).
"""
from dataclasses import dataclass, field
import importlib

DEFAULT = 'robodojo'  # Saved configurations and manifests without a benchmark field.

ROBODOJO_TOOLS = frozenset({
    'robodojo_status', 'robodojo_observe', 'robodojo_pixel_to_position',
    'robodojo_pose_math', 'robodojo_step', 'robodojo_step_eef',
    'robodojo_free_space_move', 'robodojo_execute_motion_plan', 'robodojo_fk_preview',
    'robot_preview_describe', 'robot_preview_run',
})
BEHAVIOR_TOOLS = frozenset({
    'behavior_status', 'behavior_observe', 'behavior_step', 'behavior_base_move', 'behavior_step_eef',
    'behavior_free_space_move', 'behavior_execute_motion_plan', 'behavior_pixel_to_position',
    'behavior_preview_describe', 'behavior_preview_run',
})
ROBOTWIN_TOOLS = frozenset({'robotwin_status', 'robotwin_observe', 'robotwin_step',
                           'robotwin_step_eef', 'robotwin_constraints'})


@dataclass(frozen=True, eq=False)  # Identity: entries are singletons with dict fields.
class Benchmark:
    name: str
    label: str                      # Display name: 'RoboDojo', 'BEHAVIOR'.
    adapter: str                    # Module implementing HOOKS.
    robot_tools: frozenset
    geometry_tools: frozenset       # Removed by the RGB-only profile.
    unscoped_tools: frozenset       # Pure computation: not environment-scoped.
    mutating: frozenset             # Tools that may advance physics.
    previewable: frozenset          # preview_only=true makes these planning-only.
    status_tool: str
    observe_tool: str
    tool_prefixes: tuple
    episode_tool_prefixes: tuple    # Tools whose replies carry the live episode's step_id.
    simulator_modules: tuple        # Simulator server modules (the episode server first).
    startup_seconds: int            # Environment creation deadline (seconds).
    # Words shared workspace sources use for the simulated robot instead of branching on it.
    robot: dict = field(default_factory=dict)
    partial_credit: bool = False    # Episodes are scored with partial credit, which workspaces may name.
    wall_limit: bool = True         # Episodes have a wall-clock limit (task_timeouts episode_seconds).
    terminal_tools: frozenset = frozenset()  # Non-physics operations allowed before finalizing a terminal episode.
    episode_reviews: bool = False   # Simulator ground truth supports exploration episode reviews.
    # Live copies of a code-mode session get reports of its code's formal runs (replayed frames,
    # reviews; services/agent_formal/code_runs.py).
    code_run_reports: bool = False
    # Planning and stepping sources (project-relative) an offline parallel formal batch
    # fingerprints, so a change to them during the batch stops it.
    planner_files: tuple = ()
    # Regex of this benchmark's own robot, camera, tool and skill names, which another
    # benchmark's workspace never mentions ('' for none).
    vocabulary: str = ''
    task_tag: str = ''              # Suffix of the task in the dashboard's task menu.
    score_label: str = 'Formal average score'
    layout_note: str = ''           # Dashboard note on the layouts exploration and formal use.
    # Setup values applied when the dashboard's task switches to this benchmark (editable).
    setup_defaults: dict = field(default_factory=dict)
    # configure_auto_research.py defaults for flags the operator did not pass.
    cli_defaults: dict = field(default_factory=dict)
    # Setup fields that do not apply: name -> (fixed value, reason).
    fixed: dict = field(default_factory=dict)

    def _hooks(self):
        return importlib.import_module(self.adapter)

    def __getattr__(self, name):
        # The adapter's hooks (HOOKS), loaded only when first called.
        if name not in HOOKS:
            raise AttributeError(name)
        return getattr(self._hooks(), name)

    def planning_only(self, name, arguments):
        return name in self.previewable and isinstance(arguments, dict) and arguments.get('preview_only') is True

    def startup_limit(self, task):
        """Optional operator configuration; adapters otherwise keep their default."""
        hook = getattr(self._hooks(), 'startup_limit', None)
        return hook(task) if hook else self.startup_seconds

    def definitions(self):
        return [t for t in self._hooks().tool_definitions() if t['name'] in self.robot_tools]

    def apply_fixed(self, values):
        """values with this benchmark's fixed setup fields."""
        return {**values, **{name: value for name, (value, _) in self.fixed.items()}}


# What every adapter module implements (besides tool_definitions(), the robot tools' schemas):
#   backend(config)                        the controller backend of one environment
#   task_names(), task_timeouts(task)      task catalog and budgets
#   validate_config(config)                benchmark rules of a session configuration
#   validate_settings(values)              benchmark rules of dashboard setup values
#   validate_layouts(config)               formal and exploration layouts exist
#   exploration_seeds(task, collection, start, episodes, *, formal_collection, formal_seed,
#                     formal_episodes)     one layout per exploration episode
#   task_context(config, demonstration, *, live_formal)   TASK.md text
#   refresh_workspace(config, workspace)   update instructions on deploy and resume
#   fetch_demonstration(demos_root, task, kind)           cache visual context before deploy
#   provision_demonstration(config, target, demos_root)   copy it into a workspace
#   episode_reviewer(root)                 reviewer of closed episodes, or None
HOOKS = frozenset({
    'backend', 'task_names', 'task_timeouts', 'validate_config', 'validate_settings', 'validate_layouts',
    'exploration_seeds', 'task_context', 'refresh_workspace', 'provision_demonstration',
    'fetch_demonstration', 'episode_reviewer',
})

BENCHMARKS = {
    'robotwin': Benchmark(
        name='robotwin', label='RoboTwin', adapter='services.robotwin.benchmark',
        robot_tools=ROBOTWIN_TOOLS, geometry_tools=frozenset(), unscoped_tools=frozenset(),
        mutating=frozenset({'robotwin_step', 'robotwin_step_eef'}), previewable=frozenset(),
        status_tool='robotwin_status', observe_tool='robotwin_observe', tool_prefixes=('robotwin_',),
        episode_tool_prefixes=('robotwin_',), simulator_modules=('services.robotwin.server',),
        startup_seconds=300, wall_limit=False,
        terminal_tools=frozenset({'robotwin_constraints'}),
        robot={'benchmark': 'RoboTwin', 'status': 'robotwin_status', 'observe': 'robotwin_observe',
               'step': 'robotwin_step', 'plan': 'robotwin_step_eef', 'pixel': 'not_available',
               'hz': 'variable (one native command per row)', 'overview_camera': 'head_camera',
               'overview': 'head', 'cameras': ('head_camera', 'left_camera', 'right_camera')},
        planner_files=('services/robotwin/session.py', 'services/robotwin/runtime.py',
                       'services/robotwin/tools.py'), vocabulary=r'robotwin_|\bRoboTwin\b',
        task_tag='RoboTwin', layout_note='Clean exploration; frozen programs on randomized held-out episodes.',
        setup_defaults={'formal_episodes': 100, 'formal_workers': 1, 'formal_eval_seed': 1,
                        'exploration_start_seed': 0, 'demonstration_context': 'none'},
        fixed={'episode_reviews': ('off', 'Native results and embedded constraint histories are recorded separately.'),
               'demonstration_context': ('none', 'No demonstration provisioning in the RoboTwin adapter.'),
               'task_rules': ('public', 'Instructions come from the operator-owned episode manifest.'),
               'observation_profile': ('rgb-only', 'Initial RoboTwin adapter exposes RGB and measured proprioception.')},
    ),
    'robodojo': Benchmark(
        name='robodojo', label='RoboDojo', adapter='services.robodojo.benchmark',
        robot_tools=ROBODOJO_TOOLS, geometry_tools=frozenset({'robodojo_pixel_to_position'}),
        unscoped_tools=frozenset({'robodojo_pose_math'}),
        mutating=frozenset({'robodojo_step', 'robodojo_step_eef', 'robodojo_free_space_move',
                            'robodojo_execute_motion_plan'}),
        previewable=frozenset({'robodojo_free_space_move'}), status_tool='robodojo_status',
        observe_tool='robodojo_observe', tool_prefixes=('robodojo_', 'robot_preview_'),
        episode_tool_prefixes=('robodojo_',),
        simulator_modules=('services.robodojo.server', 'services.robot_preview.server'), startup_seconds=240,
        robot={'benchmark': 'RoboDojo', 'status': 'robodojo_status', 'observe': 'robodojo_observe',
               'step': 'robodojo_step', 'plan': 'robodojo_free_space_move', 'pixel': 'robodojo_pixel_to_position',
               'hz': 25,
               'overview_camera': 'cam_high', 'overview': 'high-camera',
               'cameras': ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
        planner_files=('RoboDojo/env/planner_manager/curobo_planner.py', 'services/robodojo/session.py',
                       'services/robodojo/trajectory.py', 'services/simulator/protocol.py'),
        vocabulary=(r'robodojo|robot_preview_|\bcam_high\b|\blink6\b|\bARX\b|pose[- ]math|'
                    r'action-preview|action-search'),
        setup_defaults={'formal_episodes': 50, 'formal_workers': 4, 'formal_eval_seed': 1,
                        'exploration_start_seed': 1, 'demonstration_context': 'terminal_state'},
        fixed={'episode_reviews': ('off', 'Episode reviews read the BEHAVIOR simulator ground truth; RoboDojo has none.')},
    ),
    'behavior': Benchmark(
        name='behavior', label='BEHAVIOR', adapter='services.behavior.benchmark',
        robot_tools=BEHAVIOR_TOOLS, geometry_tools=frozenset({'behavior_pixel_to_position'}),
        unscoped_tools=frozenset(),
        mutating=frozenset({'behavior_step', 'behavior_base_move', 'behavior_step_eef', 'behavior_free_space_move',
                            'behavior_execute_motion_plan'}),
        previewable=frozenset({'behavior_free_space_move'}), status_tool='behavior_status',
        observe_tool='behavior_observe', tool_prefixes=('behavior_',),
        # The BEHAVIOR preview reports the live step as `live_step_id`, so it is never an episode step.
        episode_tool_prefixes=('behavior_',),
        simulator_modules=('services.behavior.server', 'services.behavior.preview_server'), startup_seconds=900,
        robot={'benchmark': 'BEHAVIOR', 'status': 'behavior_status', 'observe': 'behavior_observe',
               'step': 'behavior_step', 'plan': 'behavior_free_space_move', 'pixel': 'behavior_pixel_to_position',
               'hz': 30,
               'overview_camera': 'cam_head', 'overview': 'head-camera',
               'cameras': ('cam_head', 'cam_left_wrist', 'cam_right_wrist')},
        partial_credit=True, episode_reviews=True, wall_limit=False, task_tag='BEHAVIOR', code_run_reports=True,
        planner_files=('BEHAVIOR-1K/OmniGibson/omnigibson/action_primitives/curobo.py', 'services/behavior/planner.py',
                       'services/behavior/session.py', 'services/behavior/robot.py', 'services/simulator/protocol.py'),
        score_label='Formal q-score (2026 challenge metric)',
        layout_note='BEHAVIOR: exploration cycles through 300 train instances; formal uses up to 20 held-out test instances.',
        # Train instances for exploration (cycled); 20 held-out test instances for formal.
        setup_defaults={'formal_episodes': 10, 'formal_workers': 2, 'formal_eval_seed': 1, 'exploration_start_seed': 0,
                        'demonstration_context': 'terminal_state'},
        fixed={
            'observation_profile': ('rgbd', 'The 2026 challenge allows RGB, depth and proprioception: BEHAVIOR uses RGB-D.'),
            'task_rules': ('public', 'TASK.md always gives the official BDDL task definition.'),
            'formal_eval_seed': (1, 'Formal always uses the 20 held-out public test instances.'),
            'write_handoff': ('off', 'Adaptation needs another layout collection; BEHAVIOR has one train collection.'),
            'adaptation_source': ('', 'Adaptation needs another layout collection; BEHAVIOR has one train collection.'),
            'adaptation_context': ('compact', 'Adaptation is unavailable for BEHAVIOR.'),
        },
    ),
}
ALL_ROBOT_TOOLS = frozenset().union(*(b.robot_tools for b in BENCHMARKS.values()))
ALL_MUTATING = frozenset().union(*(b.mutating for b in BENCHMARKS.values()))
ALL_TOOL_PREFIXES = tuple(p for b in BENCHMARKS.values() for p in b.tool_prefixes)
EPISODE_TOOL_PREFIXES = tuple(p for b in BENCHMARKS.values() for p in b.episode_tool_prefixes)
SIMULATOR_MODULES = tuple(m for b in BENCHMARKS.values() for m in b.simulator_modules)
EPISODE_SERVERS = tuple(b.simulator_modules[0] for b in BENCHMARKS.values())
STATUS_TOOLS = frozenset(b.status_tool for b in BENCHMARKS.values())
OBSERVE_TOOLS = frozenset(b.observe_tool for b in BENCHMARKS.values())


def benchmark(name=DEFAULT):
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark {name!r}; choose one of {', '.join(sorted(BENCHMARKS))}") from None


def for_config(config):
    return benchmark(getattr(config, 'benchmark', DEFAULT))


def for_task(task):
    """The benchmark whose catalog has the task; task names are unique across benchmarks."""
    owner = next((b for b in BENCHMARKS.values() if task in b.task_names()), None)
    if owner is None:
        raise ValueError(f'Unknown task {task!r}')
    return owner


def tool_benchmark(name):
    """The benchmark that owns a robot tool name, or None."""
    return next((b for b in BENCHMARKS.values() if name in b.robot_tools), None)


def planning_only(name, arguments):
    owner = tool_benchmark(name)
    return bool(owner and owner.planning_only(name, arguments))
