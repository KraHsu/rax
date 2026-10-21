"""The facts a workspace is generated from; roles live in recipes, not here."""
from dataclasses import dataclass, field, replace

from services.benchmarks import BENCHMARKS, benchmark as benchmark_named
from services.mcp_contract import Contract

# Experiment options: 'agent-formal' for an exploring agent in live-copy mode (copies
# of it play the held-out layouts live; exploration has no bundle tools), and
# 'formal-episode' for such a copy during its one held-out episode.
# 'handoff' installs the transfer-handoff skill for adaptation source trials.
OPTIONS = frozenset({'agent-formal', 'formal-episode', 'handoff'})


def wrist_cameras(benchmark):
    """Whether the benchmark's robot has a camera on a wrist."""
    return any('wrist' in camera for camera in benchmark_named(benchmark).robot['cameras'])


@dataclass(frozen=True)
class Setup:
    contract: Contract
    options: frozenset = field(default_factory=frozenset)
    task: str | None = None
    cli: str = 'codex'  # the agent CLI reading the workspace: codex | claude
    task_rules: str = 'public'  # which rubric text TASK.md / AGENTS.md carries
    demonstrations: bool = False  # visual demonstrations are supplied (demonstration_context != 'none')
    # Which demonstration was supplied, when known: 'terminal_state' or 'completion_sequence',
    # and how many images it has (0: unknown). Text for the other kind is left out.
    demonstration_kind: str | None = None
    demonstration_images: int = 0
    # Live-copy mode: the later copies carry handoff/ (it is in transfer_parts).
    carries_handoff: bool = True
    # Complete expert episodes mounted read-only at runtime/experts (Configuration.expert_demonstrations).
    expert_demonstrations: int = 0
    # What they hold (Configuration.expert_content): 'full', or 'images' (front-camera observations only).
    expert_content: str = 'full'

    def __post_init__(self):
        object.__setattr__(self, 'options', frozenset(self.options))
        if self.options - OPTIONS:
            raise ValueError('Unknown workspace option')
        from harness.backends import NAMES
        if self.cli not in NAMES:
            raise ValueError('Unknown agent CLI')
        if type(self.expert_demonstrations) is not int or self.expert_demonstrations < 0:
            raise ValueError('expert_demonstrations must be a nonnegative integer')
        if self.expert_content not in ('full', 'images'):
            raise ValueError('expert_content must be full or images')
        if 'agent-formal' in self.options and self.contract.phase == 'exploration':
            # Live-copy mode: exploration has no register, rehearse or submit.
            object.__setattr__(self, 'contract', replace(self.contract, bundle_tools=False))

    def axes(self):
        """Which skill layers a workspace takes: the robot it controls, how it perceives and whether it
        takes the research-method skills (None: no layer of that axis)."""
        return {'robot': self.contract.benchmark,
                'perception': 'depth' if self.contract.observation.geometry else 'rgb-only',
                'method': 'method' if benchmark_named(self.contract.benchmark).method_skills else None}

    def context(self):
        """Template variables: facts only, named for readability in the sources."""
        contract = self.contract
        bench = benchmark_named(contract.benchmark)
        formal_episode = 'formal-episode' in self.options
        return {
            'agent_cli': self.cli,
            'phase': contract.phase,
            'observation_profile': contract.observation_profile,
            'camera_geometry': contract.observation.geometry,
            'environments': contract.environments,
            'multi_env': contract.environments > 1,
            'subagent_cap': contract.subagent_cap,
            'shared_env_subagents': contract.environments == 1 and contract.subagent_cap > 0,
            'agent_formal': 'agent-formal' in self.options,
            'formal_episode': formal_episode,
            'handoff': 'handoff' in self.options,
            # The exploring agent's later live copies continue from its handoff/.
            'carries_handoff': 'agent-formal' in self.options and self.carries_handoff,
            'demonstrations': self.demonstrations,
            'demonstration_kind': self.demonstration_kind if self.demonstrations else None,
            # One image: text says "image", not "images".
            'one_demonstration_image': self.demonstrations and self.demonstration_images == 1,
            # Complete expert episodes under runtime/experts (their number; 0: none).
            'expert_demonstrations': self.expert_demonstrations,
            # The complete recordings (commanded references, measured motion, object states) are
            # mounted (their number; 0: none, or camera-only episodes).
            'expert_recordings': self.expert_demonstrations if self.expert_content == 'full' else 0,
            # The expert episodes hold front-camera observations only (video, previews, depth).
            'expert_camera_only': bool(self.expert_demonstrations) and self.expert_content == 'images',
            # The research-method skills (agent_skills/method/) are installed.
            'method_skills': bench.method_skills,
            # Episodes are scored with partial credit, which the instructions ask the agent to maximize.
            'partial_credit': bench.partial_credit,
            # Formal episodes earn graded scores; False: each succeeds or fails (share of successes).
            'graded_score': bench.graded_score,
            # Episodes have a wall-clock limit (status.episode_timeout_seconds); others end at their step limit.
            'wall_limit': bench.wall_limit,
            # Simulator benchmark (services/benchmarks.py), and a flag per benchmark for text only
            # its workspaces carry.
            'benchmark': contract.benchmark,
            **{name: name == contract.benchmark for name in BENCHMARKS},
            'robot': bench.robot_facts(self.task),
            # The robot has wrist cameras (wrist-camera-inspection applies).
            'wrist_cameras': wrist_cameras(contract.benchmark),
        }
