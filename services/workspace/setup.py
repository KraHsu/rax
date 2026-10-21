"""The facts a workspace is generated from; roles live in recipes, not here."""
from dataclasses import dataclass, field, replace

from services.benchmarks import BENCHMARKS, benchmark as benchmark_named
from services.mcp_contract import Contract

# Experiment options: 'agent-formal' for an exploring agent in live-copy mode (copies
# of it play the held-out layouts live; exploration has no bundle tools), and
# 'formal-episode' for such a copy during its one held-out episode.
# 'handoff' installs the transfer-handoff skill for adaptation source trials.
OPTIONS = frozenset({'agent-formal', 'formal-episode', 'handoff'})


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

    def __post_init__(self):
        object.__setattr__(self, 'options', frozenset(self.options))
        if self.options - OPTIONS:
            raise ValueError('Unknown workspace option')
        from harness.backends import NAMES
        if self.cli not in NAMES:
            raise ValueError('Unknown agent CLI')
        if 'agent-formal' in self.options and self.contract.phase == 'exploration':
            # Live-copy mode: exploration has no register, rehearse or submit.
            object.__setattr__(self, 'contract', replace(self.contract, bundle_tools=False))

    def axes(self):
        """Which skill layers a workspace takes: the robot it controls and how it perceives."""
        return {'robot': self.contract.benchmark,
                'perception': 'depth' if self.contract.observation.geometry else 'rgb-only'}

    def context(self):
        """Template variables: facts only, named for readability in the sources."""
        contract = self.contract
        bench = benchmark_named(contract.benchmark)
        formal_episode = 'formal-episode' in self.options
        return {
            'agent_cli': self.cli,
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
            # Episodes are scored with partial credit, which the instructions ask the agent to maximize.
            'partial_credit': bench.partial_credit,
            # A flag per simulator benchmark ('robodojo': ARX X5 table-top, 'behavior': R1Pro
            # household) for text only its workspaces carry.
            # Episodes have a wall-clock limit (status.episode_timeout_seconds); BEHAVIOR's end at their step limit.
            'wall_limit': bench.wall_limit,
            **{name: name == contract.benchmark for name in BENCHMARKS},
            'robot': bench.robot,
        }
