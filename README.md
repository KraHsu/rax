# RAX on RoboDojo and BEHAVIOR-1K (native branch)

**[Project website](https://robot-agent-exploration.com/)** · [Paper](https://robot-agent-exploration.com/rax.pdf)

This branch runs the simulation experiments of *RAX: Scaling Online Exploration for Robot
Agents*. A coding agent (Codex CLI or Claude Code) learns a manipulation task by
exploring a simulator through MCP tools. It keeps notes and verified skills in its
workspace, and then is evaluated on layouts it has never seen. Evaluation either runs the
robot program the agent wrote, or runs live copies of the agent.

The agent CLI runs natively on the host, as your user, with its own login. Its own
sandbox confines every command the agent runs: no network, writes only in the session's
workspace, and the repository, other sessions and your credentials hidden. Robot programs
the agent submits run in a bubblewrap sandbox. The agent and its programs have no GPU and
no access to the simulator's ground truth. Robot, Gemini and episode access go through
one host-owned MCP service. (The container branch runs the same experiments with the
agent in an offline Docker container instead.)

## Benchmarks

Two simulator backends run side by side behind the same harness, tools contract,
workspace generator and evaluation lifecycle. Both use Isaac Sim 5.1.

| Backend | Robot | Tasks | Exploration layouts | Formal layouts |
| --- | --- | --- | --- | --- |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | dual ARX X5 table-top arms, one overhead and two wrist cameras, 25 Hz | 54 table-top task configurations | saved-layout collection 0 | held-out collection 1 |
| [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) | Galaxea R1Pro mobile bimanual robot, head and wrist cameras, 30 Hz | the 100 household tasks of the 2026 BEHAVIOR challenge ([catalog](docs/BEHAVIOR_TASKS.md)) | the 300 train instances | the held-out public test instances |

Each session picks its backend when it is prepared, by choosing a task. Backends are
peers. Each one has an entry in `services/benchmarks.py` (its tools, robot facts, setup
defaults and capabilities) and an adapter `services/<benchmark>/benchmark.py`. The shared
harness calls only that interface, so adding a simulator does not touch the harness; see
[Adding a backend](docs/ARCHITECTURE.md#adding-a-simulator-backend).

Sessions run in parallel. Each session owns its simulators, its MCP service, its agent
process and its storage, so several sessions (on one or more GPUs) never share state.
Inside a session, exploration can use up to eight simulator environments at once, and
formal evaluation runs several episodes concurrently.

## How a session runs

1. **Explore.** The agent reads `TASK.md` (the task rubric and an optional demonstration
   image or clip), then solves the task by hand: it observes, moves the robot through MCP
   tools, and checks the result. Each exploration episode uses a new layout, and the
   operator sets the episode budget.
2. **Compile.** The agent turns what worked into `code/<project>/controller.py:main(ctx)`,
   a program made of step functions that check what they observe, and tests it unattended.
3. **Evaluate (formal).** One of two modes, chosen when the session is prepared:
   - **Code (default):** the agent registers its program, passes one clean rehearsal, and
     submits it. The frozen program then plays the held-out layouts alone.
   - **Agent formal:** after exploration, live copies of the agent each play one held-out
     layout, carrying their memory and code.

The operator dashboard prepares, launches, monitors, stops and resumes sessions. Every
dashboard action is also a terminal command.

## Quick start

```bash
git clone <this-repository> rax && cd rax
export ROBODOJO_RUNTIME_ROOT=/fast-disk/rax/runtime     # where environments, assets and sessions live
bash scripts/bootstrap_sources.sh                       # pinned RoboDojo + patches, BEHAVIOR-1K source
bash scripts/setup_robodojo_env.sh --accept-nvidia-eula # Isaac Sim 5.1, Isaac Lab, cuRobo (~23 GB)
bash scripts/setup_assets.sh                            # RoboDojo assets (~66 GB)
# Install bubblewrap (and socat for Claude Code), the Codex CLI or Claude Code, and a model
# credential (docs/SETUP.md, steps 0 and 4), e.g. an OpenRouter key in
# ~/.codex/secrets/openrouter_api_key.
bash scripts/robodojo_doctor.sh                         # checks everything above
bash scripts/run_operator_dashboard.sh                  # prepare and launch sessions
```

The full guides:

- [Agent harnesses](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md): Codex CLI, Claude Code, DeepSeek Harness and Kimi Code; providers, authentication and setup.
- [DeepSeek serving](https://github.com/robot-agent-exploration/rax/blob/main/docs/DEEPSEEK_SERVING.md): four-H200 setup, tuning and protocol checks.
- [Setup](docs/SETUP.md): host requirements, every installation step, credentials for the
  agent CLIs, the optional BEHAVIOR-1K backend, and the tests.
- [Usage](docs/USAGE.md): preparing, launching, monitoring, stopping and resuming
  sessions; agent formal, adaptation and parallel exploration; results and scores.
- [Architecture](docs/ARCHITECTURE.md): services, isolation, the MCP contract, the
  evaluation lifecycle, and how to add a backend.

## Repository layout

| Path | Contents |
| --- | --- |
| `auto_research_agent/` | The agent's instructions (`AGENTS.md`, `FORMAL_AGENTS.md`, `START_PROMPT.md`) and its Python API (`api/`) |
| `agent_skills/` | Every agent skill, written once and layered per backend, observation profile and environment count ([README](agent_skills/README.md)) |
| `harness/` | Launchers for Codex CLI and Claude Code, the agent's host process and sandbox paths, the robot-tool bridge, session control |
| `services/controller/` | The per-session MCP frontend, supervisor, bundle sandbox, gateway and formal evaluation |
| `services/robodojo/`, `services/robot_preview/` | The RoboDojo backend and its robot-only preview simulator |
| `services/behavior/` | The BEHAVIOR-1K backend |
| `services/exploration/` | Parallel exploration environments |
| `services/agent_formal/` | Live-agent formal evaluation |
| `services/workspace/` | The workspace generator |
| `services/dashboard/` | The operator dashboard ([README](services/dashboard/README.md)) |
| `patches/` | The two patches applied to the pinned RoboDojo checkout ([README](patches/README.md)) |
| `scripts/` | Setup, launch, stop, dashboard and doctor scripts |
| `local_tests/` | Unit and integration tests ([README](local_tests/README.md)) |

## Acknowledgements

This branch builds on [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo),
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K), NVIDIA Isaac Sim and Isaac Lab,
and [cuRobo](https://github.com/NVlabs/curobo). `dependencies.lock` pins every version.
The BEHAVIOR-1K datasets are for non-commercial research use under their own terms.

## Citation

BibTeX coming soon. Until then, please cite the paper as listed on the
[project website](https://robot-agent-exploration.com/).
