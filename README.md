# RAX on HumanoidArena (container branch)

**[Project website](https://robot-agent-exploration.com/)** · [Paper](https://robot-agent-exploration.com/rax.pdf) · [All branches](https://github.com/robot-agent-exploration/rax/tree/main)

This branch runs the HumanoidArena experiments of *RAX: Scaling Online Exploration for Robot
Agents*. A coding agent (Codex CLI or Claude Code) learns a whole-body humanoid task by
exploring the simulator through MCP tools. It keeps notes and verified skills in its
workspace, and then is evaluated on the benchmark's official test episodes. Evaluation either
runs the robot program the agent wrote, or runs live copies of the agent.

The agent and its programs run in an offline Docker container. They have no GPU, no
network, and no access to the simulator's ground truth. Robot, Gemini and episode access
go through one host-owned MCP service.

## Benchmark

| Backend | Robot | Tasks | Exploration layouts | Formal layouts |
| --- | --- | --- | --- | --- |
| [HumanoidArena](https://github.com/William-wAng618/HumanoidArena) | Unitree G1 (29 DoF) with Dex3 hands under the frozen [SONIC](https://huggingface.co/nvidia/GEAR-SONIC) whole-body controller, front RGB-D camera, 50 Hz | the 7 tasks with official test episodes: `opendoor`, `boxing_bag`, `pickplace_box`, `doubledesk`, `sit_sofa`, `football_single`, `warehouse_navigation` | fresh layouts of each task's distribution, never an official test episode | the 60 official test episodes of each task |

The simulator is Isaac Sim 6.1 with Isaac Lab, run from a Docker image this branch builds.
The backend sits behind the harness's benchmark interface: an entry in `services/benchmarks.py`
(its tools, robot facts, setup defaults and capabilities) and the adapter
`services/humanoid/benchmark.py`. The harness calls only that interface; see
[Adding a backend](docs/ARCHITECTURE.md#adding-a-simulator-backend).

Sessions run in parallel. Each session owns its simulators, its MCP service, its agent
container and its storage, so several sessions (on one or more GPUs) never share state.
Inside a session, exploration can use up to eight simulator environments at once, and
formal evaluation runs several episodes concurrently.

## How a session runs

1. **Explore.** The agent reads `TASK.md` (the task, its randomization and its success
   condition), then solves the task by hand: it observes, moves the robot through MCP
   tools, and checks the result. Each exploration episode uses a new layout, and the
   operator sets the episode budget.
2. **Compile.** The agent turns what worked into `code/<project>/controller.py:main(ctx)`,
   a program made of step functions that check what they observe, and tests it unattended.
3. **Evaluate (formal).** One of two modes, chosen when the session is prepared:
   - **Code (default):** the agent registers its program, passes one clean rehearsal, and
     submits it. The frozen program then plays the official test episodes alone.
   - **Agent formal:** after exploration, live copies of the agent each play one test
     episode, carrying their memory and code.

The operator dashboard prepares, launches, monitors, stops and resumes sessions. Every
dashboard action is also a terminal command.

## Quick start

```bash
git clone --branch humanoid-arena/container https://github.com/robot-agent-exploration/rax.git
cd rax
export RAX_RUNTIME_ROOT=/fast-disk/rax/runtime            # where environments, assets and sessions live
bash scripts/bootstrap_sources.sh                         # pinned HumanoidArena source
bash scripts/setup_host_env.sh                            # the harness's own Python environment
bash scripts/build_humanoid_image.sh --accept-nvidia-eula # simulator image: Isaac Sim 6.1, Isaac Lab, cuRobo
bash scripts/setup_humanoid_assets.sh                     # scene and robot assets, SONIC encoder/decoder
runtime/envs/host/bin/python scripts/build_agent_image.py # offline agent image (~12 GB)
# Install the native Codex CLI or Claude Code binary and a model credential (docs/SETUP.md, step 6),
# e.g. an OpenRouter key in OPENROUTER_API_KEY or ~/.codex/secrets/openrouter_api_key.
bash scripts/doctor.sh                                    # checks the host
runtime/envs/host/bin/python scripts/humanoid_doctor.py --gpu 0 --task opendoor   # checks the simulator
bash scripts/run_operator_dashboard.sh                    # prepare and launch sessions
```

The full guides:

- [Agent harnesses](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md): Codex CLI, Claude Code, DeepSeek Harness and Kimi Code; providers, authentication and setup.
- [DeepSeek serving](https://github.com/robot-agent-exploration/rax/blob/main/docs/DEEPSEEK_SERVING.md): four-H200 setup, tuning and protocol checks.
- [Setup](docs/SETUP.md): host requirements, every installation step, credentials for the
  agent CLIs, the optional expert episodes, and the tests.
- [Usage](docs/USAGE.md): preparing, launching, monitoring, stopping and resuming
  sessions; agent formal and parallel exploration; results and scores.
- [HumanoidArena backend](docs/HUMANOID_BACKEND.md): tasks, layouts, tools, observations,
  the simulator image and expert episodes.
- [Architecture](docs/ARCHITECTURE.md): services, isolation, the MCP contract, the
  evaluation lifecycle, and how to add a backend.

## Repository layout

| Path | Contents |
| --- | --- |
| `auto_research_agent/` | The agent's instructions (`AGENTS.md`, `FORMAL_AGENTS.md`, `START_PROMPT.md`) and its in-container Python API (`api/`) |
| `agent_skills/`, `agent_blocks/` | Every agent skill and shared text block, written once and layered per robot, observation profile and environment count ([README](agent_skills/README.md)) |
| `harness/` | Launchers for Codex CLI and Claude Code, model relays, session control |
| `services/controller/` | The per-session MCP frontend, supervisor, sandbox, gateway and formal evaluation |
| `services/humanoid/` | The HumanoidArena backend: host adapter, simulator server, task rules and the G1 model |
| `services/simulator/` | The RPC between the host and a simulator process |
| `services/exploration/` | Parallel exploration environments |
| `services/agent_formal/` | Live-agent formal evaluation |
| `services/workspace/` | The workspace generator |
| `services/dashboard/` | The operator dashboard ([README](services/dashboard/README.md)) |
| `docker/humanoid/`, `patches/` | The simulator image and the Isaac Lab patch for Isaac Sim 6.1 |
| `scripts/` | Setup, launch, stop, dashboard and doctor scripts |
| `local_tests/` | Unit and integration tests ([README](local_tests/README.md)) |

## Acknowledgements

This branch builds on [HumanoidArena](https://github.com/William-wAng618/HumanoidArena),
[SONIC](https://huggingface.co/nvidia/GEAR-SONIC), NVIDIA Isaac Sim and Isaac Lab, and
[cuRobo](https://github.com/NVlabs/curobo). `dependencies.lock` pins every version. The
HumanoidArena assets and expert recordings, and the SONIC models, are used under their own
licenses; the Unitree G1 description in `services/humanoid/g1/robot/` keeps its BSD 3-Clause
license (`LICENSE.unitree`).

## License

RAX's original source code and accompanying documentation are licensed under the
[Apache License 2.0](LICENSE). Copyright 2026 RAX contributors.

Third-party code, simulator software, assets, model weights and datasets retain their
own licenses. See [NOTICE](NOTICE) and the relevant upstream releases.
The bundled Unitree G1 robot description retains its
[BSD 3-Clause license](services/humanoid/g1/robot/LICENSE.unitree).

## Citation

If you find RAX useful, please cite:

```bibtex
@misc{huang2026scaling,
  title        = {Scaling Online Exploration for Robot Agents},
  author       = {Huang, Runhan and Zhang, Xiangcheng and Chen, Haoyu
                  and Li, Yitong and Wang, Yen-Jen and Lyu, Bohan
                  and Ye, Haotian and Hong, Chuye and Wang, Yixuan
                  and Su, Chloe and Xiong, Haoyu and Wu, Yangzhen
                  and Xie, Enze and Manderson, Travis and Ermon, Stefano
                  and Abbeel, Pieter and Torralba, Antonio and Song, Dawn
                  and Ye, Weirui and Du, Yilun},
  year         = {2026},
  howpublished = {Preprint},
  url          = {https://robot-agent-exploration.com/}
}
```
