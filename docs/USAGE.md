# Usage

This guide runs sessions on a machine set up with [Setup](SETUP.md). A session is one
agent learning one task: it explores, writes a robot program, and is then evaluated on
held-out layouts. Each session owns its simulators, MCP service, agent process and
storage, so several sessions run in parallel on one or more GPUs without sharing state.

Every action has a terminal command; the [dashboard](../services/dashboard/README.md)
is a front end to the same commands.

| Step | Dashboard | Terminal |
| --- | --- | --- |
| Prepare | **New session** → **Prepare** | `runtime/envs/robodojo/bin/python scripts/configure_auto_research.py ...` |
| Launch | **Launch** (headless) | `bash scripts/start_auto_research_agent.sh --config <session>/operator-input.json` |
| Stop | **Stop** | `bash scripts/stop_auto_research_agent.sh --config <session>/operator-input.json` |
| Resume | **Resume** (headless) | `bash scripts/start_auto_research_agent.sh --config <session>/operator-input.json --resume` |
| Move to another GPU | - | `bash scripts/move_session_gpus.sh --config <session>/operator-input.json --sim-gpu <index or UUID>` |

Sessions live in `runtime/auto-research/<session-id>/`, where the ID is the creation
time and a random suffix, for example `20261013T120000Z_1a2b3c4d5e`.

## 1. Prepare a session

Preparing writes `<session>/operator-input.json` and starts nothing: no model call and
no simulator. It checks the task's layouts, maps the GPU to its UUID, and caches the
demonstration the session will show the agent.

```bash
runtime/envs/robodojo/bin/python scripts/configure_auto_research.py \
    --task stack_blocks --sim-gpu 0 --episodes 20
```

It prints the configuration path and the launch command. A configuration is used once:
to change a setting, prepare a new session.

### Choosing the backend

The task names a backend, and each backend has its own defaults:

- **RoboDojo** (default): the 54 task configurations of the dual ARX X5 table-top
  benchmark (42 tasks plus 12 `<task>_random` variants).
- **BEHAVIOR-1K:** the 100 household tasks of the 2026 challenge on the R1Pro, listed in
  [BEHAVIOR tasks](BEHAVIOR_TASKS.md). On the terminal, pass `--benchmark behavior` as
  well; the dashboard picks the backend from the task. BEHAVIOR needs its optional setup
  ([Setup, step 5](SETUP.md#5-behavior-1k-optional)).

Fields that do not apply to a backend are fixed. Passing another value is an error that
names the reason, for example `--task-rules does not apply to BEHAVIOR sessions: ...`.

### Settings

| Flag | Default | Meaning |
| --- | --- | --- |
| `--task` | required | Task name |
| `--benchmark` | `robodojo` | `robodojo` or `behavior` |
| `--sim-gpu` | required | Simulator GPU index or UUID. Agents have no GPU |
| `--episodes` | 5 | Exploration episodes (1..1000; the dashboard allows 2..1000). Each rehearsal uses one |
| `--exploration-envs` | 1 | Simulator environments the agent can use at once (1..8); see [section 6](#6-parallel-exploration-and-subagents) |
| `--subagents` | 0 | Concurrent subagents (0..8). 0 means one per environment when there are several, none with one |
| `--agent-cli` | `codex` | `codex`, `claude`, `dsh` or `kimi` |
| `--agent-provider` | `openrouter` | Provider for the selected harness: OpenRouter, vendor API, vendor account or `custom`; see [Agent harnesses](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md). Codex uses its provider profile |
| `--agent-effort` | `xhigh` | Reasoning effort, `xhigh` or `max`, for the agent and its formal copies |
| `--observation-profile` | `rgbd` | `rgbd`, or `rgb-only` (no depth or camera geometry). BEHAVIOR: always `rgbd` |
| `--demonstration-context` | `terminal_state` | What the agent sees of a demonstration: `terminal_state` (the final frame), `completion_sequence` (ordered frames of the whole demonstration) or `none` |
| `--task-rules` | `public` | RoboDojo: `public` (the public rubric) or `code-summary` (adds the success stages, thresholds and partial credit summarized from the official task code). BEHAVIOR: always the official BDDL task definition |
| `--agent-formal` | `off` | Evaluation mode: `off` (code mode) or `on` (live copies); see [section 4](#4-evaluation) |
| `--formal-episodes` | RoboDojo 50, BEHAVIOR 10 | Held-out formal episodes (1..100) |
| `--formal-workers` | RoboDojo 4, BEHAVIOR 2 | Formal episodes run at once (1..16) |
| `--formal-eval-seed` | 1 | Held-out layout collection. BEHAVIOR: always 1, the 20 public test instances |
| `--formal-seed` | 0 | First formal layout in that collection |
| `--exploration-collection` | 0 | Layout collection for exploration (RoboDojo: 0, 1 or 2; BEHAVIOR: 0, the 300 train instances) |
| `--exploration-start-seed` | RoboDojo 1, BEHAVIOR 0 | First exploration layout. Layouts cycle when episodes outnumber them; formal layouts in the same collection are skipped |
| `--formal-context` | `compact` | Agent formal: the conversation each copy carries, `resume` (whole), `compact` (summarized) or `clear` (none) |
| `--transfer-parts` | `memory,code,handoff` | Agent formal: workspace folders each copy carries |
| `--transfer-mb` | 100 | Cap on the carried folders (1..1024 MB) |
| `--write-handoff` | `off` | RoboDojo: install the `transfer-handoff` skill, so this session can be an adaptation source |
| `--adaptation-source` | none | RoboDojo: a finished session whose agent continues here; see [section 5](#5-adaptation) |
| `--adaptation-context` | `compact` | Adaptation: the conversation carried from the source |
| `--episode-reviews` | `off` | BEHAVIOR: `on` or `detailed` ground-truth reviews of each exploration episode; see [BEHAVIOR backend](BEHAVIOR_BACKEND.md#episode-reviews-exploration-only) |
| `--recording-gib` | 10 | Size limit of a robot program's output folder and of a submitted bundle (recorded episode data is not capped) |

Not all RoboDojo tasks have 50 layouts per collection: 14 task configurations have 30
and `make_toast_random` has 45. For them, lower `--formal-episodes`, or preparing fails
with `Formal collection 1 has 30 layouts for <task>; requested range is unavailable`.

The dashboard form holds the same settings except `--exploration-collection` and
`--recording-gib`, which keep their defaults there.

## 2. Launch

```bash
bash scripts/start_auto_research_agent.sh --config runtime/auto-research/<session-id>/operator-input.json
```

The launcher builds the session's workspace and CLI home, starts its MCP service, and
runs the agent CLI natively in its sandbox, offline, with the kickoff prompt from the
workspace's `START_PROMPT.md`. Model calls start here and are paid.

- **Interactive (default):** the agent's terminal UI opens in your terminal. The session
  runs until you quit the UI; then the launcher freezes the agent and runs the formal
  evaluation. Closing the terminal stops the session like **Stop** does (a launcher
  started under `nohup` keeps running), so use `tmux` or `screen` for long sessions.
- **Headless:** `--non-interactive` runs the default kickoff without a UI, and
  `--exec-prompt "<text>"` runs your own prompt instead. The dashboard's **Launch** and
  **Resume** run headless, logging to `<session>/agent.log`.

Other launcher flags:

| Flag | Meaning |
| --- | --- |
| `--profile <name>` | Codex provider profile (default: the name in `<CODEX_HOME>/experiment-profile`, else `openrouter`) |
| `--provider-home <dir>` | Where provider profiles live (default: `$CODEX_HOME`, else `~/.codex`) |
| `--key-file <file>` | OpenRouter key file (default: `~/.codex/secrets/openrouter_api_key` when it exists). Without one, `OPENROUTER_API_KEY` serves Gemini and Claude Code's `openrouter` provider; a Codex OpenRouter session needs the file |
| `--agent-binary <file>` | DSH/Kimi: installed CLI executable |
| `--agent-model <id>` | DSH/Kimi: override the provider's default model |
| `--agent-key-file <file>` | DSH/Kimi: private API key file; keep it outside the checkout |
| `--agent-base-url <url>` | DSH/Kimi `custom`: compatible model endpoint |
| `--agent-auth-home <dir>` | DSH/Kimi account: dedicated vendor login directory outside the checkout |
| `--claude-key-file <file>` | Claude `anthropic` provider: an API key file (else `ANTHROPIC_API_KEY`). `claude-login`: a token file other than the experiment token |
| `--claude-model <id>` | Claude: override the provider's default model |
| `--prepare-only` | Deploy the workspace and the CLI's session configuration (its sandbox included) and print their paths, then stop without a model call or episode. This uses up the configuration; prepare a new one to run |

Dashboard launches use the defaults of these flags, read in the dashboard's own
environment.

A session runs one launcher at a time; a second launch exits with
`Another launcher is running this session`.

### Remote machines

The dashboard binds only to loopback. Forward its port and open the printed URL locally:

```bash
ssh -L 8765:127.0.0.1:8765 <host>
```

## 3. Monitor, stop and resume

### What a session runs

1. **Explore.** The agent reads `TASK.md` and any demonstration images, then works the
   robot through MCP tools. `start_episode` starts the next layout; `evaluate` checks the
   task; `finish` closes the episode. It keeps notes in `memory/` and code in `code/`.
2. **Compile.** It writes `code/<project>/controller.py:main(ctx)`, a program that
   checks what it observes at each step, and tests it unattended.
3. **Evaluate.** Code mode or live copies ([section 4](#4-evaluation)).

### Dashboard

The dashboard lists each session's stage, exploration used, formal success and score,
agent tokens and Gemini spend, live camera frames, every simulator episode's video,
results, and the agent and MCP logs. See its [README](../services/dashboard/README.md).

### Session files

| Path | Contents |
| --- | --- |
| `operator-input.json` | The prepared configuration |
| `configuration.json`, `state.json` | The deployed configuration and the supervisor's state (episodes, bundles, submission, formal batches, Gemini ledger). Both must agree with `operator-input.json` to resume |
| `agent-lifecycle.json` | Launcher status (`preparing`, `running`, `usage_wait`, `formal`, `exited`, `stopped`, `failed`), every earlier attempt, relaunches and agent errors |
| `agent.log`, `frontend.log` | Headless launcher output; the MCP service log, appended per launch |
| `mcp.jsonl` | Every MCP call and its result |
| `agent-login.json` | The session's login: CLI, provider, model and the key file's path (never a secret) |
| `provider-errors.jsonl` | Failed Gemini calls with the provider's raw response |
| `agent-workspace/`, `codex-home/` or `claude-home/` | The agent's workspace and CLI home (settings, conversation and transcripts, from which the dashboard reads token usage) |
| `native/results/<run>/sim/` | Each simulator episode's own records and `sensors.mp4` |
| `exploration/` | The episode ledger of parallel exploration |
| `formal.json` | What ran after exploration: `exploration_incomplete`, `frozen`, `completed` or `failed` with the error |
| `formal-scores.json`, `agent-formal-scores.json` | Formal scores per episode and averaged ([section 7](#7-results-and-scores)) |
| `formal-workers/`, `agent-formal/` | One directory per formal episode: logs, and for live copies their transcripts and notes |
| `published/` | Run artifacts the agent can read (frames, traces, reports), linked into its workspace as `runtime/autonomous_controller` |
| `subagents.json` | Subagents spawned, peak concurrency, calls per agent |
| `operator-analysis/` | BEHAVIOR episode reviews (operator copies) and their cost |
| `gpu-moves.jsonl` | GPU moves of this session |

### Stop

**Stop**, or `scripts/stop_auto_research_agent.sh`, sends the session's launcher SIGTERM
(through a pidfd, never to a recycled PID). The launcher stops the agent, everything it
started and the simulators, and records `stopped`. Budgets are not restored. With one
exploration environment, an episode that was running stays charged; with several,
it is refunded (see [section 6](#6-parallel-exploration-and-subagents)).

### Resume

A stopped, exited, failed or crashed session resumes in its own conversation (Codex
`exec resume`, Claude Code `--resume`) and in the same workspace.

- The agent gets a short prompt to reread `memory/PROGRESS.md` and call `status`;
  `--exec-prompt` replaces it.
- Episodes left live by the previous launcher are recorded as interrupted, leftover
  simulators are stopped, and exploration reopens.
- Token usage over all launches stays in the CLI's transcripts in the session, and the
  Gemini ledger continues.
- Resuming a session whose exploration has closed restarts no agent: it runs only the
  formal steps that have not run. An interrupted agent formal continues with the layouts
  that have no outcome; an interrupted code-mode formal batch is not rerun.
- Resume is refused while the launcher is alive, for a session that was only prepared,
  and for a first launch that failed before its workspace existed (prepare a new
  session then).
- A session that ran its agent in a container (on the container branch) is refused:
  resume it with the branch it started on.

### Move to another GPU

Stop the session first, then:

```bash
bash scripts/move_session_gpus.sh --config <session>/operator-input.json --sim-gpu 1
```

It refuses while any process still uses the session, rewrites the simulator GPU in all
three saved copies, moves agent-formal copies listed on the old GPU, and logs the move to
`gpu-moves.jsonl`. Resume afterwards. Any other change to a launched configuration
prevents resuming.

### Automatic relaunch

In a headless session, the launcher continues the agent's conversation while
exploration is open:

- **Turn ended:** the agent is prompted to continue, at most 3 times per launch.
- **Run failed** (provider overload, broken stream): it continues after 300 s, up to
  5 failures in a row. A run of 30 minutes or more resets the count.
- **Claude `claude-login` usage limit:** the session waits for the window's reset with
  its episodes up (lifecycle `usage_wait`, `usage_limited_until`), then continues, up to
  10 waits per launch.

An interactive session is never relaunched.

## 4. Evaluation

Each session has one evaluation mode, set when it is prepared.

### Code mode (`--agent-formal off`, default)

1. **Register** freezes `code/<project>/` as a bundle without running it.
2. **Rehearse** runs the whole bundle in a fresh exploration episode under formal
   limits, unattended. It uses one exploration episode. A clean rehearsal ran to its end
   without an error, timeout or simulator exit; the task need not succeed.
3. **Submit** takes one cleanly rehearsed bundle and closes exploration. It returns no
   score.

When the agent has ended, the launcher freezes the session (memory, code, handoff and
the conversation) and runs the formal batch: `--formal-episodes` held-out layouts, each
in a fresh bubblewrap sandbox and simulator, `--formal-workers` at a time, with no agent,
edits or retries. Programs get 8 GiB of memory, math-library threads for 4 CPUs, a 1 GiB
`/tmp`, and no network or GPU.

Episode limits: RoboDojo episodes end at the task's native step limit or after 6 s per
native step of wall time, startup included. BEHAVIOR episodes end at their step limit
and have no wall-clock limit.

### Live copies (`--agent-formal on`)

Exploration has no `register`, `rehearse` or `submit`; the agent is told to practice:
solve the task, build a controller, and use every episode to test it. Exploration closes
when no episode remains and none is live.

After exploration, one copy of the agent plays each held-out layout:

- Each copy gets a fresh workspace with `FORMAL_AGENTS.md` as its `AGENTS.md`,
  `TASK.md`, the robot skills, the carried folders and its conversation
  (`--formal-context`: `compact` runs one forced compaction first).
- Carried folders are packed in the order `handoff`, `memory`, `code`; a folder that
  does not fit in the remaining `--transfer-mb` is left out whole, and later ones may
  still fit. `state.json` records what was carried.
- Copies have robot tools, Gemini and Python on their one environment, but no lifecycle
  tools. They start 60 s apart, `--formal-workers` at a time. A copy whose run fails
  continues after 300 s, up to 5 failures in a row.
- The episode limit is the task's formal limit, with 120 s to wrap up after the episode
  ends.

To spread copies over several GPUs, add `"formal_sim_gpus": ["GPU-...", "GPU-..."]`
(UUIDs) to `operator-input.json` before the first launch. Copy *i* uses entry
*i* modulo the list length.

### More evaluations of a finished session

- **Live copies after code mode:** play a finished code-mode session's held-out layouts
  with copies of its frozen agent, beside its bundle formal:

  ```bash
  bash scripts/start_auto_research_agent.sh --config <session>/operator-input.json --resume --agent-formal
  ```

  It needs closed exploration. On BEHAVIOR, each copy also gets reports of how the
  bundle ran on the other layouts (replayed frames and a Gemini review), built first on
  the session's simulator GPU and charged to the operator review budget ($10 per
  session). Build them ahead, in parallel, with
  `PYTHONPATH=. runtime/envs/robodojo/bin/python -m services.agent_formal.code_runs <session>... --gpus 0,1`.
- **Re-evaluate a rehearsed bundle** on more layouts, without a model:

  ```bash
  PYTHONPATH=. runtime/envs/robodojo/bin/python -m services.controller.parallel_batch \
      --source-session <session> --bundle <bundle-id> \
      --output runtime/auto-research/<new-directory> --episodes 50 --workers 4
  ```

  The bundle must have a clean rehearsal; the source session is not changed.

## 5. Adaptation

RoboDojo only. An adaptation session continues a finished session's agent in a new
environment with a small episode budget, to measure what transfers.

- The new environment is this session's task and exploration collection: the source's
  task on another collection (for example `--exploration-collection 2`) or its
  `<task>_random` variant.
- The agent carries the source's frozen folders (`--transfer-parts`, `--transfer-mb`)
  and conversation (`--adaptation-context`); `ADAPTATION.md` tells it what changed.
  Nothing from the source's formal evaluation is carried.
- Carrying a conversation needs the source's agent CLI; `clear` can switch CLIs.
- `--write-handoff on` on the source asks its agent to keep a structured `handoff/`.

```bash
# Source: handoff on.
runtime/envs/robodojo/bin/python scripts/configure_auto_research.py --task stack_blocks \
    --sim-gpu 0 --episodes 20 --write-handoff on
# Adaptation: 3 episodes on the randomized variant, carrying only the handoff.
runtime/envs/robodojo/bin/python scripts/configure_auto_research.py --task stack_blocks_random \
    --sim-gpu 0 --episodes 3 --formal-episodes 30 \
    --adaptation-source runtime/auto-research/<source-session> --transfer-parts handoff
```

`--formal-only` evaluates an imported agent without exploring: prepare a session with
`--agent-formal on` and `--adaptation-source`, then launch it with `--formal-only`. Live
copies of the source's frozen agent play the held-out layouts.

## 6. Parallel exploration and subagents

With `--exploration-envs N` (2..8) the agent has N simulator environments at once, each
its own Isaac Sim process with a preview simulator, started on first use.

- Robot, preview and episode tools take `env_id`; calls to different environments run
  concurrently, and calls to one environment run in order.
- All environments share one episode budget: every start, reset and rehearsal uses one.
- An episode that never ran to its own end (the session stopped, or its environment's
  process failed) is not charged, and its layout is played again next. The agent is told.
  With one environment, every start is charged.
- Rehearsals and formal episodes always use one environment.

`--subagents N` caps concurrent subagents. With one environment and N subagents, the
subagents take turns on the environment. Codex enforces the cap; Claude Code is told it
and is measured. `subagents.json` records spawns, peak concurrency and calls per agent.

Memory grows with live environments; see [GPU memory](SETUP.md#gpu-memory).

## 7. Results and scores

Scores are operator-only; the agent never sees formal results.

- **Per episode:** 1.0 on native success, otherwise the simulator's partial score: for
  RoboDojo the native process score divided by 100, for BEHAVIOR the challenge q-score
  (the fraction of goal conditions newly satisfied). The partial score is read from the
  simulator's records whatever the episode's outcome, so an episode that ended with an
  error keeps the partial score its simulator recorded.
- **Per batch:** the mean over the fixed number of scheduled episodes. An episode
  without a valid score counts as zero (`missing_score_count`). `average_score` is set
  only when the batch has completed.
- **Success rate:** successes over scheduled episodes. Success needs native success and
  a clean run; other episodes are `unsuccessful` or `error`.

Code-mode results: `published/formal_batch.json` and `formal-scores.json`. Live copies:
`published/agent_formal_batch.json` and `agent-formal-scores.json`.

## 8. Gemini

The agent can call `gemini_generate` (`google/gemini-3.8-flash` on OpenRouter), for
example to check an image. Spending has no session limit; one ledger per session covers
exploration, rehearsals, formal episodes and live copies, and the dashboard shows the
spend. Without an OpenRouter key every call answers `Gemini is not configured`. BEHAVIOR
episode reviews and code-run reports use a separate $10 operator budget per session.

## 9. Troubleshooting

| Message | Fix |
| --- | --- |
| `Controller artifacts must be under <root> (set ROBODOJO_ARTIFACT_ROOT to change it)` | Sessions must live under the artifact root, by default the directory `runtime/` links to. Set the variable to that directory or an ancestor |
| `Formal collection 1 has 30 layouts for <task>; requested range is unavailable` | Lower `--formal-episodes` or change `--formal-seed` |
| `<flag> does not apply to BEHAVIOR sessions: <reason>` | Leave that flag out for BEHAVIOR |
| `Codex binary is required` | Install Codex on `PATH` ([Setup, step 4](SETUP.md#4-agent-cli-and-model-credentials)), or prepare with `--agent-cli claude` |
| `Claude Code binary is required (install claude on the host)` | Install Claude Code on `PATH` |
| `bubblewrap (bwrap) is required to run bundles` | Install bubblewrap ([Setup, step 0](SETUP.md#0-host-requirements)) |
| `Provider profile '<name>' needs model_provider ...` | Create `<CODEX_HOME>/<name>.config.toml`, or use the built-in `openrouter` profile |
| `The OpenRouter profile needs a key file ...` | Put the key in `~/.codex/secrets/openrouter_api_key` or pass `--key-file` |
| `Provider profile '<name>' needs its own key file ([robodojo] key_file) ...` | Add `key_file` to the profile's `[robodojo]` table; the OpenRouter key goes only to OpenRouter |
| `The Claude experiment token was rejected (HTTP 401) ...` | Rerun `bash scripts/setup_claude_experiment_login.sh` |
| `This session was already launched; use --resume or a fresh configuration` | Resume it, or prepare a new session |
| `Another launcher is running this session ...` | Stop it and wait for its cleanup |
| `--agent-formal needs --resume of a session whose exploration has closed` | The session's exploration is still open; resume it normally |
| `Session predates this dashboard server` | Tick **Earlier runs** to review it; launch and resume it from the terminal |
| `formal.json`: `exploration_incomplete` | The agent stopped with episodes left; resume the session |

## DeepSeek Harness and Kimi Code

Select `--agent-cli dsh` or `--agent-cli kimi` when preparing a session.
Both adapters support one exploration environment, no subagents and code formal
evaluation. Set `--agent-effort max` for maximum reasoning.

The [agent harness guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md)
covers installation, OpenRouter, vendor API keys, account login and compatible
endpoints. Pass credentials to the launcher with `--agent-key-file` for API
access or `--agent-auth-home` for account access.

### Claude Code and external expert demonstrations

When adding expert demonstrations through a workspace symlink, add the source
directory's real path to the read roots returned by `readable()` in
`harness/claude_cli/auto_research.py`. Resolve the source with `Path.resolve()`.
The launcher uses those roots for both `permissions.allow` (`Read`) and
`sandbox.filesystem.allowRead`; keep the workspace symlink's `Read` rule as well.
Demonstration directories remain read-only.
