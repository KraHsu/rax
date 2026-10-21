# Setup

This guide installs everything a session needs on one Linux machine. Each step is one
script, and two doctors check the result at the end. The repository holds only source
code. Everything the scripts download or build goes under `runtime/`, a link to a
directory you choose. Credentials stay in your home directory.

| Step | What | Needed for |
| --- | --- | --- |
| [0](#0-host-requirements) | Linux host, NVIDIA GPU and driver, bubblewrap and socat | everything |
| [1](#1-sources) | Pinned official DexJoCo checkout | everything |
| [2](#2-host-environment) | The harness's and the agent's Python environment `runtime/envs/host` | everything |
| [3](#3-dexjoco-environment) | The official DexJoCo simulator environment `runtime/envs/dexjoco` | every session |
| [4](#4-agent-cli-and-model-credentials) | Codex CLI or Claude Code, and a model provider | every session |
| [5](#5-demonstrations) | Three verified demonstrations per task | sessions with a demonstration (the default) or expert episodes |
| [6](#6-check-the-installation) | Doctors and tests | recommended |

## 0. Host requirements

- **Operating system:** Linux x86_64. Tested on Rocky Linux 8.
- **GPU:** one NVIDIA GPU whose driver provides EGL (`libEGL_nvidia`). MuJoCo renders the
  cameras headless through EGL on the GPU the session chooses; physics runs on the CPU.
  Only the simulators use a GPU; agents and their programs run on the CPU. On A100 GPUs the
  official scenes' 8x MSAA offscreen framebuffer fails (`Offscreen framebuffer is not
  complete, error 0x8cdd`); use an RTX or H100 GPU. See [GPUs](#gpus).
- **Sandboxes:** `bubblewrap` (`bwrap`), able to create unprivileged user namespaces as
  your (non-root) user. Submitted robot programs run in it, and Claude Code's sandbox
  uses it together with `socat`. Codex brings its own sandbox. No Docker is needed: the
  simulator runs natively, and so does the agent.
  On Ubuntu: `sudo apt install bubblewrap socat`.
- **Host tools:** `git`, `git-lfs`, `curl`, `sha256sum` and `nvidia-smi`. Everything else
  (conda, Python) is installed by the scripts.
- **Disk:** use a fast local SSD for `runtime/`.

  | Item | Size |
  | --- | --- |
  | DexJoCo source | 0.3 GB |
  | Host environment (with the agent's CPU PyTorch) and Miniforge | 2 GB |
  | DexJoCo environment | 1.7 GB |
  | Demonstrations, all tasks (downloaded candidates and the installed experts) | about 0.1 GB per task |
  | One session | from a few GB to tens of GB (see below) |

  A session keeps every episode's recording. The agent's workspace has no size cap of its
  own; it lives in the session directory on the same disk.

## 1. Sources

```bash
git clone --branch dexjoco/native https://github.com/robot-agent-exploration/rax.git
cd rax
export RAX_RUNTIME_ROOT=/fast-disk/rax/runtime    # optional; default: .local-runtime/ in the repository
bash scripts/bootstrap_sources.sh
```

The bootstrap script:

- links `runtime/` to `$RAX_RUNTIME_ROOT`, or to `.local-runtime/` when the variable is
  unset;
- clones the official DexJoCo repository at the commit pinned in `dependencies.lock` next
  to the runtime directory, linked as `DexJoCo/`. The harness never modifies it.

Sessions must live under the *artifact root*. By default that is the directory
`runtime/` links to, so nothing needs setting. If you set `RAX_ARTIFACT_ROOT`, it
must be that directory or one of its ancestors.

Run every later step from the repository root.

## 2. Host environment

```bash
bash scripts/setup_host_env.sh
```

The launchers, the MCP service, the workspace generator, the dashboard and the tests run in
`runtime/envs/host`: Python 3.12 with NumPy, Pillow, Jinja2, msgpack, PyYAML, imageio,
SciPy, OpenCV, h5py, pytest and PyTorch (CPU) at the versions in `dependencies.lock`. The script installs the pinned
Miniforge into `runtime/miniforge3` (after checking its SHA-256) and creates the prefix. It
holds no simulator: the host never imports MuJoCo or the DexJoCo package.
`scripts/host_env.sh` sets the environment variables if you want them in a shell.

This environment is also the agent's Python: the agent's commands and its submitted
programs run `runtime/envs/host/bin/python`, with NumPy, SciPy, Pillow, OpenCV, h5py and a
CPU-only PyTorch, offline and without a GPU.

## 3. DexJoCo environment

```bash
bash scripts/setup_dexjoco_env.sh
```

The simulator processes run in `runtime/envs/dexjoco`, built as the official README does:
the conda dependencies of the official `environment-dexjoco.yaml` (Python 3.11, MuJoCo 3.4.0,
Gymnasium 1.0.0, NumPy 1.26.4), taken from conda-forge only (some sites block the `defaults`
channel), and its pip section verbatim (the official `dexjoco` and `openpi-client` packages,
installed editable from the pinned checkout). The script refuses a checkout that is not at the
pinned commit or has local changes, and ends by printing the installed versions.
It uses the Miniforge of step 2 (`CONDA_EXE` selects another conda binary). `DEXJOCO_PYTHON`
overrides the simulator interpreter; `scripts/dexjoco_env.sh` sets these variables if you
want them in a shell.

## 4. Agent CLI and model credentials

For DeepSeek Harness or Kimi Code, follow the [agent harness guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md) after installing the host dependencies.

Install at least one agent CLI on the host, on your `PATH`. It runs natively as your user
and reaches its model provider directly; the harness adds no relay. Each session has its
own CLI home (`codex-home/` or `claude-home/` in the session directory), rewritten at every
launch, so your own CLI settings and history are not used. The launcher never installs
packages.

- **Codex CLI** (default): the standalone release from
  [github.com/openai/codex/releases](https://github.com/openai/codex/releases), or its
  installer. Tested with 0.162.1.
- **Claude Code:** the native installer from
  [docs.claude.com](https://docs.claude.com/en/docs/claude-code/setup). Tested with 2.1.296.
  Its sandbox needs `bwrap` and `socat` (step 0).

The agent's commands cannot read your credentials ([what the sandbox hides](ARCHITECTURE.md#the-agent-cli-and-its-sandbox)),
so keep key files in the locations below. As with the CLIs used directly, the key a CLI
runs with is in its environment, so the agent's commands see it too; they have no
internet.

### Codex providers

By default a Codex session uses OpenRouter with the model `openai/gpt-6-astra`. Store
the OpenRouter key in a private file (mode 0600, owned by you):

```bash
mkdir -p ~/.codex/secrets && chmod 700 ~/.codex/secrets
install -m 600 /dev/null ~/.codex/secrets/openrouter_api_key   # then paste the key into it
```

The launcher reads the key from `--key-file`, else `~/.codex/secrets/openrouter_api_key`.
A Codex session needs that file: `OPENROUTER_API_KEY` serves only the Gemini tool and
Claude Code's `openrouter` provider. The key is sent only to an OpenRouter endpoint.

To use another Responses API provider, or the built-in profile with another model, write
a provider profile. The launcher reads `<provider-home>/config.toml` and then
`<provider-home>/<profile>.config.toml`, where the later file wins. `--provider-home`
defaults to `$CODEX_HOME`, else `~/.codex`. `--profile` defaults to the name in
`<provider-home>/experiment-profile`, else `openrouter`. Only the harness reads these
files; it writes the session's own Codex configuration from them.

```toml
# ~/.codex/myprovider.config.toml, used with --profile myprovider
model_provider = "myprovider"

[model_providers.myprovider]
name = "My provider"
base_url = "https://api.example.com/v1"
wire_api = "responses"                  # the only supported API
# http_headers = { "X-Team" = "robotics" }

[rax]
key_file = "secrets/myprovider_api_key" # required; relative to the provider home, mode 0600
model = "gpt-6-astra"                   # the session's model (default openai/gpt-6-astra)
# native_models = true                  # strip a leading "openai/" (OpenAI's own API)
# service_tier = "<tier>"               # Codex's service_tier for this profile's sessions
```

- **Key file.** A provider other than OpenRouter needs its own `[rax] key_file`.
  Codex receives the key in the variable `RAX_MODEL_KEY`; the provider table's
  `env_key` and `auth` settings are ignored.
- **ChatGPT plan.** Set `[rax] chatgpt_auth` to a Codex ChatGPT login file (an
  `auth.json`, relative to the provider home or absolute) in place of a key file. The
  session's Codex home links to that file, so Codex refreshes the login in place and
  every session using the profile shares it; a launch refuses a file that is not a
  ChatGPT login. The model name loses a leading `openai/`.
- **Built-in profile.** Without a provider table, the `openrouter` profile uses
  `https://openrouter.ai/api/v1`.

### Claude Code providers

Choose the provider when you prepare the session (`--agent-provider`, or the dashboard):

| Provider | Model | Credential, given to Claude Code at each launch |
| --- | --- | --- |
| `openrouter` (default) | `anthropic/claude-opus-5.5` | The OpenRouter key: `--key-file`, `~/.codex/secrets/openrouter_api_key` or `OPENROUTER_API_KEY` (as `ANTHROPIC_AUTH_TOKEN` with OpenRouter's base URL) |
| `claude-login` | `claude-opus-5-5` | A token for a dedicated Claude account, set up as below (as `CLAUDE_CODE_OAUTH_TOKEN`) |
| `anthropic` | `claude-opus-5-5` | An Anthropic API key: `--claude-key-file` at launch, or `ANTHROPIC_API_KEY` |

`--claude-model` overrides the model at launch. Subagents spawned with model `sonnet` or
`haiku` use Sonnet 5.5 and Haiku 4.5 on OpenRouter, and Claude Code's own aliases on the
first-party providers.

`claude-login` never uses your personal Claude login (`~/.claude`), and refuses it. Set up
a separate account for experiments once:

```bash
bash scripts/setup_claude_experiment_login.sh [--email experiment-account@example.com]
PYTHONPATH=. runtime/envs/host/bin/python -m harness.claude_cli.experiment_login check
```

The script signs that account in under its own configuration directory, issues a
long-lived `claude setup-token` token, and stores it as
`~/.config/rax/claude-experiment.token` (mode 0600). Set
`RAX_CLAUDE_EXPERIMENT_DIR` to use a directory other than `~/.config/rax`.
`check` verifies the token with free API calls, and `status` reports the stored identity
without the network. Every agent run reads the token file again, so running the script
again rotates the token for running sessions from their next run. The dashboard checks the token when it prepares a
`claude-login` session, and the launcher checks it before every launch and resume.
`--claude-key-file` points a launch at another token file.

When a headless `claude-login` run ends, the launcher probes the account's usage with a
one-token Haiku call; if the usage window is exhausted, the session waits for its reset
([Usage](USAGE.md#automatic-relaunch)).

### Gemini tool (optional)

The agent can call Gemini (`gemini_generate`, model `google/gemini-3.8-flash`) through
the MCP service, for example to check an image. The MCP service holds the OpenRouter key
from the same file or `OPENROUTER_API_KEY`; the agent never sees it. Gemini spending has
no session limit, and the dashboard shows each session's spend. Without an OpenRouter key
the tool stays listed, and every call answers `Gemini is not configured`.

## 5. Demonstrations

By default a session shows the agent the final frames of three demonstrations of its task
(`--demonstration-context terminal_state`), and it can also show it the three complete
episodes, read-only, in its workspace (`--expert-demonstrations 3`). Both come from the
official raw teleoperation dataset (`DexJoCo/DexJoCo-Datasets-Raw` on Hugging Face, at the revision pinned
in `dependencies.lock`). Install them per task, on a machine with a GPU:

```bash
runtime/envs/host/bin/python scripts/setup_dexjoco_demos.py --gpu 0 --task click_mouse --task pick_bucket
```

For each task (the rand_full set for `_random` tasks), the script downloads the first
candidates (`--candidates`, default 8), replays each in the official environment from its
recorded initial state, and compares its layout with the 150 official evaluation layouts.
The first three that succeed and show no evaluation layout become the task's experts, with a
SHA-256 manifest, under `runtime/reference-demos/dexjoco/<task>/`. If fewer than three
qualify, it says so; rerun with more `--candidates`. Sessions with
`--demonstration-context none` and no expert episodes need no demonstrations. See
[demonstrations](DEXJOCO_BACKEND.md#demonstrations).

## 6. Check the installation

```bash
bash scripts/doctor.sh
runtime/envs/host/bin/python scripts/dexjoco_doctor.py --gpu 0 --task click_mouse
```

`doctor.sh` checks the host environment (also the agent's Python), that bubblewrap can
create its namespaces, socat and git-lfs, the agent CLIs, the OpenRouter key and the GPUs.
It prints one line per check and exits non-zero if any check fails.

`dexjoco_doctor.py` starts the official simulator for each `--task` on the given GPU (on a
formal layout by default, `--collection 0` for an exploration one) through the same host
client sessions use. It reports the startup time, the observation contract (official state,
the evaluation cameras at 640x640, depth and calibration), a hold and an official action
chunk with their wall seconds per step, depth back-projection against the true object
position, the operator files, and the simulator-side checks (official layout determinism and
timing). It writes `doctor-report.json` under `runtime/diagnostics/` and exits non-zero if a
task failed.

Then run the unit tests:

```bash
runtime/envs/host/bin/python -m pytest -m "not gpu" -q local_tests/unit
```

They take about 2 minutes, use fake simulators and local fake model endpoints, and call
no paid model. The tests that run the real Codex CLI or Claude Code need that CLI on
`PATH`. See [local_tests/README.md](../local_tests/README.md) for the GPU tests, which
start real simulators.

## GPUs

Every simulator is one MuJoCo process that renders on the session's simulator GPU. A
session runs these at once:

- one episode simulator per live exploration environment (`--exploration-envs`, default
  1), plus its robot-only preview simulator once the agent uses the preview;
- during formal evaluation, one simulator per concurrent formal episode
  (`--formal-workers`, default 4).

Physics runs on the CPU, so CPU cores bound the concurrency as much as the GPU does. On a
small machine, lower `--formal-workers` and `--exploration-envs`. Several sessions can share
a GPU, or use different GPUs.
