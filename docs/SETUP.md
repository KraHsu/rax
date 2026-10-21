# Setup

This guide installs everything a session needs on one Linux machine. Each step is one
script or package, and `scripts/robodojo_doctor.sh` checks the result at the end. The
repository holds only source code. Everything the scripts download or build goes under
`runtime/`, a link to a directory you choose. Credentials stay in your home directory.

| Step | What | Needed for |
| --- | --- | --- |
| [0](#0-host-requirements) | Linux host, NVIDIA GPU and driver, bubblewrap and socat | everything |
| [1](#1-sources) | Pinned RoboDojo checkout with its patches, BEHAVIOR-1K source | everything |
| [2](#2-simulator-environment) | Isaac Sim 5.1 environment `runtime/envs/robodojo` | everything |
| [3](#3-robodojo-assets) | RoboDojo assets | RoboDojo sessions |
| [4](#4-agent-cli-and-model-credentials) | Codex CLI or Claude Code, and a model provider | every session |
| [5](#5-behavior-1k-optional) | BEHAVIOR-1K environment, datasets and demonstrations | BEHAVIOR sessions |
| [6](#6-check-the-installation) | Doctor and tests | recommended |

## 0. Host requirements

- **Operating system:** Linux x86_64. Tested on Ubuntu 24.04.
- **GPU:** one NVIDIA RTX GPU (Isaac Sim does not run on data-center GPUs without RT cores) with a driver that
  supports Isaac Sim 5.1 and CUDA 12.8. Tested with driver 580.173.02 on RTX PRO 6000
  Blackwell GPUs. Only the simulators use a GPU; agents and their programs run on the CPU.
  Memory scales with concurrent simulators; see [GPU memory](#gpu-memory).
- **Sandboxes:** `bubblewrap` (`bwrap`), able to create unprivileged user namespaces as
  your (non-root) user. Submitted robot programs run in it, and Claude Code's sandbox
  uses it together with `socat`. Codex brings its own sandbox. No Docker is needed.
  On Ubuntu: `sudo apt install bubblewrap socat`.
- **Host tools:** `git`, `git-lfs`, `curl`, `sha256sum` and `nvidia-smi`. Everything else
  (conda, Python, CUDA compiler) is installed by the scripts.
- **Disk:** use a fast local SSD for `runtime/`.

  | Item | Size |
  | --- | --- |
  | RoboDojo assets (assets plus their git-lfs cache) | 66 GB |
  | Simulator environment and Miniforge | 23 GB |
  | BEHAVIOR-1K source | 1 GB |
  | BEHAVIOR-1K environment (optional) | 22 GB |
  | BEHAVIOR-1K datasets (optional; about 70 GB while unpacking) | 37 GB |
  | One session | from a few GB to several hundred GB (see below) |

  A session keeps every episode's recording. A long BEHAVIOR session with agent formal
  can reach hundreds of GB. The agent's workspace has no size cap of its own; it lives in
  the session directory on the same disk.

## 1. Sources

```bash
git clone <this-repository> rax
cd rax
export ROBODOJO_RUNTIME_ROOT=/fast-disk/rax/runtime    # optional; default: .local-runtime/ in the repository
bash scripts/bootstrap_sources.sh
```

The bootstrap script:

- links `runtime/` to `$ROBODOJO_RUNTIME_ROOT`, or to `.local-runtime/` when the variable
  is unset;
- clones RoboDojo at the commit pinned in `dependencies.lock` with its pinned submodules,
  and applies the two patches in `patches/` after checking their SHA-256 (see
  [patches/README.md](../patches/README.md));
- shallow-clones BEHAVIOR-1K at its pinned tag (about 1 GB) next to the runtime
  directory, linked as `BEHAVIOR-1K/`.

Sessions must live under the *artifact root*. By default that is the directory
`runtime/` links to, so nothing needs setting. If you set `ROBODOJO_ARTIFACT_ROOT`, it
must be that directory or one of its ancestors.

Run every later step from the repository root.

## 2. Simulator environment

```bash
bash scripts/setup_robodojo_env.sh --accept-nvidia-eula
```

The flag accepts the [NVIDIA Isaac Sim EULA](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
Running the harness also accepts it on every Isaac Sim launch (`OMNI_KIT_ACCEPT_EULA`).

The script:

1. installs the pinned Miniforge into `runtime/miniforge3`, after checking its SHA-256;
2. creates the conda prefix `runtime/envs/robodojo` (Python 3.11, rendering libraries,
   and the CUDA 12.8 compiler that cuRobo builds its kernels with);
3. runs the steps of RoboDojo's own installer with its pins: Isaac Sim 5.1, PyTorch
   2.7.0 (CUDA 12.8), Isaac Lab and cuRobo from the pinned submodules. It does not run
   upstream's submodule update, which would move them off their pins;
4. installs the harness's own Python dependencies;
5. checks the versions against `dependencies.lock` and that cuRobo's compiled extension
   loads.

It takes about 10 minutes with a fast connection, and reruns skip finished steps.

This environment is also the agent's Python: the agent's commands and its submitted
programs run `runtime/envs/robodojo/bin/python`, with NumPy, SciPy, Pillow, OpenCV,
h5py, scikit-learn and PyTorch, offline and without a GPU.

## 3. RoboDojo assets

```bash
bash scripts/setup_assets.sh
```

This downloads the RoboDojo assets (robots, objects, materials and saved evaluation
layouts) from the Hugging Face dataset `RoboDojo-Benchmark/RoboDojo`, at the commit
pinned in `dependencies.lock`. It needs `git-lfs` and about 66 GB. The script puts them
in `runtime/robodojo-cache` and links them as `RoboDojo/Assets`.

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

[robodojo]
key_file = "secrets/myprovider_api_key" # required; relative to the provider home, mode 0600
model = "gpt-6-astra"                   # the session's model (default openai/gpt-6-astra)
# native_models = true                  # strip a leading "openai/" (OpenAI's own API)
# service_tier = "<tier>"               # Codex's service_tier for this profile's sessions
```

- **Key file.** A provider other than OpenRouter needs its own `[robodojo] key_file`.
  Codex receives the key in the variable `ROBODOJO_MODEL_KEY`; the provider table's
  `env_key` and `auth` settings are ignored.
- **ChatGPT plan.** Set `[robodojo] chatgpt_auth` to a Codex ChatGPT login file (an
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
PYTHONPATH=. runtime/envs/robodojo/bin/python -m harness.claude_cli.experiment_login check
```

The script signs that account in under its own configuration directory, issues a
long-lived `claude setup-token` token, and stores it as
`~/.config/robodojo/claude-experiment.token` (mode 0600). Set
`ROBODOJO_CLAUDE_EXPERIMENT_DIR` to use a directory other than `~/.config/robodojo`.
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

## 5. BEHAVIOR-1K (optional)

```bash
bash scripts/setup_behavior_env.sh --accept-nvidia-eula      # runtime/envs/behavior, about 22 GB
bash scripts/setup_behavior_assets.sh --accept-dataset-tos   # datasets, about 37 GB installed
bash scripts/setup_behavior_demos.sh                         # demonstration keyframes, about 0.5 GB
runtime/envs/robodojo/bin/python scripts/behavior_doctor.py --gpu 0
```

- The environment script builds the BEHAVIOR simulator's own conda prefix, separate
  from `runtime/envs/robodojo`, with the pinned BEHAVIOR-1K packages (OmniGibson, BDDL,
  cuRobo).
- `--accept-dataset-tos` accepts the BEHAVIOR-1K dataset terms: non-commercial research
  use, with the data used only within OmniGibson. The script downloads the pinned
  dataset revisions, checks their SHA-256, and unpacks them under the runtime directory.
  It needs about 70 GB while unpacking; `--keep-zips` keeps the downloads.
- The demonstration script caches head-camera keyframes of the official teleoperation
  demonstrations, which sessions can show the agent. It takes about 15 minutes.
- The BEHAVIOR doctor starts one simulator on the given GPU and checks a short episode.
  It takes about 6 minutes on a cold cache.

See [BEHAVIOR backend](BEHAVIOR_BACKEND.md) for how the backend works.

## 6. Check the installation

```bash
bash scripts/robodojo_doctor.sh
```

The doctor checks the RoboDojo commit, submodules, patches and assets; the simulator
environment's versions and its CUDA compiler; that bubblewrap can create its namespaces,
and socat and git-lfs; the agent CLIs; the OpenRouter key; and the GPUs. It prints one
line per check and exits non-zero if any check fails.

Then run the unit tests:

```bash
runtime/envs/robodojo/bin/python -m pytest -m "not gpu" -q local_tests/unit
```

They take about 2 minutes, use fake simulators and local fake model endpoints, and call
no paid model. The tests that run the real Codex CLI or Claude Code need that CLI on
`PATH`. See [local_tests/README.md](../local_tests/README.md) for the GPU tests, which
start real simulators.

## GPU memory

Every simulator is one Isaac Sim process on the session's simulator GPU. A session
runs these at once:

- one simulator per live exploration environment (`--exploration-envs`, default 1), plus
  one robot-only preview simulator;
- during formal evaluation, one simulator per concurrent formal episode
  (`--formal-workers`, default 4 for RoboDojo and 2 for BEHAVIOR).

Measured on an RTX PRO 6000, at steady state after startup:

| Process | RoboDojo (RGB-D) | BEHAVIOR-1K |
| --- | --- | --- |
| Simulator: one per exploration environment and one per concurrent formal episode | 6.8 GB | 10.5 GB |
| Robot-only preview: starts on first use, next to its simulator | 6.0 GB | 4.5 GB |

With the defaults, exploration needs about 13 GB (RoboDojo) or 15 GB (BEHAVIOR). Formal
evaluation needs one simulator per worker, about 27 GB (4 × 6.8) or 21 GB (2 × 10.5),
plus a preview for each episode whose program uses it. On a smaller GPU, lower
`--formal-workers` and `--exploration-envs`.

Several sessions can share a GPU when their simulators fit, or use different GPUs.
