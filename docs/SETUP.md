# Setup

This guide installs everything a session needs on one Linux machine. Each step is one
script, and `scripts/robodojo_doctor.sh` checks the result at the end. The repository
holds only source code. Everything the scripts download or build goes under `runtime/`,
a link to a directory you choose. Credentials stay in your home directory.

| Step | What | Needed for |
| --- | --- | --- |
| [0](#0-host-requirements) | Linux host, NVIDIA GPU and driver, Docker | everything |
| [1](#1-sources) | Pinned RoboDojo checkout with its patches, BEHAVIOR-1K source | everything |
| [2](#2-simulator-environment) | Isaac Sim 5.1 environment `runtime/envs/robodojo` | everything |
| [3](#3-robodojo-assets) | RoboDojo assets | RoboDojo sessions |
| [4](#4-agent-image) | Offline agent image `robodojo-auto-research:dev` | every session |
| [5](#5-agent-cli-and-model-credentials) | Codex CLI or Claude Code, and a model provider | every session |
| [6](#6-behavior-1k-optional) | BEHAVIOR-1K environment, datasets and demonstrations | BEHAVIOR sessions |
| [7](#7-check-the-installation) | Doctor and tests | recommended |

## 0. Host requirements

- **Operating system:** Linux x86_64. Tested on Ubuntu 24.04.
- **GPU:** one NVIDIA RTX GPU (Isaac Sim does not run on data-center GPUs without RT cores) with a driver that
  supports Isaac Sim 5.1 and CUDA 12.8. Tested with driver 580.173.02 on RTX PRO 6000
  Blackwell GPUs. Only the simulators use a GPU; agents and their programs run on the CPU.
  Memory scales with concurrent simulators; see [GPU memory](#gpu-memory).
- **Docker:** rootful Docker with BuildKit, usable by your (non-root) user without sudo:
  add yourself to the `docker` group. The NVIDIA Container Toolkit is not needed.
  - Each session's workspace is a capped ext4 image, loop-mounted by a short-lived
    `--privileged` container. The daemon must allow privileged containers, and the host
    needs `/dev/loop-control`.
  - Put Docker's image store on a fast disk too; the agent image is about 12 GB.
- **Host tools:** `git`, `git-lfs`, `curl`, `sha256sum`, `mkfs.ext4`, `findmnt` and
  `nvidia-smi`. Everything else (conda, Python, CUDA compiler) is installed by the scripts.
- **Disk:** use a fast local SSD for `runtime/`.

  | Item | Size |
  | --- | --- |
  | RoboDojo assets (assets plus their git-lfs cache) | 66 GB |
  | Simulator environment and Miniforge | 23 GB |
  | Agent image (plus its 3.8 GB wheelhouse) | 12 GB |
  | BEHAVIOR-1K source | 1 GB |
  | BEHAVIOR-1K environment (optional) | 22 GB |
  | BEHAVIOR-1K datasets (optional; about 70 GB while unpacking) | 37 GB |
  | One session | from a few GB to several hundred GB (see below) |

  A session keeps every episode's recording. A long BEHAVIOR session with agent formal
  can reach hundreds of GB. Each session's workspace image is sparse, so its nominal
  size (`--workspace-gib`, default 1024) takes no disk until used.

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

Session artifacts, capped workspaces and image builds must live under the *artifact
root*. By default that is the directory `runtime/` links to, so nothing needs setting.
If you set `ROBODOJO_ARTIFACT_ROOT`, it must be that directory or one of its ancestors.

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

## 3. RoboDojo assets

```bash
bash scripts/setup_assets.sh
```

This downloads the RoboDojo assets (robots, objects, materials and saved evaluation
layouts) from the Hugging Face dataset `RoboDojo-Benchmark/RoboDojo`, at the commit
pinned in `dependencies.lock`. It needs `git-lfs` and about 66 GB. The script puts them
in `runtime/robodojo-cache` and links them as `RoboDojo/Assets`.

## 4. Agent image

For DeepSeek Harness or Kimi Code, follow the [agent harness guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md) after installing the host dependencies.

```bash
runtime/envs/robodojo/bin/python scripts/build_agent_image.py
```

The agent, its subagents and every robot program it writes run in this image: Python
3.12 (`python:3.12-slim`) with git, ripgrep, NumPy, SciPy, Pillow, OpenCV, h5py and a
CPU-usable PyTorch. The script downloads the pinned wheels into
`runtime/agent-image/wheels`, records their SHA-256, and builds the tag
`robodojo-auto-research:dev`. It takes about 5 minutes.

The containers run without network. Each session pins the image digest when it is
prepared, so rebuilding the tag never changes a prepared or running session.
`--offline` rebuilds from the downloaded wheels and refuses if they changed.

## 5. Agent CLI and model credentials

Install at least one agent CLI on the host as a **native binary**. The launcher mounts
the binary into the agent container, which has no Node.js, so npm-installed wrappers do
not work.

- **Codex CLI** (default): the standalone release binary from
  [github.com/openai/codex/releases](https://github.com/openai/codex/releases). Tested with 0.162.1.
- **Claude Code:** the native installer from
  [docs.claude.com](https://docs.claude.com/en/docs/claude-code/setup). Tested with 2.1.296.

Credentials never enter the container. A relay on the host holds the credential, and
the CLI in the container talks to that relay.

### Codex providers

By default a Codex session uses OpenRouter with the model `openai/gpt-6-astra`. Store
the OpenRouter key in a private file, or export it:

```bash
mkdir -p ~/.codex/secrets && chmod 700 ~/.codex/secrets
install -m 600 /dev/null ~/.codex/secrets/openrouter_api_key   # then paste the key into it
# or: export OPENROUTER_API_KEY=...
```

The launcher looks for the key in `--key-file`, then `~/.codex/secrets/openrouter_api_key`,
then `OPENROUTER_API_KEY`. It sends this key only to an OpenRouter endpoint.

To use another Responses API provider, write a provider profile. The launcher reads
`<provider-home>/config.toml` and then `<provider-home>/<profile>.config.toml`, where the
later file wins. `--provider-home` defaults to `$CODEX_HOME`, else `~/.codex`. `--profile`
defaults to the name in `<provider-home>/experiment-profile`, else `openrouter`. Only the
harness reads these files: the Codex CLI in the container is configured to talk to the
relay.

```toml
# ~/.codex/myprovider.config.toml, used with --profile myprovider
model_provider = "myprovider"

[model_providers.myprovider]
name = "My provider"
base_url = "https://api.example.com/v1"
wire_api = "responses"                  # the only supported API
env_key = "MY_PROVIDER_API_KEY"         # read when the session is deployed
# http_headers = { "X-Team" = "robotics" }

[robodojo]
model = "gpt-6-astra"                   # the session's only model (default openai/gpt-6-astra)
# key_file = "secrets/myprovider_key"   # instead of env_key; relative to the provider home
# chatgpt_auth = "auth.json"            # a Codex ChatGPT login, re-read on every call
# native_models = true                  # strip a leading "openai/" from the model name
```

The credential comes from the first of these that is set: `[robodojo] chatgpt_auth`, then
`[robodojo] key_file`, then the OpenRouter key (OpenRouter endpoints only), then
`env_key`. A provider that needs an `auth` command is refused. Without a provider table,
the `openrouter` profile uses `https://openrouter.ai/api/v1`.

The relay admits only the session's model at the session's reasoning effort
(`--agent-effort`), only client-executed tools (no hosted tools, plugins or web search),
only inline images, and no `previous_response_id`.

### Claude Code providers

Choose the provider when you prepare the session (`--agent-provider`, or the dashboard):

| Provider | Model | Credential |
| --- | --- | --- |
| `openrouter` (default) | `anthropic/claude-opus-5.5` | The OpenRouter key, as for Codex |
| `claude-login` | `claude-opus-5-5` | A token for a dedicated Claude account, set up as below |
| `anthropic` | `claude-opus-5-5` | An Anthropic API key: `--claude-key-file` at launch, or `ANTHROPIC_API_KEY` |

`--claude-model` overrides the model at launch. Claude Code's subagent alias `sonnet` maps
to Sonnet 5.5 on OpenRouter and to Haiku 4.5 on the first-party providers, and `haiku`
maps to Haiku 4.5.

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
`check` makes one free API call, and `status` reports the stored identity without the
network. The relay reads the token on every request, so running the script again
rotates the token for running sessions too. The dashboard checks the token when it
prepares a `claude-login` session, and the launcher checks it before every launch and
resume. `--claude-key-file` points a launch at another token file.

### What the relays do

- **Retries.** Provider answers 429, 502, 503 and 504 (and 529 for Claude) are retried for
  up to 10 minutes, following the provider's `Retry-After`. After that, the provider's
  status, headers and error message reach the CLI unchanged.
- **Refusals.** A request the relay refuses (another model, a server-side tool, media by
  URL) is answered 403 with a JSON body naming the reason.
- **Provider failures.** A provider that is unreachable, times out or breaks off
  mid-response gives 502 with the reason.
- **Concurrency.** A relay serves the agent, its subagents and 2 spare calls at once,
  and at least 8. A call over the limit is answered 503 with `Retry-After: 1`, and the
  CLIs retry it.

For a Codex session, the launcher writes the provider address and its headers, including
the key, to `<session>/model-relay.json` (mode 0600, never mounted into a container), so
keep session directories private; a ChatGPT login is stored only as a path and read on
every call. A Claude session stores only the key file's path, or reads the environment
variable at each launch, and its relay reads the credential for every request.

### Gemini tool (optional)

The agent can call Gemini (`gemini_generate`, model `google/gemini-3.8-flash`) through
the MCP service, for example to check an image. It uses the OpenRouter key from the
same file or `OPENROUTER_API_KEY`, and each session may spend at most $10. Without an
OpenRouter key the tool stays listed, and every call answers `Gemini is not configured`.

## 6. BEHAVIOR-1K (optional)

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

## 7. Check the installation

```bash
bash scripts/robodojo_doctor.sh
```

The doctor checks the RoboDojo commit, submodules, patches and assets; the simulator
environment's versions and its CUDA compiler; Docker access and the agent image; loop
devices and host tools; the agent CLIs; the OpenRouter key; and the GPUs. It prints one
line per check and exits non-zero if any check fails.

Then run the unit tests:

```bash
runtime/envs/robodojo/bin/python -m pytest -m "not gpu" -q local_tests/unit
```

They take about 4 minutes, use fake simulators and a local mock model provider, and
call no paid model. Tests that run real containers need the agent image. See
[local_tests/README.md](../local_tests/README.md) for the GPU tests, which start real
simulators.

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
