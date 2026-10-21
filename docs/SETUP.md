# Setup

This guide installs everything a session needs on one Linux machine. Each step is one
script, and two doctors check the result at the end. The repository holds only source
code. Everything the scripts download or build goes under `runtime/`, a link to a
directory you choose, or into Docker's image store. Credentials stay in your home directory.

| Step | What | Needed for |
| --- | --- | --- |
| [0](#0-host-requirements) | Linux host, NVIDIA RTX GPU and driver, Docker | everything |
| [1](#1-sources) | Pinned HumanoidArena checkout | everything |
| [2](#2-host-environment) | The harness's Python environment `runtime/envs/host` | everything |
| [3](#3-simulator-image) | Simulator image `rax-humanoid:isaacsim6.1` (Isaac Sim 6.1, Isaac Lab, cuRobo) | every session |
| [4](#4-humanoidarena-assets-and-sonic) | HumanoidArena assets and the SONIC encoder/decoder | every session |
| [5](#5-agent-image) | Offline agent image `rax-auto-research:dev` | every session |
| [6](#6-agent-cli-and-model-credentials) | Codex CLI or Claude Code, and a model provider | every session |
| [7](#7-expert-episodes-optional) | Three expert episodes per task | sessions with expert episodes |
| [8](#8-check-the-installation) | Doctors and tests | recommended |

## 0. Host requirements

- **Operating system:** Linux x86_64. Tested on Rocky Linux 8.
- **GPU:** one NVIDIA RTX GPU (Isaac Sim does not run on data-center GPUs without RT cores)
  with a driver that supports Isaac Sim 6.1 and CUDA 12.8. Tested with driver 610.57.04 on
  RTX PRO 6000 Blackwell GPUs. Only the simulators use a GPU; agents and their programs run
  on the CPU. Memory scales with concurrent simulators; see [GPU memory](#gpu-memory).
- **Docker:** rootful Docker with BuildKit, usable by your (non-root) user without sudo:
  add yourself to the `docker` group. The simulator image needs the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  (`docker run --gpus`); the agent image does not.
  - Each session's workspace is a capped ext4 image, loop-mounted by a short-lived
    `--privileged` container. The daemon must allow privileged containers, and the host
    needs `/dev/loop-control`.
  - Put Docker's image store on a fast disk; the simulator image is about 40 GB and the
    agent image about 12 GB.
  - Rootless Podman works too: set `CONTAINER_CLI=podman` for the simulator image scripts.
- **Host tools:** `git`, `git-lfs`, `curl`, `unzip`, `sha256sum`, `mkfs.ext4`, `findmnt` and
  `nvidia-smi`. Everything else (conda, Python) is installed by the scripts.
- **Disk:** use a fast local SSD for `runtime/`.

  | Item | Size |
  | --- | --- |
  | HumanoidArena source | 0.2 GB |
  | Host environment and Miniforge | 1 GB |
  | HumanoidArena assets (plus a 2.7 GB archive while installing) and SONIC models | 3.3 GB |
  | Simulator image (Docker store) | 40 GB |
  | Agent image (plus its 3.8 GB wheelhouse; Docker store) | 12 GB |
  | Expert episodes, all tasks (optional) | 9 GB, and 3.5 GB more for their camera-only sets |
  | One session | from a few GB to over a hundred GB (see below) |

  A session keeps every episode's recording. Each session's workspace image is sparse, so
  its nominal size (`--workspace-gib`, default 1024) takes no disk until used.

## 1. Sources

```bash
git clone --branch humanoid-arena/container https://github.com/robot-agent-exploration/rax.git
cd rax
export RAX_RUNTIME_ROOT=/fast-disk/rax/runtime    # optional; default: .local-runtime/ in the repository
bash scripts/bootstrap_sources.sh
```

The bootstrap script:

- links `runtime/` to `$RAX_RUNTIME_ROOT`, or to `.local-runtime/` when the variable is
  unset;
- clones HumanoidArena at the commit pinned in `dependencies.lock` next to the runtime
  directory, linked as `HumanoidArena/`. Its large-file pointers stay unfetched: the assets
  come in step 4.

Session artifacts, capped workspaces and image builds must live under the *artifact root*.
By default that is the directory `runtime/` links to, so nothing needs setting. If you set
`RAX_ARTIFACT_ROOT`, it must be that directory or one of its ancestors.

Run every later step from the repository root.

## 2. Host environment

```bash
bash scripts/setup_host_env.sh
```

The launchers, the MCP service, the workspace generator, the dashboard and the tests run in
`runtime/envs/host`: Python 3.12 with NumPy, Pillow, Jinja2, msgpack, PyYAML, imageio,
SciPy, OpenCV and pytest at the versions in `dependencies.lock`. The script installs the pinned
Miniforge into `runtime/miniforge3` (after checking its SHA-256) and creates the prefix. It
holds no simulator: the host never imports Isaac Sim, Isaac Lab or the HumanoidArena task code.
`scripts/host_env.sh` sets the environment variables if you want them in a shell.

## 3. Simulator image

```bash
bash scripts/build_humanoid_image.sh --accept-nvidia-eula
```

The flag accepts the [NVIDIA Isaac Sim EULA](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html).
Running the harness also accepts it on every Isaac Sim launch (`OMNI_KIT_ACCEPT_EULA`).

The image (`docker/humanoid/Dockerfile`) starts from NVIDIA's `isaac-sim:6.1.0` and adds the
CUDA 12.8 toolkit, PyTorch 2.11 (CUDA 12.8), Isaac Lab with the Isaac Sim 6.1 patch in
`patches/isaaclab-isaacsim61.patch`, cuRobo, and onnxruntime for the SONIC models, all at the
commits and versions in `dependencies.lock`. The build checks the patch's SHA-256 and ends
by loading the planner and the ONNX runtime on CUDA. It takes about half an hour.

Each simulator process runs in a container of this image, started by
`scripts/humanoid_python.sh`, the default `HUMANOID_PYTHON`. It mounts the repository, the
runtime and the HumanoidArena checkout at their host paths, gives the container the one
GPU the session chose, and uses host networking so the harness reaches the simulator's RPC
port on 127.0.0.1.

**Isaac Sim 5.0 instead.** Upstream HumanoidArena installs Isaac Sim 5.0.0 with Isaac Lab
`release/2.2.0` (`isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh` in the checkout). The
paper's experiments ran on the 6.1 image above, and only that image is tested here. To use
an upstream 5.0 environment instead, add cuRobo (the commit in `dependencies.lock`) and
onnxruntime-gpu to it and point `HUMANOID_PYTHON` at its interpreter, for example
`export HUMANOID_PYTHON=/path/to/env/bin/python`. The compatibility code in
`services/humanoid/sim/` adapts to the installed Kit API, and its reset and collider fixes
apply on either version.

## 4. HumanoidArena assets and SONIC

For DeepSeek Harness or Kimi Code, follow the [agent harness guide](https://github.com/robot-agent-exploration/rax/blob/main/docs/AGENT_HARNESSES.md) after installing the host dependencies.

```bash
bash scripts/setup_humanoid_assets.sh
```

This downloads the released HumanoidArena simulation assets (a 2.7 GB archive, 3.2 GB
installed) and the released SONIC encoder, decoder and observation configuration from the
Hugging Face repository `nvidia/GEAR-SONIC`, at the revisions pinned in `dependencies.lock`,
checks every SHA-256, and installs them into `runtime/humanoid-assets/{arena,sonic}`.
`--arena-zip` and `--sonic-dir` take local copies instead, if they match the pins.

## 5. Agent image

```bash
runtime/envs/host/bin/python scripts/build_agent_image.py
```

The agent, its subagents and every robot program it writes run in this image: Python
3.12 (`python:3.12-slim`) with git, ripgrep, NumPy, SciPy, Pillow, OpenCV, h5py and a
CPU-usable PyTorch. The script downloads the pinned wheels into
`runtime/agent-image/wheels`, records their SHA-256, and builds the tag
`rax-auto-research:dev`. It takes about 5 minutes.

The containers run without network. Each session pins the image digest when it is
prepared, so rebuilding the tag never changes a prepared or running session.
`--offline` rebuilds from the downloaded wheels and refuses if they changed.

## 6. Agent CLI and model credentials

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

[rax]
model = "gpt-6-astra"                   # the session's only model (default openai/gpt-6-astra)
# key_file = "secrets/myprovider_key"   # instead of env_key; relative to the provider home
# chatgpt_auth = "auth.json"            # a Codex ChatGPT login, re-read on every call
# native_models = true                  # strip a leading "openai/" from the model name
```

The credential comes from the first of these that is set: `[rax] chatgpt_auth`, then
`[rax] key_file`, then the OpenRouter key (OpenRouter endpoints only), then `env_key`. A
provider that needs an `auth` command is refused. Without a provider table, the
`openrouter` profile uses `https://openrouter.ai/api/v1`.

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
PYTHONPATH=. runtime/envs/host/bin/python -m harness.claude_cli.experiment_login check
```

The script signs that account in under its own configuration directory, issues a
long-lived `claude setup-token` token, and stores it as
`~/.config/rax/claude-experiment.token` (mode 0600). Set `RAX_CLAUDE_EXPERIMENT_DIR` to use
a directory other than `~/.config/rax`. `check` makes one free API call, and `status`
reports the stored identity without the network. The relay reads the token on every
request, so running the script again rotates the token for running sessions too. The
dashboard checks the token when it prepares a `claude-login` session, and the launcher
checks it before every launch and resume. `--claude-key-file` points a launch at another
token file.

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

## 7. Expert episodes (optional)

A session can mount three expert teleoperation episodes of its task, read-only, for the
exploring agent (`--expert-demonstrations 3`; the default is none). They come from the
HumanoidArena raw release (`Twang2026/HumanoidArena_raw` on ModelScope, CC BY-NC 4.0):

```bash
runtime/envs/host/bin/python scripts/setup_humanoid_demos.py              # all tasks, about 9 GB
runtime/envs/host/bin/python scripts/setup_humanoid_demos.py --images     # their camera-only sets
```

The first command downloads each task's three pinned episodes
(`services/humanoid/expert_sources.json`), checks their size and SHA-256, and derives a
per-frame JSON view, a byte-offset index and front-camera previews; `--task` limits it to
some tasks. The view's virtual points come from forward kinematics of the released G1 MuJoCo
description, downloaded from `NVlabs/GR00T-WholeBodyControl` at the pinned commit and checked
by SHA-256. The second derives, from the installed episodes, the camera-only
sets that `--expert-content images` mounts instead. See
[expert episodes](HUMANOID_BACKEND.md#expert-episodes).

## 8. Check the installation

```bash
bash scripts/doctor.sh
runtime/envs/host/bin/python scripts/humanoid_doctor.py --gpu 0 --task opendoor
```

`doctor.sh` checks the host environment, Docker access and the agent image, loop devices
and host tools, the agent CLIs, the OpenRouter key and the GPUs. It prints one line per
check and exits non-zero if any check fails.

`humanoid_doctor.py` starts one simulator on the given GPU, on an exploration layout,
through the same host client sessions use. It reports the startup time, the observation
contract (camera, depth, joints, Dex3 fingers), a standing hold through SONIC (tracking
error, tilt, wall seconds per step), the depth-pixel geometry, the peak GPU memory and a
planned arm motion (`--no-planner` skips it), and writes a JSON report next to the run. It
takes a few minutes; the first run also builds the shader and kernel caches.

Then run the unit tests:

```bash
runtime/envs/host/bin/python -m pytest -m "not gpu" -q local_tests/unit
```

They take about 3 minutes, use fake simulators and a local mock model provider, and call
no paid model. Tests that run real containers need the agent image. See
[local_tests/README.md](../local_tests/README.md).

## GPU memory

Every simulator is one Isaac Sim process on the session's simulator GPU. A session runs
these at once:

- one simulator per live exploration environment (`--exploration-envs`, default 1);
- during formal evaluation, one simulator per concurrent formal episode
  (`--formal-workers`, default 4).

Each takes about 9 GB at steady state on an RTX PRO 6000. With the defaults, exploration
needs about 9 GB and formal evaluation about 36 GB (4 × 9). On a smaller GPU, lower
`--formal-workers` and `--exploration-envs`.

Several sessions can share a GPU when their simulators fit, or use different GPUs.
