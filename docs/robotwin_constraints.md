# RoboTwin recording with embedded geometric constraints

The RoboTwin backend uses existing RAX lifecycle/controller tools. Clean
exploration can record training episodes; formal randomized evaluation runs
frozen code and cannot write training constraints. No native dependency, asset
or model is installed automatically.

Set these in the existing GPU-server environment:

```bash
export ROBOTWIN_PATH=/mnt/data/liangjiaqi/RoboTwin
export ROBOTWIN_PYTHON=/path/to/existing/robotwin/python
export ROBOTWIN_DATASET_DIR=/mnt/data/liangjiaqi/rax_train/datasets/robotwin
export ROBOTWIN_EPISODE_MANIFEST=/mnt/data/liangjiaqi/rax_train/manifests/task.json
```

`ROBOTWIN_PYTHON` must already provide native RoboTwin dependencies and h5py,
Pillow, NumPy, YAML and msgpack. The host process uses RAX's existing environment.
The adapter starts one native subprocess per episode and only exposes RGB and
measured robot proprioception. Native command budgets apply in both collections.

An operator-owned manifest selects native-feasible scenes before agent access:

```bash
"$ROBOTWIN_PYTHON" scripts/prepare_robotwin_manifest.py \
  --task adjust_bottle --clean-episodes 5 --formal-episodes 100 \
  --output "$ROBOTWIN_EPISODE_MANIFEST"
python scripts/configure_auto_research.py --benchmark robotwin \
  --task adjust_bottle --observation-profile rgb-only --episodes 5
```

Use the normal configure tool's other required session/output arguments. Clean
collection 0 uses native training mode; collection 1 uses randomized native test
mode and unseen instructions. The manifest records actual seeds and source
commit. It is an explicit reproducible scene set, not a claim of matching every
official evaluation run's selected seed list. No RoboTwin-validated RAX policy
checkpoint is included.

The agent receives [robot-interface](../agent_skills/control/robotwin/robot-interface/SKILL.md)
and [geometric-constraints](../agent_skills/control/robotwin/geometric-constraints/SKILL.md).
The latter is the annotation template and review procedure. There are no
operation categories. The tool `robotwin_constraints` lists/upserts arbitrary
compositions of implemented base constraints, with optimistic revision checks.

Recording emits native-compatible HDF5 observation/action fields plus
`trajectory_constraints/json`, revision history and an instruction attribute.
The schema lives in `services/robotwin/constraint_schema.py`, kept byte-identical
to `OpenWAM/openwam/train/constraint_schema.py`. Do not change one copy alone.
Actual recorded frame indices anchor activation; predicted actions cannot turn
off a label. Unverified bounds are retained as proposed/unknown and excluded
from training. A new mathematical base type needs versioned code and tests.

Each initial observation and subsequent native command boundary is recorded.
These are variable-duration samples; no fixed Hz or physical velocity is
assumed. Completed successful data goes under
`<dataset>/<task>/<embodiment>_clean_50/data/episode_*.hdf5`; failures remain under
`rejected/`, incomplete episodes stay hidden as partial files. OpenWAM reads the
embedded constraints and exact source action contract with each sample.

The loss never runs simulation. Source command targets, achieved EEF samples
and object dynamics are distinct: labels on recorded EEF trajectories cannot
prove grasps, support or contact forces. Full native collection/evaluation is
still required beyond the fake-environment and HDF5 integration tests.

## Local Codex with a remote GPU simulator

Codex and the RAX controller run on the operator's local machine. Only the native
simulator runs on the GPU server. SSH forwards its loopback RPC socket into a
private local Unix socket. Credentials stay local; HDF5 data stays remote.
The operator state is mirrored for scoring, while only RGB/proprioception enters
the agent's tool replies. A broken transport is never retried as an action.

On the local machine, save an operator-only JSON configuration (e.g.
`runtime/robotwin_remote.json`, ignored by git):

```json
{
  "host": "operator@gpu-host",
  "port": 22,
  "rax_root": "/work/rax",
  "python": "/existing/robotwin/env/bin/python",
  "robotwin_path": "/work/RoboTwin",
  "manifest": "/work/manifests/adjust_bottle.json",
  "dataset_root": "/work/datasets/robotwin",
  "output_root": "/work/rax-runs"
}
```

Both machines need the same RAX code. Prepare the manifest on the server first,
with at least as many entries as the requested exploration/formal counts. The
launcher fetches only that small manifest and the native task step limits. Every
episode checks that the local and remote manifest hashes match. Optional
`pythonpath` (a list of remote directories) and `environment` allow existing
simulator environment settings; neither is exposed to the agent.

Run in a **normal local terminal** with Python 3.11+, NumPy, Pillow, msgpack,
PyYAML and Jinja2, a logged-in Codex CLI, SSH access, and bubblewrap:

```bash
python scripts/run_robotwin_remote.py \
  --remote-config runtime/robotwin_remote.json \
  --task adjust_bottle --episodes 6 --formal-episodes 5 --sim-gpu 0
```

It uses the model from the local `~/.codex/config.toml` and links the existing
ChatGPT login into this session's isolated Codex home. No login is copied to the
server. External Gemini perception is disabled. The experiment remains in the
foreground; Ctrl+C cleans up the local harness and remote episode. Resume with
the same options plus `--resume /absolute/path/to/operator-input.json`.

Local commands must be allowed to create Unix sockets and bubblewrap namespaces.
An enclosing coding-agent command sandbox can prohibit these even when the normal
terminal supports them. The launcher reports this during preflight; it does not
disable RAX isolation. `--prepare-only` prepares metadata/configuration without
starting Codex or an episode. Five formal scenes are a connectivity/policy pilot,
not a statistically stable benchmark score.
