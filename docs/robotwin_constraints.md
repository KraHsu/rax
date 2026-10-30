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

The native Codex process inherits the operator's `HTTP_PROXY`, `HTTPS_PROXY`,
`ALL_PROXY` and `NO_PROXY` environment variables (including lowercase variants).
Set these in the shell launching RAX when the model endpoint requires a proxy.
RAX's separate sandbox network policy still applies to agent tool subprocesses.
If model discovery or workspace routing repeatedly times out, check the launch
environment's proxy configuration before retrying; the local robot MCP socket
and the remote SSH connection are separate from model API connectivity.

Local commands must be allowed to create Unix sockets and bubblewrap namespaces.
An enclosing coding-agent command sandbox can prohibit these even when the normal
terminal supports them. The launcher reports this during preflight; it does not
disable RAX isolation. `--prepare-only` prepares metadata/configuration without
starting Codex or an episode. Five formal scenes are a connectivity/policy pilot,
not a statistically stable benchmark score.

On hosts where cold SAPIEN/cuRobo initialization exceeds five minutes, the
operator remote JSON may set `startup_timeout_seconds` (1..1800; default 300).
This applies to process readiness and reset only. After reset, robot RPCs return
to their 300-second timeout. No failed action is automatically retried.

## Native expert data with offline RAX annotation

When autonomous RAX control is the bottleneck, use RoboTwin's own successful
expert plan and replay recording. This keeps its native `save_freq` sampling
and `merge_pkl_to_hdf5_video` writer. It is separate from the sparse observations
recorded at RAX command boundaries. Collection accepts **only clean** manifest
entries; randomized scenes remain evaluation-only.

On the existing simulator host (set `ROBOTWIN_PATH` and select a free GPU):

```bash
python scripts/collect_robotwin_expert.py --task adjust_bottle \
  --manifest /work/manifests/adjust_bottle_pilot.json --indices 0 \
  --output /work/datasets/robotwin_expert/adjust_bottle/aloha-agilex_clean_50
python scripts/robotwin_annotations.py export \
  --episode /work/datasets/robotwin_expert/adjust_bottle/aloha-agilex_clean_50/data/episode0.hdf5 \
  --output /work/reviews/adjust_bottle_0
```

The collector validates both expert planning and recording replay before
publishing the episode. Failed staging data stays outside `data/`. An empty
embedded constraint bundle initially records provenance; it supplies no training
constraint until accepted annotations are added.

Copy just the review directory to the operator machine. It contains measured
world XYZ + WXYZ, gripper openings, native frame indices, threshold events, and
selected RGB previews. It contains no simulator object poses or task code. Run:

```bash
python scripts/run_robotwin_annotation.py --review runtime/reviews/adjust_bottle_0 \
  --output runtime/annotations/adjust_bottle_0
```

This reuses RAX's native Codex environment and sandbox with an offline task and
the geometric constraint skill. The agent writes `agent-workspace/annotations.json`
and `REPORT.md`; it has no live robot MCP. Copy the proposal back to the data host:

```bash
python scripts/robotwin_annotations.py embed --episode /work/datasets/robotwin_expert/adjust_bottle/aloha-agilex_clean_50/data/episode0.hdf5 \
  --proposal /work/reviews/adjust_bottle_0/annotations.json --agent-run adjust_bottle_0
```

Embedding checks the source SHA-256, schema, frame ranges, real reference gripper
crossings, and whether each accepted interval admits the source trajectory. It
uses a per-episode lock and atomic replacement, preserving revision history in
the same HDF5. A failed check leaves the episode unchanged. The same embedded
schema is consumed by OpenWAM without another label file.

Initial annotation priors are 5 mm position margin, 5 degrees angular margin,
and 0.02 opening fraction. They define reference envelopes, not measured physical
tolerance or a verified grasp volume. The agent must explain their use and mark
unsupported geometric claims proposed/unknown, which training ignores. Native
expert success, annotation validity, and C2R policy success are different metrics.

The [2026-10-22 pilot](robotwin_pilot_20261022.md) records native expert collection,
local annotation, an OpenWAM loss/backward audit, and five randomized controller
attempts, with infrastructure errors reported separately.

## Empirical tolerance validation

Use the [diagnosis and corrected calibration workflow](robotwin_tolerance_diagnosis_20261022.md)
when interpreting scans. Successful measured points remain useful when a command
has tracking bias; they validate the observed point, not the unexecuted request.
Grasp calibration must identify its continuation. The diagnostic runner compares
frozen, native adaptive, and object-position-compensated continuations from the
same grasp checkpoint; no simulation is added to the training loss.

On the existing native GPU environment, scan clean expert targets without writing
training HDF5:

```bash
python scripts/validate_robotwin_tolerances.py \
  --manifest /work/manifests/adjust_bottle_pilot.json --index 0 \
  --output /work/validation/seed0
```

The current runner is task-specific to `adjust_bottle`. It captures the native
expert target program, restores a PhysX snapshot plus joint drive targets and
gripper command state between trials, and requires three successful baseline
replays with <=2 mm feature drift. It scans requested offsets, checks achieved
geometry, refines observed brackets, tests combinations, repeats boundary points,
and checks another baseline afterward. A 0.5-second hold supplements the native
end-of-program task verdict. Planning, task and tracking failures are distinct.

Use `--program-from /work/validation/seed0 --cases-json cases.json` to test an
existing frozen program after fresh scene initialization. The scene identity and
initial geometry must match. `cases.json` is a list of dictionaries with optional
`grasp_x`, `grasp_y`, `grasp_z`, `terminal_z` (world metres) and `opening` (command
fraction). `--baseline-only` checks the reset/replay procedure alone. Outputs must
be new directories. The runner stores measured JSON traces and results; it does
not update the original training annotations.

After copying only small result/metadata files to the development machine:

```bash
python scripts/summarize_robotwin_tolerances.py --root runtime/tolerance_study/server \
  --output runtime/tolerance_study/analysis --plot
```

`--plot` needs Matplotlib only on the reporting machine. See the
[514-trial validation report](robotwin_tolerance_20261022.md) for the measured
limits, joint confirmation, and the distinction between command and action-data
coordinates.

## Counterfactual template acceptance, version 2

Successful references fitting their own labels are necessary but do not show
that labels reject failure. New registry activation therefore uses
`services.robotwin.acceptance.assess_evidence`, which is CPU-only. Native
counterfactual collection still requires the RoboTwin GPU environment. Training
continues to use only offline actions and annotations, with no simulator in loss.

Every template hash used by a new dataset needs its own evidence report. The
pilot protocol requires at least three held-out clean seeds and scenes, two
successful constraint-admitted baseline replays per seed, three successful
nonbaseline variations, and six actual native task failures. Successes and
failures must each cover three seeds. Position, orientation, gripper timing and
coupled interventions must have completed realized trials; failures must span
at least two families. Failed variations must all be rejected, and no more than
20% of successful variations may be rejected. A filter that rejects everything
cannot pass. These small selected samples are a pilot gate, not an IID success
estimate or a continuous feasible-region guarantee.

The runner freezes `experiment_manifest.json` before trials and writes
`trials.jsonl`. Context includes the exact template, annotation implementation
hash, native revision, explicit training seed/scene exclusions, and all planned
trial IDs. Raw native outcomes and reference/actual traces live in hashed JSON
artifacts. The CPU gate reconstructs the reference labels from the sealed
template and computes violations itself. It never trusts a supplied
`constraint_pass` flag, uses reference events for activation, and uses reference
motion for window/fallback selection. EEF poses are measured, but the recorded
gripper values are nominal commands, not contact forces or grasp measurements.
Native success is independently recorded as ever satisfying the native predicate
during fixed-horizon replay. Baseline replay must establish compatibility with
that protocol before perturbations count.

Missing frames, changed artifacts, missing planned trials, unsupported templates,
failed baselines, and random/training leakage block the gate. Planner failures,
incomplete execution and unrealized interventions do not become task-failure
negative examples. Validation bundles are tagged `validation/clean_heldout` and
are rejected by the training reader. No random benchmark recordings enter the
new training dataset.

Attach evidence to a separate registry, leaving the original dataset untouched:

```python
from services.robotwin.annotation_templates import attach_evidence, registry_readiness

report = attach_evidence(new_registry_root, task, template_sha256, evidence_root,
                         reference_root=old_training_dataset)
ready = registry_readiness(new_registry_root, {task: [template_sha256]})
```

`reference_root` independently checks that the evidence exclusion list contains
all source training seeds/scenes. `registry_readiness` revalidates raw artifacts
by default. `match` uses only verified version-2 templates unless the caller
explicitly requests a read-only legacy lookup. Old `active` status alone is not
accepted. With `annotation_control.json` containing `acceptance_version: 2`,
manual publication also requires an accepted template and exactly its generated
constraints; bootstrap reviews remain in `raw/` and `reviews/`. The new dataset
publisher additionally records the template/report identities for each episode
and must check all templates it actually used before starting training.

Create the new dataset only after the per-template reports pass:

```bash
python scripts/publish_robotwin_constraint_dataset.py \
  --source-root OLD_DATASET --registry-root NEW_REGISTRY --output NEW_DATASET
python scripts/publish_robotwin_constraint_dataset.py --validate-only --output NEW_DATASET
```

The publisher copies HDF5 files to different inodes in a hidden staging directory,
then publishes atomically after all 2,500 episodes and their proofs validate.
Resume checks source, template and evidence hashes. The proof is stored separately
at `trajectory_constraints/acceptance`, preserving the strict geometric bundle
schema. Raw evidence artifacts must remain readable for publication and training
validation; copying only a boolean report is insufficient.

Raw artifact and receipt hashes are exact. Recomputed floating-point geometry and
tracking summaries use only `rtol=1e-10, atol=1e-12` for cross-Python/NumPy
consistency; keys, booleans, discrete indices and pass/fail decisions remain
exact. This numerical comparison does not widen any physical acceptance bound.


### Native arm-base frame correction for acceptance replay

The installed RoboTwin cuRobo adapter subtracts the fixed arm-base translation
but omits its URDF rotation. For the aloha-agilex asset, the left/right fixed
base yaw is 0.02/0.01 rad. This produced successful baseline replays with
7.9–11.4 mm measured EEF errors, which correctly failed the 5 mm realization
gate. A correction using the complete URDF SE(3), without fitted offsets,
reduced those three first-seed baseline maxima to 2.1–3.1 mm while preserving
native success. The physical robot asset, controller, task predicates and
acceptance thresholds are unchanged.

Acceptance installs the `urdf_fixed_arm_base_se3_v1` private transform wrapper
on each trial's planner instances. Native expert reference capture remains unchanged. The wrapper
applies the inverse fixed arm-base rotation after the world-to-root transform
and before the native translation subtraction, accounting for that subtraction
exactly once. Both position and WXYZ orientation are transformed; measured
actual trajectories and constraint bounds are never compensated.

The adapter verifies the physics and planner URDF identity, direct fixed-base
chain, end-effector link, planner configuration and translation convention.
Unsupported planner/robot configurations fail explicitly. The manifest freezes
an asset/adapter audit; references set `replay_adapter_applied: false`, while
completed trial outcomes set it to `true`. Both retain identical URDF/YAML hashes. CPU reassessment verifies their consistency and
matrix semantics without importing a simulator. Source fingerprints include the
adapter code. Earlier failed attempts retain their original code and evidence;
corrected attempts use a separate physical artifact directory.

Future policy comparisons must use and report the same controller/frame adapter
for both baseline and candidate models. A partial score collected under the
legacy adapter cannot establish the effect of constraint supervision by itself.
