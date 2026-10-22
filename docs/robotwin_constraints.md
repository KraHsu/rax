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
