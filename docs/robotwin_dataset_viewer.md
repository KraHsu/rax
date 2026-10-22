# RoboTwin Dataset Atlas

A read-only website for the live `robotwin_clean50_v1` collection. It reads the actual
HDF5 files on H200; only selected frame JPEGs, episode metadata, and requested MP4
segments travel to the local browser. It does not copy the dataset, invoke Codex,
launch simulation, or change collection/annotation state.

## Open / restart

From the RAX checkout on the local development machine:

```bash
python3 scripts/run_robotwin_dataset_viewer.py
```

Open **http://127.0.0.1:8766**. The command synchronizes only the viewer source,
starts/reuses the remote CPU service, and starts a detached SSH tunnel that
reconnects after connection failures. No Python package installation is needed
locally. SSH key access and `rsync` are required; the existing remote RoboTwin
environment provides NumPy, h5py, and Pillow.

Defaults match the experiment server:

- SSH: `root@vr.turbo-ai.com:20379`
- Remote repository: `/mnt/data/liangjiaqi/rax_train/rax`
- Dataset: `/mnt/data/liangjiaqi/rax_train/datasets/robotwin_clean50_v1`
- Remote Python: `/mnt/data/liangjiaqi/yzm/miniconda3/envs/RoboTwin/bin/python`
- Remote service: `127.0.0.1:9020`; local forwarded address: `127.0.0.1:8766`.

These are configurable with `--host`, `--ssh-port`, `--remote-repo`, `--root`,
`--remote-python`, `--remote-port`, and `--local-port`. The remote repository must
already contain the matching `services.robotwin` modules. Both HTTP listeners
are loopback only; this is a private local website, not a public deployment.

## Browse

- **Task library:** search Chinese/English task names, filter collecting/completed
  tasks, and see actual recorded/annotated counts. Progress refreshes every 15 seconds.
- **Trajectory review:** select an episode and seed, step through synchronized
  head/front and wrist images, play, scrub, and click a curve or gripper event to
  locate a frame. Native MP4 playback supports byte ranges and seeking.
- **Action inspection:** absolute world EEF XYZ, gripper command fractions,
  left/right/both-arm curves, and projected EEF trajectories. The projection is a
  visualization of measured positions, not simulated motion.
- **Constraint overlays:** enabled by default in **动作与约束**. Click a constraint's
  activation lane or the inspector to select it and seek to its first active frame.
  The lane and action chart share a frame axis. **随帧切换约束** follows active
  constraints during playback; disable it to pin the selected constraint. The
  action selector offers XYZ, gripper, and **所选约束量**. Axis-aligned position and
  gripper bounds appear directly on their curves, with event/endpoint bounds only
  on the actual activation frame. Arbitrary oblique axes and angles use their
  scalar physical feature rather than independent XYZ or Euler bounds.
- **Spatial geometry:** **空间约束** displays the selected active sphere/shell,
  axial slab, cylindrical shell, motion cone, or axis-angle cone. Switch between
  local constraint detail and the full trajectory, and between XYZ/XY/XZ
  projections. A motion cone starts at the previous measured EEF position and
  indicates direction only: the displayed length is illustrative, not a target or
  step-size limit. Its wireframe shows the cone body; the near-apex slack is
  represented precisely by the scalar bounds. Gold arrows indicate measured
  movement/current orientation; dashed orientation axes retain the original
  angular-displacement anchor. Slabs/cylinders are unbounded in their free directions.
- **Constraint inspection:** each composable primitive's allowed interval,
  activation frames, reference feature, bounds check, penalty, evidence, and fixed
  coordinate frame. Displacement plots preserve the annotated original anchor.
  Feature plots show active samples only, so large hypothetical values outside the
  supervised interval cannot hide a small allowed band. Angles are displayed in
  degrees and direction-cone violation in millimetres; stored values and exported
  annotations keep their original units. Active violations are marked red. Sphere
  XYZ bands are outer projections, explicitly labeled as such: they do not make
  three independent intervals a sufficient feasibility test. Multiple constraints
  can overlap; the overlay shows one selected constraint at a time.
- **Coverage:** position/motion coverage for every arm/action frame, with gaps
  highlighted. Orientation and gripper coverage are separate; full temporal
  coverage does not fix all degrees of freedom. Ordinary movement constrains
  direction rather than per-frame XYZ targets. New annotation revisions refresh automatically.
- **Initial-state comparison:** paginated frame-zero thumbnails, seed, and scene
  fingerprint; click an image to open that episode. Object initial XYZ/WXYZ poses
  are available under provenance. This exposes variation without equating distinct
  hashes with a quantitative diversity guarantee.
- **Export JSON:** export metadata and constraints for the selected episode.
  The export includes viewer audit fields and is not a replacement for the
  canonical embedded constraint bundle.

Reference bounds passing does **not** establish physical feasibility. RAX-reviewed
templates and explicit tolerances remain visible as annotation provenance. Gripper
values are absolute open-fraction commands, not measured contact or finger distance.
The playback rate is recorded frames/second, not the simulator's physical time.

## Operations

- Local tunnel log/PID: `~/.cache/rax-dataset-viewer/tunnel.log`, `tunnel.pid`.
- Remote service log/PID: `<remote-repo>/runtime/dataset_viewer/server.log`, `server.pid`.
- The tunnel uses a dedicated SSH connection; existing collection and annotation
  control sockets are untouched. It remains running after the launch command exits.
- To stop, terminate the PID in `tunnel.pid`, then its dedicated `ssh -N -L`
  child if still running. Stop the remote service using its `server.pid` only.
  Confirm command lines before terminating processes; do not kill collection workers.
- Static asset changes are served on the next page reload. Python service changes
  require restarting its PID and rerunning the launcher.

Direct server mode, useful when already on the dataset machine:

```bash
python -m services.dataset_viewer.server --root /path/to/robotwin_clean50_v1 --port 9020
```

## Validation

```bash
../.venv/bin/python -m pytest local_tests/unit/test_dataset_viewer.py -q
node --check services/dataset_viewer/static/app.js
node --test local_tests/unit/test_dataset_viewer_frontend.cjs
```

Tests cover the original seven feature types across all three activation modes, nonzero
anchors and rotated frames, publication transitions, distinct scene metadata,
native JPEG channel correction, video byte ranges, path confinement, and artifact
immutability. The service exposes GET-only endpoints under `/api/health`,
`/api/summary`, `/api/episodes`, `/api/episode`, `/api/frame`, and `/api/video`.
Direction cones and short-chunk coverage have additional tests in
`test_robotwin_full_coverage.py` and the OpenWAM loss tests.
The Node tests exercise physical overlay geometry, rotated/signed frames, original
anchors, activation gaps, signal/arm following, pinning, and overlay toggling with
a minimal DOM. They are not a substitute for browser layout verification.
