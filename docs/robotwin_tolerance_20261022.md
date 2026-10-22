# Empirical tolerance validation — 2026-10-22

Three clean `adjust_bottle` scenes passed 120/120 joint-intervention tests in
freshly initialized matching scenes. These tests included 16 corners repeated
twice and eight interior samples per scene. Every case passed native task
success, a further 0.5-second hold, and the measured-position tracking check.

| Joint test parameter | Requested range relative to each frozen expert program |
| --- | --- |
| Grasp X | −10 to +10 mm |
| Grasp Y | −5 to +5 mm |
| Grasp Z | 0 mm; no independent Z variation in this joint test |
| Gripper target fraction | 0 to 0.2 |
| Final EEF Z | 0 to +5 mm |

All other expert targets and orientations were fixed. This establishes successful
sampled joint configurations for these scenes; it does not prove that every
interior point succeeds or that the range transfers to new tasks, orientations,
objects, arms, dynamics or real robots. The 120 cases are not independent samples
of an unseen-task success probability.

## Execution and measurements

There were **514 logged replay/intervention trials** across seven environment
initializations: 76 exploratory trials, 285 expanded scans, and 153 fresh-scene
confirmation trials. Clean native seeds were 0, 1 and 2, all using the left arm;
native model IDs were 16, 13 and 13. No randomized evaluation scene was used.

The runner captured the native expert's target program once per scene. Grasp
translations shifted the approach and grasp targets together; subsequent lift
and placement targets stayed frozen. Opening changed the closure command.
Terminal height changed only the final EEF target. Native motion planning was
rerun for each intervention. The experiment therefore tests geometry under this
fixed waypoint program, not the exact timing of the original HDF5 action chunk
or a trained OpenWAM policy.

PhysX state, joint drives and gripper command state were restored between trials.
All seven runs passed three baseline replays before testing and one afterward.
The three initial baseline closure and endpoint positions had to match the reference within
2 mm. Fresh confirmations reused the previous program and checked object identity
and initial robot/object geometry before replaying it.

Measurements were sampled at the native recording cadence without writing camera
data or training HDF5. Results preserve closure-event position, end-of-grasp
position, endpoint position, vertical displacement and gripper command fraction.
The official gripper getter returns the commanded fraction, not measured finger
separation. Native task success here checks bottle functional-point lateral
position and height; it does not test upright orientation.

For changed grasp coordinates, the tracking threshold was max(1 mm, 10% of the
requested offset); unchanged coordinates used 2 mm. Every coordinate explicitly
included with a zero offset in the joint cases used 1 mm. Endpoint Z used
max(1 mm, 10%). All 120 joint cases passed; their largest observed coordinate
tracking error was **0.798 mm**. The full measured ranges are saved in the JSON
profile, separately from requested offsets.

## Why independent intervals are insufficient

- In the first scan, a −40 mm grasp-Z command still succeeded, but the measured
  closure position descended only about **0.85 mm**. Contact limited the motion;
  this is not evidence that a 40 mm downward geometric allowance is feasible.
- A combination drawn from the inner halves of independently successful axis
  offsets failed. Single-axis extrema cannot be multiplied into an accepted box.
- Fresh confirmation reproduced a failure at seed 0's +65 mm grasp-Y point,
  despite earlier successes. Seed 2's −75 mm grasp-X point succeeded in the task
  but failed the tracking criterion after fresh initialization.
- Different successful expert executions differed by about 8.6 mm in endpoint
  height relative to the originally annotated example. Its ±5 mm reference
  envelope was a prior, not an established task boundary.

The broader scans found the following **passing tested offsets before the first
observed nonpass**, after combining boundary repeats and fresh confirmations.
These are command-space search brackets, not certified continuous physical
intervals. A nonpass can be tracking failure or an inconsistent repeat rather
than task failure.

| Parameter | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| Grasp X, mm | [−65, +65] | [−80, +60] | [−70, +27.5] |
| Grasp Y, mm | [−30, +60] | [−60, +30] | [−55, +20] |
| Grasp Z | Positive tests through +37.5 mm; +40 inconsistent | No nonzero contiguous bracket passed the tracking criterion | No nonzero contiguous bracket passed the tracking criterion |
| Opening command upper tested point | 0.6 | 0.75 | 0.775 |
| Final EEF Z, mm | Positive tests through +240; upper failure not reached | [−2.5, +240]; upper failure not reached | [−27.5, +7.5] |

Negative grasp-Z tests generally saturated at contact. Isolated successful points
beyond a failed or untracked point are retained in the raw results, but do not
expand a contiguous bracket. The initial larger envelopes are exploratory;
the small joint domain in the first table has the stronger combination evidence.

## Data and training implications

The report and `joint_candidate_profile.json` are calibration evidence. The
profile explicitly prevents treating its requested-offset box as an automatic
replacement for measured-trajectory loss bounds. A label update must map measured
features to the specific recording, retain reference-only activation and state
the conditional scope. Event time and native sample count can change when native
planning changes trajectory duration. Existing annotation frame numbers cannot
be copied into a new trace blindly.

Training remains an offline action-decoding/geometric-loss computation. Simulation
was used only to establish experimental evidence. All five existing HDF5 paths,
sizes and modification times remained unchanged; the original expert HDF5 retained
SHA-256 `da923379b0fc60979a4b161d4c2314055ff6b643a821bf68537540bcc2b7783a`.

## Reproduction and artifacts

The native RoboTwin revision was `0008ae6800df9f75fc8de7098bacb01735fd8fd2` and the
manifest SHA-256 was
`4c4280329ba8f0d70437e90e3d9e11ba9baeebf1d961f59094909b22839576b8`.
Each run records driver/module hashes and the native task, base task and robot
source hashes. Initial exploration is retained as v1, the tracking/hold scans as
v2, and fresh confirmations separately; no earlier result was overwritten.

Relative to the server experiment work directory:

- `validation/tolerance_seed0_v1/`
- `validation/tolerance_seed{0,1,2}_v2/`
- `validation/tolerance_seed{0,1,2}_confirm/`
- `validation/confirmation_seed{0,1,2}.json`

Each run contains `run.json`, `expert_program.json`, `reference.json`,
`results.jsonl` and per-trial measured traces. Local reporting artifacts are under
`runtime/tolerance_study/analysis/`: `summary.json`, `trials.csv`,
`joint_candidate_profile.json`, and `grasp_z_tracking.png`/`.pdf`.

Ten unit tests passed for target isolation, measured event extraction, contact
saturation, repeated boundary disagreement, input validation, and fresh-scene
identity checks. Native scans and confirmations completed, and the reporting
plot was visually checked. See [workflow commands](robotwin_constraints.md#empirical-tolerance-validation).
