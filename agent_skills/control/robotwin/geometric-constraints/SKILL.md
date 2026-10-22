---
name: geometric-constraints
description: Annotate recorded RoboTwin data with composable geometric intervals and reference-only activation.
---

Use `robotwin_constraints` while recording clean training episodes. There are
no operation categories. Every annotation is a base geometric constraint;
multiple annotations mean conjunction (all should hold), not alternative modes.

1. Read `robotwin_observe` and its `recorded_frame`. Frame zero is the initial
   observation; one native command adds one observed frame. Never confuse
   recorded frames with physics ticks, video frame rate or a future WAM index.
2. Ground a fixed reference frame, unit axes, allowed bounds and their units.
   Record the source and rationale. One demonstration cannot identify the full
   feasible volume. Use `proposed`/`unknown` if bounds are not justified. Never
   invent an object's support/contact/grasp boundary from its category name.
3. Fill a constraint using `references/template.json`. Replace every example
   number and evidence string using the current recording. The example's
   proposed status is deliberate; it is not a verified label.
4. Choose activation: `interval` [start,end), `endpoint` end-1, or
   `reference_event` at a labeled gripper threshold crossing. The crossing is
   checked on the reference gripper, regardless of generated gripper behavior.
   If closure itself matters, add a separate `gripper_interval` constraint.
5. Read `robotwin_constraints(operation="list")`, then upsert with the returned
   `expected_revision`. Reuse the same id to revise a constraint; create new ids
   to add independent constraints. Revisions are archived inside the HDF5.
6. Before finishing, verify every end index is within `recorded_frames`, each
   reference event really crosses the stated threshold, and the bounds are
   consistent. Use accepted only when the reference annotation is justified;
   accepted means a valid annotation, not proven physical success.

Base types and parameter units:

| type | scalar | parameters |
| --- | --- | --- |
| axis_position_interval | position projected along an axis relative to frame origin | axis, bounds in m |
| radial_distance_interval | distance to an axis line | axis, nonnegative bounds in m |
| point_distance_interval | distance to frame origin | nonnegative bounds in m |
| axis_angle_interval | angle between EEF local axis and reference direction | local_axis, reference_axis, bounds in rad |
| axis_displacement_interval | projected displacement from activation.start | axis, bounds in m |
| angular_displacement_interval | rotation angle from activation.start | bounds in rad |
| gripper_interval | absolute gripper command fraction, 0=closed and 1=open; not measured finger distance | bounds in [0,1] |

`reference_frame` is a fixed transform into world coordinates: origin in metres
and quaternion WXYZ. Its axes are converted to world directions. A moving object
frame cannot be supplied unless the data adapter provides its actual trajectory;
this version does not infer it from robot actions.

`penalty.scale` uses the same units as the scalar. `squared_hinge` is the default;
`quartic_hinge` gives faster growth outside the interval. Both are zero inside.
Weights combine constraints, not per-frame imitation targets. Axis-angle bounds
allow free rotation about the aligned axis. A radial interval plus axial bounds
can describe a cylinder; three independent axis intervals describe a box.

Do not add arbitrary executable loss code or unrecognized base types. Existing
types and their compositions can be parameterized and revised during recording.
A new mathematical primitive requires a versioned implementation and tests.

After the final movement ends the episode, you may still review/upsert annotations
before calling `finish`; further physics is blocked. Use the final recorded frame
count to check activation ranges and reference events.

On successful episode finalization, observations and constraints are published
in one HDF5 under the training dataset. Failed/interrupted recordings are kept
outside data/. Formal evaluation cannot modify or generate training labels.

When an operator supplies a simulation calibration report, check its scene,
object/embodiment, reference pose, coordinate system, fixed orientation and
frozen-program identity before using it. Keep requested command offsets separate
from measured trajectory features: a successful command can be blocked by contact
and never reach the requested pose. Exclude such trials from geometric boundary
claims about the REQUESTED pose. A successful trial still supplies its MEASURED
pose as a feasible sample; do not discard that point because tracking is biased.
Use `measured_feasible_samples` from the calibration report. A blocked -40 mm
request that achieves -1 mm supplies only the -1 mm point. Distinguish planning
failure, contact-limited tracking and executed task failure.

Use joint-intervention evidence when composing axis bounds; individually
successful extrema do not establish that their Cartesian product succeeds.
Record repeated failures, disagreement between repeats, and search limits without
observed failure. Finite successful samples are conditional empirical evidence,
not proof of every interior point or a global feasible set. Preserve the native
task verdict separately from additional stability/hold checks. Convert measured
features into the recording's reference-only activation; never replace physical
loss bounds directly with a requested-command-offset box. Gripper command
fractions do not identify finger distance, force or contact.

Calibration must state what happens after the perturbed pose. A frozen subsequent
EEF target tests the whole frozen program. A native expert continuation recomputed
from the actual grasp tests the grasp with that continuation. Never mix the two
into one feasible set, or combine a broad grasp allowance from adaptive trials
with a narrow fixed endpoint from a different grasp. Record the continuation in
the evidence rationale/report; the training loss still uses action geometry only.
For a disputed coupled failure, branch both continuations from the same saved
post-grasp physics state and compare object/EEF relative geometry. Use the
calibration checklist in `references/calibration-template.json` before proposing
updated bounds. Extrema of successful points do not validate their box corners.
