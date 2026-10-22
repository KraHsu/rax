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
| gripper_interval | absolute opening, 0=closed and 1=open | bounds in [0,1] |

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

On successful episode finalization, observations and constraints are published
in one HDF5 under the training dataset. Failed/interrupted recordings are kept
outside data/. Formal evaluation cannot modify or generate training labels.
