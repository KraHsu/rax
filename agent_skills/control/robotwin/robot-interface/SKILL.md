---
name: robot-interface
description: Control native RoboTwin with measured observations and absolute targets.
---

Use `robotwin_status` and `robotwin_observe`. Images are head_camera, left_camera,
right_camera. Proprioception includes world EEF poses XYZ+WXYZ, gripper opening
(0 closed, 1 open), and joint state. No object poses or camera calibration are
exposed. Refer to [object localization](../object-localization/SKILL.md).

`robotwin_step_eef(actions=[...])` takes 1..32 rows, each
`[L xyz3, L wxyz4, L grip, R xyz3, R wxyz4, R grip]`. Quaternions must be unit.
Targets are absolute, in metres, world coordinates. `robotwin_step` takes native
absolute joint-target vectors. One row is one native planner command with
variable physical duration; it is not a fixed-rate dynamics tick.

Observe after motion. Planning can fail even when a command is well formed.
Record constraints using [geometric constraints](../geometric-constraints/SKILL.md)
and the returned recorded_frame indices. Success comes from the native task
evaluator, not from passing individual geometry constraints.
