# Reusable task template, version 1

The task template is declarative JSON, not Python. It selects task-relevant
geometric features and computes new bounds independently on every reference trajectory.
There are no operation classes. It must NOT contain frame numbers, measured
coordinates, copied bounds, or inferred object contact/feasibility geometry.

Write `task_template.json` with exactly:

```json
{
  "version": 1,
  "task": "COPY input/review.json task",
  "rationale": "Explain which task-sensitive reference features are retained.",
  "signatures": [["close"]],
  "rules": [
    {
      "id": "closure_location",
      "type": "point_distance_interval",
      "subjects": "active_arms",
      "selector": {"kind": "events", "direction": "decreasing", "occurrence": "all"},
      "origin": "selection_start",
      "orientation": "world",
      "parameters": {},
      "rationale": "The measured EEF location at reference closure; not the object grasp boundary."
    }
  ]
}
```

This example is a format illustration. Choose rules for the actual demonstrated
task; closure location can be irrelevant when a gripper closes in free space
before pressing. Never label that point a contact point.

`signatures` must initially be `[review.event_signature]`, copied exactly. Each
signature is a sorted list containing the event sequence of each active arm.
`["close"]` permits the same pattern on left OR right; `["close", "close,open"]`
describes two active arms. `""` means an active arm with no crossing. Unseen
patterns go back to RAX. Do not invent additional supported signatures.

Every rule requires `id`, `type`, `subjects`, `selector`, `origin`, `orientation`,
`parameters`, `rationale`. Optional `required` defaults true. `subjects` is
`active_arms`, `left`, or `right`; prefer `active_arms` for symmetric arm variants.
Only use a fixed arm if supported by the task. A missing required selector fails
the template; optional rules may have no matches.

Selectors and their EXACT fields:

| selector | Fields | Activation |
| --- | --- | --- |
| events | `kind`, `direction`: decreasing/increasing/any, `occurrence`: first/last/all | genuine reference 0.5 crossing |
| gripper_plateaus | `kind`, `state`: closed/open, `after`: episode_start/first_close/last_close | contiguous command=0/1 runs (atol 1e-6), at least 2 frames |
| terminal | `kind` only | last reference frame |
| position_extremum | `kind`, `axis`: unit world vector, `which`: min/max, `after`: episode_start/first_close/last_close | first extremum frame in the selected suffix |

`origin`: world (zero), selection_start, selection_end. For an event both measured
origins use the event frame, NOT its preceding sample. Other selectors use the
selected range endpoints. `orientation`: world (identity WXYZ), selection_start
(measured WXYZ at the selected frame). These frames are fixed for each generated
constraint, never computed from predicted actions.

`parameters` contains only the fields below; NEVER supply `bounds`:

| type | parameters |
| --- | --- |
| axis_position_interval | axis: unit vector in the selected reference frame |
| radial_distance_interval | axis: unit vector in the selected reference frame |
| point_distance_interval | empty object |
| axis_angle_interval | local_axis, reference_axis: unit vectors |
| axis_displacement_interval | axis: unit vector |
| angular_displacement_interval | empty object |
| gripper_interval | empty object |

Displacement rules add `start_anchor`:
`selection` (default), `first_close`, `last_close`, `episode_start`. Non-default
Point/event displacement rules require an explicit earlier anchor (not
`selection`); a zero-length displacement is rejected. Non-default
anchors work only with point/event selectors, never a gripper interval. The
original start anchor must be present in a training chunk; the loss masks it
otherwise. Include a justified local endpoint feature when appropriate.

The executor measures min/max over exactly the selected activation, then applies
the established operator priors: position ±0.005 m, angle ±0.0872664626 rad,
gripper ±0.02; distances/angles/gripper are clipped to their valid domains.
Penalty is squared hinge with that margin as scale. These values are reference
envelopes, NOT physical tolerance calibration. You cannot widen these priors in
the template. Temporal coverage MUST include every action frame of both arms.
This does not mean copying the demonstrated path: the executor automatically
fills remaining position/motion gaps with `motion_direction_cone` for movement
and a loose stationary hold region. Moving intervals allow a 60-degree half-angle
around the measured segment direction, without absolute position/speed targets.
Direction is split only when the reference turns outside the cone or stops.
This coverage layer uses adjacent positions, so every action chunk can evaluate
it from its own initial state. It is independent of old displacement anchors.

The versioned defaults are `full_trajectory.POLICY`: direction slack 0.1 mm,
stationary-step threshold 0.01 mm, and hold-region margin 20 mm. These are
explicit starting priors, not calibrated physical success bounds. Existing
accepted task-specific position regions take priority over coverage defaults.
Orientation and gripper are constrained only where justified; full temporal
coverage does not mean all degrees of freedom must be restricted.

Run `./validate_task_template.py task_template.json` until it passes. This uses
the same numeric implementation as the H200 executor on the review CSV. Output
includes the generated constraints, reference audit, and full-coverage report. Inspect the
selected frames and physical meanings as well; admitting the demonstration is
necessary but does not prove semantic correctness or physical robustness.

The server will additionally require three distinct clean seeds and initial
geometries before activating a template. A new event pattern, missing anchor,
or validation failure is routed back for review.
