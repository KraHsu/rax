# Rigid local points in geometric constraints (schema 1.5)

A position constraint can supervise a fixed marker attached to the decoded end
effector, rather than only its origin. For physical world pose `(p,R)` and a
constant EEF-local vector `r` in metres, the marker is `x = p + R r`.
This adds position–orientation coupling without adding a task-specific loss.
It does not infer object motion, contact, a fingertip surface, or task success.

Only `axis_position_interval`, `radial_distance_interval`,
`point_distance_interval` and `axis_displacement_interval` accept the optional
`parameters.local_point`. Other primitives reject it. All existing axes, bounds,
penalty scales and weights retain their meanings, with `x` replacing `p`:

| Primitive | Feature |
| --- | --- |
| Axial position | `a · (x-o)` |
| Radial distance | `norm((x-o) - a (a · (x-o)))` |
| Point distance | `norm(x-o)` |
| Axial displacement, predicted start | `a · (x_k-x_start)` |
| Axial displacement, fixed reference | `a · (x-o)` |

Here `a = Q n`, with `Q` the annotation's fixed WXYZ reference orientation.
Interval violation remains `relu(lower-u) + relu(u-upper)`, followed by the
existing scaled squared/quartic penalty. Gradients flow through both translation
and the physical decoded rotation. The reference parameters and activation are
constants; no simulated quantity participates in training.

An illustrative parameter object (replace the source hash and geometry before
using it in real data):

```json
{
  "axis": [0, 1, 0],
  "bounds": [-0.01, 0.01],
  "local_point": {
    "frame": "subject_eef",
    "offset_m": [0.08, 0, 0],
    "provenance": {
      "source": "known_geometry",
      "reference": "synthetic example geometry document",
      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "rationale": "Rigid marker measured from the decoded EEF origin; no contact claim"
    }
  }
}
```

`frame` is exactly `subject_eef`: the decoded **physical** EEF frame, not the
checkpoint's raw rotation convention, a planner base, the annotation frame, or
an object frame. `offset_m` is a finite three-vector in metres. Provenance is
mandatory, with `source` equal to `known_geometry` or `manual`, a nonempty source
reference and rationale, and a lowercase source SHA256. The hash is portable
source identity; validation does not fetch files or certify their physical claim.
An agent must not infer a rigid offset from a task name or tune it as a free
per-episode transform to accept failures.

Programmatic rules use the identical object inside `parameters`, without
`bounds`. `origin=selection_start/selection_end` compiles the reference marker
position at the existing reference-selected index; orientation and selectors
are unchanged. A `position_extremum` selector still chooses the recorded EEF
origin's extremum, not the marker's extremum. Fixed activation never moves in
response to a prediction. Long-span displacement publication reanchors to the
reference **marker** at the original start, preserving reference feature values.

Missing `local_point`, or the explicit offset `[0,0,0]`, uses exactly the old
position-only path and masks. A nonzero offset requires all three position and
all six rotation dimensions valid at every evaluated point and, for predicted
start displacement, at its original anchor. Missing dimensions mask the point;
valid but nonfinite/degenerate rotations fail rather than becoming identity.
Measured reference quaternions must pass the existing unit-norm tolerance;
the new nonzero-marker path then normalizes their floating-point roundoff to
match the reader's physical rotation. This does not change legacy angle features.
The original start must be visible in the chunk. It may be row zero's measured
state: the training objective detaches it, and row zero itself is never a generated
action loss sample. There is no new requirement that both endpoints be generated.
A fixed reference anchor remains usable outside the original reference frame's
chunk, with its existing meaning.

Schema 1.0–1.4 stays readable with unchanged geometry. Writing any `local_point`
requires schema 1.5. The schema copies in RAX and OpenWAM are byte-identical.
RAX reference compilation, native trajectory acceptance, HDF5 serialization,
OpenWAM inverse normalization/unified indexing/legacy quaternion correction,
and Torch loss all use this definition. The viewer plots the exact marker
feature and spatial marker, and suppresses misleading marker bounds over raw
EEF XYZ curves. Temporal coverage remains a temporal property, not a geometric
rank or success guarantee.

This feature does not activate a template, rewrite an existing dataset, change
controller targets, or establish a feasible offset/tolerance for `turn_switch`.
Those require a separately frozen candidate and independent native evidence.
