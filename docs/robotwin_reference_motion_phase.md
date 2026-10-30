# Fixed reference motion phases

`motion_segment` is a template-side selector for a fixed reference movement phase.
It does not infer contact, successful actuation, or a generated phase boundary.
The compiled constraint schema remains1.5 and OpenWAM's decoder/loss are unchanged.

```json
{
  "id": "reference_phase_progress",
  "type": "axis_displacement_interval",
  "subjects": "active_arms",
  "selector": {
    "kind": "motion_segment",
    "version": 1,
    "after": "first_close",
    "occurrence": "last",
    "evaluation": "endpoint"
  },
  "origin": "selection_start",
  "orientation": "world",
  "parameters": {"axis": "reference_segment_direction"},
  "allowed_margin": {"lower": 0.003, "upper": 0.003, "unit": "m"},
  "rationale": "Example development hypothesis; tolerances require validation."
}
```

An optional `parameters.local_point` uses the existing source-bound rigid-point
format. Segmentation always uses the reference EEF origin. The derived endpoint
axis uses the selected reference physical point, including its rotation-induced
translation. Those are distinct definitions and both are recorded in the audit.

Version1 first partitions the complete reference EEF XYZ trajectory, before
considering task geometry or automatic coverage. A step of at most10micrometers
is stationary. A moving segment ends at a stationary step or a step more than
55degrees from that segment's first step. The algorithm then chooses the last
nonstationary segment with at least two steps and at least0.1mm net EEF motion,
whose anchor is at or after the reference's first close. The anchor is the frame
before the segment's first action. A segment spanning that closure is not clipped
into a new segment. A later one-step/tiny segment does not qualify; its exclusion
is visible in the audit. These constants are fixed, versioned source policy, not
searchable template fields. `after=episode_start` and `after=last_close` are not
supported by this version.

For interval evaluation, active frames are the segment's action frames, excluding
the preceding anchor. Existing position, distance and axis-angle primitives may
use this interval. They still define one allowed region over the interval rather
than per-frame waypoints. Endpoint evaluation uses the last action frame.

Paired progress uses the original preceding anchor and the endpoint, with
`parameters.anchor="predicted_start"` in the compiled constraint. The reference
physical-point net displacement supplies a unit world axis and the bounds.
A net physical-point displacement below0.1mm is invalid; the compiler does not
select an earlier, more convenient phase. Paired progress requires endpoint
evaluation, a derived reference-segment axis and the selection's own anchor.

A paired span of more than32actions fails compilation, without the legacy
long-span conversion to a fixed reference anchor. A span of32or fewer actions
means both endpoints can occur in some chunks, not every chunk. Existing reader
selection and loss masks disable the paired term when either endpoint is absent
or invalid. If the anchor is the chunk's row0, it is the separately measured
initial state and the objective detaches it. Other chunks can contain two
predicted endpoints, which receive opposite progress gradients. Missing endpoints
never become zero-valued invented poses.

`axis_position_interval` with the same endpoint selector and derived axis is a
separate fixed-reference endpoint control. Its origin is the stored reference
anchor, so it can be evaluated without a generated anchor. It does not measure
generated start-to-end progress. These two allowed sets are generally different;
neither is asserted to contain the other.

The selector is required: `required=false`, a missing closure, no qualifying
segment, an invalid derived direction, or an unsupported paired span cannot
silently drop the feature. Development evaluation must record the invalid scene,
keep its parent coverage as a separate control, and never count an empty feature
set as acceptance. Adding a new phase region under independent `motion_coverage`
retains all previous ordinary-motion constraints. Default legacy templates are
unchanged; the new selector is opt-in.

Compilation records the full segment partition, reference boundary, selected
phase, decoded-reference hash, derived physical axis and compiled activation.
Locked HDF5 publication recomputes the template and stores this audit in both
coverage and annotation history. Reannotation of stored phase data requires an
explicit reference template. Training consumes fixed compiled labels and never
rediscovers phases from generated actions. The annotation policy fingerprint
includes the phase helper's source bytes and numeric policy.

The initial turn-switch development plan freezes20candidate templates on the
previously exposed138records, with old/new corpora, baseline/candidate, tracking
eligibility, command/actual and success/failure strata retained. This plan and its
results live under `artifacts/constraint_refinement/turn_reference_phase_v1` in
the shared workspace. CPU re-scoring is neither native tolerance validation nor
authorization to publish labels or begin training. Fixed reference time phases
can reject successful trajectories which execute the physical stroke earlier or
later; this limitation must be measured, not hidden by discarding tracking-invalid
native successes.
