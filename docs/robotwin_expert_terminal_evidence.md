# Read-only terminal evidence for the dual-shoe expert

This optional diagnostic records the final physical state and each component
of the original `place_dual_shoes.check_success` before cleanup. It does not
repair the expert, alter success, or authorize another episode. The existing
seed 10000 failure remains immutable and cannot be given a retrospective final
object state.

The frozen native task, base gripper helpers and robot getter sources are
required. Explicit `left_shoe`, `right_shoe` and `shoe_box` roles are bound to
unique rows of the complete physical snapshot. Two shoes sharing the same
name are distinguished by the named native fields and their exact measured
poses; ambiguous equal rows fail observation rather than guessing identity.
Raw position/quaternion dtypes are retained. Quaternion sign handling, strict
XY and quaternion-component limits, strict height limits relative to the box,
and nominal-gripper open comparisons match the original predicate. NumPy 1/2
scalar-promotion behavior for the height expression is frozen and recomputed.
The components diagnose the native metric, not actual contact or support.

`capture_reference(..., terminal_observer=None)` remains unchanged by default.
When enabled, the callback runs in `finally` before `close_env`, after the
original success result or a capture exception. It never calls `get_info`,
`play_once`, the predicate, robot controls, or `Scene.step`. Physical snapshots
before and after reading must be identical. The original predicate result is
retained; a separately calculated conjunction must agree when the predicate
was actually evaluated. If it was not evaluated, the components are explicitly
derived observations and are not a new native-success result.

Future StageA manifests use version 3 and bind `expert_terminal_policy`.
For the dual-shoe task this writes `expert_final_physical_state.json` and
`expert_terminal_evidence.json`. An observation error instead leaves an explicit
`expert_terminal_observation_error.json`, preserving any snapshot already
obtained and the primary native failure. The optional-hook exception does not
replace the original error. A successful expert with missing/invalid required
terminal evidence cannot start the conditional replay. All files, including
failed captures, are indexed; the verifier rechecks hashes, role binding,
numeric components and the original result. A version-3 manifest cannot be
downgraded to skip the requirement. Previous version-2 evidence is verified
with its original frozen verifier, not rewritten or upgraded.

CPU tests use original source AST methods and independently check success and
failure, strict boundaries, unchanged state, no extra predicate calls, capture
exception behavior, failure retention and rehashed artifact tampering. A
separate native-Python probe can use unattached SAPIEN bodies with no Scene,
assets, planner or GPU. Neither proves that observer wall-clock overhead has
zero effect on a future wall-budgeted planner. Any new native observation must
be a separately frozen finite plan with explicit authorization; the current
failure is not retried by this change.
