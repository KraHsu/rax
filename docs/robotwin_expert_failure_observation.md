# Fixed failed-expert observations

The new entrypoint observes exactly one expert initialization and zero replay episodes for place_dual_shoes seed10000 or put_object_cabinet seed10050. It is a new declared observational reproduction, never a replacement of the earlier failure. Existing output/owner intent cannot resume. Failed capture and original exceptions remain indexed. A complete observation can contain an expert failure; completion does not mean task success or acceptance.

Dual shoes uses the unchanged source-bound terminal hook from75744025. The related StageA history is5 initializations (3dump,2shoes); one extra shoe observation raises that group to at most6. Cabinet has one earlier failed smoke initialization and may add only one. No seed substitution, automatic retry, training or publication exists.

Cabinet observations use the original robot planner entrypoints and original Base execution methods via current-thread sys.setprofile. Parameters already present on CPU, original return status, call scopes and the first observed true-to-false plan_success transition are recorded separately. A non-Success request is not itself declared a fatal task failure. Python return events can also mean exception unwind; None has no failure-status interpretation. The source version defines a check_pose helper inside choose_grasp_pose but never calls it. No extra planner calls, RNG resets, qpos getters or GPU transfers occur in the profile callback. Other processes/solver internals are outside scope.

Cabinet terminal evidence preserves original native_success=None when the original predicate was skipped. It stores object/cabinet state, fresh origin_z dtype and arm role, the source-bound cabinet functional point configuration and observed link transform. Components are recomputed from those operands without invoking get_info, check_success, control or scene.step. Before/after raw physical states and link poses must match. Asset/local transform and world target are distinct and indexed. Numeric promotion is bound to the native NumPy major version.

Source hashes, old failed records, training exclusions, exposed development seed and production numeric context are preflight inputs. Execution additionally requires a separately frozen owner with a shared GPU lease, foreign-process refusal, bounded wait/execution and inherited guardian lease. Default/check-only never initializes a native scene. Only a reviewed exact package may be launched by the parent agent.

The observation adds wall-clock CPU work and is not a native instrumented/uninstrumented A/B equivalence proof. Ordinary observer errors are separate from original capture errors. Existing legacy BaseException/cleanup behavior is retained; no claim of unconditional cleanup under process kill is made.

## Future-only planner status codec, version 2

The first cabinet observation failed its final integrity check: five batch planner returns used object-dtype status arrays. Version1 retained their derived labels but omitted the original string contents. Those arrays cannot be recovered from the frozen evidence. The old failed output, index and original `planner_success=False` / `native_success=None` remain unchanged.

Version2 permits object arrays only in a source-bound request's `status` field: an exact NumPy ndarray, one dimension, 1–4096 exact Python strings, each 1–256 Unicode codepoints. It stores shape, dtype and values with a deterministic JSON-content SHA256. Object-pointer bytes are never hashed. Generic object arrays remain unobserved; custom objects are never converted with `str`, `repr`, `item`, array coercion or a device transfer. Scalar strings and bounded Unicode arrays retain their classifications.

The observer and offline verifier derive status through the same retained-value decoder. Malformed, mixed, missing or oversized status values produce an explicit observation error and incomplete scope, rather than success or an absent failure. A true Python `None` return remains ambiguous between return and exception unwind. Content/shape/hash tampering and dropped derived failure labels are rejected. The policy version, codec limits and implementation fingerprint change, so old manifests cannot silently adopt this revision.

The native regression fixture is a compressed byte-exact copy of the 92-event failed profile. It must continue to expose the five irrecoverable raw-status gaps. Synthetic array tests validate future recording only; they never substitute values into the historical events. This fix neither resolves the cabinet's physical planning failure nor authorizes another native episode.

### Opt-in scalar cuRobo return evidence (future-only)

`planner_policy(solver_details=True)` selects observer v3 and includes the new
`planner_result_evidence.py` implementation hash, the exact native `planner.py`
SHA and the installed `motion_gen.py` SHA. The default remains v2. A future
finite design must explicitly declare `planner_solver_details: true`, include
the helper in its frozen code and the actual cuRobo source in its external
inputs. This source change is not an execution authorization and supplies no
new owner, retry, seed substitution, or episode budget.

The existing current-thread profile observes the scalar CuroboPlanner return
inside its matching active left/right Robot request. It copies only existing
CPU locals: world base/target poses (native wxyz), transformed/biased base goal,
current qpos, source joint indices and five-decimal rounded angles, optional
constraint vector, and the original `c_time`. Offline verification reconstructs
the joint-name mapping and rounding. It does not assert that these coordinates
alone prove collision clearance or feasibility.

Before profiling, exact source-bound runtime types and methods are checked.
The normal instance dictionary of the exact audited MotionGenResult is read for
`status` and `attempts` only. Status permits the exact audited MotionGenStatus
enum (stored name/value with a fixed mapping), exact bounded Python str, or
explicit None. Unknown/subclass/device/property values are unsupported and
make the evidence incomplete; they do not become a failure or success. No
extra `.item()`, `.cpu()`, `.to()`, `.numpy()`, tensor repr, pose getter, solver,
control call, or physics step is made. The native source's own existing device
operations still execute unchanged. Existing plain plan-config fields and
wrapper joint/frame fields are recorded separately. The YAML content/hash is
the file at the runtime `yml_path` **at observation start**, not proof that an
unobserved runtime world/config was reconstructed from that file.

A returned enum is a solver-reported category, not a proof of the underlying
physical cause. This installed enum has no TIMEOUT member. A 10-second timeout
configuration or elapsed time above 10 seconds does not establish timeout as
the exit reason. `max_attempts=10` is a configuration ceiling; actual attempts
come only from the separately retained result field. None/string categories
remain exactly as observed. Batch planning is still covered by the bounded v2
status codec; this extension does not invent batch solver detail.

Adding scalar wrapper events changes global call IDs. IDs associate evidence
within one receipt only. Historical call 46 remains the old raw scalar Fail;
future comparison must use arm, enclosing native scope, Robot request order,
and target/joint evidence, not demand the same numeric call ID. The five old
object-status arrays remain permanently absent, and all old index/profile bytes
and the failed whole-observation verdict stay unchanged.

The profile and file reads have wall-time/CPU overhead. CPU source-AST probes
verify argument/return identity and unchanged control/device call sequences;
they do not prove GPU scheduling or native episode reproducibility. The frozen
old 92-event fixture is only a refusal/preservation regression. Synthetic
solver result types used in CPU probes are never backfilled into that fixture.

The audited wrapper builds a planning world containing a hard-coded table
cuboid (native `planner.py:338–356`). A future `WORLD_COLLISION` category is a
report about that planner's configured world/model. It cannot by itself prove
contact with the physical cabinet, which is not established by this evidence.

### Targeted cabinet finite package integration

`targeted_observation_version=cabinet_scalar_result_v3_one_shot_v1` requires
explicit v3 scalar-result opt-in and two distinct immutable failed seed-10050
attempts: the metadata smoke and the earlier expert-only observation. Their
source paths, copied input bytes and hashes are checked together. The history
budget is two previous initializations plus at most one new initialization,
with zero replay/retry. This is a diagnostic repetition and does not replace
or upgrade either earlier failure.

Before binding the actual planner observer, this mode persists the **already
stored** fresh-success-metadata receipt and its complete physical snapshot.
It invokes no additional physical getter. This retains the initial evidence
if actual runtime type/source compatibility fails before expert actions.
Such failure consumes the one initialization, starts zero experts, preserves
the original error, closes the native environment and never retries. Output
verification checks the two initial files even for incomplete observations;
when expert execution started, the receipt must also match the attempt's
fresh metadata and the frozen scene.

A GPU-free preflight cannot import the full installed cuRobo motion module:
its `plan_grasp` default `Pose.from_list` expressions allocate on CUDA at module
load. Source-hash/AST dataclass and constructor checks are therefore labelled
CPU evidence only. Actual source/type/method bindings are checked after the
sole authorized native setup, before any expert control. No module-import or
native-constructor success is claimed from the CPU probe.
