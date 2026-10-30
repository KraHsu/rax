# Counterexample-guided constraint refinement

`services.robotwin.constraint_refinement` separates adaptive discovery from
independent native acceptance. A proposal is JSON data in the existing primitive
DSL; the agent cannot supply executable constraint code. No development result,
including zero discovered failures, activates a template or starts training.

## Protocol and artifacts

1. **Discovery:** run `search_robotwin_constraint_counterexamples.py --split dev`.
   Verify raw campaigns with `verify_campaign`, namespace their local slot IDs
   with `campaign_rows`, then call `development_summary`. Record commanded and
   actual constraint verdicts separately. Primary counterexamples are completed,
   realized, planner-valid executions with passing commands and native failure;
   passing actual trajectories identify a stronger subset. Incomplete/planner
   failures do not become native task failures.
2. **Revision proposal:** `propose_revision` accepts cited development evidence
   and whitelisted rule edits, numeric `set_coverage_policy`, or the separate
   audited `set_motion_coverage` migration. Task, signatures, existing subjects and
   activation anchors cannot change to hide a counterexample. Rule count/growth
   is bounded. `allowed_margin` has explicit `m`, `rad`, or `fraction` units and
   is independent of penalty scale. The reference activates every rule.
3. **Seal and reserve:** `reserve_holdout` in `refinement_exposure` calls the
   pure `seal_revision` helper and registers the release in the owning registry.
   The helper alone cannot authorize a final experiment. This freezes the template, numeric implementation,
   protocol, development ledger, disjoint scene split and generator. The frozen
   generator specifies random seed, candidate budget, baseline repetitions,
   replay of outside controls, per-source scene budgets and model checkpoint.
4. **Final holdout:** no adaptive feedback or outcome-based candidate selection.
   Replay all preregistered candidates, including those the constraints reject,
   so successful alternatives can be measured. Each scene has reliable native
   baselines. Use new scene seeds excluded from training and development.
5. **Decision:** `assess_revision` rehashes raw source files through the runner;
   `attach_revision` saves the recheckable report. Failures return
   `needs_revision`; accepted revisions are eligible for publication, subject to
   the existing v2 native evidence gate as well. Once holdout outcomes are read,
   those scenes are development evidence for subsequent revisions. Close the
   round with `consume_holdout`, which burns **all reserved seeds**, records all
   observed scene identities, and prevents further collection in that round.
   Subsequent reservations in the same registry reject reused seeds or scenes;
   final publication verifies registry ownership and complete history. Include
   exposed rows in the next development ledger and allocate a fresh holdout.

A revision directory contains `sealed_revision.json`, `holdout_release.json`,
`split_plan.json`, `development_report.json`, `model_evidence.json`, and `holdout_campaigns.json` with
`{"version":1,"campaigns":["/absolute/raw/campaign/root", "..."]}`.
Campaign artifacts and revision files must remain available for later training
validation; summaries alone are insufficient.

## Sources, statistics and retained successes

Expert perturbations, crafted candidates and actual model behavior remain
separate strata. A model chunk embedded in an expert trajectory is a development
proxy. A full policy trace blended toward the reference is also transformed
discovery evidence. Neither can fill the final actual-policy stratum: its
candidate must be the original complete `model_closed_loop` trace with no
search transform or reference blending.

Default release requirements apply separately to crafted and actual model
sources: at least 29 distinct accepted scenes, at least 10 scenes with native
success, at least 80% retention of successful tested behaviors averaged equally
over scenes after deduplicating identical scene/action traces, three crafted
families, and no observed admitted failures. Within each scene, any admitted
failure counts as a scene failure. The one-sided exact 95% upper risk bound must
be at most 10%; zero failures in 29 independent accepted scenes meets that
numeric threshold. Correlated variants from three scenes never become 29
independent samples. Incomplete sampling gets no confidence claim.

At least 80% of admitted unique actions in **every tested scene** must have
completed, realized, planner-valid executions. Excluding many unexecutable
admitted actions therefore cannot make a template appear safe. Counts also
report identical scene/action traces with conflicting native outcomes: an
action-only rule cannot separate those outcomes; failures remain in the report.

Final model proof inventory has shape `{"version":1,"runs":[{"root":"...",
"checkpoint_path":"...","checkpoint_receipt_path":"...",
"server_config_path":"...","server_receipt_path":"...","collector_path":"..."}]}`.
For each run the gate rehashes the real checkpoint and collector receipts, original
raw EEF20 actions, per-step observation receipts, and original candidate file.
It independently reconstructs the declared legacy rotation conversion and
native gripper clipping. The executed candidate must have the exact original
trajectory hash; provenance booleans alone do not prove model origin. Source
runs must have `created_unix >= frozen_unix` for final holdout. Current
development captures remain descriptive if they predate this proof contract.

This describes scene-level conditional risk under the declared frozen generator
and policy distribution. It is not proof that every action satisfying the
constraints succeeds, nor evidence of task success under an untested policy.
Rejecting every action cannot pass the accepted-scene and success-retention gates.

## Mandatory publication and training gate

Write `refinement_requirement.json` in a new registry with
`{"version":1,"protocol":DEFAULT_PROTOCOL}`. The publisher then requires a ready
refinement report for every selected task/template. Also invoke both publication
and training's `--validate-only` path with `--require-refinement` so deleting the
marker cannot silently downgrade a run. `acceptance_manifest.json` freezes the
requirement and exact proof inventory; validation recomputes native evidence.
Historical pilot-only datasets remain readable without the explicit new flag.

## Development screening without GPU

`scripts/propose_robotwin_gripper_revision.py --export seed_evidence.json
--output /new/proposal/directory` independently revalidates immutable historical
native artifacts, proposes reference-stable gripper plateau rules, and rescores
every completed commanded/actual trace at four physical margins. It reports
retained successes and unresolved counterexamples. This is offline screening,
not a new simulator experiment, and every output remains `proposal_not_accepted`.
Stable plateaus include partial openings, use explicit `1e-6` stability against
the segment's first value, and span at least two reference frames.
The gripper proposer recognizes a finite family map, including `early_close`,
`never_close`, `never_release`, `open_early`, and the supported chatter names.
It can cite a completed measured trajectory that passes the constraints but
fails the native task even if exact intervention tracking failed. This remains
an observed action-space counterexample and a development hypothesis; it does
not establish exact-command causality or activate a revision. An unrealized
command whose actual trajectory violates constraints cannot trigger this rule.

## Data-only full-motion tolerance revisions

Composition is separate from numerical tolerance tuning. See
[`robotwin_motion_composition.md`](robotwin_motion_composition.md) for explicit
reference-rule exemptions and the independently versioned motion migration.
Adding a geometry rule under legacy compilation may remove generated ordinary
motion; only a fixed independent composition policy guarantees preservation.

One patch operation may be `{"op":"set_coverage_policy","policy":{
"half_angle_rad":1.2,"hold_margin_m":0.04,"direction_slack_m":0.005}}`.
These values illustrate the schema, not validated task parameters. The policy
may contain a nonempty subset of those three keys. It replaces the template's
explicit override object; omitted keys use the unchanged default coverage
policy. The existing template validator applies the physical-unit whitelist:
angle 5–85 degrees expressed in radians, hold radius 0.001–0.1 metres, and
direction slack 0–0.05 metres. These limits are search-budget priors, not
physical safety guarantees.
With the existing cone-plus-slack formula, increasing the half-angle does not
guarantee global set inclusion for small backward displacements. Revisions
must therefore be evaluated as new allowed sets, not assumed monotonic relaxations.

This operation cannot disable full coverage, move reference activation masks,
change reference segmentation, or change loss penalty scales. It must cite
development evidence, produces a new template hash, and remains only a proposal
until the revised policy and generator are frozen and pass fresh independent
native holdout. Arbitrary keys, code, booleans, nonfinite numbers and repeated
coverage replacements in one patch are rejected.

## Joint gripper candidate (schema 1.4)

`gripper_pair_max_interval`, subject `both_grippers`, measures
`u = max(left_open_fraction, right_open_fraction)`. It uses the canonical
identity reference frame, fixed interval activation, fraction bounds in `[0,1]`,
and the ordinary interval hinge and configured penalty scale. The OpenWAM
physical decoder supplies columns 9 and 19; both must be valid at a frame.
A padded hand never substitutes for the other. `torch.amax` divides tied maximum
subgradients over the two independent grip dimensions. A positive lower bound
therefore provides finite gradients to both hands at equal closed commands.
The pair counts once as a supervised feature, not twice as two marginal rules.
Schema 1.0–1.3 remain readable, including schema 1.3 window direction constraints.

The fixed template selector is `joint_gripper_plateaus`, state
`before_joint_closed`, after `episode_start`. The boundary is the first reference
frame beginning at least two consecutive frames with both opening fractions
within `1e-6` of zero. Earlier max-opening plateaus must remain within `1e-6` of
their run's first value for at least two frames. Max may stay stable while the
other hand ramps. This boundary is a reference command landmark, not measured
contact or physical grasp. A missing boundary is rejected as outside template.

A data-only `replace_pregrasp_gripper_with_pair` operation names `rule_ids` and
one `joint_rule`. Only existing per-arm `gripper_interval` rules selecting
`gripper_plateaus/all/episode_start` can be migrated. Each old rule gains
`reference_phase: after_joint_closed`; its later bounds, units, margins and loss
scale remain intact. Existing fully-closed holding rules are untouched. The
instantiator records replaced preframes and retained postwindows and requires
every removed action frame to be covered by an effective joint interval with
positive lower bound. Asynchronous patterns with remaining gaps are rejected
for review. Generic `replace_rule` cannot alter this phase or other anchors.

`joint_gripper_proposal` requires cited verified development successes and
failures and returns `revision_proposed` only. Margins are hypotheses. Joint
expressivity resolves a demonstrated product-set limitation; it does not certify
success. Fresh policy fingerprints, native counterexample search, and independent
holdout are still needed before dataset publication or training.

The separate positive-lower angular-band limitation remains unresolved: exactly
aligned axes have zero cross-product norm gradient even when the lower-angle
hinge is positive. CPU regression preserves this fact and checks finite outward
gradients on either side. Floating-point rounding around an angular mode
midpoint is not a convergence guarantee. This version does not silently change
the angular loss or release the current candidate for training.
