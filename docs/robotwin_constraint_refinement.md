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
   and only `add_rule`/`replace_rule`. Task, signatures, existing subjects and
   activation anchors cannot change to hide a counterexample. Rule count/growth
   is bounded. `allowed_margin` has explicit `m`, `rad`, or `fraction` units and
   is independent of penalty scale. The reference activates every rule.
3. **Seal:** `seal_revision` freezes the template, numeric implementation,
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
   those scenes are development evidence for subsequent revisions: include all
   exposed rows in the next development ledger and allocate a fresh holdout.

A revision directory contains `sealed_revision.json`, `holdout_release.json`,
`split_plan.json`, `development_report.json`, and `holdout_campaigns.json` with
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
success, at least 80% retention of successful tested behaviors, three crafted
families, and no observed admitted failures. Within each scene, any admitted
failure counts as a scene failure. The one-sided exact 95% upper risk bound must
be at most 10%; zero failures in 29 independent accepted scenes meets that
numeric threshold. Correlated variants from three scenes never become 29
independent samples. Incomplete sampling gets no confidence claim.

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
