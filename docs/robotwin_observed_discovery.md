# Actual-trajectory development discovery

This mode answers whether the **executed** action trajectory satisfies the
labels, and whether native RoboTwin reports success. It is separate from the
existing strict intervention protocol. The 5 mm tracking threshold, planner
checks, relative-to-control tracking report and native success rule are unchanged.

The first frozen design is `open_microwave`, already-exposed development seed
10000: one reference-command control, then at most 24 frozen candidates. The
first 20 use the existing v2 selection policy: 18 target the active left arm and
two are inactive right-arm controls (tool spin and gripper chatter). Four
additional left-arm Y/Z offsets of ±10 mm are executed without screening them
away; all four violate the current command constraints. Source, exact trajectory,
reference-blend amount, selection audit and excluded hypotheses are explicit.
These are crafted/expert-derived proposals, not model rollouts.

Generation ran under the recorded NumPy 2.5.4 context. Execution never regenerates
or re-ranks proposals: NumPy 1.26 can produce different floating-point hashes and
break ties differently. It verifies the sealed trajectory bytes, original
proposal identity, finite slot plan, selection audit and independently recomputed
physical scores using the existing evaluator comparison tolerance. This does
not change any physical constraint or strict tracking threshold.

The control must complete, have successful planner records, and attain native
success. It may violate strict tracking or the current constraint labels. A
complete planner-failed candidate still contributes an observational record,
with its planner failure kept separately. Incomplete execution, infrastructure
failure or invalid control stops the finite plan. No slot, seed or reference is
retried; an existing output directory cannot resume. Every slot creates a fresh
scene and verifies the original scene, robot initial state and native assets.

Each result retains the original per-trial strict classification under `strict`
and `strict_eligible`. The latter is **not** a formal campaign qualification:
the control may fail the original strict baseline gate. Separately:

- Complete actual-pass/native-fail is an observed false accept, including
  tracking-invalid or planner-failed trajectories.
- Complete actual-reject/native-success is an observed successful variation
  rejected by the labels, including tracking-invalid trajectories.
- Command compliance, actual compliance, native outcome, controller/planner
  status and tracking errors remain separate facts.

The output uses `observed_manifest.json`, `observed_trials.jsonl` and
`observed_report.json`, with a distinct development-only purpose. It does not
write `search_manifest.json` or an acceptance certificate. The formal search
verifier rejects even a renamed observed manifest. Every report states
`formal_evidence_eligible=false`, `acceptance_ready=false`,
`publication_allowed=false` and `resume_allowed=false`. Evidence may motivate a
development proposal; it is not independent holdout evidence or an IID success
estimate. Neither contact causes nor global success equivalence follow from it.

`scripts/discover_robotwin_observed_constraints.py` defaults to CPU validation.
Native execution requires `--execute` under a separately reviewed lease-owning
supervisor. This runner does not acquire or clean up foreign GPU processes.
