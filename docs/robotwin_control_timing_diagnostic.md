# Stage A: observe expert capture and legacy replay timing

This is development evidence only. It cannot publish a template, admit a
holdout result, or start training. It does not implement a new replay mode.

The finite design fixes `dump_bin_bigbin` and `place_dual_shoes` at development
seed 10000. Each gets at most one expert capture and, only after successful
capture and complete timing observation, one unchanged legacy EEF replay.
No seed replacement, candidate perturbation, or retry is allowed. The two tasks
are independent: one blocked task does not authorize another attempt, and does
not cancel the other task's already fixed attempt. Output directories are
one-shot. The shared GPU lease and execution timeout belong to the separately
frozen owner package; this script must not be launched outside that owner.

The existing sparse capture and EEF replay have different clocks. Capture uses
the original picture callbacks plus initial/final measurements. Replay samples
after each EEF command, whose planner can take a different number of physics
steps. The native `delay` argument is a loop count: in the audited source,
each loop issues 300 steps and continues updating gripper drives. Runtime `dt`
is measured, rather than inferring seconds from the delay argument. No repeated
EEF row, wall-clock sleep, or guessed hold is inserted by Stage A.

The observer calls the original `System.step` descriptor exactly once, with the
same instance/arguments, and counts only its normal return. Native exceptions
are re-raised unchanged. Non-target backends pass through unchanged. A profile
hook observes the Python `Scene.step` wrapper, original CPU control arguments,
and sample return values; it does not add state getters, transfers, synchronizes,
controls, or success-predicate calls. Actual SAPIEN CPU tests found no C-profile
events for the native step method, so C-profile accounting is explicitly not
used. Both profile hooks and the declaring-class descriptor are restored.
Journal failures mark the observation incomplete without replacing native
returns or errors. Journal argument hashes are of values already on the CPU;
unknown argument types are reported unobserved rather than converted.

Coverage has a precise limit. The package freezes and audits native task,
base-task, robot/planner/utility, capture and SAPIEN wrapper sources. Stored step
aliases, unresolved dynamic step lookup, omitted observed source, timestep
writes, other pre-existing Python threads, or target steps on a different
thread invalidate the clock. A cached original bound method outside that
audited Python scope can bypass the shim and is not automatically detectable.
The receipt makes no arbitrary-code or hidden native-call coverage claim.
Python return events can also occur during exception unwind: the reported
`take_action` return count is checked against actual row/sample hashes and is
never itself treated as a successful-command counter.

Dump's two original published signatures are frozen together. The single
captured reference must compile under exactly one; all compilation rejections
are preserved. Routing is data-only and does not replay an expert or choose a
more favorable seed. Full initial physical states, including velocities and
gripper commands, must match before comparing replay timing. The original
5 mm/5 degree/gripper tracking report and native outcome are retained without
relaxation. A failed baseline stops the task while preserving its useful
timing/native/constraint observations. A timing gap alone does not prove that
the gap caused a task failure.

`diagnostic_manifest.json`, `artifact_index.json`, step/control journals, clock
receipts, original reference/attempt, legacy outcome, and sample comparison
form the audit trail. `verify_output` checks committed file identities, journal
counts/hashes, sample mapping, and complete-status consistency. These are not
formal acceptance manifests; renaming a file does not change its purpose.

CPU checks:

```sh
python -m pytest -q local_tests/unit/test_robotwin_control_timing_observer.py \
  local_tests/unit/test_robotwin_capture_timing.py
python scripts/diagnose_robotwin_capture_timing.py --design /frozen/design.json \
  --task dump_bin_bigbin --output /unused/path --check-only
```

Check-only imports the installed SAPIEN wrapper class for source inspection;
it does not create a Scene, initialize CUDA, or create an experiment output.
Execution requires the explicit `--execute` flag and the reviewed owner.
