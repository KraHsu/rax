# Composing ordinary motion with geometric allowed sets

The legacy compiler fills frames without a position constraint. Consequently,
adding a broad interval on only one position axis can replace an existing
ordinary-motion direction cone. Temporal position coverage alone does not imply
direction supervision. Missing `motion_coverage` keeps that legacy behavior so
old templates and frozen evidence remain reproducible.

New templates can declare an independent, reference-only composition policy:

```json
{
  "motion_coverage": {
    "version": 1,
    "task": "turn_switch",
    "mode": "independent",
    "exemptions": [],
    "rationale": "Ordinary motion remains active alongside task regions."
  }
}
```

With a fixed policy and fixed exemption rules, adding a positional region or
rigid-local-point rule leaves every generated ordinary-motion/hold constraint
unchanged. The final allowed set is their intersection. This claim concerns
individual constraints and feasibility, not the average loss: adding zero-loss
terms can lower an existing weight-normalized average. Loss normalization is
unchanged by this patch.

For migration that preserves all old direction segments and numerical values,
`legacy_motion_coverage(template)` creates explicit exemptions for exactly the
original position/motion rule identities. An exemption has `rule_id`, `primitive`,
`rationale`, and `provenance` containing `source` (`template_migration` or
`manual_review`), `reference`, and `sha256`. Migration provenance binds the
parent template hash. It never stores an episode's frame numbers as the rule
identity. Each compilation resolves the rule's fixed reference selections,
including any existing long-span displacement conversion to a reference anchor.
New rules are not automatically added to that exemption list. Changing an exempt
rule's primitive requires an explicit reviewed policy change; retaining only its
name cannot silently change its replacement-mask semantics.

The policy's task must match the template. Missing, duplicate, reserved,
non-positional or cross-task rule identities fail validation. Compiled matches
use the complete canonical `rule_id_arm_start_end` identity, including subject
and activation checks; a rule named `a` cannot exempt `a_b`. An exemption with no
reference selection fails rather than silently applying elsewhere. Manually
reviewed exemptions are an explicit change to ordinary-motion supervision, not
a claim that one axis or a TCP region physically guarantees motion direction.

`mode="legacy_position_override"` with an empty exemption list explicitly requests
the old semantics. The numeric `coverage_policy` remains unchanged and cannot
change motion composition. RSI uses the separate `set_motion_coverage` operation;
an automatic migration must cite the actual legacy parent hash and its complete
original replacement-rule set. New migrations remain proposals and need their
own native validation before publication.

Compilation and HDF5 publication store the complete policy, resolved compiled
identities, fixed reference activations, effective replacement spans and generated
ordinary-motion spans under `coverage.motion_composition`. The pre-existing
`coverage.complete` still means temporal position/motion coverage. The separate
composition check means generated ordinary-motion/hold supervision or an
explicit replacement exists at every action frame; it does not prove progress,
contact, object state or task success. Existing independent HDF5 data cannot be
reannotated through an implicit legacy/default template.

Publication recomputes this policy under the source lock and checks final label
identity. OpenWAM's compiled constraint schema and loss are unchanged: it consumes
all stored constraints and performs the existing temporal-coverage check. It
does not independently recompile template composition policy during training.
Pointwise local-point regions require only the current decoded position and
rotation; ordinary-motion windows and long-span anchors retain their existing
32-action chunk rules.
