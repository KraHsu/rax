# Count the handover release-height feature once

`handover_mic_reference_height_once_v1` is a candidate data revision for the
exact parent template `e3739d94cec71aa07a3ca76ae24fd40249547cfe352d0b2c57787cc71189ed38`.
It removes only `release_vertical_rise` and retains `release_height`. It is not
installed in a registry, applied to HDF5, or enabled by a training config.

The parent asks for first-closure-to-release vertical displacement. For the
50 audited training references, that span is 133–142 actions. The existing
compiler changes its anchor to `reference_frame` because it exceeds the
32-action horizon. After adding the fixed origin Z to the bounds, it is the
same world-Z interval as `release_height`, at the same fixed release event.
It no longer measures a displacement between two generated endpoints.

Use `services.robotwin.handover_mic_height_revision.audit_reference` on each
complete reference before considering this revision. It rejects other parent
hashes, short `predicted_start` spans, non-world axes, rotated frames, local
points, different subjects/events/penalties, or even slightly unequal physical
bounds. It checks every 32-action window and verifies every retained compiled
physical field and automatic motion guard. The event at measured row0 is not a
generated action. Position validity and the two adjacent reference gripper
validities remain required. A future reference is not covered merely because
these 50 references passed.

There are two separate candidate products:

- `candidate_template` creates the new template. Its current compilation is
  compared with the *same compiler's* parent compilation.
- `candidate_embedded_bundle` changes a supplied bundle only in memory. It
  removes the one proved member and preserves every other field and constraint
  verbatim. This is necessary because the historical published training bundles
  contain full-motion-v1 guards and a `recipient_terminal_height` label absent
  from the current template. Recompiling the current template would also change
  those historical labels. This helper does not do that.

The example template is under `examples/constraint_revisions/`. No generic
remove-rule operation or automatic deduplication is added to the RSI DSL.
Every removal has an explicit removed-ID → retained-ID audit.

The conjunction's mathematical height interval is unchanged. **The weighted
mean objective changes intentionally**. For a chunk where the feature is
active, with original numerator N, denominator W, removed weight w and penalty
P, the new objective is `(N - w*P)/(W - w)`, multiplied by the existing global
coefficient. Other errors receive different relative weight. A per-kind scale
override can make the two original penalties different despite their equal
geometric interval; the retained axis-position override remains in force.
Neither identical loss nor unchanged gradient is claimed. Under DDP the removed
weight leaves the *global* denominator; it must not be normalized separately
per rank. No objective implementation is changed by this data revision.

Floating arithmetic also matters. On 1,100 explicit boundary probes from the
50 original training references, the different float32 subtraction origins
produced residual differences up to 5.22e-8 metres. Six probes differed in exact
zero membership, and two sitting at the 1e-6 audit threshold differed in that
threshold decision. No threshold is relaxed to conceal this. All 120 command
and actual trace classifications from the original 60 development records
remain unchanged, and all retained residuals are identical within the same
scorer. Cross-NumPy frozen-evaluator numeric agreement is checked separately.

CPU validation includes the original 50 reference arrays, all original 60
native records in three scenes, synthetic recorded HDF5 through the actual
RoboTwin reader/decoder/objective (20D, 80D, normalized, event edges and padded
tails), masks and detached measured anchors, per-kind scale overrides, and a
real two-process Gloo DDP comparison with the pooled-batch objective. The local
reader harness only bypasses unrelated optional dataset registry imports;
it does not replace reader, decoder, geometry or objective code. The 60 trials
are correlated development evidence, not new heldout acceptance or proof of
physical success. No native replay or training is part of this revision.
