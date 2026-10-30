# Fixed reference roles in task templates

Templates may declare at most two roles using exact event signatures from the
successful reference recording. This is an annotation compiler feature, not a
new training primitive or a semantic operation taxonomy.

```json
{
  "reference_roles": [
    {
      "role": "basket_carrier",
      "source": "reference_event_signature",
      "event_signature": "close",
      "rationale": "Native expert grasps the basket with the opposite arm and does not release it."
    },
    {
      "role": "object_placer",
      "source": "reference_event_signature",
      "event_signature": "close,open",
      "rationale": "Native expert first grasps and then releases the object."
    }
  ]
}
```

A rule may use `"subjects": "role:basket_carrier"`. Instantiation resolves this
to `left` or `right` before selecting reference windows, bounds and anchors. The
emitted schema 1.4 constraint uses `left_eef` or `right_eef`; OpenWAM does not
infer roles again. Generated gripper actions cannot change subjects or masks.

Each declaration must match exactly one active reference arm. Missing matches,
two matching arms, duplicate role names, and two roles resolving to one arm fail
closed. The declared role meaning must be justified independently against the
task's source/metadata. An event pattern alone does not establish object contact
or physical role identity. No per-seed fallback silently changes the template.

`instantiate` returns a `reference_roles` audit with declarations, fixed bindings,
observed reference events, recording identity, and hashes of declarations and
decoded reference inputs. The latter is canonical JSON of both pose/gripper
arrays, action contract, recording ID and expert provenance, not a byte hash of
the mutable HDF5 file. Role-dependent constraint evidence also records these
hashes.

The accepted-template publication paths pass `reference_template` into
`embed_review`. Under its source lock, annotation recompiles that template,
checks the proposed constraints, preserves its `coverage_policy`, and checks the
final constraints again. Role audits are stored in the existing
`trajectory_constraints/history` entry's `programmatic_audit`. The surrounding
entry still carries the source HDF5 byte hash. Strict bundle provenance is
unchanged. Templates without roles keep their existing audit/history shape.

Template compatibility is not acceptance. Adding roles or changing subjects is
an explicit authoring correction; the existing RSI `replace_rule` operation
still prohibits subject/activation changes and has no role-declaration setter.
A corrected template requires fresh native evidence and the existing release
gate before activation. Successful reference compilation alone cannot establish
success retention or rejection of action-satisfying failures.

For `place_object_basket`, the prior right-arm terminal rules were wrong for 19
of the 50 existing reference recordings. A single proposed template now binds
the basket carrier on either side. All 50 bindings agree independently with
native `{a}`/`{b}` scene metadata and initial actor sides; the native expert's
normal and recovery branches use the same roles. This is a CPU authoring audit,
not new native validation. Object containment, load retention, uprightness and
contact remain unobserved by these EEF height/rise features.

Chunking does not repeat role resolution. Reference event activations stay
fixed, and the existing long-displacement anchor conversion is unchanged. A
32-action chunk can only supervise the decoded features and fixed anchors that
its existing loss implementation supports.
