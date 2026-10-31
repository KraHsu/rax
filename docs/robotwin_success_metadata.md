# Fresh-scene native success metadata

`open_laptop.setup_demo` does not set `arm_tag`; its native `play_once` and
`get_info` do. The original success predicate subsequently chooses the TCP using
that field. `put_object_cabinet` similarly needs `arm_tag` and the initial
`origin_z`, which its setup does not assign. A successful expert reference does
not initialize a later, newly constructed replay or model environment.
`place_object_scale` has the same missing `arm_tag`: its success predicate checks
whether the initially selected object's carrier gripper is open.

The source-bound adapter in `services/robotwin/success_metadata.py` reproduces
only these initial assignments:

| Task | Fixed initial-state rule |
| --- | --- |
| `open_laptop` | Rotate local X by the laptop root quaternion. Positive dot with world X selects left; zero or negative selects right. |
| `put_object_cabinet` | Initial object X greater than zero selects right; otherwise left. Preserve the initial object Z scalar and its native dtype. |
| `place_object_scale` | Initial object X greater than zero selects right; otherwise left. Assign the source-bound native `ArmTag`, without adding `origin_z`. |

The task source SHA256 values are `ec5f6e22…`, `4c705f50…` and `37415b6f…`, respectively. The `ArmTag` class
and the pure `get_face_prod` helper are independently source-bound. The adapter
calls neither `get_info`, `play_once`, motion, the success predicate nor a scene
step. It changes only the declared instance fields and an audit marker. Native
classes, source files and success predicates remain unchanged.

For scale, the audited source supports mouse, stapler and bell objects and scale
model IDs 0, 1, 5 and 6. Receipt validation binds the initial object position and
dtype to one named physical actor, checks the unique scale actor, and reconstructs
the original instruction tokens. The original native predicate remains XY
distance below 35 mm, object height above scale point Z minus 10 mm, and the
initial carrier gripper open. It does not require contact, support, weight or an
upper object-height bound. Moving the object across X=0 later cannot change the
selected hand. No constraint template is changed by this initialization fix.

Call `initialize_success_metadata(task, native, expected_scene_sha256=...)`
immediately after every fresh `setup_demo`, before any actions. The fixed-case,
search and model factories now do so. The model collector performs this for both
its instruction environment and the second environment used for the rollout.
`readonly_instruction_metadata` can supply these tasks' instruction tokens but
never initializes success fields as a side effect.

The audit records exact actor pose/velocity, articulation root pose/velocity and
qpos/qvel, gripper command values and timestep before initialization. SAPIEN 3
actor velocities are read from their rigid-body components; static bodies have
explicit zero velocity. Before/after hashes must match. Repeated initialization
cannot rebase height after scene evolution, and stale existing fields are
rejected. Errors use `SuccessMetadataError`; the reference seed scan aborts on
them instead of silently selecting a different scene.

The stored metadata inputs identify one physical actor/articulation by kind and
native name. Offline validation binds the input pose and dtype to that exact
snapshot entity, recomputes the native branch/height, and reconstructs the
task scene fingerprint after excluding robot articulations. Changing metadata
and assigned values together cannot pass without the bound physical input.
Changing the physical snapshot too must still match the externally frozen scene
identity. The existing scene fingerprint quantizes geometry to 1 mm/1e-3
quaternion units; it does not identify smaller cross-run physical differences.
The before/after initialization hashes themselves are exact.

NumPy 1 and 2 promote Python scalars differently in the audited transforms3d
0.4.2 quaternion helper. `success_metadata_policy.numeric_context` freezes the
collector's NumPy version and transforms3d version. Offline verification preserves
the original quaternion dtype and norm accumulation, then explicitly reproduces
that context's float32/float64 division and subsequent arithmetic. The checker
may run another supported NumPy version. Receipt context must exactly match its
sealed manifest, and actual native search requires the source runtime context
before GPU initialization.

Evidence and release checks:

- New task configurations/manifests contain `success_metadata_policy`, including
  native/helper/adapter source identities and numeric context.
- Fixed-case references and completed replay outcomes retain `success_metadata`;
  acceptance verifies both against the manifest and frozen scene.
- Search retains original source bytes, verifies source and working reference
  receipts, and checks each completed native outcome. Supported tasks' old
  references without initialization proof cannot be relabeled as new evidence.
- The model collector records the second environment's receipt in
  `source_run.json`; raw model evidence verification checks it and its policy.
  Before creating output or initializing the native runtime, the collector
  verifies its source reference receipt and requires the frozen numeric context.
- Existing failed runs and their snapshots are immutable. New attempts need a
  separately frozen package and manifest. This adapter does not activate a
  constraint template or certify task success.

CPU tests execute the source-extracted native task methods and all three actual
factory functions with simulated physical objects. They verify both arm choices,
strict origin-height dtype/boundary equivalence, fresh-instance lifecycle,
instruction immutability, physical-state invariance, source drift, receipt
tampering, and the no-resampling error path. Separate native-Python CPU probes
exercise real SAPIEN entities/components without creating a Scene or renderer.

For the first native factory smoke, the validator supports the explicit
`--case-preset baseline_smoke` for `open_laptop` and `put_object_cabinet` only
(scale support does not expand that preset). It requires `--seeds 1`,
`--max-seed-attempts 1` and `--baseline-repeats 2`: one fixed seed, one reference
attempt, and at most two baseline replays. The second replay runs only if the
first passes the existing baseline gate. Reference failures retain their initial
scene, initialization receipt and sampled actions in `reference_attempts`; no
replacement seed or automatic resume is permitted. A new smoke output must use
a separately reviewed plan. Default fixed-case validation retains its 18 cases.
Neither a captured reference nor a completed baseline pair certifies constraint
tolerances or dataset publication.
