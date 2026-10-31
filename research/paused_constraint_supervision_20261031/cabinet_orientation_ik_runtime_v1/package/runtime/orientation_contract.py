"""Fixed reference geometry and native-command mapping, entirely CPU-only.

No native environment imports and no feedback-based target selection. The float32
quaternion norm preserves NumPy1.26 source arithmetic across local checker versions.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from geometry import angular_error, origin, pose, quaternion

PROPOSAL_SHA = "86e435fc5f6980c2893031000c611c59d58c7731dc423633977b229564fbaaaf"
INTEGRATION_SHA = "8f8e2255347c5cc8488d128730ff065bfd9be30d3d308bceaedc74ca6f58b2b4"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native_quat2mat(q):
    """Frozen transforms3d quat2mat with native NumPy1.26 scalar promotion."""
    q = np.asarray(q)
    if q.shape != (4,) or q.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(q).all():
        raise ValueError("Quaternion shape/dtype/nonfinite")
    w, x, y, z = q
    nq = w*w + x*x + y*y + z*z  # Preserve source-dtype norm rounding.
    if nq < np.finfo(float).eps:
        raise ValueError("Degenerate reference quaternion")
    s = np.float64(2.)/np.float64(nq)
    w, x, y, z = map(np.float64, (w, x, y, z))
    X, Y, Z = x*s, y*s, z*s
    wX, wY, wZ = w*X, w*Y, w*Z
    xX, xY, xZ = x*X, x*Y, x*Z
    yY, yZ, zZ = y*Y, y*Z, z*Z
    return np.array([[1.-(yY+zZ), xY-wZ, xZ+wY],
                     [xY+wZ, 1.-(xX+zZ), yZ-wX],
                     [xZ-wY, yZ+wX, 1.-(xX+yY)]])


def native_mat2quat(M):
    """Frozen transforms3d eigenvector construction, WXYZ and positive w."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (3, 3) or not np.isfinite(M).all():
        raise ValueError("Rotation shape/nonfinite")
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = np.array([[Qxx-Qyy-Qzz, 0, 0, 0], [Qyx+Qxy, Qyy-Qxx-Qzz, 0, 0],
                  [Qzx+Qxz, Qzy+Qyz, Qzz-Qxx-Qyy, 0],
                  [Qyz-Qzy, Qzx-Qxz, Qxy-Qyx, Qxx+Qyy+Qzz]])/3.
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def forward_native_mapping(command, root, bias):
    """The bound embodiment adapter is identity, then SAPIEN stores float32.

    This is only a mapping emulation. The native CPU probe additionally executes
    original source AST and checks actual Pose storage; no wrapper is modified.
    """
    cmd = np.asarray(command, dtype=np.float64)
    root = np.asarray(root, dtype=np.float32)
    if cmd.shape != (7,) or root.shape != (7,) or not np.isfinite(cmd).all() or not np.isfinite(root).all():
        raise ValueError("Invalid command/root")
    # Robot._trans_from_end_link_to_gripper always quat2mat->mat2quat even
    # when .12-gripper_bias=0 and inv_delta=I.
    stored = np.r_[cmd[:3], native_mat2quat(native_quat2mat(cmd[3:]))].astype(np.float32)
    wrb = native_quat2mat(root[3:])
    wrt = native_quat2mat(stored[3:])
    p = wrb.T @ (stored[:3]-root[:3])  # subtraction occurs in float32 in source
    q = native_mat2quat(wrb.T @ wrt)
    p += np.asarray(bias, dtype=np.float64)
    return p, q, stored


def build_targets(package):
    package = Path(package)
    d, inputs = package/"design", package/"design/orientation_inputs"
    if digest(inputs/"orientation_proposal.json") != PROPOSAL_SHA or digest(d/"integration_design.json") != INTEGRATION_SHA:
        raise ValueError("Frozen orientation proposal/integration changed")
    proposal = read(inputs/"orientation_proposal.json")
    history = read(inputs/"previous_solver_result_summary.json")
    if (history["status"] != "completed_verified_diagnostic" or history["solver_calls"] != 12
            or history["counts"] != dict(constraint_evaluations=12, fk_evaluations=12, solve_attempts=12)
            or any(history[k] for k in ("scenes", "native_initializations", "experts", "replays", "automatic_retries"))):
        raise ValueError("Previous budget/history must remain complete and exhausted")
    request = read(d/"captured_request.json")
    fields = request["scalar_evidence"]["fields"]
    root = np.array(fields["world_base_pose"]["value"], dtype=np.float32)
    bias = np.array(request["right_solver_context"]["fields"]["frame_bias"]["value"])
    configs = [x for x in read(inputs/"runtime_configs.json") if x["path"].endswith("/aloha-agilex/config.yml")]
    if len(configs) != 1:
        raise ValueError("Ambiguous embodiment config")
    config = configs[0]
    if hashlib.sha256(config["text"].encode()).hexdigest() != config["sha256"]:
        raise ValueError("Embodiment bytes drift")
    cfg = yaml.safe_load(config["text"])
    if cfg["gripper_bias"] != .12 or cfg["delta_matrix"] != np.eye(3).astype(int).tolist():
        raise ValueError("Different robot adapter; no inferred transfer")
    attempt = read(inputs/"expert_attempt.json")
    state = read(inputs/"expert_final_physical_state.json")
    if attempt["seed"] != 10050 or attempt["planner_success"] is not False or attempt["native_success"] is not None:
        raise ValueError("Wrong historical failure")
    actors = [x for x in state["actors"] if x["name"] == "048_stapler"]
    bodies = [x for x in state["articulations"] if x.get("robot_arm") is not None]
    if len(actors) != 1 or len(bodies) != 1:
        raise ValueError("Ambiguous retained physical entities")
    if not np.array_equal(bodies[0]["qpos"], fields["curr_joint_pos"]["value"]):
        raise ValueError("Reference physical joint state differs")
    sample = np.array(attempt["samples"][-1]["right"]["pose"])
    b = quaternion(sample[3:]).T @ (np.array(actors[0]["p"])-sample[:3])
    if not np.allclose(b, proposal["local_object_center_from_retained_state_m"], atol=1e-14, rtol=0):
        raise ValueError("Reference point cannot be reconstructed")
    joints = {j.get("name"): j for j in ET.parse(d/"source/1_arx5_description_isaac.urdf").getroot().findall("joint")}
    world_root = pose(root[:3].astype(float), quaternion(root[3:].astype(float)))
    world_base = world_root @ origin(joints["fr_base_joint"])
    previous_targets = read(inputs/"previous_targets.json")
    targets = previous_targets[:2]
    expected_ids = ["original", "local_y_minus20", "local_y_plus20", "local_z_minus20", "local_z_plus20"]
    if [c["candidate_id"] for c in proposal["candidates"]] != expected_ids:
        raise ValueError("Fixed orientation grid changed")
    mappings = []
    for candidate in proposal["candidates"]:
        if [s["stage"] for s in candidate["stages"]] != ["preplace", "place"]:
            raise ValueError("Fixed stage order changed")
        for stage in candidate["stages"]:
            intended = np.array(stage["world_native_eef_xyz_wxyz"])
            intended_world = pose(intended[:3], quaternion(intended[3:]))
            exact = np.array(stage["primary_diagnostic_target_full_urdf_base"])
            if not np.allclose(world_base@exact, intended_world, atol=1e-12, rtol=0):
                raise ValueError("Analytical full-URDF target mismatch")
            # Native wrapper ignores URDF base yaw. Precompensate only the input
            # command; do not edit native wrapper, table, robot, or source cfg.
            cmd_p = root[:3].astype(float)+world_root[:3, :3]@(exact[:3, 3]-bias)
            cmd_r = world_root[:3, :3]@exact[:3, :3]
            command = np.r_[cmd_p, native_mat2quat(cmd_r)]
            target_p, target_q, stored = forward_native_mapping(command, root, bias)
            target_r = quaternion(target_q)
            realized = world_base@pose(target_p, target_r)
            object_goal = np.array(stage["intended_object_center_world_m"])
            p_error = float(np.linalg.norm(realized[:3, 3]-intended[:3]))
            r_error = angular_error(realized[:3, :3], intended_world[:3, :3])
            object_error = float(np.linalg.norm(realized[:3, 3]+realized[:3, :3]@b-object_goal))
            if max(p_error, r_error, object_error) > 1e-6:
                raise ValueError("Source mapping fails physical-world point preservation")
            direct_p, direct_q, _ = forward_native_mapping(intended, root, bias)
            direct_world = world_base@pose(direct_p, quaternion(direct_q))
            item = dict(
                candidate_id=candidate["candidate_id"], stage=stage["stage"],
                intended_world_eef=stage["world_native_eef_xyz_wxyz"],
                native_compensated_command_xyz_wxyz=command.tolist(), native_stored_pose_xyz_wxyz=stored.tolist(),
                actual_source_wrapper_target_position_m=target_p.tolist(), actual_source_wrapper_target_quaternion_wxyz=target_q.tolist(),
                analytical_full_urdf_target=exact.tolist(), world_from_actual_native_mapping=realized.tolist(),
                world_from_direct_uncompensated_command=direct_world.tolist(),
                compensated_world_position_error_m=p_error, compensated_world_rotation_error_rad=r_error,
                compensated_reference_object_point_error_m=object_error,
                direct_uncompensated_object_point_error_m=float(np.linalg.norm(direct_world[:3, 3]+direct_world[:3, :3]@b-object_goal)),
                commanded_position_change_m=float(np.linalg.norm(command[:3]-intended[:3])),
                commanded_rotation_change_rad=angular_error(quaternion(command[3:]), intended_world[:3, :3]),
                world_base_transform=world_base.tolist(), local_object_reference_point_m=b.tolist(),
                intended_object_center_world_m=object_goal.tolist(),
                reference_grasp_rigid_assumption=True, evidence_is_native_task_label=False,
            )
            mappings.append(item)
            targets.append(dict(id=candidate["candidate_id"]+"_"+stage["stage"], position_m=target_p.tolist(),
                                rotation_matrix=target_r.tolist(), recorded_quaternion_wxyz=target_q.tolist(),
                                start_joints_rad=previous_targets[2]["start_joints_rad"],
                                source="Fixed reference point preserved by explicit native-command precompensation; no native call executed",
                                reference_geometry=item))
    return targets, dict(version=1, proposal_sha256=PROPOSAL_SHA, integration_sha256=INTEGRATION_SHA,
                         mode="CPU_native_mapping_emulation_bound_to_source_AST_probe", mappings=mappings,
                         native_calls=0, solver_calls=0, scenes=0)


def verify_targets(package, targets):
    expected, mapping = build_targets(package)
    if len(targets) != 12 or len({t["id"] for t in targets}) != 12:
        raise ValueError("Exactly12 unique targets required")
    # target bytes themselves are hash-bound by contracts. Native NumPy1.26 is
    # the generation authority; an offline NumPy2.x checker can differ at eigensolver
    # rounding, so numeric recomputation has an explicit sub-nanometre bound.
    def equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, float):
            return np.isfinite(a) and np.isfinite(b) and abs(a-b) < 1e-11
        return a == b
    if not equal(targets, expected):
        raise ValueError("Frozen orientation targets not derived from complete bound reference")
    stored = read(Path(package)/"design/native_mapping.json")
    if not equal(stored, mapping):
        raise ValueError("Native mapping audit differs")
    return dict(targets=12, reference_geometry_targets=10, maximum_point_error_m=max(
        x["compensated_reference_object_point_error_m"] for x in mapping["mappings"]),
        full_urdf_primary_and_native_precompensation_explicit=True, production_source_modified=False)


def analyze_world(target, fk_transforms):
    if "reference_geometry" not in target:
        return {"role": "kinematic_control_not_native_task_label"}
    g = target["reference_geometry"]
    world = np.array(g["world_base_transform"])[None]@np.asarray(fk_transforms)
    b = np.array(g["local_object_reference_point_m"])
    points = world[:, :3, 3]+world[:, :3, :3]@b
    goal = np.array(g["intended_object_center_world_m"])
    return dict(role="reference_rigid_point_diagnostic_not_task_label", rigid_grasp_assumed=True,
                local_point_m=b.tolist(), intended_world_point_m=goal.tolist(),
                all32_predicted_object_points_world_m=points.tolist(),
                all32_predicted_object_point_error_m=np.linalg.norm(points-goal, axis=1).tolist(),
                no_object_retention_or_native_success_inferred=True)
