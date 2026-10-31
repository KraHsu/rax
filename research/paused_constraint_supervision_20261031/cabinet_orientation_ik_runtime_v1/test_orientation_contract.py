"""Real-reference mapping and bounded new-package integration, CPU only."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

B = Path(__file__).parent
P = B/"package"
sys.path.insert(0, str(P/"runtime"))
import orientation_contract as oc
import contracts


def test_complete_reference_native_mapping_not_implicit_frame_switch():
    targets, mapping = oc.build_targets(P)
    result = oc.verify_targets(P, oc.read(P/"design/targets.json"))
    assert result["targets"] == 12 and result["reference_geometry_targets"] == 10
    assert result["maximum_point_error_m"] < 1e-6
    assert len({t["id"] for t in targets}) == 12
    for m in mapping["mappings"]:
        assert m["direct_uncompensated_object_point_error_m"] > .005
        assert m["compensated_reference_object_point_error_m"] < 1e-6
        assert abs(m["commanded_rotation_change_rad"]-.01) < 1e-6
        assert np.linalg.norm(m["local_object_reference_point_m"]) > .15
        assert not m["evidence_is_native_task_label"]


def test_controls_identical_and_all12_self_table_enabled():
    design, targets, _ = contracts.inputs(P, False)
    previous = oc.read(P/"design/orientation_inputs/previous_targets.json")
    for old in previous[:2]:
        actual = dict(targets[old["id"]]); actual.pop("runtime_quaternion_wxyz")
        assert actual == old
    assert len(design["modes"]) == 1
    mode = design["modes"][0]
    assert mode["self_collision_opt"] is True and mode["self_collision_check"] is True
    assert mode["world_model"]["cuboid"]["table"]["dims"] == [.7, 2., .04]
    assert {x["mode"] for x in design["slots"]} == {"source_table_and_self"}
    assert [x["target"] for x in design["slots"]] == list(targets)


@pytest.mark.parametrize("mutation", ["source_root", "physical_q", "point", "adapter", "stage"])
def test_reference_tamper_cannot_change_mapping_silently(tmp_path, mutation):
    p = tmp_path/"package"
    shutil.copytree(P, p)
    if mutation == "source_root":
        path = p/"design/captured_request.json"; x = oc.read(path)
        x["scalar_evidence"]["fields"]["world_base_pose"]["value"][0] += .01
    elif mutation == "physical_q":
        path = p/"design/orientation_inputs/expert_final_physical_state.json"; x = oc.read(path)
        body = next(b for b in x["articulations"] if b.get("robot_arm") is not None)
        body["qpos"][7] += .1
    elif mutation == "point":
        path = p/"design/orientation_inputs/expert_final_physical_state.json"; x = oc.read(path)
        actor = next(a for a in x["actors"] if a["name"] == "048_stapler")
        actor["p"][0] += .01
    elif mutation == "adapter":
        path = p/"design/orientation_inputs/runtime_configs.json"; x = oc.read(path)
        next(a for a in x if a["path"].endswith("/aloha-agilex/config.yml"))["text"] += "\ngripper_bias: .13\n"
    else:
        path = p/"design/orientation_inputs/orientation_proposal.json"; x = oc.read(path)
        x["candidates"][1]["stages"] = list(reversed(x["candidates"][1]["stages"]))
    path.write_text(json.dumps(x))
    with pytest.raises(ValueError):
        oc.build_targets(p)


def test_target_and_mapping_audit_tamper_rejected(tmp_path):
    targets = oc.read(P/"design/targets.json")
    targets[2]["reference_geometry"]["native_compensated_command_xyz_wxyz"][0] += .001
    with pytest.raises(ValueError, match="derived"):
        oc.verify_targets(P, targets)
    p = tmp_path/"package"; shutil.copytree(P, p)
    path = p/"design/native_mapping.json"; x = oc.read(path)
    x["mappings"][0]["direct_uncompensated_object_point_error_m"] = 0.
    path.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="audit"):
        oc.verify_targets(p, oc.read(P/"design/targets.json"))


def test_rigid_point_report_uses_all32_fk_not_only_reported_success():
    _, targets, _ = contracts.inputs(P, False)
    target = targets["local_y_plus20_place"]
    T = np.repeat(np.eye(4)[None], 32, axis=0)
    T[:, :3, :3] = target["rotation_matrix"]
    T[:, :3, 3] = target["position_m"]
    T[31, 0, 3] += .02
    report = oc.analyze_world(target, T)
    assert len(report["all32_predicted_object_point_error_m"]) == 32
    assert report["all32_predicted_object_point_error_m"][0] < 1e-6
    assert report["all32_predicted_object_point_error_m"][31] > .019
    assert report["no_object_retention_or_native_success_inferred"] is True


def test_original_guarded_backend_and_owner_lifecycle_unchanged():
    old = B.parent/"cabinet_ik_feasibility_runtime_v1/package"
    for name in ("driver.py", "gpu_backend.py", "geometry.py", "source_audit.py"):
        assert (P/"runtime"/name).read_bytes() == (old/"runtime"/name).read_bytes()
    oldowner = (old/"owner/ik_owner.py").read_text()
    newowner = (P/"owner/ik_owner.py").read_text()
    assert newowner[newowner.index("def run("):] == oldowner[oldowner.index("def run("):]


def test_native_float32_quaternion_norm_is_not_silently_upcast_first():
    # Fixed root witness exposes NumPy scalar-promotion and norm-rounding changes.
    q = np.array([.7070000171661377, 0, 0, .7070000171661377], dtype=np.float32)
    actual = oc.native_quat2mat(q)
    wrong = oc.native_quat2mat(q.astype(np.float64))
    assert np.max(np.abs(actual-wrong)) > 1e-9
    assert np.isfinite(actual).all()
