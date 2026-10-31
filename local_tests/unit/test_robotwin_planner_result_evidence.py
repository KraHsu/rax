"""CPU source-AST probes, never cuRobo import/Scene/device/solver execution."""

import ast
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import transforms3d as t3d

from services.robotwin import planner_result_evidence as evidence
from services.robotwin import expert_planner_observer as planner
from scripts.observe_robotwin_expert_failure import validate_planner_receipt
from test_robotwin_failure_observation import planner_native, Pose

FIX = Path(__file__).parents[1] / "fixtures/robotwin_planner_result"


def source_classes(path, names, monkeypatch, **scope):
    module = ModuleType("solver_cpu_" + path.stem)
    module.__file__ = str(path)
    module.__dict__.update(scope)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for name, methods in names.items():
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name
        )
        if name == "MotionGen":
            node.bases = []
        if methods is not None:
            # Keep exact data fields and selected unmodified methods. Never execute imports or constructors of MotionGen/CuroboPlanner.
            node.body = [
                n
                for n in node.body
                if isinstance(n, ast.AnnAssign) or getattr(n, "name", None) in methods
            ]
        nodes.append(node)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


def environment(monkeypatch, tmp_path, *, status="TRAJOPT_FAIL", success=False):
    assert evidence.file_sha(FIX / "planner.py") == evidence.PLANNER_SHA
    assert evidence.file_sha(FIX / "motion_gen.py") == evidence.MOTION_SHA
    calls = []

    class Tensor:
        def __init__(self, value):
            self.value = value

        def cuda(self):
            calls.append("original.cuda")
            return self

        def reshape(self, *shape):
            calls.append(("original.reshape", shape))
            return self

        def item(self):
            calls.append("original.item")
            return self.value

        def to(self, device):
            calls.append(("original.to", device))
            return self.value

    def tensor(x):
        calls.append(("original.tensor", list(x)))
        return Tensor(x)

    def goal(x):
        calls.append(("original.goal", list(x)))
        return SimpleNamespace(batch=1, value=x)

    def joint(x, **k):
        calls.append(("original.joint", x.value, deepcopy(k)))
        return x

    mg = source_classes(
        FIX / "motion_gen.py",
        {
            "MotionGenPlanConfig": {"__post_init__"},
            "MotionGenStatus": None,
            "MotionGenResult": set(),
            "MotionGen": {"plan_single"},
        },
        monkeypatch,
        dataclass=dataclass,
        Enum=Enum,
        log_info=lambda *x: None,
        ReacherSolveType=SimpleNamespace(SINGLE=1),
        log_error=lambda *x: None,
    )
    cls = mg.MotionGenResult
    state = (
        mg.MotionGenStatus.__members__[status]
        if status in evidence.STATUS_MEMBERS
        else status
    )
    result = cls(
        success=Tensor(success),
        status=state,
        attempts=3,
        interpolated_plan=SimpleNamespace(
            position=Tensor(np.ones((1, 2))), velocity=Tensor(np.zeros((1, 2)))
        ),
    )
    mod = source_classes(
        FIX / "planner.py",
        {"CuroboPlanner": {"plan_path", "_trans_from_world_to_base"}},
        monkeypatch,
        np=np,
        t3d=t3d,
        MotionGen=mg.MotionGen,
        MotionGenPlanConfig=mg.MotionGenPlanConfig,
        torch=SimpleNamespace(tensor=tensor),
        JointState=SimpleNamespace(from_position=joint),
        CuroboPose=SimpleNamespace(from_list=goal),
        time=SimpleNamespace(time=lambda: 10.0),
    )
    n = planner_native(monkeypatch)
    n.robot._trans_from_end_link_to_gripper = lambda p, arm_tag: Pose(p[:3], p[3:])
    config = tmp_path / "config.yml"
    config.write_text("planner:\n  frame_bias: [0.01, 0.02, 0.03]\n", encoding="utf-8")
    for arm in ("left", "right"):
        w = mod.CuroboPlanner()
        w.robot_origion_pose = Pose([0.1, 0.2, 0.3], [1, 0, 0, 0])
        w.active_joints_name = ["b", "a"]
        w.all_joints = ["a", "b"]
        w.frame_bias = [0.01, 0.02, 0.03]
        w.yml_path = str(config)
        m = mg.MotionGen()
        m.reset = lambda **k: calls.append(("original.reset", k))
        m._get_solve_state = lambda *a: 1

        def solve(*args, **kwargs):
            calls.append(("original.plan_single", args[0]))
            return result

        m._plan_attempts = solve
        w.motion_gen = m
        setattr(n.robot, arm + "_planner", w)
        getattr(n.robot, arm + "_entity").qpos = np.array(
            [0.123456789, 0.23456789], dtype=np.float32
        )
        getattr(n.robot, arm + "_entity").get_qpos = lambda: np.array(
            [0.123456789, 0.23456789], dtype=np.float32
        )
    return n, result, calls, mg


def observe(n):
    obs = planner.ExpertPlannerObserver(
        n, frozen_policy=planner.policy(solver_details=True)
    ).start()
    try:
        out = n.right_move_to_pose([0.2, -0.1, 0.9, 1, 0, 0, 0])
    finally:
        obs.stop()
    receipt = obs.receipt()
    validate_planner_receipt(receipt, receipt["policy"])
    return out, receipt


@pytest.mark.parametrize("success", [False, True])
def test_original_AST_args_returns_and_device_call_counts_unchanged(
    monkeypatch, tmp_path, success
):
    n, result, calls, _ = environment(
        monkeypatch,
        tmp_path,
        success=success,
        status="SUCCESS" if success else "TRAJOPT_FAIL",
    )
    baseline = n.robot.right_plan_path([0.2, -0.1, 0.9, 1, 0, 0, 0])
    old = deepcopy(calls)
    calls.clear()
    n.plan_success = True
    # Capture return object identity from source native wrapper without additional return conversion.
    identities = []
    fn = n.robot.right_planner.plan_path.__func__
    obs = planner.ExpertPlannerObserver(
        n, frozen_policy=planner.policy(solver_details=True)
    )
    original = obs._append

    def append(row):
        if row["event"] == "return_or_unwind" and row["source"] == "curobo_wrapper":
            identities.append(row["solver_evidence"])
        original(row)

    obs._append = append
    obs.start()
    try:
        out = n.robot.right_plan_path([0.2, -0.1, 0.9, 1, 0, 0, 0])
    finally:
        obs.stop()
    assert calls == old and n.robot.right_planner.plan_path.__func__ is fn
    assert out["status"] == baseline["status"] and n.plan_success is True
    assert calls.count("original.item") == 1
    receipt = obs.receipt()
    assert receipt["owner_thread_scope_complete"]
    validate_planner_receipt(receipt, receipt["policy"])
    r = identities[0]
    assert (
        r["result_attempts"]["value"] == 3
        and r["plan_config"]["max_attempts"]["value"] == 10
    )
    assert r["plan_config"]["timeout"]["value"] == 10.0 and r["result_status"][
        "name"
    ] == ("SUCCESS" if success else "TRAJOPT_FAIL")
    assert r["parent_request_call_id"] == 1 and r["call_id"] == 2
    assert sys.getprofile() is None
    assert all(
        r[k] == 0
        for k in ("added_tensor_reads", "added_solver_calls", "added_control_calls")
    )


@pytest.mark.parametrize(
    "status", [*evidence.STATUS_MEMBERS, "custom returned status", None]
)
def test_each_audited_enum_exact_string_and_none_retained(
    monkeypatch, tmp_path, status
):
    n, result, _, _ = environment(monkeypatch, tmp_path, status=status)
    _, r = observe(n)
    assert r["owner_thread_scope_complete"]
    value = next(e["solver_evidence"] for e in r["events"] if "solver_evidence" in e)
    assert value["result_status"]["observed"] is True
    assert value["reported_category"] == (
        status if status in evidence.STATUS_MEMBERS else None
    )


class Dangerous:
    def __getattribute__(self, key):
        raise AssertionError("unknown getattr " + key)

    def __str__(self):
        raise AssertionError("str")

    def __repr__(self):
        raise AssertionError("repr")

    def __array__(self, *args):
        raise AssertionError("array")

    def item(self):
        raise AssertionError("item")

    def cpu(self):
        raise AssertionError("cpu")


@pytest.mark.parametrize(
    "change",
    [
        "unknown_status",
        "unknown_result",
        "unknown_attempts",
        "nonfinite_time",
        "missing_result",
        "config_unknown",
    ],
)
def test_unknown_missing_values_stay_incomplete_without_coercion(
    monkeypatch, tmp_path, change
):
    n, result, _, _ = environment(monkeypatch, tmp_path)
    obs = planner.ExpertPlannerObserver(
        n, frozen_policy=planner.policy(solver_details=True)
    )
    if change == "unknown_status":
        result.status = Dangerous()
    elif change == "unknown_attempts":
        result.attempts = Dangerous()
    elif change == "unknown_result":
        # Native original .success is allowed, new evidence must not access unknown .status.
        class Unknown:
            success = result.success

            @property
            def status(self):
                raise AssertionError("unknown property")

        n.robot.right_planner.motion_gen._plan_attempts = lambda *a, **k: Unknown()
    elif change == "missing_result":
        sentinel = RuntimeError("original solver exception")

        def boom(*a, **k):
            raise sentinel

        n.robot.right_planner.motion_gen._plan_attempts = boom
    elif change == "config_unknown":
        # Mutate known config stored value in native stub, never call arbitrary descriptors.
        old = n.robot.right_planner.motion_gen._plan_attempts

        def changeconfig(*a, **k):
            a[3].timeout = Dangerous()
            return old(*a, **k)

        n.robot.right_planner.motion_gen._plan_attempts = changeconfig
    else:
        gl = n.robot.right_planner.plan_path.__func__.__globals__
        gl["time"] = SimpleNamespace(time=lambda: float("nan"))
    obs.start()
    try:
        if change == "missing_result":
            with pytest.raises(
                RuntimeError, match="original solver exception"
            ) as caught:
                n.right_move_to_pose([0.2, -0.1, 0.9, 1, 0, 0, 0])
            assert caught.value is sentinel
        else:
            assert n.right_move_to_pose([0.2, -0.1, 0.9, 1, 0, 0, 0]) is None
    finally:
        obs.stop()
    r = obs.receipt()
    assert not r["owner_thread_scope_complete"]
    validate_planner_receipt(r, r["policy"])
    assert sys.getprofile() is None


@pytest.mark.parametrize(
    "kind",
    [
        "cycle",
        "nested",
        "array",
        "custom",
        "largeint",
        "longstr",
        "2d",
        "object",
        "nan",
        "extended",
    ],
)
def test_cpu_formatter_rejects_without_recursive_or_unknown_execution(kind):
    if kind == "cycle":
        value = []
        value.append(value)
    elif kind == "nested":
        value = [[1]]
    elif kind == "array":
        value = [np.array([1])]
    elif kind == "custom":
        value = Dangerous()
    elif kind == "largeint":
        value = 1 << 129
    elif kind == "longstr":
        value = "x" * 257
    elif kind == "2d":
        value = np.zeros((1, 1))
    elif kind == "object":
        value = np.array([1], dtype=object)
    elif kind == "extended":
        value = np.array([1], dtype=np.longdouble)
    else:
        value = float("nan")
    assert evidence.cpu_field(value, "test")["observed"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "rowhash",
        "category",
        "calls",
        "context",
        "missing",
        "complete",
        "dtype",
        "rawhash",
        "enum",
        "config",
        "parent",
        "remove_solver",
    ],
)
def test_rehashed_tamper_is_rejected(monkeypatch, tmp_path, kind):
    n, _, _, _ = environment(monkeypatch, tmp_path)
    _, r = observe(n)
    row = next(e for e in r["events"] if "solver_evidence" in e)
    v = row["solver_evidence"]
    if kind == "rowhash":
        v["sha256"] = "0" * 64
    elif kind == "category":
        v["reported_category"] = "IK_FAIL"
    elif kind == "calls":
        v["added_solver_calls"] = 1
    elif kind == "context":
        r["solver_contexts"]["right"]["yaml"]["text"] += "x"
    elif kind == "missing":
        del v["fields"]["target_pose_p"]
    elif kind == "complete":
        v["complete"] = False
    elif kind == "dtype":
        v["fields"]["world_target_pose"]["dtype"] = "object"
    elif kind == "rawhash":
        v["fields"]["world_target_pose"]["sha256"] = "0" * 64
    elif kind == "enum":
        v["result_status"]["value"] = "Something else"
    elif kind == "config":
        v["plan_config"]["max_attempts"]["type"] = "bool"
    elif kind == "parent":
        v["parent_request_call_id"] = 1
    else:
        del row["solver_evidence"]
    if kind not in ("rowhash", "remove_solver"):
        v["sha256"] = evidence.digest({k: x for k, x in v.items() if k != "sha256"})
    with pytest.raises((ValueError, KeyError)):
        validate_planner_receipt(r, r["policy"])


def test_source_policy_optin_and_config_limit(monkeypatch, tmp_path):
    n, _, _, _ = environment(monkeypatch, tmp_path)
    assert "solver_details" not in planner.policy()
    for flag in (None, 1, "true"):
        with pytest.raises(ValueError):
            planner.policy(solver_details=flag)
    file = Path(n.robot.right_planner.yml_path)
    file.write_text("x" * (evidence.MAX_CONFIG_BYTES + 1))
    _, r = observe(n)
    assert not r["owner_thread_scope_complete"]
    n.robot.right_planner.plan_path = lambda *a: None
    with pytest.raises(ValueError):
        planner.ExpertPlannerObserver(
            n, frozen_policy=planner.policy(solver_details=True)
        )


@pytest.mark.parametrize("change", ["index", "rounded", "names"])
def test_cpu_joint_mapping_rederived_after_rehash(monkeypatch, tmp_path, change):
    n, _, _, _ = environment(monkeypatch, tmp_path)
    _, r = observe(n)
    v = next(e["solver_evidence"] for e in r["events"] if "solver_evidence" in e)
    c = r["solver_contexts"]["right"]
    if change == "index":
        v["fields"]["joint_indices"] = evidence.cpu_field(
            [0, 1], "plan_path.local.joint_indices"
        )
    elif change == "rounded":
        v["fields"]["joint_angles"] = evidence.cpu_field(
            [0.3, 0.2], "plan_path.local.joint_angles"
        )
    else:
        c["fields"]["active_joints_name"] = evidence.cpu_field(
            ["a", "b"], "CuroboPlanner.__dict__.active_joints_name"
        )
    c["sha256"] = evidence.digest({k: x for k, x in c.items() if k != "sha256"})
    v["context_sha256"] = c["sha256"]
    v["sha256"] = evidence.digest({k: x for k, x in v.items() if k != "sha256"})
    with pytest.raises(ValueError, match="Joint|Rounded"):
        validate_planner_receipt(r, r["policy"])


def test_profile_return_identity_and_no_additional_control(monkeypatch, tmp_path):
    n, _, calls, _ = environment(monkeypatch, tmp_path)
    seen = []

    class Observer(planner.ExpertPlannerObserver):
        def profile(self, frame, event, arg):
            identity = self.identities.get(
                (frame.f_code, id(frame.f_locals.get("self")))
            )
            if (
                event == "return"
                and identity is not None
                and identity[0] in ("plan_path", "right_plan_path")
            ):
                seen.append(arg)
            super().profile(frame, event, arg)

    o = Observer(n, frozen_policy=planner.policy(solver_details=True)).start()
    try:
        out = n.robot.right_plan_path([0.2, -0.1, 0.9, 1, 0, 0, 0])
    finally:
        o.stop()
    assert len(seen) == 2 and all(value is out for value in seen)
    assert calls.count("original.item") == 1
    validate_planner_receipt(o.receipt(), o.policy)


def test_source_boundaries_unknown_properties_and_process_mode(monkeypatch, tmp_path):
    n, result, _, mg = environment(monkeypatch, tmp_path)

    def forbidden(*a):
        raise AssertionError("result Tensor/property read")

    mg.MotionGenResult.motion_time = property(forbidden)
    # All these preexisting device fields are deliberately outside the whitelist.
    result.optimized_dt = Dangerous()
    result.solve_time = Dangerous()
    _, r = observe(n)
    assert r["owner_thread_scope_complete"]
    n.robot.communication_flag = True
    with pytest.raises(ValueError, match="owner-process"):
        planner.ExpertPlannerObserver(
            n, frozen_policy=planner.policy(solver_details=True)
        )


def test_event_budget_incomplete_and_profile_cleanup(monkeypatch, tmp_path):
    n, _, _, _ = environment(monkeypatch, tmp_path)
    o = planner.ExpertPlannerObserver(
        n, frozen_policy=planner.policy(solver_details=True)
    )
    # Fill the declared budget before observing; no second episode or solver retry.
    o.events = [{}] * o.policy["max_events"]
    previous = sys.getprofile()
    o.start()
    try:
        assert n.robot.right_plan_path([0.2, -0.1, 0.9, 1, 0, 0, 0])["status"] == "Fail"
    finally:
        o.stop()
    assert (
        len(o.events) == o.policy["max_events"] and "event_budget_exhausted" in o.errors
    )
    assert (
        sys.getprofile() is previous and not o.receipt()["owner_thread_scope_complete"]
    )
