"""Opt-in evidence from existing CPU locals at the native scalar planner return.

No solver, control, descriptor, Tensor method or pose getter is invoked here.
"""

import hashlib
import inspect
import json
import math
from pathlib import Path
from types import GetSetDescriptorType

import numpy as np

VERSION = "curobo_scalar_return_cpu_evidence_v1"
PLANNER_SHA = "178a72a7b6ededee66e7f78ebc35e7c39c4a855a1d713b90a01a32868d49f24a"
MOTION_SHA = "348ee2fa858b1d651a70fa2b2b5a5d4e240ea724b7817d7769473c3f7a7e46e8"
MAX_ITEMS = 128
MAX_TEXT = 256
MAX_CONFIG_BYTES = 131072
STATUS_MEMBERS = {
    "IK_FAIL": "IK Fail",
    "GRAPH_FAIL": "Graph Fail",
    "TRAJOPT_FAIL": "TrajOpt Fail",
    "FINETUNE_TRAJOPT_FAIL": "Finetune TrajOpt Fail",
    "DT_EXCEPTION": "dt exceeded maximum allowed trajectory dt",
    "INVALID_QUERY": "Invalid Query",
    "INVALID_START_STATE_UNKNOWN_ISSUE": "Invalid Start State, unknown issue",
    "INVALID_START_STATE_WORLD_COLLISION": "Start state is colliding with world",
    "INVALID_START_STATE_SELF_COLLISION": "Start state is in self-collision",
    "INVALID_START_STATE_JOINT_LIMITS": "Start state is out of joint limits",
    "INVALID_PARTIAL_POSE_COST_METRIC": "Invalid partial pose metric",
    "SUCCESS": "Success",
    "NOT_ATTEMPTED": "Not Attempted",
}
LOCAL_FIELDS = (
    "curr_joint_pos",
    "world_base_pose",
    "world_target_pose",
    "target_pose_p",
    "target_pose_q",
    "joint_indices",
    "joint_angles",
    "constraint_pose",
    "c_time",
)
CONFIG_FIELDS = (
    "max_attempts",
    "timeout",
    "enable_graph",
    "enable_opt",
    "enable_graph_attempt",
    "disable_graph_attempt",
    "ik_fail_return",
    "num_ik_seeds",
    "num_graph_seeds",
    "num_trajopt_seeds",
    "fail_on_invalid_query",
    "check_start_validity",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy():
    return dict(
        version=VERSION,
        implementation_sha256=file_sha(__file__),
        native_planner_sha256=PLANNER_SHA,
        motion_gen_sha256=MOTION_SHA,
        max_items=MAX_ITEMS,
        max_text_codepoints=MAX_TEXT,
        max_config_bytes=MAX_CONFIG_BYTES,
        status_members=STATUS_MEMBERS,
        scope="Existing CPU locals and audited stored result/config fields at scalar plan_path return; no Tensor/solver access",
    )


def unsupported(reason, source):
    return dict(observed=False, reason=reason, source=source)


def cpu_field(value, source):
    """Exact finite CPU types only; never item/str/repr/coercion on unknowns."""
    t = type(value)
    if value is None:
        return dict(observed=True, type="none", value=None, source=source)
    if t is int and value.bit_length() > 128:
        return unsupported("integer_limit", source)
    if t in (bool, int):
        return dict(observed=True, type=t.__name__, value=value, source=source)
    if t is float or t in (np.float32, np.float64):
        if not math.isfinite(float(value)):
            return unsupported("nonfinite", source)
        return dict(observed=True, type=t.__name__, value=float(value), source=source)
    if t in (np.int32, np.int64):
        return dict(observed=True, type=t.__name__, value=int(value), source=source)
    if t is str:
        if len(value) > MAX_TEXT:
            return unsupported("string_limit", source)
        return dict(observed=True, type="str", value=value, source=source)
    if t is np.ndarray:
        if (
            value.ndim != 1
            or value.size > MAX_ITEMS
            or value.dtype.kind not in "biuf"
            or value.dtype.itemsize > 8
            or not np.isfinite(value).all()
        ):
            return unsupported("unsupported_CPU_array", source)
        return dict(
            observed=True,
            type="numpy",
            dtype=str(value.dtype),
            shape=list(value.shape),
            value=value.tolist(),
            sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            source=source,
        )
    if t in (list, tuple):
        if len(value) > MAX_ITEMS:
            return unsupported("sequence_limit", source)
        if any(
            type(x)
            not in (
                type(None),
                bool,
                int,
                float,
                str,
                np.float32,
                np.float64,
                np.int32,
                np.int64,
            )
            for x in value
        ):
            return unsupported("non_scalar_sequence", source)
        rows = [cpu_field(x, source) for x in value]
        if any(
            not r["observed"] or r["type"] in ("list", "tuple", "numpy") for r in rows
        ):
            return unsupported("non_scalar_sequence", source)
        return dict(
            observed=True,
            type=t.__name__,
            value=[r["value"] for r in rows],
            element_types=[r["type"] for r in rows],
            source=source,
        )
    return unsupported("unsupported_exact_type", source)


def stored(obj, cls):
    """Read the normal instance dictionary only after exact audited type match."""
    if type(obj) is not cls:
        return None
    descriptor = cls.__dict__.get("__dict__")
    if type(descriptor) is not GetSetDescriptorType:
        return None
    value = descriptor.__get__(obj, cls)
    return value if type(value) is dict else None


def _class(cls, name, sha):
    if (
        not isinstance(cls, type)
        or cls.__name__ != name
        or file_sha(inspect.getfile(cls)) != sha
    ):
        raise ValueError("Unaudited solver class: " + name)


class Bindings:
    def __init__(self, native):
        self.by_object = {}
        self.contexts = {}
        self.methods = []
        if native.robot.__dict__.get("communication_flag") is not False:
            raise ValueError(
                "Scalar observation requires source-bound owner-process planner"
            )
        for arm in ("left", "right"):
            wrapper = native.robot.__dict__[arm + "_planner"]
            cls = type(wrapper)
            _class(cls, "CuroboPlanner", PLANNER_SHA)
            fn = cls.__dict__.get("plan_path")
            if (
                getattr(wrapper.plan_path, "__func__", None) is not fn
                or file_sha(inspect.getfile(fn)) != PLANNER_SHA
            ):
                raise ValueError("Scalar wrapper method replaced")
            globals_ = fn.__globals__
            data = stored(wrapper, cls)
            if data is None:
                raise ValueError("Scalar wrapper has no audited stored configuration")
            mg = data.get("motion_gen")
            mgcls = globals_["MotionGen"]
            _class(mgcls, "MotionGen", MOTION_SHA)
            if type(mg) is not mgcls or getattr(
                mg.plan_single, "__func__", None
            ) is not mgcls.__dict__.get("plan_single"):
                raise ValueError("MotionGen binding replaced")
            scope = mgcls.plan_single.__globals__
            resultcls = scope["MotionGenResult"]
            enumcls = scope["MotionGenStatus"]
            configcls = globals_["MotionGenPlanConfig"]
            for c, n in [
                (resultcls, "MotionGenResult"),
                (enumcls, "MotionGenStatus"),
                (configcls, "MotionGenPlanConfig"),
            ]:
                _class(c, n, MOTION_SHA)
            members = enumcls.__dict__.get("_member_map_")
            if type(members) is not dict or set(members) != set(STATUS_MEMBERS):
                raise ValueError("Audited status enum changed")
            for name, member in members.items():
                # Enum has an inherited normal dictionary; inspect only this exact audited enum.
                raw = object.__getattribute__(member, "__dict__")
                if (
                    type(member) is not enumcls
                    or raw.get("_name_") != name
                    or raw.get("_value_") != STATUS_MEMBERS[name]
                ):
                    raise ValueError("Status enum mapping changed")
            fields = {
                key: cpu_field(data.get(key), "CuroboPlanner.__dict__." + key)
                if key in data
                else unsupported("missing", "CuroboPlanner.__dict__." + key)
                for key in ("active_joints_name", "all_joints", "frame_bias")
            }
            path = data.get("yml_path")
            yaml = unsupported("unsupported_path", "CuroboPlanner.__dict__.yml_path")
            if type(path) is str and len(path) <= 4096:
                try:
                    p = Path(path)
                    size = p.stat().st_size
                    if size > MAX_CONFIG_BYTES:
                        yaml = unsupported(
                            "config_byte_limit", "CuroboPlanner.__dict__.yml_path"
                        )
                    else:
                        with p.open("rb") as stream:
                            raw = stream.read(MAX_CONFIG_BYTES + 1)
                        if len(raw) > MAX_CONFIG_BYTES:
                            raise ValueError("config_byte_limit")
                        text = raw.decode("utf-8")
                        yaml = dict(
                            observed=True,
                            path=str(p.resolve()),
                            bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest(),
                            text=text,
                            source="existing runtime yml_path file before profile",
                        )
                except (OSError, UnicodeError, ValueError):
                    yaml = unsupported(
                        "config_unreadable", "CuroboPlanner.__dict__.yml_path"
                    )
            context = dict(
                arm=arm,
                fields=fields,
                yaml=yaml,
                planner_source_sha256=PLANNER_SHA,
                motion_source_sha256=MOTION_SHA,
            )
            context["sha256"] = digest(context)
            self.contexts[arm] = context
            self.by_object[id(wrapper)] = (arm, resultcls, enumcls, configcls)
            self.methods.append((wrapper, fn, arm))

    def collect(self, wrapper, locals_, call_id, parent_request_call_id):
        arm, resultcls, enumcls, configcls = self.by_object[id(wrapper)]
        fields = {
            key: cpu_field(locals_[key], "plan_path.local." + key)
            if key in locals_
            else unsupported("missing", "plan_path.local." + key)
            for key in LOCAL_FIELDS
        }
        config = stored(locals_.get("plan_config"), configcls)
        config_fields = {
            key: cpu_field(config[key], "MotionGenPlanConfig.__dict__." + key)
            if config is not None and key in config
            else unsupported(
                "missing_or_unrecognized_config", "MotionGenPlanConfig.__dict__." + key
            )
            for key in CONFIG_FIELDS
        }
        result = stored(locals_.get("result"), resultcls)
        status = unsupported(
            "missing_or_unrecognized_result", "MotionGenResult.__dict__.status"
        )
        attempts = unsupported(
            "missing_or_unrecognized_result", "MotionGenResult.__dict__.attempts"
        )
        if result is not None:
            if "attempts" in result:
                attempts = (
                    cpu_field(result["attempts"], "MotionGenResult.__dict__.attempts")
                    if type(result["attempts"]) is int
                    and 0 <= result["attempts"] <= 100000
                    else unsupported(
                        "unsupported_attempt_count", "MotionGenResult.__dict__.attempts"
                    )
                )
            if "status" in result:
                value = result["status"]
                if type(value) is enumcls:
                    raw = object.__getattribute__(value, "__dict__")
                    name = raw.get("_name_")
                    text = raw.get("_value_")
                    if (
                        type(name) is str
                        and type(text) is str
                        and STATUS_MEMBERS.get(name) == text
                    ):
                        status = dict(
                            observed=True,
                            type="MotionGenStatus",
                            name=name,
                            value=text,
                            source="MotionGenResult.__dict__.status",
                        )
                elif value is None or type(value) is str:
                    status = cpu_field(value, "MotionGenResult.__dict__.status")
                else:
                    status = unsupported(
                        "unsupported_status_exact_type",
                        "MotionGenResult.__dict__.status",
                    )
        context = self.contexts[arm]
        observed = [
            *fields.values(),
            *config_fields.values(),
            status,
            attempts,
            *context["fields"].values(),
            context["yaml"],
        ]
        row = dict(
            version=VERSION,
            arm=arm,
            call_id=call_id,
            parent_request_call_id=parent_request_call_id,
            context_sha256=context["sha256"],
            fields=fields,
            plan_config=config_fields,
            result_status=status,
            result_attempts=attempts,
            complete=all(x["observed"] for x in observed),
            reported_category=status.get("name")
            if status.get("type") == "MotionGenStatus"
            else None,
            interpretation="Reported solver category only; no proof of physical impossibility or timeout. No TIMEOUT enum exists in the audited source.",
            added_tensor_reads=0,
            added_solver_calls=0,
            added_control_calls=0,
        )
        row["sha256"] = digest(row)
        return row


def _hash(value):
    if type(value) is not dict or value.get("sha256") != digest(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("Solver evidence content hash mismatch")


def validate_cpu_field(field, source):
    """Decode bounded, finite retained CPU data, then require canonical re-encoding."""
    if type(field) is not dict or field.get("source") != source:
        raise ValueError("Wrong CPU field provenance")
    if field.get("observed") is False:
        if (
            set(field) != {"observed", "reason", "source"}
            or type(field["reason"]) is not str
            or not 0 < len(field["reason"]) <= MAX_TEXT
        ):
            raise ValueError("Invalid unsupported CPU field")
        return None
    if field.get("observed") is not True:
        raise ValueError("Invalid CPU observed flag")
    t = field.get("type")
    v = field.get("value")
    types = {
        "none": type(None),
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    try:
        if t in types:
            if t == "none":
                if v is not None:
                    raise ValueError("Invalid None")
                value = None
            elif t in ("bool", "int", "str"):
                if type(v) is not types[t]:
                    raise ValueError("Wrong scalar type")
                value = v
            elif t in ("int32", "int64"):
                if (
                    type(v) is not int
                    or not np.iinfo(types[t]).min <= v <= np.iinfo(types[t]).max
                ):
                    raise ValueError("Wrong integer range")
                value = types[t](v)
            else:
                if type(v) is not float or not math.isfinite(v):
                    raise ValueError("Wrong float type")
                value = types[t](v)
        elif t == "numpy":
            shape = field.get("shape")
            dtype = np.dtype(field.get("dtype"))
            if (
                type(shape) is not list
                or len(shape) != 1
                or type(shape[0]) is not int
                or not 0 <= shape[0] <= MAX_ITEMS
                or dtype.kind not in "biuf"
                or dtype.itemsize > 8
                or type(v) is not list
                or len(v) != shape[0]
                or any(type(x) not in (bool, int, float) for x in v)
            ):
                raise ValueError("Invalid retained array")
            value = np.asarray(v, dtype=dtype)
            if value.tolist() != v:
                raise ValueError("Lossy array reconstruction")
        elif t in ("list", "tuple"):
            et = field.get("element_types")
            if (
                type(v) is not list
                or len(v) > MAX_ITEMS
                or type(et) is not list
                or len(et) != len(v)
            ):
                raise ValueError("Invalid retained sequence")
            value = []
            for item, tag in zip(v, et):
                if tag not in types:
                    raise ValueError("Nested/unknown retained sequence")
                f = dict(observed=True, type=tag, value=item, source=source)
                validate_cpu_field(f, source)
                value.append(None if tag == "none" else types[tag](item))
            if t == "tuple":
                value = tuple(value)
        else:
            raise ValueError("Unknown retained CPU type")
        if cpu_field(value, source) != field:
            raise ValueError("CPU field is not canonical")
    except (TypeError, OverflowError, KeyError) as exc:
        raise ValueError("Invalid retained CPU field") from exc
    return value


def validate_context(context, arm):
    _hash(context)
    if (
        set(context)
        != {
            "arm",
            "fields",
            "yaml",
            "planner_source_sha256",
            "motion_source_sha256",
            "sha256",
        }
        or context["arm"] != arm
        or context["planner_source_sha256"] != PLANNER_SHA
        or context["motion_source_sha256"] != MOTION_SHA
    ):
        raise ValueError("Wrong solver context source")
    if set(context["fields"]) != {"active_joints_name", "all_joints", "frame_bias"}:
        raise ValueError("Solver context fields changed")
    for key, value in context["fields"].items():
        validate_cpu_field(value, "CuroboPlanner.__dict__." + key)
    y = context["yaml"]
    if y.get("observed") is False:
        validate_cpu_field(y, "CuroboPlanner.__dict__.yml_path")
    else:
        if (
            set(y) != {"observed", "path", "bytes", "sha256", "text", "source"}
            or y["observed"] is not True
            or type(y["text"]) is not str
            or type(y["path"]) is not str
            or not Path(y["path"]).is_absolute()
            or len(y["path"]) > 4096
            or y["source"] != "existing runtime yml_path file before profile"
        ):
            raise ValueError("Invalid retained YAML context")
        raw = y["text"].encode("utf-8")
        if (
            len(raw) > MAX_CONFIG_BYTES
            or type(y["bytes"]) is not int
            or len(raw) != y["bytes"]
            or hashlib.sha256(raw).hexdigest() != y["sha256"]
        ):
            raise ValueError("Retained configuration hash/limit changed")
    return all(x["observed"] for x in [*context["fields"].values(), y])


def validate_evidence(row, context, *, call_id, parent_request_call_id, arm):
    _hash(row)
    context_complete = validate_context(context, arm)
    keys = {
        "version",
        "arm",
        "call_id",
        "parent_request_call_id",
        "context_sha256",
        "fields",
        "plan_config",
        "result_status",
        "result_attempts",
        "complete",
        "reported_category",
        "interpretation",
        "added_tensor_reads",
        "added_solver_calls",
        "added_control_calls",
        "sha256",
    }
    if (
        set(row) != keys
        or row["version"] != VERSION
        or row["arm"] != arm
        or row["call_id"] != call_id
        or row["parent_request_call_id"] != parent_request_call_id
        or row["context_sha256"] != context["sha256"]
        or type(call_id) is not int
        or type(parent_request_call_id) is not int
        or parent_request_call_id >= call_id
        or any(
            type(row[k]) is not int or row[k] != 0
            for k in ("added_tensor_reads", "added_solver_calls", "added_control_calls")
        )
    ):
        raise ValueError("Wrong solver evidence call/source/side-effect contract")
    if set(row["fields"]) != set(LOCAL_FIELDS) or set(row["plan_config"]) != set(
        CONFIG_FIELDS
    ):
        raise ValueError("Solver fields omitted")
    for key, value in row["fields"].items():
        validate_cpu_field(value, "plan_path.local." + key)
    for key, value in row["plan_config"].items():
        validate_cpu_field(value, "MotionGenPlanConfig.__dict__." + key)
    status = row["result_status"]
    source = "MotionGenResult.__dict__.status"
    if status.get("type") == "MotionGenStatus":
        if (
            set(status) != {"observed", "type", "name", "value", "source"}
            or status["observed"] is not True
            or status["source"] != source
            or type(status["name"]) is not str
            or STATUS_MEMBERS.get(status["name"]) != status["value"]
        ):
            raise ValueError("Unaudited solver status category")
    else:
        validate_cpu_field(status, source)
        if status["observed"] and status["type"] not in ("str", "none"):
            raise ValueError("Unsupported result status type")
    attempts = row["result_attempts"]
    validate_cpu_field(attempts, "MotionGenResult.__dict__.attempts")
    if attempts["observed"] and (
        attempts["type"] != "int" or not 0 <= attempts["value"] <= 100000
    ):
        raise ValueError("Wrong actual attempt count")
    expected_category = (
        status.get("name") if status.get("type") == "MotionGenStatus" else None
    )
    if row["reported_category"] != expected_category:
        raise ValueError("Solver category differs from retained result")
    complete = context_complete and all(
        v["observed"]
        for v in [
            *row["fields"].values(),
            *row["plan_config"].values(),
            status,
            attempts,
        ]
    )
    if row["complete"] is not complete:
        raise ValueError("Solver completeness differs from retained evidence")
    validate_joint_mapping(row, context)
    return complete


def validate_joint_mapping(row, context):
    """Recompute the source's CPU index selection and five-decimal rounding."""
    fields = row["fields"]
    config = context["fields"]
    relevant = [
        fields[k] for k in ("curr_joint_pos", "joint_indices", "joint_angles")
    ] + [config[k] for k in ("active_joints_name", "all_joints")]
    if not all(v["observed"] for v in relevant):
        return
    active = validate_cpu_field(
        config["active_joints_name"], "CuroboPlanner.__dict__.active_joints_name"
    )
    all_names = validate_cpu_field(
        config["all_joints"], "CuroboPlanner.__dict__.all_joints"
    )
    if (
        type(active) not in (list, tuple)
        or type(all_names) not in (list, tuple)
        or any(type(v) is not str for v in [*active, *all_names])
    ):
        raise ValueError("Invalid retained joint names")
    indices = [all_names.index(name) for name in active if name in all_names]
    if cpu_field(indices, "plan_path.local.joint_indices") != fields["joint_indices"]:
        raise ValueError("Joint index ordering differs from source mapping")
    qpos = validate_cpu_field(
        fields["curr_joint_pos"], "plan_path.local.curr_joint_pos"
    )
    if type(qpos) not in (np.ndarray, list, tuple) or any(
        i >= len(qpos) for i in indices
    ):
        raise ValueError("Joint index outside retained qpos")
    rounded = [round(qpos[i], 5) for i in indices]
    if cpu_field(rounded, "plan_path.local.joint_angles") != fields["joint_angles"]:
        raise ValueError("Rounded joint values differ from source mapping")
