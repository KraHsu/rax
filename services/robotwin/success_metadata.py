"""Initialize native success-predicate metadata immediately after fresh setup.

Only two audited task classes are supported. No get_info, expert, motion,
predicate, simulator step or global native monkeypatch is invoked. Instruction
reading remains separate and never assigns these native instance fields.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from services.robotwin.collection import initial_scene, scene_fingerprint
from services.robotwin.instruction_metadata import audited_native_source

ADAPTER_VERSION = 'fresh_scene_success_metadata_v1'
SOURCE_SHA256 = {
    'open_laptop': 'ec5f6e22b49d8da4e3bf5eacccbda4f147c74715e058cef1e0366e9884c111e9',
    'put_object_cabinet': '4c705f50df805965acad4d09257dc58af1aa3ecc903d3c34d117aa1b155d0b64',
}
DEPENDENCY_SHA256 = {
    'ArmTag': '7bf69e226b98ae7f5b21a523f722271530885cf4778873900f3564003a9b591d',
    'get_face_prod': '44df7f5dca83990e95ac4936951a9a81a68903ea01456caf9df2a6143a79f6e4',
}
CABINET_OBJECTS = ('047_mouse', '048_stapler', '057_toycar', '073_rubikscube', '075_bread',
                   '077_phone', '081_playingcards', '112_tea-box', '113_coffee-box', '107_soap')
MARKER = '_rax_fresh_success_metadata'


class SuccessMetadataError(ValueError):
    """Infrastructure/source mismatch: never a reason to select another seed."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def metadata_policy(task):
    if task not in SOURCE_SHA256:
        return None
    dependencies = {name: DEPENDENCY_SHA256[name] for name in
                    (('ArmTag', 'get_face_prod') if task == 'open_laptop' else ('ArmTag',))}
    import transforms3d
    if int(np.__version__.split('.')[0]) not in (1, 2) or transforms3d.__version__ != '0.4.2':
        raise SuccessMetadataError('Unreviewed metadata numeric runtime')
    return dict(version=ADAPTER_VERSION, task=task, native_source_sha256=SOURCE_SHA256[task],
                dependency_sha256=dependencies,
                numeric_context=dict(numpy_version=np.__version__, transforms3d_version=transforms3d.__version__),
                implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ('success_metadata.py', 'instruction_metadata.py', 'collection.py')})


def validate_metadata_policy(task, recorded):
    expected = metadata_policy(task)
    if expected is None:
        if recorded is not None:
            raise SuccessMetadataError('Unexpected success metadata policy')
        return
    if not isinstance(recorded, dict):
        raise SuccessMetadataError('Frozen success metadata policy is missing')
    context = recorded.get('numeric_context', {})
    if (set(context) != {'numpy_version', 'transforms3d_version'}
            or not isinstance(context['numpy_version'], str) or context['numpy_version'].split('.')[0] not in ('1', '2')
            or context['transforms3d_version'] != '0.4.2'
            or {k:v for k,v in recorded.items() if k != 'numeric_context'} !=
               {k:v for k,v in expected.items() if k != 'numeric_context'}):
        raise SuccessMetadataError('Frozen success metadata policy is missing or differs from audited implementation')


def _native_face(q, numpy_version):
    """Exact audited quat2mat R00 arithmetic across NumPy scalar promotions."""
    w, x, y, z = q
    nq = w*w + x*x + y*y + z*z
    if nq < np.finfo(float).eps:
        return 1.
    # NumPy 1 promotes Python float / np.float32 to float64; NumPy 2 keeps
    # float32. Freeze this choice in the externally sealed manifest policy.
    dtype = np.float64 if numpy_version.split('.')[0] == '1' or q.dtype == np.float64 else np.float32
    s = dtype(2.)/dtype(nq)
    yy = dtype(y)*(dtype(y)*s); zz = dtype(z)*(dtype(z)*s)
    return float(dtype(1.)-(yy+zz))


def validate_metadata_receipt(task, receipt, scene_sha256, *, policy=None):
    current_policy = metadata_policy(task)
    if current_policy is None:
        if receipt is not None:
            raise SuccessMetadataError('Unexpected success metadata receipt for unsupported task')
        return
    if not isinstance(receipt, dict):
        raise SuccessMetadataError('Missing fresh success metadata receipt')
    validate_metadata_policy(task, receipt.get('policy'))
    if policy is None:
        policy = current_policy
    validate_metadata_policy(task, policy)
    if receipt['policy'] != policy:
        raise SuccessMetadataError('Receipt numeric/source policy differs from its sealed manifest')
    if (receipt.get('adapter') != policy['version'] or receipt.get('task') != task
            or receipt.get('native_source_sha256') != policy['native_source_sha256']
            or receipt.get('dependency_sha256') != policy['dependency_sha256']
            or receipt.get('adapter_sha256') != policy['implementation_sha256']['success_metadata.py']
            or receipt.get('native_get_info_called') is not False
            or receipt.get('expert_actions_for_instruction') is not False):
        raise SuccessMetadataError('Success metadata provenance disagrees with its sealed policy')
    if (receipt.get('scene_before') != scene_sha256 or receipt.get('scene_after') != scene_sha256
            or receipt.get('fresh_state_sha256') != canonical_hash(receipt.get('fresh_state'))
            or receipt.get('after_state_sha256') != receipt['fresh_state_sha256']
            or receipt.get('native_actions_executed') is not False or receipt.get('predicate_unchanged') is not True
            or receipt.get('initialization_boundary') != 'after_fresh_setup_demo_before_any_action'):
        raise SuccessMetadataError('Fresh success metadata evidence changed or lacks unchanged physical state')
    state = receipt['fresh_state']
    scene = dict(actors=[dict(name=r['name'], pose=r['p']+r['q']) for r in state['actors']],
                 task_articulations=[dict(name=r['name'], pose=r['p']+r['q'], qpos=r['qpos'])
                                     for r in state['articulations'] if r['robot_arm'] is None])
    for rows in scene.values(): rows.sort(key=lambda r: (r['name'], r['pose']))
    if scene_fingerprint(scene) != scene_sha256:
        raise SuccessMetadataError('Fresh physical snapshot differs from frozen scene identity')
    source = receipt['source_inputs']
    identity = source.get('scene_entity', {})
    kind = 'articulations' if task == 'open_laptop' else 'actors'
    if identity.get('kind') != kind or not isinstance(identity.get('name'), str) or not identity['name']:
        raise SuccessMetadataError('Metadata input has no unique physical scene entity')
    matches = [row for row in receipt['fresh_state'][kind] if row['name'] == identity['name']]
    if len(matches) != 1:
        raise SuccessMetadataError('Metadata scene entity is missing or ambiguous')
    entity = matches[0]
    if task == 'open_laptop':
        if (identity['name'] != '015_laptop' or entity['quaternion_dtype'] not in ('float32', 'float64')
                or source['laptop_quaternion_wxyz'] != entity['q']):
            raise SuccessMetadataError('Laptop metadata quaternion differs from the physical scene')
        # Preserve the dtype that the original get_pose/quat2mat helper saw.
        q = np.asarray(entity['q'], dtype=entity['quaternion_dtype'])
        face = _native_face(q, policy['numeric_context']['numpy_version'])
        if not math.isfinite(face) or source['face_product'] != face:
            raise SuccessMetadataError('Laptop face-product branch differs from native quaternion computation')
        arm = 'left' if face > 0 else 'right'
        expected = {'arm_tag': arm}
    else:
        objects = [row for row in state['actors'] if row['name'] in CABINET_OBJECTS]
        if (len(objects) != 1 or identity['name'] != objects[0]['name']
                or source['object_position'] != entity['p'] or source['origin_z_dtype'] != entity['position_dtype']
                or entity['position_dtype'] not in ('float32', 'float64')):
            raise SuccessMetadataError('Cabinet initial height/position differs from the physical scene')
        arm = 'right' if source['object_position'][0] > 0 else 'left'
        expected = {'arm_tag': arm, 'origin_z': source['object_position'][2]}
    if receipt.get('assigned_fields') != expected:
        raise SuccessMetadataError('Assigned success fields differ from fixed native initial-state rule')


def _dependency(native, name):
    value = type(native).play_once.__globals__[name]
    digest = hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()
    if getattr(value, '__name__', None) != name or digest != DEPENDENCY_SHA256[name]:
        raise ValueError('Native metadata dependency changed: '+name)
    return value


def _asset_id(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('Invalid native metadata asset ID')
    return int(value)


def _derive(task, native):
    native_hash = audited_native_source(task, native, SOURCE_SHA256)
    arm_type = _dependency(native, 'ArmTag')
    dependencies = {'ArmTag': DEPENDENCY_SHA256['ArmTag']}
    if task == 'open_laptop':
        if (native.model_name != '015_laptop' or native.laptop.get_name() != native.model_name
                or _asset_id(native.model_id) not in range(11)):
            raise ValueError('Unexpected audited laptop asset')
        q = native.laptop.get_pose().q
        if np.asarray(q).shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError('Invalid initial laptop quaternion')
        # Call only this source-bound pure helper, preserving its exact >0 tie
        # rule and native transforms3d arithmetic rather than an Euler proxy.
        face = _dependency(native, 'get_face_prod')(q, [1, 0, 0], [1, 0, 0])
        if not math.isfinite(face):
            raise ValueError('Invalid initial laptop face product')
        dependencies['get_face_prod'] = DEPENDENCY_SHA256['get_face_prod']
        arm = 'left' if face > 0 else 'right'
        values = {'arm_tag': arm_type(arm)}
        inputs = dict(laptop_quaternion_wxyz=np.asarray(q).tolist(), face_product=float(face),
                      scene_entity=dict(kind='articulations', name=native.laptop.get_name()),
                      native_branch='left if face_product > 0 else right', model_id=int(native.model_id))
        info = {'{A}': f'015_laptop/base{int(native.model_id)}', '{a}': arm}
    else:
        if (native.model_name != '036_cabinet' or native.model_id != 46653
                or native.selected_modelname not in CABINET_OBJECTS or native.object.get_name() != native.selected_modelname):
            raise ValueError('Unexpected audited cabinet/object asset')
        model_id = _asset_id(native.selected_model_id)
        position = native.object.get_pose().p
        if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid initial object position')
        arm = 'right' if position[0] > 0 else 'left'
        # Preserve the native NumPy scalar dtype at strict height boundaries.
        values = {'arm_tag': arm_type(arm), 'origin_z': position[2]}
        inputs = dict(object_position=np.asarray(position).tolist(), origin_z_dtype=str(np.asarray(position[2]).dtype),
                      scene_entity=dict(kind='actors', name=native.object.get_name()),
                      native_branch='right if object_initial_x > 0 else left', object_model_id=model_id)
        info = {'{A}': f'{native.selected_modelname}/base{model_id}', '{B}': '036_cabinet/base0',
                '{a}': arm, '{b}': str(arm_type(arm).opposite)}
    provenance = dict(adapter=ADAPTER_VERSION, task=task, native_source_sha256=native_hash,
                      adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      dependency_sha256=dependencies, source_inputs=inputs,
                      expert_actions_for_instruction=False, native_get_info_called=False)
    return values, info, provenance


def readonly_success_instruction_metadata(task, native):
    if task not in SOURCE_SHA256:
        return None
    _, info, provenance = _derive(task, native)
    return dict(info=deepcopy(info), provenance=provenance)


def fresh_physical_state(native):
    """Exact physical state; the coarser scene identity alone is insufficient."""
    actors = []
    for actor in native.scene.get_all_actors():
        pose = actor.get_pose()
        bodies = [c for c in actor.get_components() if type(c).__name__ in
                  ('PhysxRigidDynamicComponent', 'PhysxRigidStaticComponent')]
        if len(bodies) != 1:
            raise SuccessMetadataError('Expected one audited SAPIEN rigid body component per actor')
        body = bodies[0]; dynamic = type(body).__name__ == 'PhysxRigidDynamicComponent'
        actors.append(dict(name=actor.get_name(), p=np.asarray(pose.p).tolist(), q=np.asarray(pose.q).tolist(),
                           position_dtype=str(np.asarray(pose.p).dtype), quaternion_dtype=str(np.asarray(pose.q).dtype),
                           body_type='dynamic' if dynamic else 'static',
                           kinematic=bool(body.get_kinematic()) if dynamic else None,
                           velocity=np.asarray(body.get_linear_velocity()).tolist() if dynamic else [0., 0., 0.],
                           angular_velocity=np.asarray(body.get_angular_velocity()).tolist() if dynamic else [0., 0., 0.]))
    arts = []
    for art in native.scene.get_all_articulations():
        pose = art.get_root_pose()
        robot_arm = next((arm for arm in ('left', 'right') if art == getattr(native.robot, arm+'_entity')), None)
        arts.append(dict(name=art.get_name(), p=np.asarray(pose.p).tolist(), q=np.asarray(pose.q).tolist(),
                         robot_arm=robot_arm,
                         root_linear_velocity=np.asarray(art.get_root_linear_velocity()).tolist(),
                         root_angular_velocity=np.asarray(art.get_root_angular_velocity()).tolist(),
                         position_dtype=str(np.asarray(pose.p).dtype), quaternion_dtype=str(np.asarray(pose.q).dtype),
                         qpos=np.asarray(art.get_qpos()).tolist(), qvel=np.asarray(art.get_qvel()).tolist()))
    key = lambda row: json.dumps(row, sort_keys=True, allow_nan=False)
    return dict(actors=sorted(actors, key=key), articulations=sorted(arts, key=key),
                gripper_commands={arm: float(getattr(native.robot, f'get_{arm}_gripper_val')())
                                  for arm in ('left', 'right')},
                timestep=float(native.scene.get_timestep()))


def initialize_success_metadata(task, native, *, expected_scene_sha256, expected_state_sha256=None):
    """Initialize only audited instance metadata at the fresh pre-action boundary.

    Failures are infrastructure errors, never evidence for skipping a scene.
    The returned receipt must be stored and checked against a sealed policy.
    """
    try:
        return _initialize_success_metadata(task, native, expected_scene_sha256=expected_scene_sha256,
                                            expected_state_sha256=expected_state_sha256)
    except Exception as exc:
        raise SuccessMetadataError(str(exc)) from exc


def _initialize_success_metadata(task, native, *, expected_scene_sha256, expected_state_sha256=None):
    """Set audited per-instance fields once, before reference/model/replay actions.

    The caller supplies its frozen scene identity immediately after setup_demo.
    A repeated call may verify the same fresh state, but cannot rebase origin_z
    after an action. Unsupported tasks return None and remain untouched.
    """
    if task not in SOURCE_SHA256:
        return None
    if getattr(native, 'take_action_cnt', 0) != 0:
        raise ValueError('Success metadata must be initialized before any action')
    scene_before = scene_fingerprint(initial_scene(native))
    if scene_before != expected_scene_sha256:
        raise ValueError('Success metadata scene differs from frozen fresh reference')
    before = fresh_physical_state(native); state_hash = canonical_hash(before)
    if expected_state_sha256 is not None and state_hash != expected_state_sha256:
        raise ValueError('Success metadata physical state differs from fresh snapshot')
    previous = getattr(native, MARKER, None)
    if previous is not None and previous['fresh_state_sha256'] != state_hash:
        raise ValueError('Cannot rebase native success metadata after scene evolution')
    values, info, provenance = _derive(task, native)
    missing = object(); old = {key: getattr(native, key, missing) for key in values}
    for key, value in values.items():
        if old[key] is not missing and old[key] != value:
            raise ValueError('Existing success metadata differs from initial native rule: '+key)
    try:
        for key, value in values.items():
            setattr(native, key, value)
        after = fresh_physical_state(native)
        scene_after = scene_fingerprint(initial_scene(native))
        if canonical_hash(after) != state_hash or scene_after != scene_before:
            raise ValueError('Success metadata initialization changed physical state')
    except BaseException:
        for key, value in old.items():
            if value is missing:
                if hasattr(native, key): delattr(native, key)
            else:
                setattr(native, key, value)
        raise
    receipt = dict(provenance, info=info,
                   policy=metadata_policy(task), fresh_state=before,
                   assigned_fields={key: str(value) if key == 'arm_tag' else float(value) for key, value in values.items()},
                   scene_before=scene_before, scene_after=scene_after,
                   fresh_state_sha256=state_hash, after_state_sha256=canonical_hash(after),
                   predicate_unchanged=True, native_actions_executed=False,
                   initialization_boundary='after_fresh_setup_demo_before_any_action')
    try:
        validate_metadata_receipt(task, receipt, scene_before)
    except BaseException:
        for key, value in old.items():
            if value is missing:
                if hasattr(native, key): delattr(native, key)
            else:
                setattr(native, key, value)
        raise
    setattr(native, MARKER, deepcopy(receipt))
    return receipt
