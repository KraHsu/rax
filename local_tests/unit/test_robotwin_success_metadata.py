"""CPU executions of source-bound native metadata and success predicates."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import transforms3d as t3d

from services.robotwin.collection import initial_scene, scene_fingerprint
from services.robotwin.instruction_metadata import readonly_instruction_metadata
from services.robotwin.success_metadata import (DEPENDENCY_SHA256, MARKER, SOURCE_SHA256, canonical_hash,
                                                fresh_physical_state, initialize_success_metadata, metadata_policy,
                                                validate_metadata_policy, validate_metadata_receipt, SuccessMetadataError)

FIXTURES = Path(__file__).parents[1]/'fixtures/robotwin_success_metadata'
OPENWAM = next(parent/'OpenWAM' for parent in Path(__file__).resolve().parents
               if (parent/'OpenWAM/scripts/robotwin_constraint_policy_candidates.py').is_file())


def load_node(path, name, monkeypatch, **globals_):
    tree = ast.parse(path.read_text()); node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    module = ModuleType('audited_success_'+name); module.__file__ = str(path); module.__dict__.update(globals_)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), module.__dict__)
    return getattr(module, name), module


class Entity:
    def __init__(self, name, p=(0., 0., .74), q=(1., 0., 0., 0.)):
        self.name = name; self.pose = SimpleNamespace(p=np.asarray(p, np.float32), q=np.asarray(q, np.float32))
        self.qpos = np.array([.2], np.float32); self.qvel = np.zeros(1, np.float32)
    def get_name(self): return self.name
    def get_pose(self): return self.pose
    def get_root_pose(self): return self.pose
    def get_components(self): return [PhysxRigidDynamicComponent()]
    def get_root_linear_velocity(self): return np.zeros(3, np.float32)
    def get_root_angular_velocity(self): return np.zeros(3, np.float32)
    def get_qpos(self): return self.qpos
    def get_qvel(self): return self.qvel
    def get_qlimits(self): return np.array([[0., 1.]], np.float32)
    def get_contact_point(self, index): return [0., 0., .8, 1., 0., 0., 0.]
    def get_functional_point(self, index): return np.array([0., .1, .8, 1., 0., 0., 0.])


class PhysxRigidDynamicComponent:
    def get_linear_velocity(self): return np.zeros(3, np.float32)
    def get_angular_velocity(self): return np.zeros(3, np.float32)
    def get_kinematic(self): return False


class PhysxRigidStaticComponent:
    pass


def environment(task, monkeypatch, *, x=.25, q=(1., 0., 0., 0.)):
    source = FIXTURES/(task+'.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256[task]
    arm_type, _ = load_node(FIXTURES/'action.py', 'ArmTag', monkeypatch)
    face, _ = load_node(FIXTURES/'transforms.py', 'get_face_prod', monkeypatch, np=np, t3d=t3d)
    for name, path in [('ArmTag', 'action.py'), ('get_face_prod', 'transforms.py')]:
        assert hashlib.sha256((FIXTURES/path).read_bytes()).hexdigest() == DEPENDENCY_SHA256[name]
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata path called motion, expert, predicate, get_info or scene.step')
    class Base:
        def close_env(self):
            self.closed = True
        def is_left_gripper_open(self): return self.robot.is_left_gripper_open()
        def is_right_gripper_open(self): return self.robot.is_right_gripper_open()
        def _init_task_env_(self, **kwargs):
            self.info = {'info': {}, 'unchanged': 'keep'}; self.take_action_cnt = 0
            self.robot = SimpleNamespace(left_entity=Entity('left_robot'), right_entity=Entity('right_robot'),
                get_left_gripper_val=lambda: 1., get_right_gripper_val=lambda: 1.,
                get_left_tcp_pose=lambda: [0., 0., .8], get_right_tcp_pose=lambda: [0., 0., .8],
                is_left_gripper_open=lambda: True, is_right_gripper_open=lambda: True)
            self.object = Entity('073_rubikscube', [x, 0., .74]); self.laptop = Entity('015_laptop', q=q)
            self.cabinet = Entity('036_cabinet')
            actors = [self.object] if task == 'put_object_cabinet' else []
            arts = [self.robot.left_entity, self.robot.right_entity, self.laptop if task == 'open_laptop' else self.cabinet]
            self.scene = SimpleNamespace(get_all_actors=lambda: actors, get_all_articulations=lambda: arts,
                                         get_timestep=lambda: 1/250, step=forbidden)
            self.model_name = '015_laptop' if task == 'open_laptop' else '036_cabinet'
            self.model_id = 3 if task == 'open_laptop' else 46653
            self.selected_modelname = '073_rubikscube'; self.selected_model_id = np.int64(0)
            if task == 'place_object_scale':
                self.selected_modelname = '047_mouse'; self.object.name = self.selected_modelname
                self.scale = Entity('072_electronicscale'); self.scale_id = np.int64(5)
                actors[:] = [self.object, self.scale]
                arts[:] = [self.robot.left_entity, self.robot.right_entity]
    cls, module = load_node(source, task, monkeypatch, Base_Task=Base, ArmTag=arm_type, get_face_prod=face, np=np)
    native = cls(); native.setup_demo(seed=7, now_ep_num=0, is_test=False)
    # Class get_info remains available only to a separate oracle fixture.
    for name in ('get_info', 'play_once', 'move', 'grasp_actor', 'place_actor', 'open_gripper', 'move_by_displacement', 'take_action'):
        setattr(native, name, forbidden)
    return native, module


def initialize(task, native):
    return initialize_success_metadata(task, native, expected_scene_sha256=scene_fingerprint(initial_scene(native)),
                                       expected_state_sha256=canonical_hash(fresh_physical_state(native)))


@pytest.mark.parametrize('q,expected', [([1,0,0,0], 'left'), ([0,0,0,1], 'right'),
                                      ([.5,.5,.5,.5], 'right'), ([.5,.5,-.5,-.5], 'right')])
def test_laptop_fresh_setup_missing_arm_and_exact_native_face_branch(monkeypatch, q, expected):
    native, _ = environment('open_laptop', monkeypatch, q=q)
    assert not hasattr(native, 'arm_tag')
    with pytest.raises(AttributeError, match='arm_tag'): native.check_success()
    before = fresh_physical_state(native); original = native.check_success.__func__
    receipt = initialize('open_laptop', native)
    assert str(native.arm_tag) == expected
    assert receipt['assigned_fields'] == {'arm_tag': expected}
    assert fresh_physical_state(native) == before and receipt['fresh_state_sha256'] == receipt['after_state_sha256']
    assert native.check_success.__func__ is original and not native.check_success()
    native.laptop.qpos[0] = .5
    assert native.check_success()


@pytest.mark.parametrize('x,expected', [(-.25, 'left'), (.25, 'right'), (0., 'left')])
def test_cabinet_initial_z_and_arm_match_original_get_info_preserving_dtype(monkeypatch, x, expected):
    native, _ = environment('put_object_cabinet', monkeypatch, x=x)
    assert not hasattr(native, 'arm_tag') and not hasattr(native, 'origin_z')
    with pytest.raises(AttributeError, match='origin_z'): native.check_success()
    original = native.check_success.__func__; before_info = deepcopy(native.info)
    oracle, _ = environment('put_object_cabinet', monkeypatch, x=x)
    expected_info = type(oracle).get_info(oracle)
    receipt = initialize('put_object_cabinet', native)
    assert receipt['info'] == expected_info and str(native.arm_tag) == expected
    assert type(native.origin_z) is type(oracle.origin_z) is np.float32
    assert native.origin_z == oracle.origin_z and native.info == before_info
    assert native.check_success.__func__ is original
    # Sweep strict native rise/XY boundaries with the untouched AST predicate.
    for rise in (0., .00699999, .007, .00700001, .05, .11999999, .12, .12000001):
        for offset in (0., .04999999, .05, .05000001):
            position = np.array([offset, .1, native.origin_z+np.float32(rise)], np.float32)
            native.object.pose.p = position.copy(); oracle.object.pose.p = position.copy()
            assert bool(native.check_success()) == bool(oracle.check_success())


@pytest.mark.parametrize('task', list(SOURCE_SHA256))
def test_instruction_reader_never_initializes_predicate_fields_and_matches_native_tokens(monkeypatch, task):
    native, _ = environment(task, monkeypatch)
    before = fresh_physical_state(native); info_before = deepcopy(native.info)
    result = readonly_instruction_metadata(task, native)
    assert not hasattr(native, 'arm_tag') and not hasattr(native, 'origin_z')
    assert native.info == info_before and fresh_physical_state(native) == before
    assert result['provenance']['expert_actions_for_instruction'] is False
    for texts in json.loads((FIXTURES/(task+'.json')).read_text()).values():
        if isinstance(texts, list):
            for text in texts:
                assert set(re.findall(r'\{[A-Za-z]\}', text)) <= result['info'].keys()
    oracle, _ = environment(task, monkeypatch)
    assert result['info'] == type(oracle).get_info(oracle)


@pytest.mark.parametrize('change', ['source', 'helper', 'class', 'nonfinite', 'asset', 'scene', 'physical'])
def test_unreviewed_or_mismatched_inputs_fail_without_partial_assignment(monkeypatch, tmp_path, change):
    task = 'open_laptop'; native, module = environment(task, monkeypatch)
    scene = scene_fingerprint(initial_scene(native)); state = canonical_hash(fresh_physical_state(native))
    if change in ('source', 'helper'):
        changed = tmp_path/'changed.py'; changed.write_text('# changed\n')
        if change == 'source': monkeypatch.setattr(module, '__file__', str(changed))
        else:
            helper = type(native).play_once.__globals__['get_face_prod']
            monkeypatch.setattr(helper, '__code__', helper.__code__.replace(co_filename=str(changed)))
    elif change == 'class': task = 'put_object_cabinet'
    elif change == 'nonfinite': native.laptop.pose.q[0] = np.nan
    elif change == 'asset': native.model_id = 15
    elif change == 'scene': scene = '0'*64
    else: state = '0'*64
    with pytest.raises(ValueError):
        initialize_success_metadata(task, native, expected_scene_sha256=scene, expected_state_sha256=state)
    assert not hasattr(native, 'arm_tag') and not hasattr(native, MARKER)


def test_later_actions_and_repeated_initialization_cannot_rebase_success_height(monkeypatch):
    native, _ = environment('put_object_cabinet', monkeypatch)
    receipt = initialize('put_object_cabinet', native); origin = native.origin_z
    assert initialize('put_object_cabinet', native) == receipt
    native.object.pose.p[2] += .05
    with pytest.raises(ValueError, match='rebase'):
        initialize('put_object_cabinet', native)
    assert native.origin_z == origin
    native.take_action_cnt = 1
    with pytest.raises(ValueError, match='before any action'): initialize('put_object_cabinet', native)


def test_post_assignment_physics_change_fails_and_rolls_back_only_metadata(monkeypatch):
    import services.robotwin.success_metadata as adapter
    native, _ = environment('put_object_cabinet', monkeypatch); real = adapter.fresh_physical_state
    calls = []
    def changed_state(env):
        value = real(env); calls.append(True)
        if len(calls) == 2: value['timestep'] *= 2
        return value
    monkeypatch.setattr(adapter, 'fresh_physical_state', changed_state)
    with pytest.raises(ValueError, match='changed physical state'):
        initialize_success_metadata('put_object_cabinet', native, expected_scene_sha256=scene_fingerprint(initial_scene(native)))
    assert not hasattr(native, 'arm_tag') and not hasattr(native, 'origin_z') and not hasattr(native, MARKER)


def test_existing_stale_metadata_is_not_overwritten_and_unknown_task_untouched(monkeypatch):
    native, _ = environment('put_object_cabinet', monkeypatch)
    native.origin_z = np.float32(.3)
    with pytest.raises(ValueError, match='Existing success metadata'): initialize('put_object_cabinet', native)
    assert native.origin_z == np.float32(.3) and not hasattr(native, 'arm_tag')
    assert initialize_success_metadata('unreviewed', object(), expected_scene_sha256='none') is None


@pytest.mark.parametrize('change', ['missing', 'policy', 'scene', 'state_hash', 'arm', 'origin', 'actions'])
def test_offline_receipt_validation_requires_bound_policy_and_fresh_state(monkeypatch, change):
    native, _ = environment('put_object_cabinet', monkeypatch)
    receipt = initialize('put_object_cabinet', native); scene = receipt['scene_before']
    validate_metadata_receipt('put_object_cabinet', receipt, scene)
    if change == 'missing': receipt = None
    elif change == 'policy': receipt['policy']['version'] = 'unreviewed'
    elif change == 'scene': receipt['scene_after'] = 'different'
    elif change == 'state_hash': receipt['fresh_state']['timestep'] *= 2
    elif change == 'arm': receipt['assigned_fields']['arm_tag'] = 'left'
    elif change == 'origin': receipt['assigned_fields']['origin_z'] += .05
    else: receipt['native_actions_executed'] = True
    with pytest.raises(SuccessMetadataError):
        validate_metadata_receipt('put_object_cabinet', receipt, scene)
    with pytest.raises(SuccessMetadataError): validate_metadata_policy('put_object_cabinet', None)


@pytest.mark.parametrize('task,change', [('put_object_cabinet','coupled_origin'), ('open_laptop','coupled_face'),
                                       ('open_laptop','coupled_quaternion'), ('put_object_cabinet','duplicate_entity'),
                                       ('put_object_cabinet','dtype'), ('put_object_cabinet','whole_state')])
def test_receipt_inputs_are_bound_to_unique_physical_entity_not_self_consistent_claims(monkeypatch, task, change):
    native, _ = environment(task, monkeypatch); receipt = initialize(task, native); scene = receipt['scene_before']
    if change == 'coupled_origin':
        receipt['source_inputs']['object_position'][2] += .05
        receipt['assigned_fields']['origin_z'] += .05
    elif change == 'coupled_face':
        receipt['source_inputs']['face_product'] *= -1
        receipt['assigned_fields']['arm_tag'] = 'right'
    elif change == 'coupled_quaternion':
        receipt['source_inputs']['laptop_quaternion_wxyz'] = [0.,0.,0.,1.]
        receipt['source_inputs']['face_product'] = -1.
        receipt['assigned_fields']['arm_tag'] = 'right'
    elif change == 'dtype': receipt['source_inputs']['origin_z_dtype'] = 'float64'
    elif change == 'duplicate_entity':
        receipt['fresh_state']['actors'].append(deepcopy(receipt['fresh_state']['actors'][0]))
        receipt['fresh_state_sha256'] = receipt['after_state_sha256'] = canonical_hash(receipt['fresh_state'])
    else:
        receipt['fresh_state']['actors'][0]['p'][2] += .05
        receipt['source_inputs']['object_position'][2] += .05
        receipt['assigned_fields']['origin_z'] += .05
        receipt['fresh_state_sha256'] = receipt['after_state_sha256'] = canonical_hash(receipt['fresh_state'])
    with pytest.raises(SuccessMetadataError): validate_metadata_receipt(task, receipt, scene)


def test_numpy126_native_face_is_recomputed_from_snapshot_under_frozen_numeric_context(monkeypatch):
    from services.robotwin.success_metadata import _native_face
    q = np.array([.8170540928840637, 3.522428960422985e-5, 5.205589332035743e-5, .5765609741210938], np.float32)
    # Independently measured by the H200 native Python (NumPy1.26.4/t3d0.4.2)
    # without a scene/GPU. Promoting q before Nq would change this value.
    assert _native_face(q, '1.26.4') == .3351548411935794
    assert _native_face(q, '1.26.4') != _native_face(q, '2.5.4')
    native, _ = environment('open_laptop', monkeypatch, q=q)
    receipt = initialize('open_laptop', native); scene = receipt['scene_before']
    for version in ('1.26.4', '2.5.4'):
        policy = deepcopy(metadata_policy('open_laptop'))
        policy['numeric_context']['numpy_version'] = version
        variant = deepcopy(receipt); variant['policy'] = policy
        variant['source_inputs']['face_product'] = _native_face(q, version)
        validate_metadata_receipt('open_laptop', variant, scene, policy=policy)
        other = deepcopy(policy); other['numeric_context']['numpy_version'] = '2.5.4' if version.startswith('1') else '1.26.4'
        with pytest.raises(SuccessMetadataError, match='sealed manifest'):
            validate_metadata_receipt('open_laptop', variant, scene, policy=other)


def test_sapien3_component_velocity_and_static_state_do_not_require_entity_velocity_methods(monkeypatch):
    native, _ = environment('put_object_cabinet', monkeypatch)
    assert not hasattr(native.object, 'get_velocity')
    dynamic = fresh_physical_state(native)['actors'][0]
    assert dynamic['body_type'] == 'dynamic' and dynamic['velocity'] == [0.,0.,0.]
    native.object.get_components = lambda: [PhysxRigidStaticComponent()]
    static = fresh_physical_state(native)['actors'][0]
    assert static['body_type'] == 'static' and static['velocity'] == [0.,0.,0.]
    assert static['kinematic'] is None
    native.object.get_components = lambda: []
    with pytest.raises(SuccessMetadataError, match='one audited'):
        fresh_physical_state(native)


@pytest.mark.parametrize('task', list(SOURCE_SHA256))
@pytest.mark.parametrize('runner', ['fixedcase', 'search', 'model'])
def test_every_actual_factory_initializes_each_fresh_native_instance(monkeypatch, task, runner):
    native, _ = environment(task, monkeypatch); cls = type(native)
    scene = scene_fingerprint(initial_scene(native))
    root = Path(__file__).resolve().parents[2]
    paths = dict(fixedcase=root/'scripts/validate_robotwin_constraint_acceptance.py',
                 search=root/'scripts/search_robotwin_constraint_counterexamples.py',
                 model=OPENWAM/'scripts/robotwin_constraint_policy_candidates.py')
    tree = ast.parse(paths[runner].read_text())
    factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'factory')
    calls = []
    def checked_initializer(*args, **kwargs):
        # This assertion distinguishes a true fresh factory from carrying the
        # first instruction/expert environment's fields into a later rollout.
        assert not hasattr(args[1], 'arm_tag') and not hasattr(args[1], 'origin_z')
        value = initialize_success_metadata(*args, **kwargs); calls.append(value); return value
    scope = dict(random=SimpleNamespace(seed=lambda _: None), seed=7, kwargs={}, cls=cls, native_class=cls,
                 template={'task':task}, manifest={'task':task, 'scene_sha256':scene}, task=task,
                 ref={'provenance':{'scene_sha256':scene}}, scene_fingerprint=scene_fingerprint,
                 initial_scene=initial_scene, metadata_policy=metadata_policy,
                 initialize_success_metadata=checked_initializer)
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(paths[runner]), 'exec'), scope)
    first = scope['factory']() if runner == 'model' else scope['factory'](7)
    first.close_env()
    second = scope['factory']() if runner == 'model' else scope['factory'](7)
    assert first is not second and len(calls) == 2 and hasattr(second, MARKER)
    assert str(first.arm_tag) == str(second.arm_tag)
    if task == 'put_object_cabinet': assert type(second.origin_z) is np.float32


def test_fixed_reference_seed_scan_aborts_metadata_errors_instead_of_resampling():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root/'scripts/validate_robotwin_constraint_acceptance.py').read_text())
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and isinstance(n.type, ast.Tuple)
                and any(isinstance(e, ast.Name) and e.id == 'SuccessMetadataError' for e in n.type.elts)]
    assert len(handlers) == 1 and len(handlers[0].body) == 1 and isinstance(handlers[0].body[0], ast.Raise)


def test_search_reference_ingestion_requires_original_initialization_evidence(monkeypatch):
    from scripts.search_robotwin_constraint_counterexamples import prepare_reference
    from services.robotwin.acceptance import annotation_policy_fingerprint, json_hash
    from test_robotwin_acceptance import reference, template
    native, _ = environment('open_laptop', monkeypatch); receipt = initialize('open_laptop', native)
    t = template(); t['task'] = 'open_laptop'; source = reference(7, t)
    source['initial_scene'] = initial_scene(native); source['success_metadata'] = receipt
    source['provenance']['scene_sha256'] = receipt['scene_before']
    source['provenance']['source_episode_sha256'] = json_hash(dict(scene=source['initial_scene'], trajectory=source['trajectory']))
    manifest = dict(mode='demo_clean', split='validation', task=t['task'], native_commit='native',
                    alignment='reference_command_index_v1', native_success_rule='native_ever_success',
                    annotation_policy_sha256=annotation_policy_fingerprint(), template=t,
                    template_sha256=json_hash(t), replay_adapter=source['replay_adapter'],
                    success_metadata_policy=metadata_policy(t['task']))
    exclusions = dict(native_commit='native', excluded_training_seeds=[0], excluded_training_scene_sha256=['training'])
    working = prepare_reference(source, manifest, t, exclusions)
    assert working['success_metadata'] == source['success_metadata']
    bad = deepcopy(source); del bad['success_metadata']
    with pytest.raises(SuccessMetadataError, match='Missing fresh'):
        prepare_reference(bad, manifest, t, exclusions)


def test_model_collector_rejects_missing_forged_or_wrong_runtime_reference_before_native_work(monkeypatch):
    path = OPENWAM/'scripts/robotwin_constraint_policy_candidates.py'
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_reference_initialization')
    scope = {}; exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), scope)
    validate = scope['validate_reference_initialization']
    native, _ = environment('open_laptop', monkeypatch); receipt = initialize('open_laptop', native)
    reference = dict(provenance=dict(scene_sha256=receipt['scene_before']), success_metadata=receipt)
    source = dict(success_metadata_policy=metadata_policy('open_laptop'))
    validate('open_laptop', reference, source)
    with pytest.raises(SuccessMetadataError):
        validate('open_laptop', {k:v for k,v in reference.items() if k != 'success_metadata'}, source)
    forged = deepcopy(reference); forged['success_metadata']['source_inputs']['face_product'] *= -1
    forged['success_metadata']['assigned_fields']['arm_tag'] = 'right'
    with pytest.raises(SuccessMetadataError): validate('open_laptop', forged, source)
    other_source = deepcopy(source)
    other_source['success_metadata_policy']['numeric_context']['numpy_version'] = '1.26.4' if np.__version__.startswith('2.') else '2.5.4'
    other_ref = deepcopy(reference); other_ref['success_metadata']['policy'] = other_source['success_metadata_policy']
    # The identity quaternion is numerically identical in both contexts; a
    # valid offline receipt still cannot silently change the execution runtime.
    with pytest.raises(ValueError, match='runtime differs'):
        validate('open_laptop', other_ref, other_source)


def test_raw_model_proof_cannot_drop_or_rehash_corrupted_initialization_receipt(monkeypatch, tmp_path):
    from services.robotwin.acceptance import file_hash
    from services.robotwin.collection import atomic_json
    from services.robotwin.refinement_model_evidence import read, verify_model_evidence
    from test_robotwin_refinement_model_evidence import evidence
    inventory, root, row, release = evidence.__wrapped__(tmp_path)
    native, _ = environment('open_laptop', monkeypatch); receipt = initialize('open_laptop', native)
    run = read(root/'source_run.json')
    run.update(task='open_laptop', scene_sha256=receipt['scene_before'],
               success_metadata_policy=metadata_policy('open_laptop'), success_metadata=receipt)
    candidates = read(root/'model_candidates.json')
    candidates.update(task=run['task'], scene_sha256=run['scene_sha256'])
    row.update(task=run['task'], scene_sha256=run['scene_sha256'])
    def write_run(value):
        atomic_json(root/'source_run.json', value); atomic_json(root/'configuration.json', value)
        digest = file_hash(root/'source_run.json')
        candidates['candidates'][0]['model_provenance']['source_run_sha256'] = digest
        row['model_provenance']['source_run_sha256'] = digest
        atomic_json(root/'model_candidates.json', candidates)
    write_run(run)
    assert verify_model_evidence(inventory, [row], release)['runs']
    bad = deepcopy(run); del bad['success_metadata']; write_run(bad)
    with pytest.raises(SuccessMetadataError, match='Missing fresh'):
        verify_model_evidence(inventory, [row], release)
    bad = deepcopy(run); bad['success_metadata']['source_inputs']['face_product'] *= -1
    bad['success_metadata']['assigned_fields']['arm_tag'] = 'right'; write_run(bad)
    with pytest.raises(SuccessMetadataError, match='face-product'):
        verify_model_evidence(inventory, [row], release)
