"""CPU fakes exercise bounded state transitions; no claim of physical validation."""
import ast
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P/'package/runtime'))
import contracts
import driver
import predicate_observer as observer
import verifier

CORPUS = json.loads(gzip.decompress((P.parent/'dual_bottle_revision_v1/verified_corpus.json.gz').read_bytes()))
REFERENCE = CORPUS['refs']['10000']
SOURCE = P.parent/'native_task_batch_v6/package/inputs/pick_dual_bottles/native_task.py'


def bind_native_predicate(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'check_success')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path.resolve()), 'exec'), ns)
    return ns['check_success']


class Actor:
    def __init__(self, xyz):
        self.point = np.asarray(xyz, dtype=np.float32)
        self.point_reads = 0
        self.root_reads = 0
    def get_functional_point(self, index):
        assert index == 0
        self.point_reads += 1
        return self.point.copy()
    def get_pose(self):
        self.root_reads += 1
        return types.SimpleNamespace(p=self.point.copy(), q=np.asarray([1, 0, 0, 0], dtype=np.float32))


def fake_native(path):
    value = types.SimpleNamespace(bottle1=Actor([-.06, -.105, 1]), bottle2=Actor([.06, -.105, 1]),
                                 left_target_pose=[-.06, -.105, 1], right_target_pose=[.06, -.105, 1],
                                 actor_pose=True, take_action_cnt=0, plan_success=True, frame=0)
    value.check_success = types.MethodType(bind_native_predicate(path), value)
    value.close_env = lambda: None
    def take_action(*_args, **_kwargs):
        value.take_action_cnt += 1
        return 'unchanged_return'
    value.take_action = take_action
    return value


def test_observer_preserves_native_calls_return_and_side_effect():
    native = fake_native(SOURCE)
    monitor = observer.PredicateObserver(native, SOURCE)
    monitor.install()
    try:
        assert native.check_success()
        native.bottle1.point[2] = .77
        assert not native.check_success()
        assert native.actor_pose is False
    finally:
        monitor.close()
    assert native.bottle1.point_reads == native.bottle2.point_reads == 2
    assert native.bottle1.root_reads == 0
    result = observer.verify_observations(monitor.result())
    assert result['recorded_calls'] == 2 and result['successful_calls'] == 1
    assert sys.getprofile() is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('z', [.78, .89, float(np.nextafter(.89, 1.))])
def test_source_ast_strict_predicate_and_packed_types_match(dtype, z):
    native = fake_native(SOURCE)
    native.bottle1.point = np.array([-.06, -.105, z], dtype=dtype)
    monitor = observer.PredicateObserver(native, SOURCE)
    monitor.install()
    try:
        outcome = bool(native.check_success())
    finally:
        monitor.close()
    data = monitor.result()['predicate_rle'][0]['data']
    assert observer.unpack(data['bottle1_pose']).dtype == dtype
    assert observer.components(data)['success'] is outcome


def test_diagnostic_read_error_does_not_change_native_result():
    native = fake_native(SOURCE)
    monitor = observer.PredicateObserver(native, SOURCE)
    native.bottle1.get_pose = lambda: (_ for _ in ()).throw(RuntimeError('read-only getter unavailable'))
    monitor.install()
    try:
        before = native.check_success()
        monitor.observe_pose('boundary')
        after = native.check_success()
    finally:
        monitor.close()
    assert before == after and monitor.result()['observation_complete'] is False
    assert monitor.result()['errors'][0]['where'] == 'root_pose'


def test_observer_refuses_other_profiler_and_source_drift(tmp_path):
    native = fake_native(SOURCE)
    monitor = observer.PredicateObserver(native, SOURCE)
    old = lambda *_: None
    sys.setprofile(old)
    try:
        with pytest.raises(ValueError, match='another'):
            monitor.install()
        assert sys.getprofile() is old
    finally:
        sys.setprofile(None)
    path = tmp_path/'task.py'
    path.write_text(SOURCE.read_text()+'\n')
    with pytest.raises(ValueError, match='exact original'):
        observer.PredicateObserver(native, path)


def test_replaced_profile_is_explicitly_incomplete_without_clobbering_replacement():
    native = fake_native(SOURCE)
    monitor = observer.PredicateObserver(native, SOURCE)
    other = lambda *_: None
    monitor.install()
    try:
        sys.setprofile(other)
        assert native.check_success()
        monitor.close()
        assert sys.getprofile() is other
    finally:
        sys.setprofile(None)
    assert monitor.result()['observation_complete'] is False
    assert monitor.result()['errors'][0]['where'] == 'profile_lost'


def test_observer_capacity_does_not_change_or_add_native_calls(monkeypatch):
    monkeypatch.setattr(observer, 'MAX_PREDICATE_ROWS', 2)
    monkeypatch.setattr(observer, 'MAX_ROOT_POSE_ROWS', 2)
    monkeypatch.setattr(observer, 'MAX_ERRORS', 1)
    native = fake_native(SOURCE)
    monitor = observer.PredicateObserver(native, SOURCE)
    monitor.install()
    try:
        for i in range(5):
            native.bottle1.point[2] = .91+i*.01
            assert native.check_success()
            monitor.observe_pose('sample', i)
    finally:
        monitor.close()
    result = monitor.result()
    assert len(result['predicate_rle']) == len(result['root_pose_samples']) == 2
    assert result['predicate_calls'] == native.bottle1.point_reads == 5
    assert result['omitted_predicate_returns'] == result['omitted_root_poses'] == 3
    assert len(result['errors']) == 1 and result['errors_total'] == 2
    assert result['observation_complete'] is False
    observer.verify_observations(result)


def test_current_campaign_scene_claim_and_reserved_consumed_exclusion(tmp_path):
    ledger = tmp_path/'ledger.json'
    contracts.write_once(ledger, contracts.new_development_ledger('a'*64))
    contracts.claim_scene(ledger, 13000, 'b'*64, [])
    with pytest.raises(ValueError, match='already claimed'):
        contracts.claim_scene(ledger, 13000, 'c'*64, [])
    with pytest.raises(ValueError, match='duplicates'):
        contracts.claim_scene(ledger, 13001, 'b'*64, [])
    with pytest.raises(ValueError, match='duplicates'):
        contracts.claim_scene(ledger, 13001, 'c'*64, ['c'*64])
    train = {'exclusions': {'excluded_training_seeds': []}}
    for status in ('reserved', 'consumed'):
        with pytest.raises(ValueError, match='no reseed'):
            contracts.assert_seed_exclusion([13000, 13001, 13002], train,
                [{'excluded_seeds': [13001], 'state': status}], [])


@pytest.fixture
def finite_scene(tmp_path, monkeypatch):
    native_root = tmp_path/'native'
    source = native_root/'envs/pick_dual_bottles.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(SOURCE.read_bytes())
    design = dict(native_root=str(native_root), identity_sha256='d'*64, dataset_version_sha256='e'*64)
    ledger = tmp_path/'ledger.json'
    contracts.write_once(ledger, contracts.new_development_ledger('d'*64))
    instances = []
    settings = {'bad_baseline': None, 'native_false_baseline': None, 'expert_failed': False, 'cleanup_failed': False,
                'cleanup_failure_at': None, 'replay_exception': False}
    def factory(_seed):
        obj = fake_native(source)
        obj.ordinal = len(instances)
        obj.closed = False
        if obj.ordinal == 0 and settings['expert_failed']:
            obj.plan_success = False
        if settings['native_false_baseline'] == obj.ordinal:
            obj.bottle1.point[2] = .77
        def close():
            obj.closed = True
            if ((settings['cleanup_failed'] and obj.ordinal == 0)
                    or settings['cleanup_failure_at'] == obj.ordinal):
                raise RuntimeError('cleanup failed')
        obj.close_env = close
        instances.append(obj)
        return obj
    def sample(native):
        return {arm: {'pose': copy.deepcopy(values['pose'][native.frame]), 'gripper': values['gripper'][native.frame]}
                for arm, values in REFERENCE['trajectory'].items()}
    def expert(native):
        for frame in range(1, len(REFERENCE['trajectory']['left']['pose'])-1):
            native.frame = frame
            native._take_picture()
            native.check_success()
        native.frame += 1
    def fake_replay(native, command):
        actual = copy.deepcopy(command)
        n = len(command['left']['pose'])
        success = []
        for _ in range(1, n):
            native.take_action([], action_type='ee')
            success.append(bool(native.check_success()))
        success.append(bool(native.check_success()))
        if settings['replay_exception']:
            raise RuntimeError('unclassified controller exception')
        if settings['bad_baseline'] == native.ordinal:
            for arm in actual:
                for pose in actual[arm]['pose'][1:]:
                    pose[0] += .02
        return dict(actual=actual, native_success=any(success), ever_success=any(success), final_success=success[-1],
                    execution_complete=True, planner_success=True, error=None,
                    planner_results=[dict(frame=i, arm=a, status='Success') for i in range(1, n) for a in ('left', 'right')])
    import scripts.collect_robotwin_clean50 as collect
    monkeypatch.setattr(collect, 'play_expert', expert)
    monkeypatch.setattr(driver, 'sample', sample)
    monkeypatch.setattr(driver, 'initial_scene', lambda _native: copy.deepcopy(REFERENCE['initial_scene']))
    monkeypatch.setattr(driver, 'install_base_frame_adapter', lambda *_a, **_kw: copy.deepcopy(REFERENCE['replay_adapter']))
    monkeypatch.setattr(driver, 'replay', fake_replay)
    return tmp_path/'scene', design, ledger, factory, instances, settings


def test_unique_reference_failure_has_no_baseline_or_next_attempt(finite_scene):
    out, design, ledger, factory, instances, settings = finite_scene
    settings['expert_failed'] = True
    result = driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    assert result['status'] == 'blocked_reference' and len(instances) == 1
    assert not (out/'trials').exists()
    verifier.verify_scene(out, design, previous_claims=contracts.read(ledger)['claims'])


@pytest.mark.parametrize('bad,expected', [(1, 2), (2, 3)])
def test_invalid_baseline_stops_exact_prefix_without_retry(finite_scene, bad, expected):
    out, design, ledger, factory, instances, settings = finite_scene
    settings['bad_baseline'] = bad
    result = driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    assert result['status'] == 'blocked_baseline' and len(instances) == expected
    report = verifier.verify_scene(out, design, previous_claims=contracts.read(ledger)['claims'])
    assert len(report['trials']) == bad


def test_cleanup_exception_preserves_attempt_and_stops(finite_scene):
    out, design, ledger, factory, instances, settings = finite_scene
    settings['cleanup_failed'] = True
    with pytest.raises(RuntimeError, match='infrastructure'):
        driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    assert len(instances) == 1
    assert contracts.read(out/'capture_attempt.json')['cleanup_error']
    assert contracts.read(out/'state.json')['status'] == 'infrastructure_failed'


def test_controller_exception_preserves_trial_and_halts(finite_scene):
    out, design, ledger, factory, instances, settings = finite_scene
    settings['replay_exception'] = True
    with pytest.raises(RuntimeError, match='infrastructure'):
        driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    assert len(instances) == 2
    assert contracts.read(out/'trials/baseline_0/record.json')['infrastructure_failure'] is True
    assert not (out/'trials/baseline_1').exists()


def test_capture_observer_close_failure_still_cleans_and_commits(finite_scene, monkeypatch):
    out, design, ledger, factory, instances, settings = finite_scene
    original = observer.PredicateObserver.close
    def broken(monitor):
        original(monitor)
        raise RuntimeError('observer teardown failed')
    monkeypatch.setattr(observer.PredicateObserver, 'close', broken)
    with pytest.raises(RuntimeError, match='infrastructure'):
        driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    assert len(instances) == 1 and instances[0].closed
    attempt = contracts.read(out/'capture_attempt.json')
    assert attempt['native_success'] is True and attempt['observer_close_error']
    assert attempt['samples'] and sys.getprofile() is None


def test_replay_primary_observer_and_cleanup_errors_all_preserved(finite_scene, monkeypatch):
    out, design, ledger, factory, instances, settings = finite_scene
    settings.update(replay_exception=True, cleanup_failure_at=1)
    original = observer.PredicateObserver.close
    def broken(monitor):
        original(monitor)
        if monitor.phase == 'replay_existing_predicate_cadence':
            raise RuntimeError('observer teardown failed')
    monkeypatch.setattr(observer.PredicateObserver, 'close', broken)
    with pytest.raises(RuntimeError, match='infrastructure'):
        driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    assert len(instances) == 2 and all(n.closed for n in instances)
    raw = contracts.read(out/'trials/baseline_0/outcome.json')
    record = contracts.read(out/'trials/baseline_0/record.json')
    assert 'unclassified controller exception' in raw['error']
    assert 'observer teardown' in raw['observer_close_error'] and raw['cleanup_error']
    assert 'unclassified controller exception' in record['infrastructure_error']
    assert not (out/'trials/baseline_1').exists()


def test_full_finite_scene_and_tampered_successor_rejected(finite_scene):
    out, design, ledger, factory, instances, settings = finite_scene
    result = driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    assert result['status'] == 'scene_complete' and len(instances) == 21
    report = verifier.verify_scene(out, design, previous_claims=contracts.read(ledger)['claims'])
    assert report['native_initializations'] == 21 and len(report['trials']) == 20
    with pytest.raises(ValueError, match='new fixed scene'):
        driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    (out/'trials/uncommitted_extra').mkdir()
    with pytest.raises(ValueError, match='partial trial'):
        verifier.verify_scene(out, design, previous_claims=contracts.read(ledger)['claims'])


def test_every_manifest_semantic_field_is_frozen(finite_scene):
    out, design, ledger, factory, instances, settings = finite_scene
    settings['bad_baseline'] = 1
    driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    original = contracts.read(out/'manifest.json')
    for key, value in [('task', 'other'), ('seed', 13001), ('version', 2), ('source', 'model_closed_loop'),
                       ('predicate_policy', 'final_only'), ('annotation_policy_sha256', 'f'*64),
                       ('publication_allowed', True), ('training_allowed', True), ('formal_holdout', True),
                       ('early_termination_suppressed', False), ('alignment', 'other'), ('unreviewed_field', True)]:
        changed = dict(original, **{key: value})
        contracts.update_state(out/'manifest.json', changed)
        with pytest.raises(ValueError, match='manifest identities'):
            verifier.verify_scene(out, design, previous_claims=contracts.read(ledger)['claims'])
    contracts.update_state(out/'manifest.json', original)


def test_capture_final_result_is_not_replaced_by_any_ever(finite_scene):
    out, design, ledger, factory, instances, settings = finite_scene
    settings['bad_baseline'] = 1
    driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    path = out/'capture_observations.json.gz'
    data = json.loads(gzip.decompress(path.read_bytes()))
    final = next(r for r in data['predicate_rle'] if r['data']['phase'] == 'original_capture_final_check')
    # Dropping the actual final call while preserving prior successful calls is
    # rejected even if an attacker adjusts generic timeline totals consistently.
    data['predicate_rle'].remove(final)
    data['predicate_calls'] -= 1
    data['last_return'] = dict(call=data['predicate_calls'], data=data['predicate_rle'][-1]['data'])
    path.write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
    with pytest.raises(ValueError, match='Unique original capture final'):
        verifier.verify_scene(out, design, previous_claims=contracts.read(ledger)['claims'])


def test_registry_reserved_consumed_and_explicit_authority_discovery(tmp_path):
    registry, revision = tmp_path/'registry', tmp_path/'revision'
    release = {'task': 'pick_dual_bottles', 'holdout_seeds': [13000, 14000]}
    contracts.write_once(revision/'holdout_release.json', release)
    entry = dict(state='reserved', reservation_id='r1', holdout_seeds=[13000, 14000],
                 revision_root=str(revision), release_sha256=contracts.digest(release))
    path = registry/'refinement_exposure/pick_dual_bottles.json'
    contracts.write_once(path, dict(version=1, task='pick_dual_bottles', reservations=[entry]))
    assert contracts.inspect_registry(registry)['excluded_seeds'] == [13000, 14000]
    receipt = dict(scene_sha256=['a'*64])
    contracts.write_once(revision/'exposure_receipt.json', receipt)
    entry.update(state='consumed', exposure_receipt_sha256=contracts.file_hash(revision/'exposure_receipt.json'))
    contracts.update_state(path, dict(version=1, task='pick_dual_bottles', reservations=[entry]))
    contracts.write_once(tmp_path/'manifest.json', dict(registry_root=str(registry)))
    scope = contracts.exposure_scope([str(tmp_path)])
    assert len(scope['explicit_registry_fields']) == 1
    assert 'a'*64 in scope['excluded_scene_sha256'] and 13000 in scope['excluded_seeds']
    contracts.update_state(revision/'exposure_receipt.json', dict(scene_sha256=['b'*64]))
    with pytest.raises(ValueError, match='Consumed exposure receipt drift'):
        contracts.inspect_registry(registry)


def test_preexpert_claim_prefix_cannot_be_replaced(finite_scene):
    out, design, ledger, factory, instances, settings = finite_scene
    settings['expert_failed'] = True
    driver.run_scene(design, 13000, out, ledger, set(), factory=factory)
    claims = contracts.read(ledger)['claims']
    attempt = contracts.read(out/'capture_attempt.json')
    attempt['development_ledger_after_claim_sha256'] = 'f'*64
    contracts.update_state(out/'capture_attempt.json', attempt)
    with pytest.raises(ValueError, match='ledger prefix changed'):
        verifier.verify_scene(out, design, previous_claims=claims)


def test_predicate_numeric_context_must_not_silently_change():
    native = fake_native(SOURCE)
    monitor = observer.PredicateObserver(native, SOURCE)
    monitor.install()
    try:
        native.check_success()
    finally:
        monitor.close()
    value = monitor.result()
    value['numeric_context']['numpy_version'] = 'different'
    with pytest.raises(ValueError, match='original NumPy'):
        observer.verify_observations(value)


def test_worker_requires_exact_live_coordinator_before_factory(tmp_path, monkeypatch):
    design = dict(output='runs/pick_dual_bottles')
    path = tmp_path/'design.json'; contracts.write_once(path, design)
    output = tmp_path/design['output']
    ledger = output/'development_registry/pick_dual_bottles.json'
    contracts.write_once(ledger, contracts.new_development_ledger(contracts.file_hash(path)))
    parent = dict(pid=os.getppid(), start_ticks=10)
    contracts.write_once(output/'worker_intent_13000.json', dict(seed=13000, coordinator=parent))
    monkeypatch.setattr(driver, 'PACKAGE', tmp_path)
    monkeypatch.setattr(driver, 'validate_design', lambda *_: ({'exclusions': {'excluded_training_scene_sha256': []}}, [], {'excluded_scene_sha256': []}))
    monkeypatch.setattr(driver, 'process_identity', lambda pid: dict(pid=pid, start_ticks=11))
    calls = []
    monkeypatch.setattr(driver, 'run_scene', lambda *a, **kw: calls.append(a))
    args = types.SimpleNamespace(design=path, check_only=False, execute=True, seed=13000,
                                 output=output/'scenes/seed13000', ledger=ledger)
    with pytest.raises(ValueError, match='exact live coordinator'):
        driver.run(args)
    assert calls == [] and not (output/'worker_receipts').exists()
    monkeypatch.setattr(driver, 'process_identity', lambda pid: parent if pid == os.getppid() else dict(pid=pid, start_ticks=11))
    assert driver.run(args) == 0 and len(calls) == 1
    with pytest.raises(FileExistsError):
        driver.run(args)
    assert len(calls) == 1


def test_coordinator_worker_failure_never_starts_next_seed_or_retries(tmp_path, monkeypatch):
    design = dict(output='runs/pick_dual_bottles')
    path = tmp_path/'design.json'; contracts.write_once(path, design)
    monkeypatch.setattr(driver, 'PACKAGE', tmp_path)
    monkeypatch.setattr(driver, 'validate_design', lambda *_: ({}, [], {}))
    calls = []
    def fail_worker(argv, **kwargs):
        calls.append(argv)
        return types.SimpleNamespace(returncode=7)
    monkeypatch.setattr(driver.subprocess, 'run', fail_worker)
    args = types.SimpleNamespace(design=path, check_only=False, execute=True, seed=None,
                                 output=tmp_path/design['output'], ledger=None)
    with pytest.raises(RuntimeError, match='no next scene/retry'):
        driver.run(args)
    assert len(calls) == 1 and calls[0][calls[0].index('--seed')+1] == '13000'
    assert contracts.read(args.output/'state.json')['status'] == 'infrastructure_failed'
    with pytest.raises(FileExistsError):
        driver.run(args)
    assert len(calls) == 1


def test_native_false_compliant_baseline_is_not_a_crafted_counterexample(finite_scene):
    """Full CPU fake campaign: one invalid baseline, two duplicate-scene stops."""
    out, design, old_ledger, factory, instances, settings = finite_scene
    settings['native_false_baseline'] = 1
    root = out.parent/'campaign'
    ledger = root/'development_registry/pick_dual_bottles.json'
    contracts.write_once(ledger, contracts.new_development_ledger(design['identity_sha256']))
    coordinator = dict(pid=123, start_ticks=1, boot_id='cpu-fixture', command_sha256='a'*64)
    contracts.write_once(root/'intent.json', dict(design_sha256=design['identity_sha256'],
        seeds=list(contracts.SEEDS), coordinator=coordinator))
    for index, seed in enumerate(contracts.SEEDS):
        driver.run_scene(design, seed, root/'scenes'/f'seed{seed}', ledger, set(), factory=factory)
        contracts.write_once(root/f'worker_intent_{seed}.json', dict(coordinator=coordinator, seed=seed, automatic_retries=0))
        contracts.write_once(root/'worker_receipts'/f'seed{seed}.json', dict(coordinator=coordinator, seed=seed,
            automatic_retries=0, design_sha256=design['identity_sha256'],
            worker=dict(pid=124+index, start_ticks=1, boot_id='cpu-fixture', command_sha256='b'*64)))
    report = verifier.verify_campaign(root, design)
    assert len(instances) == 4 and report['counts']['native_initializations'] == 4
    assert report['counts']['crafted_replays'] == 0
    assert report['counts']['strict_strong_failures'] == 0
    assert report['counts']['strict_command_only_failures'] == 0
    assert report['counts']['command_admitted_controller_failures'] == 0
    assert report['baseline_counts']['native_failures_with_both_constraints_pass'] == 1
    assert report['baseline_counts']['valid'] == 0
    assert report['baseline_diagnostics'][0]['result']['strong_counterexample'] is True
    assert report['baseline_diagnostics'][0]['result']['native_success'] is False
    for scene in report['scene_reports']:
        path = root/'scenes'/f"seed{scene['seed']}"
        for name, expected in scene['evidence_sha256'].items():
            assert expected == (contracts.file_hash(path/name) if (path/name).exists() else None)
    original_digest = contracts.digest(report)
    attempt_path = root/'scenes/seed13000/capture_attempt.json'
    attempt = contracts.read(attempt_path)
    attempt['seconds'] += 1  # Semantics unchanged; final proof bytes must differ.
    contracts.update_state(attempt_path, attempt)
    changed = verifier.verify_campaign(root, design)
    assert contracts.digest(changed) != original_digest
