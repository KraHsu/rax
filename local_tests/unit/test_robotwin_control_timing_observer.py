import importlib.util
import json
import sys
import threading

import numpy as np
import pytest

from services.robotwin.control_timing_observer import ControlTimingObserver, cpu_value, source_scope

FIXTURE = '''import numpy as np
class Backend:
    def __init__(self):self.count=0;self.fail=False
    def step(self):
        if self.fail:raise RuntimeError('original backend failure')
        self.count+=1
        return 'original-return'
class Scene:
    def __init__(self):self.physx_system=Backend();self.dt=.004
    def step(self):return self.physx_system.step()
    def get_timestep(self):return self.dt
    def set_timestep(self,value):self.dt=value
class Robot:
    def __init__(self):self.calls=[]
    def set_arm_joints(self,position,velocity,arm):
        self.calls.append(('arm',position,velocity,arm));return 'arm-return'
    def set_gripper(self,value,arm,per_step):
        self.calls.append(('grip',value,arm,per_step));return 'grip-return'
class Native:
    def __init__(self):self.scene=Scene();self.robot=Robot();self.pictures=[]
    def take_dense_action(self,control_seq,save_freq=-1):
        for row in control_seq:
            self.robot.set_arm_joints(row,np.zeros(2),'left')
            self.robot.set_gripper(.5,'right',0.)
            self.scene.step()
        return 'dense-return'
    def delay(self,n):
        for _ in range(n):self.scene.step()
    def take_action(self,row,action_type='ee'):
        self.scene.step();return 'take-return'
def sample(native):
    return {'left':{'pose':[0.,0.,0.,1.,0.,0.,0.],'gripper':0.},
            'right':{'pose':[0.,0.,0.,1.,0.,0.,0.],'gripper':1.}}
'''


@pytest.fixture
def fixture(tmp_path):
    p=tmp_path/'native_fixture.py';p.write_text(FIXTURE)
    spec=importlib.util.spec_from_file_location('timing_fixture',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,p


def observe(fixture,**kw):
    m,p=fixture;n=m.Native()
    return n,ControlTimingObserver(n,m.sample,phase=kw.pop('phase','expert_capture'),source_paths=[p],**kw)


def test_transparent_controls_targets_returns_and_samples(fixture):
    m,p=fixture;baseline=m.Native();n,o=observe(fixture)
    rows=np.array([[1.,2.],[3.,4.]])
    expected=baseline.take_dense_action(rows)
    original=vars(m.Backend)['step']
    with o:
        m.sample(n);result=n.take_dense_action(rows);m.sample(n);m.sample(n)
    assert result==expected=='dense-return' and n.scene.physx_system.count==baseline.scene.physx_system.count==2
    assert vars(m.Backend)['step'] is original
    for actual,reference in zip(n.robot.calls,baseline.robot.calls):
        assert actual[0]==reference[0]
        for a,b in zip(actual[1:],reference[1:]):
            if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
            else:assert a==b
    assert n.robot.calls[0][1] is rows[0] or np.shares_memory(n.robot.calls[0][1],rows)
    r=o.receipt();assert r['clock_complete'] and r['timing_compatibility_ready']
    assert [s['physics_step_completed'] for s in r['samples']]==[0,2,2]
    assert r['scene_wrapper_calls']==2 and r['direct_backend_calls']==0


def test_direct_backend_and_declaring_class_calls_are_counted(fixture):
    m,_=fixture;n,o=observe(fixture)
    with o:
        assert n.scene.physx_system.step()=='original-return'
        assert m.Backend.step(n.scene.physx_system)=='original-return'
        n.scene.step()
    assert o.receipt()['clock_complete']
    assert o.steps_completed==3 and o.direct_backend_calls==2 and o.scene_calls==1


def test_native_style_property_dt_setter_audited_and_observed(tmp_path):
    p=tmp_path/'property_fixture.py'
    source=FIXTURE.replace('    def get_timestep(self):return self.dt',
        '    @property\n    def timestep(self):return self.dt\n'
        '    @timestep.setter\n    def timestep(self,v):self.dt=v\n'
        '    def get_timestep(self):return self.timestep').replace(
        '    def set_timestep(self,value):self.dt=value','    def set_timestep(self,value):self.timestep=value')
    p.write_text(source);spec=importlib.util.spec_from_file_location('property_fixture',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);n=m.Native()
    with ControlTimingObserver(n,m.sample,phase='expert_capture',source_paths=[p]) as o:
        n.scene.timestep=.004;n.scene.step()
    assert o.scope['passed'] and not o.receipt()['clock_complete']
    assert 'timestep_setter_called_during_observation' in o.errors


def test_non_target_backend_unchanged_and_not_counted(fixture):
    m,_=fixture;n,o=observe(fixture);other=m.Backend()
    with o:
        assert other.step()=='original-return';n.scene.step()
    assert other.count==1 and o.steps_completed==1 and o.receipt()['clock_complete']


def test_exception_unwind_does_not_count_python_none_return_as_step(fixture):
    m,_=fixture;n,o=observe(fixture);original=vars(m.Backend)['step'];n.scene.physx_system.fail=True
    with pytest.raises(RuntimeError,match='original backend failure'):
        with o:n.scene.step()
    r=o.receipt();assert r['started_backend_steps']==1 and r['failed_backend_steps']==1 and r['completed_physics_steps']==0
    assert not r['clock_complete'] and vars(m.Backend)['step'] is original
    assert sys.getprofile() is None and threading.getprofile() is None


def test_cached_raw_method_is_not_claimed_detectable_source_audit_marks_scope(fixture,tmp_path):
    m,p=fixture;n=m.Native();cached=n.scene.physx_system.step
    alias=tmp_path/'cached_alias.py';alias.write_text('cached = backend.step\ncached()\n')
    o=ControlTimingObserver(n,m.sample,phase='expert_capture',source_paths=[p,alias])
    with o:cached()
    assert n.scene.physx_system.count==1 and o.steps_completed==0
    assert not o.receipt()['clock_complete'] and not o.scope['passed']
    assert 'not detectable' in o.receipt()['count_scope']


def test_source_audit_missing_tracked_method_file_is_incomplete(fixture,tmp_path):
    m,_=fixture;p=tmp_path/'unrelated.py';p.write_text('pass\n');n=m.Native()
    with ControlTimingObserver(n,m.sample,phase='expert_capture',source_paths=[p]) as o:n.scene.step()
    assert not o.receipt()['clock_complete']
    assert 'observed_method_source_missing_from_audit_scope' in o.errors


def test_legacy_rows_bind_completed_command_indices(fixture):
    m,_=fixture;n,o=observe(fixture,phase='legacy_eef_replay')
    with o:
        m.sample(n)
        for i in range(2):n.take_action([i]);m.sample(n)
    r=o.receipt();assert r['clock_complete']
    assert [v['command_index'] for v in r['samples']]==[0,1,2]
    assert [v['sample_index'] for v in r['samples']]==[0,1,2]


def test_unknown_argument_never_invokes_conversion(fixture):
    class Forbidden:
        def __array__(self,*a):raise AssertionError('No tensor/array conversion')
        def cpu(self):raise AssertionError('No CPU transfer')
        def __repr__(self):raise AssertionError('No repr')
    n,o=observe(fixture)
    with o:n.robot.set_arm_joints(Forbidden(),None,'left');n.scene.step()
    assert o.receipt()['clock_complete'] and not o.receipt()['control_arguments_complete']
    assert not o.receipt()['timing_compatibility_ready']


@pytest.mark.parametrize('failure', ['write','flush','close'])
def test_journal_failures_do_not_replace_native_return_or_exception(fixture,failure):
    class Broken:
        def write(self,_):
            if failure=='write':raise OSError('disk full')
        def flush(self):
            if failure=='flush':raise OSError('disk full')
        def close(self):
            if failure=='close':raise OSError('close failed')
    m,_=fixture;n,o=observe(fixture);original=vars(m.Backend)['step']
    with o:
        o.journal=Broken()
        assert n.scene.step()=='original-return'
    assert not o.receipt()['clock_complete'] and vars(m.Backend)['step'] is original
    n,o=observe(fixture);n.scene.physx_system.fail=True
    with pytest.raises(RuntimeError,match='original backend failure'):
        with o:
            o.journal=Broken();n.scene.step()
    assert not o.receipt()['clock_complete'] and vars(m.Backend)['step'] is original
    assert sys.getprofile() is None and threading.getprofile() is None


def test_existing_profiles_are_not_overwritten(fixture):
    n,o=observe(fixture);marker=lambda *args:None
    sys.setprofile(marker)
    try:
        with pytest.raises(RuntimeError,match='Existing profiler'):o.start()
        assert sys.getprofile() is marker
    finally:sys.setprofile(None)
    threading.setprofile(marker)
    try:
        with pytest.raises(RuntimeError,match='Existing profiler'):o.start()
        assert threading.getprofile() is marker
    finally:threading.setprofile(None)


def test_new_thread_backend_call_is_seen_but_scope_not_claimed_ordered(fixture):
    n,o=observe(fixture)
    with o:
        t=threading.Thread(target=n.scene.physx_system.step);t.start();t.join()
    assert o.steps_completed==1 and not o.receipt()['clock_complete']
    assert 'target_backend_step_on_nonowner_thread' in o.errors
    assert threading.getprofile() is None


def test_preexisting_unobserved_python_thread_is_explicitly_incomplete(fixture):
    ready=threading.Event();finish=threading.Event()
    def worker():ready.set();finish.wait()
    t=threading.Thread(target=worker);t.start();ready.wait()
    try:
        n,o=observe(fixture)
        with o:n.scene.step()
        assert not o.receipt()['clock_complete'] and t.ident in o.initial_other_threads
    finally:finish.set();t.join()


def test_timestep_change_even_restored_is_not_equivalent(fixture):
    n,o=observe(fixture)
    with o:n.scene.set_timestep(.01);n.scene.step();n.scene.set_timestep(.004)
    assert not o.receipt()['clock_complete']
    assert 'timestep_setter_called_during_observation' in o.errors


def test_journal_retains_prefix_and_observer_is_one_shot(fixture,tmp_path):
    p=tmp_path/'clock.jsonl';n,o=observe(fixture,journal_path=p)
    with o:n.scene.step()
    events=[json.loads(line) for line in p.read_text().splitlines()]
    assert len(events)==o.receipt()['event_count']
    assert events[-1]['kind']=='backend_complete'
    with pytest.raises(ValueError,match='one-shot'):o.start()
    assert o.receipt()['formal_evidence_eligible'] is False


def test_cpu_serialization_keeps_numpy_dtype_and_bytes():
    x=np.array([.1,.2],dtype=np.float32);out,ok=cpu_value(x)
    assert ok and out['dtype']=='<f4' and out['shape']==[2]
    assert cpu_value(np.array([object()],dtype=object))[1] is False


def test_source_scope_rejects_dynamic_alias_and_timestep_assignment(tmp_path):
    p=tmp_path/'bad.py';p.write_text("a=getattr(system,'step')\nsystem.timestep=.1\n")
    r=source_scope([p]);assert not r['passed'] and len(r['findings'])==2
