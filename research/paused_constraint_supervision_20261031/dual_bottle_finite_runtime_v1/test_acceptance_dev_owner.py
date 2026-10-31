import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE=Path(__file__).parent/'package/owner/dual_bottle_owner.py'
spec=importlib.util.spec_from_file_location('reviewed_owner',SOURCE)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(owner,'PACKAGE',tmp_path)
    job=dict(task='pick_dual_bottles',gpu=0,output='runs/pick_dual_bottles')
    plan=dict(version=1,kind='dual_bottle_finite_development_owner_v1',task='pick_dual_bottles',new_native_initializations_max=63,legacy_replays=0,automatic_retries=0,previous_failures_immutable=True,package_root=str(tmp_path),rax_commit=owner.COMMIT,
              lease_wait_timeout_seconds=43200,execution_timeout_seconds=10800,
              python=owner.NATIVE_PYTHON,native_root='/native',
              lease_root='/mnt/data/liangjiaqi/rax_train/experiments/.robotwin_dev_gpu_leases',jobs=[job],design='design/runtime_design.json',seeds=[13000,13001,13002])
    driver=SimpleNamespace(run=lambda args:0)
    monkeypatch.setattr(owner,'verify_inputs',lambda *args:(driver,{},{}))
    return plan,job,tmp_path

@pytest.mark.parametrize('mutation',[lambda j:j.update(task='unreviewed'),lambda j:j.update(gpu=1),lambda j:j.update(command=['rm','-rf','/']),lambda j:j.update(seed_start=11000)])
def test_fixed_command_rejects_mutations(setup,mutation):
    plan,job,_=setup;mutation(job)
    with pytest.raises(ValueError):owner.validate_plan(plan,'pick_dual_bottles')

def test_existing_intent_never_restarts(setup):
    plan,job,p=setup;(p/'ownership/pick_dual_bottles').mkdir(parents=True)
    (p/'ownership/pick_dual_bottles/intent.json').write_text('{}')
    with pytest.raises(FileExistsError):owner.run(plan,job,False)

def test_existing_output_never_restarts(setup):
    plan,job,p=setup;(p/'runs/pick_dual_bottles').mkdir(parents=True)
    assert owner.run(plan,job,False)==2
    assert json.loads((p/'ownership/pick_dual_bottles/state.json').read_text())['phase']=='blocked_existing'

def test_nonzero_native_is_failure_and_releases_lease(setup,monkeypatch):
    plan,job,p=setup;leases=p/'leases';leases.mkdir();plan['lease_root']=str(leases)
    monkeypatch.setattr(owner,'gpu_occupancy',lambda gs:{g:[] for g in gs})
    logs=[]
    class Registry:
        def add(self,child,log):logs.append(log)
        def terminate_all(self):
            for log in logs:log.close()
    monkeypatch.setattr(owner,'ProcessRegistry',Registry)
    monkeypatch.setattr(owner,'spawn_resource',lambda *a,**kw:SimpleNamespace(pid=os.getpid(),returncode=2,poll=lambda:2))
    monkeypatch.setattr(owner,'checked_completion',lambda *a:pytest.fail('Must not accept nonzero native'))
    assert owner.run(plan,job,False)==2
    assert json.loads((p/'ownership/pick_dual_bottles/state.json').read_text())['phase']=='failed'
    import fcntl
    with (leases/'gpu_0.lock').open('a+') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)

def test_blocked_native_cannot_be_completed(setup,monkeypatch):
    _,_,p=setup
    (p/'design').mkdir();(p/'design/runtime_design.json').write_text('{}')
    (p/'state.json').write_text('{"status":"infrastructure_failed"}')
    import sys
    monkeypatch.setitem(sys.modules,'verifier',SimpleNamespace(verify_campaign=lambda *a:{}))
    with pytest.raises(ValueError,match='complete output'):
        owner.checked_completion(None,{},None,p)


def test_valid_finite_plan_and_command(setup):
    plan,job,p=setup
    assert owner.validate_plan(plan,'pick_dual_bottles') is job
    cmd=owner.command(plan,job)
    assert cmd[-1]=='--execute' and cmd.count('--design')==1 and '--task' not in cmd and '10001' not in cmd



def test_hung_owned_child_expires_and_cleans_only_registry(setup,monkeypatch):
    plan,job,p=setup;leases=p/'leases';leases.mkdir();plan['lease_root']=str(leases)
    now=[0.];monkeypatch.setattr(owner.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(owner,'gpu_occupancy',lambda gs:{g:[] for g in gs})
    logs=[];cleanup=[]
    class Registry:
        def add(self,child,log):logs.append(log)
        def terminate_all(self):
            cleanup.append('owned_only')
            for log in logs:log.close()
    class Event:
        def is_set(self):return False
        def set(self):pass
        def wait(self,seconds):now[0]=10801.;return False
    monkeypatch.setattr(owner,'ProcessRegistry',Registry)
    monkeypatch.setattr(owner.threading,'Event',Event)
    monkeypatch.setattr(owner,'spawn_resource',lambda *a,**k:SimpleNamespace(pid=os.getpid(),poll=lambda:None))
    monkeypatch.setattr(owner,'checked_completion',lambda *a:pytest.fail('Timed-out child cannot complete'))
    assert owner.run(plan,job,False)==2
    assert json.loads((p/'ownership/pick_dual_bottles/state.json').read_text())['phase']=='expired_execution'
    assert cleanup==['owned_only']
    import fcntl
    with (leases/'gpu_0.lock').open('a+') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)




@pytest.mark.parametrize('key,value',[('new_native_initializations_max',2),('legacy_replays',1),('automatic_retries',1),('previous_failures_immutable',False)])
def test_history_or_budget_cannot_be_hidden(setup,key,value):
    plan,job,_=setup;plan[key]=value
    with pytest.raises(ValueError,match='budget/no-retry'):owner.validate_plan(plan,job['task'])


@pytest.mark.parametrize('task',['pick_dual_bottles','put_object_cabinet'])
def test_each_intent_has_new_budget_without_stale_history(setup,monkeypatch,task):
    plan,job,p=setup;plan['task']=task;job.update(task=task,output='runs/'+task)
    if task != 'pick_dual_bottles':
        with pytest.raises(ValueError,match='Wrong finite task'):
            owner.validate_plan(plan,task)
        assert not (p/'ownership').exists()
        return
    class Stopped:
        def is_set(self):return True
        def set(self):pass
    monkeypatch.setattr(owner.threading,'Event',Stopped)
    monkeypatch.setattr(owner,'spawn_resource',lambda *a,**k:pytest.fail('Stopped owner must not launch'))
    monkeypatch.setattr(owner,'gpu_occupancy',lambda *a:pytest.fail('No GPU access in this intent test'))
    assert owner.validate_plan(plan,task)is job and owner.run(plan,job,False)==2
    intent=json.loads((p/'ownership'/task/'intent.json').read_text())
    assert intent['task']==task and intent['new_native_initializations_max']==63 and intent['legacy_replays']==0
    assert 'startup_attempt'not in intent and 'previous_native_initializations'not in intent
    assert json.loads((p/'ownership'/task/'state.json').read_text())['phase']=='interrupted'
