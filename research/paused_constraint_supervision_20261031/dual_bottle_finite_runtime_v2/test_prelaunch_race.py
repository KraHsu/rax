"""Real foreign process + flock: busy-after-preflight waits, then launches once."""
import fcntl
import json
import subprocess
import sys
import time
from test_acceptance_dev_owner import owner, setup as setup_fixture
setup = setup_fixture


def test_busy_after_preflight_releases_real_flock_and_launches_once(setup,monkeypatch):
    plan,job,p=setup;leases=p/'leases';leases.mkdir();plan['lease_root']=str(leases)
    foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
    calls=[];spawned=[];wait_states=[]
    def occupancy(gs):
        calls.append(1)
        # Initial check free, race occupied, next initial check still occupied,
        # then free twice so exactly one actual owned child may start.
        return {g:([foreign.pid] if len(calls) in (2,3) else []) for g in gs}
    monkeypatch.setattr(owner,'gpu_occupancy',occupancy)
    real_spawn=owner.spawn_resource
    def spawn(_command,**kwargs):
        spawned.append(1);assert foreign.poll() is None
        kwargs['cwd']=str(p)
        program=f"from pathlib import Path; p=Path({str(p/'launch_count')!r}); p.open('a').write('launch\\n')"
        return real_spawn([sys.executable,'-c',program],**kwargs)
    monkeypatch.setattr(owner,'spawn_resource',spawn)
    monkeypatch.setattr(owner,'checked_completion',lambda *args:{'phase':'completed_development'})
    class Event:
        def is_set(self):return False
        def set(self):pass
        def wait(self,_seconds):
            if not spawned:
                state=json.loads((p/'ownership/pick_dual_bottles/state.json').read_text());wait_states.append(state)
                assert state['occupants']==[foreign.pid]
                assert not (p/'runs/pick_dual_bottles').exists()
                assert not (p/'ownership/pick_dual_bottles/native.log').exists()
                with (leases/'gpu_1.lock').open('a+') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                assert foreign.poll() is None
            time.sleep(.02)
            return False
    monkeypatch.setattr(owner.threading,'Event',Event)
    try:
        assert owner.run(plan,job,False)==0
        assert len(spawned)==1 and (p/'launch_count').read_text()=='launch\n'
        assert [s['reason'] for s in wait_states]==['gpu_occupied_after_preflight','gpu_occupied']
        assert len(calls)==5 and foreign.poll() is None
        history=[json.loads(line) for line in (p/'ownership/pick_dual_bottles/occupancy_waits.jsonl').read_text().splitlines()]
        assert [r['occupants'] for r in history]==[[foreign.pid],[foreign.pid]]
        assert json.loads((p/'ownership/pick_dual_bottles/state.json').read_text())['phase']=='completed_development'
    finally:
        foreign.terminate();foreign.wait(timeout=5)


def test_race_wait_uses_original_deadline_without_any_child(setup,monkeypatch):
    plan,job,p=setup;leases=p/'leases';leases.mkdir();plan['lease_root']=str(leases)
    times=iter([0,1,12*3600+1]);monkeypatch.setattr(owner.time,'monotonic',lambda:next(times,12*3600+1))
    inventory=iter([[],[123456]])
    monkeypatch.setattr(owner,'gpu_occupancy',lambda gs:{g:next(inventory) for g in gs})
    monkeypatch.setattr(owner,'spawn_resource',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No child before deadline')))
    class Event:
        def is_set(self):return False
        def set(self):pass
        def wait(self,_seconds):
            assert not (p/'runs/pick_dual_bottles').exists()
            with (leases/'gpu_1.lock').open('a+') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return False
    monkeypatch.setattr(owner.threading,'Event',Event)
    assert owner.run(plan,job,False)==2
    assert json.loads((p/'ownership/pick_dual_bottles/state.json').read_text())['phase']=='expired_wait'


def test_cpu_preflight_expiring_deadline_cannot_launch_even_on_free_gpu(setup,monkeypatch):
    plan,job,p=setup;leases=p/'leases';leases.mkdir();plan['lease_root']=str(leases)
    times=iter([0,1,12*3600+1]);monkeypatch.setattr(owner.time,'monotonic',lambda:next(times,12*3600+1))
    monkeypatch.setattr(owner,'gpu_occupancy',lambda gs:{g:[] for g in gs})
    monkeypatch.setattr(owner,'spawn_resource',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No launch after deadline')))
    assert owner.run(plan,job,False)==2
    assert not (p/'runs/pick_dual_bottles').exists()
    assert json.loads((p/'ownership/pick_dual_bottles/state.json').read_text())['phase']=='expired_wait'
    with (leases/'gpu_1.lock').open('a+') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)


def test_stop_during_cpu_preflight_cannot_launch_on_free_gpu(setup,monkeypatch):
    plan,job,p=setup;leases=p/'leases';leases.mkdir();plan['lease_root']=str(leases)
    event=owner.threading.Event();monkeypatch.setattr(owner.threading,'Event',lambda:event)
    original=owner.verify_inputs;checks=[]
    def verify(*args):
        checks.append(1)
        if len(checks)==2:event.set()
        return original(*args)
    monkeypatch.setattr(owner,'verify_inputs',verify)
    monkeypatch.setattr(owner,'gpu_occupancy',lambda gs:{g:[] for g in gs})
    monkeypatch.setattr(owner,'spawn_resource',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No launch after stop')))
    assert owner.run(plan,job,False)==2 and len(checks)==2
    assert not (p/'runs/pick_dual_bottles').exists()
    assert json.loads((p/'ownership/pick_dual_bottles/state.json').read_text())['phase']=='interrupted'
    with (leases/'gpu_1.lock').open('a+') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
