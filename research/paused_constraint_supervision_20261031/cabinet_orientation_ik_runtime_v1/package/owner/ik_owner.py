#!/usr/bin/env python3
"""One-shot owner for fixed 12-slot standalone IK diagnostic; no native episodes."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'code/owner_dependencies'))
from scripts.robotwin_dev_queue import atomic, file_hash, gpu_occupancy, process_identity, same_process
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource

TASK_GPUS = {'cabinet_orientation_ik':0}
COMMIT = 'standalone_orientation_ik_v1_source_frozen_by_plan'
NATIVE_PYTHON = '/mnt/data/liangjiaqi/yzm/miniconda3/envs/RoboTwin/bin/python'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def contained(relative):
    path=(PACKAGE/relative).resolve()
    if Path(relative).is_absolute()or not path.is_relative_to(PACKAGE)or path==PACKAGE:raise ValueError('Contained package path required')
    return path


def validate_plan(plan,task):
    if (plan.get('version')!=1 or plan.get('kind')!='cabinet_orientation_ik_runtime_v1'
        or Path(plan['package_root']).resolve()!=PACKAGE or task not in TASK_GPUS
        or plan.get('task')!=task):raise ValueError('Wrong fixed diagnostic/root')
    if (plan.get('new_native_initializations_max')!=0 or plan.get('legacy_replays')!=0
        or plan.get('automatic_retries')!=0 or plan.get('solver_calls_max')!=12
        or plan.get('seeds_per_call')!=32 or plan.get('previous_failures_immutable')is not True):raise ValueError('Fixed diagnostic budget changed')
    if plan.get('lease_wait_timeout_seconds')!=43200 or plan.get('execution_timeout_seconds')!=900:raise ValueError('Fixed12h wait/15min execution required')
    if plan.get('python')!=NATIVE_PYTHON or len(plan['jobs'])!=1:raise ValueError('Single diagnostic job required')
    job=plan['jobs'][0]
    if set(job)!={'task','gpu','output'} or job['task']!=task or job['gpu']!=0 or contained(job['output'])!=PACKAGE/'runs'/task:raise ValueError('Reviewed GPU0/job/output required')
    if plan['lease_root']!='/mnt/data/liangjiaqi/rax_train/experiments/.robotwin_dev_gpu_leases' or plan['design']!='design/probe_design.json':raise ValueError('Fixed lease/design required')
    return job


def verify_inputs(plan,job):
    for relative,h in plan['frozen_files'].items():
        if file_hash(contained(relative))!=h:raise ValueError('Frozen input changed: '+relative)
    for absolute,h in plan['external_files'].items():
        if file_hash(absolute)!=h:raise ValueError('External source/history changed: '+absolute)
    required={'owner/ik_owner.py','runtime/driver.py','runtime/contracts.py','runtime/geometry.py','runtime/gpu_backend.py','runtime/source_audit.py','runtime/orientation_contract.py',
              'design/probe_design.json','design/targets.json','design/native_mapping.json','design/integration_design.json',
              'design/orientation_inputs/orientation_proposal.json','design/orientation_inputs/expert_attempt.json',
              'design/orientation_inputs/expert_final_physical_state.json','design/orientation_inputs/planner.py',
              'design/orientation_inputs/runtime_configs.json','code/owner_dependencies/benchmarks/utils/owned_processes.py',
              'code/owner_dependencies/benchmarks/utils/resource_process.py','code/owner_dependencies/scripts/robotwin_dev_queue.py'}
    if not required<=set(plan['frozen_files']):raise ValueError('Required finite source absent')
    spec=importlib.util.spec_from_file_location('frozen_ik_driver',PACKAGE/'runtime/driver.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    driver.preflight(PACKAGE,external=True)
    return driver,read(contained(plan['design'])),None


def command(plan,job):
    return [plan['python'],str(PACKAGE/'runtime/driver.py'),'--design',str(contained(plan['design'])),
            '--output',str(contained(job['output'])),'--execute']


def checked_completion(driver,design,_unused,output):
    report=driver.verify_output(output)
    if report['status'] not in ('completed_diagnostic','blocked_known_fk_control'):raise ValueError('Diagnostic evidence incomplete; no retry')
    if any(report[k]is not False for k in ('native_task_label_available','formal_evidence_eligible','acceptance_ready','publication_allowed','training_allowed')):raise ValueError('Diagnostic cannot be acceptance')
    return dict(report,phase=report['status'],automatic_retries=0)


def run(plan, job, check_only):
    driver, exclusions, template = verify_inputs(plan, job)
    args=SimpleNamespace(design=contained(plan['design']),task=job['task'],
                         output=PACKAGE/'unused_preflight_output',check_only=True,execute=False)
    driver.run(args)
    if check_only:
        print(json.dumps({'checked':job['task'],'gpu_started':False,'intended_gpu':job['gpu'],
                          'native_episodes_max':0,'solver_calls_max':12,'plan_role':'standalone_ik_diagnostic_only'}))
        return 0
    directory = PACKAGE/'ownership'/job['task']
    directory.parent.mkdir(parents=True,exist_ok=True)
    directory.mkdir()  # Existing intent/state, including failure, is never retried.
    output = contained(job['output'])
    if output.exists():
        atomic(directory/'state.json',{'phase':'blocked_existing','reason':'Run output already exists'})
        return 2
    stop = threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,lambda *_:stop.set())
    intent = {'task':job['task'],'gpu':job['gpu'],'owner':process_identity(os.getpid()),
              'command':command(plan,job),'created_unix':time.time(),
              'plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              'role':'diagnostic_only','automatic_retries':0,'new_native_initializations_max':0,'legacy_replays':0,'solver_calls_max':12}
    atomic(directory/'intent.json',intent)
    registry, lease, child = ProcessRegistry(), None, None
    try:
        deadline = time.monotonic()+plan['lease_wait_timeout_seconds']
        while not stop.is_set() and time.monotonic()<deadline:
            handle = (Path(plan['lease_root'])/f'gpu_{job["gpu"]}.lock').open('a+')
            occupants, reason = None, 'shared_lease_busy'
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                handle.seek(0); previous=handle.read()
                if previous and same_process(json.loads(previous).get('process')):
                    reason = 'prior_guardian_still_alive'
                else:
                    occupants = gpu_occupancy([job['gpu']])[job['gpu']]
                    if occupants:
                        reason = 'gpu_occupied'
                    else:
                        lease=handle
            except BlockingIOError:
                pass
            finally:
                if lease is not handle:handle.close()
            if lease is not None:
                driver,exclusions,template=verify_inputs(plan,job)
                occupants = gpu_occupancy([job['gpu']])[job['gpu']]
                if not occupants:
                    break
                # An external process can arrive while CPU input checks run.
                # No output or child exists yet: release and wait on the SAME deadline.
                lease.close(); lease=None
                reason = 'gpu_occupied_after_preflight'
            waiting={'phase':'waiting_for_free_gpu','gpu':job['gpu'],
                     'occupants':occupants,'reason':reason,'updated_unix':time.time()}
            atomic(directory/'state.json',waiting)
            if occupants:
                with (directory/'occupancy_waits.jsonl').open('a') as journal:
                    journal.write(json.dumps(waiting)+'\n')
            stop.wait(10)
        if lease is None or stop.is_set() or time.monotonic() >= deadline:
            atomic(directory/'state.json',{'phase':'interrupted' if stop.is_set() else 'expired_wait','automatic_retries':0})
            return 2
        # Child owns the one-shot output mkdir; parent must not pre-create it.
        output.parent.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(job['gpu']),ROBOTWIN_PATH=plan['native_root'],
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                 VK_ICD_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
                 VK_DRIVER_FILES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
        log=(directory/'native.log').open('x')
        execution_deadline = time.monotonic()+plan['execution_timeout_seconds']
        try:
            child=spawn_resource(command(plan,job),cwd=str(PACKAGE/'runtime'),env=env,stdout=log,pass_fds=(lease.fileno(),))
        except BaseException:
            log.close();raise
        registry.add(child,log)
        identity=process_identity(child.pid)
        if identity is None:raise RuntimeError('Guardian exited before identity receipt')
        lease.seek(0);lease.truncate();lease.write(json.dumps({'process':identity,'owner':intent['owner'],'task':job['task']}));lease.flush()
        atomic(directory/'launch.json',dict(intent,guardian=identity,guardian_pid=child.pid))
        atomic(directory/'state.json',{'phase':'running_owned','guardian':identity,'gpu':job['gpu'],
                                                'execution_timeout_seconds':plan['execution_timeout_seconds']})
        while child.poll() is None and not stop.wait(5):
            if time.monotonic() >= execution_deadline:
                atomic(directory/'state.json', {'phase':'expired_execution',
                       'execution_timeout_seconds':plan['execution_timeout_seconds'],'automatic_retries':0})
                return 2
        if stop.is_set():
            atomic(directory/'state.json',{'phase':'interrupted','automatic_retries':0});return 2
        if child.returncode!=0:
            atomic(directory/'state.json',{'phase':'failed','returncode':child.returncode,'automatic_retries':0});return 2
        atomic(directory/'state.json',checked_completion(driver,exclusions,template,output))
        return 0
    except BaseException as exc:
        atomic(directory/'state.json',{'phase':'failed','error':f'{type(exc).__name__}: {exc}','automatic_retries':0})
        raise
    finally:
        try:
            registry.terminate_all()
        finally:
            if lease is not None:lease.close()  # Never LOCK_UN: guardian inherits the lease until cleanup ends.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path);parser.add_argument('--task',required=True,choices=TASK_GPUS)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();plan=read(args.plan)
    raise SystemExit(run(plan,validate_plan(plan,args.task),args.check_only))


if __name__=='__main__':main()
