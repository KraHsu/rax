"""Observe source-bound expert planner requests without wrapping control methods.

Only already-CPU locals/results on the owner thread are copied. Python return
may be exception unwind; only a recognized returned status is classified.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

from services.robotwin.instruction_metadata import audited_native_source

VERSION = 'readonly_expert_planner_requests_v2'
STATUS_CODEC = 'bounded_planner_status_strings_v1'
MAX_STATUS_ITEMS = 4096
MAX_STATUS_STRING_LENGTH = 256
TASK = 'put_object_cabinet'
SOURCE_SHA256 = {
    TASK: '4c705f50df805965acad4d09257dc58af1aa3ecc903d3c34d117aa1b155d0b64',
    'base': 'aa9d717ad214b9d68c6eeb83ed03f2491db6addc98deaa988a48a3c074178bdd',
    'robot': '1f4d0284eaf47fe5de2c0ba98a173b78beb83f2f2cc43409d37072a4774df4c8',
}
REQUESTS = ('left_plan_path','right_plan_path','left_plan_multi_path','right_plan_multi_path')
SCOPES = ('move','left_move_to_pose','right_move_to_pose','together_move_to_pose',
          'choose_grasp_pose','grasp_actor','place_actor','move_by_displacement','get_place_pose')
PARAMETERS = ('target_pose','target_lst','constraint_pose','use_point_cloud','use_attach','last_qpos',
              'now_qpos','trans_target_pose','target_lst_copy','pose','left_target_pose','right_target_pose',
              'left_constraint_pose','right_constraint_pose')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy(*,solver_details=False):
    if type(solver_details)is not bool:raise ValueError('Explicit boolean solver-details opt-in required')
    result=dict(version=VERSION, task=TASK, source_sha256=SOURCE_SHA256,
                implementation_sha256=sha(__file__), max_events=2000,
                status_codec=dict(version=STATUS_CODEC,max_items=MAX_STATUS_ITEMS,
                                  max_string_codepoints=MAX_STATUS_STRING_LENGTH,
                                  object_array='exact ndarray, one dimensional, nonempty, exact Python str elements'),
                scope='Source-bound robot/base/task Python methods on the owner thread; no solver/GPU state reads')
    if solver_details:
        from services.robotwin.planner_result_evidence import policy as result_policy
        result.update(version='readonly_expert_planner_requests_v3',solver_details=result_policy(),
                      scope='Source-bound owner-thread Robot/base/task and scalar cuRobo wrapper returns; audited existing CPU locals only, no Tensor reads')
    return result


def cpu_value(value, depth=0):
    """No repr(), arbitrary properties, tensor transfer or array coercion."""
    if depth > 5:return {'observed':False,'reason':'nested_depth_limit'}
    if value is None or type(value) in (bool,int,str):return value
    if type(value) is float:
        return value if np.isfinite(value) else {'observed':False,'reason':'nonfinite'}
    if isinstance(value,np.generic) and value.dtype.kind in 'biufUS':return cpu_value(value.item(),depth+1)
    if type(value) is np.ndarray:
        if value.dtype.kind not in 'biufUS' or value.size>500000:
            return {'observed':False,'reason':'unsupported_or_large_CPU_array','shape':list(value.shape),'dtype':str(value.dtype)}
        raw=value.tobytes(order='C')
        return {'observed':True,'kind':'numpy','shape':list(value.shape),'dtype':str(value.dtype),
                'sha256':hashlib.sha256(raw).hexdigest(),'values':cpu_value(value.tolist(),depth+1) if value.size<=4096 else None}
    if type(value) in (tuple,list):
        if len(value)>4096:return {'observed':False,'reason':'sequence_limit','length':len(value)}
        return [cpu_value(v,depth+1) for v in value]
    if type(value) is dict:
        return {k:cpu_value(v,depth+1) for k,v in value.items() if type(k) is str}
    return {'observed':False,'reason':'non_CPU_whitelist_type','type':type(value).__name__}


def returned_status(value):
    if type(value) is not dict or 'status' not in value:return None
    status=value['status']
    if isinstance(status,np.ndarray):status=status.tolist()
    if isinstance(status,np.generic):status=status.item()
    if type(status) is str:return dict(values=[status],all_success=status=='Success')
    if type(status) in (tuple,list) and status and all(type(x) is str for x in status):
        return dict(values=list(status),all_success=all(x=='Success'for x in status))
    return None


def _status_strings(values):
    return (type(values)is list and 0<len(values)<=MAX_STATUS_ITEMS
            and all(type(x)is str and 0<len(x)<=MAX_STATUS_STRING_LENGTH for x in values))


def _status_content_hash(shape,values):
    # Object-array bytes contain pointers, not portable string content.
    payload=dict(codec=STATUS_CODEC,dtype='object',shape=shape,values=values)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()


def encode_request_return(value):
    """Retain only bounded exact-string object arrays in the status field.

    The generic CPU serializer still rejects every other object array. No
    str/repr/item/device conversion or mutation of a returned object is used.
    """
    retained=cpu_value(value)
    if type(value)is not dict or 'status'not in value:return retained
    status=value['status']
    if type(status)is np.ndarray and status.dtype.kind=='O':
        if status.ndim==1 and 0<status.size<=MAX_STATUS_ITEMS:
            values=list(status.flat)
            if _status_strings(values):
                shape=list(status.shape)
                retained['status']=dict(observed=True,kind=STATUS_CODEC,dtype='object',shape=shape,
                    values=values,content_sha256=_status_content_hash(shape,values))
                return retained
        retained['status']=dict(observed=False,reason='unsupported_object_status_array',
                                shape=list(status.shape),dtype=str(status.dtype))
    return retained


def retained_status(value):
    """Return (classification, error) from the retained CPU evidence alone."""
    if value is None:return None,None  # Python return/unwind remains ambiguous.
    if type(value)is not dict or 'status'not in value:return None,'missing_or_unrecognized_status'
    raw=value['status']
    if type(raw)is dict and raw.get('kind')==STATUS_CODEC:
        required={'observed','kind','dtype','shape','values','content_sha256'}
        if set(raw)!=required or raw['observed']is not True or raw['dtype']!='object':
            return None,'invalid_object_status_codec'
        shape=raw['shape'];values=raw['values']
        if (type(shape)is not list or len(shape)!=1 or type(shape[0])is not int
                or not _status_strings(values)or shape[0]!=len(values)):
            return None,'invalid_object_status_shape_or_values'
        if raw['content_sha256']!=_status_content_hash(shape,values):return None,'object_status_content_hash_mismatch'
    elif type(raw)is dict and raw.get('kind')=='numpy':
        # Preserve existing Unicode ndarray semantics, verifying the retained
        # array's shape and byte hash rather than trusting its derived label.
        try:
            dtype=np.dtype(raw['dtype']);shape=raw['shape'];values=raw['values']
            if (raw.get('observed')is not True or dtype.kind!='U' or dtype.itemsize>4*MAX_STATUS_STRING_LENGTH or type(shape)is not list
                    or len(shape)>1 or any(type(n)is not int or n<0 or n>MAX_STATUS_ITEMS for n in shape)):
                return None,'unsupported_numpy_status'
            sequence=[values]if shape==[] else values
            if not _status_strings(sequence):return None,'invalid_numpy_status_values'
            array=np.asarray(values,dtype=dtype)
            if list(array.shape)!=shape or array.tolist()!=values or hashlib.sha256(array.tobytes(order='C')).hexdigest()!=raw['sha256']:
                return None,'numpy_status_content_mismatch'
            values=sequence
        except (ValueError,TypeError,KeyError,OverflowError):return None,'invalid_numpy_status_codec'
    else:
        values=[raw]if type(raw)is str else raw
        if not _status_strings(values):return None,'unobserved_or_unsupported_status'
    return dict(values=list(values),all_success=all(x=='Success'for x in values)),None


class ExpertPlannerObserver:
    def __init__(self,native,*,frozen_policy):
        if frozen_policy!=policy(solver_details='solver_details'in frozen_policy):raise ValueError('Planner observation policy/source changed')
        audited_native_source(TASK,native,SOURCE_SHA256)
        self.native=native;self.policy=frozen_policy;self.events=[];self.errors=[];self.pending={};self.serial=0
        self.identities={};self.first_non_success=None;self.last_non_success=None;self.first_fatal_scope=None;self.callback_seconds=0.
        self.thread=threading.get_ident();self.previous=None;self.previous_threading=None;self.started=False;self.used=False
        for obj,names,source in [(native.robot,REQUESTS,'robot'),(native,SCOPES,'base'),(native,('play_once',),TASK)]:
            for name in names:
                fn=getattr(obj,name)
                if (getattr(fn,'__func__',None) is not getattr(type(obj),name) or fn.__name__!=name
                        or sha(inspect.getfile(fn))!=SOURCE_SHA256[source]):
                    raise ValueError('Source-bound planner/control method changed: '+name)
                self.identities[(fn.__func__.__code__,id(obj))]=(name,source)
        self.solver=None
        if 'solver_details'in frozen_policy:
            from services.robotwin.planner_result_evidence import Bindings
            self.solver=Bindings(native)
            for obj,fn,arm in self.solver.methods:
                self.identities[(fn.__code__,id(obj))]=('plan_path','curobo_wrapper')
        self.codes={k[0]for k in self.identities}

    def _stack(self,frame):
        rows=[];parent=frame.f_back
        while parent is not None and len(rows)<20:
            if parent.f_code in self.codes:
                rows.append(dict(function=parent.f_code.co_name,line=parent.f_lineno,filename=parent.f_code.co_filename))
            parent=parent.f_back
        return rows

    def _append(self,row):
        if len(self.events)>=self.policy['max_events']:
            if 'event_budget_exhausted'not in self.errors:self.errors.append('event_budget_exhausted')
            return
        self.events.append(row)

    def profile(self,frame,event,arg):
        if event not in ('call','return') or frame.f_code not in self.codes:return
        identity=self.identities.get((frame.f_code,id(frame.f_locals.get('self'))))
        if identity is None:return
        started=time.perf_counter()
        try:
            if threading.get_ident()!=self.thread:
                self.errors.append('unexpected_observer_thread');return
            name,source=identity;key=id(frame);flag=self.native.__dict__.get('plan_success')
            if event=='call':
                self.serial+=1
                row=dict(event='call',call_id=self.serial,method=name,source=source,line=frame.f_lineno,
                         plan_success=flag if type(flag)is bool else None,stack=self._stack(frame),
                         arguments={k:cpu_value(frame.f_locals[k])for k in PARAMETERS if k in frame.f_locals})
                if source=='curobo_wrapper':
                    parent=frame.f_back
                    while parent is not None:
                        p=self.pending.get(id(parent))
                        if p is not None and p['method']in REQUESTS:break
                        parent=parent.f_back
                    row['parent_request_call_id']=None if parent is None else p['call_id']
                    row['arm']=self.solver.by_object[id(frame.f_locals['self'])][0]
                self.pending[key]=row;self._append(row)
            else:
                entry=self.pending.pop(key,None)
                if entry is None:self.errors.append('return_without_observed_call');return
                if source=='curobo_wrapper'and type(arg)is dict:
                    retained={'status':cpu_value(arg['status'])}if 'status'in arg else None
                else:retained=encode_request_return(arg)if name in REQUESTS else cpu_value(arg)
                status,error=retained_status(retained)if name in REQUESTS else(None,None)
                row=dict(event='return_or_unwind',call_id=entry['call_id'],method=name,source=source,line=frame.f_lineno,
                         plan_success=flag if type(flag)is bool else None,returned_cpu_value=retained,
                         returned_status=status,status_observation_error=error,
                         parameters_at_return={k:cpu_value(frame.f_locals[k])for k in PARAMETERS if k in frame.f_locals})
                if source=='curobo_wrapper':
                    evidence=self.solver.collect(frame.f_locals['self'],frame.f_locals,entry['call_id'],entry['parent_request_call_id'])
                    row['solver_evidence']=evidence
                    if not evidence['complete']or entry['parent_request_call_id']is None:self.errors.append('incomplete_solver_evidence:'+str(entry['call_id']))
                self._append(row)
                if error is not None:self.errors.append('unobserved_status:'+str(entry['call_id'])+':'+error)
                if status is not None and not status['all_success']:
                    self.last_non_success=entry['call_id']
                    if self.first_non_success is None:self.first_non_success=entry['call_id']
                if entry['plan_success'] is True and flag is False and self.first_fatal_scope is None:
                    self.first_fatal_scope=dict(call_id=entry['call_id'],method=name,line=frame.f_lineno,
                        stack=entry['stack'],last_recognized_non_success_request=self.last_non_success,
                        meaning='Earliest observed source-bound scope returning with true-to-false plan_success. Not proof the first non-success request caused it.')
        except Exception as exc:self.errors.append(type(exc).__name__+': '+str(exc))
        finally:self.callback_seconds+=time.perf_counter()-started

    def start(self):
        if self.used:raise ValueError('Planner observer is one-shot')
        self.previous=sys.getprofile();self.previous_threading=threading.getprofile()
        if self.previous is not None:raise ValueError('Existing profiler must not be replaced')
        self.initial_other_threads=[t.name for t in threading.enumerate()if t.ident!=self.thread]
        self.started=True;self.used=True;sys.setprofile(self.profile);return self

    def stop(self):
        if self.started:
            current=sys.getprofile()
            if current!=self.profile:self.errors.append('owner_thread_profile_changed')
            sys.setprofile(self.previous);self.started=False
            if threading.getprofile() is not self.previous_threading:self.errors.append('threading_profile_changed')
        if self.pending:self.errors.append('unmatched_call_scopes')

    def receipt(self):
        result=dict(version=self.policy['version'],policy=self.policy,events=self.events,errors=self.errors,
            first_non_success_request_id=self.first_non_success,first_true_to_false_scope=self.first_fatal_scope,
            callback_seconds=self.callback_seconds,initial_other_threads=getattr(self,'initial_other_threads',[]),
            owner_thread_scope_complete=not self.errors,
            all_process_or_solver_calls_observed=False,threading_profiler_modified=False,
            control_methods_wrapped=False,original_arguments_modified=False,extra_state_getters=0,
            extra_physics_steps=0,extra_get_info_calls=0,extra_predicate_calls=0,
            timing_limit='callback_seconds excludes dispatch/filter overhead and later JSON serialization; no native A/B equivalence claim',
            python_return_limit='return_or_unwind includes exception unwinds; None is never classified as a returned planner failure',
            formal_evidence_eligible=False)
        if self.solver is not None:result['solver_contexts']=self.solver.contexts
        return result
