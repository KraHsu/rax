"""Future-only codec; the immutable v1 native evidence remains incomplete."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.robotwin import expert_planner_observer as planner
from scripts.observe_robotwin_expert_failure import validate_planner_receipt
from test_robotwin_failure_observation import planner_native

FIX=Path(__file__).parents[1]/'fixtures/robotwin_planner_status'


def test_real_v1_profile_has_five_permanently_missing_raw_status_arrays():
    source=(FIX/'failed_v1_profile.json.gz').read_bytes();meta=json.loads((FIX/'provenance.json').read_text());raw=gzip.decompress(source)
    assert hashlib.sha256(source).hexdigest()==meta['compressed_fixture_sha256']
    assert hashlib.sha256(raw).hexdigest()==meta['original_profile_sha256']
    receipt=json.loads(raw);assert len(receipt['events'])==92
    missing=[]
    for row in receipt['events']:
        if row['event']=='return_or_unwind'and row['method']in planner.REQUESTS:
            classified,error=planner.retained_status(row['returned_cpu_value'])
            if error:
                missing.append(row['call_id']);assert classified is None
                assert row['returned_cpu_value']['status']['observed']is False
                assert 'values'not in row['returned_cpu_value']['status']
            else:assert classified==row['returned_status']
    assert missing==meta['missing_raw_status_calls']==[4,5,6,14,15]
    with pytest.raises(ValueError,match='legacy missing values'):
        validate_planner_receipt(receipt,receipt['policy'])
    assert gzip.decompress(source)==raw  # No backfill or revision of original bytes.


@pytest.mark.parametrize('status',[
    'Success','Fail',np.str_('Failure'),['Success','Fail'],('Success','Success'),
    np.array('Success'),np.array(['Success','Failure']),
    np.array(['Success']*10,dtype=object),np.array(['Failure']+['Success']*9,dtype=object),
])
def test_supported_statuses_derive_from_same_retained_value(status):
    original={'status':status};raw=planner.encode_request_return(original)
    assert original['status']is status
    value,error=planner.retained_status(json.loads(json.dumps(raw,allow_nan=False)))
    assert error is None and value==planner.returned_status(original)
    if type(status)is np.ndarray and status.dtype.kind=='O':
        assert raw['status']['kind']==planner.STATUS_CODEC and raw['status']['dtype']=='object'
        assert 'sha256'not in raw['status']  # No pointer-byte digest.
        assert raw['status']['shape']==list(status.shape)


class Untouchable:
    def __str__(self):raise AssertionError('str called')
    def __repr__(self):raise AssertionError('repr called')
    def item(self):raise AssertionError('item called')
    def __array__(self,*a):raise AssertionError('array called')
    def cpu(self):raise AssertionError('cpu called')


@pytest.mark.parametrize('kind',['mixed','custom','scalar','matrix','empty','oversized','long','nan','none','np_string_object'])
def test_unsupported_status_never_becomes_success_or_executes_object_methods(kind):
    if kind=='mixed':status=np.array(['Success',1],dtype=object)
    elif kind=='custom':
        status=np.empty(1,dtype=object);status[0]=Untouchable()
    elif kind=='scalar':status=np.array('Success',dtype=object)
    elif kind=='matrix':status=np.array([['Success']],dtype=object)
    elif kind=='empty':status=np.array([],dtype=object)
    elif kind=='oversized':status=np.array(['Success']*(planner.MAX_STATUS_ITEMS+1),dtype=object)
    elif kind=='long':status=np.array(['x'*(planner.MAX_STATUS_STRING_LENGTH+1)],dtype=object)
    elif kind=='nan':status=float('nan')
    elif kind=='none':status=None
    else:
        status=np.empty(1,dtype=object);status[0]=np.str_('Success')
    raw=planner.encode_request_return({'status':status});value,error=planner.retained_status(raw)
    assert value is None and error is not None
    assert planner.retained_status(None)==(None,None)  # Genuine return/unwind stays distinct.


def test_generic_object_arrays_remain_unobserved_and_content_hash_is_portable():
    a=np.array(['Success','Fail'],dtype=object);b=np.array(['Suc'+'cess','F'+'ail'],dtype=object)
    assert planner.cpu_value(a)['observed']is False
    assert planner.encode_request_return({'status':a})==planner.encode_request_return({'status':b})
    assert planner.encode_request_return({'other':a})['other']['observed']is False


@pytest.mark.parametrize('change',['values','shape','dtype','digest','derived','missing_error'])
def test_object_codec_tamper_rejected_from_live_source_receipt(monkeypatch,change):
    n=planner_native(monkeypatch);out={'status':np.array(['Success','Failure'],dtype=object)}
    n.robot.left_planner=SimpleNamespace(plan_batch=lambda *a,**k:out)
    obs=planner.ExpertPlannerObserver(n,frozen_policy=planner.policy()).start()
    try:returned=n.robot.left_plan_multi_path([[0,0,.8,1,0,0,0]])
    finally:obs.stop()
    assert returned is out and n.plan_success
    r=obs.receipt();validate_planner_receipt(r,planner.policy())
    row=r['events'][-1];raw=row['returned_cpu_value']['status']
    if change=='values':raw['values'][1]='Success'
    elif change=='shape':raw['shape']=[3]
    elif change=='dtype':raw['dtype']='float64'
    elif change=='digest':raw['content_sha256']='0'*64
    elif change=='derived':row['returned_status']=None;r['first_non_success_request_id']=None
    else:
        row['returned_cpu_value']['status']={'observed':False,'reason':'missing'}
        row['returned_status']=None;row['status_observation_error']='unobserved_or_unsupported_status';r['first_non_success_request_id']=None
    with pytest.raises(ValueError):validate_planner_receipt(r,planner.policy())


def test_batch_source_control_args_return_and_unknown_evidence_completeness(monkeypatch):
    n=planner_native(monkeypatch);calls=[];out={'status':np.array(['Success']*10,dtype=object)}
    def plan(*a,**k):calls.append((deepcopy(a),deepcopy(k)));return out
    n.robot.left_planner=SimpleNamespace(plan_batch=plan);target=[[0.,0.,.8,1.,0.,0.,0.]]*10
    baseline=n.robot.left_plan_multi_path(target);before=deepcopy(target)
    obs=planner.ExpertPlannerObserver(n,frozen_policy=planner.policy()).start()
    try:actual=n.robot.left_plan_multi_path(target)
    finally:obs.stop()
    assert baseline is actual is out and target==before
    assert np.array_equal(calls[0][0][0],calls[1][0][0])and calls[0][0][1]==calls[1][0][1]and calls[0][1]==calls[1][1]
    assert obs.receipt()['owner_thread_scope_complete'];validate_planner_receipt(obs.receipt(),planner.policy())
    out['status']=np.array(['Success',None],dtype=object)
    obs=planner.ExpertPlannerObserver(n,frozen_policy=planner.policy()).start()
    try:assert n.robot.left_plan_multi_path(target)is out
    finally:obs.stop()
    receipt=obs.receipt();assert not receipt['owner_thread_scope_complete']and receipt['first_non_success_request_id']is None
    validate_planner_receipt(receipt,planner.policy())
