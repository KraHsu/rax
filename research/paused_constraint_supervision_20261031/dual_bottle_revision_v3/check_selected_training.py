"""CPU compiled-schema, every reference chunk, gradients and historical-label audit."""
import copy,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
P=Path(__file__).resolve().parent;W=P.parents[2];sys.path[:0]=[str(W/'.worktrees/rax-dual-bottle-calibration'),str(W/'OpenWAM')]
from services.robotwin.programmatic_annotation import instantiate,template_hash
from services.robotwin.constraint_schema import HDF5_KEY
from services.robotwin.annotation import rotation
from openwam.train.constraint_schema import validate_bundle, SCHEMA_VERSION
from openwam.train.constraint_coverage import chunk_candidate
from openwam.train.meta_constraints import residual,penalty
class Handle(dict):pass
def h(r):
 v=Handle({HDF5_KEY:np.asarray(json.dumps(r['bundle']))});v.attrs={'robotwin_expert_provenance':json.dumps(r['expert_provenance'])}
 for a,p in r['arms'].items():v['endpose/'+a+'_endpose']=np.asarray(p['pose']);v['endpose/'+a+'_gripper']=np.asarray(p['gripper'])
 return v
def physical(arms):
 n=len(arms['left']['pose']);x=torch.zeros(n,20,dtype=torch.float64)
 for arm,k in [('left',0),('right',10)]:
  p=np.asarray(arms[arm]['pose']);q=p[:,3:];r=rotation(q/np.linalg.norm(q,axis=-1,keepdims=True));x[:,k:k+3]=torch.tensor(p[:,:3]);x[:,k+3:k+6]=torch.tensor(r[:,:,0]);x[:,k+6:k+9]=torch.tensor(r[:,:,1]);x[:,k+9]=torch.tensor(arms[arm]['gripper'])
 return x
train=json.loads((P.parent/'dual_bottle_revision_v2/training_inputs.json').read_text());parent=train['registries'][0]['entry']['template'];chosen='open0.60__axis030__closure0.021__hold0.021__closureX0.015';template=json.loads((P/'axial_candidates'/chosen/'template.json').read_text());rows=[];totals=Counter();history=[];first=None
for item in train['rows']:
 handle=h(item);old,_=instantiate(handle,parent);new,audit=instantiate(handle,template);b=copy.deepcopy(item['bundle']);b['constraints']=new;b['schema_version']=SCHEMA_VERSION;validate_bundle(b,len(item['arms']['left']['pose']))
 raw=physical(item['arms']);n=len(raw);bad=[];terms=0
 for start in range(n-1):
  end=min(n,start+33);x=raw[start:end];valid=torch.ones_like(x,dtype=torch.bool)
  for c in new:
   if not chunk_candidate(c,start,len(x)-1):continue
   v=residual(x,x,valid,c,start_frame=start,rotation_convention='physical')
   if v is not None:
    terms+=1
    if float(v.max())>1e-6:bad.append(dict(start=start,id=c['id'],max_violation=float(v.max())))
 totals['windows']+=n-1;totals['terms']+=terms;totals['rejected_reference_windows']+=len({x['start'] for x in bad})
 oldnamed={c['id']:{k:v for k,v in c.items() if k!='evidence'} for c in old if not c['id'].startswith('__full_motion')};storednamed={c['id']:{k:v for k,v in c.items() if k!='evidence'} for c in item['bundle']['constraints'] if not c['id'].startswith('__full_motion')}
 storedauto=[c for c in item['bundle']['constraints'] if c['id'].startswith('__full_motion')];currentauto=[c for c in old if c['id'].startswith('__full_motion')]
 history.append(dict(path=item['path'],named_stored_current_physics_equal=storednamed==oldnamed,stored_auto_types=dict(Counter(c['type'] for c in storedauto)),current_auto_types=dict(Counter(c['type'] for c in currentauto)),stored_auto_prefixes=sorted({c['id'].split('__')[0]+'__'+c['id'].split('__')[1]+'__' for c in storedauto}),current_auto_prefix='__full_motion_v2__',direct_reannotation_is_not_label_preserving=True))
 rows.append(dict(path=item['path'],T=n,windows=n-1,active_terms=terms,bad_reference_chunks=bad,compiled_openwam_schema_valid=True,terminal_anchors=[dict(id=c['id'],anchor=c['parameters'].get('anchor'),activation=c['activation']) for c in new if c['id'].startswith('terminal_lift_')]))
 if first is None:first=(item,new,raw)
assert totals['rejected_reference_windows']==0
# Real first exported reference, fixed measured row0, generated closure and reverse motion.
item,cs,raw=first;c=next(c for c in cs if c['id'].startswith('closure_location_left_'));event=c['activation']['event']['frame'];start=event-1;ref=raw[start:start+33].clone();initial=ref[:1].clone().requires_grad_();generated=ref[1:].clone();generated[0,0]+=.03;generated[:,9]=1.;generated.requires_grad_();pred=torch.cat((initial.detach(),generated));valid=torch.ones_like(pred,dtype=torch.bool);v=residual(pred,ref,valid,c,start_frame=start,rotation_convention='physical');loss=penalty(v,c['penalty']);loss.backward();assert v is not None and float(v.max())>.0089 and generated.grad[0,0]>0 and initial.grad is None
closure=dict(reference_frame=event,predicted_gripper_all_open=True,fixed_event_remains_active=True,violation_m=float(v.max()),scale_m=c['penalty']['scale'],gradient_finite=bool(torch.isfinite(generated.grad).all()),measured_row0_detached=initial.grad is None)
motion=None
for c in cs:
 if c['type']!='window_direction_cone':continue
 lo=c['activation']['start']-1;hi=min(c['activation']['end']-1,lo+8);arm=0 if c['subject']=='left_eef' else 10
 if hi-lo<2 or torch.linalg.vector_norm(raw[hi,arm:arm+3]-raw[lo,arm:arm+3])<.005:continue
 ref=raw[lo:min(len(raw),lo+33)].clone();initial=ref[:1].clone().requires_grad_();generated=ref[1:].clone();axis=torch.tensor(c['parameters']['axis'],dtype=torch.float64);generated[hi-lo-1,arm:arm+3]=ref[0,arm:arm+3]-.01*axis;generated.requires_grad_();pred=torch.cat((initial.detach(),generated));v=residual(pred,ref,torch.ones_like(pred,dtype=torch.bool),c,start_frame=lo,rotation_convention='physical');penalty(v,c['penalty']).backward();assert v is not None and float(v.max())>.007 and generated.grad.abs().sum()>0 and initial.grad is None
 motion=dict(id=c['id'],negative_net_displacement_m=.01,allowed_residual_slack_m=c['parameters']['bounds'][1],violation_m=float(v.max()),scale_m=c['penalty']['scale'],finite_nonzero_gradient=bool(torch.isfinite(generated.grad).all() and generated.grad.abs().sum()>0),motion_not_disabled=True,measured_row0_detached=True);break
assert motion is not None
# Newly proposed event-local constraints keep fixed masks and finite gradients.
new_checks=[]
for prefix,change in [('dev_first_close_event_axis0_left_', 'rotate'),('dev_closure_worldX_left_','x'),('dev_reference_open_left_','grip')]:
 c=next(c for c in cs if c['id'].startswith(prefix))
 event=c['activation'].get('event',{}).get('frame',1);start=event-1;ref=raw[start:start+33].clone();initial=ref[:1].clone().requires_grad_();generated=ref[1:].clone()
 if change=='rotate':
  angle=np.deg2rad(45);rot=torch.tensor([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],dtype=torch.float64)
  rr=torch.stack((ref[1,3:6],ref[1,6:9],torch.linalg.cross(ref[1,3:6],ref[1,6:9])),dim=-1);rr=rr@rot;generated[0,3:6]=rr[:,0];generated[0,6:9]=rr[:,1]
 elif change=='x':generated[0,0]-=.02
 else:generated[:,9]=.55
 generated.requires_grad_();pred=torch.cat((initial.detach(),generated));v=residual(pred,ref,torch.ones_like(pred,dtype=torch.bool),c,start_frame=start,rotation_convention='physical');assert v is not None and float(v.max())>0
 penalty(v,c['penalty']).backward();assert torch.isfinite(generated.grad).all()and generated.grad.abs().sum()>0 and initial.grad is None
 new_checks.append(dict(id=c['id'],perturbation=change,violation=float(v.max()),finite_nonzero_gradient=True,measured_row0_detached=True,activation=c['activation']))
result=dict(version=1,candidate=chosen,source_template_sha256=template_hash(template),training_reference_count=50,compiled_schema_version=SCHEMA_VERSION,stored_schema_versions=sorted({r['bundle']['schema_version'] for r in train['rows']}),reference_chunk_totals=dict(totals),rows=rows,closure_gradient=closure,motion_gradient=motion,new_feature_gradients=new_checks,rotation_mode='Physical WXYZ converted to two matrix columns; tests compiled feature/window semantics, not reader or normalization replacement',observability='Reference-event geometry is local, stable plateau intervals use fixed reference ranges, terminal_lift remains fixed-reference because its original anchor lies outside32. Ordinary-motion windows truncate at chunk start using only reference choices.',training_data_not_modified=True,gpu_started=False)
(P/'selected_training_chunk_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(P/'historical_training_label_audit.json').write_text(json.dumps(dict(version=1,rows=history,all_named_stored_current_physics_equal=all(x['named_stored_current_physics_equal'] for x in history),warning='All50 old embedded auto labels use historical full-motion-v1. Proposed current-template compilation creates reviewed full-motion-v2. This is not an in-place bound-only edit preserving every stored label; any later dataset rewrite needs explicit version migration review. No data is rewritten here.'),indent=2)+'\n')
print(json.dumps(dict(totals),indent=2));print('gradient checks passed; historical named',all(x['named_stored_current_physics_equal'] for x in history))
