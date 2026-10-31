import copy,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;W=P.parents[2];R=W/'.worktrees/rax-dual-bottle-calibration';sys.path.insert(0,str(R))
from services.robotwin.acceptance import trajectory_violations
from services.robotwin.constraint_refinement import propose_revision
from services.robotwin.programmatic_annotation import instantiate,template_hash
from services.robotwin.constraint_schema import HDF5_KEY
from services.robotwin.constraint_coverage import chunk_candidate
from scripts.validate_robotwin_constraint_acceptance import reference_bundle
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def readgz(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def savegz(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(gzip.compress(json.dumps(d,allow_nan=False,separators=(',',':')).encode(),mtime=0))
def physics(c):return {k:copy.deepcopy(v) for k,v in c.items()if k!='evidence'}
class Handle(dict):pass
def handle(item):
 h=Handle({HDF5_KEY:np.asarray(json.dumps(item['bundle']))});h.attrs={'robotwin_expert_provenance':json.dumps(item['expert_provenance'])}
 for arm,v in item['arms'].items():h[f'endpose/{arm}_endpose']=np.asarray(v['pose']);h[f'endpose/{arm}_gripper']=np.asarray(v['gripper'])
 return h

def load_corpus():
 old=readgz(P.parent/'dual_bottle_revision_v1/verified_corpus.json.gz');new=readgz(P/'new_corpus.json.gz')
 verified=read(P.parent/'dual_bottle_finite_runtime_v1/result_review_1793435169/verified_results.json')
 assert verified['status']=='completed_campaign_verified'and verified['selected_completed_seeds']==[13000,13001,13002]
 vrows={(str(s['seed']),r['trial_id']):r for s in verified['scene_reports']for r in s['trials']}
 for path,h in new['source_hashes'].items():
  relative=path.removeprefix('runs/pick_dual_bottles/');assert verified['completed_scene_artifacts'][relative]==h,path
 refs={**old['refs'],**new['refs']};rows=[]
 for item in old['rows']:
  r=item['record'];rows.append(dict(uid='old/'+r['trial_id'],corpus='old',trial_id=r['trial_id'],seed=r['seed'],kind='baseline'if r['variant_family']=='baseline'else'candidate',family=r['variant_family'],native_success=item['outcome']['native_success'],eligible=item['verified']['eligible'],execution_complete=item['outcome']['execution_complete'],trajectory=item['trajectory'],reference_sha256=r['reference_sha256'],trajectory_sha256=r['trajectory_sha256']))
 for item in new['rows']:
  r=item['record'];seed=str(item['seed']);v=vrows[(seed,item['trial_id'])];assert r['classification']==v['result']
  assert v['record_sha256']==new['source_hashes']['runs/pick_dual_bottles/scenes/seed'+seed+'/trials/'+item['trial_id']+'/record.json']
  rows.append(dict(uid='new/'+seed+'/'+item['trial_id'],corpus='new',trial_id=item['trial_id'],seed=item['seed'],kind='baseline'if v['category']=='baseline'else'candidate',family=v['family'],native_success=v['result']['native_success'],eligible=v['result']['eligible'],execution_complete=v['result']['execution_complete'],trajectory=item['trajectory'],reference_sha256=r['reference_sha256'],trajectory_sha256=r['trajectory']['sha256']))
 assert len(rows)==120 and all(r['execution_complete']for r in rows)
 return refs,rows,old,new,verified

def summarize(rows):
 scopes={'all120':rows,'requested_old60_plus_new54':[r for r in rows if r['corpus']=='old'or r['kind']!='baseline'],'all_baselines':[r for r in rows if r['kind']=='baseline'],'all_candidates':[r for r in rows if r['kind']=='candidate']}
 for corpus in ['old','new']:
  rr=[r for r in rows if r['corpus']==corpus];scopes[corpus+'_all']=rr;scopes[corpus+'_candidates']=[r for r in rr if r['kind']=='candidate']
  scopes[corpus+'_strict_candidates']=[r for r in rr if r['kind']=='candidate'and r['eligible']];scopes[corpus+'_tracking_invalid_candidates']=[r for r in rr if r['kind']=='candidate'and not r['eligible']]
 result={}
 for scope,rr in scopes.items():
  pos=[r for r in rr if r['native_success']];neg=[r for r in rr if not r['native_success']];s=dict(records=len(rr),native_success=len(pos),native_failure=len(neg))
  for stream in ['commanded','actual']:
   s[stream+'_success_retained']=sum(r['scores'][stream]['passed']for r in pos);s[stream+'_failure_rejected']=sum(not r['scores'][stream]['passed']for r in neg)
   s[stream+'_success_rejected_ids']=[r['uid']for r in pos if not r['scores'][stream]['passed']];s[stream+'_failure_accepted_ids']=[r['uid']for r in neg if r['scores'][stream]['passed']]
  s['strict_strong_failure_ids']=[r['uid']for r in neg if r['eligible']and all(r['scores'][stream]['passed']for stream in ['commanded','actual'])]
  s['strict_command_only_failure_ids']=[r['uid']for r in neg if r['eligible']and r['scores']['commanded']['passed']and not r['scores']['actual']['passed']]
  result[scope]=s
 return result
