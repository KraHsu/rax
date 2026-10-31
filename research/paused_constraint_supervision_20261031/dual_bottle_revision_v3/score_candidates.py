from common import *
import math,time,subprocess
plan=read(P/'selection_plan.json');assert sha(P/'selection_plan.json')=='0adcdfcb00af6fa2865e7fb0ac4b61c5cb7994683880d1768b64ea69b92d57ca'
for path,h in plan['input_sha256'].items():assert sha(P.parent/path)==h,path
for path,h in plan['source_sha256'].items():assert sha(R/path)==h,path
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==plan['source_commit']
refs,items,old,new,verified=load_corpus();training=read(P.parent/'dual_bottle_revision_v2/training_inputs.json');dev=read(P.parent/'dual_bottle_revision_v1/development_report.json')
base=read(P.parent/'dual_bottle_revision_v2/candidates/A__cal_c0.02_h0.02_s0.0025/template.json');tcp_rule=read(P.parent/'dual_bottle_revision_v1/B_open_TCP/template.json')['rules'][-1]
original=old['manifest']['template'];basecompiled={};trainbase=[]
for seed,ref in refs.items():basecompiled[seed]=reference_bundle(base,ref['trajectory'],int(seed),ref['provenance']['scene_sha256'])[0]
numeric_deltas=[]
def numeric_compare(a,b,path):
 if type(a)is dict:
  assert set(a)==set(b)
  for k in a:numeric_compare(a[k],b[k],path+'/'+k)
 elif type(a)is list:
  assert len(a)==len(b)
  for k,(x,y)in enumerate(zip(a,b)):numeric_compare(x,y,path+'/'+str(k))
 elif a!=b:
  assert isinstance(a,(int,float))and isinstance(b,(int,float))and abs(a-b)<1e-12,(path,a,b)
  numeric_deltas.append(dict(path=path,local_numpy_value=a,frozen_native_value=b,difference=a-b))
for seed,ref in new['refs'].items():numeric_compare([physics(c)for c in basecompiled[seed]['constraints']],[physics(c)for c in ref['bundle']['constraints']],seed)
save(P/'native_recompile_numeric_delta.json',dict(local_numpy=np.__version__,native_numpy='1.26.4',differences=numeric_deltas,reason='Exact frozen reference is retained. Different NumPy normalization arithmetic may differ at binary64 rounding; all command/actual baseline classifications and violations are checked separately, with no change to the actual pass threshold.'))
for item in items:
 if item['corpus']=='new':
  seed=str(item['seed']);v=next(r for s in verified['scene_reports']if str(s['seed'])==seed for r in s['trials']if r['trial_id']==item['trial_id'])
  for stream,key in [('commanded','command_score'),('actual','actual_score')]:
   score=trajectory_violations(basecompiled[seed],refs[seed]['trajectory'],item['trajectory'][stream]);recorded=v['result'][key]
   assert score['passed']==recorded['passed']and abs(score['max_violation']-recorded['max_violation'])<1e-12

for item in training['rows']:
 h=handle(item);cs,a=instantiate(h,base);trainbase.append((h,cs,a))

def build(setting):
 ops=[];by={r['id']:r for r in base['rules']}
 for key,value in [('closure_location',setting['closure_radius_m']),('dev_reference_open',1-setting['open_floor'])]:
  rule=copy.deepcopy(by[key]);rule['allowed_margin']['lower']=value;rule['allowed_margin']['upper']=value;rule['rationale']+=' V3 coarse exposed-development candidate; fixed reference activation and penalty scale preserved; no object-state labels.'
  ops.append(dict(op='replace_rule',rule=rule))
 geometry=setting['geometry']
 if geometry['kind']=='axis0':
  rule=copy.deepcopy(by['dev_first_close_event_axis1']);rule.update(id='dev_first_close_event_axis0',parameters=dict(local_axis=[1,0,0],reference_axis=[1,0,0]),allowed_margin=dict(lower=math.radians(geometry['upper_deg']),upper=math.radians(geometry['upper_deg']),unit='rad'),rationale='Second nonparallel EEF localX axis at fixed first reference closure, with coarse angular freedom. Complements localY while avoiding full-trajectory orientation fitting; does not prove contact.');ops.append(dict(op='add_rule',rule=rule))
 elif geometry['kind']=='tcp':
  rule=copy.deepcopy(tcp_rule);rule['allowed_margin'].update(lower=geometry['radius_m'],upper=geometry['radius_m']);rule['rationale']+=' V3 conjunction retains the existing localY relation and all ordinary motion guards.';ops.append(dict(op='add_rule',rule=rule))
 patch=dict(version=1,base_template_sha256=template_hash(base),reason='V3 coarse offline development screen: all old60 and new54 crafted plus6 baseline controls. No activation or native validation.',evidence_ids=[r['trial_id']for r in dev['rows']],operations=ops)
 proposal=propose_revision(base,patch,dev);proposal['additional_development_evidence']=dict(source_report_sha256=sha(P.parent/'dual_bottle_finite_runtime_v1/result_review_1793435169/verified_results.json'),new_corpus_sha256=sha(P/'new_corpus.json.gz'),role='separate finite-native observational dev record format, not a substituted refinement acceptance ledger',publishable=False)
 return proposal['template'],proposal

def invariant(oldcs,newcs):
 by={c['id']:c for c in newcs};changes=[]
 for c in oldcs:
  before=physics(c);after=physics(by[c['id']]);assert before['activation']==after['activation']and before['penalty']==after['penalty']and before['reference_frame']==after['reference_frame']
  if c['id'].startswith('closure_location_')or c['id'].startswith('dev_reference_open_'):
   ob=before['parameters'].pop('bounds');nb=after['parameters'].pop('bounds');assert before==after
   if ob!=nb:changes.append(dict(id=c['id'],before=ob,after=nb))
  else:assert before==after,c['id']
 assert all(physics(c)==physics(by[c['id']])for c in oldcs if c['id'].startswith('__full_motion'))
 return dict(all_prior_ids_and_masks_preserved=True,all_prior_scales_and_weights_preserved=True,full_motion_exact=True,bounds_changes=changes,added_ids=[c['id']for c in newcs if c['id']not in {x['id']for x in oldcs}])

settings=[dict(candidate_id='original_parent_control')]+plan['settings'];summaries=[];started=time.time()
for setting in settings:
 name=setting['candidate_id'];iscontrol=name=='original_parent_control';template,proposal=(copy.deepcopy(original),dict(state='original_control'))if iscontrol else build(setting);dest=P/'candidates'/name;save(dest/'template.json',template);save(dest/'revision_proposal.json',proposal)
 training_rows=[];compiled={};errors=[]
 for item,(h,oldcs,oa)in zip(training['rows'],trainbase,strict=True):
  try:
   cs,audit=instantiate(h,template);proof=None if iscontrol else invariant(oldcs,cs);assert audit['coverage']['complete']and audit['coverage']['chunk_safe']
   training_rows.append(dict(path=item['path'],seed=item['expert_provenance']['seed'],constraints_sha256=template_hash(cs),coverage_sha256=template_hash(audit['coverage']),proof=proof))
  except Exception as e:errors.append(dict(scope='training',path=item['path'],error=type(e).__name__+':'+str(e)))
 for seed,ref in refs.items():
  try:
   bundle,audit=reference_bundle(template,ref['trajectory'],int(seed),ref['provenance']['scene_sha256']);proof=None if iscontrol else invariant(basecompiled[seed]['constraints'],bundle['constraints']);assert audit['coverage']['complete']and audit['coverage']['chunk_safe']
   compiled[seed]=dict(bundle=bundle,audit=audit,proof=proof,local_observability=[dict(id=c['id'],activation=c['activation'],valid_chunk_starts=[s for s in range(len(ref['trajectory']['left']['pose'])-1)if chunk_candidate(c,s,32)])for c in bundle['constraints']if not c['id'].startswith('__full_motion')])
  except Exception as e:errors.append(dict(scope='reference',seed=seed,error=type(e).__name__+':'+str(e)))
 rows=[]
 for item in items:
  seed=str(item['seed']);row={k:copy.deepcopy(v)for k,v in item.items()if k!='trajectory'};row['scores']={stream:trajectory_violations(compiled[seed]['bundle'],refs[seed]['trajectory'],item['trajectory'][stream])for stream in ['commanded','actual']};rows.append(row)
 stats=summarize(rows);report=dict(candidate_id=name,settings=setting,template_sha256=template_hash(template),training_rows=training_rows,compiled=compiled,rows=rows,stats=stats,errors=errors,all50_and6_compiled=len(training_rows)==50 and len(compiled)==6 and not errors,native_started=0,data_or_template_activated=False);savegz(dest/'report.json.gz',report)
 summaries.append(dict(candidate_id=name,settings=setting,template_sha256=template_hash(template),report_path=str((dest/'report.json.gz').relative_to(P)),report_sha256=sha(dest/'report.json.gz'),stats=stats,errors=errors,all50_and6_compiled=report['all50_and6_compiled']))
 s=stats['all120'];print(name,'successC/A',s['commanded_success_retained'],s['actual_success_retained'],'rejectC/A',s['commanded_failure_rejected'],s['actual_failure_rejected'],'strong',len(s['strict_strong_failure_ids']),'errors',len(errors),flush=True)
for path,h in plan['source_sha256'].items():assert sha(R/path)==h
save(P/'screen_report.json',dict(version=1,status='CPU_exposed_development_screen',selection_plan_sha256=sha(P/'selection_plan.json'),source_commit=plan['source_commit'],source_sha256=plan['source_sha256'],script_sha256={n:sha(P/n)for n in ['common.py','score_candidates.py']},input_sha256=plan['input_sha256'],candidates=summaries,reference_compilations=49*56,stream_scores=49*120*2,duration_seconds=time.time()-started,native_started=0,training_started=False,publication_allowed=False))
