from common import *
import time
plan=read(P/'adaptive_selection_plan.json');assert sha(P/'adaptive_selection_plan.json')=='119d65286cf5ce87a8351fe21912e8f41ef4dc7c65106000873a0c254f0a1735';assert sha(P/'screen_report.json')==plan['base_screen_sha256']
refs,items,old,new,verified=load_corpus();training=read(P.parent/'dual_bottle_revision_v2/training_inputs.json');dev=read(P.parent/'dual_bottle_revision_v1/development_report.json');summaries=[];started=time.time()

def invariant(oldcs,newcs):
 assert len(oldcs)==len(newcs);changes=[]
 for c,n in zip(oldcs,newcs,strict=True):
  a,b=physics(c),physics(n)
  if a['id'].startswith('__full_motion')and a['type']=='window_direction_cone':
   x=a['parameters'].pop('bounds');y=b['parameters'].pop('bounds');assert x==[0.,.0025]and y==[0.,.003]and a==b,c['id'];changes.append(dict(id=c['id'],before=x,after=y))
  else:assert a==b,c['id']
 return dict(all_constraint_ids_masks_frames_scales_weights_preserved=True,only_direction_slack_changed=True,changes=changes)
for setting in plan['candidates']:
 parent=P/'candidates'/setting['parent_candidate'];base=read(parent/'template.json');assert template_hash(base)==setting['parent_template_sha256']and sha(parent/'report.json.gz')==setting['parent_report_sha256'];parentreport=readgz(parent/'report.json.gz')
 patch=dict(version=1,base_template_sha256=template_hash(base),reason=plan['reason'],evidence_ids=[r['trial_id']for r in dev['rows']],operations=[dict(op='set_coverage_policy',policy=dict(direction_slack_m=.003))]);proposal=propose_revision(base,patch,dev);template=proposal['template'];dest=P/'adaptive_candidates'/setting['candidate_id'];save(dest/'template.json',template);save(dest/'revision_proposal.json',proposal)
 train=[];compiled={}
 for item in training['rows']:
  h=handle(item);before,_=instantiate(h,base);after,audit=instantiate(h,template);proof=invariant(before,after);assert audit['coverage']['complete']and audit['coverage']['chunk_safe'];train.append(dict(path=item['path'],constraints_sha256=template_hash(after),coverage_sha256=template_hash(audit['coverage']),proof=proof))
 for seed,ref in refs.items():
  b,audit=reference_bundle(template,ref['trajectory'],int(seed),ref['provenance']['scene_sha256']);proof=invariant(parentreport['compiled'][seed]['bundle']['constraints'],b['constraints']);assert audit['coverage']['complete']and audit['coverage']['chunk_safe'];compiled[seed]=dict(bundle=b,audit=audit,proof=proof,local_observability=parentreport['compiled'][seed]['local_observability'])
 rows=[]
 for item in items:
  seed=str(item['seed']);row={k:copy.deepcopy(v)for k,v in item.items()if k!='trajectory'};row['scores']={s:trajectory_violations(compiled[seed]['bundle'],refs[seed]['trajectory'],item['trajectory'][s])for s in ['commanded','actual']};rows.append(row)
 stats=summarize(rows);report=dict(candidate_id=setting['candidate_id'],settings=setting,template_sha256=template_hash(template),training_rows=train,compiled=compiled,rows=rows,stats=stats,errors=[],all50_and6_compiled=True,native_started=0,data_or_template_activated=False);savegz(dest/'report.json.gz',report)
 summaries.append(dict(candidate_id=setting['candidate_id'],template_sha256=template_hash(template),report_path=str((dest/'report.json.gz').relative_to(P)),report_sha256=sha(dest/'report.json.gz'),stats=stats,all50_and6_compiled=True))
 s=stats['all120'];print(setting['candidate_id'],'successC/A',s['commanded_success_retained'],s['actual_success_retained'],'failureRejectC/A',s['commanded_failure_rejected'],s['actual_failure_rejected'],'strong',len(s['strict_strong_failure_ids']),flush=True)
save(P/'adaptive_screen_report.json',dict(version=1,status='adaptive_CPU_exposed_development_screen',adaptive_plan_sha256=sha(P/'adaptive_selection_plan.json'),prior_screen_sha256=sha(P/'screen_report.json'),script_sha256=sha(__file__),candidates=summaries,reference_compilations=5*56,stream_scores=5*120*2,duration_seconds=time.time()-started,native_started=0,training_started=False,publication_allowed=False))
