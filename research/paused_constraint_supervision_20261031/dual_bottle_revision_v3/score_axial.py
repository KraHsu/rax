from common import *
import time
plan=read(P/'axial_selection_plan.json');assert sha(P/'axial_selection_plan.json')=='24a6199192ca471ac955105ce62ddcbd5939636d3558d79f830ef94c18057d54';assert sha(P/'fallback_screen_report.json')==plan['fallback_screen_sha256']
refs,items,old,new,verified=load_corpus();training=read(P.parent/'dual_bottle_revision_v2/training_inputs.json');dev=read(P.parent/'dual_bottle_revision_v1/development_report.json');summaries=[];started=time.time()

def invariant(oldcs,newcs):
 by={c['id']:c for c in newcs}
 for c in oldcs:assert physics(c)==physics(by[c['id']]),c['id']
 added=[c['id']for c in newcs if c['id']not in {x['id']for x in oldcs}]
 assert len(added)==2 and all(x.startswith('dev_closure_worldX_')for x in added)
 return dict(all_parent_constraints_physics_exact=True,all_motion_guards_exact=True,added_constraint_ids=added)
for setting in plan['candidates']:
 parent=P/'fallback_candidates'/setting['parent_candidate'];base=read(parent/'template.json');assert template_hash(base)==setting['parent_template_sha256']and sha(parent/'report.json.gz')==setting['parent_report_sha256'];parentreport=readgz(parent/'report.json.gz')
 patch=dict(version=1,base_template_sha256=template_hash(base),reason=plan['reason'],evidence_ids=[r['trial_id']for r in dev['rows']],operations=[dict(op='add_rule',rule=dict(id='dev_closure_worldX',type='axis_position_interval',subjects='active_arms',selector=dict(kind='events',direction='decreasing',occurrence='first'),origin='selection_start',orientation='world',parameters=dict(axis=[1,0,0]),allowed_margin=dict(lower=setting['half_width_m'],upper=setting['half_width_m'],unit='m'),required=True,rationale=plan['physical_scope']+' Exposed-development hypothesis, not object-clearance proof.'))]);proposal=propose_revision(base,patch,dev);template=proposal['template'];dest=P/'axial_candidates'/setting['candidate_id'];save(dest/'template.json',template);save(dest/'revision_proposal.json',proposal)
 train=[];compiled={}
 for item in training['rows']:
  h=handle(item);before,_=instantiate(h,base);after,audit=instantiate(h,template);proof=invariant(before,after);assert audit['coverage']['complete']and audit['coverage']['chunk_safe'];train.append(dict(path=item['path'],constraints_sha256=template_hash(after),coverage_sha256=template_hash(audit['coverage']),proof=proof))
 for seed,ref in refs.items():
  b,audit=reference_bundle(template,ref['trajectory'],int(seed),ref['provenance']['scene_sha256']);proof=invariant(parentreport['compiled'][seed]['bundle']['constraints'],b['constraints']);assert audit['coverage']['complete']and audit['coverage']['chunk_safe'];compiled[seed]=dict(bundle=b,audit=audit,proof=proof,local_observability=[dict(id=c['id'],activation=c['activation'],valid_chunk_starts=[start for start in range(len(ref['trajectory']['left']['pose'])-1)if chunk_candidate(c,start,32)])for c in b['constraints']if not c['id'].startswith('__full_motion')])
 rows=[]
 for item in items:
  seed=str(item['seed']);row={k:copy.deepcopy(v)for k,v in item.items()if k!='trajectory'};row['scores']={s:trajectory_violations(compiled[seed]['bundle'],refs[seed]['trajectory'],item['trajectory'][s])for s in ['commanded','actual']};rows.append(row)
 stats=summarize(rows);report=dict(candidate_id=setting['candidate_id'],settings=setting,template_sha256=template_hash(template),training_rows=train,compiled=compiled,rows=rows,stats=stats,errors=[],all50_and6_compiled=True,native_started=0,data_or_template_activated=False);savegz(dest/'report.json.gz',report)
 summaries.append(dict(candidate_id=setting['candidate_id'],template_sha256=template_hash(template),report_path=str((dest/'report.json.gz').relative_to(P)),report_sha256=sha(dest/'report.json.gz'),stats=stats,all50_and6_compiled=True))
 s=stats['all120'];print(setting['candidate_id'],'successC/A',s['commanded_success_retained'],s['actual_success_retained'],'failureRejectC/A',s['commanded_failure_rejected'],s['actual_failure_rejected'],'strong',len(s['strict_strong_failure_ids']),flush=True)
save(P/'axial_screen_report.json',dict(version=1,status='adaptive_CPU_exposed_development_screen',adaptive_plan_sha256=sha(P/'axial_selection_plan.json'),prior_screen_sha256=sha(P/'fallback_screen_report.json'),script_sha256=sha(__file__),candidates=summaries,reference_compilations=6*56,stream_scores=6*120*2,duration_seconds=time.time()-started,native_started=0,training_started=False,publication_allowed=False))
