from common import *
import time
screen=read(P/'screen_report.json');names=['open0.51__none__closure0.021','open0.60__none__closure0.021','open0.60__axis030__closure0.021','open0.60__tcp0.03__closure0.021','open0.60__tcp0.04__closure0.021']
rows=[]
for name in names:
 c=next(x for x in screen['candidates']if x['candidate_id']==name);report=readgz(P/c['report_path'])
 assert report['stats']['all120']['actual_success_rejected_ids']==['new/13002/candidate_06']
 rr=next(r for r in report['rows']if r['uid']=='new/13002/candidate_06');violations=[v for v in rr['scores']['actual']['constraints']if v['max_violation']>1e-6]
 assert len(violations)==1 and violations[0]['type']=='window_direction_cone'
 rows.append(dict(parent_candidate=name,parent_template_sha256=c['template_sha256'],parent_report_sha256=c['report_sha256'],candidate_id=name+'__slack0.003',direction_slack_m=.003,observed_inherited_success_rejection=violations))
p=P/'adaptive_selection_plan.json';assert not p.exists();save(p,dict(version=1,status='adaptive_development_addendum_not_holdout',created_unix=time.time(),base_selection_plan_sha256=sha(P/'selection_plan.json'),base_screen_sha256=sha(P/'screen_report.json'),candidates=rows,maximum_candidates=5,
reason='The frozen49 grid preserves80/81 actual native successes: one tracking-invalid but native-success trace is rejected only by a7.12um old direction-cone excess. Keep the original49 tradeoffs intact. Test a coarse .5mm slack increase(.0025→.003), not a threshold fitted7um above this trace. Retain guard IDs, directions, masks, weights/scales. No change to hold/position fallback constraints.',
selection='No candidate is declared acceptable unless it preserves all81 known successes, including the 15 tracking-invalid positive records. Report commanded and actual separately and retain all39 failures.',native_started=0,training_started=False,publication_allowed=False));print(sha(p))
