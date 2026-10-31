from common import *
import time
rows=[]
for parent in ['open0.60__axis030__closure0.021__hold0.021','open0.60__tcp0.04__closure0.021__hold0.021']:
 for width in [.0125,.015,.0175]:rows.append(dict(parent_candidate=parent,parent_template_sha256=template_hash(read(P/'fallback_candidates'/parent/'template.json')),parent_report_sha256=sha(P/'fallback_candidates'/parent/'report.json.gz'),candidate_id=parent+'__closureX'+str(width),half_width_m=width))
p=P/'axial_selection_plan.json';assert not p.exists();save(p,dict(version=1,status='third_adaptive_development_addendum_not_holdout',created_unix=time.time(),fallback_screen_sha256=sha(P/'fallback_screen_report.json'),feature_audit_sha256=sha(P/'feature_alias_diagnostic.json'),candidates=rows,maximum_candidates=6,total_candidate_limit_this_round=72,
reason='Enlarging inherited short-segment fallback to preserve the real20mm-Z successful bump newly admits3 native failures whose commanded left EEF X is−20mm at closure. All81 known successful command X residuals are within±10mm and actual within−10.62..10.96mm; coarse12.5/15/17.5mm X bands test lateral alignment freedom without tightening Z or full paths. This choice was informed by dev outcomes and is not object-geometry calibration.',
physical_scope='A single world-X coordinate interval centered on the fixed reference closure point, conjoined with existing radial/angle/TCP constraints. Task world frame and mechanical TCP provenance are explicit. No newwhole-pathXYZ box and no objects/native labels used by loss.',
limits='Known−X10mm outcomes can differ by scene; those smaller offsets are deliberately allowed by this coarse grid, not micro-tuned away. No claim that this axis alone expresses successful grasp.',native_started=0,publication_allowed=False));print(sha(p))
