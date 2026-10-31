from common import *
import importlib.util,math
from services.robotwin.annotation import rotation
source=P.parent/'dual_bottle_native_dev_v1/candidate_design.py';spec=importlib.util.spec_from_file_location('frozen_design',source);gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
refs,items,old,new,v=load_corpus();names={'A20':P/'candidates/open0.51__none__closure0.020/report.json.gz','axisX15':P/'axial_candidates/open0.60__axis030__closure0.021__hold0.021__closureX0.015/report.json.gz','TCP_X15':P/'axial_candidates/open0.60__tcp0.04__closure0.021__hold0.021__closureX0.015/report.json.gz'}
reports={n:readgz(p)for n,p in names.items()};plan=read(P/'selection_plan.json')['synthetic_feasibility'];slots=[]
for f in plan['open_fractions']:slots.append(dict(family='pregrasp_open_floor',parameters=dict(fraction=f,ramp_frames=8)))
for arm in ['left','right']:
 for a in plan['localY_degrees']:slots.append(dict(family='local_y_roll',parameters=dict(degrees={arm:a})))
for dx in plan['worldX_offsets_m']:slots.append(dict(family='smooth_offset',parameters=dict(offset_world_m=[dx,0,0],arms=['left','right'])))
assert len(slots)==24;rows=[];checks=[]
for seed,ref in refs.items():
 for slot in slots:
  candidate,events=gen.build_raw(ref['trajectory'],slot)
  for arm in ['left','right']:
   assert candidate[arm]['pose'][0]==ref['trajectory'][arm]['pose'][0]and candidate[arm]['gripper'][0]==ref['trajectory'][arm]['gripper'][0]
  if slot['family']=='local_y_roll':
   arm=next(iter(slot['parameters']['degrees']));deg=slot['parameters']['degrees'][arm];frame=events[arm]['first_close'];rp=np.asarray(ref['trajectory'][arm]['pose'][frame]);cp=np.asarray(candidate[arm]['pose'][frame]);rr=rotation(rp[3:]);cr=rotation(cp[3:]);y_error=float(np.linalg.norm(cr[:,1]-rr[:,1]));tcp_delta=(cp[:3]+cr@np.array([.12,0,0]))-(rp[:3]+rr@np.array([.12,0,0]));expected=.24*abs(math.sin(math.radians(deg)/2));assert y_error<1e-14 and abs(np.linalg.norm(tcp_delta)-expected)<1e-14
   angle=math.atan2(np.linalg.norm(np.cross(rr[:,0],cr[:,0])),np.dot(rr[:,0],cr[:,0]));assert abs(angle-abs(math.radians(deg)))<1e-14
   checks.append(dict(seed=int(seed),arm=arm,degrees=deg,reference_close_frame=frame,existing_Y_axis_error=y_error,new_X_axis_angle_rad=angle,tcp_delta_norm_m=float(np.linalg.norm(tcp_delta)),mechanical_expected_m=expected))
  scores={name:trajectory_violations(r['compiled'][seed]['bundle'],ref['trajectory'],candidate)for name,r in reports.items()}
  rows.append(dict(seed=int(seed),slot=slot,candidate_sha256=template_hash(candidate),scores=scores,native_success=None,source='crafted_CPU_feasibility_not_executed'))
save(P/'synthetic_feasibility.json',dict(version=1,generator_source=str(source.relative_to(W)),generator_sha256=sha(source),script_sha256=sha(__file__),original_selection_plan_sha256=sha(P/'selection_plan.json'),slots=slots,candidates=len(rows),stream_scores=len(rows)*len(reports),rows=rows,local_rotation_TCP_geometry_checks=checks,initial_measured_row_unchanged=True,native_executions=0,meaning='Multiple amplitudes and signs test CPU geometry/constraint feasibility, not empirical failure or success; never included in native-positive/negative counts.'))
print(len(rows),'crafted CPU candidates;',len(checks),'mechanical/quat checks;',len(rows)*len(reports),'scores')
