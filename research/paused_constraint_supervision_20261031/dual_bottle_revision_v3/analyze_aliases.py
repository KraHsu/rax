from common import *
from services.robotwin.annotation import rotation
refs,items,old,new,v=load_corpus();rows=[]
for item in items:
 ref=refs[str(item['seed'])]['trajectory']
 for stream in ['commanded','actual']:
  tr=item['trajectory'][stream];features={};eq={};deltas={}
  for arm in ['left','right']:
   rp=np.asarray(ref[arm]['pose']);ap=np.asarray(tr[arm]['pose']);rg=np.asarray(ref[arm]['gripper']);ag=np.asarray(tr[arm]['gripper']);frame=int(np.flatnonzero((rg[:-1]>.5)&(rg[1:]<=.5))[0]+1);rr=rotation(rp[frame,3:]);ar=rotation(ap[frame,3:]);d=ap[frame,:3]-rp[frame,:3];tcp=d+(ar-rr)@np.array([.12,0,0]);local=rr.T@d;localtcp=rr.T@tcp
   # All axes/values are reference/action quantities; no object pose/label enters features.
   for space,values in [('world_delta_m',d),('reference_eef_delta_m',local),('tcp_world_delta_m',tcp),('tcp_reference_eef_delta_m',localtcp)]:
    for axis,value in zip('xyz',values):features[arm+'/'+space+'/'+axis]=float(value)
   features[arm+'/eef_distance_m']=float(np.linalg.norm(d));features[arm+'/tcp_distance_m']=float(np.linalg.norm(tcp))
   for axis in range(3):features[arm+'/local_axis_angle_rad/'+str(axis)]=float(np.arctan2(np.linalg.norm(np.cross(rr[:,axis],ar[:,axis])),np.dot(rr[:,axis],ar[:,axis])))
   inds=np.flatnonzero((np.abs(rg-1.)<=1e-6)&(np.arange(len(rg))<frame)&(np.arange(len(rg))>0));features[arm+'/open_plateau_min_fraction']=float(ag[inds].min());features[arm+'/fixed_reference_close_fraction']=float(ag[frame]);deltas[arm]=d
   for key in ['world_delta_m/x','world_delta_m/y','world_delta_m/z','eef_distance_m','tcp_distance_m','local_axis_angle_rad/1','open_plateau_min_fraction','fixed_reference_close_fraction']:eq[arm+'/'+key]=features[arm+'/'+key]
  for axis,z in zip('xyz',deltas['left']-deltas['right']):features['relative_arms_delta_m/'+axis]=float(z)
  rows.append(dict(uid=item['uid'],seed=item['seed'],kind=item['kind'],native_success=item['native_success'],eligible=item['eligible'],stream=stream,features=features,alias_feature_subset=eq))
positive={s:[r for r in rows if r['stream']==s and r['native_success']]for s in ['commanded','actual']};envelopes={s:{key:dict(lower=min(r['features'][key]for r in rr),upper=max(r['features'][key]for r in rr))for key in rr[0]['features']}for s,rr in positive.items()}
checks=[]
for r in rows:
 if r['native_success']:continue
 keys=r['features'];outside=[]
 for key,value in keys.items():
  env=envelopes[r['stream']][key];distance=max(env['lower']-value,0)+max(value-env['upper'],0)
  if distance>1e-12:outside.append(dict(feature=key,value=value,envelope=env,excess=distance))
 nearest=min(positive[r['stream']],key=lambda pos:max(abs(r['alias_feature_subset'][key]-pos['alias_feature_subset'][key])for key in r['alias_feature_subset']))
 diff={k:r['alias_feature_subset'][k]-nearest['alias_feature_subset'][k]for k in r['alias_feature_subset']}
 checks.append(dict(uid=r['uid'],stream=r['stream'],all_tested_scalars_inside_positive_envelope=not outside,outside=outside,nearest_positive_in_documented_subset=nearest['uid'],subset_max_abs_difference=max(abs(x)for x in diff.values()),subset_differences=diff))
save(P/'feature_alias_diagnostic.json',dict(version=1,rows=rows,positive_envelopes=envelopes,negative_checks=checks,interpretation='These are fixed-reference closure/action scalar feature families, not all possible action information. A failure inside every all-success scalar envelope cannot be rejected by any conjunction of one interval per listed scalar that retains all those positives. Near equality in the documented subset is not a claim of identical full action/scene/reference or impossibility for all nonlinear coupled representations.',native_object_state_used_for_features=False,native_started=0,script_sha256=sha(__file__)))
print('rows',len(rows),'positive streams',len(positive['commanded']),len(positive['actual']))
for r in checks:
 if r['stream']=='commanded'and (r['uid'].endswith('position_0_-1')or r['uid'].endswith('candidate_07')or r['uid'].endswith('candidate_08')):print(r['uid'],'inside',r['all_tested_scalars_inside_positive_envelope'],'nearest',r['nearest_positive_in_documented_subset'],'delta',r['subset_max_abs_difference'],'outside',[(x['feature'],x['excess'])for x in r['outside']])
