"""Clean collection bookkeeping and physical initial-state identity."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def initial_scene(native):
    actors=[]
    for actor in native.scene.get_all_actors():
        pose=actor.get_pose()
        actors.append({'name':actor.get_name(),'pose':list(map(float,pose.p))+list(map(float,pose.q))})
    actors.sort(key=lambda a:(a['name'],a['pose']))
    articulations=[]
    for art in native.scene.get_all_articulations():
        if art==native.robot.left_entity or art==native.robot.right_entity:continue
        pose=art.get_root_pose()
        articulations.append({'name':art.get_name(),'pose':list(map(float,pose.p))+list(map(float,pose.q)),
                              'qpos':np.asarray(art.get_qpos()).tolist()})
    articulations.sort(key=lambda a:(a['name'],a['pose']))
    return {'actors':actors,'task_articulations':articulations}


def scene_fingerprint(scene):
    """Exclude seed/time; quantize physical geometry to 1 mm / 1e-3 quaternion units."""
    rows=[]
    for kind,items in scene.items():
        for item in items:
            pose=np.asarray(item['pose'],dtype=float).copy()
            if pose.shape!=(7,) or not np.isfinite(pose).all():raise ValueError('Invalid initial pose')
            # q and -q represent the same physical rotation.
            q=pose[3:]
            nonzero=np.flatnonzero(np.abs(q)>1e-8)
            if len(nonzero) and q[nonzero[0]]<0:pose[3:]*=-1
            values=np.rint(pose/.001).astype(np.int64).tolist()
            joints=np.rint(np.asarray(item.get('qpos',[]))/.001).astype(np.int64).tolist()
            rows.append([kind,item['name'],values,joints])
    rows.sort(key=lambda row:json.dumps(row))
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def install_planner_cache():
    """Reuse immutable-config cuRobo models, not scene state or trajectories.

    Robot.set_planner still constructs scene-bound MPLib planners normally.
    Native CuroboPlanner.plan_path resets its solver seed on every request.
    """
    from envs.robot.planner import CuroboPlanner
    original=CuroboPlanner.__init__;cache={};stats={'misses':0,'hits':0}
    def cached(self,origin,names,all_joints,yml_path=None):
        key=json.dumps([str(Path(yml_path).resolve()),hashlib.sha256(Path(yml_path).read_bytes()).hexdigest(),
                        list(map(float,origin.p)),list(map(float,origin.q)),list(names),list(all_joints)])
        if key not in cache:
            original(self,origin,names,all_joints,yml_path=yml_path)
            cache[key]=dict(self.__dict__);stats['misses']+=1
        else:
            self.__dict__.update(cache[key]);stats['hits']+=1
    CuroboPlanner.__init__=cached
    return stats
