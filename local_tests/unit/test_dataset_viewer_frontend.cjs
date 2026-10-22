// Run with node --test. Pure geometry and mock-DOM behavior, not browser/layout QA.
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('services/dataset_viewer/static/app.js','utf8');
const html=fs.readFileSync('services/dataset_viewer/static/index.html','utf8');
function harness(){
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(([,id])=>[id,{value:'',checked:false,innerHTML:'',textContent:'',hidden:false,
    setAttribute(){},classList:{add(){},remove(){},toggle(){}},addEventListener(){}}]));
  for(const[id,value]of Object.entries({'signal-kind':'position','arm-select':'left','projection':'iso','spatial-zoom':'local'}))elements.get(id).value=value;
  for(const id of ['show-constraints','follow-constraints'])elements.get(id).checked=true;
  const ctx=vm.createContext({document:{getElementById:id=>elements.get(id)||null},console});
  vm.runInContext(source.slice(0,source.indexOf("$('search').addEventListener"))+`
    frameImages=async()=>{};
    globalThis.api={state,rotate3,frameRuns,positionBands,spatialGeometry,featureDisplay,activeFeature,isViolation,lineChart,renderCharts,selectConstraint,setFrame};`,ctx);
  return {api:ctx.api,elements};
}
function near(a,b,eps=1e-9){assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);}
function near3(a,b){a.forEach((v,i)=>near(v,b[i]));}
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),norm=a=>Math.hypot(...a);
function data(){const pose=Array.from({length:6},(_,f)=>[1+f*.01,2+f*.02,3+f*.03,1,0,0,0]);return {frames:6,annotation_method:'programmatic',constraints:[],arms:{left:{pose,gripper:[1,1,0,0,0,0]},right:{pose:structuredClone(pose),gripper:[1,1,1,1,1,1]}}};}
function constraint(type,parameters={}){return {id:type,type,subject:'left_eef',parameters:{bounds:[0,.1],...parameters},
  reference_frame:{origin_m:[.5,1.5,2.5],quaternion_wxyz:[1,0,0,0]},activation:{mode:'interval',start:2,end:5},
  active_frames:[2,3,4],values:[0,0,0,0,0,0],unit:type.includes('angle')||type.includes('angular')?'rad':type==='gripper_interval'?'fraction':'m',
  reference_violation:0,penalty:{kind:'squared_hinge',scale:.01},evidence:{status:'accepted',rationale:'test'}};}

test('rotated/signed world axes and original displacement anchor give physical bands',()=>{
  const{api}=harness(),d=data(),c=constraint('axis_position_interval',{axis:[1,0,0],bounds:[.1,.3]});
  c.reference_frame.quaternion_wxyz=[Math.SQRT1_2,0,0,Math.SQRT1_2];
  near3(api.rotate3(c.reference_frame.quaternion_wxyz,[1,0,0]),[0,1,0]);
  near3(api.rotate3([2,0,0,0],[1,2,3]),[1,2,3]);
  let bands=api.positionBands(c,d).bands;near(bands[0].bounds[0],1.6);near(bands[0].bounds[1],1.8);
  c.parameters.axis=[-1,0,0];bands=api.positionBands(c,d).bands;near(bands[0].bounds[0],1.2);near(bands[0].bounds[1],1.4);
  c.type='axis_displacement_interval';bands=api.positionBands(c,d).bands;near(bands[0].bounds[0],d.arms.left.pose[2][1]-.3);
  c.reference_frame.quaternion_wxyz=[1,0,0,0];c.parameters.axis=[Math.SQRT1_2,Math.SQRT1_2,0];assert.equal(api.positionBands(c,d).bands.length,0);
});

test('sphere and cylinder wireframes preserve physical radii, not an XYZ box',()=>{
  const{api}=harness(),d=data();
  for(const type of ['point_distance_interval','radial_distance_interval']){
    const c=constraint(type,{axis:[0,0,1],bounds:[.01,.03]}),g=api.spatialGeometry(c,d,3);
    for(const line of g.lines)for(const p of line.points){const v=sub(p,c.reference_frame.origin_m),r=type==='point_distance_interval'?norm(v):Math.hypot(v[0],v[1]);assert.ok(Math.min(Math.abs(r-.01),Math.abs(r-.03))<1e-9);}
  }
});

test('slab boundaries use rotated axis and retain the start-frame anchor',()=>{
  const{api}=harness(),d=data(),c=constraint('axis_displacement_interval',{axis:[0,0,1],bounds:[.02,.08]});
  c.reference_frame.quaternion_wxyz=[Math.SQRT1_2,0,Math.SQRT1_2,0];
  const g=api.spatialGeometry(c,d,4),origin=d.arms.left.pose[2].slice(0,3);
  for(const [i,line] of g.lines.slice(0,2).entries())for(const p of line.points)near(p[0]-origin[0],[.02,.08][i]);
});

test('movement cone is attached to previous frame and rotated into world, not to a fixed target',()=>{
  const{api}=harness(),d=data(),c=constraint('motion_direction_cone',{axis:[1,0,0],half_angle_rad:Math.PI/3,bounds:[0,.0001]});
  c.reference_frame.quaternion_wxyz=[Math.SQRT1_2,0,0,Math.SQRT1_2];
  const g=api.spatialGeometry(c,d,4);near3(g.anchor,d.arms.left.pose[3].slice(0,3));near3(g.axis,[0,1,0]);
  for(const p of g.lines[0].points){const v=sub(p,g.anchor);near(Math.acos(dot(v,[0,1,0])/norm(v)),Math.PI/3);}
  near3(g.arrows.find(a=>a.role==='actual').end,d.arms.left.pose[4].slice(0,3));
  assert.equal(api.spatialGeometry(c,d,5).lines.length,0); // inactive: no plausible-looking geometry
});

test('angle cones support obtuse limits; angular displacement shows original orientation',()=>{
  const{api}=harness(),d=data(),c=constraint('axis_angle_interval',{local_axis:[1,0,0],reference_axis:[0,0,1],bounds:[Math.PI/6,Math.PI*5/6]});
  const g=api.spatialGeometry(c,d,3),origin=d.arms.left.pose[3].slice(0,3);
  for(const [lineIndex,angle]of [[0,Math.PI/6],[5,Math.PI*5/6]])for(const p of g.lines[lineIndex].points){const v=sub(p,origin);near(Math.acos(v[2]/norm(v)),angle);}
  near(api.featureDisplay(c).bounds[1],150);
  assert.equal(JSON.stringify(api.activeFeature(c,[0,100,0,0,0,100])), '[null,null,0,0,0,null]');
  const angular=constraint('angular_displacement_interval',{bounds:[0,.5]});
  d.arms.left.pose[2].splice(3,4,Math.SQRT1_2,0,0,Math.SQRT1_2);
  const a=api.spatialGeometry(angular,d,4).arrows[0];near3(sub(a.end,a.start),[0,.05,0]);
});

test('event and disjoint activation intervals do not paint continuous full-range bounds',()=>{
  const{api,elements}=harness(),d=data(),c=constraint('gripper_interval',{bounds:[0,.02]});
  c.active_frames=[3];c.values=[1,1,1,0,1,1];api.state.detail=d;api.state.frame=3;
  assert.equal(JSON.stringify(api.frameRuns([1,2,4,6,7])),'[[1,2],[4,4],[6,7]]');
  api.lineChart('signal-chart',[{values:c.values,color:'#fff'}],{bands:[{bounds:c.parameters.bounds,frames:c.active_frames}],constraint:c});
  let svg=elements.get('signal-chart').innerHTML;
  assert.equal((svg.match(/class="allowed-band"/g)||[]).length,1);assert.equal((svg.match(/violation-point/g)||[]).length,0);
  c.values[3]=.5;api.lineChart('signal-chart',[{values:c.values,color:'#fff'}],{constraint:c});
  assert.equal((elements.get('signal-chart').innerHTML.match(/violation-point/g)||[]).length,1);
});

test('all eight primitives render finite curves and geometry; overlays can be disabled',()=>{
  const{api,elements}=harness(),d=data();api.state.detail=d;api.state.frame=3;
  for(const type of ['motion_direction_cone','point_distance_interval','radial_distance_interval','axis_position_interval','axis_displacement_interval','axis_angle_interval','angular_displacement_interval','gripper_interval']){
    d.constraints=[constraint(type,{axis:[0,0,1],local_axis:[1,0,0],reference_axis:[0,0,1],half_angle_rad:Math.PI/3})];
    elements.get('signal-kind').value='feature';api.renderCharts();
    for(const id of ['signal-chart','spatial-chart','constraint-chart','constraint-timeline']){const svg=elements.get(id).innerHTML;assert.ok(svg.includes('<svg'),id);assert.doesNotMatch(svg,/NaN|Infinity/);}
    assert.ok(elements.get('signal-chart').innerHTML.includes('allowed-band'));
  }
  elements.get('show-constraints').checked=false;api.renderCharts();
  assert.equal(elements.get('constraint-timeline-wrap').hidden,true);
  assert.ok(!elements.get('signal-chart').innerHTML.includes('allowed-band'));
  assert.ok(!elements.get('spatial-chart').innerHTML.includes('constraint-geometry'));
});

test('frame following respects signal type, arm filter, and pinned selection',async()=>{
  const{api,elements}=harness(),d=data(),grip=constraint('gripper_interval'),cone=constraint('motion_direction_cone',{axis:[1,0,0],half_angle_rad:Math.PI/3}),right=structuredClone(cone);
  right.subject='right_eef';cone.id='__full_motion_v1__left';right.id='__full_motion_v1__right';d.constraints=[grip,cone,right];api.state.detail=d;api.state.frame=3;
  api.renderCharts();assert.equal(api.state.constraint,1); // grip is task-specific but cannot shadow movement
  elements.get('arm-select').value='right';api.renderCharts();assert.equal(api.state.constraint,2);
  api.selectConstraint(0);assert.equal(elements.get('signal-kind').value,'gripper');assert.equal(api.state.constraint,0);assert.equal(api.state.frame,2);
  elements.get('follow-constraints').checked=false;await api.setFrame(5,false);assert.equal(api.state.constraint,0);assert.ok(elements.get('overlay-status').innerHTML.includes('当前未激活'));
  api.selectConstraint(1);assert.equal(elements.get('signal-kind').value,'feature');assert.ok(elements.get('spatial-chart').innerHTML.includes('constraint-geometry'));
});
