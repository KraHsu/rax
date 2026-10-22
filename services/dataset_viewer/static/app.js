'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names = {adjust_bottle:'调整瓶子',beat_block_hammer:'锤击积木',blocks_ranking_rgb:'积木颜色排序',blocks_ranking_size:'积木大小排序',click_alarmclock:'按下闹钟',click_bell:'按铃',dump_bin_bigbin:'倾倒垃圾桶',grab_roller:'双臂抓取滚筒',handover_block:'积木交接',handover_mic:'麦克风交接',hanging_mug:'悬挂马克杯',lift_pot:'抬起锅具',move_can_pot:'移动罐子到锅中',move_pillbottle_pad:'移动药瓶到垫子',move_playingcard_away:'移开扑克牌',move_stapler_pad:'移动订书机到垫子',open_laptop:'打开笔记本电脑',open_microwave:'打开微波炉',pick_diverse_bottles:'抓取不同瓶子',pick_dual_bottles:'双臂抓瓶',place_a2b_left:'左侧物体放置',place_a2b_right:'右侧物体放置',place_bread_basket:'面包放入篮子',place_bread_skillet:'面包放入煎锅',place_burger_fries:'汉堡与薯条放置',place_can_basket:'罐子放入篮子',place_cans_plasticbox:'罐子放入塑料箱',place_container_plate:'容器放到盘子上',place_dual_shoes:'双鞋放置',place_empty_cup:'放置空杯',place_fan:'放置风扇',place_mouse_pad:'鼠标放到垫子上',place_object_basket:'物体放入篮子',place_object_scale:'物体放到秤上',place_object_stand:'物体放到支架',place_phone_stand:'手机放到支架',place_shoe:'放置鞋子',press_stapler:'按压订书机',put_bottles_dustbin:'瓶子放入垃圾桶',put_object_cabinet:'物体放入柜子',rotate_qrcode:'旋转二维码',scan_object:'扫描物体',shake_bottle:'摇晃瓶子',shake_bottle_horizontally:'水平摇晃瓶子',stack_blocks_three:'堆叠三块积木',stack_blocks_two:'堆叠两块积木',stack_bowls_three:'堆叠三个碗',stack_bowls_two:'堆叠两个碗',stamp_seal:'盖章',turn_switch:'转动开关'};
const kinds = {motion_direction_cone:'移动方向锥',axis_position_interval:'轴向位置',radial_distance_interval:'到轴线的距离',point_distance_interval:'到参考点的距离',axis_angle_interval:'轴向夹角',axis_displacement_interval:'轴向位移',angular_displacement_interval:'角位移',gripper_interval:'夹爪指令区间'};
const statusNames = {queued:'等待采集',initializing:'初始化',recording:'采集中',recorded:'录制完成',failed:'需要处理',retry_queued:'等待重试'};
const armName = arm => arm.startsWith('left') ? '左臂' : '右臂';
const number = v => Number(v).toLocaleString('en-US');
const fmt = (v, decimals=4) => Number(v).toFixed(decimals);
const colors = ['#76b8ff','#e8b36d','#b7a4ff'];
const state = {summary:null,task:null,episodes:[],detail:null,index:0,frame:0,constraint:0,filter:'all',view:'inspect',page:0,playing:false,loadToken:0,imageToken:0,frameCache:new Map(),refreshing:false};
let frameAbort = null, seekTimer = null, playToken = 0;

// Geometry stays in physical world coordinates. Never turn a direction cone
// into a positional tube, or an axis-angle interval into Euler component bounds.
const add3=(a,b)=>a.map((v,i)=>v+b[i]);
const sub3=(a,b)=>a.map((v,i)=>v-b[i]);
const mul3=(a,s)=>a.map(v=>v*s);
const dot3=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit3=a=>mul3(a,1/Math.max(Math.hypot(...a),1e-15));
function rotate3(q,v) {
  const norm=Math.hypot(...q),[w,x,y,z]=q.map(n=>n/norm),u=[x,y,z];
  return add3(v,add3(mul3(cross3(u,v),2*w),mul3(cross3(u,cross3(u,v)),2)));
}
function frameRuns(frames) {
  const runs=[];
  for(const f of frames){const last=runs[runs.length-1];if(last&&f===last[1]+1)last[1]=f;else runs.push([f,f]);}
  return runs;
}
function isActive(c,f){return !!c?.active_frames.includes(f);}
function isViolation(c,f){const v=c.values[f],[lo,hi]=c.parameters.bounds;return isActive(c,f)&&(v<lo-1e-9||v>hi+1e-9);}
function constraintArm(c){return c.subject.split('_')[0];}
function featureDisplay(c) {
  const factor=c.unit==='rad'?180/Math.PI:c.type==='motion_direction_cone'?1000:1;
  return {values:c.values.map(v=>v*factor),bounds:c.parameters.bounds.map(v=>v*factor),
    unit:c.unit==='rad'?'°':c.type==='motion_direction_cone'?'mm':c.unit==='fraction'?'0–1':'m',
    label:c.type==='motion_direction_cone'?'方向锥越界量':kinds[c.type]};
}
function activeFeature(c,values) {
  // Inactive hypothetical values can be orders of magnitude larger and flatten
  // a sub-millimetre tolerance band. Plot the actual supervised feature only.
  const frames=new Set(c.active_frames);return values.map((v,f)=>frames.has(f)?v:null);
}
function worldAxis(c){return unit3(rotate3(c.reference_frame.quaternion_wxyz,c.parameters.axis||c.parameters.reference_axis));}
function constraintOrigin(c,d){return c.type==='axis_displacement_interval'?d.arms[constraintArm(c)].pose[c.activation.start].slice(0,3):c.reference_frame.origin_m;}
function positionBands(c,d) {
  if(!c)return {bands:[],note:''};
  const origin=constraintOrigin(c,d),[lo,hi]=c.parameters.bounds;
  if(c.type==='point_distance_interval')return {
    bands:[0,1,2].map(i=>({bounds:[origin[i]-hi,origin[i]+hi],color:colors[i],frames:c.active_frames})),
    note:'色带为球形外边界在 XYZ 上的投影；实际允许区域是球体/球壳，三个区间不能独立判断可行性。'};
  if(['axis_position_interval','axis_displacement_interval'].includes(c.type)){
    const axis=worldAxis(c),i=axis.findIndex(v=>Math.abs(v)>1-1e-8);
    if(i>=0){const bounds=[origin[i]+lo/axis[i],origin[i]+hi/axis[i]].sort((a,b)=>a-b);
      return {bands:[{bounds,color:colors[i],frames:c.active_frames}],note:`${'XYZ'[i]} 轴允许范围${c.type==='axis_displacement_interval'?`，位移从原始 F${c.activation.start} 还原`:''}；其他位置自由度不由此项限制。`};}
    return {bands:[],note:'斜轴约束不拆成 XYZ 区间；切换「所选约束量」查看精确允许范围。'};
  }
  return {bands:[],note:c.type==='motion_direction_cone'?'移动只约束相邻步的大致方向；方向锥见空间图，精确越界量见下方曲线。':
    c.type==='gripper_interval'?'该项作用于夹爪；切换「夹爪指令」查看允许范围。':'该项的允许范围见「所选约束量」及下方曲线。'};
}
function perpendicularBasis(axis) {
  const u=unit3(cross3(axis,Math.abs(axis[2])<.9?[0,0,1]:[1,0,0]));
  return [u,cross3(axis,u)];
}
function circle3(center,axis,radius) {
  const [u,v]=perpendicularBasis(axis);
  return Array.from({length:49},(_,i)=>add3(center,add3(mul3(u,radius*Math.cos(i*Math.PI/24)),mul3(v,radius*Math.sin(i*Math.PI/24)))));
}
function spatialGeometry(c,d,f) {
  const geometry={lines:[],arrows:[],focus:[],note:''};
  if(!c||!isActive(c,f)){geometry.note=c?'所选约束当前未激活；点击生效区间或开启随帧切换。':'当前没有约束标注。';return geometry;}
  const poses=d.arms[constraintArm(c)].pose,p=poses[f].slice(0,3),origin=constraintOrigin(c,d),[lo,hi]=c.parameters.bounds;
  const line=(points,role='allowed')=>{geometry.lines.push({points,role});geometry.focus.push(...points);};
  const arrow=(start,end,role='axis')=>{geometry.arrows.push({start,end,role});geometry.focus.push(start,end);};
  geometry.focus.push(p);
  if(c.type==='point_distance_interval'){
    for(const r of [lo,hi].filter(r=>r>0))for(const axis of [[1,0,0],[0,1,0],[0,0,1]])line(circle3(origin,axis,r));
    geometry.focus.push(origin);arrow(origin,p,'actual');
    geometry.note=`${lo>0?'球壳':'球形'}允许区域：半径 ${fmt(lo*1000,1)}–${fmt(hi*1000,1)} mm。线框按真实尺寸投影。`;
  }else if(['axis_position_interval','axis_displacement_interval'].includes(c.type)){
    const axis=worldAxis(c),[u,v]=perpendicularBasis(axis),tangent=sub3(p,mul3(axis,dot3(sub3(p,origin),axis)));
    const centers=[lo,hi].map(b=>add3(tangent,mul3(axis,b)));
    const corners=centers.map(center=>[[1,1],[-1,1],[-1,-1],[1,-1],[1,1]].map(([a,b])=>add3(center,add3(mul3(u,a*.04),mul3(v,b*.04)))));
    corners.forEach(points=>line(points));for(let i=0;i<4;i++)line([corners[0][i],corners[1][i]],'guide');
    arrow(centers[0],add3(centers[1],mul3(axis,.025)));
    geometry.note=`两平面之间为允许区域，平面横向无限延伸；线框宽度仅用于示意。${c.type==='axis_displacement_interval'?`锚点保留原始 F${c.activation.start}。`:''}`;
  }else if(c.type==='radial_distance_interval'){
    const axis=worldAxis(c),center=add3(origin,mul3(axis,dot3(sub3(p,origin),axis))),ends=[-.04,.04].map(t=>add3(center,mul3(axis,t)));
    for(const r of [lo,hi].filter(r=>r>0)){
      const rings=ends.map(e=>circle3(e,axis,r));rings.forEach(ring=>line(ring));
      for(const i of [0,12,24,36])line([rings[0][i],rings[1][i]]);
    }
    arrow(ends[0],ends[1]);geometry.note='圆柱/圆柱壳允许区域，沿轴无限延伸；图中长度仅用于示意。';
  }else if(c.type==='motion_direction_cone'){
    const axis=worldAxis(c),anchor=poses[Math.max(0,f-1)].slice(0,3),length=.025;
    const ring=circle3(add3(anchor,mul3(axis,length)),axis,length*Math.tan(c.parameters.half_angle_rad));
    line(ring);for(const i of [0,6,12,18,24,30,36,42])line([anchor,ring[i]]);
    arrow(anchor,add3(anchor,mul3(axis,length*1.25)));arrow(anchor,p,'actual');
    geometry.anchor=anchor;geometry.axis=axis;
    geometry.note=`F${Math.max(0,f-1)} → F${f} · 方向半角 ${fmt(c.parameters.half_angle_rad*180/Math.PI,0)}°，越界容差 ${fmt(hi*1000,2)} mm。锥体线框为主体，未画近锥顶的容差扩张；锥体无限延伸，图示轴长 25 mm，无步长限制。金色箭头为实际一步，细小位移可能显示为点。`;
  }else if(c.type==='axis_angle_interval'){
    const axis=worldAxis(c),length=.06,actual=unit3(rotate3(poses[f].slice(3),c.parameters.local_axis));
    for(const angle of [lo,hi]){
      const ring=circle3(add3(p,mul3(axis,length*Math.cos(angle))),axis,length*Math.sin(angle));
      line(ring);for(const i of [0,12,24,36])line([p,ring[i]]);
    }
    arrow(p,add3(p,mul3(axis,length)));arrow(p,add3(p,mul3(actual,length)),'actual');
    geometry.note=`夹爪指定轴与参考轴的允许夹角 ${fmt(lo*180/Math.PI,1)}–${fmt(hi*180/Math.PI,1)}°；箭头长度仅用于示意，绕轴旋转自由。`;
  }else if(c.type==='angular_displacement_interval'){
    for(const axis of [[1,0,0],[0,1,0],[0,0,1]]){
      arrow(p,add3(p,mul3(rotate3(poses[c.activation.start].slice(3),axis),.05)),'anchor');
      arrow(p,add3(p,mul3(rotate3(poses[f].slice(3),axis),.05)),'actual');
    }
    geometry.note=`虚线为原始 F${c.activation.start} 朝向，实线为当前朝向。允许旋转角 ${fmt(lo*180/Math.PI,1)}–${fmt(hi*180/Math.PI,1)}° 由下方角位移曲线精确表示，不等同于分别约束三根轴。`;
  }else geometry.note='夹爪约束显示在动作曲线中；它是开合指令范围，不是末端位置区域。';
  return geometry;
}

async function request(path, attempt=0) {
  const response = await fetch(path, {signal:AbortSignal.timeout(30000)});
  if ((response.status===503 || response.status===404) && attempt<1) {
    await new Promise(resolve=>setTimeout(resolve,300)); return request(path,attempt+1);
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function api(route, params={}) { return `/api/${route}?${new URLSearchParams(params)}`; }
function showError(message) { $('error-banner').textContent=message; $('error-banner').hidden=!message; }
function updateHash() {
  if(state.task) history.replaceState(null,'',`#${new URLSearchParams({task:state.task,episode:state.index,view:state.view})}`);
}
function summaryUI() {
  const s=state.summary;
  $('recorded').textContent=number(s.recorded);$('target').textContent=number(s.target);
  $('annotated').textContent=number(s.annotated);$('ready').textContent=s.ready_tasks;
  $('total-meter').style.width=`${100*s.recorded/s.target}%`;
  $('backlog').textContent=s.recorded===s.annotated ? '标注已跟上录制' : `${s.recorded-s.annotated} 条等待标注`;
  $('workers').textContent=`${s.running_count} 个任务采集中`;
  $('dataset-line').textContent=`${s.dataset} · ${s.tasks.length} 个任务 · 每任务 ${s.tasks[0]?.target || 50} 条成功轨迹`;
  const stale=Date.now()/1000-s.heartbeat_unix>90 && s.batch_status==='running';
  $('connection').innerHTML=`<i></i>${stale?'采集状态待更新':'数据源已连接'}`;
  $('connection').classList.toggle('offline',stale);
  $('updated').textContent=`更新于 ${new Date(s.updated_unix*1000).toLocaleTimeString('zh-CN',{hour12:false})}`;
  renderTasks();taskHeading();
}
function renderTasks() {
  const query=$('search').value.toLowerCase().trim();
  const tasks=state.summary.tasks.filter(t=>(!query || `${t.id} ${names[t.id]||''}`.toLowerCase().includes(query)) &&
    (state.filter==='all' || state.filter==='ready' && t.annotated===t.target || state.filter==='running' && t.gpu!==null));
  $('task-count').textContent=`${tasks.length} / ${state.summary.tasks.length}`;
  $('task-list').innerHTML=tasks.map(t=>`<button class="task-row ${t.id===state.task?'selected':''} ${t.annotated===t.target?'ready':''} ${t.recorded===0?'queued':''}" data-task="${escapeHTML(t.id)}" title="${escapeHTML(t.id)} · ${statusNames[t.status]||t.status}" aria-current="${t.id===state.task?'true':'false'}"><div class="task-top"><span class="task-name">${escapeHTML(names[t.id]||t.id)}</span><span class="task-count">${t.recorded}<span class="muted"> / ${t.target}</span></span></div><div class="task-english">${escapeHTML(t.id)}</div><div class="task-meter"><i style="width:${100*t.recorded/t.target}%"></i></div></button>`).join('') || '<div class="empty">没有匹配的任务</div>';
}
function taskHeading() {
  if(!state.task || !state.summary)return;
  const task=state.summary.tasks.find(t=>t.id===state.task);
  $('task-title').textContent=names[state.task]||state.task;
  $('task-description').textContent=`${state.task} · ${statusNames[task.status]||task.status} · ${task.unique_scenes} 个独立初始场景`;
  $('task-progress').textContent=`${task.recorded} / ${task.target} EPISODES`;
}
function tabsUI() {
  const compare=state.view==='compare', hasData=state.episodes.length>0;
  $('inspect-tab').classList.toggle('active',!compare);$('inspect-tab').setAttribute('aria-selected',String(!compare));
  $('compare-tab').classList.toggle('active',compare);$('compare-tab').setAttribute('aria-selected',String(compare));
  $('inspect-content').hidden=compare||!hasData;$('compare-content').hidden=!compare||!hasData;$('no-data').hidden=hasData;
  $('compare-count').textContent=state.episodes.length;
  if(compare&&hasData)renderCompare();
}
async function selectTask(task, index=0) {
  stop();const token=++state.loadToken;state.task=task;state.detail=null;state.episodes=[];state.page=0;
  renderTasks();taskHeading();showError('');$('instruction').textContent='正在读取真实轨迹…';
  try {
    const data=await request(api('episodes',{task}));if(token!==state.loadToken)return;
    state.episodes=data.episodes;tabsUI();
    if(data.episodes.length) {
      state.index=data.episodes.some(e=>e.index===index)?index:data.episodes[0].index;
      episodeOptions();await loadEpisode(state.index);
    } else { $('no-data').hidden=false;updateHash(); }
  } catch(error) {if(token===state.loadToken)showError(error.message);}
}
function episodeOptions() {
  $('episode-select').innerHTML=state.episodes.map(e=>`<option value="${e.index}">EP ${String(e.index).padStart(3,'0')} · seed ${e.seed}</option>`).join('');
  $('episode-select').value=state.index;
  const position=state.episodes.findIndex(e=>e.index===state.index);
  $('prev-episode').disabled=position<=0;$('next-episode').disabled=position>=state.episodes.length-1;
}
async function loadEpisode(index, preserve=false) {
  stop();const token=++state.loadToken;const previousFrame=preserve?state.frame:0;
  state.index=index;state.detail=null;state.frame=previousFrame;state.constraint=0;
  frameAbort?.abort();episodeOptions();updateHash();
  $('instruction').textContent='正在读取真实轨迹…';$('frame-loading').classList.add('visible');
  for(const id of ['main-image','left-image','right-image'])$(id).removeAttribute('src');
  try {
    const detail=await request(api('episode',{task:state.task,episode:index}));if(token!==state.loadToken)return;
    state.detail=detail;state.frame=Math.min(previousFrame,detail.frames-1);
    $('instruction').textContent=detail.instruction;
    $('episode-badges').innerHTML=`<span class="${detail.native_success?'success':''}">${detail.native_success?'✓ 原生成功':'未确认成功'}</span><span>${detail.frames} FRAMES</span><span>${detail.annotated?'已标注':'待标注'}</span>`;
    $('native-video').disabled=!detail.video;
    $('frame-slider').max=detail.frames-1;$('frame-slider').value=state.frame;
    $('arm-select').value=detail.constraints[0]?.subject.startsWith('right')?'right':'left';
    $('camera-select').innerHTML=detail.cameras.map(c=>`<option value="${escapeHTML(c)}">${escapeHTML(c.replace('_camera','').toUpperCase())} CAMERA</option>`).join('');
    $('camera-select').value=detail.cameras.includes('head_camera')?'head_camera':detail.cameras[0];
    renderCoverage();renderConstraints();renderEvents();renderProvenance();renderCharts();await frameImages();
  } catch(error) {if(token===state.loadToken)showError(`轨迹暂不可用：${error.message}`);}
}
function stop() {playToken++;state.playing=false;$('play').textContent='▶';$('play').setAttribute('aria-label','播放轨迹');}
async function playLoop(token) {
  if(!state.detail || !state.playing || token!==playToken)return;
  const began=performance.now();
  if(state.frame>=state.detail.frames-1){stop();return;}
  await setFrame(state.frame+1);
  if(state.playing && token===playToken)setTimeout(()=>playLoop(token),Math.max(0,1000/Number($('speed').value)-(performance.now()-began)));
}
async function setFrame(frame, images=true) {
  if(!state.detail)return;
  state.frame=Math.max(0,Math.min(state.detail.frames-1,Math.round(frame)));
  $('frame-slider').value=state.frame;renderCharts();
  if(images)await frameImages();
}
async function frameImages() {
  if(!state.detail)return;
  const detail=state.detail,frame=state.frame,token=++state.imageToken;
  frameAbort?.abort();frameAbort=new AbortController();const signal=frameAbort.signal;
  $('frame-loading').textContent=`载入第 ${frame} 帧…`;$('frame-loading').classList.add('visible');
  const cameras=[['main-image',$('camera-select').value],['left-image','left_camera'],['right-image','right_camera']];
  try {
    const images=await Promise.all(cameras.map(async([id,camera])=>{
      if(!detail.cameras.includes(camera))return[id,null];
      const url=api('frame',{task:detail.task,episode:detail.index,frame,camera,width:id==='main-image'?640:320});
      let blobURL=state.frameCache.get(url);
      if(!blobURL){const response=await fetch(url,{signal});if(!response.ok)throw new Error('画面暂不可用');blobURL=URL.createObjectURL(await response.blob());state.frameCache.set(url,blobURL);}
      return[id,blobURL];
    }));
    if(token!==state.imageToken)return;
    for(const [id,url] of images)if(url)$(id).src=url;
    $('frame-stamp').textContent=`FRAME ${String(frame).padStart(3,'0')}`;
    $('frame-loading').classList.remove('visible');
    while(state.frameCache.size>90){const key=state.frameCache.keys().next().value;URL.revokeObjectURL(state.frameCache.get(key));state.frameCache.delete(key);}
  } catch(error){if(error.name!=='AbortError'&&token===state.imageToken){$('frame-loading').textContent='画面加载失败，点击刷新重试';stop();}}
}
function renderCoverage() {
  const report=state.detail.coverage;
  $('coverage-status').textContent=report.complete?'运动全时段覆盖':`运动覆盖 ${(100*report.ratio).toFixed(1)}%`;
  $('coverage-bars').innerHTML=['left','right'].map(arm=>{
    const r=report.per_arm[arm],motion=r.position_or_motion;
    const gaps=motion.gaps.map(([start,end])=>`<i class="coverage-gap" style="left:${100*(start-1)/report.total}%;width:${100*(end-start)/report.total}%" title="缺少几何约束 F${start}–${end-1}"></i>`).join('');
    return `<div class="coverage-row"><span>${armName(arm)}</span><div class="coverage-track">${gaps}</div><b>${motion.covered} / ${motion.total}</b></div><p class="coverage-sub">朝向标注 ${(100*r.orientation.ratio).toFixed(0)}% · 夹爪标注 ${(100*r.gripper.ratio).toFixed(0)}% · 未标注自由度不额外限制</p>`;
  }).join('');
}
function activationLabel(c) {
  const a=c.activation;
  return a.mode==='reference_event'?`${a.event.direction==='decreasing'?'闭合':'打开'}事件 · F${a.event.frame}`:a.mode==='endpoint'?`终点 · F${a.end-1}`:`区间 · F${a.start}–${a.end-1}`;
}
function selectedConstraint(){return state.detail?.constraints[state.constraint];}
function renderConstraints() {
  const d=state.detail;if(!d)return;
  $('constraint-count').textContent=d.constraints.length;
  $('annotation-method').textContent=({programmatic:'规则模板 · 逐轨迹数值标注',rax:'RAX · 独立轨迹审核',pending:'成功轨迹 · 等待约束标注'})[d.annotation_method];
  $('constraint-list').innerHTML=d.constraints.map((c,i)=>`<button class="constraint-item ${i===state.constraint?'active':''}" data-constraint="${i}" title="${escapeHTML(c.id)}"><strong>${escapeHTML(kinds[c.type]||c.type)}<span>${c.evidence.status==='accepted'?'CHECKED':escapeHTML(c.evidence.status.toUpperCase())}</span></strong><small>${armName(c.subject)} · ${escapeHTML(activationLabel(c))}</small></button>`).join('')||'<div class="empty">约束标注正在排队<br>画面和轨迹已经可以查看</div>';
  renderConstraintDetail();
}
function renderConstraintDetail() {
  const c=selectedConstraint();if(!c){$('constraint-detail').innerHTML='';return;}
  const a=c.activation;const valid=c.reference_violation<=1e-6;
  const unit=c.unit==='fraction'?'比例':c.unit;
  $('constraint-detail').innerHTML=`<div class="detail-label">${c.type==='motion_direction_cone'?'方向锥越界量允许值':'允许区间'} · ${unit}</div><div class="bounds">${fmt(c.parameters.bounds[0])}<span>↔</span>${fmt(c.parameters.bounds[1])}</div><div class="detail-row"><span>当前参考值</span><b id="current-feature">${fmt(c.values[state.frame])} ${unit}</b></div><div class="detail-row"><span>当前帧激活</span><b id="current-active">${c.active_frames.includes(state.frame)?'是':'否'}</b></div><div class="detail-row"><span>参考校验</span><b class="${valid?'mint':''}">${valid?'通过':'超出区间'}</b></div><div class="detail-row"><span>惩罚 / 尺度</span><b>${c.penalty.kind==='squared_hinge'?'平方 hinge':'四次 hinge'} / ${c.penalty.scale}</b></div>${c.type==='motion_direction_cone'?`<div class="detail-row"><span>允许方向半角</span><b>${fmt(c.parameters.half_angle_rad*180/Math.PI,0)}°</b></div><div class="detail-row"><span>参考方向轴</span><b>[${c.parameters.axis.map(v=>fmt(v,2)).join(', ')}]</b></div><p class="coverage-note">作用于相邻末端位置之差；允许改变路径和步长。该项本身不保证到达目标。</p>`:''}<div class="evidence"><details><summary>标注依据与参考坐标系</summary><p>${escapeHTML(c.evidence.rationale)}</p><p class="mono">origin: [${c.reference_frame.origin_m.map(v=>fmt(v,4)).join(', ')}]<br>WXYZ: [${c.reference_frame.quaternion_wxyz.map(v=>fmt(v,3)).join(', ')}]</p>${a.mode==='endpoint'&&c.type.includes('displacement')?'<p>位移锚点必须在训练 chunk 内；缺少原始起点时屏蔽该约束。</p>':''}</details></div>`;
}
function renderEvents() {
  const d=state.detail;
  $('events').innerHTML=d.events.map(e=>`<button class="event-chip ${e.arm}" data-frame="${e.frame}">${armName(e.arm)} ${e.direction==='decreasing'?'闭合':'打开'} <span class="mono">F${e.frame}</span></button>`).join('')||'<span class="muted small">无夹爪阈值交叉事件</span>';
  $('event-markers').innerHTML=d.events.map(e=>`<button class="event-tick" data-frame="${e.frame}" style="left:${100*e.frame/(d.frames-1)}%;background:${e.arm==='left'?'#76b8ff':'#83dfbf'}" title="${armName(e.arm)} ${e.direction==='decreasing'?'闭合':'打开'} · F${e.frame}" aria-label="定位第 ${e.frame} 帧"></button>`).join('');
}
function renderProvenance() {
  const d=state.detail;
  const rows=[['Seed',d.seed],['初始场景指纹',d.initial_sha256.slice(0,12)],['位置 / 坐标系','absolute / world'],['旋转表示','WXYZ quaternion'],['夹爪指令','0 closed · 1 open'],['约束修订',`revision ${d.revision}`],['RoboTwin 版本',d.provenance.source_commit?.slice(0,8)||'—']];
  $('provenance').innerHTML=rows.map(([k,v])=>`<dt>${escapeHTML(k)}</dt><dd title="${k==='初始场景指纹'?escapeHTML(d.initial_sha256):escapeHTML(v)}">${escapeHTML(v)}</dd>`).join('');
  const objects=[...(d.initial_scene?.actors||[]),...(d.initial_scene?.task_articulations||[])].filter(x=>!['ground','table','wall'].includes(x.name));
  $('initial-objects').innerHTML=objects.map(o=>`<div class="object-row"><b>${escapeHTML(o.name)}</b><span>xyz [${o.pose.slice(0,3).map(v=>fmt(v,4)).join(', ')}] m</span><span>q [${o.pose.slice(3).map(v=>fmt(v,3)).join(', ')}]</span></div>`).join('')||'<p class="muted small">该记录未包含物体初始位姿。</p>';
}
function lineChart(id, series, options={}) {
  const d=state.detail;if(!d)return;
  const W=640,H=190,L=48,R=15,T=14,B=28,n=d.frames,bands=options.bands||[];
  const values=series.flatMap(s=>s.values).filter(Number.isFinite).concat(bands.flatMap(b=>b.bounds));
  let lo=Math.min(...values),hi=Math.max(...values);
  if(!Number.isFinite(lo)){ $(id).innerHTML='<div class="empty">暂无可显示数据</div>';return; }
  const span=Math.max(hi-lo,options.minSpan||.0001);lo-=span*.12;hi+=span*.12;
  const x=f=>L+f/Math.max(n-1,1)*(W-L-R),y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
  const cell=([a,b])=>[Math.max(L,x(a-.5)),Math.min(W-R,x(b+.5))];
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHTML(options.label||'轨迹曲线')}"><defs><clipPath id="${id}-clip"><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}"/></clipPath></defs>`;
  for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3,yy=y(v);svg+=`<line class="grid" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text x="${L-8}" y="${yy+3}" text-anchor="end">${fmt(v,hi-lo<.003?4:hi-lo<.03?3:2)}</text>`;}
  for(const f of [0,Math.round((n-1)/2),n-1])svg+=`<text x="${x(f)}" y="${H-7}" text-anchor="middle">${f}</text>`;
  svg+=`<g clip-path="url(#${id}-clip)">`;
  for(const run of frameRuns(options.active||[])){
    const[a,b]=cell(run);svg+=`<rect class="activation-band" x="${a}" y="${T}" width="${Math.max(2,b-a)}" height="${H-T-B}" fill="#a2edb4" opacity=".07"/>`;
  }
  for(const band of bands)for(const run of frameRuns(band.frames)){
    const[a,b]=cell(run),[lower,upper]=band.bounds;
    svg+=`<rect class="allowed-band" x="${a}" y="${y(upper)}" width="${Math.max(2,b-a)}" height="${Math.max(1,y(lower)-y(upper))}" fill="${band.color||'#a2edb4'}" fill-opacity=".18" stroke="${band.color||'#a2edb4'}" stroke-opacity=".8" stroke-width="1"/>`;
  }
  for(const s of series){
    let pen=false;const points=s.values.map((v,i)=>{if(!Number.isFinite(v)){pen=false;return '';}const step=`${pen?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`;pen=true;return step;}).join(' ');
    svg+=`<path d="${points}" fill="none" stroke="${s.color}" stroke-width="1.8" stroke-linejoin="round" ${s.dashed?'stroke-dasharray="4 3"':''}/>`;
    s.values.forEach((v,f)=>{if(Number.isFinite(v)&&!Number.isFinite(s.values[f-1])&&!Number.isFinite(s.values[f+1]))svg+=`<circle cx="${x(f)}" cy="${y(v)}" r="2.5" fill="${s.color}"/>`;});
    if(Number.isFinite(s.values[state.frame]))svg+=`<circle cx="${x(state.frame)}" cy="${y(s.values[state.frame])}" r="3" fill="${s.color}" stroke="#101923" stroke-width="1.2"/>`;
  }
  if(options.constraint)for(const f of options.constraint.active_frames)if(isViolation(options.constraint,f))
    svg+=`<circle class="violation-point" cx="${x(f)}" cy="${y(series[0].values[f])}" r="2.5" fill="#ff8e8e"/>`;
  svg+=`<line class="cursor" x1="${x(state.frame)}" x2="${x(state.frame)}" y1="${T}" y2="${H-B}"/></g></svg>`;
  $(id).innerHTML=svg;
  $(id).onclick=event=>{const rect=$(id).querySelector('svg').getBoundingClientRect();stop();setFrame(((event.clientX-rect.left)/rect.width*W-L)/(W-L-R)*(n-1));};
}
function followConstraint() {
  if(!$('follow-constraints').checked)return;
  const arm=$('arm-select').value,current=selectedConstraint(),mode=$('signal-kind').value;
  const relevant=c=>mode==='feature'||(mode==='gripper'?c.type==='gripper_interval':!['gripper_interval','axis_angle_interval','angular_displacement_interval'].includes(c.type));
  if(current&&isActive(current,state.frame)&&(arm==='both'||constraintArm(current)===arm)&&relevant(current))return;
  const candidates=state.detail.constraints.map((c,i)=>({c,i})).filter(({c})=>isActive(c,state.frame)&&(arm==='both'||constraintArm(c)===arm)&&relevant(c));
  // Prefer a task-specific constraint to generic motion coverage; preserve a
  // user's active selection until it expires. This changes the view, not labels.
  const next=candidates.find(({c})=>!c.id.startsWith('__full_motion'))||candidates[0];
  if(next&&next.i!==state.constraint){state.constraint=next.i;renderConstraints();}
}
function selectConstraint(index,frame) {
  const c=state.detail?.constraints[index];if(!c)return;
  state.constraint=index;$('arm-select').value=constraintArm(c);
  $('signal-kind').value=c.type==='gripper_interval'?'gripper':
    ['axis_position_interval','axis_displacement_interval','point_distance_interval'].includes(c.type)?'position':'feature';
  renderConstraints();stop();setFrame(frame??c.active_frames[0]);
}
function renderConstraintTimeline() {
  const host=$('constraint-timeline'),d=state.detail,arm=$('arm-select').value;
  $('constraint-timeline-wrap').hidden=!$('show-constraints').checked;
  if(!$('show-constraints').checked)return;
  const groups=new Map();
  d.constraints.forEach((c,i)=>{if(arm!=='both'&&constraintArm(c)!==arm)return;const key=`${constraintArm(c)}:${c.type}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push({c,i});});
  if(!groups.size){host.innerHTML='<div class="empty">当前机械臂尚无约束</div>';return;}
  const short={motion_direction_cone:'方向',point_distance_interval:'球域',axis_position_interval:'轴位置',axis_displacement_interval:'轴位移',radial_distance_interval:'径向',axis_angle_interval:'轴夹角',angular_displacement_interval:'角位移',gripper_interval:'夹爪'};
  const W=640,L=48,R=15,H=groups.size*22+12,n=d.frames,x=f=>L+f/Math.max(n-1,1)*(W-L-R);
  let svg=`<svg viewBox="0 0 ${W} ${H}" aria-label="各约束激活帧，可点击选择">`;
  let row=0;
  for(const list of groups.values()){
    const y=6+row++*22,c=list[0].c;svg+=`<text x="${L-5}" y="${y+10}" text-anchor="end" class="lane-label">${constraintArm(c)==='left'?'左':'右'}·${short[c.type]}</text><line x1="${L}" x2="${W-R}" y1="${y+6}" y2="${y+6}" class="grid"/>`;
    for(const {c,i} of list)for(const run of frameRuns(c.active_frames)){
      const a=Math.max(L,x(run[0]-.5)),b=Math.min(W-R,x(run[1]+.5)),selected=i===state.constraint,invalid=c.active_frames.some(f=>isViolation(c,f));
      svg+=`<rect data-constraint="${i}" x="${a}" y="${y}" width="${Math.max(3,b-a)}" height="13" rx="2" fill="${invalid?'#ff8e8e':constraintArm(c)==='left'?'#76b8ff':'#83dfbf'}" fill-opacity="${selected?'.85':'.32'}" stroke="${selected?'#e1f9ed':'#56766c'}" stroke-width="${selected?'1.5':'.5'}" tabindex="0" role="button" aria-label="${escapeHTML(`${armName(c.subject)} ${kinds[c.type]} ${activationLabel(c)}`)}"><title>${escapeHTML(`${c.id} · ${activationLabel(c)}`)}</title></rect>`;
    }
  }
  svg+=`<line class="cursor" x1="${x(state.frame)}" x2="${x(state.frame)}" y1="0" y2="${H}" pointer-events="none"/></svg>`;host.innerHTML=svg;
  host.onclick=event=>{const item=event.target.closest('[data-constraint]');if(item)selectConstraint(Number(item.dataset.constraint));};
  host.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();const item=event.target.closest('[data-constraint]');if(item)selectConstraint(Number(item.dataset.constraint));}};
}
function renderCharts() {
  const d=state.detail;if(!d)return;followConstraint();
  $('frame-counter').textContent=`${state.frame} / ${d.frames-1}`;
  const arm=$('arm-select').value,mode=$('signal-kind').value,show=$('show-constraints').checked,c=selectedConstraint();
  const arms=arm==='both'?['left','right']:[arm],matches=c&&arms.includes(constraintArm(c)),visible=show&&matches;
  const series=[],options={label:'参考末端位置'},projection=positionBands(matches?c:null,d);
  let note='';
  if(mode==='feature'&&c){
    const feature=featureDisplay(c);series.push({values:activeFeature(c,feature.values),color:'#b6cbdf'});
    Object.assign(options,{label:feature.label,bands:visible?[{bounds:feature.bounds,frames:c.active_frames}]:[],constraint:visible?c:null});
    $('signal-legend').innerHTML=`<span><i class="line-key"></i>${armName(c.subject)} · ${feature.label} / ${feature.unit}</span>`;
    note=c.type==='motion_direction_cone'?'仅绘制生效帧的方向锥越界量：允许区间内损失为零；它不是位置偏差或步长目标。':'仅绘制所选约束生效帧的参考物理特征，绿色区间是精确约束。';
  }else{
    for(const a of arms){if(mode==='gripper')series.push({values:d.arms[a].gripper,color:a==='left'?'#76b8ff':'#83dfbf'});
      else for(let axis=0;axis<3;axis++)series.push({values:d.arms[a].pose.map(p=>p[axis]),color:colors[axis],dashed:a==='right'&&arms.length===2});}
    if(mode==='gripper'){
      $('signal-legend').innerHTML=arms.map(a=>`<span><i style="background:${a==='left'?'#76b8ff':'#83dfbf'}"></i>${armName(a)}</span>`).join('');
      options.minSpan=.1;options.label='参考夹爪指令';
      if(visible&&c.type==='gripper_interval'){options.bands=[{bounds:c.parameters.bounds,frames:c.active_frames}];if(arms.length===1)options.constraint=c;}
      note='夹爪 0 为闭合，1 为打开；色带仅适用于所选约束的机械臂。';
    }else{
      $('signal-legend').innerHTML=['X','Y','Z'].map((a,i)=>`<span><i style="background:${colors[i]}"></i>${a}</span>`).join('')+(arms.length===2?'<span>实线左臂 / 虚线右臂</span>':'');
      options.bands=visible?projection.bands:[];note=projection.note;
    }
  }
  if(visible)options.active=c.active_frames;
  lineChart('signal-chart',series,options);
  const active=isActive(c,state.frame),bad=c&&isViolation(c,state.frame);
  $('overlay-status').innerHTML=!show?'约束叠加已关闭':!c?'轨迹尚无约束标注':!matches?'所选约束属于另一只机械臂；选择对应机械臂或开启随帧切换。':
    `<span class="constraint-state ${bad?'violated':active?'in-range':'inactive'}">${active?bad?'当前越界':'当前满足':'当前未激活'}</span><b>${armName(c.subject)} · ${escapeHTML(kinds[c.type])}</b><span>${escapeHTML(activationLabel(c))}</span><p>${escapeHTML(note)}</p>`;
  renderConstraintTimeline();renderSpatial();
  if(c){const feature=featureDisplay(c);$('constraint-plot-label').textContent=`${armName(c.subject)} · ${feature.label} · ${activationLabel(c)}`;$('constraint-unit').textContent=feature.unit;
    lineChart('constraint-chart',[{values:activeFeature(c,feature.values),color:'#b6cbdf'}],{active:show?c.active_frames:[],bands:show?[{bounds:feature.bounds,frames:c.active_frames}]:[],constraint:show?c:null,label:feature.label});
    if($('current-feature'))$('current-feature').textContent=`${fmt(c.values[state.frame])} ${c.unit==='fraction'?'比例':c.unit}`;
    if($('current-active'))$('current-active').textContent=active?'是':'否';
  }else{$('constraint-plot-label').textContent='约束标注完成后自动显示';$('constraint-unit').textContent='—';$('constraint-chart').innerHTML='<div class="empty">当前轨迹尚未包含约束</div>';}
}
function renderSpatial() {
  const d=state.detail,mode=$('projection').value,c=selectedConstraint(),show=$('show-constraints').checked;
  const project=p=>mode==='xy'?[p[0],p[1]]:mode==='xz'?[p[0],p[2]]:[p[0]-.7*p[1],p[2]+.25*(p[0]+p[1])];
  const geometry=show?spatialGeometry(c,d,state.frame):{lines:[],arrows:[],focus:[],note:'约束叠加已关闭。'};
  const paths=['left','right'].map(a=>d.arms[a].pose.map(project)),local=$('spatial-zoom').value==='local'&&geometry.focus.length>1;
  const points=local?geometry.focus.map(project):paths.flat().concat(geometry.focus.map(project));
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const scale=Math.min(300/Math.max(.03,xmax-xmin),120/Math.max(.03,ymax-ymin));
  const x=v=>200+(v-(xmin+xmax)/2)*scale,y=v=>83-(v-(ymin+ymax)/2)*scale;
  const screen=p=>{const q=project(p);return [x(q[0]),y(q[1])];};
  const pathData=points=>points.map((p,i)=>{const [a,b]=screen(p);return `${i?'L':'M'}${a.toFixed(2)},${b.toFixed(2)}`;}).join(' ');
  const violation=c&&isViolation(c,state.frame),allowed=violation?'#ff8e8e':'#a2edb4';
  let svg='<svg viewBox="0 0 400 190" role="img" aria-label="末端轨迹与当前约束的空间投影"><defs><clipPath id="spatial-clip"><rect x="15" y="8" width="370" height="152"/></clipPath></defs>';
  for(let i=0;i<5;i++)svg+=`<line class="grid" x1="20" x2="380" y1="${20+i*32}" y2="${20+i*32}"/>`;
  svg+='<g clip-path="url(#spatial-clip)">';
  paths.forEach((path,i)=>{const color=i===0?'#76b8ff':'#83dfbf';svg+=`<path d="${path.map((p,j)=>`${j?'L':'M'}${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ')}" fill="none" stroke="${color}" opacity=".55" stroke-width="1.4"/><circle cx="${x(path[0][0])}" cy="${y(path[0][1])}" r="3" fill="#111b24" stroke="${color}"/>`;});
  for(const line of geometry.lines)svg+=`<path class="constraint-geometry" d="${pathData(line.points)}" fill="none" stroke="${allowed}" stroke-width="1.2" opacity="${line.role==='guide'?'.25':'.8'}" ${line.role==='guide'?'stroke-dasharray="3 4"':''}/>`;
  for(const arrow of geometry.arrows){
    const [a,b]=screen(arrow.start),[u,v]=screen(arrow.end),angle=Math.atan2(v-b,u-a),color=arrow.role==='actual'?'#f4db9c':arrow.role==='anchor'?'#8a9da9':allowed;
    const length=Math.hypot(u-a,v-b);svg+=`<path d="M${a},${b} L${u},${v}" stroke="${color}" stroke-width="${arrow.role==='actual'?2.4:1.4}" ${arrow.role==='anchor'?'stroke-dasharray="3 3"':''}/>`;
    if(length>3)svg+=`<path d="M${u},${v} L${u-5*Math.cos(angle-.45)},${v-5*Math.sin(angle-.45)} L${u-5*Math.cos(angle+.45)},${v-5*Math.sin(angle+.45)} Z" fill="${color}"/>`;
    else svg+=`<circle cx="${u}" cy="${v}" r="2" fill="${color}"/>`;
  }
  paths.forEach((path,i)=>{svg+=`<circle cx="${x(path[state.frame][0])}" cy="${y(path[state.frame][1])}" r="3" fill="${i===0?'#76b8ff':'#83dfbf'}" stroke="#111b24" stroke-width="1"/>`;});
  svg+='</g>';
  const barLength=.01*10**Math.floor(Math.log10(50/(scale*.01))),barPixels=barLength*scale;
  svg+=`<line x1="22" x2="${22+barPixels}" y1="168" y2="168" stroke="#8aa6ba"/><text x="22" y="182">${fmt(barLength*1000,0)} mm${mode==='iso'?'（投影单位）':''}</text><text x="380" y="182" text-anchor="end">${local?'局部 · ':''}${mode==='iso'?'XYZ PROJECTION':mode.toUpperCase()}</text></svg>`;
  $('spatial-chart').innerHTML=svg;$('spatial-note').textContent=geometry.note;
  $('spatial-chart').onclick=event=>{const rect=$('spatial-chart').querySelector('svg').getBoundingClientRect(),px=(event.clientX-rect.left)/rect.width*400,py=(event.clientY-rect.top)/rect.height*190;let best=Infinity,index=0;
    for(const path of paths)path.forEach((p,i)=>{const distance=(x(p[0])-px)**2+(y(p[1])-py)**2;if(distance<best){best=distance;index=i;}});stop();setFrame(index);};
}
function renderCompare() {
  const count=state.episodes.length,pages=Math.ceil(count/12);state.page=Math.min(state.page,Math.max(0,pages-1));
  const unique=new Set(state.episodes.map(e=>e.initial_sha256)).size,seeds=new Set(state.episodes.map(e=>e.seed)).size;
  $('diversity-note').textContent=`${count} 条轨迹 · ${seeds} 个 seed · ${unique} 个独立初始场景。每张图为该轨迹的第 0 帧。`;
  $('compare-grid').innerHTML=state.episodes.slice(state.page*12,(state.page+1)*12).map(e=>`<button class="compare-card" data-episode="${e.index}" aria-label="查看 seed ${e.seed} 的轨迹"><div class="compare-photo"><img loading="lazy" src="${escapeHTML(api('frame',{task:state.task,episode:e.index,frame:0,camera:'head_camera',width:384}))}" alt="seed ${e.seed} 的初始画面"><span>SEED ${e.seed}</span></div><div class="compare-info"><div><span>EP ${String(e.index).padStart(3,'0')} · ${e.frames} F</span><span class="${e.annotated?'mint':''}">${e.annotated?'✓ 已标注':'待标注'}</span></div><small>SCENE ${e.initial_sha256.slice(0,12)} ↗</small></div></button>`).join('');
  $('compare-page').textContent=`${state.page+1} / ${pages}`;$('compare-prev').disabled=state.page===0;$('compare-next').disabled=state.page>=pages-1;
}
async function refresh(initial=false) {
  if(state.refreshing)return;state.refreshing=true;
  try{
    state.summary=await request('/api/summary');summaryUI();showError('');
    if(initial){const hash=new URLSearchParams(location.hash.slice(1));state.view=hash.get('view')==='compare'?'compare':'inspect';const requested=hash.get('task');const task=state.summary.tasks.some(t=>t.id===requested)?requested:state.summary.tasks.find(t=>t.recorded)?.id||state.summary.tasks[0].id;await selectTask(task,Number(hash.get('episode')||0));}
    else if(state.task){const task=state.task,data=await request(api('episodes',{task}));if(task!==state.task)return;const hadData=state.episodes.length>0;state.episodes=data.episodes;episodeOptions();tabsUI();
      if(!hadData&&data.episodes.length)await loadEpisode(data.episodes[0].index);
      else if(state.detail&&!state.playing){const current=data.episodes.find(e=>e.index===state.index);if(current&&(current.annotated&&!state.detail.annotated||current.revision>state.detail.revision))await loadEpisode(state.index,true);}}
  }catch(error){$('connection').innerHTML='<i></i>连接暂时中断';$('connection').classList.add('offline');showError(`无法同步数据：${error.message}。页面保留上次结果，稍后自动重试。`);}
  finally{state.refreshing=false;}
}

$('search').addEventListener('input',()=>state.summary&&renderTasks());
document.querySelector('.filter-group').onclick=event=>{const button=event.target.closest('[data-filter]');if(!button)return;state.filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===button));renderTasks();};
$('task-list').onclick=event=>{const button=event.target.closest('[data-task]');if(button)selectTask(button.dataset.task);};
$('refresh').onclick=()=>refresh();
$('episode-select').onchange=()=>loadEpisode(Number($('episode-select').value));
for(const [id,direction] of [['prev-episode',-1],['next-episode',1]])$(id).onclick=()=>{const index=state.episodes.findIndex(e=>e.index===state.index)+direction;if(state.episodes[index])loadEpisode(state.episodes[index].index);};
for(const [id,view] of [['inspect-tab','inspect'],['compare-tab','compare']])$(id).onclick=()=>{stop();state.view=view;tabsUI();updateHash();};
$('play').onclick=()=>{if(state.playing){stop();return;}if(!state.detail)return;if(state.frame===state.detail.frames-1)state.frame=0;state.playing=true;$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暂停轨迹');playLoop(++playToken);};
$('prev-frame').onclick=()=>{stop();setFrame(state.frame-1);};$('next-frame').onclick=()=>{stop();setFrame(state.frame+1);};
$('frame-slider').oninput=()=>{stop();setFrame(Number($('frame-slider').value),false);clearTimeout(seekTimer);seekTimer=setTimeout(frameImages,65);};
$('camera-select').onchange=()=>frameImages();
for(const id of ['signal-kind','arm-select','projection','show-constraints','follow-constraints','spatial-zoom'])$(id).onchange=renderCharts;
$('constraint-list').onclick=event=>{const button=event.target.closest('[data-constraint]');if(button)selectConstraint(Number(button.dataset.constraint));};
for(const id of ['events','event-markers'])$(id).onclick=event=>{const button=event.target.closest('[data-frame]');if(button){stop();setFrame(Number(button.dataset.frame));}};
$('compare-grid').onclick=event=>{const button=event.target.closest('[data-episode]');if(button){state.view='inspect';tabsUI();loadEpisode(Number(button.dataset.episode));$('inspect-tab').scrollIntoView({block:'start',behavior:'smooth'});}};
$('compare-prev').onclick=()=>{state.page--;renderCompare();};$('compare-next').onclick=()=>{state.page++;renderCompare();};
$('export').onclick=()=>{if(!state.detail)return;const data=structuredClone(state.detail);delete data.arms;data.constraints.forEach(c=>{delete c.values;delete c.active_frames;});const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`${data.task}-episode${data.index}-constraints.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('native-video').onclick=()=>{if(!state.detail?.video)return;stop();$('video').src=api('video',{task:state.task,episode:state.index});$('video-dialog').showModal();};
$('close-video').onclick=()=>$('video-dialog').close();$('video-dialog').onclose=()=>{$('video').pause();$('video').removeAttribute('src');$('video').load();};
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName)||$('video-dialog').open)return;if(event.code==='Space'){event.preventDefault();$('play').click();}if(event.code==='ArrowLeft'){event.preventDefault();stop();setFrame(state.frame-1);}if(event.code==='ArrowRight'){event.preventDefault();stop();setFrame(state.frame+1);}});
refresh(true);setInterval(()=>refresh(),15000);
