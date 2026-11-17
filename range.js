// =================== 練習場模式 ===================
// 用手機鏡頭 + MediaPipe 姿勢偵測，自動切出每一次揮桿，分析節奏與動作。
// 球路（起飛方向、彎曲、擊球品質、高度）由使用者每球點選。

// ---------- 姿勢模型 ----------
const MP_VERSIONS=['0.10.22-rc.20250304','0.10.14'];
const MP_MODEL='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
let POSE=null;
async function loadPose(onMsg){
  if(POSE)return POSE;
  let lastErr;
  for(const v of MP_VERSIONS){
    try{
      onMsg&&onMsg('正在下載姿勢分析模型（第一次約 10 MB）⋯');
      const mp=await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${v}/vision_bundle.mjs`);
      const fs=await mp.FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${v}/wasm`);
      const mk=delegate=>mp.PoseLandmarker.createFromOptions(fs,{baseOptions:{modelAssetPath:MP_MODEL,delegate},runningMode:'VIDEO',numPoses:1,
        minPoseDetectionConfidence:0.5,minPosePresenceConfidence:0.5,minTrackingConfidence:0.5});
      try{POSE=await mk('GPU')}catch(e){POSE=await mk('CPU')}
      return POSE;
    }catch(e){lastErr=e}
  }
  throw lastErr||new Error('load failed');
}

// ---------- 關鍵點 ----------
// MediaPipe 33 點：0 鼻、11/12 左右肩、13/14 左右肘、15/16 左右腕、23/24 左右髖、27/28 左右腳踝
const LM={nose:0,ls:11,rs:12,le:13,re:14,lw:15,rw:16,lh:23,rh:24,lk:25,rk:26,la:27,ra:28};
const BONES=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
const mid=(a,b)=>({x:(a.x+b.x)/2,y:(a.y+b.y)/2});
const dist2=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
// 把一格的關鍵點轉成像素座標與常用的身體部位
function bodyOf(lm,W,H){
  if(!lm||lm.length<29)return null;
  const P=i=>({x:lm[i].x*W,y:lm[i].y*H,v:lm[i].visibility??1});
  const ls=P(11),rs=P(12),lh=P(23),rh=P(24),lw=P(15),rw=P(16);
  if(Math.min(ls.v,rs.v,lh.v,rh.v)<0.3)return null;
  const sm=mid(ls,rs),hp=mid(lh,rh);
  return{nose:P(0),ls,rs,le:P(13),re:P(14),lw,rw,lh,rh,la:P(27),ra:P(28),sm,hp,hand:mid(lw,rw),
    torso:dist2(sm,hp),shW:dist2(ls,rs),hipW:dist2(lh,rh)};
}
function angleAt(a,b,c){const v1={x:a.x-b.x,y:a.y-b.y},v2={x:c.x-b.x,y:c.y-b.y};
  const d=(v1.x*v2.x+v1.y*v2.y)/((Math.hypot(v1.x,v1.y)*Math.hypot(v2.x,v2.y))||1);return Math.acos(Math.max(-1,Math.min(1,d)))*180/Math.PI}

// ---------- 揮桿偵測（每一格呼叫一次） ----------
// 狀態：idle（等待站好）→ ready（準備姿勢）→ swing（上桿、下桿）→ follow（收桿）→ 冷卻
function newDetector(onSwing){
  // 用「那一格本身」記住準備、開始、頂點、擊球，不用索引（舊畫面會被丟掉，索引會位移）
  const D={st:'idle',fr:[],addr:null,sF:null,tS:0,topF:null,impF:null,impT:0,cool:0,lost:0,msg:'請站到畫面中，全身入鏡'};
  const keep=6;// 秒
  D.push=f=>{ // f={t,b(body)}
    D.fr.push(f);while(D.fr.length&&f.t-D.fr[0].t>keep)D.fr.shift();
    const n=D.fr.length-1,b=f.b;
    if(!b){if(++D.lost>8&&D.st!=='idle'){D.st='idle';D.msg='看不到身體，請整個人入鏡'}return}
    D.lost=0;
    if(f.t<D.cool)return;
    const T=D.addr?D.addr.b.torso:b.torso;
    if(D.st==='idle'||D.st==='ready'){
      // 0.4 秒內手幾乎不動，而且手在胸口以下 → 準備姿勢
      const win=D.fr.filter(x=>x.b&&f.t-x.t<=0.4);
      const still=win.length>=4&&win[0].t<=f.t-0.3&&win.every(x=>dist2(x.b.hand,b.hand)<0.07*b.torso);
      const low=b.hand.y>b.sm.y+0.55*b.torso;
      if(still&&low){D.st='ready';D.addr=f;D.msg='準備好了，可以揮桿'}
      else if(D.st==='ready'&&dist2(b.hand,D.addr.b.hand)>0.25*T){
        // 開始上桿：往回找最後一格「手還沒離開」的畫面，再在兩格之間內插出起動時間
        const A=D.addr.b.hand,th=0.03*T;let s=n;
        while(s>0&&D.fr[s-1].b&&dist2(D.fr[s-1].b.hand,A)>th)s--;
        const p0=D.fr[Math.max(0,s-1)],p1=D.fr[s],d0=p0.b?dist2(p0.b.hand,A):0,d1=dist2(p1.b.hand,A);
        D.tS=p0===p1?p1.t:p0.t+(p1.t-p0.t)*Math.max(0,Math.min(1,(th-d0)/((d1-d0)||1)));
        D.sF=p0;D.st='swing';D.topF=f;D.impF=null;D.msg='揮桿中⋯';
      }else if(D.st==='idle')D.msg=low?'站好準備姿勢，手停住一下':'請站好準備姿勢';
      return;
    }
    if(D.st==='swing'){
      if(f.t-D.tS>3){D.st='idle';D.msg='沒有偵測到完整揮桿，請再試一次';return}
      if(b.hand.y<D.topF.b.hand.y)D.topF=f;
      const rose=D.addr.b.hand.y-D.topF.b.hand.y;
      if(rose>0.6*T&&b.hand.y>D.topF.b.hand.y+0.3*T){
        // 下桿：手回到準備姿勢高度附近 → 擊球（在兩格之間內插時間）
        const yI=D.addr.b.hand.y-0.08*T;
        if(b.hand.y>=yI){const p=D.fr[n-1],py=p&&p.b?p.b.hand.y:b.hand.y,k=py<yI&&b.hand.y!==py?(yI-py)/(b.hand.y-py):1;
          D.impF=f;D.impT=p.t+(f.t-p.t)*Math.max(0,Math.min(1,k));D.st='follow';D.msg='收桿中⋯'}
      }
      return;
    }
    if(D.st==='follow'){
      const since=f.t-D.impT,win=D.fr.filter(x=>x.b&&f.t-x.t<=0.25);
      const settled=since>0.5&&win.length>=3&&win.every(x=>dist2(x.b.hand,b.hand)<0.06*T);
      if(settled||since>1.4){
        const sw=analyzeSwing(D.addr,D.tS,D.topF,D.impF,D.impT,f);
        const from=D.fr.indexOf(D.sF);
        D.st='idle';D.cool=f.t+0.8;D.addr=null;
        if(sw){D.msg='分析完成，可以再打下一球';onSwing(sw,D.fr.slice(Math.max(0,from-6)))}
        else D.msg='這一下不像完整的揮桿，請再試一次';
      }
    }
  };
  return D;
}

// ---------- 揮桿分析 ----------
// 回傳時間、節奏與動作指標；距離用「肩膀中點到髖部中點約 50 公分」換算成公分（估算）
function analyzeSwing(addr,tS,topF,impF,impT,finF){
  const tT=topF.t,back=tT-tS,down=impT-tT;
  if(!(back>=0.3&&back<=2.2&&down>=0.08&&down<=0.8))return null;
  const A=addr.b,Tb=topF.b,I=impF.b,F=finF.b;
  const cm=50/A.torso;
  const s=RG.hand==='R'?1:-1; // 正面：往目標方向的 x 正負；後方：往球的方向
  const m={};
  if(RG.view==='face'){
    m.headTop=Math.round(s*(Tb.nose.x-A.nose.x)*cm);          // 上桿頭部側移（負＝遠離目標）
    m.headImp=Math.round(s*(I.nose.x-A.nose.x)*cm);
    m.headRise=Math.round(-(I.nose.y-A.nose.y)*cm);           // 擊球時頭抬起（正＝抬高）
    m.hipBack=Math.round(s*(Tb.hp.x-A.hp.x)*cm);              // 上桿臀部側移（負＝往後滑）
    m.hipImp=Math.round(s*(I.hp.x-A.hp.x)*cm);                // 擊球時重心往目標移動
    m.hipFin=Math.round(s*(F.hp.x-A.hp.x)*cm);
    m.shTurn=Math.round(Math.acos(Math.max(0,Math.min(1,Tb.shW/A.shW)))*180/Math.PI); // 2D 估算
    const lead=RG.hand==='R'?[Tb.ls,Tb.le,Tb.lw]:[Tb.rs,Tb.re,Tb.rw];
    m.leadArm=Math.round(angleAt(...lead));
  }else{
    const spine=b=>Math.atan2(Math.abs(b.sm.x-b.hp.x),b.hp.y-b.sm.y)*180/Math.PI; // 與垂直線夾角
    m.spineAddr=Math.round(spine(A));m.spineImp=Math.round(spine(I));
    m.spineLoss=m.spineAddr-m.spineImp;                       // 正＝擊球時起身
    m.hipThrust=Math.round(s*(I.hp.x-A.hp.x)*cm);             // 正＝臀部往球靠近（提早伸展）
    m.headRise=Math.round(-(I.nose.y-A.nose.y)*cm);
    const lead=RG.hand==='R'?[Tb.ls,Tb.le,Tb.lw]:[Tb.rs,Tb.re,Tb.rw];
    m.leadArm=Math.round(angleAt(...lead));
    m.handsTop=Math.round(-(Tb.hand.y-Tb.sm.y)*cm);           // 頂點時手比肩膀高多少
  }
  return{back:+back.toFixed(2),down:+down.toFixed(2),ratio:+(back/down).toFixed(1),m,tS,tT,tI:impT,tF:finF.t};
}

// ---------- 建議 ----------
function swingTips(sw){
  const t=[],m=sw.m,r=sw.ratio;
  if(r<2.3)t.push({lv:'warn',k:'tempo',t:`節奏 ${r}:1，上桿偏急。試著心裡數「一、二、三」慢慢上桿到頂點，「打」再下桿。職業選手大多在 3:1 左右。`});
  else if(r>4.0)t.push({lv:'warn',k:'tempo',t:`節奏 ${r}:1，下桿相對太慢，或在頂點停太久。到頂點後讓下桿自然加速，不要刻意等。`});
  else t.push({lv:'ok',k:'tempo',t:`節奏 ${r}:1，在理想範圍（約 2.5–3.5:1）。`});
  if(sw.back>1.3)t.push({lv:'info',k:'slow',t:`上桿用了 ${sw.back} 秒，偏慢。大多數好手的上桿約 0.7–1.0 秒。`});
  if(RG.view==='face'){
    if(m.hipBack<=-8)t.push({lv:'warn',k:'sway',t:`上桿時臀部往後滑了約 ${-m.hipBack} 公分（側移）。試著原地轉身，右膝保持彎曲，感覺右臀往後轉而不是往右推。`});
    if(m.headTop<=-10)t.push({lv:'info',k:'head',t:`上桿頂點頭部往後移了約 ${-m.headTop} 公分。頭可以稍微跟著轉，但移太多會讓擊球點不穩定。`});
    if(m.hipImp<3)t.push({lv:'warn',k:'shift',t:`擊球時重心幾乎沒有移到${RG.hand==='R'?'左':'右'}腳（髖部只往目標移了 ${m.hipImp} 公分）。下桿先把重心移到前腳，再轉身。`});
    if(m.headRise>=6)t.push({lv:'warn',k:'lift',t:`擊球時頭抬高了約 ${m.headRise} 公分，容易打薄。試著讓頭的高度保持到擊球後。`});
    if(m.leadArm<140)t.push({lv:'info',k:'arm',t:`上桿頂點${RG.hand==='R'?'左':'右'}手臂彎曲較多（約 ${m.leadArm}°）。保持手臂伸直可以加大揮桿弧度，距離更穩定。`});
    if(m.shTurn<55)t.push({lv:'info',k:'turn',t:`肩膀轉動看起來不足（估計約 ${m.shTurn}°）。上桿時試著讓背部朝向目標。這個數字是從畫面估算的，僅供參考。`});
    if(m.hipFin<8)t.push({lv:'info',k:'finish',t:`收桿時重心沒有完全到前腳。好的收桿應該能用前腳站穩、後腳腳尖點地。`});
  }else{
    if(m.spineLoss>=8)t.push({lv:'warn',k:'early',t:`擊球時上半身起身了約 ${m.spineLoss}°（提早伸展），常造成打薄或右曲。試著保持臀部往後、胸口朝下到擊球。`});
    if(m.hipThrust>=6)t.push({lv:'warn',k:'thrust',t:`下桿時臀部往球的方向靠近了約 ${m.hipThrust} 公分。想像臀部貼著身後的牆轉身。`});
    if(m.headRise>=6)t.push({lv:'warn',k:'lift',t:`擊球時頭抬高了約 ${m.headRise} 公分。`});
    else if(m.headRise<=-6)t.push({lv:'info',k:'dip',t:`擊球時頭往下沉了約 ${-m.headRise} 公分，容易打厚。`});
    if(m.leadArm<140)t.push({lv:'info',k:'arm',t:`上桿頂點${RG.hand==='R'?'左':'右'}手臂彎曲較多（約 ${m.leadArm}°）。`});
  }
  return t;
}
// 球路（依使用者點選）累積的建議
const FL={start:{L:'左',C:'中',R:'右'},curve:{L:'左曲',S:'直',R:'右曲'},contact:{T:'打薄',G:'扎實',F:'打厚'},height:{L:'低',M:'中',H:'高'}};
function flightTips(list){
  const f=list.filter(x=>x.fl);if(f.length<5)return[];
  const cnt=(k,v)=>f.filter(x=>x.fl[k]===v).length,pct=n=>Math.round(n/f.length*100),t=[];
  const slice=RG.hand==='R'?'R':'L',hook=RG.hand==='R'?'L':'R';
  if(pct(cnt('curve',slice))>=40)t.push(`${pct(cnt('curve',slice))}% 的球${FL.curve[slice]}（${RG.hand==='R'?'右曲':'左曲'}球）：通常是擊球時桿面打開，或下桿路徑由外往內。可以試著把握桿稍微加強，下桿讓手從身體內側下來。`);
  if(pct(cnt('curve',hook))>=40)t.push(`${pct(cnt('curve',hook))}% 的球${FL.curve[hook]}：通常是桿面關閉，或路徑太由內往外。握桿可以放鬆一點，轉身不要停。`);
  if(pct(cnt('start',slice))>=40&&cnt('curve','S')>=cnt('curve',hook))t.push(`很多球一開始就往${FL.start[slice]}飛：先檢查站位與瞄準方向，雙腳、髖部、肩膀要和目標線平行。`);
  if(pct(cnt('start',hook))>=40&&cnt('curve',slice)>=1)t.push(`很多球往${FL.start[hook]}起飛又${FL.curve[slice]}：典型的由外往內路徑，常見是下桿時上半身先轉。試著讓下半身先帶動。`);
  if(pct(cnt('contact','T'))>=30)t.push(`打薄 ${pct(cnt('contact','T'))}%：常見原因是擊球時起身或重心留在後腳。`);
  if(pct(cnt('contact','F'))>=30)t.push(`打厚 ${pct(cnt('contact','F'))}%：揮桿最低點在球的後面，下桿要把重心移到前腳。`);
  if(pct(cnt('contact','G'))>=70)t.push(`扎實擊球 ${pct(cnt('contact','G'))}%，很棒！`);
  return t;
}

// =================== 練習場畫面 ===================
const RG=Object.assign({view:'face',hand:'R',club:null,cam:'environment',voice:true},
  (()=>{try{return JSON.parse(localStorage.getItem('golf-range-set'))||{}}catch(e){return{}}})(),
  {on:false,stream:null,det:null,ring:[],pool:[],swings:[],loading:false,err:'',vid:null,cv:null,lastT:-1,lock:null,flashT:0});
function saveRG(){try{localStorage.setItem('golf-range-set',JSON.stringify({view:RG.view,hand:RG.hand,club:RG.club,cam:RG.cam,voice:RG.voice}))}catch(e){}}
let RHIST=(()=>{try{return JSON.parse(localStorage.getItem('golf-range-hist'))||[]}catch(e){return[]}})();
function saveRH(){try{localStorage.setItem('golf-range-hist',JSON.stringify(RHIST.slice(-300)))}catch(e){}}
const clubName=id=>{const c=BAG.clubs.find(x=>x.id===id);return c?c.name:'未指定球桿'};

// 畫面縮圖用的小 canvas（重複使用，避免手機記憶體不夠）
function grabFrame(v){
  const w=144,h=Math.round(w*v.videoHeight/v.videoWidth)||256;
  const c=RG.pool.pop()||document.createElement('canvas');if(c.width!==w||c.height!==h){c.width=w;c.height=h}
  c.getContext('2d').drawImage(v,0,0,w,h);return c;
}
function freeFrame(c){if(RG.pool.length<60)RG.pool.push(c)}
function drawBody(ctx,b,sx,sy,color){
  if(!b)return;const P={11:b.ls,12:b.rs,13:b.le,14:b.re,15:b.lw,16:b.rw,23:b.lh,24:b.rh,27:b.la,28:b.ra};
  ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,3*sx);ctx.lineCap='round';
  BONES.forEach(([a,c])=>{const p=P[a],q=P[c];if(!p||!q)return;
    if(a===23&&c===25||a===24&&c===26||a===25||a===26)return;
    ctx.beginPath();ctx.moveTo(p.x*sx,p.y*sy);ctx.lineTo(q.x*sx,q.y*sy);ctx.stroke()});
  [[11,23],[12,24]].forEach(([a,c])=>{});
  ctx.fillStyle=color;[b.nose,b.lw,b.rw].forEach(p=>{ctx.beginPath();ctx.arc(p.x*sx,p.y*sy,Math.max(3,5*sx),0,7);ctx.fill()});
  // 手的位置特別標示
  ctx.fillStyle='#ffd84d';ctx.beginPath();ctx.arc(b.hand.x*sx,b.hand.y*sy,Math.max(4,6*sx),0,7);ctx.fill();
}
function nearest(arr,t){let best=null,bd=1e9;for(const x of arr){const d=Math.abs(x.t-t);if(d<bd){bd=d;best=x}}return best}

async function startCam(){
  RG.err='';RG.loading=true;renderRange();
  try{
    const pr=loadPose(m=>{const e=document.getElementById('rgLoad');if(e){e.open=true;e.innerHTML='<summary>ⓘ</summary>'+m}});
    RG.stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:RG.cam,width:{ideal:720},height:{ideal:1280},frameRate:{ideal:60}}});
    if(!RG.vid){RG.vid=document.createElement('video');RG.vid.playsInline=true;RG.vid.muted=true;RG.vid.setAttribute('playsinline','');RG.cv=document.createElement('canvas')}
    RG.vid.srcObject=RG.stream;await RG.vid.play().catch(()=>{});
    await pr;
  }catch(e){
    RG.loading=false;
    RG.err=e&&e.name==='NotAllowedError'?'沒有相機權限。請到手機設定允許這個網頁使用相機。':
      (e&&e.name&&e.name.includes('NotFound'))?'找不到相機。':'無法啟動：'+(POSE?'相機打不開':'姿勢分析模型下載失敗，請確認有網路')+'。';
    stopCam(true);renderRange();return;
  }
  RG.loading=false;RG.on=true;RG.det=newDetector(onSwing);RG.lastT=-1;
  try{navigator.wakeLock&&navigator.wakeLock.request('screen').then(l=>RG.lock=l).catch(()=>{})}catch(e){}
  if(RG.voice&&window.speechSynthesis){try{const u=new SpeechSynthesisUtterance('開始，請站好準備姿勢');u.lang='zh-TW';speechSynthesis.speak(u)}catch(e){}}
  renderRange();requestAnimationFrame(camLoop);
}
function stopCam(silent){
  RG.on=false;try{RG.stream&&RG.stream.getTracks().forEach(t=>t.stop())}catch(e){}RG.stream=null;
  try{RG.lock&&RG.lock.release()}catch(e){}RG.lock=null;
  RG.ring.forEach(f=>{if(!f.keep)freeFrame(f.cv)});RG.ring=[];
  if(!silent)renderRange();
}
function camLoop(){
  if(!RG.on)return;requestAnimationFrame(camLoop);
  const v=RG.vid;if(!v||v.readyState<2||v.currentTime===RG.lastT)return;RG.lastT=v.currentTime;
  const ts=performance.now(),t=ts/1000;let res;
  try{res=POSE.detectForVideo(v,ts)}catch(e){return}
  const W=v.videoWidth,H=v.videoHeight,lm=res&&res.landmarks&&res.landmarks[0],b=bodyOf(lm,W,H);
  RG.det.push({t,b});
  RG.ring.push({t,cv:grabFrame(v)});
  while(RG.ring.length&&t-RG.ring[0].t>5){const o=RG.ring.shift();if(!o.keep)freeFrame(o.cv)}
  // 畫骨架
  const cv=RG.cv;if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H}
  const ctx=cv.getContext('2d');ctx.clearRect(0,0,W,H);
  drawBody(ctx,b,1,1,RG.det.st==='ready'?'#7CFC9A':RG.det.st==='idle'?'rgba(255,255,255,.8)':'#ffd84d');
  if(RG.flashT>t){ctx.fillStyle='rgba(0,0,0,.55)';ctx.fillRect(0,H*0.04,W,H*0.13);ctx.fillStyle='#fff';ctx.font=`bold ${Math.round(H*0.07)}px sans-serif`;ctx.textAlign='center';
    ctx.fillText(RG.flash,W/2,H*0.13)}
  const st=document.getElementById('rgStat');if(st&&st.textContent!==RG.det.msg){st.textContent=RG.det.msg;st.className='rgstat '+RG.det.st}
}
function onSwing(sw,frames){
  sw.t=Date.now();sw.club=RG.club;sw.view=RG.view;sw.hand=RG.hand;sw.fl=null;
  sw.tips=swingTips(sw);sw.det=frames;
  // 重播用的畫面：準備前 0.25 秒到收桿
  sw.rep=RG.ring.filter(f=>f.t>=sw.tS-0.25&&f.t<=sw.tF+0.1);sw.rep.forEach(f=>f.keep=true);
  // 四張關鍵畫面
  const keys=[['準備',sw.tS],['頂點',sw.tT],['擊球',sw.tI],['收桿',sw.tF]];
  sw.kf=keys.map(([n,tk])=>{const f=nearest(sw.rep,tk),d=nearest(frames.filter(x=>x.b),tk);if(!f)return{n,src:''};
    const c=document.createElement('canvas');c.width=f.cv.width;c.height=f.cv.height;const x=c.getContext('2d');x.drawImage(f.cv,0,0);
    if(d)drawBody(x,d.b,c.width/RG.vid.videoWidth,c.height/RG.vid.videoHeight,'#7CFC9A');return{n,src:c.toDataURL('image/jpeg',0.7)}});
  RG.swings.unshift(sw);
  // 只留最近 3 球的重播畫面
  RG.swings.slice(3).forEach(s=>{if(s.rep){s.rep.forEach(f=>{f.keep=false;freeFrame(f.cv)});s.rep=null;s.det=null}});
  RHIST.push({t:sw.t,club:sw.club,view:sw.view,hand:sw.hand,back:sw.back,down:sw.down,ratio:sw.ratio,m:sw.m,fl:null});saveRH();sw.hi=RHIST.length-1;
  RG.flash=`節奏 ${sw.ratio} : 1`;RG.flashT=performance.now()/1000+3;
  if(RG.voice&&window.speechSynthesis){try{const u=new SpeechSynthesisUtterance(`上桿 ${sw.back} 秒，下桿 ${sw.down} 秒，節奏 ${sw.ratio} 比 1`);u.lang='zh-TW';u.rate=1.05;speechSynthesis.cancel();speechSynthesis.speak(u)}catch(e){}}
  drawRangeList();
}

function rangeSummaryHTML(){
  const L=RG.swings;if(!L.length)return'';
  const rs=L.map(s=>s.ratio),avg=a=>a.reduce((x,y)=>x+y,0)/a.length,sd=a=>{const m=avg(a);return Math.sqrt(avg(a.map(x=>(x-m)**2)))};
  const ar=avg(rs).toFixed(1),sdr=sd(rs).toFixed(1);
  // 常出現的問題（超過四成的球都有）
  const cnt={};L.forEach(s=>s.tips.filter(t=>t.lv==='warn').forEach(t=>{cnt[t.k]=(cnt[t.k]||0)+1}));
  const common=Object.entries(cnt).filter(([k,n])=>n/L.length>=0.4&&L.length>=3).map(([k])=>L.find(s=>s.tips.some(t=>t.k===k)).tips.find(t=>t.k===k).t);
  const ft=flightTips(L);
  const fl=L.filter(s=>s.fl);
  const dist=k=>Object.entries(FL[k]).map(([v,n])=>{const c=fl.filter(s=>s.fl[k]===v).length;return c?`${n} ${c}`:''}).filter(Boolean).join('、');
  return`<div class="card"><b>📊 這次練習（${L.length} 球）</b>
    <div class="stats" style="margin-top:10px">
      <div class="stat"><b>${ar} : 1</b><span>平均節奏（±${sdr}）</span></div>
      <div class="stat"><b>${avg(L.map(s=>s.back)).toFixed(2)} / ${avg(L.map(s=>s.down)).toFixed(2)}</b><span>上桿 / 下桿（秒）</span></div>
    </div>
    ${L.length>=3&&+sdr>0.4?`<div class="adv warn" style="margin-top:8px">⚠️ 節奏不太穩定（每球差異 ±${sdr}）。可以邊揮邊在心裡固定數拍子。</div>`:''}
    ${common.map(t=>`<div class="adv warn" style="margin-top:8px">🔁 常出現：${t}</div>`).join('')}
    ${fl.length?`<div style="font-size:13px;color:var(--mute);margin-top:10px">球路紀錄（${fl.length} 球）：${['start','curve','contact'].map(dist).filter(Boolean).join('｜')}</div>`:''}
    ${ft.map(t=>`<div class="adv tip" style="margin-top:8px">👉 ${t}</div>`).join('')}
    ${fl.length<5?`<div class="hint">每球打完點一下球路，累積 5 球以上會分析球路問題。</div>`:''}
  </div>`;
}
function swingCardHTML(s,idx,n){const i=idx;
  const m=s.m,chips=s.view==='face'?[
      ['上桿臀部側移',`${m.hipBack>0?'+':''}${m.hipBack} cm`],['擊球重心移動',`${m.hipImp>0?'+':''}${m.hipImp} cm`],['擊球頭部高度',`${m.headRise>0?'+':''}${m.headRise} cm`],
      ['肩膀轉動（估）',`${m.shTurn}°`],['頂點前手臂',`${m.leadArm}°`]]:
    [['脊椎角度',`${m.spineAddr}° → ${m.spineImp}°`],['臀部往球移動',`${m.hipThrust>0?'+':''}${m.hipThrust} cm`],['擊球頭部高度',`${m.headRise>0?'+':''}${m.headRise} cm`],['頂點前手臂',`${m.leadArm}°`]];
  const FI={start:[['L','left',''],['C','arrow',''],['R','right','']],curve:[['L','curveL',''],['S','arrow',''],['R','curveR','']],
    contact:[['T','','薄'],['G','check',''],['F','','厚']],height:[['L','hlow',''],['M','hmid',''],['H','hhigh','']]};
  const FH={start:['arrow','起飛'],curve:['curveR','彎曲'],contact:['club','擊球'],height:['hhigh','高度']};
  const fb=(k)=>`<div class="flrow"><span>${ic(FH[k][0],16)}<small>${FH[k][1]}</small></span>${FI[k].map(([v,i,t])=>`<button data-fl="${idx}:${k}:${v}" class="${s.fl&&s.fl[k]===v?'on':''}" aria-label="${FL[k][v]}">${i?ic(i,22):''}${t?`<b>${t}</b>`:''}</button>`).join('')}</div>`;
  return`<div class="card swing">
    <div class="row" style="justify-content:space-between"><b>第 ${n} 球</b><span style="font-size:13px;color:var(--mute)">${esc(clubName(s.club))}・${s.view==='face'?'正面':'後方'}・${new Date(s.t).toLocaleTimeString('zh-TW',{hour:'2-digit',minute:'2-digit'})}</span></div>
    <div class="tempo"><div><b>${s.back}</b><span>上桿（秒）</span></div><div><b>${s.down}</b><span>下桿（秒）</span></div><div class="hi"><b>${s.ratio} : 1</b><span>節奏</span></div></div>
    <div class="kf">${s.kf.map(k=>`<div>${k.src?`<img src="${k.src}" alt="${k.n}">`:''}<span>${k.n}</span></div>`).join('')}</div>
    ${s.rep?`<div class="rpwrap"><canvas class="rp" id="rp${i}"></canvas><div class="row"><button class="btn ghost sm" data-rp="${i}:0.25">▶ 慢動作</button><button class="btn ghost sm" data-rp="${i}:1">▶ 原速</button></div></div>`:''}
    <div class="chips">${chips.map(([a,b])=>`<div><span>${a}</span><b>${b}</b></div>`).join('')}</div>
    <div class="advbox">${s.tips.map(t=>`<div class="adv ${t.lv}">${t.lv==='warn'?'⚠️':t.lv==='ok'?'✅':'ℹ️'} ${t.t}</div>`).join('')}</div>
    <div class="flhead">${ic('target',16)} 球路</div>
    ${['start','curve','contact','height'].map(fb).join('')}
  </div>`;
}
function drawRangeList(){
  const sum=document.getElementById('rgSum'),list=document.getElementById('rgList');if(!list)return;
  sum.innerHTML=rangeSummaryHTML();
  const N=RG.swings.length;
  list.innerHTML=RG.swings.map((s,i)=>swingCardHTML(s,i,N-i)).join('');
  RG.swings.forEach((s,i)=>{const c=document.getElementById('rp'+i);if(c&&s.rep&&s.rep.length){c.width=s.rep[0].cv.width;c.height=s.rep[0].cv.height;drawRep(c,s,0)}});
  list.querySelectorAll('[data-fl]').forEach(b=>b.onclick=()=>{const[i,k,v]=b.dataset.fl.split(':');const s=RG.swings[+i];
    s.fl=s.fl||{};s.fl[k]=s.fl[k]===v?undefined:v;if(RHIST[s.hi])RHIST[s.hi].fl=s.fl;saveRH();drawRangeList()});
  list.querySelectorAll('[data-rp]').forEach(b=>b.onclick=()=>{const[i,sp]=b.dataset.rp.split(':');playRep(+i,+sp)});
}
function drawRep(c,s,k){
  const f=s.rep[k];if(!f)return;const x=c.getContext('2d');x.drawImage(f.cv,0,0);
  const d=s.det&&nearest(s.det.filter(q=>q.b),f.t);if(d&&RG.vid)drawBody(x,d.b,c.width/RG.vid.videoWidth,c.height/RG.vid.videoHeight,'#7CFC9A');
  const ph=f.t<s.tT?'上桿':f.t<s.tI?'下桿':'收桿';x.fillStyle='rgba(0,0,0,.55)';x.fillRect(0,0,c.width,20);x.fillStyle='#fff';x.font='bold 12px sans-serif';x.fillText(ph,6,14);
}
function playRep(i,speed){
  const s=RG.swings[i],c=document.getElementById('rp'+i);if(!s||!s.rep||!c)return;
  const t0=performance.now(),base=s.rep[0].t,end=s.rep[s.rep.length-1].t;
  const step=()=>{const el=(performance.now()-t0)/1000*speed+base;let k=s.rep.findIndex(f=>f.t>=el);if(k<0)k=s.rep.length-1;drawRep(c,s,k);if(el<end)requestAnimationFrame(step)};
  requestAnimationFrame(step);
}

const VIEWHELP={
  face:'正面：手機放在你正前方約 3 公尺、腰的高度，整個人（含揮桿範圍）都要入鏡。',
  dtl:'後方：手機放在你身後、沿目標線延長約 3 公尺，腰到胸口的高度，整個人都要入鏡。'};
const clubOpts=()=>BAG.clubs.map(c=>`<option value="${c.id}" ${RG.club===c.id?'selected':''}>${esc(c.name)}</option>`).join('');
// 在練習場直接新增球桿（加進球袋）
function addClubRange(){
  $('#modalBody').innerHTML=`<div style="text-align:center;color:var(--g)">${ic('bag',36)}</div>
    <label>球桿名稱</label><input id="ncName" placeholder="例如：3 號混血">
    <label>平常打多遠（${BAG.unit}，可不填）</label><input id="ncEst" type="number" inputmode="numeric">
    <div class="row"><button class="btn ghost" id="mNo">${ic('x',18)}</button><button class="btn" id="mYes" style="flex:3">${ic('plus',18)} 加入球袋</button></div>`;
  $('#modal').classList.add('show');setTimeout(()=>$('#ncName').focus(),50);
  $('#mNo').onclick=()=>$('#modal').classList.remove('show');
  $('#mYes').onclick=()=>{const n=$('#ncName').value.trim();if(!n)return;const id='c'+Date.now();
    BAG.clubs.push({id,name:n,est:+$('#ncEst').value||null,shots:[]});saveBag();RG.club=id;saveRG();$('#modal').classList.remove('show');
    if(RG.on){const sel=document.getElementById('rgClub2');if(sel)sel.innerHTML=clubOpts()}else renderRange()};
}
function renderRange(){
  document.body.classList.add('starting');
  if(!RG.club&&BAG.clubs.length)RG.club=(BAG.clubs.find(c=>/7/.test(c.name))||BAG.clubs[0]).id;
  const seg=(k,arr)=>`<div class="icrow">${arr.map(([v,i,t])=>`<button data-rg="${k}:${v}" class="${RG[k]===v?'on':''}">${ic(i,24)}<small>${t}</small></button>`).join('')}</div>`;
  const clubRow=id=>`<div class="rgclub">${ic('club',24)}<select id="${id}">${clubOpts()}</select><button class="btn sm" id="rgAddClub" aria-label="新增球桿">${ic('plus',20)}</button></div>`;
  $('#app').innerHTML=`<div class="card">
    <div class="row" style="justify-content:space-between"><span class="rghead">${ic('golfer',28)} 練習場</span><button class="icbtn dark" id="rgExit" aria-label="離開">${ic('x',20)}</button></div>
    ${RG.on?`<div class="rgcam" id="rgCam"><div class="rgstat idle" id="rgStat">請站到畫面中</div></div>
      <div class="row" style="margin-top:8px">${clubRow('rgClub2')}<button class="btn sm" id="rgStop" style="margin:0;background:var(--red)">${ic('stop',20)}</button></div>`:
    `${seg('view',[['face','front','正面'],['dtl','side','後方']])}
    <div class="hint" style="margin-top:6px">${VIEWHELP[RG.view]}</div>
    ${seg('hand',[['R','right','右打'],['L','left','左打']])}
    ${clubRow('rgClub')}
    ${seg('cam',[['environment','camera','後鏡頭'],['user','swap','前鏡頭']])}
    ${seg('voice',[[true,'speaker','語音開'],[false,'mute','語音關']])}
    ${RG.err?`<div class="adv warn" style="margin-top:10px">${RG.err}</div>`:''}
    <button class="btn rgstart" id="rgStart" ${RG.loading?'disabled':''}>${ic('camera',28)} ${RG.loading?'準備中⋯':'開始'}</button>
    <details class="hint" id="rgLoad"><summary>ⓘ</summary>${RG.loading?'正在開啟相機與分析模型⋯':'第一次使用需要下載姿勢分析模型（約 10 MB），請在有網路的地方開啟。影像只在你的手機上分析，不會上傳。'}</details>`}
  </div>
  <div id="rgSum"></div><div id="rgList"></div>`;
  $('#rgAddClub').onclick=addClubRange;
  if(RG.on){const box=$('#rgCam');box.prepend(RG.cv);box.prepend(RG.vid);RG.vid.play().catch(()=>{});
    $('#rgStop').onclick=()=>stopCam();$('#rgClub2').onchange=e=>{RG.club=e.target.value;saveRG()}}
  else{
    document.querySelectorAll('[data-rg]').forEach(b=>b.onclick=()=>{const[k,v]=b.dataset.rg.split(':');RG[k]=v==='true'?true:v==='false'?false:v;saveRG();renderRange()});
    $('#rgClub').onchange=e=>{RG.club=e.target.value;saveRG()};
    $('#rgStart').onclick=()=>{if(RG.voice&&window.speechSynthesis){try{speechSynthesis.speak(new SpeechSynthesisUtterance(''))}catch(e){}}startCam()};
  }
  $('#rgExit').onclick=()=>{stopCam(true);START.step='home';view='start';render();window.scrollTo(0,0)};
  drawRangeList();
}
