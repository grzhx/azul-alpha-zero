const $=id=>document.getElementById(id);const colors=['蓝','黄','红','黑','白'];let revision=-1;
let modelEntries=[], modelPlayerCount=null, gamePlayerCount=null;
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function tile(color,ghost=false){return el('span',`tile ${color===null?'empty':'c'+color} ${ghost?'ghost':''}`,color===null?'':colors[color])}
function source(data,index,center=false){const box=el('div','source');box.append(el('div','source-title',center?'中央共享区':`工厂 ${index+1}`));const list=el('div','tiles');data.forEach((n,c)=>{for(let i=0;i<n;i++)list.append(tile(c))});if(!data.some(Boolean))list.append(el('span','muted','无彩砖'));box.append(list);return box}
function board(player,index,current){const box=el('section',`panel board ${index===current?'current':''}`);const title=el('div','board-title');title.append(el('h2','',player.name),el('span','score',`${player.score} 分`));box.append(title);const grid=el('div','board-grid');grid.append(el('span','muted','图案行'),el('span'),el('span','muted','墙面'));player.patterns.forEach((p,r)=>{const row=el('div','row');for(let i=0;i<r+1;i++)row.append(tile(i>=r+1-p.count?p.color:null));const wall=el('div','wall');for(let c=0;c<5;c++)wall.append(tile((c+5-r)%5,!player.wall[r][c]));grid.append(row,el('span','arrow','→'),wall)});box.append(grid);const floor=el('div','floor');floor.append(el('div','muted',`地板 · ${player.floor.length}/7`));const f=el('div','floor-tiles');player.floor.forEach(c=>f.append(c===5?el('span','tile token','①'):tile(c)));for(let i=player.floor.length;i<7;i++)f.append(tile(null));floor.append(f);box.append(floor);return box}
function render(state){
  const watching=state.status==='watching'&&state.connected;
  $('connection').textContent=state.connected?(watching?'BGA 已连接':'实时连接'):'等待/断线';
  $('connection').className=`badge ${state.connected?'live':'offline'}`;
  $('message').textContent=state.message;
  $('error').hidden=!state.error;$('error').textContent=state.error||'';
  const game=state.game;
  gamePlayerCount=game?.players.length??null;
  filterModels(gamePlayerCount);
  $('waiting').hidden=!!game;$('game').hidden=!game;$('table-link').hidden=!state.table_url;
  if(state.table_url)$('table-link').href=state.table_url;
  $('table-meta').textContent=game?`桌号 ${state.table_id} · ${game.game_name} · ${game.players.length} 人 · ${Number(state.age_seconds??0).toFixed(1)} 秒前更新`:watching?'正在监测 BGA 页面；进入双人或三人 Azul 后会自动显示棋盘。':'请保持 BGA 标签和本页打开。';
  $('waiting-title').textContent=watching?'等待进入 Azul 对局':'等待 BGA 连接';
  $('waiting-copy').textContent=watching?'已检测到 BGA 页面。进入双人或三人 Azul 后将自动同步。':'采集连接断开；重新挂载 BGA 页面后恢复。';
  if(!game)return;
  const factoryCount=game.sources.length-1;
  $('round').textContent=`第 ${game.round} 轮 · ${game.phase}`;
  $('factories').classList.toggle('three',game.players.length===3);
  $('factories').replaceChildren(...game.sources.slice(0,factoryCount).map((data,index)=>source(data,index)));
  const center=source(game.sources[factoryCount],factoryCount,true);
  if(game.token_available)center.querySelector('.tiles').prepend(el('span','tile token','①'));
  $('center').replaceChildren(...center.childNodes);
  $('boards').classList.toggle('three',game.players.length===3);
  $('boards').replaceChildren(...game.players.map((p,i)=>{
    const node=board(p,i,game.current);node.classList.add(`seat-${i}`);
    if(i===game.current)node.querySelector('h2').append(el('span','turn-label','走棋'));
    return node;
  }));
  $('history').replaceChildren(...game.log.map(x=>el('li','',x)));
}
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store'});const s=await r.json();revision=s.revision;render(s);renderSituation(s,s.game);renderAdvice(s,s.game)}catch(e){$('connection').textContent='服务断开';$('connection').className='badge offline'}}setInterval(poll,500);poll();

const monitorToken=document.querySelector('meta[name="monitor-token"]').content;
if(new URLSearchParams(location.search).has('relay'))window.name='azul-bga-monitor';
window.addEventListener('message',event=>{
  let host='';
  try{host=new URL(event.origin).hostname}catch{return}
  if(host!=='boardgamearena.com'&&!host.endsWith('.boardgamearena.com'))return;
  const message=event.data;
  if(!message||message.type!=='azul-bga-monitor'||!['/api/ingest','/api/presence','/api/error'].includes(message.route))return;
  fetch(message.route,{method:'POST',headers:{'Content-Type':'application/json','X-BGA-Monitor-Token':monitorToken},body:JSON.stringify(message.value)})
    .then(response=>{if(response.status===403)location.reload()}).catch(()=>{});
});
let adviceBusy=false;
async function postAdvice(operation){
  if(adviceBusy)return;
  adviceBusy=true;
  try{
    const response=await fetch(`/api/advice/${operation}`,{method:'POST',headers:{'Content-Type':'application/json','X-BGA-Monitor-Token':monitorToken},body:JSON.stringify({checkpoint:$('advice-model').value,budget:$('advice-budget').value})});
    const value=await response.json();
    if(!response.ok)throw new Error(value.error||`HTTP ${response.status}`);
    renderAdvice(value,value.game);
  }catch(error){$('advice-status').textContent=`建议设置失败：${error.message}`}
  finally{adviceBusy=false}
}
function renderAdvice(state,game){
  const advice=state.advice||{status:'waiting',suggestions:[],message:'等待局面'};
  const running=advice.status==='running',ready=advice.status==='ready';
  $('advice-state').textContent=running?'分析中':ready?'建议已更新':advice.status==='error'?'分析失败':'等待局面';
  $('advice-state').className=`badge ${running||ready?'live':'offline'}`;
  const side=game?.players?.[advice.perspective??game?.current]?.name||'当前行动方';
  $('advice-side').textContent=`针对 ${side}`;
  const details=[];
  if(advice.simulations)details.push(`${advice.simulations} 次模拟`);
  if(advice.device)details.push(advice.device);
  if(advice.bag_mode)details.push(`袋内颜色：${advice.bag_mode}`);
  if(advice.checkpoint_iteration!==undefined)details.push(`第 ${advice.checkpoint_iteration} 次发布`);
  $('advice-status').textContent=[advice.message,...details].filter(Boolean).join(' · ');
  if(advice.checkpoint&&[...$('advice-model').options].some(option=>option.value===advice.checkpoint))$('advice-model').value=advice.checkpoint;
  if(advice.budget)$('advice-budget').value=advice.budget;
  $('advice-list').replaceChildren(...(advice.suggestions||[]).map(item=>{
    const row=el('li');
    row.append(el('strong','',`#${item.rank}`),document.createTextNode(item.text),el('span','muted',` · ${(item.confidence*100).toFixed(1)}%`));
    return row;
  }));
  $('advice-start').disabled=adviceBusy||!game||!$('advice-model').value;
  $('advice-stop').disabled=adviceBusy||!advice.enabled;
}
function renderSituation(state,game){
  const panel=$('situation-panel');
  panel.hidden=!game;
  if(!game)return;
  const situation=state.advice?.situation;
  const three=game.players.length===3;
  $('situation-ranks').hidden=!three;
  document.querySelector('.situation-bar').hidden=three;
  document.querySelector('.situation-labels').hidden=three;
  if(three){
    const ranks=situation?.kind==='rank_utility'?situation.expected_ranks:null;
    $('situation-ranks').replaceChildren(...game.players.map((player,index)=>{
      const row=el('div',`rank-row seat-${index}`);
      const value=Number(ranks?.[index]);const known=!!ranks&&Number.isFinite(value);
      row.append(el('span','rank-name',player.name));
      const track=el('div','rank-track');const fill=el('span','rank-fill');
      fill.style.width=known?`${Math.max(0,Math.min(1,(3-value)/2))*100}%`:'0%';
      track.append(fill);row.append(track,el('strong','',known?`名次 ${value.toFixed(2)}`:'名次 --'));
      return row;
    }));
    const best=ranks?Math.min(...ranks):null;
    const leaders=ranks?game.players.filter((_,index)=>Math.abs(ranks[index]-best)<1e-6).map(p=>p.name):[];
    $('situation-summary').textContent=leaders.length?`${leaders.join(' / ')}${situation.exact?'领先':'较优'}`:'等待评估';
    $('situation-note').textContent=situation?.label||state.advice?.message||'等待模型评估';
    return;
  }
  const current=game.current??0;
  const leftName=game.players[0]?.name||'左侧玩家';
  const rightName=game.players[1]?.name||'右侧玩家';
  if(!situation){
    $('situation-left').style.width='0%';$('situation-draw').style.width='0%';$('situation-right').style.width='0%';
    $('situation-label-left').textContent=`${leftName}胜 --`;
    $('situation-label-draw').textContent='平局 --';
    $('situation-label-right').textContent=`${rightName}胜 --`;
    $('situation-summary').textContent='等待评估';
    $('situation-note').textContent=state.advice?.message||'等待模型评估';
    return;
  }
  let wdl=[situation.win,situation.draw,situation.loss].map(value=>Number.isFinite(Number(value))?Math.max(0,Number(value)):0);
  const total=wdl.reduce((a,b)=>a+b,0)||1;
  wdl=wdl.map(value=>value/total);
  const values=current===0?wdl:[wdl[2],wdl[1],wdl[0]];
  const percentages=values.map(value=>(value*100).toFixed(1));
  $('situation-left').style.width=`${values[0]*100}%`;
  $('situation-draw').style.width=`${values[1]*100}%`;
  $('situation-right').style.width=`${values[2]*100}%`;
  $('situation-label-left').textContent=`${leftName}胜 ${percentages[0]}%`;
  $('situation-label-draw').textContent=`平局 ${percentages[1]}%`;
  $('situation-label-right').textContent=`${rightName}胜 ${percentages[2]}%`;
  const best=values.indexOf(Math.max(...values));
  $('situation-summary').textContent=best===0?`${leftName}较优`:best===2?`${rightName}较优`:'平局倾向';
  $('situation-note').textContent=situation.label||'模型 W/D/L 估计';
  document.querySelector('.situation-bar')?.setAttribute('aria-label',`${leftName}胜 ${percentages[0]}%，平局 ${percentages[1]}%，${rightName}胜 ${percentages[2]}%`);
}
async function loadAdviceModels(){
  try{
    const [modelsResponse,optionsResponse]=await Promise.all([fetch('/api/checkpoints',{cache:'no-store'}),fetch('/api/options',{cache:'no-store'})]);
    const models=await modelsResponse.json(),options=await optionsResponse.json();
    modelEntries=models.checkpoints;
    modelPlayerCount=undefined;filterModels(gamePlayerCount);
    if(options.default_checkpoint)$('advice-model').value=options.default_checkpoint;
    if(options.default_budget)$('advice-budget').value=options.default_budget;
  }catch(error){$('advice-status').textContent=`读取 checkpoint 失败：${error.message}`}
}
function filterModels(count){
  if(modelPlayerCount===count||!modelEntries.length)return;
  const selected=$('advice-model').value;
  modelPlayerCount=count;
  const entries=modelEntries.filter(item=>count===null||item.player_count===count);
  $('advice-model').replaceChildren(...entries.map(item=>{
    const option=el('option','',`${item.player_count}人 · ${item.id} · ${item.time}`);option.value=item.id;return option;
  }));
  if(entries.some(item=>item.id===selected))$('advice-model').value=selected;
  if(!entries.length){const option=el('option','','没有适配当前人数的模型');option.value='';$('advice-model').append(option)}
}
$('advice-start').onclick=()=>postAdvice('start');
$('advice-stop').onclick=()=>postAdvice('stop');
$('advice-model').onchange=()=>postAdvice('configure');
$('advice-budget').onchange=()=>postAdvice('configure');
loadAdviceModels();
async function prepareBookmark(){
  const link=$('mount-bookmark');
  try{
    const response=await fetch('/collector.js',{cache:'no-store'});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const source=await response.text();
    link.href='javascript:'+source;
    link.onclick=event=>{
      event.preventDefault();
      link.textContent='请拖到书签栏';
      setTimeout(()=>link.textContent='挂载 Azul 监控',1800);
    };
  }catch(error){
    link.removeAttribute('href');
    link.textContent='书签生成失败';
    link.title=error.message;
  }
}
prepareBookmark();
