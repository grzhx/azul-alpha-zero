const $ = id => document.getElementById(id);
const colors = ['蓝','黄','红','黑','白'];
const token = document.querySelector('meta[name="azul-token"]').content;
let state = null, selected = null, sending = false, adviceSending = false, catalog = [], lastRendered = -1, lastAdviceRendered = -1, defaults = null;
const configKeys=['device','simulations','candidates','chance_cap','max_depth','chance_initial','chance_coefficient','chance_exponent','chance_sensitivity','q_mode','q_floor','value_scale','maxvisit_init','gumbel_scale','afterstate_prior','cache_capacity','subtree_reuse','bf16','cuda_graph'];
function readConfig(){const c={};for(const k of configKeys){const n=$(k.replaceAll('_','-'));if(n.type==='number'){const value=Number(n.value);if(!Number.isFinite(value))throw new Error(`参数 ${k} 必须是数字`);c[k]=value}else c[k]=n.type==='checkbox'?n.checked:(k==='q_mode'?Number(n.value):n.value)}return c}
function setConfig(c){for(const k of configKeys){const n=$(k.replaceAll('_','-'));if(n.type==='checkbox')n.checked=!!c[k];else n.value=c[k]}}
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function tile(color, ghost=false, small=false) {
  const node = el('span', `tile ${color===null?'empty':'c'+color} ${ghost?'ghost':'filled'} ${small?'small':''}`, color===null?'':colors[color]);
  node.title = color===null?'空格':`${colors[color]}砖${ghost?'预印位置（未铺）':''}`;
  return node;
}
async function api(path, data) {
  const response = await fetch(path, data===undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-Azul-Token':token},body:JSON.stringify(data)});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `请求失败 ${response.status}`);
  return body;
}
function showError(message) { $('error').textContent=message || ''; $('error').hidden=!message; }
function modelInfo() {
  const item=catalog.find(x=>x.id===$('checkpoint').value);
  $('model-info').textContent=item ? `${item.time} · ${(item.bytes/1048576).toFixed(1)} MiB${item.kind==='latest'?' · 完整训练快照，加载较慢，优先选 actor':''}` : '未找到模型。可以训练产生 checkpoint 后刷新。';
}
async function refreshModels() {
  $('refresh').disabled=true;
  try {
    const previous=$('checkpoint').value;
    catalog=(await api('/api/checkpoints')).checkpoints;
    $('checkpoint').replaceChildren();
    for(const x of catalog) {
      const option=el('option','',`${x.id} · ${x.time}`);option.value=x.id;$('checkpoint').append(option);
    }
    if(catalog.some(x=>x.id===previous)) $('checkpoint').value=previous;
    else {
      const preferred=catalog.find(x=>x.run==='optimized' && x.kind==='actor') || catalog[0];
      if(preferred) $('checkpoint').value=preferred.id;
    }
    modelInfo(); render();
  } catch(e) {showError(e.message);} finally {$('refresh').disabled=false;}
}
const playable=()=>state?.game && !state.busy && !sending && !state.game.terminal && !state.game.waiting_deal && (state.game.mode==='human'||state.game.current===state.game.human);
let dealValues=Array(20).fill(-1),dealKey='';
function renderDeal(){
  const g=state?.game;$('deal-panel').hidden=!g?.waiting_deal;if(!g?.waiting_deal)return;
  const key=g.seed+':'+g.round+':'+g.moves;if(key!==dealKey){dealValues=Array(20).fill(-1);dealKey=key;}
  $('deal-stock').textContent='当前袋：'+colors.map((c,i)=>`${c} ${g.bag[i]}`).join(' · ')+' ｜ 弃砖：'+colors.map((c,i)=>`${c} ${g.discard[i]}`).join(' · ');
  const bag=[...g.bag],discard=[...g.discard];let valid=true,complete=true;
  $('deal-slots').replaceChildren();
  for(let f=0;f<5;f++){
    const box=el('div','source');box.append(el('div','source-title',`工厂 ${f+1}`));
    for(let j=0;j<4;j++){
      const i=f*4+j;if(!bag.some(x=>x)){for(let c=0;c<5;c++){bag[c]=discard[c];discard[c]=0;}}
      const select=el('select');select.setAttribute('aria-label',`工厂 ${f+1} 第 ${j+1} 块`);
      const blank=el('option','','请选择');blank.value=-1;select.append(blank);
      colors.forEach((c,k)=>{const o=el('option','',`${c}（剩 ${bag[k]}）`);o.value=k;o.disabled=valid&&complete&&bag[k]===0;select.append(o);});
      if(valid&&complete&&!bag.some(x=>x)){const empty=el('option','','砖已用尽');empty.value=255;select.append(empty);dealValues[i]=255;}
      select.value=dealValues[i];select.disabled=state.busy||sending;
      select.onchange=()=>{dealValues[i]=Number(select.value);renderDeal();};box.append(select);
      const c=dealValues[i];if(c===-1)complete=false;else if(valid&&complete){if(c===255){if(bag.some(x=>x))valid=false;}else if(!bag[c])valid=false;else bag[c]--;}
    }$('deal-slots').append(box);
  }
  $('deal-status').textContent=!valid?'颜色超出当前袋可用数量，请修改。':!complete?'请选择全部 20 个槽位。':'发牌有效，可以提交。';
  $('deal-submit').disabled=!valid||!complete||state.busy||sending;
}
function actionFor(destination) { return (selected.source*5+selected.color)*6+destination; }
function selectGroup(source,color,destination=null) {selected={source,color,destination};renderGame();}
function sourceContent(game,source) {
  const groups=el('div','color-groups');
  game.sources[source].forEach((count,color)=>{
    if(!count)return;
    const button=el('button','pick');
    button.append(tile(color,false,true),el('span','',`×${count}`));
    button.disabled=!playable();
    button.setAttribute('aria-label',`${source===5?'中央':'工厂 '+(source+1)} ${colors[color]}砖 ${count} 块`);
    button.setAttribute('aria-pressed',String(selected?.source===source && selected?.color===color));
    if(selected?.source===source && selected?.color===color)button.classList.add('selected');
    button.onclick=()=>selectGroup(source,color);groups.append(button);
  });
  if(!game.sources[source].some(x=>x))groups.append(el('span','muted','无彩砖'));
  return groups;
}
function renderBoard(game,player) {
  const p=game.boards[player],mine=game.mode==='human'?player===game.current:player===game.human;
  const playerName=game.mode==='human'?`玩家 ${player+1}`:(player===game.human?'你':'AI');
  const section=el('section',`panel board ${!game.terminal && game.current===player?'active':''}`);
  section.setAttribute('aria-label',`${playerName}棋盘`);
  const heading=el('div','board-title');
  heading.append(el('h2','',`${playerName}${game.mode==='ai'?` · 玩家 ${player+1}`:''}${game.token_owner===player?' · 下轮先手':''}`));
  const score=el('span','score',String(p.score));score.append(el('small','','分'));heading.append(score);section.append(heading);
  const grid=el('div','board-grid');grid.append(el('span','column-title','图案行（右侧填入）'),el('span'),el('span','column-title','墙面'));
  p.patterns.forEach((pattern,row)=>{
    const target=el('button','pattern-row target');
    target.append(el('span','pattern-label',`${row+1} 行`));
    for(let c=0;c<row+1;c++)target.append(tile(c>=row+1-pattern.count?pattern.color:null));
    const can=mine && playable() && selected && game.legal.includes(actionFor(row));
    target.disabled=!can;
    if(selected && mine)target.classList.add(can?'available':'unavailable');
    if(can && selected.destination===row)target.classList.add('recommended');
    target.setAttribute('aria-label',`放入第 ${row+1} 行`);
    if(can) {
      const total=game.sources[selected.source][selected.color],kept=Math.min(total,row+1-pattern.count);
      target.title=`放入 ${kept} 块，${total-kept} 块落地`;
      target.onclick=()=>send('move',{action:actionFor(row)});
    }
    const wall=el('div','wall-row');
    for(let col=0;col<5;col++)wall.append(tile((col+5-row)%5,!p.wall[row][col]));
    grid.append(target,el('span','arrow','→'),wall);
  });section.append(grid);
  const floor=el('div','floor'),title=el('div','floor-title');
  const penalties=[0,1,2,4,6,8,11,14];
  title.append(el('span','','地板'),el('span','',`本轮扣 ${penalties[p.floor_count]} 分`));floor.append(title);
  const slots=el('div','floor-slots'),contents=[];
  if(p.floor_marker)contents.push('token');
  p.floor_tiles.forEach((n,c)=>{for(let i=0;i<n;i++)contents.push(c);});
  [1,1,2,2,2,3,3].forEach((penalty,i)=>{
    const cell=el('span','floor-cell');cell.append(el('span','penalty',`−${penalty}`));
    cell.append(contents[i]==='token'?el('span','tile token','1'):tile(i<contents.length?contents[i]:null));slots.append(cell);
  });floor.append(slots);
  if(mine) {
    const button=el('button','floor-target','整组放到地板');
    const can=playable() && selected && game.legal.includes(actionFor(5));button.disabled=!can;
    if(can) {button.classList.add('available');button.onclick=()=>send('move',{action:actionFor(5)});}
    if(can && selected.destination===5)button.classList.add('recommended');
    floor.append(button);
  }
  section.append(floor,el('div','footer-note',`完整横行：${p.completed_rows} · 浅色墙格尚未铺砖`));return section;
}
function renderGame() {
  const game=state?.game;if(!game)return;
  renderDeal();
  if(selected && !game.sources[selected.source][selected.color])selected=null;
  $('round').textContent=`第 ${game.round} 轮 · 第 ${game.moves} 步`;
  $('factories').replaceChildren();
  for(let s=0;s<5;s++) {const box=el('div','source');box.append(el('div','source-title',`工厂 ${s+1}`),sourceContent(game,s));$('factories').append(box);}
  const centerTitle=el('div','source-title','中央共享区');
  if(game.token_available)centerTitle.append(document.createTextNode('　'),el('span','token','① 起始标记'));
  $('center').replaceChildren(centerTitle,sourceContent(game,5));
  $('selection').replaceChildren();
  if(selected && playable()) {
    const count=game.sources[selected.source][selected.color];
    $('selection').append(tile(selected.color,false,true),el('span','',`已选 ${count} 块${colors[selected.color]}砖，请点击下方绿色图案行或地板。`));
    const cancel=el('button','secondary','取消选择');cancel.onclick=()=>{selected=null;renderGame();};$('selection').append(cancel);
  } else $('selection').textContent=game.terminal?game.result:(playable()?'① 选择来源中的一种颜色　② 选择当前玩家的图案行或地板':'等待 AI 行动…');
  $('boards').replaceChildren(renderBoard(game,game.human),renderBoard(game,1-game.human));
  $('last-move').textContent=game.last_move?.text || '暂无行动';
  $('history').replaceChildren(...game.log.map(text=>el('li','',text)));
  $('history').scrollTop=$('history').scrollHeight;
}
function render() {
  if(!state)return;
  const game=state.game;
  $('welcome').hidden=!!game;$('game').hidden=!game;
  $('status').textContent=state.busy?state.message:(game?.terminal?game.result:state.message);
  $('status').classList.toggle('thinking',state.busy);
  $('game-meta').textContent=game?`本盘模型：${game.checkpoint} · 第 ${game.checkpoint_iteration} 次发布 · ${game.device} · ${game.config.simulations} 次搜索 · ${game.config.candidates} 候选` : '';
  $('new').disabled=state.busy || sending || !$('checkpoint').value;
  $('undo').disabled=state.busy || sending || !game?.can_undo;
  $('apply-config').disabled=state.busy||sending||!game;
  $('device-badge').textContent=game?`${game.device} · ${game.config.bf16?'BF16':'FP32'}`:'CPU / CUDA';
  $('retry').hidden=!state.error || !game || game.mode==='human' || game.terminal || game.current===game.human;
  $('retry').disabled=state.busy || sending;
  renderSituation(game);
  renderAdvice(game);
  showError(state.error);
  renderGame();
}
function renderSituation(game){
  const panel=$('situation-panel');panel.hidden=!game;if(!game)return;
  const s=game.situation||{human:0,draw:1,ai:0,label:'暂无评估'};
  const hp=s.human*100,dp=s.draw*100,ap=s.ai*100;
  $('situation-human').style.width=`${hp}%`;$('situation-draw').style.width=`${dp}%`;$('situation-ai').style.width=`${ap}%`;
  $('situation-human-label').textContent=`${s.left_label||'你'} ${hp.toFixed(1)}%`;
  $('situation-draw-label').textContent=`和棋 ${dp.toFixed(1)}%`;
  $('situation-ai-label').textContent=`${s.right_label||'AI'} ${ap.toFixed(1)}%`;
  $('situation-note').textContent=s.exact?'终局确定结果':`${s.label} · 当前 checkpoint 的 W/D/L 预测`;
  $('situation-summary').textContent=s.exact?game.result:hp>ap?`${s.left_label||'你'} 领先 ${(hp-ap).toFixed(1)}%`:ap>hp?`${s.right_label||'AI'} 领先 ${(ap-hp).toFixed(1)}%`:'局势均衡';
  $('situation-bar').setAttribute('aria-label',`${s.left_label||'你'}获胜 ${hp.toFixed(1)}%，和棋 ${dp.toFixed(1)}%，${s.right_label||'AI'}获胜 ${ap.toFixed(1)}%`);
}
const adviceAvailable=game=>game&&!game.terminal&&!game.waiting_deal&&(game.mode==='human'||game.current===game.human);
function renderAdvice(game){
  const panel=$('advice-panel');panel.hidden=!game;if(!game)return;
  const advice=state.advice||{status:'idle',suggestions:[],simulations:0};
  const available=adviceAvailable(game);
  $('advice-start').disabled=!available||adviceSending;
  $('advice-stop').hidden=advice.status!=='running';
  $('advice-stop').disabled=adviceSending;
  const side=game.mode==='human'?`玩家 ${game.current+1}`:'玩家方';
  $('advice-status').textContent=!available?(game.waiting_deal?'完成发牌后可获取建议':game.terminal?'对局已结束':'AI 回合不显示玩家建议'):
    advice.status==='running'?`${side}：已搜索 ${advice.simulations} 次，${advice.message||'持续分析中'}`:
    advice.status==='ready'?`${side}：${advice.simulations} 次模拟，分析完成`:
    advice.status==='stopped'?`${side}：分析已停止，可点击“重新分析”继续`:
    advice.status==='error'?`分析失败：${advice.message}`:`展开或点击“重新分析”获取${side}建议`;
  $('advice-list').replaceChildren();
  for(const item of advice.suggestions||[]){
    const li=el('li','advice-item');
    const text=el('span','',`${item.text} · ${(item.confidence*100).toFixed(1)}%`);
    const show=el('button','secondary','在棋盘标出');show.disabled=!available;
    show.onclick=()=>selectGroup(item.source,item.color,item.destination);
    li.append(text,show);$('advice-list').append(li);
  }
  if(panel.open&&available&&advice.status==='idle'&&!adviceSending)setTimeout(()=>adviceCommand('start'),0);
}
async function adviceCommand(operation){
  if(adviceSending)return;adviceSending=true;render();
  try{state=await api(`/api/advice/${operation}`,operation==='start'?{budget:$('advice-budget').value}:{ });lastAdviceRendered=state.advice_revision;}
  catch(e){showError(e.message)}finally{adviceSending=false;render()}
}
async function send(operation,data={}) {
  if(sending || state?.busy)return;
  sending=true;render();selected=null;
  try {state=await api('/api/'+operation,{...data,revision:state.revision});lastRendered=state.revision;lastAdviceRendered=state.advice_revision;}
  catch(e) {showError(e.message);sending=false;await poll();return;}
  finally {sending=false;}
  render();
}
async function poll() {
  try {
    const updated=await api('/api/state');
    if(!state || updated.revision!==lastRendered || updated.advice_revision!==lastAdviceRendered) {const gameChanged=!state||updated.revision!==lastRendered;state=updated;lastRendered=updated.revision;lastAdviceRendered=updated.advice_revision;if(gameChanged)selected=null;render();}
  }catch(e){showError('无法连接对战服务，请检查服务是否仍在运行。');}
}
$('new').onclick=()=>{
  if(state?.game && !state.game.terminal && !window.confirm('结束当前对局并重新开始？'))return;
  try{send('new',{checkpoint:$('checkpoint').value,mode:$('mode').value,human:Number($('human').value),seed:$('seed').value,manual_deal:$('manual-deal').checked,config:readConfig()});}catch(e){showError(e.message)}
};
$('checkpoint').onchange=()=>{modelInfo();if(state)$('new').disabled=state.busy || sending || !$('checkpoint').value;};
$('refresh').onclick=refreshModels;$('undo').onclick=()=>send('undo');$('retry').onclick=()=>send('retry');
$('defaults').onclick=()=>{if(defaults)setConfig(defaults)};
$('deal-submit').onclick=()=>send('deal',{colors:dealValues.slice()});
$('deal-clear').onclick=()=>{dealValues=Array(20).fill(-1);renderDeal();};
$('apply-config').onclick=()=>{try{send('configure',{config:readConfig()})}catch(e){showError(e.message)}};
$('mode').onchange=()=>{$('human-field').hidden=$('mode').value==='human';};
$('advice-panel').ontoggle=()=>{if($('advice-panel').open&&adviceAvailable(state?.game)&&state?.advice?.status!=='running')adviceCommand('start');else if(!$('advice-panel').open&&state?.advice?.status==='running')adviceCommand('stop');};
$('advice-start').onclick=()=>adviceCommand('start');$('advice-stop').onclick=()=>adviceCommand('stop');
$('advice-budget').onchange=()=>{if($('advice-panel').open&&adviceAvailable(state?.game))adviceCommand('start');};
(async()=>{try{const o=await api('/api/options');defaults=o.defaults;setConfig(defaults);if(!o.cuda)$('device').value='cpu'}catch(e){showError(e.message)}await poll();if(state?.game){setConfig(state.game.config);$('mode').value=state.game.mode;$('human').value=state.game.human;$('manual-deal').checked=state.game.manual_deal;$('human-field').hidden=state.game.mode==='human'}await refreshModels();setInterval(poll,500)})();
