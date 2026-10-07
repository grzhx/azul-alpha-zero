// Exercise the collector against BGA's DOM IDs and public-data array ordering.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const code = fs.readFileSync(path.join(__dirname, '../bga_web/collector.js'), 'utf8');
const engineToBga = [3, 4, 5, 1, 2, 0];
const tileObjects = counts => counts.flatMap((count, color) =>
  Array.from({length: count}, () => ({type: engineToBga[color]})));
const nodeFor = tiles => ({querySelectorAll: () => tiles.map(tile =>
  ({id: '', className: `tile tile${tile.type}`}))});

function collect(data, history, useDom = true) {
  const count = Object.keys(data.players).length;
  const factoryCount = count === 3 ? 7 : 5;
  const document = {
    getElementById(id) {
      const index = Number(id.replace('factory', ''));
      return useDom && index >= 1 && index <= factoryCount ? nodeFor(data.factories[index]) : null;
    },
    querySelector(selector) {
      if (selector === '.factory-center') return useDom ? nodeFor(data.factories[0]) : null;
      if (selector === '#bag-counter, #bag') return useDom ? {textContent: String(data.remainingTiles)} : null;
      if (selector === '[id^="chatwindowlogs_tablelog_"]') return {querySelectorAll: () => history.map(textContent => ({textContent}))};
      if (useDom) {
        const line = selector.match(/^#player-table-(\d+)-line(\d+)$/);
        if (line) return nodeFor(data.players[line[1]].lines.filter(tile => tile.line === Number(line[2])));
      }
      return null;
    },
    // BGA also assigns .factory to temporary player hands. These must be excluded.
    querySelectorAll: selector => selector === '.factory' ? Array(factoryCount + count).fill(nodeFor([])) : []
  };
  const location = {href: 'https://boardgamearena.com/10/azul?table=925669787', pathname: '/10/azul'};
  const window = {location, document, gameui: {game_name: 'azul', table_id: '925669787', gamedatas: data},
    open: () => ({closed: false, postMessage() {}})};
  window.top = window;
  const context = vm.createContext({window, document, location, URL, CSS: {escape: String},
    fetch: async () => ({ok: true}), setInterval: () => 1, clearInterval() {}});
  vm.runInContext(code, context);
  return JSON.parse(JSON.stringify(window.__azulBgaReadSnapshot()));
}

for (const count of [2, 3]) {
  const n = count === 3 ? 7 : 5;
  const counts = Array.from({length: n}, (_, index) => [0, 0, 0, 0, 0].map((_, color) => color === index % 5 ? 4 : 0));
  const data = {
    players: Object.fromEntries(Array.from({length: count}, (_, index) => [String(index + 1),
      {name: `player${index}`, lines: [], wall: [], score: 0}])),
    playerorder: Array.from({length: count}, (_, index) => index + 1),
    factories: [[{type: 0}], ...counts.map(tileObjects)], remainingTiles: 100 - n * 4,
    gamestate: {name: 'chooseTile', active_player: String(count)}
  };
  for (const useDom of [true, false]) {
    const result = collect(data, ['新的一轮开始了!'], useDom);
    assert.equal(result.players.length, count);
    assert.equal(result.factory_count, n);
    assert.equal(result.current, count - 1);
    assert.equal(result.round, 1);
    assert.equal(result.token_available, true);
    assert.equal(result.bag_colors_exact, true);
    assert.deepEqual(result.sources, [...counts, [0, 0, 0, 0, 0]]);
  }
}

const history = ['新的一轮开始了!', 'kukusumusu 拿取 2 块 青色 瓷砖',
  'kukusumusu 将 2 块 青色 瓷砖放置在第 4 行上', 'jacobcookeefc 拿取 2 块 蓝色 瓷砖',
  'jacobcookeefc 将 2 块 蓝色 瓷砖放置在第 3 行上', 'grzhx 拿取 2 块 黑色 瓷砖和起始玩家标记',
  'grzhx 将 2 块 黑色 瓷砖放置在第 5 行上', 'kukusumusu 拿取 2 块 青色 瓷砖',
  'kukusumusu 将 2 块 青色 瓷砖放置在第 4 行上'];
const sources = [[0, 2, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
  [0, 1, 1, 1, 1], [0, 2, 1, 0, 1], [1, 1, 1, 0, 1], [0, 0, 0, 0, 0], [1, 1, 2, 0, 0]];
const real = {
  players: {
    92785155: {name: 'grzhx', score: 0, wall: [], lines: [{line: 5, type: 1}, {line: 5, type: 1}, {line: 0, type: 0}]},
    95175631: {name: 'kukusumusu', score: 0, wall: [], lines: Array.from({length: 4}, () => ({line: 4, type: 2}))},
    99223478: {name: 'jacobcookeefc', score: 0, wall: [], lines: [{line: 3, type: 3}, {line: 3, type: 3}]}
  },
  playerorder: [92785155, 95175631, 99223478], firstPlayerTokenPlayerId: 92785155,
  factories: [tileObjects(sources[7]), ...sources.slice(0, 7).map(tileObjects)], remainingTiles: 72,
  gamestate: {name: 'chooseTile', active_player: '99223478'}
};
for (const useDom of [true, false]) {
  const result = collect(real, history, useDom);
  assert.deepEqual(result.sources, sources);
  assert.equal(result.token_owner, 0);
  assert.equal(result.next_start, 0);
  assert.equal(result.token_available, false);
  assert.equal(result.bag_colors_exact, true);
  assert.deepEqual(result.bag_colors, [16, 13, 14, 17, 12]);
  assert.deepEqual(result.players.map(player => player.patterns.map(row => row.count)),
    [[0, 0, 0, 0, 2], [0, 0, 0, 4, 0], [0, 0, 2, 0, 0]]);
}
const pending = collect({...real, gamestate: {...real.gamestate, name: 'chooseLine'}}, history);
assert.equal(pending.actionable, false);
// Stale gamedatas owner must not override a marker on the current player's floor.
for (const useDom of [true, false]) {
  const stale = collect({...real, firstPlayerTokenPlayerId: 95175631}, history, useDom);
  assert.equal(stale.token_owner, 0);
  assert.equal(stale.next_start, 0);
  const fullFloor = collect({...real, firstPlayerTokenPlayerId: 95175631,
    players: {...real.players, 92785155: {...real.players[92785155], lines: Array.from({length: 7}, () => ({line: 0, type: 3}))}}}, history, useDom);
  assert.equal(fullFloor.token_owner, 0);
  assert.equal(fullFloor.next_start, 0);
  const newRound = collect({...real, firstPlayerTokenPlayerId: 95175631,
    players: {...real.players, 92785155: {...real.players[92785155], lines: []}},
    factories: [[{type: 0}], ...real.factories.slice(1)]}, [...history, '新的一轮开始了!'], useDom);
  assert.equal(newRound.token_owner, null);
  assert.equal(newRound.next_start, 1);
}
const notTaken = collect({...real, firstPlayerTokenPlayerId: null,
  players: {...real.players, 92785155: {...real.players[92785155], lines: []}},
  factories: [[{type: 0}], ...real.factories.slice(1)]}, history.slice(0, 3));
assert.equal(notTaken.next_start, 1);
assert.throws(() => collect({...real, players: {...real.players, 4: {name: 'fourth'}}, playerorder: [...real.playerorder, 4]}, history), /2\/3/);
console.log('BGA collector regressions passed: 2P/3P, factory IDs, DOM/data fallback, exact history, pending actions.');
