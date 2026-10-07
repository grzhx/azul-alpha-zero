(() => {
  'use strict';
  if (window.top !== window) return;
  if (typeof window.__azulBgaCollectorStop === 'function') window.__azulBgaCollectorStop();
  else if (window.__azulBgaCollector) return;
  window.__azulBgaCollector = true;

  const ENDPOINT = 'http://127.0.0.1:8770';
  const TOKEN = '__TOKEN__';
  const tracker = { round: 1, tableId: '', previousTileCount: null, lastError: '', lastPresence: 0 };
  let relayWindow = window.__azulBgaRelay;
  if (!relayWindow || relayWindow.closed) {
    relayWindow = window.open(ENDPOINT + '/?relay=1', `azul-bga-monitor-${Date.now()}`);
    window.__azulBgaRelay = relayWindow;
  }

  const post = async (route, value) => {
    try {
      const response = await fetch(ENDPOINT + route, {
        method: 'POST', mode: 'cors', cache: 'no-store',
        headers: {'Content-Type': 'application/json', 'X-BGA-Monitor-Token': TOKEN},
        body: JSON.stringify(value)
      });
      if (!response.ok) throw new Error(`monitor HTTP ${response.status}`);
    } catch (error) {
      if (!relayWindow || relayWindow.closed) {
        relayWindow = window.open(ENDPOINT + '/?relay=1', `azul-bga-monitor-${Date.now()}`);
        window.__azulBgaRelay = relayWindow;
      }
      if (!relayWindow) throw error;
      relayWindow.postMessage({type: 'azul-bga-monitor', route, value}, ENDPOINT);
    }
  };

  const integer = (value, fallback = 0) => {
    const result = Number.parseInt(value, 10);
    return Number.isFinite(result) ? result : fallback;
  };

  /* BGA: token, black, white, blue, yellow, red. Engine: blue, yellow, red, black, white, token. */
  const bgaToEngine = type => [5, 3, 4, 0, 1, 2][type] ?? null;

  const colorOf = value => {
    if (value == null) return null;
    if (typeof value === 'object') {
      for (const key of ['color', 'colorId', 'color_id', 'tile', 'tileId', 'type']) {
        if (value[key] != null) return colorOf(value[key]);
      }
      return null;
    }
    const match = String(value).match(/(?:tile)?([0-5])(?:\D|$)/);
    return match ? bgaToEngine(integer(match[1], -1)) : null;
  };

  const validTiles = root => [...(root?.querySelectorAll?.('.tile') || [])].filter(tile => {
    const text = `${tile.id || ''} ${tile.className || ''}`;
    return !text.includes('to-be-destroyed') && !text.includes('destroyed');
  });

  const tileColor = tile => {
    const match = String(tile.className || '').match(/(?:^|\s)tile([0-5])(?:\s|$)/);
    return match ? bgaToEngine(integer(match[1], -1)) : null;
  };

  const countTiles = root => {
    const counts = [0, 0, 0, 0, 0];
    for (const tile of validTiles(root)) {
      const color = tileColor(tile);
      if (color != null && color < 5) counts[color]++;
    }
    return counts;
  };

  const playerOrder = data => {
    const players = data.players || {};
    const order = Array.isArray(data.playerorder) ? data.playerorder.map(String) : Object.keys(players);
    return order.filter(id => players[id] || players[Number(id)]);
  };

  const patternFromData = (player, row) => {
    if (Array.isArray(player.lines) && player.lines.some(tile => typeof tile === 'object' && tile?.line != null)) {
      const tiles = player.lines.filter(tile => integer(tile?.line, -1) === row + 1);
      const colors = tiles.map(colorOf).filter(color => color != null && color < 5);
      return {count: Math.min(row + 1, colors.length), color: colors[0] ?? null};
    }
    const source = Array.isArray(player.lines) ? player.lines[row] : player.lines?.[row + 1] ?? player.lines?.[row];
    if (source == null) return null;
    if (Array.isArray(source)) {
      const colors = source.map(colorOf).filter(color => color != null && color < 5);
      return {count: Math.min(row + 1, colors.length), color: colors[0] ?? null};
    }
    if (typeof source === 'object') {
      const count = integer(source.count ?? source.number ?? source.tiles?.length, 0);
      return {count: Math.min(row + 1, Math.max(0, count)), color: count ? colorOf(source) : null};
    }
    return null;
  };

  const parsePlayer = (doc, raw, id, index) => {
    const board = doc.querySelector(`#player-table-${CSS.escape(String(id))}`) ||
      doc.querySelector(`.player-table-${CSS.escape(String(id))}`) || doc.querySelectorAll('.player-table')[index];
    const patterns = [];
    for (let row = 0; row < 5; row++) {
      const line = doc.querySelector(`#player-table-${CSS.escape(String(id))}-line${row + 1}`) ||
        board?.querySelector(`[id$="-line${row + 1}"]`);
      const colors = validTiles(line).map(tileColor).filter(color => color != null && color < 5);
      patterns.push(line ? {count: Math.min(row + 1, colors.length), color: colors[0] ?? null} :
        (patternFromData(raw, row) || {count: 0, color: null}));
    }

    const wall = Array.from({length: 5}, () => [0, 0, 0, 0, 0]);
    for (let row = 0; row < 5; row++) for (let column = 0; column < 5; column++) {
      const spot = doc.querySelector(`#player-table-${CSS.escape(String(id))}-wall-spot-${row + 1}-${column + 1}`) ||
        board?.querySelector(`[id$="-wall-spot-${row + 1}-${column + 1}"]`);
      wall[row][column] = validTiles(spot).length ? 1 : 0;
    }
    if (!wall.some(row => row.some(Boolean)) && Array.isArray(raw.wall)) {
      if (raw.wall.length === 5 && raw.wall.every(row => Array.isArray(row))) {
        raw.wall.forEach((source, row) => source.slice(0, 5).forEach((cell, col) => wall[row][col] = cell ? 1 : 0));
      } else {
        for (const cell of raw.wall) {
          const row = integer(cell?.row ?? cell?.line, -1), col = integer(cell?.column ?? cell?.col, -1);
          if (row >= 0 && row < 5 && col >= 0 && col < 5) wall[row][col] = 1;
        }
      }
    }

    const floorNode = doc.querySelector(`#player-table-${CSS.escape(String(id))}-line0`) ||
      board?.querySelector('[id$="-line0"]');
    const floor = (floorNode ? validTiles(floorNode).map(tileColor) :
      (Array.isArray(raw.lines) ? raw.lines.filter(tile => integer(tile?.line, -1) === 0).map(colorOf) : []))
      .filter(color => color != null).slice(0, 7);
    const domScore = doc.querySelector(`#player_score_${CSS.escape(String(id))} .player_score_value`) ||
      doc.querySelector(`#player_score_${CSS.escape(String(id))}`) || board?.querySelector('.player_score_value');
    return {
      id: String(id), name: String(raw.name ?? raw.player_name ?? `玩家 ${index + 1}`),
      score: Math.max(0, integer(domScore?.textContent ?? raw.score, 0)), wall, patterns, floor
    };
  };

  const parseSources = (doc, data, factoryCount) => {
    const sources = [];
    for (let index = 1; index <= factoryCount; index++) {
      const factory = doc.getElementById(`factory${index}`);
      const raw = data.factories?.[index];
      const counts = factory ? countTiles(factory) : [0, 0, 0, 0, 0];
      if (!factory && Array.isArray(raw)) raw.forEach(value => {
        const color = colorOf(value);
        if (color != null && color < 5) counts[color]++;
      });
      sources.push(counts);
    }
    const center = doc.querySelector('.factory-center');
    const counts = countTiles(center);
    if (!center && Array.isArray(data.factories?.[0])) data.factories[0].forEach(value => {
      const color = colorOf(value);
      if (color != null && color < 5) counts[color]++;
    });
    sources.push(counts);
    return sources;
  };

  const logs = doc => {
    const timeline = doc.querySelector('[id^="chatwindowlogs_tablelog_"]');
    const nodes = timeline ? [...timeline.querySelectorAll('.game_move_notif')] :
      [...doc.querySelectorAll('#logs .log')].reverse();
    return nodes.map(node => (node.textContent || '').replace(/\s+/g, ' ').trim())
      .filter(line => line && line.length <= 500).slice(-400);
  };

  const bagFromHistory = (history, sources, bagCount, players) => {
    if (bagCount < 0) return null;
    const zero = () => [0, 0, 0, 0, 0];
    const names = {蓝色: 0, 黄色: 1, 红色: 2, 黑色: 3, 青色: 4, 白色: 4,
      blue: 0, yellow: 1, red: 2, black: 3, teal: 4, white: 4};
    const colorPattern = '(蓝色|黄色|红色|黑色|青色|白色|blue|yellow|red|black|teal|white)';
    const replay = players.map(() => ({
      patterns: Array.from({length: 5}, () => ({count: 0, color: null})),
      floorTiles: zero(), floorCount: 0
    }));
    let bag = [20, 20, 20, 20, 20], discard = zero(), drawn = zero();
    const pending = Array(players.length).fill(null);
    const playerOf = line => players.findIndex(player =>
      line === player.name || line.startsWith(`${player.name} `));
    const drawFromBag = draw => {
      const requested = draw.reduce((a, b) => a + b, 0);
      const available = bag.reduce((a, b) => a + b, 0);
      if (!requested) return false;
      if (requested <= available) {
        if (draw.some((count, color) => count > bag[color])) return false;
        bag = bag.map((count, color) => count - draw[color]);
        return true;
      }
      if (draw.some((count, color) => count < bag[color])) return false;
      const afterRefill = draw.map((count, color) => count - bag[color]);
      if (afterRefill.some((count, color) => count > discard[color])) return false;
      bag = discard.map((count, color) => count - afterRefill[color]);
      discard = zero();
      return true;
    };
    const addFloor = (player, color, count) => {
      const kept = Math.min(count, 7 - player.floorCount);
      player.floorTiles[color] += kept;
      player.floorCount += kept;
      discard[color] += count - kept;
    };
    const settle = () => {
      for (const player of replay) {
        player.patterns.forEach((pattern, row) => {
          if (pattern.count === row + 1) {
            discard[pattern.color] += row;
            pattern.count = 0;
            pattern.color = null;
          }
        });
        player.floorTiles.forEach((count, color) => discard[color] += count);
        player.floorTiles = zero();
        player.floorCount = 0;
      }
    };
    for (const line of history) {
      if (/新的一轮开始|new round/i.test(line)) {
        if (!drawn.some(Boolean)) continue; /* BGA logs the first round start before any move. */
        if (!drawFromBag(drawn) || pending.some(Boolean)) return null;
        settle();
        drawn = zero();
        continue;
      }
      const player = playerOf(line);
      if (player < 0) continue;
      const take = line.match(new RegExp(`(?:拿取|takes?)\\s*(\\d+)\\s*(?:块|tiles?)?\\s*${colorPattern}`, 'i'));
      if (take) {
        const count = integer(take[1]), color = names[take[2].toLowerCase()] ?? names[take[2]];
        if (color == null || count < 1 || pending[player]) return null;
        drawn[color] += count;
        pending[player] = {count, color};
        if (/起始玩家标记|first.player/i.test(line) && replay[player].floorCount < 7)
          replay[player].floorCount++;
        continue;
      }
      const placement = line.match(new RegExp(`(?:将|places?)\\s*(\\d+)\\s*(?:块|tiles?)?\\s*${colorPattern}.*?(?:第\\s*(\\d+)\\s*行|line\\s*(\\d+)|地板|floor)`, 'i'));
      if (!placement) continue;
      const move = pending[player];
      const color = names[placement[2].toLowerCase()] ?? names[placement[2]];
      if (!move || move.color !== color) return null;
      const rowText = placement[3] ?? placement[4];
      if (rowText == null) {
        addFloor(replay[player], color, move.count);
      } else {
        const row = integer(rowText, 0) - 1;
        if (row < 0 || row >= 5) return null;
        const pattern = replay[player].patterns[row];
        if (pattern.count && pattern.color !== color) return null;
        const kept = Math.min(move.count, row + 1 - pattern.count);
        pattern.count += kept;
        pattern.color = color;
        addFloor(replay[player], color, move.count - kept);
      }
      pending[player] = null;
    }
    if (pending.some(Boolean)) return null;
    for (let color = 0; color < 5; color++) drawn[color] +=
      sources.reduce((sum, source) => sum + source[color], 0);
    if (!drawFromBag(drawn)) return null;
    const matchesBoard = replay.every((player, index) => {
      const actual = players[index];
      const patternsMatch = player.patterns.every((pattern, row) =>
        pattern.count === actual.patterns[row].count &&
        (pattern.count === 0 || pattern.color === actual.patterns[row].color));
      const actualFloor = zero();
      actual.floor.filter(color => color < 5).forEach(color => actualFloor[color]++);
      return patternsMatch && player.floorCount === actual.floor.length &&
        player.floorTiles.every((count, color) => count === actualFloor[color]);
    });
    return matchesBoard && bag.every(count => count >= 0) &&
      bag.reduce((a, b) => a + b, 0) === bagCount ? bag : null;
  };

  const phaseName = state => {
    const name = String(state?.name || '');
    if (/gameEnd|endGame|gameSetup/i.test(name)) return /setup/i.test(name) ? '准备' : '结束';
    if (/score|wall|endRound/i.test(name)) return '墙面计分';
    if (/choose|take|tile/i.test(name)) return '选砖';
    return name || '进行中';
  };

  const hash = value => {
    const text = JSON.stringify(value);
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
    return (h >>> 0).toString(36);
  };

  const gameContext = () => {
    const frame = document.getElementById('gameIframe');
    try {
      if (frame?.contentWindow?.gameui) return frame.contentWindow;
    } catch (_) {}
    if (window.gameui && /azul/i.test(String(window.gameui.game_name || location.pathname))) return window;
    return null;
  };

  const snapshot = context => {
    const ui = context.gameui, data = ui.gamedatas || {}, doc = context.document;
    if (!/azul/i.test(String(ui.game_name || context.location.pathname)) || !data.players) return null;
    const order = playerOrder(data);
    if (order.length < 2 || order.length > 3) throw new Error('当前不是支持的 2/3 人 Azul 对局');
    const players = order.map((id, index) => parsePlayer(doc, data.players[id] || data.players[Number(id)], id, index));
    const factoryCount = order.length === 3 ? 7 : 5;
    const sources = parseSources(doc, data, factoryCount);
    const history = logs(doc);
    const state = data.gamestate || ui.gamestate || {};
    const tableId = String(ui.table_id || data.table_id || new URL(context.location.href).searchParams.get('table') ||
      new URL(location.href).searchParams.get('table') || '');
    if (tracker.tableId !== tableId) {
      tracker.tableId = tableId;
      tracker.round = 1;
      tracker.previousTileCount = null;
    }
    const total = sources.reduce((sum, source) => sum + source.reduce((a, b) => a + b, 0), 0);
    if (tracker.previousTileCount === 0 && total > 0) tracker.round++;
    tracker.previousTileCount = total;
    const bagCount = integer(doc.querySelector('#bag-counter, #bag')?.textContent ?? data.remainingTiles, -1);
    const inferredRound = bagCount >= 0 && bagCount <= 100 - factoryCount * 4 ?
      Math.floor((100 - bagCount) / (factoryCount * 4)) : 1;
    const historyRound = Math.max(1, history.filter(line => /新的一轮开始|new round/i.test(line)).length);
    tracker.round = Math.max(tracker.round, inferredRound, historyRound,
      integer(data.round ?? data.roundNumber, 1));
    const active = String(state.active_player ?? state.activePlayer ?? '');
    const floorOwners = players.map(player => player.floor.includes(5));
    const dataTokenOwner = order.indexOf(String(data.firstPlayerTokenPlayerId ?? ''));
    const floorOwner = floorOwners.findIndex(Boolean);
    const center = doc.querySelector('.factory-center');
    const tokenAvailable = center ? validTiles(center).some(tile => tileColor(tile) === 5) :
      Array.isArray(data.factories?.[0]) && data.factories[0].some(tile => colorOf(tile) === 5);
    const startLog = history.findLastIndex(line => /新的一轮开始|new round/i.test(line));
    const roundHistory = history.slice(startLog + 1);
    const tokenLog = roundHistory.findLast(line => /起始玩家标记|first[ -]player/i.test(line) &&
      players.some(player => line.startsWith(`${player.name} `)));
    const historyTokenOwner = tokenLog ? players.findIndex(player => tokenLog.startsWith(`${player.name} `)) : -1;
    // gamedatas may keep the previous round's owner after live notifications.
    const tokenOwner = floorOwner >= 0 ? floorOwner : historyTokenOwner >= 0 ? historyTokenOwner : dataTokenOwner;
    let nextStart = tokenAvailable ? dataTokenOwner : tokenOwner;
    if (nextStart < 0) {
      const firstTake = roundHistory.find(line => /拿取|takes?\s/i.test(line));
      nextStart = firstTake ? players.findIndex(player => firstTake.startsWith(`${player.name} `)) : order.indexOf(active);
    }
    if (nextStart < 0) nextStart = Math.max(0, order.indexOf(active));
    const terminal = /gameEnd|endGame/i.test(String(state.name || '')) || String(state.type || '') === 'gameEnd';
    const bagColors = bagFromHistory(history, sources, bagCount, players);
    const core = {sources, players, active, state: state.name, round: tracker.round, bagCount,
      tokenAvailable, tokenOwner, nextStart, bagColors};
    return {
      table_id: tableId, table_url: `https://boardgamearena.com/tableview?table=${encodeURIComponent(tableId)}`,
      game_name: 'Azul', player_count: players.length, factory_count: sources.length - 1,
      round: tracker.round, current: Math.max(0, order.indexOf(active)), players, sources,
      token_available: tokenAvailable, token_owner: tokenAvailable ? null :
        (tokenOwner >= 0 ? tokenOwner : floorOwner >= 0 ? floorOwner : null),
      next_start: nextStart,
      bag_count: bagCount >= 0 ? bagCount : null,
      bag_colors: bagColors, bag_colors_exact: bagColors !== null,
      phase: phaseName(state), actionable: state.name === 'chooseTile', terminal, log: history,
      raw_state_id: `bga:${tableId}:${data.notifications?.move_nbr || '0'}:${hash(core)}`
    };
  };

  const tick = async () => {
    try {
      const context = gameContext();
      const value = context ? snapshot(context) : null;
      if (value) await post('/api/ingest', value);
      else if (Date.now() - tracker.lastPresence > 1500) {
        tracker.lastPresence = Date.now();
        await post('/api/presence', {location: location.href});
      }
      tracker.lastError = '';
    } catch (error) {
      const message = String(error?.message || error);
      if (message !== tracker.lastError) {
        tracker.lastError = message;
        try { await post('/api/error', {error: message}); } catch (_) {}
      }
    }
  };

  window.__azulBgaReadSnapshot = () => {
    const context = gameContext();
    return context ? snapshot(context) : null;
  };
  const timer = setInterval(tick, 1000);
  window.__azulBgaCollectorStop = () => {
    clearInterval(timer);
    window.__azulBgaCollector = false;
    delete window.__azulBgaReadSnapshot;
    delete window.__azulBgaCollectorStop;
  };
  tick();
})();
