"""Local, read-only BGA Azul mirror. Ingests normalized public table snapshots."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import torch
from azul import Batch
from azul3 import Batch3
from . import ENGINE_VERSION, OPTIMIZED_VERSION
from .model import PolicyValue
from .model3 import PolicyValue3
from .optimized_search import FrozenActor, PipelineSearch, Tuning
from .search import SearchConfig
from .search3 import Search3, SearchConfig3, Evaluator3
from .selfplay import split_seed
from .web_game import catalog, default_config, describe_action

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "bga_web"
RUNS = ROOT / "runs"
THREE_PLAYER_ENGINE_VERSIONS = {"azul3-v1", "azul3-stable-v2"}
ADVICE_BUDGETS = {
    "quick": (1, 64), "standard": (4, 64), "deep": (8, 128),
    "maximum": (16, 256), "infinite": (16, 256),
}


def monitor_catalog(root):
    entries = catalog(root)
    present = {item["id"] for item in entries}
    for path in root.rglob("actor-3p-*.pt"):
        if not path.stem.removeprefix("actor-3p-").isdigit():
            continue
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            checkpoint_id = path.relative_to(root).as_posix()
            if checkpoint_id in present:
                continue
            stat = path.stat()
            entries.append(dict(id=checkpoint_id, run=path.parent.relative_to(root).as_posix(),
                                name=path.name, bytes=stat.st_size, modified=stat.st_mtime,
                                time=datetime.fromtimestamp(stat.st_mtime).strftime("%m-%d %H:%M"), kind="actor"))
        except OSError:
            continue
    return sorted(entries, key=lambda item: (item["kind"] == "actor", item["modified"]), reverse=True)


def empty_state():
    return dict(status="waiting", message="等待 BGA Azul 对局", connected=False,
                table_id=None, table_url=None, updated_at=None, revision=0, game=None,
                source="BGA 公开对局数据", error=None)


def _allowed_origin(value):
    if not value:
        return None
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and (host == "boardgamearena.com" or host.endswith(".boardgamearena.com")):
        return value
    return None


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} 必须是 {low}–{high} 的整数")
    return value


def validate_snapshot(raw):
    if not isinstance(raw, dict):
        raise ValueError("快照必须是 JSON 对象")
    required = {"table_id", "table_url", "round", "current", "players", "sources"}
    missing = required - set(raw)
    if missing:
        raise ValueError("快照缺少字段：" + ", ".join(sorted(missing)))
    players = raw["players"]
    if not isinstance(players, list) or len(players) not in (2, 3):
        raise ValueError("监控页只接受双人或三人 Azul 对局")
    player_count = len(players)
    normalized_players = []
    for index, player in enumerate(players):
        if not isinstance(player, dict):
            raise ValueError("玩家数据格式错误")
        wall = player.get("wall")
        if not isinstance(wall, list) or len(wall) != 5 or any(
                not isinstance(row, list) or len(row) != 5 or any(cell not in (0, 1) for cell in row)
                for row in wall):
            raise ValueError(f"玩家 {index + 1} 墙面必须是 5×5 的 0/1 数组")
        patterns = player.get("patterns")
        if not isinstance(patterns, list) or len(patterns) != 5:
            raise ValueError(f"玩家 {index + 1} 图案行必须有 5 行")
        normalized_patterns = []
        for row, pattern in enumerate(patterns):
            if not isinstance(pattern, dict):
                raise ValueError("图案行格式错误")
            count = _integer(pattern.get("count"), "图案行数量", 0, row + 1)
            color = pattern.get("color")
            if count == 0:
                color = None
            elif type(color) is not int or not 0 <= color < 5:
                raise ValueError("非空图案行必须有合法颜色")
            normalized_patterns.append(dict(count=count, color=color, capacity=row + 1))
        floor = player.get("floor", [])
        if not isinstance(floor, list) or len(floor) > 7 or any(
                type(tile) is not int or not 0 <= tile <= 5 for tile in floor):
            raise ValueError("地板必须是最多 7 个颜色/标记编号；5 表示起始标记")
        normalized_players.append(dict(id=str(player.get("id", index)),
                                       name=str(player.get("name", f"玩家 {index + 1}"))[:80],
                                       score=_integer(player.get("score", 0), "分数", 0, 1000),
                                       wall=wall, patterns=normalized_patterns, floor=floor,
                                       completed_rows=sum(all(row) for row in wall)))
    sources = raw["sources"]
    expected_factories = 7 if player_count == 3 else 5
    if not isinstance(sources, list) or len(sources) != expected_factories + 1 or any(
            not isinstance(source, list) or len(source) != 5 or any(
                type(count) is not int or not 0 <= count <= 20 for count in source)
            for source in sources):
        raise ValueError(f"{player_count} 人对局来源必须是 {expected_factories + 1}×5 的颜色数量数组")
    for source in sources[:expected_factories]:
        if sum(source) > 4:
            raise ValueError("工厂盘不能超过 4 块砖")
    log = raw.get("log", [])
    if not isinstance(log, list):
        raise ValueError("行动记录必须是数组")
    token_owner = raw.get("token_owner")
    if token_owner is not None:
        token_owner = _integer(token_owner, "先手标记玩家", 0, player_count - 1)
    next_start = _integer(raw.get("next_start", token_owner if token_owner is not None else raw["current"]),
                          "下轮先手玩家", 0, player_count - 1)
    token_available = bool(raw.get("token_available", False))
    floor_owners = [index for index, player in enumerate(normalized_players) if 5 in player["floor"]]
    if sum(player["floor"].count(5) for player in normalized_players) > 1:
        raise ValueError("局面中存在多个起始玩家标记，等待 BGA 动画完成")
    if floor_owners:
        # BGA gamedatas.firstPlayerTokenPlayerId can still refer to the previous round.
        # A marker visible on a player's floor is the authoritative current owner.
        token_available = False
        token_owner = next_start = floor_owners[0]
    elif token_available:
        token_owner = None
    else:
        # With a full floor, taking the marker may leave no visible marker slot.
        # Use only this round's marker notification, never a previous round's owner.
        for line in reversed(log):
            text = str(line)
            if re.search(r"新的一轮开始|new round", text, re.I):
                break
            if not re.search(r"起始玩家标记|first[ -]player", text, re.I):
                continue
            owner = next((index for index, player in enumerate(normalized_players)
                          if text.startswith(player["name"] + " ")), None)
            if owner is not None:
                token_owner = next_start = owner
                break
    bag_count = raw.get("bag_count")
    if bag_count is not None:
        bag_count = _integer(bag_count, "牌袋余量", 0, 100)
    bag_colors = raw.get("bag_colors")
    if bag_colors is not None:
        if (not isinstance(bag_colors, list) or len(bag_colors) != 5 or
                any(type(count) is not int or not 0 <= count <= 20 for count in bag_colors)):
            raise ValueError("牌袋颜色必须是 5 个 0–20 的整数")
        if bag_count is None or sum(bag_colors) != bag_count:
            raise ValueError("牌袋颜色合计必须等于牌袋余量")
    return dict(table_id=str(raw["table_id"])[:40], table_url=str(raw["table_url"])[:500],
                game_name=str(raw.get("game_name", "Azul"))[:80],
                round=_integer(raw["round"], "轮数", 0, 1000),
                current=_integer(raw["current"], "当前玩家", 0, player_count - 1),
                players=normalized_players, sources=sources, player_count=player_count,
                factory_count=expected_factories,
                token_available=token_available,
                token_owner=token_owner, next_start=next_start, bag_count=bag_count, bag_colors=bag_colors,
                bag_colors_exact=bool(raw.get("bag_colors_exact", False) and bag_colors is not None),
                phase=str(raw.get("phase", "draft"))[:40],
                actionable=bool(raw.get("actionable", True)),
                terminal=bool(raw.get("terminal", False)),
                log=[str(item)[:500] for item in log[-400:]],
                raw_state_id=str(raw.get("raw_state_id", ""))[:120])


class AdviceManager:
    """Asynchronous imperfect-information analysis of the latest public position."""
    def __init__(self, checkpoints=RUNS):
        self.checkpoints = checkpoints.resolve()
        entries = monitor_catalog(self.checkpoints)
        preferred = next((item for item in entries
                          if item["id"] == "optimized/latest.pt"), entries[0] if entries else None)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bga-advice")
        self.cancel = threading.Event()
        self.generation = 0
        self.enabled = True
        self.budget = "maximum"
        self.checkpoint = preferred["id"] if preferred else None
        self.position_id = None
        self.cached = None
        self.evaluator3_cache = None
        self.metadata_lock = threading.Lock()
        self.metadata_cache = {}
        self.selections = {2: self.checkpoint, 3: None}
        self.view = dict(status="waiting", enabled=self.enabled, budget=self.budget,
                         checkpoint=self.checkpoint, simulations=0, suggestions=[],
                         message="等待可分析的 BGA 局面")

    def state(self):
        with self.lock:
            return dict(self.view)

    def checkpoint_players(self, checkpoint_id):
        path = (self.checkpoints / checkpoint_id).resolve()
        if not path.is_relative_to(self.checkpoints) or not path.is_file():
            raise ValueError("checkpoint 不存在")
        stat = path.stat()
        signature = (str(path), stat.st_mtime_ns, stat.st_size)
        with self.metadata_lock:
            if signature not in self.metadata_cache:
                # Metadata inspection maps tensor storage without copying model/optimizer weights.
                payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                version = payload.get("engine_version")
                self.metadata_cache[signature] = 3 if version in THREE_PLAYER_ENGINE_VERSIONS else (
                    2 if version in (ENGINE_VERSION, OPTIMIZED_VERSION) else 0)
            return self.metadata_cache[signature]

    def models(self):
        entries = monitor_catalog(self.checkpoints)
        for item in entries:
            try:
                item["player_count"] = self.checkpoint_players(item["id"])
            except (OSError, RuntimeError, ValueError):
                item["player_count"] = 0
        return [item for item in entries if item["player_count"] in (2, 3)]

    def select_for_game(self, game):
        count = len(game["players"])
        selected = self.selections[count]
        if selected:
            try:
                if self.checkpoint_players(selected) == count:
                    self.checkpoint = selected
                    return
            except (OSError, ValueError):
                pass
        entries = [item for item in self.models() if item["player_count"] == count]
        preferred_id = "three_player_maxn_longrun/latest.pt" if count == 3 else "optimized/latest.pt"
        preferred = next((item for item in entries if item["id"] == preferred_id), entries[0] if entries else None)
        self.checkpoint = preferred["id"] if preferred else None
        self.selections[count] = self.checkpoint

    def _cancel_locked(self):
        self.cancel.set()
        self.cancel = threading.Event()
        self.generation += 1
        return self.generation, self.cancel

    def update(self, game, force=False):
        with self.lock:
            position_id = game.get("raw_state_id") if game else None
            if not force and position_id == self.position_id:
                return
            self.position_id = position_id
            generation, cancel = self._cancel_locked()
            if game:
                self.select_for_game(game)
            if not self.enabled:
                self.view.update(status="stopped", enabled=False, budget=self.budget,
                                 checkpoint=self.checkpoint, simulations=0, suggestions=[],
                                 message="实时建议已停止")
                return
            if game and game.get("terminal"):
                ranks = [(player["score"], player["completed_rows"]) for player in game["players"]]
                current = game["current"]
                if len(ranks) == 3:
                    expected_ranks = [1 + sum(other > rank for other in ranks) +
                                      (sum(other == rank for other in ranks) - 1) / 2 for rank in ranks]
                    situation = dict(kind="rank_utility", expected_ranks=expected_ranks,
                                     utilities=[2 - rank for rank in expected_ranks],
                                     winners=[i for i, rank in enumerate(ranks) if rank == max(ranks)],
                                     exact=True, label="终局确定名次（同分比较完整横行）")
                else:
                    situation = dict(win=0.0, draw=0.0, loss=0.0, exact=True, label="终局确定结果")
                    situation["draw" if ranks[0] == ranks[1] else
                              "win" if ranks[current] > ranks[current ^ 1] else "loss"] = 1.0
                self.view.update(status="waiting", enabled=True, budget=self.budget,
                                 checkpoint=self.checkpoint, simulations=0, suggestions=[],
                                 situation=situation, message="对局已结束")
                return
            if not game or not game.get("actionable", True) or not any(map(any, game.get("sources", []))):
                self.view.update(status="waiting", enabled=True, budget=self.budget,
                                 checkpoint=self.checkpoint, simulations=0, suggestions=[],
                                 situation=None, message="等待走棋局面")
                return
            if not self.checkpoint:
                self.view.update(status="error", enabled=True, budget=self.budget,
                                 checkpoint=None, simulations=0, suggestions=[],
                                 message="没有可用 checkpoint")
                return
            snapshot = json.loads(json.dumps(game, ensure_ascii=False))
            self.view = dict(status="running", enabled=True, budget=self.budget,
                             checkpoint=self.checkpoint, simulations=0, suggestions=[],
                             perspective=game["current"], message="正在分析当前行动方…",
                             bag_mode="精确历史重建" if game.get("bag_colors_exact") else "历史约束估计")
            self.executor.submit(self._work, generation, cancel, snapshot,
                                 self.checkpoint, self.budget)

    def command(self, operation, data, game):
        with self.lock:
            if operation == "stop":
                self.enabled = False
                self._cancel_locked()
                self.view.update(status="stopped", enabled=False, simulations=0,
                                 suggestions=[], message="实时建议已停止")
                return
            if operation not in ("start", "configure"):
                raise ValueError("未知建议操作")
            checkpoint = str(data.get("checkpoint", self.checkpoint or ""))
            valid = {item["id"] for item in monitor_catalog(self.checkpoints)}
            if checkpoint not in valid:
                raise ValueError("无效或已被训练清理的 checkpoint")
            count = self.checkpoint_players(checkpoint)
            if not count or (game and count != len(game["players"])):
                raise ValueError("所选模型的玩家数与当前对局不匹配")
            budget = data.get("budget", self.budget)
            if budget not in ADVICE_BUDGETS:
                raise ValueError("无效的建议搜索预算")
            self.checkpoint, self.budget, self.enabled = checkpoint, budget, True
            self.selections[count] = checkpoint
        self.update(game, force=True)

    def _load_actor(self, checkpoint_id):
        path = (self.checkpoints / checkpoint_id).resolve()
        if not path.is_relative_to(self.checkpoints) or not path.is_file():
            raise ValueError("checkpoint 不存在")
        signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if self.cached and self.cached[0] == signature:
            return self.cached[1:]
        with path.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        if payload.get("engine_version") not in (ENGINE_VERSION, OPTIMIZED_VERSION):
            raise ValueError("checkpoint 引擎版本不受支持")
        model = PolicyValue(**payload["architecture"])
        model.load_state_dict(payload["model"])
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actor = FrozenActor(model, device, bf16=False)
        result = (actor, payload, device)
        self.evaluator3_cache = None
        self.cached = (signature, *result)
        return result

    def _load_actor3(self, checkpoint_id):
        path = (self.checkpoints / checkpoint_id).resolve()
        if not path.is_relative_to(self.checkpoints) or not path.is_file():
            raise ValueError("三人 checkpoint 不存在")
        signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size, "3p")
        if self.cached and self.cached[0] == signature:
            return self.cached[1:]
        with path.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        if payload.get("engine_version") not in THREE_PLAYER_ENGINE_VERSIONS:
            raise ValueError("所选 checkpoint 不是受支持的三人模型")
        model = PolicyValue3(**{k: payload["architecture"][k] for k in ("width", "blocks")})
        model.load_state_dict(payload["model"]); model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = (model.to(device), payload, device)
        self.evaluator3_cache = None
        self.cached = (signature, *result)
        return result

    @staticmethod
    def _visible_arrays(game):
        sources = np.asarray(game["sources"], dtype=np.uint8)
        walls, scores, pattern_colors, pattern_counts = [], [], [], []
        floor_tiles, floor_counts = [], []
        known = sources.astype(np.int32).sum(axis=0)
        for player in game["players"]:
            wall_bits = 0
            for row, cells in enumerate(player["wall"]):
                for column, occupied in enumerate(cells):
                    if occupied:
                        wall_bits |= 1 << (row * 5 + column)
                        known[(column + 5 - row) % 5] += 1
            walls.append(wall_bits)
            scores.append(player["score"])
            colored_floor = [0] * 5
            for color in player["floor"]:
                if color < 5:
                    colored_floor[color] += 1
                    known[color] += 1
            floor_tiles.append(colored_floor)
            floor_counts.append(len(player["floor"]))
            for pattern in player["patterns"]:
                count = pattern["count"]
                color = pattern["color"] if count else 5
                pattern_colors.append(color)
                pattern_counts.append(count)
                if count:
                    known[color] += count
        unseen = 20 - known
        if np.any(unseen < 0):
            raise ValueError("BGA 公开局面的颜色数量不守恒")
        owner = game.get("token_owner")
        return dict(sources=sources.tolist(), walls=walls, scores=scores,
                    pattern_colors=pattern_colors, pattern_counts=pattern_counts,
                    floor_tiles=floor_tiles, floor_counts=floor_counts,
                    unseen=unseen, next_start=game.get("next_start", game["current"] if owner is None else owner))

    @staticmethod
    def _bag_split(game, visible, rng):
        exact = game.get("bag_colors") if game.get("bag_colors_exact") else None
        unseen = visible["unseen"]
        total = int(unseen.sum())
        bag_size = game.get("bag_count")
        if bag_size is None:
            deal_size = (7 if len(game["players"]) == 3 else 5) * 4
            bag_size = min(total, max(0, 100 - game.get("round", 1) * deal_size))
        bag_size = min(int(bag_size), total)
        if exact is not None:
            bag = np.asarray(exact, dtype=np.int32)
            if np.any(bag > unseen) or int(bag.sum()) != bag_size:
                raise ValueError("历史重建的袋内颜色与公开局面不一致")
        else:
            pool = np.repeat(np.arange(5, dtype=np.int8), unseen)
            rng.shuffle(pool)
            bag = np.bincount(pool[:bag_size], minlength=5)
        return bag.tolist(), (unseen - bag).tolist()

    def _work(self, generation, cancel, game, checkpoint_id, budget):
        if len(game.get("players", [])) == 3:
            return self._work3(generation, cancel, game, checkpoint_id, budget)
        env = search = None
        total = 0
        try:
            time.sleep(0.2)  # Avoid analyzing BGA's intermediate animation DOM.
            if cancel.is_set():
                return
            actor, payload, device = self._load_actor(checkpoint_id)
            roots, simulations = ADVICE_BUDGETS[budget]
            visible = self._visible_arrays(game)
            native = ROOT / "build-bga-3p" / "azul.dll"
            if not native.exists():
                native = ROOT / "build-bga" / "azul.dll"
            env = Batch(roots, threads=min(8, roots), seed=0,
                        library=native if native.exists() else None)
            base = default_config()
            base.update({key: value for key, value in payload.get("search", {}).items() if key in base})
            tuning_data = payload.get("tuning", {})
            tuning = Tuning(q_mode=int(tuning_data.get("q_mode", base["q_mode"])),
                            q_floor=float(tuning_data.get("q_floor", base["q_floor"])),
                            chance_coefficient=float(tuning_data.get("chance_coefficient", base["chance_coefficient"])),
                            chance_exponent=float(tuning_data.get("chance_exponent", base["chance_exponent"])),
                            chance_sensitivity=float(tuning_data.get("chance_sensitivity", base["chance_sensitivity"])),
                            afterstate_prior=float(tuning_data.get("afterstate_prior", base["afterstate_prior"]))
                            if payload.get("iteration", 0) and actor.model.kind == "equivariant" else 0.0,
                            cache_capacity=int(tuning_data.get("cache_capacity", base["cache_capacity"])),
                            subtree_reuse=0, symmetric_cache=int(actor.model.kind == "equivariant"))
            config = SearchConfig(simulations=simulations,
                                  candidates=min(int(base["candidates"]), simulations),
                                  max_depth=int(base["max_depth"]),
                                  chance_initial=int(base["chance_initial"]),
                                  chance_cap=int(base["chance_cap"]), gumbel_scale=0,
                                  value_scale=float(base["value_scale"]),
                                  maxvisit_init=float(base["maxvisit_init"]))
            search = PipelineSearch(env, actor, config, tuning, threads=min(8, roots),
                                    queues=1, graphs=device == "cuda")
            accumulated = np.zeros(180, dtype=np.float64)
            accumulated_wdl = np.zeros(3, dtype=np.float64)
            waves = 0
            seed = int.from_bytes(hashlib.blake2b(game["raw_state_id"].encode(), digest_size=8).digest(), "little")
            rng = np.random.default_rng(seed)
            while not cancel.is_set():
                for index in range(roots):
                    bag, discard = self._bag_split(game, visible, rng)
                    env.import_public(index, sources=visible["sources"], walls=visible["walls"],
                                      scores=visible["scores"], pattern_colors=visible["pattern_colors"],
                                      pattern_counts=visible["pattern_counts"], floor_tiles=visible["floor_tiles"],
                                      floor_counts=visible["floor_counts"], bag=bag, discard=discard,
                                      round=game["round"], current=game["current"],
                                      next_start=visible["next_start"],
                                      token_available=int(game["token_available"]))
                env.observe()
                observations = np.ctypeslib.as_array(env.observations).reshape(roots, -1).copy()
                tensor = torch.as_tensor(observations, device=actor.device, dtype=actor.dtype)
                with torch.inference_mode():
                    _, wdl_logits = actor.model(tensor)
                    wave_wdl = wdl_logits.float().softmax(-1).mean(0).cpu().numpy()
                _, policies, values = search.run(split_seed(seed, waves))
                accumulated += policies.mean(axis=0)
                accumulated_wdl += wave_wdl
                waves += 1
                total += roots * simulations
                policy = accumulated / waves
                ranked = [int(action) for action in np.argsort(policy)[::-1] if policy[action] > 0][:5]
                suggestions = []
                for rank, action in enumerate(ranked, 1):
                    item = describe_action(action, game["sources"])
                    item.update(rank=rank, confidence=float(policy[action]))
                    suggestions.append(item)
                win, draw, loss = (accumulated_wdl / waves).tolist()
                situation = dict(win=win, draw=draw, loss=loss,
                                 exact=False, label="模型 W/D/L 估计")
                with self.lock:
                    if generation != self.generation or cancel.is_set():
                        return
                    running = budget == "infinite"
                    self.view = dict(status="running" if running else "ready", enabled=True,
                                     budget=budget, checkpoint=checkpoint_id, simulations=total,
                                     suggestions=suggestions, perspective=game["current"], device=device.upper(),
                                     checkpoint_iteration=payload.get("iteration", 0),
                                     value=float(np.asarray(values).mean()),
                                     situation=situation,
                                     bag_mode="精确历史重建" if game.get("bag_colors_exact") else f"{roots} 组历史约束估计",
                                     message="持续搜索中，可随时停止" if running else "分析完成")
                if budget != "infinite":
                    return
        except Exception as exc:
            with self.lock:
                if generation == self.generation:
                    self.view.update(status="error", simulations=total, suggestions=[], message=str(exc))
        finally:
            if search:
                search.close()
            if env:
                env.close()

    def _work3(self, generation, cancel, game, checkpoint_id, budget):
        env = search = None; total = 0
        try:
            time.sleep(0.2)
            if cancel.is_set(): return
            model, payload, device = self._load_actor3(checkpoint_id)
            roots, simulations = ADVICE_BUDGETS[budget]
            visible = self._visible_arrays(game)
            native = ROOT / "build-bga-3p" / "azul.dll"
            env = Batch3(roots, threads=min(8, roots), seed=0, library=native if native.exists() else None)
            search_args = payload.get("search", payload.get("args", {}))
            if not isinstance(search_args, dict):
                search_args = vars(search_args)
            search = Search3(env, SearchConfig3(simulations=simulations,
                                                 candidates=min(int(search_args.get("candidates", 32)), simulations),
                                                 max_depth=int(search_args.get("max_depth", 128)),
                                                 chance_initial=int(search_args.get("chance_initial", 4)),
                                                 chance_cap=int(search_args.get("chance_cap", 128)),
                                                 value_scale=float(search_args.get("value_scale", .1)),
                                                 maxvisit_init=float(search_args.get("maxvisit_init", 50)),
                                                 gumbel_scale=0), min(8, roots))
            evaluator_key = (id(model), roots, device)
            if self.evaluator3_cache is None or self.evaluator3_cache[0] != evaluator_key:
                self.evaluator3_cache = (evaluator_key, Evaluator3(model, roots, device, use_graph=device == "cuda"))
            evaluator = self.evaluator3_cache[1]
            observations = env.numpy_views()["observations"]
            accumulated = np.zeros(240, dtype=np.float64); utilities_sum = np.zeros(3, dtype=np.float64)
            waves = 0; seed = int.from_bytes(hashlib.blake2b(game["raw_state_id"].encode(), digest_size=8).digest(), "little")
            rng = np.random.default_rng(seed)
            while not cancel.is_set():
                for index in range(roots):
                    bag, discard = self._bag_split(game, visible, rng)
                    env.import_public(index, sources=visible["sources"],
                                     walls=visible["walls"], scores=visible["scores"],
                                     pattern_colors=visible["pattern_colors"], pattern_counts=visible["pattern_counts"],
                                     floor_tiles=visible["floor_tiles"],
                                     floor_counts=visible["floor_counts"], bag=bag, discard=discard,
                                     round=game["round"], current=game["current"], next_start=visible["next_start"],
                                     token_available=int(game["token_available"]))
                env.observe()
                np.copyto(evaluator.observations, observations)
                _, utilities = evaluator.evaluate()
                utilities_sum += utilities.mean(0)
                result = search.run(evaluator, split_seed(seed, waves), cancel=cancel)
                if result is None:
                    return
                _, policies, values = result
                accumulated += policies.mean(axis=0); waves += 1; total += roots * simulations
                policy = accumulated / waves
                ranked = [int(action) for action in np.argsort(policy)[::-1] if policy[action] > 0][:5]
                suggestions = []
                for rank, action in enumerate(ranked, 1):
                    item = describe_action(action, game["sources"]); item.update(rank=rank, confidence=float(policy[action])); suggestions.append(item)
                # Utility slots rotate with the actor; player boards remain in fixed seat order.
                utilities = np.roll(utilities_sum / waves, game["current"])
                situation = dict(kind="rank_utility", utilities=utilities.tolist(),
                                 expected_ranks=(2 - utilities).tolist(), exact=False,
                                 label="三人模型预期名次（1 为最好，3 为最差）")
                with self.lock:
                    if generation != self.generation or cancel.is_set(): return
                    self.view = dict(status="running" if budget == "infinite" else "ready", enabled=True,
                                     budget=budget, checkpoint=checkpoint_id, simulations=total, suggestions=suggestions,
                                     perspective=game["current"], device=str(device).upper(), checkpoint_iteration=payload.get("iteration", 0),
                                     value=float(values.mean()), situation=situation,
                                     bag_mode="精确历史重建" if game.get("bag_colors_exact") else f"{roots} 组历史约束估计",
                                     message="持续搜索中，可随时停止" if budget == "infinite" else "分析完成")
                if budget != "infinite": return
        except Exception as exc:
            with self.lock:
                if generation == self.generation: self.view.update(status="error", simulations=total, suggestions=[], message=str(exc))
        finally:
            if search: search.close()
            if env: env.close()

    def close(self):
        with self.lock:
            self.cancel.set()
            self.generation += 1
        self.executor.shutdown(wait=True)


class Monitor:
    def __init__(self, checkpoints=RUNS):
        self.lock = threading.Lock()
        self.data = empty_state()
        self.token = secrets.token_urlsafe(24)
        self.advice = AdviceManager(checkpoints)

    def state(self):
        with self.lock:
            data = dict(self.data)
        if data["updated_at"] is not None:
            age = max(0.0, time.time() - data["updated_at"])
            data["age_seconds"] = age
            if age > 10 and data["connected"]:
                data["connected"] = False
                data["message"] = "BGA 数据超过 10 秒未更新"
        data["advice"] = self.advice.state()
        return data

    def game(self):
        with self.lock:
            return self.data.get("game")

    def ingest(self, snapshot):
        game = validate_snapshot(snapshot)
        now = time.time()
        with self.lock:
            duplicate = (self.data.get("game") or {}).get("raw_state_id") == game["raw_state_id"]
            self.data.update(status="live" if not game["terminal"] else "finished",
                             message="实时同步中" if not game["terminal"] else "BGA 对局已结束",
                             connected=True, table_id=game["table_id"], table_url=game["table_url"],
                             updated_at=now, game=game, error=None,
                             revision=self.data["revision"] + (0 if duplicate else 1))
        if not duplicate:
            self.advice.update(game)

    def presence(self, payload):
        location = str(payload.get("location", ""))[:500]
        now = time.time()
        with self.lock:
            had_game = self.data.get("game") is not None
            self.data.update(status="watching", connected=True, game=None, table_id=None,
                             table_url=None, updated_at=now, error=None,
                             message="已连接 BGA，等待进入 Azul 对局",
                             bga_location=location,
                             revision=self.data["revision"] + (1 if had_game else 0))
        if had_game:
            self.advice.update(None)

    def failure(self, message):
        with self.lock:
            self.data.update(status="error", connected=False, error=str(message)[:500],
                             message="BGA 数据解析失败", revision=self.data["revision"] + 1)

    def close(self):
        self.advice.close()


def make_server(monitor, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            origin = _allowed_origin(self.headers.get("Origin"))
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                if self.headers.get("Access-Control-Request-Private-Network") == "true":
                    self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, status, value):
            self.send_bytes(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def local(self):
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.local():
                return self.json(403, {"error": "仅允许本机访问"})
            route = urlsplit(self.path).path
            if route == "/api/state":
                return self.json(200, monitor.state())
            if route == "/api/checkpoints":
                return self.json(200, {"checkpoints": monitor.advice.models()})
            if route == "/api/options":
                return self.json(200, {"cuda": torch.cuda.is_available(),
                                       "default_checkpoint": monitor.advice.checkpoint,
                                       "default_budget": monitor.advice.budget})
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8"),
                     "/advice.css": ("advice.css", "text/css; charset=utf-8"),
                     "/collector.js": ("collector.js", "text/javascript; charset=utf-8")}
            if route not in files:
                return self.json(404, {"error": "Not found"})
            name, content_type = files[route]
            body = (WEB / name).read_bytes()
            if name == "index.html":
                body = body.replace(b"__TOKEN__", monitor.token.encode())
            elif name == "collector.js":
                body = body.replace(b"__TOKEN__", monitor.token.encode())
            self.send_bytes(200, body, content_type)

        def do_OPTIONS(self):
            if not self.local() or not _allowed_origin(self.headers.get("Origin")):
                return self.json(403, {"error": "仅允许 BGA 页面连接"})
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-BGA-Monitor-Token")
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_POST(self):
            if not self.local() or self.headers.get("X-BGA-Monitor-Token") != monitor.token:
                return self.json(403, {"error": "无效的本机监控令牌"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 262144:
                    raise ValueError("快照大小不合法")
                payload = json.loads(self.rfile.read(length))
                route = urlsplit(self.path).path
                if route.startswith("/api/advice/"):
                    operation = route.rsplit("/", 1)[-1]
                    monitor.advice.command(operation, payload, monitor.game())
                elif route == "/api/ingest":
                    monitor.ingest(payload)
                elif route == "/api/presence":
                    monitor.presence(payload)
                elif route == "/api/error":
                    monitor.failure(payload.get("error", "未知错误"))
                else:
                    return self.json(404, {"error": "Not found"})
                self.json(202, monitor.state())
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.json(400, {"error": str(exc)})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="BGA Azul 公开对局只读监控面板")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--checkpoints", type=Path, default=RUNS)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    monitor = Monitor(args.checkpoints)
    server = make_server(monitor, args.port)
    print(f"BGA Azul 监控：http://127.0.0.1:{args.port}", flush=True)
    print(f"采集令牌：{monitor.token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        monitor.close()


if __name__ == "__main__":
    main()
