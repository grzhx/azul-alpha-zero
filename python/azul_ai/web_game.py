"""Local/ngrok human vs checkpoint UI. Separate process, read-only model access."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import torch
from azul import Batch
from . import ENGINE_VERSION, OPTIMIZED_VERSION
from .model import PolicyValue
from .optimized_search import FrozenActor, PipelineSearch, Tuning
from .search import SearchConfig
from .selfplay import split_seed

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
COLORS = ["蓝", "黄", "红", "黑", "白"]


def default_config():
    return dict(device="cuda" if torch.cuda.is_available() else "cpu", simulations=512,
                candidates=32, chance_initial=4, chance_cap=128, max_depth=128,
                q_mode=1, q_floor=0.25, value_scale=0.1, maxvisit_init=50.0,
                gumbel_scale=0.0, chance_coefficient=2.0, chance_exponent=0.5,
                chance_sensitivity=1.0, afterstate_prior=0.25, cache_capacity=4096,
                subtree_reuse=True, bf16=False, cuda_graph=True)


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("AI 配置须为对象")
    config = default_config()
    if set(raw)-set(config):
        raise ValueError("存在不支持的 AI 配置项")
    config.update(raw)
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("设备须为 CPU 或 CUDA")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前对战进程未检测到 CUDA，请选择 CPU")
    for key, low, high in (("simulations", 2, 4096), ("candidates", 1, 180), ("max_depth", 1, 512),
                           ("chance_initial", 1, 256), ("chance_cap", 1, 256),
                           ("q_mode", 0, 2), ("cache_capacity", 0, 65536)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"{key} 必须是 {low}–{high} 的整数")
    for key, low, high in (("q_floor", 0.00001, 2), ("value_scale", 0.00001, 10),
                           ("maxvisit_init", 0, 1000), ("gumbel_scale", 0, 10),
                           ("chance_coefficient", 0.01, 16), ("chance_exponent", 0, 1),
                           ("chance_sensitivity", 0, 10), ("afterstate_prior", 0, 16)):
        value = config[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} 必须在 {low}–{high} 之间")
        config[key] = float(value)
    for key in ("subtree_reuse", "bf16", "cuda_graph"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} 必须为布尔值")
    if config["candidates"] > config["simulations"]:
        raise ValueError("根候选数不能超过模拟次数")
    if config["chance_cap"] < config["chance_initial"]:
        raise ValueError("机会样本上限不能低于初始样本数")
    return config


def catalog(root):
    """Read only metadata; never load changing training/replay files to populate UI."""
    entries = []
    for path in root.rglob("*.pt"):
        if path.name not in ("initial.pt", "latest.pt") and not (
                path.name.startswith("actor-") and path.stem[6:].isdigit()):
            continue
        try:
            actual = path.resolve()
            if not actual.is_relative_to(root.resolve()):
                continue
            stat = path.stat()
            entries.append(dict(id=path.relative_to(root).as_posix(), run=path.parent.relative_to(root).as_posix(),
                                name=path.name, bytes=stat.st_size, modified=stat.st_mtime,
                                time=datetime.fromtimestamp(stat.st_mtime).strftime("%m-%d %H:%M"),
                                kind="actor" if path.name.startswith("actor-") else path.stem))
        except OSError:
            continue  # Training may atomically rotate/prune actors during discovery.
    entries.sort(key=lambda x: (x["kind"] == "actor", x["modified"]), reverse=True)
    return entries


def board_from_features(features):
    wall = np.rint(features[:25]).astype(int).reshape(5, 5).tolist()
    color_hot = features[25:50].reshape(5, 5)
    patterns = []
    for row in range(5):
        count = round(float(features[50 + row]) * (row + 1))
        patterns.append(dict(count=count, color=int(color_hot[row].argmax()) if count else None,
                             capacity=row + 1))
    floor_tiles = [round(float(x) * 7) for x in features[55:60]]
    floor_count = round(float(features[60]) * 7)
    return dict(wall=wall, patterns=patterns, floor_tiles=floor_tiles, floor_count=floor_count,
                floor_marker=floor_count > sum(floor_tiles), score=round(float(features[61]) * 200),
                completed_rows=sum(all(row) for row in wall))


def situation_from_wdl(probabilities, current, human):
    """Convert current-player W/D/L output into stable human/draw/AI shares."""
    win, draw, loss = (max(0.0, float(x)) for x in probabilities)
    total = win + draw + loss
    if not math.isfinite(total) or total <= 0:
        return dict(human=0.0, draw=1.0, ai=0.0)
    win, draw, loss = win / total, draw / total, loss / total
    human_win, ai_win = (win, loss) if current == human else (loss, win)
    return dict(human=human_win, draw=draw, ai=ai_win)


def describe_action(action, sources):
    source, color, destination = action // 30, (action // 6) % 5, action % 6
    origin = "中央" if source == len(sources) - 1 else f"工厂 {source + 1}"
    target = "地板" if destination == 5 else f"第 {destination + 1} 行"
    count = sources[source][color]
    return dict(action=action, source=source, color=color, destination=destination, count=count,
                text=f"从{origin}取 {count} 块{COLORS[color]}砖 → {target}")


class Match:
    def __init__(self, checkpoint, checkpoint_id, human, seed, config, manual=False, mode="ai"):
        # Checkpoints are local files produced by this project. No upload or URL loading.
        with checkpoint.open("rb") as stream:
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        if payload.get("engine_version") not in (ENGINE_VERSION, OPTIMIZED_VERSION):
            raise ValueError("不支持该 checkpoint 的引擎版本")
        model = PolicyValue(**payload["architecture"])
        model.load_state_dict(payload["model"])
        model.eval()
        self.checkpoint_id, self.iteration = checkpoint_id, payload.get("iteration", 0)
        self.engine_version = payload["engine_version"]
        self.model = model
        self.human, self.seed, self.mode = human, seed, mode
        self.config = config
        self.moves = 0
        self.manual = manual
        self.log = []
        self.undo_points = []
        self.last_move = None
        self.result = None
        self._situation_key = None
        self._situation = None
        manual_lib=ROOT/'build-play'/'azul.dll'
        self.env = Batch(1, threads=1, seed=seed, library=manual_lib if manual_lib.exists() else None)
        self.search = None
        try:
            self.env.reset_at(0, seed, 0)
            if manual:self.env.manual_reset(seed)
            self.configure(config)
        except BaseException:
            self.close()
            raise
        first = ("你" if human == 0 else "AI") if mode == "ai" else "玩家 1"
        self.add_log(f"对局开始：{first}先手。")

    def configure(self, config):
        config = validate_config(config)
        actor = FrozenActor(self.model, config["device"], bf16=config["bf16"])
        tuning = Tuning(q_mode=config["q_mode"], q_floor=config["q_floor"],
                        chance_coefficient=config["chance_coefficient"], chance_exponent=config["chance_exponent"],
                        chance_sensitivity=config["chance_sensitivity"], cache_capacity=config["cache_capacity"],
                        subtree_reuse=int(config["subtree_reuse"]), symmetric_cache=int(self.model.kind=="equivariant"),
                        afterstate_prior=config["afterstate_prior"] if self.iteration and self.model.kind=="equivariant" else 0)
        search_config = SearchConfig(**{k: config[k] for k in (
            "simulations", "candidates", "max_depth", "chance_initial", "chance_cap", "gumbel_scale", "value_scale", "maxvisit_init")})
        # One real game / root: a single CPU worker avoids occupying training cores.
        search = PipelineSearch(self.env, actor, search_config, tuning, threads=1, queues=1,
                                graphs=config["device"]=="cuda" and config["cuda_graph"])
        old = self.search
        self.search, self.actor, self.device, self.config = search, actor, config["device"], config
        self._situation_key = None
        self._situation = None
        if old:
            old.close()

    def add_log(self, text):
        self.log.append(text)

    @torch.inference_mode()
    def situation(self, observation, current, terminal, winner):
        if terminal:
            result = dict(human=0.0, draw=0.0, ai=0.0)
            result["draw" if winner == -1 else "human" if winner == self.human else "ai"] = 1.0
            return dict(**result, exact=True, label="终局结果")
        key = self.env.snapshot(0)
        if key != self._situation_key:
            tensor = torch.as_tensor(observation.copy(), device=self.actor.device,
                                     dtype=self.actor.dtype).reshape(1, -1)
            _, wdl = self.actor.model(tensor)
            probabilities = wdl.float().softmax(-1)[0].cpu().tolist()
            self._situation = dict(**situation_from_wdl(probabilities, current, self.human),
                                   exact=False, label="模型估计")
            self._situation_key = key
        return self._situation

    def state(self):
        self.env.observe()
        obs = np.ctypeslib.as_array(self.env.observations)
        current = int(self.env.players[0])
        boards = [None, None]
        boards[current] = board_from_features(obs[30:92])
        boards[1-current] = board_from_features(obs[92:154])
        terminal = bool(obs[169] > 0.5)
        if terminal:
            rank = [(p["score"], p["completed_rows"]) for p in boards]
            winner = -1 if rank[0] == rank[1] else (0 if rank[0] > rank[1] else 1)
            result = ("平局" if winner < 0 else
                      ("你赢了" if winner == self.human else "AI 获胜") if self.mode == "ai" else
                      f"玩家 {winner + 1} 获胜")
        else:
            winner, result = None, None
        situation = self.situation(obs, current, terminal, winner)
        situation["left_label"] = "你" if self.mode == "ai" else "玩家 1"
        situation["right_label"] = "AI" if self.mode == "ai" else "玩家 2"
        token = bool(obs[164] > 0.5)
        next_start = current if obs[165] > 0.5 else 1-current
        return dict(checkpoint=self.checkpoint_id, checkpoint_iteration=self.iteration,
                    human=self.human, mode=self.mode, current=current, round=round(float(obs[170])*10), moves=self.moves,
                    sources=np.rint(obs[:30]*20).astype(int).reshape(6, 5).tolist(),
                    boards=boards, bag=np.rint(obs[154:159]*20).astype(int).tolist(),
                    discard=np.rint(obs[159:164]*20).astype(int).tolist(), token_available=token,
                    next_start=next_start, token_owner=None if token else next_start,
                    terminal=terminal, winner=winner, result=result,
                    manual_deal=self.manual, waiting_deal=bool(obs[168]>.5),
                    legal=[a for a, legal in enumerate(self.env.masks) if legal],
                    can_undo=bool(self.undo_points), log=self.log[-150:], last_move=self.last_move,
                    seed=str(self.seed), config=self.config, device=self.device.upper(),
                    auxiliary_enabled=self.iteration>0 and self.model.kind=="equivariant",
                    checkpoint_engine=self.engine_version, situation=situation)

    def step(self, action, ai_seconds=None):
        before = self.state()
        src, color, dest = action//30, (action//6)%5, action%6
        count = before["sources"][src][color]
        actor = before["current"]
        result = self.env.manual_step(action) if self.manual else self.env.step([action])[0]
        if result.invalid_action:
            raise ValueError("该落点不合法，请重新选择")
        self.moves += 1
        who = ("你" if actor == self.human else "AI") if self.mode == "ai" else f"玩家 {actor + 1}"
        source = "中央" if src == 5 else f"工厂 {src+1}"
        destination = "地板" if dest == 5 else f"第 {dest+1} 行"
        marker = src == 5 and before["token_available"]
        text = f"{who}：从{source}取 {count} 块{COLORS[color]}砖 → {destination}"
        if marker:
            text += "，取得下轮先手"
        if ai_seconds is not None:
            text += f"（思考 {ai_seconds:.2f} 秒）"
        self.add_log(text)
        self.last_move = dict(actor=actor, source=src, color=color, destination=dest, count=count, text=text)
        if result.round_finished:
            after = self.state()
            old_scores = [p["score"] for p in before["boards"]]
            new_scores = [p["score"] for p in after["boards"]]
            self.add_log(f"第 {before['round']} 轮结算：玩家 1 {old_scores[0]} → {new_scores[0]} 分，"
                         f"玩家 2 {old_scores[1]} → {new_scores[1]} 分。" +
                         ("含终局奖励。" if result.terminated else "等待手动发牌。" if after['waiting_deal'] else f"第 {after['round']} 轮开始。"))
        if result.terminated:
            self.add_log(self.state()["result"] + "。同分时按完整横行数判定。")

    def ai_step(self):
        if self.state()['waiting_deal']:raise ValueError('请先完成手动发牌')
        start = time.perf_counter()
        action, _, _ = self.search.run(split_seed(self.seed ^ 0xA0761D6478BD642F, self.moves))
        self.step(int(action[0]), time.perf_counter()-start)

    def human_step(self, action):
        state = self.state()
        if state["terminal"] or (self.mode == "ai" and state["current"] != self.human):
            raise ValueError("现在不是你的回合")
        if action not in state["legal"]:
            raise ValueError("该动作不合法")
        self.undo_points.append((self.env.snapshot(0), len(self.log), self.moves, self.last_move))
        self.step(action)

    def deal(self,colors):
        if not self.manual or not self.state()['waiting_deal']:raise ValueError('当前不需要手动发牌')
        if not isinstance(colors,list):raise ValueError('请选择发牌颜色')
        self.env.manual_deal(colors)
        self.search.actor.version+=1
        self.add_log(f"第 {self.state()['round']} 轮手动发牌完成。")

    def undo(self):
        if not self.undo_points:
            raise ValueError("没有可撤回的人类行动")
        snapshot, log_size, self.moves, self.last_move = self.undo_points.pop()
        self.env.restore(0, snapshot)
        self.log = self.log[:log_size]
        # Invalidate retained trees/cache so undone future evaluations do not affect replay.
        self.search.actor.version += 1
        self.add_log("已撤回你上次行动及其后的 AI 行动。")

    def close(self):
        if getattr(self, "search", None):
            self.search.close()
            self.search = None
        if getattr(self, "env", None):
            self.env.close()
            self.env = None


class Application:
    """One match per local server, serialized mutation worker, nonblocking UI polling."""
    def __init__(self, checkpoints):
        self.checkpoints = checkpoints.resolve()
        self.lock = threading.Lock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="azul-game")
        self.advice_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="azul-advice")
        self.match = None
        self.view = None
        self.busy = False
        self.message = "请选择模型并开始对局"
        self.error = None
        self.revision = 0
        self.advice_revision = 0
        self.advice_generation = 0
        self.advice_cancel = threading.Event()
        self.advice = dict(status="closed", budget="standard", simulations=0, suggestions=[])
        self.token = secrets.token_urlsafe(24)

    def state(self):
        with self.lock:
            return dict(game=self.view, busy=self.busy, message=self.message,
                        error=self.error, revision=self.revision, advice=self.advice,
                        advice_revision=self.advice_revision)

    def _cancel_advice_locked(self, status="idle"):
        self.advice_cancel.set()
        self.advice_generation += 1
        self.advice = dict(status=status, budget=self.advice.get("budget", "standard"),
                           simulations=0, suggestions=[])
        self.advice_revision += 1

    def advice_command(self, operation, data):
        budgets = {"quick": 64, "standard": 256, "deep": 1024, "maximum": 4096, "infinite": None}
        with self.lock:
            if operation == "stop":
                self._cancel_advice_locked("stopped")
                return
            if operation != "start":
                raise ValueError("未知建议操作")
            if self.busy or not self.match or not self.view:
                raise ValueError("当前不能分析局面")
            game = self.view
            if game["terminal"] or game["waiting_deal"]:
                raise ValueError("当前局面不需要走棋建议")
            if game["mode"] == "ai" and game["current"] != game["human"]:
                raise ValueError("人机对战只提供玩家方建议")
            budget = data.get("budget", "standard")
            if budget not in budgets:
                raise ValueError("无效的建议搜索预算")
            self._cancel_advice_locked("running")
            self.advice_cancel = threading.Event()
            cancel = self.advice_cancel
            generation = self.advice_generation
            snapshot = self.match.env.snapshot(0)
            model = self.match.model
            config = dict(self.match.config)
            iteration = self.match.iteration
            sources = [list(row) for row in game["sources"]]
            perspective = game["current"]
            self.advice = dict(status="running", budget=budget, simulations=0, suggestions=[],
                               perspective=perspective, message="正在搜索…")
            self.advice_revision += 1
        self.advice_worker.submit(self._advice_work, generation, cancel, snapshot, model,
                                  config, iteration, sources, perspective, budget, budgets[budget])

    def _advice_work(self, generation, cancel, snapshot, model, config, iteration,
                     sources, perspective, budget, fixed_budget):
        total = 0
        accumulated = np.zeros(180, dtype=np.float64)
        chunks = 0
        env = search = None
        try:
            manual_lib = ROOT / "build-play" / "azul.dll"
            env = Batch(1, threads=1, seed=0, library=manual_lib if manual_lib.exists() else None)
            env.restore(0, snapshot)
            actor = FrozenActor(model, config["device"], bf16=config["bf16"])
            chunk = fixed_budget or 1024
            tuning = Tuning(q_mode=config["q_mode"], q_floor=config["q_floor"],
                            chance_coefficient=config["chance_coefficient"],
                            chance_exponent=config["chance_exponent"],
                            chance_sensitivity=config["chance_sensitivity"],
                            afterstate_prior=config["afterstate_prior"] if iteration and model.kind == "equivariant" else 0,
                            cache_capacity=config["cache_capacity"], subtree_reuse=0,
                            symmetric_cache=int(model.kind == "equivariant"))
            search_config = SearchConfig(simulations=chunk, candidates=min(config["candidates"], chunk),
                                         max_depth=config["max_depth"], chance_initial=config["chance_initial"],
                                         chance_cap=config["chance_cap"], gumbel_scale=0,
                                         value_scale=config["value_scale"], maxvisit_init=config["maxvisit_init"])
            search = PipelineSearch(env, actor, search_config, tuning, threads=1, queues=1,
                                    graphs=config["device"] == "cuda" and config["cuda_graph"])
            while not cancel.is_set():
                _, policies, _ = search.run(split_seed(0x9E3779B97F4A7C15, total + chunks))
                accumulated += policies[0]
                chunks += 1
                total += chunk
                policy = accumulated / chunks
                ranked = [int(a) for a in np.argsort(policy)[::-1] if policy[a] > 0][:3]
                suggestions = []
                for rank, action in enumerate(ranked, 1):
                    suggestion = describe_action(action, sources)
                    suggestion.update(rank=rank, confidence=float(policy[action]))
                    suggestions.append(suggestion)
                with self.lock:
                    if generation != self.advice_generation or cancel.is_set():
                        return
                    status = "running" if fixed_budget is None else "ready"
                    self.advice = dict(status=status, budget=budget, simulations=total,
                                       suggestions=suggestions, perspective=perspective,
                                       message="持续分析中，可随时停止" if status == "running" else "分析完成")
                    self.advice_revision += 1
                if fixed_budget is not None:
                    return
        except Exception as exc:
            with self.lock:
                if generation == self.advice_generation:
                    self.advice = dict(status="error", budget=budget, simulations=total,
                                       suggestions=[], perspective=perspective, message=str(exc))
                    self.advice_revision += 1
        finally:
            if search:
                search.close()
            if env:
                env.close()

    def publish(self, message=None):
        view = self.match.state() if self.match else None
        with self.lock:
            self.view = view
            if message:
                self.message = message
            self.revision += 1

    def _ai_turns(self):
        if not self.match or self.match.mode != "ai":
            return
        for _ in range(32):
            state = self.match.state()
            if state["terminal"] or state['waiting_deal'] or state["current"] == self.match.human:
                return
            self.publish("AI 思考中…")
            self.match.ai_step()
            self.publish()
        raise RuntimeError("AI 连续行动次数异常，请重新开局")

    def submit(self, operation, data):
        if operation not in ("new", "move", "undo", "retry", "configure", "deal"):
            raise ValueError("未知操作")
        with self.lock:
            if self.busy:
                raise ValueError("AI 正在处理，请稍候")
            if data.get("revision") != self.revision:
                raise ValueError("页面局面已更新，请重试")
            self._cancel_advice_locked("idle")
            self.busy = True
            self.error = None
            self.message = "加载模型中…" if operation == "new" else "处理中…"
            self.revision += 1
        self.worker.submit(self._work, operation, data)

    def _work(self, operation, data):
        try:
            if operation == "new":
                checkpoint_id = data.get("checkpoint", "")
                path = (self.checkpoints / checkpoint_id).resolve()
                if not path.is_relative_to(self.checkpoints) or checkpoint_id not in {x["id"] for x in catalog(self.checkpoints)}:
                    raise ValueError("模型文件不存在或已被训练清理，请刷新模型列表")
                mode = data.get("mode", "ai")
                if mode not in ("ai", "human"):
                    raise ValueError("无效的对局模式")
                human = data.get("human", 0) if mode == "ai" else 0
                config = validate_config(data.get("config", {}))
                if type(human) is not int or human not in (0, 1):
                    raise ValueError("无效的先手设置")
                seed_text = str(data.get("seed", "")).strip()
                seed = secrets.randbits(63) if not seed_text else int(seed_text)
                if seed < 0 or seed >= 2**64:
                    raise ValueError("种子须为 0 到 2^64−1 的整数")
                manual=data.get('manual_deal',False)
                if type(manual) is not bool:raise ValueError('手动发牌设置无效')
                new_match = Match(path, checkpoint_id, human, seed, config, manual, mode)
                old = self.match
                self.match = new_match
                if old:
                    old.close()
            else:
                if self.match is None:
                    raise ValueError("请先开始对局")
                if operation == "move":
                    action = data.get("action")
                    if type(action) is not int or not 0 <= action < 180:
                        raise ValueError("无效的动作")
                    self.match.human_step(action)
                elif operation == "undo":
                    self.match.undo()
                elif operation == "deal":
                    self.match.deal(data.get('colors'))
                elif operation == "configure":
                    self.match.configure(data.get("config", {}))
                    self.match.add_log("AI 配置已更新，从下一次 AI 思考开始生效。")
            self.publish()
            self._ai_turns()
            current = self.match.state()
            message = ("对局结束" if current["terminal"] else
                       "请为下一轮的 5 个工厂手动发牌" if current["waiting_deal"] else
                       f"轮到玩家 {current['current'] + 1}" if current["mode"] == "human" else
                       "轮到你了：选择来源和颜色，再选择落点")
            self.publish(message)
        except Exception as exc:
            self.publish()
            with self.lock:
                self.error = str(exc)
                self.message = "操作未完成"
        finally:
            with self.lock:
                self.busy = False
                self.revision += 1

    def close(self):
        with self.lock:
            self._cancel_advice_locked("closed")
        self.worker.shutdown(wait=True)
        self.advice_worker.shutdown(wait=True)
        if self.match:
            self.match.close()


def normalize_public_origin(value):
    if not value:
        return None, set()
    value = str(value).strip().rstrip('/')
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('public-origin 必须是形如 https://example.ngrok-free.app 的源地址')
    return value, {parsed.netloc.lower()}


def make_server(app, port=8765, public_origin=None):
    public_origin, public_hosts = normalize_public_origin(public_origin)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, status, payload):
            self.send_bytes(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def allowed(self):
            host = self.headers.get("Host", "").lower().split(':', 1)[0]
            local = host in {'127.0.0.1', 'localhost', '::1'}
            configured = any(host == item.split(':', 1)[0] for item in public_hosts)
            return local or configured

        def allowed_origin(self):
            origin = self.headers.get("Origin")
            if not origin:
                return True
            if origin in (f"http://127.0.0.1:{self.server.server_port}",
                          f"http://localhost:{self.server.server_port}"):
                return True
            return public_origin is not None and origin.rstrip('/') == public_origin

        def do_GET(self):
            if not self.allowed():
                return self.json(403, {"error": "仅允许本机访问"})
            route = urlsplit(self.path).path
            if route == "/api/state":
                return self.json(200, app.state())
            if route == "/api/checkpoints":
                return self.json(200, {"checkpoints": catalog(app.checkpoints)})
            if route == "/api/options":
                return self.json(200, {"defaults": default_config(), "cuda": torch.cuda.is_available()})
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            assets["/config.css"] = ("config.css", "text/css; charset=utf-8")
            if route not in assets:
                return self.json(404, {"error": "Not found"})
            file, mime = assets[route]
            body = (WEB/file).read_bytes()
            if file == "index.html":
                body = body.replace(b"__TOKEN__", app.token.encode())
            self.send_bytes(200, body, mime)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.allowed() or not self.allowed_origin() or self.headers.get("X-Azul-Token") != app.token:
                return self.json(403, {"error": "请求来源无效，请刷新本机页面"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 8192:
                    raise ValueError("请求大小不合法")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("请求格式不合法")
                route = urlsplit(self.path).path
                if route in ("/api/advice/start", "/api/advice/stop"):
                    app.advice_command(route.rsplit("/", 1)[-1], data)
                    return self.json(202, app.state())
                if route not in ("/api/new", "/api/move", "/api/undo", "/api/retry", "/api/configure", "/api/deal"):
                    return self.json(404, {"error": "Not found"})
                app.submit(route.rsplit("/", 1)[-1], data)
                self.json(202, app.state())
            except (ValueError, TypeError) as exc:
                self.json(400, {"error": str(exc)})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Azul 本地/Ngrok 对战，支持 checkpoint、CPU/CUDA 和 AI 建议")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoints", type=Path, default=ROOT/"runs")
    parser.add_argument("--public-origin", default=None,
                        help="ngrok 公网源，例如 https://abc.ngrok-free.app；也可使用 AZUL_PUBLIC_ORIGIN")
    args = parser.parse_args(argv)
    public_origin = args.public_origin or os.environ.get('AZUL_PUBLIC_ORIGIN')
    # Validate before allocating the application worker when configuration is malformed.
    public_origin, _ = normalize_public_origin(public_origin)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    app = Application(args.checkpoints)
    server = make_server(app, args.port, public_origin)
    address = public_origin or f"http://127.0.0.1:{server.server_port}"
    print(f"Azul 对战：{address}  （人机 / 人人 / AI 建议）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
