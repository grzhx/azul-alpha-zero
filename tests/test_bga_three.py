"""Public-state import, seat mapping, and BGA 2P/3P advice regressions."""
import sys
import json
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from azul3 import Batch3
from azul import Batch
from azul_ai.bga_monitor import AdviceManager, validate_snapshot
from azul_ai.search3 import Evaluator3, Search3, SearchConfig3
from azul_ai.web_game import board_from_features, describe_action

LIBRARY = ROOT / "build-bga-3p" / "azul.dll"


def initial_public(env, index=0):
    env.observe()
    observations = env.numpy_views()["observations"]
    obs = observations[index]
    current = int(env.players[index])
    players = [None] * 3
    for slot in range(3):
        board = board_from_features(obs[40 + slot * 62:40 + (slot + 1) * 62])
        board["floor"] = []
        players[(current + slot) % 3] = board
    return validate_snapshot(dict(table_id="test", table_url="https://boardgamearena.com/tableview?table=test",
                                  round=1, current=current, players=players,
                                  sources=np.rint(obs[:40] * 20).astype(int).reshape(8, 5).tolist(),
                                  token_available=True, bag_count=72,
                                  bag_colors=np.rint(obs[226:231] * 20).astype(int).tolist(),
                                  bag_colors_exact=True, raw_state_id=f"test:{current}"))


def import_game(env, game, index=0):
    visible = AdviceManager._visible_arrays(game)
    bag, discard = AdviceManager._bag_split(game, visible, np.random.default_rng(42))
    fields = {key: value for key, value in visible.items() if key != "unseen"}
    env.import_public(index, **fields, bag=bag, discard=discard, round=game["round"],
                      current=game["current"], token_available=int(game["token_available"]))


class BgaThreeTests(unittest.TestCase):
    def test_stale_marker_owner_actual_round_two(self):
        raw = json.loads((ROOT / "tests/fixtures/bga_stale_token_round2.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["token_owner"], 1)
        self.assertIn(5, raw["players"][0]["floor"])
        game = validate_snapshot(raw)
        self.assertEqual(game["token_owner"], 0)
        self.assertEqual(game["next_start"], 0)
        self.assertTrue(game["bag_colors_exact"])
        with Batch(1, 1, library=LIBRARY) as env:
            import_game(env, game)
            env.observe()
            self.assertGreater(sum(env.masks), 0)
        manager = AdviceManager()
        try:
            manager.budget = "quick"
            manager.update(game)
            manager.executor.submit(lambda: None).result(timeout=30)
            result = manager.state()
            self.assertEqual(result["status"], "ready", result)
            self.assertTrue(result["suggestions"])
        finally:
            manager.close()

    def test_full_floor_marker_owner_and_round_boundary(self):
        raw = json.loads((ROOT / "tests/fixtures/bga_stale_token_round2.json").read_text(encoding="utf-8"))
        raw["players"][0]["floor"] = [0] * 7
        self.assertEqual(validate_snapshot(raw)["token_owner"], 0)
        raw["log"].append("新的一轮开始了!")
        self.assertEqual(validate_snapshot(raw)["token_owner"], 1)
        raw["token_available"] = True
        self.assertIsNone(validate_snapshot(raw)["token_owner"])
        raw["players"][0]["floor"] = [5]
        raw["players"][1]["floor"] = [5]
        with self.assertRaisesRegex(ValueError, "多个起始玩家标记"):
            validate_snapshot(raw)

    def test_roundtrip_all_seats_and_atomic_invalid_import(self):
        with Batch3(3, 1, library=LIBRARY) as original, Batch3(3, 1, library=LIBRARY) as restored:
            for index in range(3):
                import_game(restored, initial_public(original, index), index)
            original.observe(); restored.observe()
            np.testing.assert_array_equal(original.numpy_views()["observations"], restored.numpy_views()["observations"])
            np.testing.assert_array_equal(original.numpy_views()["masks"], restored.numpy_views()["masks"])
            before = restored.numpy_views()["observations"].copy()
            invalid = initial_public(original)
            invalid["players"][0]["patterns"][0] = dict(count=2, color=0)
            invalid["bag_colors"][0] -= 2
            invalid["bag_count"] -= 2
            with self.assertRaises(RuntimeError):
                import_game(restored, invalid)
            restored.observe()
            np.testing.assert_array_equal(before, restored.numpy_views()["observations"])
            with self.assertRaises(ValueError):
                restored.import_public(0, sources=[], walls=[], scores=[], pattern_colors=[],
                                       pattern_counts=[], floor_tiles=[], floor_counts=[], bag=[], discard=[],
                                       round=1, current=0, next_start=0, token_available=1)

    def test_three_sources_validation_and_action_names(self):
        with Batch3(1, 1, library=LIBRARY) as env:
            game = initial_public(env)
        self.assertEqual(game["factory_count"], 7)
        self.assertEqual(game["player_count"], 3)
        with self.assertRaises(ValueError):
            validate_snapshot(dict(game, current=3))
        with self.assertRaises(ValueError):
            validate_snapshot(dict(game, sources=game["sources"][:6]))
        self.assertIn("工厂 6", describe_action(5 * 30, game["sources"])["text"])
        self.assertIn("中央", describe_action(7 * 30, game["sources"])["text"])

    def test_gpu_advice_and_model_compatibility(self):
        manager = AdviceManager()
        try:
            manager.budget = "quick"
            with Batch3(3, 1, library=LIBRARY) as env:
                game = initial_public(env, 2)
            manager.update(game)
            manager.executor.submit(lambda: None).result(timeout=30)
            result = manager.state()
            self.assertEqual(result["status"], "ready", result)
            # Training can publish a newer engine version; selection must remain compatible.
            self.assertEqual(manager.checkpoint_players(result["checkpoint"]), 3)
            self.assertEqual(result["perspective"], 2)
            self.assertEqual(result["simulations"], 64)
            self.assertEqual(result["situation"]["kind"], "rank_utility")
            self.assertTrue(result["suggestions"])
            model, _, device = manager._load_actor3(result["checkpoint"])
            with Batch3(1, 1, library=LIBRARY) as env:
                import_game(env, game); env.observe()
                with torch.inference_mode():
                    _, utilities = model(torch.tensor(env.numpy_views()["observations"], device=device))
                np.testing.assert_allclose(result["situation"]["utilities"],
                                           np.roll(utilities.cpu().numpy()[0], 2), atol=.01)
                evaluator = Evaluator3(model, 1, device, use_graph=False)
                cancelled = threading.Event(); cancelled.set()
                with Search3(env, SearchConfig3(simulations=16, candidates=8), 1) as search:
                    self.assertIsNone(search.run(evaluator, 42, cancel=cancelled))
            with self.assertRaises(ValueError):
                manager.command("configure", {"checkpoint": "optimized/latest.pt"}, game)
            manager.update(dict(game, terminal=True, players=[dict(player, score=score) for player, score in zip(game["players"], [10, 30, 20])]), force=True)
            self.assertEqual(manager.state()["situation"]["expected_ranks"], [3, 1, 2])
        finally:
            manager.close()

    def test_two_player_advice_regression(self):
        with Batch(1, 1, library=LIBRARY) as env:
            env.observe()
            obs = env.numpy_views()["observations"][0]
            current = int(env.players[0]); players = [None] * 2
            for slot in range(2):
                board = board_from_features(obs[30 + slot * 62:30 + (slot + 1) * 62])
                board["floor"] = []
                players[(current + slot) % 2] = board
            game = validate_snapshot(dict(table_id="test2", table_url="https://boardgamearena.com/tableview?table=test2",
                                          round=1, current=current, players=players,
                                          sources=np.rint(obs[:30] * 20).astype(int).reshape(6, 5).tolist(),
                                          token_available=True, bag_count=80,
                                          bag_colors=np.rint(obs[154:159] * 20).astype(int).tolist(),
                                          bag_colors_exact=True, raw_state_id="test2"))
        manager = AdviceManager()
        try:
            manager.budget = "quick"
            manager.update(game)
            manager.executor.submit(lambda: None).result(timeout=30)
            result = manager.state()
            self.assertEqual(result["status"], "ready", result)
            self.assertEqual(result["checkpoint"], "optimized/latest.pt")
            self.assertAlmostEqual(sum(result["situation"][key] for key in ("win", "draw", "loss")), 1, places=5)
            manager.update(dict(game, actionable=False), force=True)
            self.assertEqual(manager.state()["status"], "waiting")
        finally:
            manager.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
