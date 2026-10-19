import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from azul import Batch
from azul_ai.model import EndFeatures, PolicyValue, permute_factories
from azul_ai.optimized_learning import DeviceReplay, GraphLearner
from azul_ai.optimized_search import FrozenActor, BucketEvaluator


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_committed_ending_and_color_geometry(self):
        obs = torch.zeros(4, 172)
        # Player 1, row 3: column 1 is missing, hence color 3 fills it.
        start = 92
        obs[:, start + 15:start + 20] = 1
        obs[:, start + 16] = 0
        obs[:, start + 25 + 3 * 5 + 3] = 1
        obs[:, start + 50 + 3] = torch.tensor([1., .75, 0., 1.])
        # Last sample has a full row with a different color, not an ending move.
        obs[3, start + 25 + 3 * 5 + 3] = 0
        obs[3, start + 25 + 3 * 5 + 1] = 1
        obs[:, start + 60] = 6 / 7
        features = EndFeatures()(obs)
        self.assertEqual(features.shape, (4, 45))
        self.assertEqual(features[:, -1].tolist(), [1., 0., 0., 0.])
        self.assertTrue(torch.equal(features[:, 22 + 3], torch.full((4,), .2)))
        self.assertEqual(features[:, 22 + 15 + 3].tolist(), [1., 0., 0., 0.])
        self.assertAlmostEqual(features[0, 22 + 20].item(), 2 / 95)
        self.assertAlmostEqual(features[0, 22 + 21].item(), 11 / 14)

        partial = obs[:1].clone()
        partial[:, start + 50 + 3] = .5
        self.assertEqual(EndFeatures()(partial)[0, 22 + 15 + 3].item(), 0.)

    def test_bonuses_and_terminal_observation(self):
        obs = torch.zeros(1, 172)
        obs[:, 30:55] = 1
        features = EndFeatures()(obs)
        self.assertEqual(features[0, 20].item(), 1.)
        self.assertEqual(features[0, -1].item(), 1.)
        self.assertEqual(features[0, :5].sum().item(), 0.)

    def test_native_games_committed_wall_and_perspectives(self):
        rng = np.random.default_rng(7)
        ending_count = 0
        with Batch(8, 2, 23) as env:
            for _ in range(180):
                env.observe()
                views = env.numpy_views()
                obs = views["observations"].copy()
                actual = EndFeatures()(torch.from_numpy(obs)).numpy()
                for lane in range(8):
                    ends = False
                    for player in range(2):
                        p = obs[lane, 30 + 62 * player:92 + 62 * player]
                        wall = p[:25].reshape(5, 5).astype(bool).copy()
                        colors = p[25:50].reshape(5, 5)
                        for row in range(5):
                            if p[50 + row] >= .999 and colors[row].any():
                                wall[row, (row + colors[row].argmax()) % 5] = True
                        ends |= bool(wall.all(1).any())
                        self.assertTrue(np.allclose(actual[lane, 22 * player + 10:22 * player + 15], (~wall).sum(1) / 5))
                        bonus = 2 * wall.all(1).sum() + 7 * wall.all(0).sum()
                        bonus += 10 * sum(all(wall[r, (r + c) % 5] for r in range(5)) for c in range(5))
                        self.assertAlmostEqual(float(actual[lane, 22 * player + 20]), bonus / 95, places=6)
                    self.assertEqual(bool(actual[lane, -1]), ends)
                    ending_count += ends
                actions = np.where(views["masks"], rng.random((8, 180)), -1).argmax(1)
                env.step(actions)
                if all(x.terminated for x in env.results):
                    break
        self.assertGreater(ending_count, 0)

    def test_old_model_migration_and_feature_gradients(self):
        for kind in ("mlp", "equivariant"):
            old = PolicyValue(32, 1, kind=kind)
            new = PolicyValue(32, 1, kind=kind, end_features=True)
            new.load_compatible_state_dict(old.state_dict())
            obs = torch.rand(8, 172)
            expected = old(obs)
            actual = new(obs)
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
            sum(x.square().mean() for x in actual).backward()
            self.assertGreater(new.input.weight.grad[:, -45:].abs().sum().item(), 0)
            restored = PolicyValue(**new.architecture)
            restored.load_state_dict(new.state_dict())
            for a, b in zip(actual, restored(obs)):
                torch.testing.assert_close(a, b)

    def test_value_tower_migration_preserves_old_outputs(self):
        old = PolicyValue(32, 1, kind="equivariant", end_features=True)
        new = PolicyValue(32, 1, kind="equivariant", end_features=True,
                          value_tower=True, value_tower_blocks=2)
        new.load_compatible_state_dict(old.state_dict())
        obs = torch.rand(8, 172)
        old_out = old(obs)
        new_out = new(obs)
        for a, b in zip(old_out, new_out):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        loss = new_out[1].square().mean()
        loss.backward()
        self.assertGreater(new.value_gate.grad.abs().item(), 0)
        self.assertTrue(torch.isfinite(new.value_delta[1].weight.grad).all())
        broken = dict(old.state_dict())
        del broken["value.weight"]
        with self.assertRaises(RuntimeError):
            new.load_compatible_state_dict(broken)

    def test_actor_initialization_starts_a_new_training_line(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="azul-value-init-") as tmp:
            path = Path(tmp)
            source = path / "source.pt"
            old = PolicyValue(32, 1, kind="equivariant", end_features=True)
            torch.save({"engine_version": "azul-gumbel-v2", "architecture": old.architecture,
                        "model": old.state_dict(), "iteration": 1900}, source)
            output = path / "new-run"
            command = [sys.executable, str(root / "scripts/train_optimized.py"),
                       "--output", str(output), "--init-from", str(source), "--end-features",
                       "--value-tower", "--value-tower-blocks", "1", "--value-rank-weight", ".1",
                       "--width", "32", "--blocks", "1", "--iterations", "1901", "--envs", "4",
                       "--positions-per-update", "160", "--threads", "2", "--queues", "1",
                       "--simulations", "4", "--candidates", "4", "--batch-size", "32",
                       "--replay-capacity", "1000", "--warmup", "1", "--updates-per-iteration", "2",
                       "--device", "cpu"]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            initial = torch.load(output / "initial.pt", weights_only=False)
            for key, value in old.state_dict().items():
                self.assertTrue(torch.equal(initial["model"][key], value), key)
            ck = torch.load(output / "latest.pt", weights_only=False)
            self.assertEqual(ck["iteration"], 1901)
            self.assertTrue(ck["architecture"]["value_tower"])
            self.assertEqual(ck["updates"], 2)
            self.assertGreater(ck["model"]["value_gate"].abs().item(), 0)

    def test_factory_symmetry_with_active_features(self):
        model = PolicyValue(32, 1, kind="equivariant", end_features=True)
        with torch.no_grad():
            model.input.weight[:, -45:].normal_(0, .1)
        obs = torch.rand(8, 172)
        logits, value = model(obs)
        perms = torch.rand(8, 5).argsort(-1)
        transformed, expected, _ = permute_factories(obs, logits, torch.ones_like(logits), perms)
        actual, v = model(transformed)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v, value, atol=1e-6, rtol=1e-6)

    def test_full_checkpoint_migration_with_old_history_and_resume(self):
        root = Path(__file__).resolve().parents[1]
        def run(argv, success=True):
            result = subprocess.run([sys.executable, str(root / "scripts/train_optimized.py"),
                                     *argv, "--device", "cpu"], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        with tempfile.TemporaryDirectory(prefix="azul-end-features-") as tmp:
            path = Path(tmp)
            run(["--output", tmp, "--iterations", "1", "--envs", "4", "--positions-per-update", "160",
                 "--threads", "2", "--queues", "1", "--simulations", "4", "--candidates", "4",
                 "--width", "32", "--blocks", "1", "--batch-size", "32", "--warmup", "1",
                 "--replay-capacity", "1000", "--updates-per-iteration", "2"])
            latest = str(path / "latest.pt")
            old = torch.load(latest, map_location="cpu", weights_only=False)
            run(["--resume", latest, "--iterations", "2", "--end-features"], False)
            run(["--resume", latest, "--retune", "--iterations", "2", "--end-features",
                 "--history-fraction", ".4", "--history-iterations", "1"])
            ck = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertTrue(ck["architecture"]["end_features"])
            self.assertEqual(ck["model"]["input.weight"].shape[1], old["model"]["input.weight"].shape[1] + 45)
            self.assertEqual(ck["updates"], old["updates"] + 2)
            replay = DeviceReplay(1000, "cpu")
            from azul_ai.optimized_learning import CheckpointJournal
            CheckpointJournal.restore(Path(latest), replay, ck)
            self.assertTrue((replay.birth[:replay.size] == 1).any())
            run(["--resume", latest, "--iterations", "3"])
            final = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertTrue(final["args"]["end_features"])
            self.assertEqual(final["phase_history"], ck["phase_history"])
            self.assertIn("end_features", ck["phase_history"][0]["changes"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_graph_inference_and_learning(self):
        model = PolicyValue(32, 1, kind="equivariant", end_features=True, value_tower=True).cuda()
        replay = DeviceReplay(128, "cuda")
        obs = np.random.default_rng(3).random((128, 172)).astype(np.float16)
        obs[:, 168] = 0
        masks = np.ones((128, 180), np.uint8)
        replay.add((obs, masks.astype(np.float16) / 180, masks, np.arange(128, dtype=np.uint8) % 3), 1)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, fused=True, capturable=True)
        learner = GraphLearner(model, opt, replay, 32, value_rank_weight=.1)
        metrics = learner.run(2)
        self.assertEqual(metrics["invalid_updates"], 0)
        self.assertGreater(model.input.weight[:, -45:].abs().sum().item(), 0)
        actor = FrozenActor(model, "cuda")
        evaluator = BucketEvaluator(actor, 8)
        evaluator.observations[:] = obs[:8]
        evaluator.launch(np.ones(8, np.uint8))
        evaluator.finish()
        self.assertTrue(np.isfinite(evaluator.values).all())


if __name__ == "__main__":
    unittest.main()
