"""Check native vote selection, score accounting and complete-pair resume."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_tree_counts import restore_records, save_json, summarize, vote_actions


class EvaluationTests(unittest.TestCase):
    def test_native_vote_ignores_policy_maximum_and_breaks_ties_by_id(self):
        actions = np.asarray([8, 8, 7, 7, 5, 5, 5, 9])
        policy = np.zeros((8, 180), np.float32)
        policy[:, 42] = 1
        self.assertEqual(vote_actions(actions, policy, [0, 1], 4), {0: 7, 1: 5})

    def test_record_counts_and_game_result_scores(self):
        contract = dict(candidate="A", opponent="B", simulations=[1024, 1024], seed=8, pairs=3)
        scores = [1., 1., .5, 0., 1., 0.]
        records = [dict(pair=i // 2, seat=i % 2, score=v, scores=[30, 40])
                   for i, v in enumerate(scores)]
        report = summarize(contract, records, 10., final=True)
        self.assertEqual((report["wins"], report["draws"], report["losses"]), (3, 1, 2))
        self.assertEqual((report["paired_score"]["a_points"], report["paired_score"]["b_points"]), (1, 1))
        self.assertAlmostEqual(report["score_rate"], 3.5 / 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.json"
            save_json(path, report)
            self.assertEqual(restore_records(path, contract), (records, 10.))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                restore_records(path, dict(contract, seed=9))
            report["records"][2]["pair"] = 5
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                restore_records(path, contract)


if __name__ == "__main__":
    unittest.main()
