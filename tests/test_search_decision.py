import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from azul_ai.search_decision import rank_search_actions, require_selected_move


class SearchDecisionTests(unittest.TestCase):
    def test_native_action_wins_over_policy_maximum(self):
        policy = np.array([.9, .1, .8], dtype=float)
        votes = np.zeros(3, dtype=np.int64)
        selected, order, method = rank_search_actions(np.array([2]), policy, votes)
        self.assertEqual(selected, 2)
        self.assertEqual(order[0], 2)
        self.assertEqual(method, 'native_sequential_halving')
        self.assertEqual(votes.tolist(), [0, 0, 1])

    def test_multiple_root_vote_is_explicit(self):
        policy = np.ones(3)
        votes = np.zeros(3, dtype=np.int64)
        selected, order, method = rank_search_actions(np.array([1, 2, 1]), policy, votes)
        self.assertEqual((selected, method), (1, 'native_action_vote'))
        self.assertEqual(order[0], 1)

    def test_execution_requires_native_selection(self):
        with self.assertRaises(ValueError):
            require_selected_move({'suggestions':[{'action':0}]})
        move = {'action': 2, 'selected': True, 'text': 'native'}
        got = require_selected_move({'selected_action':2, 'selected_move':move,
                                     'selection_method':'native_sequential_halving'})
        self.assertEqual(got['action'], 2)


if __name__ == '__main__':
    unittest.main()
