"""Keep Sequential Halving decisions separate from the training policy target."""
import numpy as np


def rank_search_actions(actions, policy, votes, limit=5):
    """One root uses its native action. Multiple roots/waves vote on native actions.

    Training probabilities are used only to order supplementary alternatives.
    Tied votes use action ID, never resurrect a non-selected policy maximum.
    """
    policy = np.asarray(policy)
    native = np.asarray(actions).reshape(-1)
    if policy.ndim != 1 or votes.shape != policy.shape:
        raise ValueError('Search decision dimensions do not match')
    for value in native:
        action = int(value)
        if value != action or not 0 <= action < len(policy) or not np.isfinite(policy[action]):
            raise ValueError('Native search returned an invalid final action')
        votes[action] += 1
    if not len(native):
        raise ValueError('Native search returned no final action')
    selected = int(np.argmax(votes))
    order = [selected] + [int(a) for a in np.argsort(policy)[::-1]
                          if int(a) != selected and np.isfinite(policy[a]) and policy[a] > 0]
    method = 'native_sequential_halving' if int(votes.sum()) == 1 else 'native_action_vote'
    return selected, order[:limit], method


def selection_fields(action, selected, policy, votes):
    return dict(confidence=float(policy[action]), policy_probability=float(policy[action]),
                selected=action == selected, native_votes=int(votes[action]),
                native_vote_fraction=float(votes[action] / votes.sum()))


def require_selected_move(advice):
    """Execution must use the explicit decision, regardless of list ordering."""
    move = advice.get('selected_move')
    if (not isinstance(move, dict) or advice.get('selected_action') != move.get('action')
            or not move.get('selected') or advice.get('selection_method') not in
            ('native_sequential_halving', 'native_action_vote', 'native_partial_sequential_halving', 'native_partial_action_vote')):
        raise ValueError('等待原生 Sequential Halving 最终下法，不能按训练策略最大概率执行')
    return dict(move)
