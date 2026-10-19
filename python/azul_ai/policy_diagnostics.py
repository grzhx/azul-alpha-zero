"""Search-target policy metrics over legal actions; no training state changes."""
import torch


@torch.no_grad()
def policy_metrics(logits, target, mask):
    logits = logits.float().masked_fill(~mask.bool(), -1e9)
    target = target.float()
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = logits.log_softmax(-1)
    entropy = -(target * target.clamp_min(1e-12).log()).sum(-1)
    kl = -(target * logp).sum(-1) - entropy
    predicted = logits.topk(3, dim=-1).indices
    top = target.argmax(-1)
    return dict(policy_target_kl=kl, policy_target_entropy=entropy,
                policy_top1_agreement=(predicted[:, 0] == top).float(),
                policy_top3_agreement=(predicted == top[:, None]).any(-1).float(),
                policy_top3_target_mass=target.gather(-1, predicted).sum(-1))
