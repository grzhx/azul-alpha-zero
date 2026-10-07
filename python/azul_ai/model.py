import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x):
        return x + self.second(F.silu(self.first(self.norm(x))))


class PolicyValue(nn.Module):
    def __init__(self, width=256, blocks=4, kind="mlp"):
        super().__init__()
        self.architecture = {"width": width, "blocks": blocks}
        self.kind = kind
        if kind not in ("mlp", "equivariant"):
            raise ValueError("Unknown network architecture")
        if kind == "equivariant":
            self.architecture["kind"] = kind
            self.factory_width = max(16, width // 4)
            self.factory_encoder = nn.Sequential(nn.Linear(5, self.factory_width), nn.SiLU(),
                                                 nn.Linear(self.factory_width, self.factory_width), nn.SiLU())
            self.input = nn.Linear(147 + self.factory_width, width)
            self.context = nn.Linear(width, self.factory_width)
            self.factory_policy = nn.Sequential(nn.Linear(2*self.factory_width, self.factory_width), nn.SiLU(),
                                                 nn.Linear(self.factory_width, 30))
            self.afterstate = nn.Linear(width, 3)
        else:
            self.input = nn.Linear(172, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.norm = nn.LayerNorm(width)
        self.policy = nn.Linear(width, 30 if kind == "equivariant" else 180)
        self.value = nn.Linear(width, 3)  # win, draw, loss, current player's view

    def forward(self, x):
        if self.kind == "equivariant":
            f = self.factory_encoder(x[:, :25].reshape(-1, 5, 5))
            h = self.norm(self.blocks(F.silu(self.input(torch.cat((x[:, 25:], f.mean(1)), -1)))))
            context = self.context(h)[:, None, :].expand(-1, 5, -1)
            factory_logits = self.factory_policy(torch.cat((f, context), -1)).reshape(-1, 150)
            policy = torch.cat((factory_logits, self.policy(h)), -1)
            wdl = torch.where(x[:, 168:169] > 0, self.afterstate(h), self.value(h))
            return policy, wdl
        h = self.norm(self.blocks(F.silu(self.input(x))))
        return self.policy(h), self.value(h)


def permute_factories(obs, policy, mask, permutations):
    """permutations[b,new_source] = old_source. Center source 5 stays fixed.

    Changes only source slots; wall/row/color permutations are NOT symmetries.
    This returns new tensors and never modifies replay samples in place.
    """
    b = obs.shape[0]
    sources = torch.cat((permutations, torch.full((b, 1), 5, device=obs.device, dtype=torch.long)), 1)
    out_obs = obs.clone()
    out_obs[:, :30] = obs[:, :30].reshape(b, 6, 5).gather(1, sources[:, :, None].expand(-1, -1, 5)).reshape(b, 30)
    out_policy = policy.reshape(b, 6, 30).gather(1, sources[:, :, None].expand(-1, -1, 30)).reshape(b, 180)
    out_mask = mask.reshape(b, 6, 30).gather(1, sources[:, :, None].expand(-1, -1, 30)).reshape(b, 180)
    return out_obs, out_policy, out_mask


def loss_for_batch(model, obs, targets, masks, outcomes):
    logits, wdl_logits = model(obs)
    # FP32: avoid 0 * -inf NaNs, and never train the network to prefer illegal moves.
    logits = logits.float().masked_fill(~masks.bool(), -1e9)
    policy_loss = -(targets.float() * F.log_softmax(logits, -1)).sum(-1).mean()
    value_loss = F.cross_entropy(wdl_logits.float(), outcomes.long())
    return policy_loss + value_loss, policy_loss, value_loss
