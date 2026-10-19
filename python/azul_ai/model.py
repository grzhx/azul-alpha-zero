import torch
from torch import nn
from torch.nn import functional as F


END_FEATURE_COUNT = 45


class EndFeatures(nn.Module):
    """Rule-derived features in observation player order; no lookahead or RNG."""

    def __init__(self):
        super().__init__()
        rows = torch.arange(5)
        # A color c belongs at column (row + c) % 5 on the standard wall.
        self.register_buffer("color_columns", ((rows[:, None] + rows) % 5)[None, None], persistent=False)
        self.register_buffer("column_colors", ((rows[None, :] - rows[:, None]) % 5)[None, None], persistent=False)
        self.register_buffer("penalties", torch.tensor([0., 1., 2., 4., 6., 8., 11., 14.]), persistent=False)

    def forward(self, x):
        players = x[:, 30:154].reshape(-1, 2, 62).float()
        wall = players[:, :, :25].reshape(-1, 2, 5, 5) > 0.5
        colors = players[:, :, 25:50].reshape(-1, 2, 5, 5) > 0.5
        full = players[:, :, 50:55] >= 0.999
        missing = (~wall).sum(-1)
        at_color = wall.gather(-1, self.color_columns.expand(wall.shape[0], 2, -1, -1))
        # Empty pattern lines can still take any missing color; nonempty ones are locked.
        compatible = ((~at_color) & (colors | ~colors.any(-1, keepdim=True))).any(-1)
        triggers = full & (missing == 1) & compatible
        placements = (full[..., None] & colors).gather(-1, self.column_colors.expand(wall.shape[0], 2, -1, -1))
        projected = wall | placements
        projected_missing = (~projected).sum(-1)
        ends = (projected_missing == 0).any(dim=(1, 2))
        # Bonuses on the committed wall, before any further choices this round.
        bonus = (2 * (projected_missing == 0).sum(-1)
                 + 7 * projected.all(-2).sum(-1)
                 + 10 * projected.gather(-1, self.color_columns.expand(wall.shape[0], 2, -1, -1)).all(-2).sum(-1))
        floor_index = (players[:, :, 60] * 7).round().long().clamp(0, 7)
        penalty = self.penalties[floor_index].float()
        per_player = torch.cat((missing.float() / 5, full.float(), projected_missing.float() / 5,
                                triggers.float(), bonus[..., None].float() / 95,
                                penalty[..., None] / 14), -1)
        return torch.cat((per_player.flatten(1), ends[:, None].float()), -1).to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x):
        return x + self.second(F.silu(self.first(self.norm(x))))


class PolicyValue(nn.Module):
    def __init__(self, width=256, blocks=4, kind="mlp", end_features=False,
                 value_tower=False, value_tower_blocks=2):
        super().__init__()
        self.architecture = {"width": width, "blocks": blocks}
        self.kind = kind
        self.value_tower_enabled = bool(value_tower)
        self.value_tower_blocks = int(value_tower_blocks)
        self.end_features = EndFeatures() if end_features else None
        if end_features:
            self.architecture["end_features"] = True
        if self.value_tower_enabled:
            if self.value_tower_blocks < 0:
                raise ValueError("value_tower_blocks must be nonnegative")
            self.architecture["value_tower"] = True
            self.architecture["value_tower_blocks"] = self.value_tower_blocks
        extra = END_FEATURE_COUNT if end_features else 0
        if kind not in ("mlp", "equivariant"):
            raise ValueError("Unknown network architecture")
        if kind == "equivariant":
            self.architecture["kind"] = kind
            self.factory_width = max(16, width // 4)
            self.factory_encoder = nn.Sequential(nn.Linear(5, self.factory_width), nn.SiLU(),
                                                 nn.Linear(self.factory_width, self.factory_width), nn.SiLU())
            self.input = nn.Linear(147 + self.factory_width + extra, width)
            self.context = nn.Linear(width, self.factory_width)
            self.factory_policy = nn.Sequential(nn.Linear(2*self.factory_width, self.factory_width), nn.SiLU(),
                                                 nn.Linear(self.factory_width, 30))
            self.afterstate = nn.Linear(width, 3)
        else:
            self.input = nn.Linear(172 + extra, width)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.norm = nn.LayerNorm(width)
        self.policy = nn.Linear(width, 30 if kind == "equivariant" else 180)
        self.value = nn.Linear(width, 3)  # win, draw, loss, current player's view
        if self.value_tower_enabled:
            self.value_delta = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                *(ResidualBlock(width) for _ in range(self.value_tower_blocks)),
                nn.Linear(width, width))
            # Zero preserves the 1900 policy/value outputs at migration. It is
            # learned immediately by the utility-calibration loss.
            self.value_gate = nn.Parameter(torch.zeros(()))
        if extra:
            with torch.no_grad():
                self.input.weight[:, -extra:].zero_()

    def load_compatible_state_dict(self, state):
        """Explicit one-way migration; normal load_state_dict remains strict."""
        state = dict(state)
        old = state["input.weight"]
        new = self.input.weight
        if self.end_features is not None and old.shape == (new.shape[0], new.shape[1] - END_FEATURE_COUNT):
            state["input.weight"] = F.pad(old, (0, END_FEATURE_COUNT))
        if self.value_tower_enabled and "value_gate" not in state:
            # Only the new value branch may be initialized during migration.
            # Missing legacy weights and all shape mismatches remain errors.
            for key, value in self.state_dict().items():
                if key == "value_gate" or key.startswith("value_delta."):
                    state[key] = value
        return self.load_state_dict(state)

    def forward(self, x):
        x = x.to(self.input.weight.dtype)
        if self.kind == "equivariant":
            f = self.factory_encoder(x[:, :25].reshape(-1, 5, 5))
            inputs = torch.cat((x[:, 25:], f.mean(1)), -1)
            if self.end_features is not None:
                inputs = torch.cat((inputs, self.end_features(x)), -1)
            pre = self.input(inputs)
            h = self.norm(self.blocks(F.silu(pre)))
            context = self.context(h)[:, None, :].expand(-1, 5, -1)
            factory_logits = self.factory_policy(torch.cat((f, context), -1)).reshape(-1, 150)
            policy = torch.cat((factory_logits, self.policy(h)), -1)
            value_h = h if not self.value_tower_enabled else h + self.value_gate * self.value_delta(h)
            wdl = torch.where(x[:, 168:169] > 0, self.afterstate(value_h), self.value(value_h))
            return policy, wdl
        inputs = x if self.end_features is None else torch.cat((x, self.end_features(x)), -1)
        pre = self.input(inputs)
        h = self.norm(self.blocks(F.silu(pre)))
        value_h = h if not self.value_tower_enabled else h + self.value_gate * self.value_delta(h)
        policy = self.policy(h)
        return policy, self.value(value_h)


def policy_value_from_architecture(architecture):
    """Construct a paired-player model from old or end-feature checkpoint metadata."""
    architecture = dict(architecture or {})
    if any(architecture.get(k) for k in ("rule_features", "action_features", "structured_value")):
        raise ValueError("Post-2300 structured architectures are outside this release")
    return PolicyValue(width=int(architecture.get("width", 256)),
                       blocks=int(architecture.get("blocks", 4)),
                       kind=str(architecture.get("kind", "mlp")),
                       end_features=bool(architecture.get("end_features", False)),
                       value_tower=bool(architecture.get("value_tower", False)),
                       value_tower_blocks=int(architecture.get("value_tower_blocks", 2)))


def upgraded_architecture(source, target):
    """Only enable additive branches; never silently resize learned towers."""
    source = dict(source)
    for key in ("end_features",):
        if target.get(key) and not source.get(key):
            source[key] = True
    if target.get("value_tower") and not source.get("value_tower"):
        source.update(value_tower=True, value_tower_blocks=target.get("value_tower_blocks", 2))
    return source


def load_compatible_optimizer(optimizer, model, payload):
    """Migrate by parameter name, never by changed parameter list positions."""
    old = payload["optimizer"]
    if len(old["param_groups"]) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError("Compatible optimizer migration requires one parameter group")
    with torch.random.fork_rng(devices=[]):
        original = policy_value_from_architecture(payload["architecture"])
    old_names = list(dict(original.named_parameters()))
    old_ids = old["param_groups"][0]["params"]
    if len(old_names) != len(old_ids):
        raise ValueError("Optimizer parameter metadata mismatch")
    new_params = dict(model.named_parameters())
    new = optimizer.state_dict()
    new_ids = dict(zip(new_params, new["param_groups"][0]["params"]))
    migrated = {}
    for name, identifier in zip(old_names, old_ids):
        if name not in new_params:
            raise ValueError(f"Removed optimizer parameter: {name}")
        values = {}
        for key, value in old["state"].get(identifier, {}).items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape != new_params[name].shape:
                shape = new_params[name].shape
                if name != "input.weight" or value.shape != (shape[0], shape[1]-END_FEATURE_COUNT):
                    raise ValueError(f"Optimizer moment shape mismatch: {name}/{key}")
                value = F.pad(value, (0, END_FEATURE_COUNT))
            values[key] = value
        if values:
            migrated[new_ids[name]] = values
    group = dict(old["param_groups"][0], params=new["param_groups"][0]["params"])
    optimizer.load_state_dict(dict(state=migrated, param_groups=[group]))


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
