"""python -m azul_ai.train --help (set PYTHONPATH=python)."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from azul import Batch
from . import ENGINE_VERSION
from .model import PolicyValue, permute_factories, loss_for_batch
from .search import Search, SearchConfig, Evaluator
from .selfplay import collect, split_seed


class Replay:
    def __init__(self, capacity):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.obs = np.empty((capacity, 172), np.float16)
        self.policy = np.empty((capacity, 180), np.float16)
        self.masks = np.empty((capacity, 180), np.uint8)
        self.outcomes = np.empty(capacity, np.uint8)

    def add(self, data):
        n = len(data[0])
        if n > self.capacity:
            data = tuple(x[-self.capacity:] for x in data)
            n = self.capacity
        indices = (np.arange(n) + self.cursor) % self.capacity
        for target, source in zip((self.obs, self.policy, self.masks, self.outcomes), data):
            target[indices] = source
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.capacity, self.size+n)

    def state(self):
        return dict(size=self.size, cursor=self.cursor, capacity=self.capacity,
                    obs=self.obs[:self.size], policy=self.policy[:self.size],
                    masks=self.masks[:self.size], outcomes=self.outcomes[:self.size])

    @classmethod
    def restore(cls, state):
        r = cls(state["capacity"])
        r.size, r.cursor = state["size"], state["cursor"]
        for name in ("obs", "policy", "masks", "outcomes"):
            getattr(r, name)[:r.size] = state[name]
        return r


def emit(row, path=None):
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    print(text, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text+"\n")


def save_checkpoint(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def device_setup(name, cpu_threads):
    torch.set_num_threads(cpu_threads)
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use the project .venv CUDA PyTorch or --device cpu")
    if name == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.device(name)


def parser():
    p = argparse.ArgumentParser(description="Full Gumbel pure self-play; factory augmentation and chance reuse ON")
    p.add_argument("--output", type=Path, default=Path("runs/gumbel"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--iterations", type=int, default=100, help="Total target wave count, including resumed waves")
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--chance-cap", type=int, default=64)
    p.add_argument("--max-depth", type=int, default=96)
    p.add_argument("--max-steps", type=int, default=4096)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--replay-capacity", type=int, default=500000)
    p.add_argument("--warmup", type=int, default=20000)
    p.add_argument("--sample-reuse", type=float, default=4.0)
    p.add_argument("--updates-per-iteration", type=int, default=0, help="Optional cap; 0 means sample-reuse budget")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--fp32", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.resume and ((args.output/"initial.pt").exists() or (args.output/"latest.pt").exists()):
        raise FileExistsError("Run already exists: use --resume <run>/latest.pt or a new --output directory")
    if min(args.iterations, args.envs, args.threads, args.cpu_threads, args.batch_size,
           args.replay_capacity, args.max_steps, args.width, args.blocks) < 1 or args.sample_reuse <= 0:
        raise ValueError("Counts and sample reuse must be positive")
    if args.updates_per_iteration < 0 or args.warmup < 0 or args.learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    device = device_setup(args.device, args.cpu_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(split_seed(args.seed, 991))
    search_config = SearchConfig(simulations=args.simulations, candidates=args.candidates,
                                 chance_cap=args.chance_cap, max_depth=args.max_depth)
    model = PolicyValue(args.width, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=device.type=="cuda")
    replay = Replay(args.replay_capacity)
    start_iteration, updates, games, credit = 0, 0, 0, 0.0
    payload = None
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload["engine_version"] != ENGINE_VERSION or payload["architecture"] != model.architecture:
            raise ValueError("Checkpoint engine/architecture mismatch")
        # State transitions and future episode identities must retain their protocol.
        for key in ("envs", "seed", "max_steps", "simulations", "candidates", "chance_cap", "max_depth",
                    "batch_size", "sample_reuse", "updates_per_iteration", "warmup", "fp32", "learning_rate"):
            if payload["args"][key] != getattr(args, key):
                raise ValueError(f"Resume requires same --{key.replace('_','-')}")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        replay = Replay.restore(payload["replay"])
        start_iteration, updates, games, credit = (payload[k] for k in ("iteration", "updates", "games", "credit"))
        rng.bit_generator.state = payload["numpy_rng"]
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "metrics.jsonl"
    emit(dict(event="start", engine=ENGINE_VERSION, device=str(device), torch=torch.__version__,
              gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
              parameters=sum(p.numel() for p in model.parameters()), search=asdict(search_config),
              factory_augmentation=True, chance_reuse=True, cuda_graph=device.type=="cuda" and not args.no_cuda_graph,
              resumed_iteration=start_iteration), log_path)
    # CPU staging tensors stay pinned across all optimizer updates.
    pin = device.type == "cuda"
    train_cpu = [torch.empty(shape, dtype=dtype, pin_memory=pin) for shape, dtype in (
        ((args.batch_size,172),torch.float16), ((args.batch_size,180),torch.float16),
        ((args.batch_size,180),torch.uint8), ((args.batch_size,),torch.uint8))]
    train_np = [t.numpy() for t in train_cpu]
    train_gpu = [torch.empty_like(t, device=device) for t in train_cpu]
    with Batch(args.envs, threads=args.threads, seed=args.seed) as env, Search(env, search_config, args.threads) as search:
        evaluator = Evaluator(model, args.envs, device, not args.no_cuda_graph, not args.fp32)
        # Graph warmup must not change the resumed augmentation RNG sequence.
        if payload:
            torch.set_rng_state(payload["torch_rng"])
            if pin and payload["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        if not args.resume:
            save_checkpoint(args.output/"initial.pt", dict(engine_version=ENGINE_VERSION, architecture=model.architecture,
                            model=model.state_dict(), iteration=0, search=asdict(search_config)))
        for iteration in range(start_iteration, args.iterations):
            model.eval()
            data, metrics = collect(env, search, evaluator, split_seed(args.seed, iteration), args.max_steps,
                                    lambda row: emit(row, log_path))
            metrics["iteration"] = iteration+1
            games += metrics["games"]
            replay.add(data)
            # Only newly completed samples earn training budget, even before warmup.
            credit += len(data[0])*args.sample_reuse
            if not len(data[0]):
                raise RuntimeError("No completed games: increase cutoff; truncations are never draw labels")
            emit(metrics, log_path)
            n_updates = int(credit//args.batch_size) if replay.size>=args.warmup else 0
            if args.updates_per_iteration:
                n_updates = min(n_updates,args.updates_per_iteration)
            model.train()
            losses = torch.zeros(3, device=device)
            begin = time.perf_counter()
            for _ in range(n_updates):
                indices = rng.integers(replay.size, size=args.batch_size)
                for source, dst in zip((replay.obs,replay.policy,replay.masks,replay.outcomes), train_np):
                    np.take(source,indices,axis=0,out=dst,mode="clip")
                for dst, source in zip(train_gpu,train_cpu):
                    dst.copy_(source,non_blocking=pin)
                obs, policy, mask, outcome = train_gpu
                policy = policy.float()
                policy = policy / policy.sum(-1,keepdim=True).clamp_min(1e-12)
                # Independent, uniform factory permutation for EACH sampled position.
                permutations = torch.rand((args.batch_size,5),device=device).argsort(-1)
                obs, policy, mask = permute_factories(obs.float(),policy,mask,permutations)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=pin and not args.fp32,cache_enabled=False):
                    total, pl, vl = loss_for_batch(model,obs,policy,mask,outcome)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0, error_if_nonfinite=True)
                optimizer.step()
                losses += torch.stack((total.detach(),pl.detach(),vl.detach()))
                updates += 1; credit -= args.batch_size
                # Staging buffers cannot be rewritten until H2D completed.
                if pin:
                    torch.cuda.current_stream(device).synchronize()
            means = (losses/max(1,n_updates)).cpu().tolist()
            emit(dict(event="train",iteration=iteration+1,updates=n_updates,total_updates=updates,
                      replay_size=replay.size,loss=means[0],policy_loss=means[1],value_loss=means[2],
                      seconds=time.perf_counter()-begin),log_path)
            state = dict(engine_version=ENGINE_VERSION, architecture=model.architecture,
                         args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                         search=asdict(search_config),model=model.state_dict(),optimizer=optimizer.state_dict(),
                         iteration=iteration+1,updates=updates,games=games,credit=credit,replay=replay.state(),
                         numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if pin else None)
            save_checkpoint(args.output/"latest.pt",state)
            # Small immutable actor snapshot for tournaments; full replay only in latest.pt.
            save_checkpoint(args.output/f"actor-{iteration+1:06d}.pt",{k:state[k] for k in (
                "engine_version","architecture","search","model","iteration")})
            emit(dict(event="checkpoint",iteration=iteration+1,path=str(args.output/"latest.pt")),log_path)


if __name__ == "__main__":
    main()
