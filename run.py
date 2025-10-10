import argparse, time
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from model import Model


def mse_autoencode_step(model, x, cu):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        y = model(x, cu)
        loss = F.mse_loss(y, x.detach())
    return loss


def run_one_length(model, optimizer, dim, B, L, steps, warmup, profile_dir="baseline"):
    device = torch.device("cuda")

    cu = torch.arange(0, (B+1)*L, L, dtype=torch.int32, device=device) # [0, L, 2L, ..., B*L]
    x  = torch.randn(B*L, dim, device=device) # [B*L, dim]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = mse_autoencode_step(model, x, cu)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=10, repeat=1),
        on_trace_ready=tensorboard_trace_handler(profile_dir),
        record_shapes=True, profile_memory=True, with_stack=True,
    ) as prof:
        for _ in range(14):
            optimizer.zero_grad(set_to_none=True)
            loss = mse_autoencode_step(model, x, cu)
            loss.backward()
            optimizer.step()
            prof.step()

    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = mse_autoencode_step(model, x, cu)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)

    avg_t = sum(times)/len(times)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    toks = B*L
    print(f"[baseline] L={L:5d} B={B:2d}  time/it={avg_t:.4f}s  tok/s={toks/avg_t:,.0f}  peak={peak_mb:,.1f} MB")
    return {"L": L, "B": B, "avg_time_s": avg_t, "tok_per_s": toks/avg_t, "peak_mem_mb": peak_mb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--ff_dim", type=int, default=4096)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--L_list", type=int, nargs="+", default=[2048, 4096, 8192, 12288, 16384])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile_first", action="store_true")
    ap.add_argument("--profile_dir", default="baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = Model(
        in_dim=args.dim, 
        hidden_dim=args.dim, 
        ff_dim=args.ff_dim,
        num_layers=args.num_layers, 
        head_dim= args.head_dim
    ).to(device)

    model = model.to(memory_format=torch.channels_last).to(dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    results = []
    for i, L in enumerate(args.L_list):
        res = run_one_length(
            model, optimizer, dim=args.dim, B=args.B, L=L,
            steps=args.steps, warmup=args.warmup,
            profile_dir=args.profile_dir
        )
        results.append(res)


if __name__ == "__main__":
    main()
