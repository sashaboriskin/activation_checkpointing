import argparse
import time
import torch
import torch.nn.functional as F
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from model import Model, ModelCheckpoint


def mse_autoencode_step(model, x, cu):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        y = model(x, cu)
        loss = F.mse_loss(y, x.detach())
    return loss


def run_one_length(model, optimizer, dim, B, L, profile_dir="baseline"):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=10, repeat=1),
        on_trace_ready=tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        cu = torch.arange(
            0, (B + 1) * L, L, dtype=torch.int32, device=device
        )  # [0, L, 2L, ..., B*L]
        x = torch.randn(
            B * L, dim, device=device
        )  # [B*L, dim] because of varlen_flash_attention

        for _ in range(14):
            optimizer.zero_grad(set_to_none=True)
            loss = mse_autoencode_step(model, x, cu)
            loss.backward()
            optimizer.step()
            prof.step()

    print(
        f"{profile_dir} L={L} B={B} peak={torch.cuda.max_memory_allocated() / (1024**2)} MB time={time.perf_counter()-start_time} S"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="checkpoint")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--ff_dim", type=int, default=4096)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument(
        "--L_list", type=int, nargs="+", default=[2048, 4096, 8192, 12288, 16384]
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile_dir", default="checkpoint")
    ap.add_argument("--use_rng_state", type=bool, default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    if args.method == "checkpoint":
        model = ModelCheckpoint(
            in_dim=args.dim,
            hidden_dim=args.dim,
            ff_dim=args.ff_dim,
            num_layers=args.num_layers,
            head_dim=args.head_dim,
            use_rng_state=args.use_rng_state,
        ).to(device)

    elif args.method == "baseline":
        model = Model(
            in_dim=args.dim,
            hidden_dim=args.dim,
            ff_dim=args.ff_dim,
            num_layers=args.num_layers,
            head_dim=args.head_dim,
        ).to(device)

    model = model.to(dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for i, L in enumerate(args.L_list):
        res = run_one_length(
            model,
            optimizer,
            dim=args.dim,
            B=args.B,
            L=L,
            profile_dir=args.profile_dir,
        )


if __name__ == "__main__":
    main()
