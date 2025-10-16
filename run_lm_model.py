import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import (
    profile,
    schedule,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from datasets import load_dataset
from transformers import AutoTokenizer

from model import LMModel


class Dataset:
    def __init__(self, tokenizer, seq_len: int = 2048, batch_size: int = 1):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", streaming=True
        )

    def _iter_tokens(self):
        for ex in self.ds:
            text = ex["text"]
            if not text or text.strip() == "":
                continue
            out = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            for t in out:
                yield t
        eos = self.tokenizer.eos_token_id
        while True:
            yield eos

    def __iter__(self):
        token_iter = self._iter_tokens()
        device = torch.device("cuda")
        while True:
            seqs = []
            for _ in range(self.batch_size):
                ids = [next(token_iter) for _ in range(self.seq_len)]
                seqs.append(torch.tensor(ids, dtype=torch.long))
            packed = torch.cat(seqs, dim=0).to(device)
            cu = torch.arange(
                0,
                (self.batch_size + 1) * self.seq_len,
                self.seq_len,
                dtype=torch.int32,
                device=device,
            )
    
            targets = packed.clone()
            for i in range(self.batch_size):
                start = i * self.seq_len
                end = (i + 1) * self.seq_len
                targets[start : end - 1] = packed[start + 1 : end]
                targets[end - 1] = self.tokenizer.eos_token_id

            yield packed, targets, cu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="checkpoint")
    ap.add_argument("--tokenizer_name", type=str, default="gpt2")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--ff_dim", type=int, default=4096)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument(
        "--L_list", type=int, nargs="+", default=[2048, 4096, 8192]
    )  # default=[2048,4096,8192,12288,16384]
    ap.add_argument("--profile_dir", type=str, default="checkpoint")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_rng_state", type=bool, default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    for L in args.L_list:
        writer = SummaryWriter(log_dir=f"{args.profile_dir}/L{L}_B{args.B}")

        model = (
            LMModel(
                vocab_size=tokenizer.vocab_size,
                dim=args.dim,
                ff_dim=args.ff_dim,
                num_layers=args.num_layers,
                head_dim=args.head_dim,
                checkpointing=args.method == "checkpoint",
                use_rng_state=args.use_rng_state,
            )
            .to(device)
            .to(dtype=torch.float32)
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_loader = iter(Dataset(tokenizer, L, args.B))

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=2, warmup=2, active=10, repeat=1),
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()

            for step in range(1, 500):
                input_ids, targets, cu = next(train_loader)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    logits = model(input_ids, cu)
                    loss = F.cross_entropy(logits.float(), targets, reduction="mean")

                loss.backward()
                optimizer.step()
                prof.step()

                writer.add_scalar("Loss/train", loss.item(), step)
                # print(step, "loss", loss.item())
            
            writer.close()

        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(
            f"{args.profile_dir} L={L} B={args.B} peak={peak_mb:.1f} MB, time={time.perf_counter() - start:.2f}s"
        )


if __name__ == "__main__":
    main()
