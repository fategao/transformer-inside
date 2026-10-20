"""Train the small character-level Transformer used by the site.

The checkpoint is intentionally small enough to run in a browser: four
decoder blocks, four heads, 128 embedding dimensions, context 64.

Usage:
    python training/train.py
    python training/train.py --steps 3000 --device cpu
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "data"
OUT_DIR = ROOT / "public" / "model"
CORPUS = DATA_DIR / "tinyshakespeare.txt"


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    n_ff: int = 512


def ensure_corpus() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CORPUS.exists():
        return CORPUS.read_text(encoding="utf-8")

    attempts = [
        "https://api.github.com/repos/karpathy/char-rnn/contents/data/tinyshakespeare/input.txt",
        "https://api.github.com/repos/karpathy/char-rnn/git/blobs/7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9",
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    ]
    last_error: Exception | None = None
    for url in attempts:
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                payload = response.read()
            if "api.github.com" in url:
                meta = json.loads(payload)
                if not meta.get("content"):
                    raise ValueError("GitHub API returned no inline content")
                text = base64.b64decode(meta["content"]).decode("utf-8")
            else:
                text = payload.decode("utf-8")
            if len(text) < 100_000:
                raise ValueError(f"corpus too small: {len(text)} chars")
            CORPUS.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:  # noqa: BLE001 - report the last failure
            last_error = exc
    # Some networks block Python's TLS stack but allow curl. Keep the same
    # public artifact and verify its size before accepting it.
    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "--max-time",
                "60",
                "-s",
                "https://api.github.com/repos/karpathy/char-rnn/git/blobs/7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9",
                "-o",
                str(DATA_DIR / "blob.json"),
            ],
            check=True,
        )
        meta = json.loads((DATA_DIR / "blob.json").read_text(encoding="utf-8"))
        text = base64.b64decode(meta["content"]).decode("utf-8")
        if len(text) > 100_000:
            CORPUS.write_text(text, encoding="utf-8")
            return text
    except Exception as exc:  # noqa: BLE001
        last_error = exc
    raise RuntimeError(f"could not download Tiny Shakespeare: {last_error}")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq, seq, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq, channels)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.fc = nn.Linear(cfg.n_embd, cfg.n_ff)
        self.mlp_proj = nn.Linear(cfg.n_ff, cfg.n_embd)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        hidden = self.act(self.fc(self.ln2(x)))
        x = x + self.mlp_proj(hidden)
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        batch, seq = idx.shape
        positions = torch.arange(seq, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)[None, :, :]
        trace: list[dict] = []
        for block in self.blocks:
            residual = x
            normed = block.ln1(x)
            qkv = block.attn.qkv(normed)
            channels = self.cfg.n_embd
            q, k, v = qkv.split(channels, dim=2)
            q = q.view(batch, seq, self.cfg.n_head, -1).transpose(1, 2)
            k = k.view(batch, seq, self.cfg.n_head, -1).transpose(1, 2)
            v = v.view(batch, seq, self.cfg.n_head, -1).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            mask = torch.tril(torch.ones(seq, seq, device=idx.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            attn_out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq, channels)
            attn_out = block.attn.proj(attn_out)
            x = residual + attn_out
            mlp_in = block.ln2(x)
            mlp_hidden = block.act(block.fc(mlp_in))
            mlp_out = block.mlp_proj(mlp_hidden)
            x = x + mlp_out
            trace.append(
                {
                    "attention": weights.detach(),
                    "mlp_hidden": mlp_hidden.detach(),
                    "residual_in": residual.detach(),
                    "residual_out": x.detach(),
                }
            )
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, trace

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def batch_from(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: TinyGPT, splits: dict[str, torch.Tensor], batch_size: int, block_size: int, device, iters=40):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = batch_from(data, batch_size, block_size, device)
            logits, _ = model(x)
            losses[i] = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        out[name] = losses.mean().item()
    model.train()
    return out


def export_weights(model: TinyGPT, cfg: ModelConfig, vocab: list[str], metrics: dict, sample: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tensors: list[tuple[str, torch.Tensor]] = [
        ("tok_emb.weight", model.tok_emb.weight),
        ("pos_emb.weight", model.pos_emb.weight),
    ]
    for index, block in enumerate(model.blocks):
        prefix = f"blocks.{index}"
        tensors.extend(
            [
                (f"{prefix}.ln1.weight", block.ln1.weight),
                (f"{prefix}.ln1.bias", block.ln1.bias),
                (f"{prefix}.attn.qkv.weight", block.attn.qkv.weight),
                (f"{prefix}.attn.qkv.bias", block.attn.qkv.bias),
                (f"{prefix}.attn.proj.weight", block.attn.proj.weight),
                (f"{prefix}.attn.proj.bias", block.attn.proj.bias),
                (f"{prefix}.ln2.weight", block.ln2.weight),
                (f"{prefix}.ln2.bias", block.ln2.bias),
                (f"{prefix}.fc.weight", block.fc.weight),
                (f"{prefix}.fc.bias", block.fc.bias),
                (f"{prefix}.mlp_proj.weight", block.mlp_proj.weight),
                (f"{prefix}.mlp_proj.bias", block.mlp_proj.bias),
            ]
        )
    tensors.extend([("ln_f.weight", model.ln_f.weight), ("ln_f.bias", model.ln_f.bias)])

    arrays = []
    manifest = []
    offset = 0
    for name, tensor in tensors:
        array = tensor.detach().cpu().float().contiguous().numpy()
        manifest.append(
            {
                "name": name,
                "shape": list(array.shape),
                "offset": offset,
                "count": int(array.size),
            }
        )
        arrays.append(array.ravel())
        offset += int(array.size)
    flat = np.concatenate(arrays).astype("<f4")
    weight_path = OUT_DIR / "weights.bin"
    flat.tofile(weight_path)
    digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()

    model_json = {
        "name": "Tiny Shakespeare 4L",
        "config": {
            "vocabSize": cfg.vocab_size,
            "contextLength": cfg.context_length,
            "nLayer": cfg.n_layer,
            "nHead": cfg.n_head,
            "dModel": cfg.n_embd,
            "dFF": cfg.n_ff,
        },
        "vocab": vocab,
        "tokenizer": "character",
        "weights": {
            "file": "weights.bin",
            "dtype": "float32",
            "sha256": digest,
            "tensors": manifest,
        },
        "training": metrics,
        "sample": sample,
    }
    (OUT_DIR / "model.json").write_text(json.dumps(model_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported {offset:,} parameters to {weight_path} ({weight_path.stat().st_size / 1e6:.2f} MB)")


@torch.no_grad()
def export_reference(model: TinyGPT, vocab: list[str], prompt: str) -> None:
    stoi = {ch: i for i, ch in enumerate(vocab)}
    ids = [stoi[ch] for ch in prompt]
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits, trace = model(idx)
    reference = {
        "prompt": prompt,
        "tokenIds": ids,
        "tokens": list(prompt),
        "logits": [round(float(v), 7) for v in logits[0, -1].cpu()],
        "attention": [
            [[[round(float(v), 7) for v in row] for row in head.cpu()] for head in layer["attention"][0]]
            for layer in trace
        ],
    }
    (OUT_DIR / "reference.json").write_text(json.dumps(reference, ensure_ascii=False), encoding="utf-8")
    print(f"exported reference for prompt {prompt!r} ({len(ids)} tokens)")


@torch.no_grad()
def generate(model: TinyGPT, stoi: dict[str, int], itos: list[str], prompt: str, max_new_tokens=240, temperature=0.8):
    ids = [stoi[ch] for ch in prompt]
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.context_length :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    text = ensure_corpus()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = list(vocab)
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    splits = {"train": data[:split], "val": data[split:]}

    cfg = ModelConfig(vocab_size=len(vocab), context_length=args.block_size)
    model = TinyGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"device={device} vocab={len(vocab)} params={model.num_parameters():,}")

    started = time.time()
    for step in range(1, args.steps + 1):
        if step <= 100:
            lr = args.lr * step / 100
        else:
            progress = (step - 100) / max(1, args.steps - 100)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = batch_from(splits["train"], args.batch_size, args.block_size, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:5d} loss {loss.item():.4f} lr {lr:.2e} elapsed {time.time() - started:.0f}s")

    losses = estimate_loss(model, splits, args.batch_size, args.block_size, device)
    torch.save(model.state_dict(), ROOT / "training" / "checkpoint.pt")
    sample = generate(model, stoi, itos, "ROMEO: ", max_new_tokens=240, temperature=0.8)
    metrics = {
        "steps": args.steps,
        "batchSize": args.batch_size,
        "blockSize": args.block_size,
        "learningRate": args.lr,
        "seed": args.seed,
        "device": str(device),
        "trainLoss": round(losses["train"], 5),
        "valLoss": round(losses["val"], 5),
        "parameterCount": model.num_parameters(),
        "corpusCharacters": len(text),
        "corpusSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "trainingSeconds": round(time.time() - started, 1),
    }
    export_weights(model, cfg, itos, metrics, sample)
    export_reference(model, itos, "ROMEO: ")
    (OUT_DIR / "training-log.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "sample.txt").write_text(sample, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
