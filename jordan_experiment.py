# jordan_experiment.py

# import numpy as np
# import matplotlib.pyplot as plt

# def run_experiment():
#     print("Running Jordan's experiment...")
#     # Example: generate and plot a sine wave
#     x = np.linspace(0, 2 * np.pi, 100)
#     y = np.sin(x)
#     plt.plot(x, y)
#     plt.title("Sine Wave")
#     plt.xlabel("x")
#     plt.ylabel("sin(x)")
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     run_experiment()


def beta2_from_half_life(batch_size: int, seq_len: int, t_half: int) -> float:
    """
    Compute β2 given batch size, sequence length, and desired half-life in tokens.
    β2 = 2^(-(B*T)/t_half)
    """
    return 2 ** (-(batch_size * seq_len) / t_half)

import torch
from torch.optim import SGD, AdamW
try:
    from torch.optim import Adafactor  # if installed via huggingface/transformers
except ImportError:
    Adafactor = None

def make_optimizer(model, opt_name: str, lr: float, weight_decay: float,
                   batch_size: int, seq_len: int, t_half: int = 10_000_000):
    if opt_name.lower() == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == "adam":
        beta1 = 0.9
        beta2 = beta2_from_half_life(batch_size, seq_len, t_half)
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                     betas=(beta1, beta2))
    elif opt_name.lower() == "adafactor" and Adafactor is not None:
        return Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay,
                         scale_parameter=True, relative_step=False)
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")
    

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += (logits.argmax(1) == targets).float().sum().item()
        n += bs
    return total_loss / n, total_acc / n


from dataclasses import dataclass
@dataclass
class RunConfig:
    optimizer: str   # "sgd", "adam", "adafactor"
    lr: float
    weight_decay: float
    batch_size: int
    seq_len: int
    drop_path_rate: float
    t_half: int = 10_000_000  # token half-life for β2 scaling











import math
import time
import itertools
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import datasets, transforms

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

# Warmup + Cosine LR (step per batch)
class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=1e-5, last_epoch=-1):
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * float(step) / float(max(1, self.warmup_steps))
            else:
                progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            lrs.append(lr)
        return lrs

# ---------------------------
# Stochastic depth (DropPath)
# ---------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ---------------------------
# ViT-Small for 32x32, patch=4
# ---------------------------

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # B, N, 3C
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # B, heads, N, head_dim
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size  # 8
        self.num_patches = self.grid * self.grid  # 64
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, C, 8, 8
        x = x.flatten(2).transpose(1, 2)  # B, 64, C
        return x

class ViTSmallCIFAR(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch_size=4,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 cls_norm=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_norm = cls_norm
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B, 64, C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # B, 65, C
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.cls_norm:
            x = self.norm(x)
        cls_tok = x[:, 0]
        logits = self.head(cls_tok)
        return logits

# ---------------------------
# Data
# ---------------------------

def build_dataloaders(batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, transform=train_tf, download=True)
    test_ds = datasets.CIFAR10(root="./data", train=False, transform=test_tf, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# ---------------------------
# Paper heuristics
# ---------------------------

def beta2_from_half_life(batch_size: int, seq_len: int, t_half_tokens: int) -> float:
    """
    β2 = 2^(-(B * T) / t_half_tokens)
    """
    return 2 ** (-(batch_size * seq_len) / float(t_half_tokens))

# Optional Adafactor import (from transformers). If not available, we skip it.
def _try_import_adafactor():
    try:
        import importlib
        mod = importlib.import_module("transformers")
        return getattr(mod, "Adafactor", None)
    except Exception:
        return None

# ---------------------------
# Train / Eval
# ---------------------------

@dataclass
class RunConfig:
    optimizer: str         # "sgd", "adam", "adamw", "adafactor"
    lr: float
    weight_decay: float
    batch_size: int
    drop_path_rate: float
    seq_len: int = 65      # tokens per sample (CLS + 64 patches) for ViT; used for β2 scaling
    t2_half_tokens: int = 10000000  # desired half-life in tokens for β2 scaling

def make_optimizer(model, cfg: RunConfig):
    opt = cfg.optimizer.lower()
    if opt == "sgd":
        # vanilla SGD, no momentum per paper's small-batch findings
        return SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif opt in ("adam", "adamw"):
        beta1 = 0.9
        beta2 = beta2_from_half_life(cfg.batch_size, cfg.seq_len, cfg.t2_half_tokens)
        # Use AdamW to match common practice; set eps small and keep weight decay configurable
        return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                     betas=(beta1, beta2), eps=1e-8)
    elif opt == "adafactor":
        Adafactor = _try_import_adafactor()
        if Adafactor is None:
            raise ValueError("Adafactor not available. Install transformers or remove 'adafactor' from the grid.")
        # Adafactor has its own decay strategy; weight_decay applies if not zero
        # Use fixed step (relative_step=False) to align with LR grid
        return Adafactor(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                         scale_parameter=True, relative_step=False, warmup_init=False)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # per-batch scheduler step

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), targets) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_acc, n = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        bs = images.size(0)
        total_acc += accuracy(logits, targets) * bs
        n += bs
    return total_acc / n

def run_training(cfg: RunConfig, epochs: int, warmup_epochs: int, min_lr: float,
                 device: torch.device, num_workers: int = 4) -> Dict[str, Any]:
    train_loader, test_loader = build_dataloaders(cfg.batch_size, num_workers)
    model = ViTSmallCIFAR(
        num_classes=10, img_size=32, patch_size=4,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=cfg.drop_path_rate
    ).to(device)

    optimizer = make_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    steps_per_epoch = math.ceil(50000 / cfg.batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    scheduler = WarmupCosineLR(optimizer, total_steps=total_steps,
                               warmup_steps=warmup_steps, min_lr=min_lr)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device)
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        print(f"epoch {epoch+1:03d}/{epochs} | loss {train_loss:.4f} | train_acc {train_acc*100:5.2f}% | test_acc {test_acc*100:5.2f}%")

    return {
        "config": cfg,
        "best_acc": best_acc
    }

# ---------------------------
# Grid Search
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)         # deafult=60
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Grids
    parser.add_argument("--optimizers", type=str, nargs="+", default=["sgd", "adamw", "adafactor"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 2e-4, 3e-4, 5e-4, 8e-4])
    parser.add_argument("--wds", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--bss", type=int, nargs="+", default=[64, 128, 192, 256, 384])
    parser.add_argument("--dprs", type=float, nargs="+", default=[0.00, 0.05, 0.10, 0.15])
    parser.add_argument("--t2_halves", type=int, nargs="+", default=[1_000_000, 10_000_000, 100_000_000])

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # If Adafactor is not available, remove it from the grid
    if _try_import_adafactor() is None and "adafactor" in args.optimizers:
        print("Adafactor not available; removing it from the optimizer grid.")
        args.optimizers = [o for o in args.optimizers if o.lower() != "adafactor"]

    search_space = list(itertools.product(args.optimizers, args.lrs, args.wds, args.bss, args.dprs, args.t2_halves))
    print(f"total runs: {len(search_space)}")
    results: List[Dict[str, Any]] = []

    start_all = time.time()
    for i, (opt, lr, wd, bs, dpr, t2h) in enumerate(search_space, 1):
        print("\n" + "=" * 72)
        print(f"run {i}/{len(search_space)} | opt={opt} lr={lr} wd={wd} bs={bs} drop_path={dpr} t2_half={t2h}")
        print("=" * 72)
        cfg = RunConfig(optimizer=opt, lr=lr, weight_decay=wd, batch_size=bs,
                        drop_path_rate=dpr, seq_len=65, t2_half_tokens=t2h)
        res = run_training(cfg, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                           min_lr=args.min_lr, device=device, num_workers=args.num_workers)
        results.append(res)

    elapsed = time.time() - start_all
    print(f"\nGrid search finished in {elapsed/60:.1f} min\n")

    # Leaderboard
    results = sorted(results, key=lambda r: r["best_acc"], reverse=True)
    print("Leaderboard (best test accuracy):")
    for rank, r in enumerate(results, 1):
        cfg = r["config"]
        print(f"{rank:2d}) acc={r['best_acc']*100:5.2f}% | opt={cfg.optimizer} lr={cfg.lr} wd={cfg.weight_decay} bs={cfg.batch_size} dpr={cfg.drop_path_rate} t2_half={cfg.t2_half_tokens}")

if __name__ == "__main__":
    main()
