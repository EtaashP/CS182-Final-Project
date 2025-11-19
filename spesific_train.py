# grid_vit_cifar10.py
# ViT-Small (embed_dim=384, depth=12, heads=6, mlp_ratio=4) with patch size 4 for CIFAR-10.
# Grid search over: learning rate, weight decay, batch size, stochastic depth rate.

import math
import time
import itertools
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
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

# Warmup + Cosine LR
class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=1e-5, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
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

        # stochastic depth decay rule
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
    # Standard CIFAR-10 augments
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
# Train / Eval
# ---------------------------

@dataclass
class RunConfig:
    lr: float
    weight_decay: float
    batch_size: int
    drop_path_rate: float

def train_one_epoch(model, loader, optimizer, scaler, device, mixup_alpha=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()

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

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), targets) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_acc = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        bs = images.size(0)
        total_acc += accuracy(logits, targets) * bs
        n += bs
    return total_acc / n

def run_training(cfg: RunConfig, epochs: int, warmup_epochs: int, min_lr: float,
                 device: torch.device, num_workers: int = 4,
                 log_fn=print, initial_state=None) -> Dict[str, Any]:
    train_loader, test_loader = build_dataloaders(cfg.batch_size, num_workers)
    model = ViTSmallCIFAR(
        num_classes=10, img_size=32, patch_size=4,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=cfg.drop_path_rate
    ).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    total_steps = epochs * math.ceil(50000 / cfg.batch_size)
    warmup_steps = warmup_epochs * math.ceil(50000 / cfg.batch_size)
    scheduler = WarmupCosineLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=min_lr)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)

        # step LR scheduler per iteration equivalently by calling .step() repeated times.
        # Here we approximate by stepping once per epoch across epoch-length steps:
        # do it properly: step per batch in train loop would be ideal.
        # Quick fix: recompute steps done and set last_epoch accordingly.
        scheduler.last_epoch = (epoch + 1) * math.ceil(50000 / cfg.batch_size) - 1
        scheduler.step()

        log_fn(f"epoch {epoch+1:03d}/{epochs} | loss {train_loss:.4f} | train_acc {train_acc*100:5.2f}% | test_acc {test_acc*100:5.2f}%")

    return {
        "config": cfg,
        "best_acc": best_acc,
        "final_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}
    }

# ---------------------------
# Grid Search
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase_epochs", type=int, default=None,
                        help="Override epochs per phase (defaults to total/3).")

    # Baseline (phase 1) hyperparameters.
    parser.add_argument("--baseline_lrs", type=float, nargs="+", default=[3.3e-4])
    parser.add_argument("--baseline_wds", type=float, nargs="+", default=[0.07])
    parser.add_argument("--baseline_bss", type=int, nargs="+", default=[128])
    parser.add_argument("--baseline_dprs", type=float, nargs="+", default=[0.20])

    # Grid (phases 2 & 3) hyperparameters.
    parser.add_argument("--grid_lrs", type=float, nargs="+", default=[1e-4, 3.3e-4, 1e-3])
    parser.add_argument("--grid_wds", type=float, nargs="+", default=[0.02, 0.07, 0.15])
    parser.add_argument("--grid_bss", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--grid_dprs", type=float, nargs="+", default=[0.10, 0.20, 0.40])
    parser.add_argument("--results_out", type=str, default="phase_results.txt",
                        help="Path to save printed results.")

    args = parser.parse_args()
    set_seed(args.seed)

    log_lines: List[str] = []
    def log(msg: str = ""):
        print(msg)
        log_lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    baseline_space = list(itertools.product(args.baseline_lrs, args.baseline_wds,
                                            args.baseline_bss, args.baseline_dprs))
    grid_space = list(itertools.product(args.grid_lrs, args.grid_wds,
                                        args.grid_bss, args.grid_dprs))

    phase_epochs = args.phase_epochs or args.epochs // 3
    phase_lengths = [int(phase_epochs/2), phase_epochs, max(1, args.epochs - int(phase_epochs/2)- phase_epochs)]

    def run_phase(search_space, epochs, label, initial_state=None):
        log(f"\n{label}: total runs={len(search_space)}, epochs per run={epochs}")
        results: List[Dict[str, Any]] = []
        start_phase = time.time()
        for i, (lr, wd, bs, dpr) in enumerate(search_space, 1):
            log("\n" + "=" * 64)
            log(f"{label} run {i}/{len(search_space)} | lr={lr} wd={wd} bs={bs} drop_path_rate={dpr}")
            log("=" * 64)
            cfg = RunConfig(lr=lr, weight_decay=wd, batch_size=bs, drop_path_rate=dpr)
            res = run_training(cfg, epochs=epochs, warmup_epochs=args.warmup_epochs,
                               min_lr=args.min_lr, device=device, num_workers=args.num_workers,
                               log_fn=log, initial_state=initial_state)
            results.append(res)

        elapsed_phase = time.time() - start_phase
        log(f"\n{label} finished in {elapsed_phase/60:.1f} min\n")

        sorted_results = sorted(results, key=lambda r: r["best_acc"], reverse=True)
        if sorted_results:
            log(f"{label} leaderboard (best test accuracy):")
            for rank, r in enumerate(sorted_results, 1):
                cfg = r["config"]
                log(f"{rank:2d}) acc={r['best_acc']*100:5.2f}% | lr={cfg.lr} wd={cfg.weight_decay} bs={cfg.batch_size} dpr={cfg.drop_path_rate}")
        else:
            log(f"{label} skipped: no configs found.")
        return sorted_results

    phase_results = []
    phase1 = run_phase(baseline_space, phase_lengths[0], "Phase 1 (baseline)")
    phase1_best_state = phase1[0]["final_state"] if phase1 else None
    phase_results.append(phase1)

    phase2 = run_phase(grid_space, phase_lengths[1], "Phase 2 (grid search)", initial_state=phase1_best_state)
    phase2_best_state = phase2[0]["final_state"] if phase2 else None
    phase_results.append(phase2)

    phase3 = run_phase(grid_space, phase_lengths[2], "Phase 3 (grid search)", initial_state=phase2_best_state)
    phase_results.append(phase3)

    log("\nSummary of best configs per phase:")
    for label, results in zip(
        ["Phase 1 (baseline)", "Phase 2 (grid search)", "Phase 3 (grid search)"],
        phase_results
    ):
        if results:
            best = results[0]["config"]
            acc = results[0]["best_acc"] * 100
            log(f"- {label}: {acc:5.2f}% | lr={best.lr} wd={best.weight_decay} bs={best.batch_size} dpr={best.drop_path_rate}")
        else:
            log(f"- {label}: no runs executed")

    with open(args.results_out, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\nSaved results to {args.results_out}")

if __name__ == "__main__":
    main()
