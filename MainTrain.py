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
import torch
import pandas as pd
from model import ViTSmallCIFAR
from optimizer import *


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
    
    #TODO: re-adjust when sending to Mark.
    # Pick first 100 indices
    train_ds = torch.utils.data.Subset(train_ds, list(range(100)))
    test_ds = torch.utils.data.Subset(test_ds, list(range(100)))


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# ---------------------------
# Train / Eval
# ---------------------------

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
                 device: torch.device, num_workers: int = 4) -> Dict[str, Any]:
    train_loader, test_loader = build_dataloaders(cfg.batch_size, num_workers)
    
    model = ViTSmallCIFAR(
        num_classes=10, img_size=32, patch_size=4,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=cfg.drop_path_rate
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    total_steps = epochs * math.ceil(50000 / cfg.batch_size)
    warmup_steps = warmup_epochs * math.ceil(50000 / cfg.batch_size)
    scheduler = WarmupCosineLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=min_lr)
    training_log = []
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
        update_str = f"epoch {epoch+1:03d}/{epochs} | loss {train_loss:.4f} | train_acc {train_acc*100:5.2f}% | test_acc {test_acc*100:5.2f}%"
        training_log.append([cfg.lr, cfg.weight_decay, cfg.batch_size, cfg.drop_path_rate, train_loss, train_acc, test_acc])
        print(update_str)

    return {
        "config": cfg,
        "best_acc": best_acc,
        "training_log":training_log
    }

# ---------------------------
# Grid Search
# ---------------------------

def run_one_hypertune(seed):
    random.seed(seed)
    parser = argparse.ArgumentParser()
    #TODO: re-adjust parameters when sending to Mark.
    #parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Default grids. Adjust as needed.
    # parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3.3e-4, 1e-3])
    # parser.add_argument("--wds", type=float, nargs="+",default=[0.02, 0.07, 0.15])
    # parser.add_argument("--bss", type=int, nargs="+", default=[64, 128, 256])
    # parser.add_argument("--dprs", type=float, nargs="+", default=[0.00, 0.10, 0.20, 0.40])

    parser.add_argument("--lrs", type=float, nargs="+", default=[3.3e-4])
    parser.add_argument("--wds", type=float, nargs="+",default=[0.07])
    parser.add_argument("--bss", type=int, nargs="+", default=[128])
    parser.add_argument("--dprs", type=float, nargs="+", default=[0.20])

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    search_space = list(itertools.product(args.lrs, args.wds, args.bss, args.dprs))
    print(f"total runs: {len(search_space)}")
    results: List[Dict[str, Any]] = []

    start_all = time.time()
    training_logs = []
    best_training_log = None
    best_acc = float('-inf')
    for i, (lr, wd, bs, dpr) in enumerate(search_space, 1):
        print("\n" + "=" * 64)
        print(f"run {i}/{len(search_space)} | lr={lr} wd={wd} bs={bs} drop_path_rate={dpr}")
        print("=" * 64)
        cfg = RunConfig(lr=lr, weight_decay=wd, batch_size=bs, drop_path_rate=dpr)
        res = run_training(cfg, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                           min_lr=args.min_lr, device=device, num_workers=args.num_workers)
        if res['best_acc'] > best_acc:
            best_training_log = res['training_log']
        results.append({"config": res['config'], "best_acc": res['best_acc']})
    elapsed = time.time() - start_all
    print(f"\nGrid search finished in {elapsed/60:.1f} min\n")

    training_logs = pd.DataFrame(best_training_log)
    training_logs.columns = ['lr', 'wd', 'bs', 'dpr', 'train loss', 'train accuracy', 'test accuracy']
    training_logs.to_csv(f'base_run_with_seed_{seed}.csv')
    #now that training results are written, clear training_log to free memory
    training_logs = []
    # Leaderboard
    results = sorted(results, key=lambda r: r["best_acc"], reverse=True)
    print("Leaderboard (best test accuracy):")
    for rank, r in enumerate(results, 1):
        cfg = r["config"]
        print(f"{rank:2d}) acc={r['best_acc']*100:5.2f}% | lr={cfg.lr} wd={cfg.weight_decay} bs={cfg.batch_size} dpr={cfg.drop_path_rate}")


def main():
    for seed in [4721, 8154, 2398, 6932, 1567]: run_one_hypertune(seed)

if __name__ == "__main__":
    main()
