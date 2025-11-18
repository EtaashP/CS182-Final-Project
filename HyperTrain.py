# HyperTrain.py
# Hypergradient descent on lr and weight decay for ViT-Small on CIFAR-10.

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
import higher

from MainTrain import ViTSmallCIFAR, build_dataloaders, accuracy


# =============================
# 1. Hyperparameters as tensors
# =============================

class HyperParams:
    def __init__(self, init_lr: float = 3e-4, init_wd: float = 0.05, device: torch.device | str = "cuda"):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.log_lr = torch.tensor(math.log(init_lr), requires_grad=True, device=self.device)
        self.log_wd = torch.tensor(math.log(init_wd), requires_grad=True, device=self.device)

        # Optimizer that updates the hyperparameters
        self.hyper_opt = torch.optim.Adam([self.log_lr, self.log_wd], lr=1e-3)

    def lr(self) -> float:
        return float(torch.exp(self.log_lr).item())

    def wd(self) -> float:
        return float(torch.exp(self.log_wd).item())


# ===========================================
# 2. Hypergradient inner-loop using `higher`
# ===========================================

def hyper_step(
    model: nn.Module,
    hyper: HyperParams,
    train_loader,
    val_loader,
    device: torch.device,
    T_inner: int = 3,
    train_batches: int = 2,
    val_batches: int = 2,
) -> float:
    """
    One hypergradient update:
      - Unroll a few inner SGD steps on train data
      - Evaluate val loss at the end
      - Backprop through inner steps into hyperparameters
    """

    criterion = nn.CrossEntropyLoss()

    base_opt = torch.optim.SGD(model.parameters(), lr=hyper.lr())

    with higher.innerloop_ctx(model, base_opt, copy_initial_weights=False) as (fmodel, diffopt):

        train_iter = iter(train_loader)

        # Inner optimization on training data
        for _ in range(T_inner):
            for _ in range(train_batches):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x, y = x.to(device), y.to(device)

                logits = fmodel(x)
                loss = criterion(logits, y)

                # manual L2 weight decay so gradients flow into log_wd
                wd = torch.exp(hyper.log_wd)
                l2_reg = 0.0
                for p in fmodel.parameters():
                    l2_reg = l2_reg + (p ** 2).sum()
                loss = loss + 0.5 * wd * l2_reg

                diffopt.zero_grad()
                loss.backward()
                diffopt.step()

        # Validation loss after inner-loop
        val_iter = iter(val_loader)
        val_loss = 0.0
        for _ in range(val_batches):
            try:
                x_val, y_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_val, y_val = next(val_iter)

            x_val, y_val = x_val.to(device), y_val.to(device)
            logits_val = fmodel(x_val)
            val_loss = val_loss + criterion(logits_val, y_val)

        val_loss = val_loss / val_batches

    # Backprop into hyperparameters
    hyper.hyper_opt.zero_grad()
    val_loss.backward()
    hyper.hyper_opt.step()

    return float(val_loss.item())


# =============================
# 3. Simple evaluation helper
# =============================

@torch.no_grad()
def evaluate_simple(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total += images.size(0)
        correct += (preds == targets).sum().item()
    return correct / total


# =============================
# 4. Main training loop with HGD
# =============================

def train_with_hypergrad(
    epochs: int = 30,
    batch_size: int = 256,
    hyper_every: int = 5,
    T_inner: int = 3,
    train_batches: int = 2,
    val_batches: int = 2,
) -> Tuple[nn.Module, HyperParams, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Data
    train_loader, val_loader = build_dataloaders(batch_size=batch_size, num_workers=4)

    # Model
    model = ViTSmallCIFAR(
        num_classes=10,
        img_size=32,
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,  # pick a reasonable fixed dpr for HGD run
    ).to(device)

    # Hyperparameters to adapt
    hyper = HyperParams(init_lr=3e-4, init_wd=0.05, device=device)

    # Main optimizer; lr and wd will be overwritten from hyper each epoch
    optimizer = AdamW(
        model.parameters(),
        lr=hyper.lr(),
        weight_decay=hyper.wd(),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()

        # update optimizer learning rate & weight decay from hyperparameters
        for g in optimizer.param_groups:
            g["lr"] = hyper.lr()
            g["weight_decay"] = hyper.wd()

        total_loss = 0.0
        total_correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total += bs

        train_loss = total_loss / total
        train_acc = total_correct / total

        val_acc = evaluate_simple(model, val_loader, device)
        best_acc = max(best_acc, val_acc)

        print(
            f"Epoch {epoch+1:03d}/{epochs} "
            f"| train_loss {train_loss:.4f} "
            f"| train_acc {train_acc*100:5.2f}% "
            f"| val_acc {val_acc*100:5.2f}% "
            f"| lr {hyper.lr():.2e} "
            f"| wd {hyper.wd():.2e}"
        )

        # Periodic hypergradient update
        if (epoch + 1) % hyper_every == 0:
            val_loss = hyper_step(
                model,
                hyper,
                train_loader,
                val_loader,
                device=device,
                T_inner=T_inner,
                train_batches=train_batches,
                val_batches=val_batches,
            )
            print(
                f"    [hyper] val_loss={val_loss:.4f} "
                f"-> lr {hyper.lr():.2e}, wd {hyper.wd():.2e}"
            )

    print(f"\nBest val_acc: {best_acc*100:.2f}%")
    return model, hyper, best_acc


if __name__ == "__main__":
    # You can tweak epochs / batch_size here if runs are too slow.
    train_with_hypergrad()
