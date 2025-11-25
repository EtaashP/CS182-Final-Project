from dataclasses import dataclass
# ---------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import datasets, transforms
import torch
import math

@dataclass
class RunConfig:
    lr: float
    weight_decay: float
    batch_size: int
    drop_path_rate: float
    epochs: int = None
    warmup_epochs: int = None

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