import math
import os
from collections import deque

import torch
import numpy as np
import torch_xla.core.xla_model as xm


def get_warmup_cosine_scheduler(optimizer, warmup_iteration, max_iteration):
    def _warmup_cosine(step):
        if step < warmup_iteration:
            lr_ratio = step * 1.0 / warmup_iteration
        else:
            where = (step - warmup_iteration) * 1.0 / (max_iteration - warmup_iteration)
            lr_ratio = 0.5 * (1 + math.cos(math.pi * where))

        return lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine)


def save_ckpt(ckpt_path, model, optimizer, lr_scheduler, master_only=True):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    xm.save(ckpt, ckpt_path, master_only=master_only, global_master=True)
    print(f"checkpoint saved to {ckpt_path}\n", end="")


def load_ckpt(ckpt_path, model, optimizer, lr_scheduler):
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    print(f"resumed from checkpoint {ckpt_path}\n", end="")


class FakeImageNetDataset:
    def __init__(self, image_size, length):
        self.image_size = image_size
        self.length = length

    def __getitem__(self, idx):
        return (torch.zeros(3, self.image_size, self.image_size), 0)

    def __len__(self):
        return self.length


# adapted from
# https://github.com/facebookresearch/mmf/blob/master/mmf/common/meter.py
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.averaged_value_deque = deque(maxlen=self.window_size)
        self.batch_sizes = deque(maxlen=self.window_size)
        self.total_samples = 0
        self.total = 0.0
        self.count = 0

    def update(self, value, batch_size):
        self.deque.append(value * batch_size)
        self.averaged_value_deque.append(value)
        self.batch_sizes.append(batch_size)

        self.count += 1
        self.total_samples += batch_size
        self.total += value * batch_size

    @property
    def median(self):
        d = np.median(list(self.averaged_value_deque))
        return d

    @property
    def avg(self):
        d = np.sum(list(self.deque))
        s = np.sum(list(self.batch_sizes))
        return d / s

    @property
    def global_avg(self):
        return self.total / self.total_samples

    def get_latest(self):
        return self.averaged_value_deque[-1]
