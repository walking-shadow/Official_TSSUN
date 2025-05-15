import math
import torch
from functools import partial
import numpy as np

# step scheduler
def fn_LinearWarmup(warmup_steps, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def Scheduler_LinearWarmup(warmup_steps):
    return partial(fn_LinearWarmup, warmup_steps)


def fn_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min, step):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:  # cosine learning rate schedule
        multipler = 0.5 * (math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1)
        return max(multipler, multipler_min)

def Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multipler_min):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps, multipler_min)

def fn_LinearWarmup_CosineDecay_BSQ(warmup_steps, lr_min, lr_max, lr_start, max_decay_steps, step):

    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
        return lr
    else:
        t = (step - warmup_steps) / (max_decay_steps - warmup_steps)
        t = min(t, 1.0)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (
                1 + np.cos(t * np.pi))
        return lr

def Scheduler_LinearWarmup_CosineDecay_BSQ(warmup_steps, lr_min, lr_max, lr_start, max_decay_steps,):
    return partial(fn_LinearWarmup_CosineDecay_BSQ, warmup_steps, lr_min, lr_max, lr_start, max_decay_steps,)