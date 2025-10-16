import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import math
import logging

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_logger(name=None, console_level=logging.INFO, file_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(f"{name}.log", mode="a")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger



def seed_torch(seed=0):
    print("Seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0 

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
