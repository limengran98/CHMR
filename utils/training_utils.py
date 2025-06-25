import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import math
import logging

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_logger(name, logfile=None):
    """create a nice logger"""
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
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