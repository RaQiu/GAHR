import os
import random
import numpy as np
import torch
import torch.distributed as dist


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_using_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    if not is_using_distributed():
        return True
    return dist.get_rank() == 0


def get_rank():
    if not is_using_distributed():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_using_distributed():
        return 1
    return dist.get_world_size()
