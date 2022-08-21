import random
import torch
import numpy


def fix_random_seeds(seed: int) -> None:
    """
    Parameters
    ----------
    seed: `int`, required
        The integer seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
