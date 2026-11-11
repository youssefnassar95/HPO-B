"""Random number generator control for sweeps."""

import random

import numpy as np
import torch


def set_seed(seed):
    """Seed every generator the HPO methods draw from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
