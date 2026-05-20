"""Global seed management for reproducible experiments."""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value to use across numpy, random, os, and optionally torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.debug("Global seed set to %d", seed)
