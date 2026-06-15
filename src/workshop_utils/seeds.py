import random
import numpy as np
import torch

WORKSHOP_SEED = 469


def set_seed(seed: int = WORKSHOP_SEED) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + GPU).

    Parameters
    ----------
    seed : int — seed value (default 469, the CSU area code)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


if __name__ == "__main__":
    set_seed()