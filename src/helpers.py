from typing import Optional

import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Determines the device to send model parameters to for training.
    Prioritizes platforms that have GPU first. If none are found, resorts
    to using CPU.

    Parameters
    ----------
    device: str
        Desired device to use (if any). Must be a recognizeable device by
        torch.device
    """
    if device:
        return torch.device(device)
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return torch.device(device)

