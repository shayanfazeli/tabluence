import torch


def get_device(device: int) -> torch.device:
    """
    Returns a torch.device object.

    Parameters
    ----------
    device: `int`
        The device to use. -1 means cpu.
    """
    if device == -1:
        return torch.device("cpu")
    else:
        return torch.device("cuda:" + str(device))
