from abc import ABCMeta

import torch
import torch.utils.data


class DatasetBase(torch.utils.data.Dataset, metaclass=ABCMeta):
    """
    This is the base class for all of the Nowoe datasets.
    """
    def __init__(self, *args, **kwargs):
        super(DatasetBase, self).__init__(*args, **kwargs)
