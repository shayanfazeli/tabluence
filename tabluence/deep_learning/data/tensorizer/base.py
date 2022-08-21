import abc
from typing import Dict, Any
import torch
import torch.nn


class TensorizerBase(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for the Tesorizer conceptual model.
    The goal for modules in this family is to prepare the input from the unstructured tabular/numpy format
    to tensors ready to be used by the neural network.

    Please note that approaches such as preparing the packed input for sequence-to-sequence models are to be
    implemented in the subclasses as well.

    Parameters
    ----------
    config: `Dict[str, Any]`, required
        Config

    device: `torch.device`, required
        Device to use for the tensors
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super(TensorizerBase, self).__init__()
        self.config = config
        self.device = device

    @abc.abstractmethod
    def tensorize(self, batch_data):
        """
        Tensorize function, which is the same as the :cls:`__call__` method of this class.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.tensorize(*args, **kwargs)
