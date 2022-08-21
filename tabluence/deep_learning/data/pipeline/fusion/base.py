import abc
from typing import Dict, Any


class DataSideFusionBase(metaclass=abc.ABCMeta):
    """
    The base class for data side fusion.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def fuse(self, batch_data):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.fuse(*args, **kwargs)
