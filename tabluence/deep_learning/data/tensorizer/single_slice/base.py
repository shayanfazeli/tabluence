import abc
from tabluence.deep_learning.data.tensorizer.base import TensorizerBase


class SingleSliceTensorizerBase(TensorizerBase, metaclass=abc.ABCMeta):
    """
    Base class for the tensorizers that are to work with the single slice dataset.
    """
    def __init__(self, *args, **kwargs):
        super(SingleSliceTensorizerBase, self).__init__(*args, **kwargs)
