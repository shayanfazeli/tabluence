import abc
from tabluence.deep_learning.data.pipeline.augmentation.base import DataSideAugmentationBase


class SingleSliceDataSideAugmentationBase(DataSideAugmentationBase, metaclass=abc.ABCMeta):
    """
    The base class for augmentation of single-slice datasets.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
