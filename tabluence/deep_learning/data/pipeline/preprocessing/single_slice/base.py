import abc
from tabluence.deep_learning.data.pipeline.preprocessing.base import DataSidePreprocessingBase


class SingleSliceDataSidePreprocessingBase(DataSidePreprocessingBase, metaclass=abc.ABCMeta):
    """
    Base class for preprocessing of data side of a single slice.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
