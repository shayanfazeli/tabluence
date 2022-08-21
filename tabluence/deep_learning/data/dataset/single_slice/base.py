from tabluence.deep_learning.data.dataset.base import DatasetBase
from abc import ABCMeta


class SingleSliceDatasetBase(DatasetBase, metaclass=ABCMeta):
    """
    Single slice dataset base class. This applies on cases in which
    we have one single tabular time-series per data source.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
