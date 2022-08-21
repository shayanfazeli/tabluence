from tabluence.deep_learning.data.dataset.base import DatasetBase
from abc import ABCMeta


class SliceSequenceDatasetBase(DatasetBase, metaclass=ABCMeta):
    """
    This is the base class for the tasks and datasets in which data points comprise of
    a sequence of slices (for instance, a many-to-one mapping of a sequence of slices, each
    being a one-hour tabular time-series).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
