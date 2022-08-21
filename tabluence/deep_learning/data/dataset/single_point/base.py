from tabluence.deep_learning.data.dataset.base import DatasetBase


class SinglePointDatasetBase(DatasetBase):
    """
    Single point dataset base class. This is the base class for tasks in which
    the data is comprised of multiple-source tabular datasets (meaning a single record
    per source, no timestamp).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
