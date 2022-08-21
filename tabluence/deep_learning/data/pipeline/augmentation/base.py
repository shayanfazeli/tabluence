import abc
from typing import List, Dict
import pandas


class DataSideAugmentationBase(metaclass=abc.ABCMeta):
    """
    The base class for augmentation pipelines operated on the data side
    """
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def augment(self, batch_data: Dict[str, List[pandas.DataFrame]]) -> Dict[str, List[pandas.DataFrame]]:
        """
        The main augmentation function, which upon calling an object of this class
        will apply on the batch.

        Parameters
        ----------
        batch_data : dict
            The batch data to be augmented.

        Returns
        -------
        `Dict[str, List[pandas.DataFrame]]`: The augmented batch data.

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.augment(*args, **kwargs)
