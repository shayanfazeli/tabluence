import abc
from typing import Dict, Any


class DataSidePreprocessingBase(metaclass=abc.ABCMeta):
    "The base class for all data side preprocessors."
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abc.abstractmethod
    def preprocess(self, batch_data):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)
