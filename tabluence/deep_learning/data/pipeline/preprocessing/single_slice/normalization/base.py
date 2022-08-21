import abc
from typing import List, Dict
import collections
import pandas
import torch.utils.data.dataloader
import numpy
from tqdm import tqdm
import copy
import logging
from tabluence.deep_learning.data.pipeline.preprocessing.base import DataSidePreprocessingBase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataSideNormalizationBase(DataSidePreprocessingBase, metaclass=abc.ABCMeta):
    """
    Base class for all augmentations that sample from a distribution.

    Please note that the configuration for this class needs to contain the following keys:
    - feature_names_per_data_source: `Dict[str, List[str]]`, required
        A dictionary including the data source names as keys and the feature names as values.
        The feature names will be buffered for normalization training
    """
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(DataSideNormalizationBase, self).__init__(*args, **kwargs)
        self.feature_names_per_data_source = self.config['feature_names_per_data_source']
        self.normalizer_models = None

    def learn(
            self,
            dataloader: torch.utils.data.dataloader.DataLoader
    ):
        """
        Parameters
        ----------
        dataloader: `torch.utils.data.dataloader.DataLoader`, required
            A single slice dataset's dataloader. Please note that this is the dataloader that will be used
            to learn the normalizers, thus, it should be a train dataloader.
        """
        distributions = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

        # - reading this dataloader
        logger.info('[learning normalizers] sampling from the given dataloader...')
        for batch in tqdm(dataloader):
            for data_source, features_per_ds in self.feature_names_per_data_source.items():
                for featurename in features_per_ds:
                    distributions[data_source][featurename].append(numpy.concatenate([e[data_source][featurename].to_numpy() for e in batch['slice']], axis=0))

        # - forming 1D numpy arrays
        for data_source in distributions:
            for featurename in distributions[data_source]:
                distributions[data_source][featurename] = numpy.concatenate(distributions[data_source][featurename], axis=0)

        # - making sure nans are not considered
        logger.info(f'[learning distributions] fitting models (of class {self.normalizer_model}) to the distributions...')
        for data_source in distributions:
            for featurename in distributions[data_source]:
                tmp = distributions[data_source][featurename]
                tmp = tmp[~numpy.isnan(tmp)]
                distributions[data_source][featurename] = copy.deepcopy(tmp)

        self.normalizer_models = collections.defaultdict(lambda: dict())
        for data_source in distributions:
            for featurename in distributions[data_source]:
                self.normalizer_models[data_source][featurename] = self.normalizer_model(**self.normalizer_args).fit(distributions[data_source][featurename].reshape(-1, 1))
        logger.info(f'[learning distributions] completed.')

    def preprocess(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]]) -> Dict[str, List[pandas.DataFrame]]:
        """
        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The collated batch of a single-slice dataloader.

        Returns
        ----------
        `Dict[str, List[pandas.DataFrame]]`: The batch after modification.
        """
        batch = copy.deepcopy(batch)
        for item_index in range(len(batch['slice'])):
            for data_source, features_per_ds in self.feature_names_per_data_source.items():
                for feature_name in features_per_ds:
                    assert feature_name in batch['slice'][item_index][data_source]
                    # batch['slice'][item_index][data_source][feature_name] = self.normalizer_models[data_source][feature_name].transform()
                    tmp = batch['slice'][item_index][data_source][feature_name].to_numpy().reshape(-1, 1)
                    if tmp.shape[0] > 0:
                        non_nan_indices = numpy.nonzero(~numpy.isnan(tmp))[0]
                        tmp[non_nan_indices] = self.normalizer_models[data_source][feature_name].transform(tmp[non_nan_indices])
                        batch['slice'][item_index][data_source][feature_name] = tmp.ravel()

        return batch

    @property
    @abc.abstractmethod
    def normalizer_model(self):
        pass

    @property
    @abc.abstractmethod
    def normalizer_args(self):
        pass
