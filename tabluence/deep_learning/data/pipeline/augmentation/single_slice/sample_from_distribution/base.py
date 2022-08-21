import abc
from typing import List, Dict, Any, Union, Tuple
import collections
import pandas
import torch.utils.data.dataloader
import numpy
from tqdm import tqdm
import copy
import logging
from tabluence.deep_learning.data.pipeline.augmentation.single_slice.base import SingleSliceDataSideAugmentationBase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SampleFromDistributionAugmentionBase(SingleSliceDataSideAugmentationBase, metaclass=abc.ABCMeta):
    """
    Base class for all augmentations that sample from a distribution.

    Please note that the config for this module must contain the following keys:
        - feature_names_per_data_source: `Dict[str, List[str]]`, required
            A dictionary that maps data sources to the list of names of the features that are to be augmented.
    """
    def __init__(
            self,
            *args,
            rows_to_add: Union[int, Tuple[int, int]] = (0, 20),
            **kwargs
    ):
        super(SampleFromDistributionAugmentionBase, self).__init__(*args, **kwargs)
        self.feature_names_per_data_source = self.config['feature_names_per_data_source']
        self.distr_models = None
        self.rows_to_add = rows_to_add

    def sample_a_row_count_to_add(self) -> int:
        """
        Sample row count to add

        Returns
        -------
        `int`: the number of rows to add (serving as a maximum)
        """
        if isinstance(self.rows_to_add, int):
            return self.rows_to_add
        else:
            return numpy.random.randint(self.rows_to_add[0], self.rows_to_add[1])

    def learn(
            self,
            dataloader: torch.utils.data.dataloader.DataLoader
    ):
        """
        Parameters
        ----------
        dataloader: `torch.utils.data.dataloader.DataLoader`, required
            A single slice dataset's dataloader. Please note that this is the dataloader that will be used
            to learn the distributions, thus, it should be a train dataloader.
        """
        distributions = collections.defaultdict(lambda: collections.defaultdict(lambda: []))

        # - reading this dataloader
        logger.info('[learning distributions] sampling from the given dataloader...')
        for batch in tqdm(dataloader):
            for data_source, features_per_ds in self.feature_names_per_data_source.items():
                for featurename in features_per_ds:
                    distributions[data_source][featurename].append(numpy.concatenate([e[data_source][featurename].to_numpy() for e in batch['slice']], axis=0))

        # - forming 1D numpy arrays
        for data_source in distributions:
            for featurename in distributions[data_source]:
                distributions[data_source][featurename] = numpy.concatenate(distributions[data_source][featurename], axis=0)

        # - making sure nans are not considered
        logger.info(f'[learning distributions] fitting models (of class {self.distribution_model}) to the distributions...')
        for data_source in distributions:
            for featurename in distributions[data_source]:
                tmp = distributions[data_source][featurename]
                tmp = tmp[~numpy.isnan(tmp)]
                distributions[data_source][featurename] = copy.deepcopy(tmp)

        self.distr_models = collections.defaultdict(lambda: dict())
        for data_source in distributions:
            for featurename in distributions[data_source]:
                self.distr_models[data_source][featurename] = self.distribution_model(**self.distribution_args).fit(distributions[data_source][featurename].reshape(-1, 1))
        logger.info(f'[learning distributions] completed.')

    def augment(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]]) -> Dict[str, List[pandas.DataFrame]]:
        """
        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The collated batch of a single-slice dataloader.

        Returns
        ----------
        `Dict[str, List[pandas.DataFrame]]`: The batch after modification.
        """
        rows_to_add = self.sample_a_row_count_to_add()
        for item_index in range(len(batch['slice'])):
            for ds_to_augment in self.distr_models.keys():
                column_names = batch['slice'][item_index][ds_to_augment].columns.tolist()
                timestamps = batch['slice'][item_index][ds_to_augment].utc_timestamp.to_numpy()
                if timestamps.shape[0] == 0:
                    # - ignore, nothing's possible
                    continue

                # - getting the timestamp range and those that are left
                min_timestamp = timestamps.min()
                max_timestamp = timestamps.max()
                timestamps_left_blank = numpy.arange(min_timestamp, max_timestamp + 1)
                timestamps_left_blank[(timestamps - min_timestamp).astype('int')] = -1
                timestamps_left_blank = timestamps_left_blank[timestamps_left_blank > -1].tolist()

                number_of_rows_to_add = min(rows_to_add, len(timestamps_left_blank))
                new_rows = pandas.DataFrame([{x: None for x in column_names} for _ in range(number_of_rows_to_add)])
                new_rows['utc_timestamp'] = numpy.random.choice(timestamps_left_blank, number_of_rows_to_add)
                for feature_name in self.distr_models[ds_to_augment].keys():
                    new_rows[feature_name] = self.distr_models[ds_to_augment][feature_name].sample(number_of_rows_to_add)[0].ravel()

                # - adding the new rows to the original dataframe
                batch['slice'][item_index][ds_to_augment] = pandas.concat([batch['slice'][item_index][ds_to_augment], new_rows], ignore_index=True)
                # - sorting
                batch['slice'][item_index]['daily'].sort_values(by='utc_timestamp', inplace=True)
                # - filling nans
                batch['slice'][item_index]['daily'] = batch['slice'][item_index]['daily'].ffill().bfill()
        return batch

    def impute(self, batch: Dict[str, List[pandas.DataFrame]]) -> Dict[str, List[pandas.DataFrame]]:
        """
        Parameters
        -----------
        batch: `Dict[str, List[pandas.DataFrame]]`, required
            The collated batch of a single-slice dataloader.

        Returns
        ----------
        `Dict[str, List[pandas.DataFrame]]`: The batch after modification.
        """
        for item_index in range(len(batch['slice'])):
            for ds_to_augment in self.distr_models.keys():
                timestamps = batch['slice'][item_index][ds_to_augment].utc_timestamp.to_numpy()
                if timestamps.shape[0] == 0:
                    # - ignore, nothing's possible
                    continue

                for feature_name in self.distr_models[ds_to_augment].keys():
                    item_series = batch['slice'][item_index][ds_to_augment][feature_name]
                    item_series = item_series.to_numpy()
                    if item_series.shape[0] == 0:
                        continue
                    number_of_missing_items = int(numpy.isnan(item_series).sum())
                    if number_of_missing_items == 0:
                        continue
                    item_series[numpy.isnan(item_series)] = self.distr_models[ds_to_augment][feature_name].sample(number_of_missing_items)[0].ravel()
        return batch

    @property
    @abc.abstractmethod
    def distribution_model(self):
        pass

    @property
    @abc.abstractmethod
    def distribution_args(self):
        pass
