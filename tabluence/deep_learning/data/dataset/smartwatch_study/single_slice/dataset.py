import collections
import os
import copy
import gzip
import pickle
import numpy
import multiprocessing
import threading
import concurrent.futures
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Callable
from tabluence.data.api.smartwatch.data_manager import SmartwatchDataManager
from tabluence.deep_learning.data.dataset.single_slice.base import SingleSliceDatasetBase
from tabluence.data.api.smartwatch.utilities.timestamp import get_utc_date_from_utc_timestamp
from .utilities import dict_hash

import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SmartwatchStudySingleSliceDataset(SingleSliceDatasetBase):
    """
    The main dataset class for the Smartwatch study.

    Parameters
    ----------
    data_manager: `SmartwatchDataManager`, required
        The data manager for our smartwatch dataset.
    slice_lengths: `List[int]`, required
        The lengths of the slices to use.
    slice_time_step: `int`, required
        The slice stride
    overall_stress_quantization_bins: `List[float]`, optional (default=`numpy.linspace(0, 9.0, (3*3)*4).tolist() + [9.0]`)
        The bins to use for overall stress quantization
    specific_stress_quantization_bins: `List[float]`, optional (default=numpy.linspace(0, 3.0, (3)*4).tolist() + [9.0]`)
        The bins to use for specific stress quantization
    label_milestone_per_window: `float`, optional (default=1.0)
        The ratio of the window at which point the system queries a label and assigns it to the window. `1.0` means
        that the endpoint of the window is where the label is queried for.
    metadata_cache_filepath: `str`, optional (default=None)
        If provided and exists, the metadata will be loaded from this filepath. If not, the metadata
        will be created and saved to this filepath. This process might take a while depending on the
        other parameters.
    no_cache: `bool`, optional (default=False)
        If set to `True`, even if a cache file is provided it will be ignored and not loaded. However,
        the resulting metadata will overwrite the file provided on `metadata_cache_filepath`, if something
        is provided.`
    parallel_threads: `int`, optional (default=5)
        Number of parallel threads that will be used in processing subjects in creating the metadata (if not
        loading from a cache file).
    """
    def __init__(
            self,
            data_manager: SmartwatchDataManager,
            slice_lengths: List[int],
            slice_time_step: int,
            overall_stress_quantization_bins: List[float],
            specific_stress_quantization_bins: List[float],
            label_milestone_per_window: float = 1.0,
            metadata_cache_filepath: str = None,
            no_cache: bool = False,
            parallel_threads: int = 5,
            **kwargs
    ):
        """
        constructor
        """
        super(SmartwatchStudySingleSliceDataset, self).__init__(**kwargs)
        self.contents_all = None
        self.data_manager = data_manager
        self.slice_lengths = [float(e) for e in slice_lengths]
        self.slice_time_step = slice_time_step
        self.overall_stress_quantization_bins = overall_stress_quantization_bins
        self.specific_stress_quantization_bins = specific_stress_quantization_bins
        self.label_milestone_per_window = label_milestone_per_window
        self.metadata_cache_filepath = os.path.abspath(metadata_cache_filepath)
        self.datapoint_counts = None
        self.metadata = None
        self.no_cache = no_cache
        self.parallel_threads = parallel_threads
        self.build_metadata_parallel()
        self.sanity_checks()

    def get_quantized_stress_value_and_bin(self, x: float, stress_category: str) -> Tuple[int, numpy.ndarray]:
        """
        Parameters
        ----------
        x: `float`, required
            The stress value to quantize
        stress_category: `str`, required
            The stress category to quantize. One of `overall`, or specific ones (`general`, `interpersonal`, and `induced`)

        Returns
        -------
        `Tuple[int, numpy.ndarray]`: The quantized value and the quantization bins for the stress category
        """
        bins = self.overall_stress_quantization_bins if stress_category == 'overall' else self.specific_stress_quantization_bins
        # vals, bins = numpy.histogram(numpy.array([x]), bins=bins)
        bin_index = numpy.digitize(x, bins) - 1
        return bin_index, bins

    def initialize_datapoint_counts(self) -> None:
        """
        Preparing the `datapoint_counts` which will hold the basic stats for the item metadata over this dataset.
        """
        self.datapoint_counts = {
            'total': 0,
        }
        for k in ['overall', 'general', 'induced', 'interpersonal']:
            bins = self.overall_stress_quantization_bins if k == 'overall' else self.specific_stress_quantization_bins
            # _, bins = numpy.histogram(numpy.array([0]), bins=bins)
            self.datapoint_counts.update(
                {f'{k}/bin-{i}/{bins[i]:.2f}-{bins[i+1]:.2f}': 0 for i in range(len(bins) - 1)}
            )

    def enforce_new_quantization_based_label_layout(self) -> None:
        """
        This method will run (whether or not cache option is requested) when initializing the dataset, and
        it will re-generate the metadata based on the possibly new quantization information.
        """
        self.label_layouts = {
            x + '_quantized_stress_value': set() for x in ['general', 'induced', 'interpersonal', 'overall']
        }
        self.initialize_datapoint_counts()
        logger.info("\n\t~> processing the metadata for building quantization based label layout")

        for item_index in tqdm(range(len(self.metadata))):
            tmp = copy.deepcopy(self.metadata[item_index])
            for k in ['general', 'induced', 'interpersonal']:
                i, bins = self.get_quantized_stress_value_and_bin(tmp[f'{k}_stress_value'], stress_category=k)
                tmp[f'{k}_quantized_stress_value'] = bins[i]
                self.label_layouts[k + '_quantized_stress_value'].add(bins[i])
                self.datapoint_counts[f'{k}/bin-{i}/{bins[i]:.2f}-{bins[i+1]:.2f}'] += 1

            i, bins = self.get_quantized_stress_value_and_bin(tmp['overall_stress_value'], stress_category='overall')
            tmp[f'overall_quantized_stress_value'] = bins[i]
            self.label_layouts['overall_quantized_stress_value'].add(bins[i])
            self.datapoint_counts[f'overall/bin-{i}/{bins[i]:.2f}-{bins[i+1]:.2f}'] += 1
            self.metadata[item_index] = copy.deepcopy(tmp)

        for k in self.label_layouts.keys():
            self.label_layouts[k] = sorted(list(self.label_layouts[k]))
            logger.info(f"\n\t\t~> label layout for {k} stress category: {self.label_layouts[k]}")

    def build_metadata(self, slice_ignore_conditions: List[Callable[[Dict[str, Any]], bool]] = []) -> None:
        """
        This method builds the metadata including time-ranges and labels for the dataset.

        Parameters
        ----------
        slice_ignore_conditions: `List[Callable[[Dict[str, Any]], bool]]`, optional (default=`[]`)
            A list of functions that will be called for each slice. If any of them returns `True`, the slice will be ignored.
        """
        if (not self.no_cache) and os.path.isfile(self.metadata_cache_filepath):
            with gzip.open(self.metadata_cache_filepath, 'rb') as handle:
                self.metadata, self.datapoint_counts = pickle.load(handle)
                self.enforce_new_quantization_based_label_layout()
                logger.info("~> loaded and processed metadata from cache file: {}".format(self.metadata_cache_filepath))

        else:
            logger.info("\nBuilding dataset metadata...\n")
            self.initialize_datapoint_counts()
            self.metadata = list()
            self.metadata_md5 = set()
            logger.info("processing subjects:")
            for subject_id in tqdm(self.data_manager.get_existing_subject_id_list()):
                logger.info("processing subject_id: %s", subject_id)
                subject_overall_stress_function = self.data_manager.get_subject_general_stress_function(subject_id=subject_id)
                subject_specific_stress_function = self.data_manager.get_subject_specific_stress_function(subject_id=subject_id, stress_types=["induced", "general", "interpersonal"])

                t_min, t_max = self.data_manager.get_utc_timestamp_range_for_subject(subject_id=subject_id)
                for slice_length_in_seconds in self.slice_lengths:
                    logger.info("processing slice_length_in_seconds: %s\n\n", slice_length_in_seconds)
                    t_start = t_min

                    while t_start + slice_length_in_seconds <= t_max:
                        print(f"remaining timespan: {t_max - t_start} seconds == {float(t_max - t_start) / 3600.0:.2f} hours       ", end='\r')
                        # - getting window end time
                        t_end = t_start + slice_length_in_seconds

                        signals_slice = self.data_manager.get_subject_signals_for_utc_timestamp_range(
                            subject_id=subject_id,
                            timestamp_range=(t_start, t_end)
                        )

                        ignore_slice = False

                        # - ignoring certain hours
                        dt_start, dt_end = get_utc_date_from_utc_timestamp(t_start), get_utc_date_from_utc_timestamp(t_end)
                        if dt_start.hour in list(range(0, 6, 1)):
                            ignore_slice = True

                        for ignore_condition in slice_ignore_conditions:
                            if ignore_condition(slice=signals_slice):
                                ignore_slice = True
                        if ignore_slice:
                            # - pushing forward
                            t_start += self.slice_time_step
                            continue

                        if not self.data_manager.is_slice_empty(slice=signals_slice):
                            # - getting stress labels
                            query_timestamp = t_start + slice_length_in_seconds * self.label_milestone_per_window
                            overall_stress = subject_overall_stress_function(numpy.array([query_timestamp])).item()
                            specific_stress = subject_specific_stress_function(numpy.array([query_timestamp]))
                            specific_stress = {x: specific_stress[x].item() for x in specific_stress}

                            item = {
                                'subject_id': subject_id,
                                'utc_timestamp_window': (t_start, t_end),
                                'overall_stress_value': overall_stress,
                                'general_stress_value': specific_stress['general'],
                                'interpersonal_stress_value': specific_stress['interpersonal'],
                                'utc_timestamp_for_stress_query': query_timestamp,
                                'induced_stress_value': specific_stress['induced']
                            }

                            slice = self.data_manager.get_subject_signals_for_utc_timestamp_range(
                                subject_id=subject_id,
                                timestamp_range=(t_start, t_end)
                            )

                            item_md5 = dict_hash(dictionary={x: slice[x].utc_timestamp.tolist() for x in slice})

                            if item_md5 not in self.metadata_md5:
                                self.datapoint_counts['total'] += 1

                                for k in ['general', 'induced', 'interpersonal']:
                                    i, bins = self.get_quantized_stress_value_and_bin(specific_stress[k], stress_category=k)
                                    item.update({f'{k}_quantized_stress_value': bins[i]})
                                    self.datapoint_counts[f'{k}/bin-{i}/{bins[i]:.2f}-{bins[i+1]:.2f}'] += 1

                                i, bins = self.get_quantized_stress_value_and_bin(overall_stress, stress_category='overall')
                                item.update({f'overall_quantized_stress_value': bins[i]})
                                self.datapoint_counts[f'overall/bin-{i}/{bins[i]:.2f}-{bins[i+1]:.2f}'] += 1

                                self.metadata.append(
                                    copy.deepcopy(item)
                                )
                                self.metadata_md5.add(item_md5)

                        # - pushing forward
                        t_start += self.slice_time_step
            logger.info("\nDataset metadata built.\n")

            if self.metadata_cache_filepath is not None:
                logger.info("\nCaching dataset metadata...\n")
                with gzip.open(self.metadata_cache_filepath, 'wb') as handle:
                    pickle.dump((self.metadata, self.datapoint_counts), handle)
                    logger.info(f"\nCached it. File is stored at f{self.metadata_cache_filepath}\n")

    def parallel_processing_function_to_process_subject_slice(
            self,
            subject_id: str,
            slice_ignore_conditions: List[Callable[[Dict[str, Any]], bool]],
    ):
        """
        The parallel processing function for subject slices.

        Parameters
        ----------
        subject_id: `str`, required
            The subject id.
        slice_ignore_conditions: `List[Callable[[Dict[str, Any]], bool]]`, optional (default=`[]`)
            A list of functions that will be called for each slice. If any of them returns `True`, the slice will be ignored.
        """
        # - get subject stress functions
        subject_overall_stress_function = self.data_manager.get_subject_general_stress_function(subject_id=subject_id)
        subject_specific_stress_function = self.data_manager.get_subject_specific_stress_function(subject_id=subject_id, stress_types=["induced", "general", "interpersonal"])
        t_min, t_max = self.data_manager.get_utc_timestamp_range_for_subject(subject_id=subject_id)

        for slice_length_in_seconds in self.slice_lengths:
            t_start = t_min

            while t_start + slice_length_in_seconds <= t_max:
                if subject_id == self.first_subject_id_in_progress:
                    print(f"remaining timespan: {t_max - t_start} seconds == {float(t_max - t_start) / 3600.0:.2f} hours       ", end='\r')
                # - getting window end time
                t_end = t_start + slice_length_in_seconds

                signals_slice = self.data_manager.get_subject_signals_for_utc_timestamp_range(
                    subject_id=subject_id,
                    timestamp_range=(t_start, t_end)
                )

                ignore_slice = False

                # - ignoring certain hours
                dt_start, dt_end = get_utc_date_from_utc_timestamp(t_start), get_utc_date_from_utc_timestamp(t_end)
                if dt_start.hour in list(range(0, 6, 1)):
                    ignore_slice = True

                for ignore_condition in slice_ignore_conditions:
                    if ignore_condition(slice=signals_slice):
                        ignore_slice = True

                if not self.data_manager.is_slice_empty(slice=signals_slice) and not ignore_slice:
                    # - getting stress labels
                    query_timestamp = t_start + slice_length_in_seconds * self.label_milestone_per_window
                    overall_stress = subject_overall_stress_function(numpy.array([query_timestamp])).item()
                    specific_stress = subject_specific_stress_function(numpy.array([query_timestamp]))
                    specific_stress = {x: specific_stress[x].item() for x in specific_stress}

                    item = {
                        'subject_id': subject_id,
                        'utc_timestamp_window': (t_start, t_end),
                        'overall_stress_value': overall_stress,
                        'general_stress_value': specific_stress['general'],
                        'interpersonal_stress_value': specific_stress['interpersonal'],
                        'utc_timestamp_for_stress_query': query_timestamp,
                        'induced_stress_value': specific_stress['induced']
                    }

                    slice = self.data_manager.get_subject_signals_for_utc_timestamp_range(
                        subject_id=subject_id,
                        timestamp_range=(t_start, t_end)
                    )

                    item_md5 = dict_hash(dictionary={x: slice[x].utc_timestamp.tolist() for x in slice})

                    with self.rlock:
                        if item_md5 not in self.metadata_md5:
                            self.datapoint_counts['total'] += 1

                            for k in ['general', 'induced', 'interpersonal']:
                                i, bins = self.get_quantized_stress_value_and_bin(specific_stress[k], stress_category=k)
                                item.update({f'{k}_quantized_stress_value': bins[i]})
                                self.datapoint_counts[f'{k}/bin-{i}/{bins[i]:.2f}-{bins[i+1]:.2f}'] += 1

                            i, bins = self.get_quantized_stress_value_and_bin(overall_stress, stress_category='overall')
                            item.update({f'overall_quantized_stress_value': bins[i]})
                            self.datapoint_counts[f'overall/bin-{i}/{bins[i]:.2f}-{bins[i+1]:.2f}'] += 1

                            self.metadata.append(
                                copy.deepcopy(item)
                            )
                            self.metadata_md5.add(item_md5)

                # - pushing forward
                t_start += self.slice_time_step

    def build_metadata_parallel(self, slice_ignore_conditions: List[Callable[[Dict[str, Any]], bool]] = []) -> None:
        """
        This method builds the metadata including time-ranges and labels for the dataset.

        Parameters
        ----------
        slice_ignore_conditions: `List[Callable[[Dict[str, Any]], bool]]`, optional (default=`[]`)
            A list of functions that will be called for each slice. If any of them returns `True`, the slice will be ignored.
        """
        if (not self.no_cache) and os.path.isfile(self.metadata_cache_filepath):
            with gzip.open(self.metadata_cache_filepath, 'rb') as handle:
                self.metadata, self.datapoint_counts = pickle.load(handle)
                self.enforce_new_quantization_based_label_layout()
                logger.info("~> loaded / processed metadata from cache file: {}".format(self.metadata_cache_filepath))
        else:
            logger.info("\nBuilding dataset metadata...\n")
            self.initialize_datapoint_counts()
            self.metadata = list()
            self.metadata_md5 = set()

            # - making dicts threadsafe:
            manager = multiprocessing.Manager()
            self.datapoint_counts = manager.dict(self.datapoint_counts)
            self.rlock = threading.RLock()
            futures = []

            logger.info("processing subjects:")
            self.first_subject_id_in_progress = None
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_threads) as executor:
                for subject_id in tqdm(self.data_manager.get_existing_subject_id_list()):
                    if self.first_subject_id_in_progress is None:
                        self.first_subject_id_in_progress = subject_id
                    futures.append(executor.submit(
                        self.parallel_processing_function_to_process_subject_slice,
                        subject_id,
                        slice_ignore_conditions
                    ))

                logger.info('Waiting for tasks to complete...')
                for future_index, future in tqdm(enumerate(futures)):
                    self.first_subject_id_in_progress = self.data_manager.get_existing_subject_id_list()[future_index]
                    _ = future.result()
                # concurrent.futures.wait(futures)
                executor.shutdown(wait=True, cancel_futures=False)

            self.datapoint_counts = dict(self.datapoint_counts)
            logger.info("\nDataset metadata built.\n")

            if self.metadata_cache_filepath is not None:
                logger.info("\nCaching dataset metadata...\n")
                with gzip.open(self.metadata_cache_filepath, 'wb') as handle:
                    pickle.dump((self.metadata, self.datapoint_counts), handle)
                    logger.info(f"\nCached it. File is stored at f{self.metadata_cache_filepath}\n")

    def sanity_checks(self):
        """sanity checks"""
        assert self.slice_time_step > 0, "slice_time_step must be greater than 0"

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        meta_item = self.metadata[index]
        slice_item = self.data_manager.get_subject_signals_for_utc_timestamp_range(
            subject_id=meta_item['subject_id'],
            timestamp_range=meta_item['utc_timestamp_window']
        )
        return {'meta': meta_item, 'slice': slice_item}
