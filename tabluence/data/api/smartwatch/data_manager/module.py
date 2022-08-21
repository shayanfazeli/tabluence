import os
import pdb
import pickle
import gzip
import sys
import copy
import json
import numpy
import pandas
import datetime
import dateutil.parser as datetime_parser
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import functools

import tabluence.data.api.smartwatch.secured_meta.metadata as smartwatch_metadata
from tabluence.data.api.smartwatch.utilities.timestamp import get_utc_date_from_utc_timestamp, \
    get_utc_timestamp_from_naive_datetime
from tabluence.data.api.smartwatch.secured_meta.stress_labels import stress_labels  # replace this with a module of yours for a new dataset
from .stress_poke_functions import *


class SmartwatchDataManager:
    """
    The :cls:`SmartwatchDataManager` class is a data manager for the Smartwatch data. It includes all of
    the required functionalities for easy access to a clean version of data, along with per-subject api.

    Parameters
    ----------
    root_dir: `str`, required
        The root directory which contains the json files that are expected to be there.

    subject_id_list: `List[str]`, required
        The list of subject ids, which will be used as a filter. The resulting data bundle
        will include data ONLY if the subject ids are present in the json_contents. Please
        note that it is supposed to act as a filter, therefore, it is possible that
        some of a subject id mentioned in this variable has no data, and therefore,
        will not be present in the resulting data bundle.

    cache_filepath: `str`, required
        If provided, the data manager will try to load the data from the cache file, or write to it
        if it does not exist. It helps as the process of loading the smartwatch study's file manager
        can be relatively slow otherwise.
    """
    def __init__(
            self,
            root_dir: str,
            subject_id_list: List[str],
            cache_filepath: str = None
    ):
        """Constructor"""
        load_cache = False
        if cache_filepath is not None:
            if os.path.isfile(cache_filepath):
                load_cache = True

        if not load_cache:
            # - owning the variables
            self.root_dir = os.path.abspath(root_dir)
            self.subject_id_list = subject_id_list

            # - preparing the placeholders for data
            self.json_contents = None
            self.per_subject_dataset = None
            self.stress_labels = None

            # - loading and building the data
            self.get_all_json_contents_from_smartwatch_repo()
            self.prepare_per_subject_dataset()
            self.post_process_per_subject_dataset()

            if cache_filepath is not None:
                with gzip.open(cache_filepath, 'wb') as handle:
                    pickle.dump(dict(
                        root_dir=self.root_dir,
                        subject_id_list=self.subject_id_list,
                        json_contents=self.json_contents,
                        per_subject_dataset=self.per_subject_dataset,
                        stress_labels=self.stress_labels
                    ), handle)
        else:
            with gzip.open(cache_filepath, 'rb') as handle:
                cache_data = pickle.load(handle)
                self.root_dir = cache_data['root_dir']
                assert self.root_dir == os.path.abspath(root_dir)
                self.subject_id_list = cache_data['subject_id_list']
                assert len(self.subject_id_list) == len(subject_id_list)
                assert len(set(self.subject_id_list).intersection(set(subject_id_list))) == len(subject_id_list)
                self.json_contents = cache_data['json_contents']
                self.per_subject_dataset = cache_data['per_subject_dataset']
                self.stress_labels = cache_data['stress_labels']

        self.build_stress_probe_functions_per_subject()
        # - sanity checks
        self.initialization_sanity_checks()

    def get_subject_general_stress_function(
            self,
            subject_id: str,
            stress_labels: List[Dict[str, Any]] = None
    ) -> Callable[[float], float]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            Subject id

        Returns
        ----------
        `Callable[[float], float]`: the general stress function for the subject
        """
        if stress_labels is None:
            stress_labels = copy.deepcopy(self.stress_labels)
        output = [lambda x: numpy.zeros(x.shape[0])]
        for stress_label in stress_labels:
            if stress_label['subject_id'] == subject_id:
                output.append(stress_poke_function_gaussian_1(stress_label))

        return lambda x: numpy.sum(numpy.array([f(x) for f in output]), axis=0)

    def get_subject_specific_stress_function(
            self,
        subject_id: str,
        stress_types: List[str] = [
            "induced",
            "general",
            "interpersonal"
        ]
    ) -> Dict[str, Callable[[float], float]]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            subject id

        stress_types: `List[str]`, optional (default=['induced', 'general', 'interpersonal'])
            The specific stress types that we want to include in the label function. All possible
            options for the smartwatch study are included by default, as it is better for the
            metadata to be inclusive (not much additional overhead).

        Returns
        ----------
        `Dict[Callable[[float], float]]`: the stress function for the subject, each key is a provided stress type
        """
        stress_types = list(set(stress_types))
        assert len(stress_types) > 0, "Please provide at least one stress type"
        output = {stress_type: self.get_subject_general_stress_function(
            subject_id,
            [e for e in self.stress_labels if e['stress_type_2'] == stress_type]
        ) for stress_type in stress_types}

        return lambda x: {t: output[t](x) for t in output.keys()}

    def build_stress_probe_functions_per_subject(self) -> None:
        """
        Building the stress labels for all subjects.
        """
        self.stress_labels = copy.deepcopy(stress_labels)
        for i in range(len(stress_labels)):
            if isinstance(stress_labels[i]['probe_datetime'], tuple):
                self.stress_labels[i]['probe_timestamp'] = (
                    get_utc_timestamp_from_naive_datetime(stress_labels[i]['probe_datetime'][0]), get_utc_timestamp_from_naive_datetime(stress_labels[i]['probe_datetime'][1]))
            else:
                self.stress_labels[i]['probe_timestamp'] = get_utc_timestamp_from_naive_datetime(stress_labels[i]['probe_datetime'])

    def initialization_sanity_checks(self):
        """
        Base sanity checks for initializing a smartwatch data manager object.
        """
        assert os.path.isdir(self.root_dir), "The root directory {} does not exist".format(self.root_dir)
        assert self.json_contents is not None, "The json contents are not loaded yet"
        assert self.per_subject_dataset is not None, "The per-subject dataset is not loaded yet"
        assert len(set(self.subject_id_list)) == len(self.subject_id_list), "The subject id list is not unique"
        assert self.stress_labels is not None, "The stress labels are not loaded yet"

    def get_existing_subject_id_list(self) -> List[str]:
        """
        This method returns a list of subject ids that are present in the data manager.

        Returns
        -------
        `List[str]`: the list of subject ids that are present in the data manager.
        """
        subject_id_lists = []
        for source in self.per_subject_dataset:
            subject_id_lists += [list(self.per_subject_dataset[source].keys())]
        subject_id_list = functools.reduce(lambda x, y: list(set(x).intersection(set(y))), subject_id_lists)
        return sorted(list(set(subject_id_list)))

    def get_all_json_contents_from_smartwatch_repo(self) -> None:
        """
        This method loads all of the JSON contents into the data manager memory.

        It will be stored as variable `json_contents` with the type `Dict[str, Any]` such that:
        For each json file (e.g., `g_activity.json`), the output will contain
        a key (e.g., `activity`) with the corresponding data.
        """
        repo = self.root_dir
        json_files = [e for e in os.listdir(repo) if e.endswith('.json')]
        json_contents = dict()
        for json_file in json_files:
            if json_file.split('.')[0][2:] in smartwatch_metadata.known_sources:
                with open(os.path.join(repo, json_file), 'r') as handle:
                    json_contents[json_file.split('.')[0][2:]] = json.load(handle)

        for source in smartwatch_metadata.known_sources:
            assert source in json_contents, "Source {} not found in repo {}".format(source, repo)
        self.json_contents = json_contents

    def prepare_per_subject_dataset(
            self,
    ) -> None:
        """
        Please note that the json_contents must be provided and data must be already loaded in it.

        json_contents: `Dict[str, Any]`, required
            The parsed json contents from the dumpfiles recorded by the smartwatch API
        subject_ids: `List[str]`, required
            The list of subject ids, which will be used as a filter. The resulting data bundle
            will include data ONLY if the subject ids are present in the json_contents. Please
            note that it is supposed to act as a filter, therefore, it is possible that
            some of a subject id mentioned in this variable has no data, and therefore,
            will not be present in the resulting data bundle.

        The internal variable `per_subject_dataset` will be formed which has the following type
        and characteristics:

        `Dict[str, Any]`:
            The per-subject data bundle, which for our smartwatch dataset will include
            the following keys:
            ```
            ['activity',
            'daily',
            'respiration',
            'activityDetail',
            'stress',
            'sleep',
            'pulseOx']
            ```
            In each of these keys, we have a data bundle which includes the corresponding dataframe,
            sorted by `utc_timestamp`, for subject id (activityDetail is an exception as it does not
            include timestamp).
        """
        # - building the subject-oriented dataset
        per_subject_dataset = dict()

        # - for each one of the sources
        for source in smartwatch_metadata.known_sources:
            # - merge daily information into one source (map them to the same target source)
            if source.startswith('daily'):
                target_source = 'daily'
            else:
                target_source = source

            # - build a per-subject-subset of the dataset (for each source)
            if target_source not in per_subject_dataset:
                per_subject_dataset[target_source] = dict()

            # - for each subject id we will proceed to add the data
            for value in self.json_contents[f'{source}']:
                subject_id = value['user_id']
                if subject_id not in self.subject_id_list:
                    continue
                else:
                    if subject_id not in per_subject_dataset[target_source].keys():
                        per_subject_dataset[target_source][subject_id] = []
                    per_subject_dataset[target_source][subject_id].append(value)

        # - now, the postprocessing will take place which includes
        # adding the utc-like timestamp (note that the idea is to represent the
        # OFFSETED local timestamp as UTC timestamp. The assumption for timezones
        # is that all of them are local.
        for source in per_subject_dataset:
            for key in per_subject_dataset[source]:
                df = pandas.DataFrame(per_subject_dataset[source][key])
                if source not in ['activityDetail']:
                    df['utc_timestamp'] = df.apply(lambda x: x['startTimeInSeconds'] + x['startTimeOffsetInSeconds'], axis=1)
                    df['utc_date'] = df.apply(lambda x: str(get_utc_date_from_utc_timestamp(x['utc_timestamp'])), axis=1)
                    df.sort_values(by='utc_timestamp', inplace=True)
                per_subject_dataset[source][key] = copy.deepcopy(df)
        self.per_subject_dataset = per_subject_dataset

    def post_process_per_subject_dataset(self) -> None:
        for source in smartwatch_metadata.ts_fields_per_source:
            for processed_subjects, subject_id in enumerate(self.per_subject_dataset[source].keys()):
                self.per_subject_dataset[source][subject_id] = self.unfold_timeseries_recordings(
                    df=self.per_subject_dataset[source][subject_id],
                    fields=smartwatch_metadata.ts_fields_per_source[source],
                )

        # - picking the first monitoring period for each second
        for source in smartwatch_metadata.ts_fields_per_source:
            for processed_subjects, subject_id in enumerate(self.per_subject_dataset[source].keys()):
                tmp_df = copy.deepcopy(self.per_subject_dataset[source][subject_id])
                tmp_df = tmp_df.sort_values(by=['utc_timestamp', 'durationInSeconds']).groupby('utc_timestamp').first().reset_index()
                self.per_subject_dataset[source][subject_id] = copy.deepcopy(tmp_df)

    def get_utc_timestamp_range_for_subject_signal(self, subject_id, signal_name) -> Optional[Tuple[float, float]]:
        """
        Parameters
        ----------
        subject_id: `str`, required
        signal_name: `str`, required
            The signal name. For available options, please refer to the `target_source_and_column` in
             smartwatch metadata.

        Returns
        ----------
        If the time-range is found, then it will be returned as a tuple of two floats. Else, `None` will be returned.
        """
        source, _ = smartwatch_metadata.target_source_and_column[signal_name]

        sub_df = copy.deepcopy(self.per_subject_dataset[source][subject_id])

        timestamp_min = float(sub_df['utc_timestamp'].min())
        timestamp_max = float(sub_df['utc_timestamp'].max())

        if numpy.isnan(timestamp_min) or numpy.isnan(timestamp_max):
            return None
        else:
            if not isinstance(timestamp_min, float) or not isinstance(timestamp_max, float):
                raise ValueError(f'The timestamp range for subject {subject_id} and signal {signal_name} '
                                 f'is not a float: it is -> {timestamp_min}, {timestamp_max}')
            return timestamp_min, timestamp_max

    def get_utc_timestamp_range_for_signal(self, signal_name) -> Tuple[float, float]:
        """
        The UTC timestamp range for the signal.

        Parameters
        ----------
        signal_name: `str`, required
            Signal name

        Returns
        ----------
        `Tuple[float, float]`: If exception is raised, it means that there is no data is available for the signal.
        Else, the output would be the timestamp range in which there is data for the given signal.
        """
        timestamp_min = numpy.inf
        timestamp_max = -numpy.inf

        for subject_id in self.subject_id_list:
            t1, t2 = self.get_utc_timestamp_range_for_subject_signal(subject_id, signal_name)
            timestamp_min = min(timestamp_min, t1)
            timestamp_max = max(timestamp_max, t2)

        if numpy.isnan(timestamp_min) or numpy.isnan(timestamp_max) or numpy.isinf(timestamp_min) or numpy.isinf(timestamp_max):
            raise Exception("The timestamp range for signal {} is not valid: {}".format(signal_name, (timestamp_min, timestamp_max)))

        return timestamp_min, timestamp_max

    def get_utc_datetime_range_for_signal(self, signal_name: str) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Parameters
        ----------
        signal_name: `str`, required
            Signal name

        Returns
        ----------
        `Tuple[datetime.datetime, datetime.datetime]`: The date range in which there is some data available for some subject for the queried signal.
        """
        t_min, t_max = self.get_utc_timestamp_range_for_signal(signal_name=signal_name)
        return get_utc_date_from_utc_timestamp(t_min), get_utc_date_from_utc_timestamp(t_max)

    def get_utc_dates_for_subject_signal(self, subject_id: str, signal_name: str) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            Subject ID.
        signal_name: `str`, required
            The signal

        Returns
        ----------
        `Tuple[datetime.datetime, datetime.datetime]`: the list of utc dates which have data available for the queried subject id and
        for the corresponding signal.
        """
        sub_df = self.per_subject_dataset[signal_name].copy()
        sub_df = sub_df[sub_df['user_id'] == subject_id]
        return sub_df['utc_date'].apply(lambda x: datetime_parser.parse(x).date()).tolist()

    def get_utc_datetime_range_for_subject_signal(self, subject_id: str, signal_name: str) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            The subject id.
        signal_name: `str`, required
            The signal name, please refer to the specific metadata to get the options.

        Returns
        ----------
        `Tuple[datetime.datetime, datetime.datetime]`: the maximum date range for the subject id in input, for which the data is available.
        """
        t_min, t_max = self.get_utc_timestamp_range_for_subject_signal(subject_id=subject_id, signal_name=signal_name)
        return get_utc_date_from_utc_timestamp(t_min), get_utc_date_from_utc_timestamp(t_max)

    def get_utc_datetime_range_for_subject(self, subject_id: str) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            The subject id.

        Returns
        ----------
        `Tuple[datetime.datetime, datetime.datetime]`: the maximum date range for the subject id in input, for which the data is available.
        """
        t_min = numpy.inf
        t_max = -numpy.inf
        for signal in smartwatch_metadata.target_source_and_column.keys():
            t_min_tmp, t_max_tmp = self.get_utc_timestamp_range_for_subject_signal(subject_id=subject_id, signal_name=signal)
            t_min = min(t_min, t_min_tmp)
            t_max = max(t_max, t_max_tmp)

        if numpy.isnan(t_min) or numpy.isnan(t_max) or numpy.isinf(t_min) or numpy.isinf(t_max):
            raise Exception("this user has no data available")

        return get_utc_date_from_utc_timestamp(t_min), get_utc_date_from_utc_timestamp(t_max)

    def is_slice_empty(self, slice):
        for key in slice.keys():
            if slice[key].shape[0] > 0:
                return False
        return True

    def get_utc_timestamp_range_for_subject(self, subject_id: str) -> Optional[Tuple[datetime.datetime, datetime.datetime]]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            The subject id.

        Returns
        ----------
        `Tuple[float, float]`: the maximum utc timestamp range for the subject id in input, for which the data is available.
        """
        return [e.timestamp() for e in self.get_utc_datetime_range_for_subject(subject_id=subject_id)]

    def get_subject_signals_for_date(self, subject_id: str, date_string: str) -> Dict[str, pandas.DataFrame]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            Subject ID.
        date_string: `str`, required
            The date in 'yyyy-mm-dd' format.
        """
        t_min = datetime_parser.parse(date_string).replace(tzinfo=datetime.timezone.utc).timestamp()
        t_max = t_min + 24 * 60 * 60
        return self.get_subject_signals_for_utc_timestamp_range(subject_id=subject_id, timestamp_range=(t_min, t_max))

    def get_subject_signals_for_utc_timestamp_range(self, subject_id, timestamp_range: Tuple[float, float]) -> Dict[str, pandas.DataFrame]:
        """
        For a given timestamp range, this method will return all the subject data

        Parameters
        ----------
        subject_id: `str`, required
            Subject id for whom the data is queried.
        timestamp_range: `Tuple[float, float]`, required
            The timestamp range, minimum first, followed by the maximum. The query applies to [t_min, t_max) range.

        Returns
        ----------
        `Dict[str, Any]`: The dictionary of time range data for the subject.
            The keys would be the available data sources, and the contents for each pertains only to the queried subject.
        """
        assert timestamp_range[1] > timestamp_range[0], "bad timestamp range"
        output = dict()
        for source in self.per_subject_dataset:
            if source in ['activityDetail']:  # s: add the sleep features to the system as well.
                continue
            df = copy.deepcopy(self.per_subject_dataset[source][subject_id])
            df = df[
                (df['utc_timestamp'] >= timestamp_range[0]) & (df['utc_timestamp'] < timestamp_range[1])
                ]
            output[source] = copy.deepcopy(df)
        return output

    def get_subject_signals_for_utc_datetime_range(self, subject_id: str, datetime_range: Tuple[datetime.datetime, datetime.datetime]) -> Dict[str, pandas.DataFrame]:
        """
        For a given timestamp range, this method will return all the subject data

        Parameters
        ----------
        subject_id: `str`, required
            Subject id for whom the data is queried.
        datetime_range: `Tuple[datetime.datetime, datetime.datetime]`, required
            The datetime range. Note that no matter what timezone these objects have, they are treated as naive
            utc objects, and their timezone will be replaced with `datetime.timezone.utc`. Thus, correct use
            is caller's responsibility.

        Returns
        ----------
        `Dict[str, Any]`: The dictionary of time range data for the subject.
            The keys would be the available data sources, and the contents for each pertains only to the queried subject.
        """
        t_min = datetime_range[0].replace(tzinfo=datetime.timezone.utc).timestamp()
        t_max = datetime_range[1].replace(tzinfo=datetime.timezone.utc).timestamp()
        return self.get_subject_signals_for_utc_timestamp_range(subject_id=subject_id, timestamp_range=(t_min, t_max))

    def get_all_data_for_subject(self, subject_id: str) -> Dict[str, pandas.DataFrame]:
        """
        Parameters
        ----------
        subject_id: `str`, required
            The subject id for whom we want all of the available data.

        Returns
        ----------
        `Dict[str, Any]`: The dictionary of all the data for the subject.
            The keys would be the available data sources, and the contents for each pertains only to the queried subject.
        """
        return copy.deepcopy({x: self.per_subject_dataset[x][subject_id] for x in self.per_subject_dataset.keys()})

    @staticmethod
    def unfold_timeseries_recordings(
            df: pandas.DataFrame,
            fields: Dict[str, str],
            drop_columns: List[str] = None,
            nan_ffill: bool = False,
            nan_bfill: bool = False
    ) -> pandas.DataFrame:
        """
        The dataframe

        Parameters
        ----------
        df: `pandas.DataFrame`, required
            The garmin dataframe

        fields: `Dict[str, str]`, required
            The fields map, the keys being columns that include time_series_bundles, and the keys will be the names for unfolded timeseries.
            Please note that we have assertion that all the mapped names end in "_tsvalue" for consistency.

        drop_columns: `List[str]`, optional:
            If provided, columns will be dropped.

        nan_ffill: `bool`, optional (default False)
            The option to forward fill the output dataframe

        nan_bfill: `bool`, optional (default False)
            The option to backward fill the output dataframe

        Returns
        ----------
        `pandas.DataFrame`: The dataframe with the timeseries unfolded. Please note that
        this method will NOT fill the NaNs, unless specifically requested.
        """
        if drop_columns is not None:
            df.drop(columns=drop_columns, inplace=True)
        if 'sleepLevelsMap' in df.columns:
            df.drop(columns=['sleepLevelsMap'], inplace=True)
        timestamp_field = 'utc_timestamp'
        if fields is None:
            return df

        if df.shape[0] == 0:
            return df.rename(fields, axis=1, errors='raise')

        # - taking a copy
        df = copy.deepcopy(df)

        internal_timeseries_dfs = defaultdict(lambda: [])

        # bringing out the fields one by one
        for i in range(df.shape[0]):
            row = df.iloc[i]
            baseline_time = row['startTimeInSeconds'] + row['startTimeOffsetInSeconds']
            for input_field, output_field in fields.items():
                if pandas.isna(row[input_field]):
                    continue
                t = sorted([int(e) for e in row[input_field].keys()])
                v = [row[input_field][str(e)] for e in t]
                tmp_df = pandas.DataFrame({timestamp_field: [baseline_time + e for e in t], output_field: v})
                for c in df.columns:
                    if c in ['utc_timestamp'] + list(fields.keys()) + list(fields.values()):
                        continue
                    else:
                        tmp_df[c] = [row[c]] * len(t)
                internal_timeseries_dfs[output_field].append(copy.deepcopy(tmp_df))

        # - now we need to prepare all these datasets and merge.
        for key in internal_timeseries_dfs.keys():
            internal_timeseries_dfs[key] = pandas.concat(internal_timeseries_dfs[key])

        def combine(df_x, df_y):
            merge_on = list(set(df_x.columns).intersection(df_y.columns))
            merge_on = [e for e in merge_on if e not in fields.keys()]
            return pandas.merge(left=df_x, right=df_y, on=merge_on, how='outer')

        output_df = functools.reduce(combine, internal_timeseries_dfs.values())

        output_df = output_df.sort_values(by=[timestamp_field]).reset_index(drop=True)
        output_df['utc_date'] = output_df['utc_timestamp'].copy().apply(lambda x: str(get_utc_date_from_utc_timestamp(x)))

        if nan_ffill:
            output_df.ffill()

        if nan_bfill:
            output_df.bfill()

        return output_df
