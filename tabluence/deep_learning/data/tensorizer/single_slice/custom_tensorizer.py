import copy
from typing import Dict, Any, List, Tuple
import collections
from tqdm import tqdm
import torch
import torch.nn
import torch.utils.data.dataloader
from tabluence.deep_learning.data.tensorizer.single_slice.base import SingleSliceTensorizerBase


class CustomTensorizer(SingleSliceTensorizerBase):
    """
    The :cls:`CustomTensorizer` class is used to make sequence of tensors ready to be fed
    into PyTorch models, obtained from the single slice dataset.


    The tensorization configuration. Example:
    ```
    {
        'timestamp_column': 'utc_timestamp',
        'value_config': {
            'daily': {
                'embed': {
                    'columns': ['heart_rate_tsvalue'],
                    'embedding_dim': 10,
                },
                'bring': ['heart_rate_tsvalue']
            }
        }
    }
    ```
    """
    def __init__(self, *args, **kwargs):
        """constructor"""
        super(CustomTensorizer, self).__init__(*args, **kwargs)
        self.embedding_layouts = None

    def tensorize_single(self, slice_data, meta_data) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        tensorization of a single item within the batch of slices.
        Parameters
        ----------
        slice_data: `Dict[str, pandas.DataFrame]`, required
            The data of a single slice.
        meta_data: `Dict[str, Any]`, required
            The meta data of a single slice.

        Returns
        ----------
        `Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]`: The batch item after tensorization (in almost all cases, the list item is a torch.Tensor).
        """
        new_timestamp_data = dict()
        new_slice_data = dict()
        new_meta_data = meta_data
        for data_source_name in slice_data.keys():
            if data_source_name not in self.config['value_config']:
                continue
            slice_data[data_source_name].sort_values(by=self.config['timestamp_column'], inplace=True)
            data_source_config = self.config['value_config'][data_source_name]
            sequence_reps = []
            timestamps = torch.from_numpy(slice_data[data_source_name][self.config['timestamp_column']].to_numpy()).to(self.device)
            sequence_reps += [torch.from_numpy(slice_data[data_source_name][column].to_numpy().astype('float')).unsqueeze(-1).to(self.device) for column in data_source_config['bring']]
            if 'embed' in data_source_config:
                for column_to_be_embedded in data_source_config['embed']['columns']:
                    categorizer = lambda x: self.embedding_layouts[f"{data_source_name}___{column_to_be_embedded}"].index(x)
                    category_indices = torch.from_numpy(slice_data[data_source_name].loc[:, column_to_be_embedded].apply(categorizer).to_numpy().astype('int')).long().to(self.device)
                    sequence_reps += [getattr(self, f"embedding_{data_source_name}___{column_to_be_embedded}")(category_indices)]
            if len(sequence_reps) > 1:
                sequence_reps = torch.cat(sequence_reps, dim=1)
            elif len(sequence_reps) == 1:
                sequence_reps = sequence_reps[0]
            else:
                sequence_reps = None
            new_slice_data[data_source_name] = sequence_reps
            new_timestamp_data[data_source_name] = timestamps
        return new_slice_data, new_timestamp_data, new_meta_data

    def tensorize(self, batch) -> Dict[str, List[Any]]:
        """
        Parameters
        ----------
        batch: `Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The collated batch of a single-slice dataloader.

        Returns
        ----------
        `Dict[str, List[Any]]`: The batch after tensorization (in almost all cases, the list item is a torch.Tensor).
        """
        batch = copy.deepcopy(batch)
        modified_batch = {
            'slice': [],
            'meta': [],
            'timestamp': []
        }
        for meta_data, slice_data in zip(batch['meta'], batch['slice']):
            new_slice_data, new_timestamp_data, new_meta_data = self.tensorize_single(slice_data, meta_data)
            modified_batch['slice'] += [{x: new_slice_data[x].to(self.device) for x in new_slice_data.keys()}]
            modified_batch['meta'] += [new_meta_data]
            modified_batch['timestamp'] += [{x: new_timestamp_data[x].to(self.device) for x in new_timestamp_data.keys()}]

        return modified_batch

    def sanity_check(self) -> None:
        # - for configuration:
        assert 'timestamp_column' in self.config, '"timestamp_column" is missing in the configuration'
        assert 'value_config' in self.config, '"value_config" is missing in the configuration'
        assert isinstance(self.config['value_config'], dict), '"value_config" must be a dictionary'
        for ds_name, ds_config in self.config['value_config'].items():
            if 'embed' in ds_config:
                assert len(ds_config['embed']['columns']) == len(ds_config['bring']), "the embedding dims are not provided for all columns"
                for i, k in enumerate(ds_config['embed']['embedding_dim']):
                    assert k > 1, f"the embedding dims must be greater than 1. It is not for the column {ds_config['embed']['columns'][i]}"

    def build_embeddings(self) -> None:
        """
        Builds the embedding modules for the columns that are to be embedded.
        """
        assert self.embedding_layouts is not None, "the embeddings layouts are not built, please run `learn` first"
        for key in self.embedding_layouts:
            data_source_name, feature_name = key.split('___')
            ds_config = self.config['value_config'][data_source_name]
            embedding_dim = ds_config['embed']['embedding_dim'][ds_config['embed']['columns'].index(feature_name)]
            self.add_module(f"embedding_{data_source_name}___{feature_name}", torch.nn.Embedding(len(self.embedding_layouts[key]), embedding_dim))

    def get_embedding_layout(self, data_source_name: str, feature_name: str) -> List[str]:
        """
        getting the embedding layout for a specific feature.

        Parameters
        ----------
        data_source_name: `str`, required
            The name of the data source.
        feature_name: `str`, required
            The name of the feature.

        Returns
        ----------
        `List[str]`: The embedding layout for the feature associated with the given data source.
        """
        assert self.embedding_layouts is not None, "run `learn` first"
        return self.embedding_layouts[f"{data_source_name}___{feature_name}"]

    def learn(self, dataloader: torch.utils.data.dataloader.DataLoader) -> None:
        """
        Parameters
        ----------
        dataloader: `torch.utils.data.dataloader.DataLoader`, required
            A single slice dataset's dataloader. Please note that this is the dataloader that will be used
            to learn all possible values for those columns to be embedded (`None`, corresponding to `pandas.nan`, is included as well).

            When this process is done, the layouts themselves can also be accessed by calling :meth:`get_embedding_layout`.
        """

        there_is_embedding_to_learn = False
        for ds in self.config['value_config'].keys():
            if 'embed' in self.config['value_config'][ds]:
                there_is_embedding_to_learn = True

        if not there_is_embedding_to_learn:
            return

        embedding_layouts = collections.defaultdict(lambda: set())

        for batch in tqdm(dataloader):
            batch = copy.deepcopy(batch)
            for slice_data in batch['slice']:
                for ds in slice_data.keys():
                    if ds in self.config['value_config']:
                        if 'embed' in self.config['value_config'][ds].keys():
                            for feature_name_to_be_embedded in self.config['value_config'][ds]['embed']['columns']:
                                embedding_layouts[ds +'___' + feature_name_to_be_embedded] = embedding_layouts[ds +'___' + feature_name_to_be_embedded].union(
                                    slice_data[ds][feature_name_to_be_embedded].unique().tolist())

        self.embedding_layouts = dict()
        for key in embedding_layouts.keys():
            self.embedding_layouts[key] = list(embedding_layouts[key])

        self.build_embeddings()
