import collections
import copy
import functools
from typing import List, Dict, Any
import pandas

from tabluence.deep_learning.data.pipeline.fusion.base import DataSideFusionBase


class SliceToSliceFusion(DataSideFusionBase):
    """
    The :cls:`SliceToSliceFusion` class is a data side fusion that allows slice manipulation. This
    can be used to turn slices into newly designed slices, without having to save them elsewhere.
    The operation takes place on the fly.

    Here is an example configuration for this module
    ```
    {
        'timestamp_column': 'utc_timestamp',
        'sources': {
            'hr_and_pulseox': {
                'daily': ['heart_rate_tsvalue'],
                'pulseOX': ['spo2_tsvalue'],
            }
        },
        'nan_fill_method': ['ffill', 'bfill'],
        'drop_remaining_nans': True
    }
    ```
    Using the above, the new slice that will be returned will contain the sources defined in the `sources` key,
    and the columns would come from the designated columns of the sources from the original slice.


    """
    def __init__(
            self,
            *args,
            **kwargs
    ):
        """constructor"""
        super(SliceToSliceFusion, self).__init__(*args, **kwargs)
        self.config = self.config

    def fuse(self, batch: Dict[str, List[Dict[str, pandas.DataFrame]]]) -> Dict[str, List[Dict[str, pandas.DataFrame]]]:
        """
        Parameters
        ----------
        batch: ` Dict[str, List[Dict[str, pandas.DataFrame]]]`, required
            The collated batch of a single-slice dataloader.

        Returns
        -------
        `Dict[str, List[Dict[str, pandas.DataFrame]]]`: the fused slice
        """
        # - taking a copy of the batch
        batch = copy.deepcopy(batch)

        timestamp_column = self.config['timestamp_column']

        batch_slice_new = []
        for i in range(len(batch['slice'])):
            item_slice_new = collections.defaultdict(lambda: [])
            for new_source_name, new_source_info in self.config['sources'].items():
                new_source_columns = [timestamp_column]
                for column_name_list in new_source_info.values():
                    new_source_columns += column_name_list
                for orig_name, orig_columns in new_source_info.items():
                    item_slice_new[new_source_name].append(batch['slice'][i][orig_name].loc[:, [timestamp_column] + orig_columns])

                if len(item_slice_new[new_source_name]) > 1:
                    item_slice_new[new_source_name] = functools.reduce(lambda x, y: pandas.merge(x, y, on=timestamp_column, how='outer'), item_slice_new[new_source_name])
                elif len(item_slice_new[new_source_name]) == 1:
                    item_slice_new[new_source_name] = item_slice_new[new_source_name][0]
                else:
                    item_slice_new[new_source_name] = pandas.DataFrame(columns=new_source_columns)
                item_slice_new[new_source_name].sort_values(by=timestamp_column, inplace=True)
                if 'ffill' in self.config['nan_fill_method']:
                    item_slice_new[new_source_name].fillna(method='ffill', inplace=True)
                if 'bfill' in self.config['nan_fill_method']:
                    item_slice_new[new_source_name].fillna(method='bfill', inplace=True)

                constant_fill_strategy = [strategy for strategy in self.config['nan_fill_method'] if strategy.startswith('fill_constant_')]
                assert len(constant_fill_strategy) <= 1, 'Only one constant fill strategy can be used at a time'
                if len(constant_fill_strategy) == 1:
                    constant_fill_strategy = constant_fill_strategy[0]
                    fill_value = float(constant_fill_strategy.split('_')[-1])
                    item_slice_new[new_source_name].fillna(value=fill_value, inplace=True)
                elif 'drop_remaining_nans' in self.config:
                    if self.config['drop_remaining_nans']:
                        item_slice_new[new_source_name].dropna(inplace=True)
            batch_slice_new.append(item_slice_new)

        batch['slice'] = batch_slice_new
        return batch
