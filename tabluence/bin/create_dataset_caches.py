#!/usr/bin/env python
import os
from tabluence.data.api.smartwatch.data_manager.module import SmartwatchDataManager
from tabluence.deep_learning.data.dataset.smartwatch_study.single_slice import SmartwatchStudySingleSliceDataset, single_slice_collate_fn


def main():
    root = os.path.abspath('../../resources/warrior_wellness/Analysis/local_repo/')
    data_manager = SmartwatchDataManager(
        root_dir = root,
        subject_id_list=[f'SWS_{i:02d}' for i in range(0,15)]
    )

    dataset_cache_meta = [
        # {
        #     'name': 'dataset-cache-1',
        #     'args': {
        #         'data_manager': data_manager,
        #         'slice_lengths': [3600], #numpy.arange(1*3600, 2*3600, 15*60).tolist(),
        #         'slice_time_step': (1),
        #         'label_milestone_per_window': 1.0,
        #         'metadata_cache_filepath': './dataset_cache/dataset-cache-1.pkl.gz',
        #         'no_cache': False,
        #         'parallel_threads': 10
        #     }
        # },
        {
            'name': 'dataset-cache-2',
            'args': {
                'data_manager': data_manager,
                'slice_lengths': [3600], #numpy.arange(1*3600, 2*3600, 15*60).tolist(),
                'slice_time_step': (5 * 60),
                'label_milestone_per_window': 1.0,
                'metadata_cache_filepath': './dataset_cache/dataset-cache-2.pkl.gz',
                'no_cache': False,
                'parallel_threads': 10
            }
        },
    ]

    for i, ds_meta in enumerate(dataset_cache_meta):
        _ = SmartwatchStudySingleSliceDataset(**ds_meta['args'])
        print("{}/{} dataset is processed ***.".format(i+1, len(dataset_cache_meta)))


if __name__ == "__main__":
    # todo: make the arguments in argparse format
    main()
