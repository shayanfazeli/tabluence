from typing import List, Dict, Any
import torch
import torch.utils.data.dataloader
from tabluence.data.api.smartwatch.data_manager import SmartwatchDataManager
from tabluence.deep_learning.data.dataset.smartwatch_study.single_slice import SmartwatchStudySingleSliceDataset, \
    BalancedStressLevelSampler, single_slice_collate_fn

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_dataloaders(
        batch_size: int,
        root_dir: str,
        subject_splits: Dict[str, List[str]],
        dataset_config: Dict[str, Any],
        sampler_configs: Dict[str, Dict[str, Any]],
        data_manager_cache_filepath: str
) -> Dict[str, torch.utils.data.dataloader.DataLoader]:
    """
    Parameters
    ----------
    batch_size: `int`, required
        The batch size to use.
    root_dir: `str`, required
        The root directory in which the dump files for the Smartwatch study reside.

    subject_splits: `Dict[str, List[str]]`, required
        The subject splits. Its keys can be either 'train', 'validation', or 'test', and the contents
        are the subject IDs used for that split.

    dataset_config: `Dict[str, Any]`, required
        The dataset configuration, please refer to the :cls:`SmartwatchStudySingleSliceDataset` documentation.

    sampler_configs: `Dict[str, Dict[str, Any]]`, required
        The sampler configurations, one per each split.

    data_manager_cache_filepath: `str`, required
        The filepath to the data manager cache file. If provided, the data manager will be loaded from
        that file (or written to it).

    Returns
    -------
    `Dict[str, torch.utils.data.dataloader.DataLoader]`: the dataloaders per required split.
    """
    # - sanity checks
    if 'timeline_portions' not in sampler_configs['train']:
        enforce_disjoint_splits = True
    elif sampler_configs['train']['timeline_portions'] is None:
        enforce_disjoint_splits = True
    else:
        enforce_disjoint_splits = False

    if enforce_disjoint_splits:
        for split_name, split_ids in subject_splits.items():
            for split_name2, split_ids2 in subject_splits.items():
                if split_name != split_name2:
                    assert len(set(split_ids).intersection(set(split_ids2))) == 0, "bad subject splits"

    for k in sampler_configs:
        assert k in subject_splits, "bad sampler configs"

    for k in subject_splits:
        assert k in sampler_configs, "bad subject splits"

    # - preparations
    all_subjects = []
    for split_ids in subject_splits.values():
        all_subjects += split_ids

    # - enforcing uniqueness of subject ids
    all_subjects = list(set(all_subjects))

    # - creating the data manager
    logger.info("initializing data manager...")
    data_manager = SmartwatchDataManager(
        root_dir=root_dir,
        subject_id_list=all_subjects,
        cache_filepath=data_manager_cache_filepath
    )

    # - creating the dataset (full)
    logger.info("preparing the dataset...")
    dataset_config.update({"data_manager": data_manager})
    dataset = SmartwatchStudySingleSliceDataset(**dataset_config)

    # - creating the samplers
    logger.info("preparing samplers...")
    samplers = dict()
    for x in sampler_configs:
        sampler_configs[x].update({"dataset": dataset, "subject_ids": subject_splits[x]})
        samplers[x] = BalancedStressLevelSampler(**sampler_configs[x])

    return {
        x: torch.utils.data.DataLoader(
            dataset,
            sampler=samplers[x],
            batch_size=batch_size,
            collate_fn=single_slice_collate_fn
        ) for x in sampler_configs
    }
