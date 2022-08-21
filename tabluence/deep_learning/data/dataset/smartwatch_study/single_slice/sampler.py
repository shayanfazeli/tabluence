from typing import Iterator, Dict, List, Any
import torch
import torch.utils.data.sampler
import numpy

from tabluence.deep_learning.data.dataset.smartwatch_study.single_slice.dataset import SmartwatchStudySingleSliceDataset

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




class BalancedStressLevelSampler(torch.utils.data.sampler.Sampler):
    """
    We want to build the probability distribution of retaining an element from each one of the bins.

    Thus, we will follow the approach below:
    * First, we get the metadata and datapoint_counts obtained from the fully instantiated sample of
    the :cls:`SmartwatchStudySingleSliceDataset`.
    * We get the metadata and stats from the dataset object.
    * The quantized value of overall-stress (or any other value passed as `target_variable`) will be used as label here
    * For each label, we will sample (with replacement) a total of `positive_sample_count` (if non-zero) samples,
    and of course `negative_sample_count` items for the negative class.
    * We will randomly permute the resulting indices and then use them.

    Parameters
    ----------
    subject_ids: `List[str]`, required
        The split in this dataset is carried out in the sampler, thus, the list
        of subjects used in the dataset should be provided here.
    dataset: `SmartwatchStudySingleSliceDataset`, required
        The dataset object.
    target_variable: `str`, required
        The name of the variable to use as label.
    positive_sample_count: `int`, required
        The number of samples to retain for the positive class.
    negative_sample_count: `int`, required
        The number of samples to retain for the negative class.
    """
    def __init__(
            self,
            subject_ids: List[str],
            negative_sample_count: int,
            positive_sample_count: int,
            dataset: SmartwatchStudySingleSliceDataset,
            target_variable: str = 'overall_quantized_stress_value',
            split_name: str = 'not_given',
            timeline_portions: Dict[str, float] = None,
    ) -> None:
        """
        constructor
        """
        assert target_variable in dataset.metadata[0].keys(), "the chosen target variable (=%s) is not in the metadata" % target_variable
        self.metadata = dataset.metadata
        self.index_mask = numpy.array([e['subject_id'] in subject_ids for e in self.metadata], dtype=bool)
        self.negative_sample_count = negative_sample_count
        self.positive_sample_count = positive_sample_count
        assert split_name in ['train', 'validation', 'test'], "the split name should be one of 'train', 'validation', 'test'"

        if timeline_portions is not None:
            assert numpy.sum(list(timeline_portions.values())) == 1.0, "the timeline portions should sum to 1.0"
            self.timeline_portions = timeline_portions
            subject_timeline_mask = numpy.array([False for _ in self.metadata], dtype=bool)
            for subject_id in subject_ids:
                # try:
                t_start, t_end = dataset.data_manager.get_utc_timestamp_range_for_subject(subject_id)
                train_ratio = timeline_portions['train']
                if 'validation' not in timeline_portions:
                    val_ratio = 0.0
                else:
                    val_ratio = timeline_portions['validation']
                full_len = t_end - t_start
                if split_name == 'train':
                    chosen_window = (t_start, t_start + full_len * train_ratio)
                elif split_name == 'validation':
                    chosen_window = (t_start + full_len * train_ratio, t_start + full_len * (train_ratio + val_ratio))
                elif split_name == 'test':
                    chosen_window = (t_start + full_len * (train_ratio + val_ratio), t_end)
                else:
                    raise ValueError("the split name should be one of 'train', 'validation', 'test'")

                self.chosen_window = chosen_window

                tmp_mask = []
                for meta in self.metadata:
                    if meta['subject_id'] == subject_id:
                        intersection = get_timespan_overlap(meta['utc_timestamp_window'], chosen_window)
                        tmp_mask.append((intersection / (float(meta['utc_timestamp_window'][1] - meta['utc_timestamp_window'][0]))) == 1.0)
                    else:
                        tmp_mask.append(False)
                subject_timeline_mask |= numpy.array(tmp_mask, dtype=bool)
                # except Exception as e:
                #     logger.warning(f"subject {subject_id} does not have any"
                #                    f" windows in the {split_name} split. skipping...")

            self.index_mask &= subject_timeline_mask
        else:
            self.timeline_portions = None

        quantized_stress_value_layout = sorted(list(set([e[target_variable] for e in self.metadata])))
        labels_per_index = numpy.array([quantized_stress_value_layout.index(e[target_variable]) for e in self.metadata])

        indices = []
        for label_index in range(len(quantized_stress_value_layout)):
            possible_indices = numpy.nonzero(self.index_mask & (labels_per_index == label_index))[0]
            if possible_indices.sum() == 0:
                logger.warning(f"no samples for label {quantized_stress_value_layout[label_index]} was found in the split: {split_name}")
            else:
                if self.negative_sample_count is not None and self.positive_sample_count is not None:
                    tmp_indices = numpy.random.choice(
                        possible_indices,
                        self.negative_sample_count if label_index == 0 else self.positive_sample_count,
                        replace=True
                    )
                    indices.append(tmp_indices)
                else:
                    assert self.negative_sample_count is None and self.positive_sample_count is None, "if negative_sample_count and positive_sample_count are None, then they should be both None"
                    indices.append(possible_indices)

        indices = numpy.concatenate(indices, axis=0)

        # - permutation
        random_perm = numpy.random.permutation(indices.shape[0])
        indices = indices[random_perm]

        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


def get_timespan_overlap(x1, x2):
    if x1[1] <= x2[0] or x2[1] <= x1[0]:
        return 0
    intersection = min(x2[1], x1[1]) - max(x2[0], x1[0])

    return intersection
