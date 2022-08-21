from typing import List, Tuple, Dict
import numpy
import torch
import torch.nn
import torch.nn.functional
import torch.nn.utils.rnn


def pack_single_slice_batch_for_rnn(
        batch: Dict[str, List[torch.Tensor]],
        data_sources_to_pack: List[str]) -> Dict[str, torch.nn.utils.rnn.PackedSequence]:
    """
    Parameters
    ----------
    batch: `Dict[str, List[torch.Tensor]]`, required
        A single slice batch, containing 'slice' and 'meta' keys.

    data_sources_to_pack: `List[str]`, required
        The list of data sources to pack.

    Returns
    -------
    `Dict[str, torch.nn.utils.rnn.PackedSequence]`: The packed sequence
    """
    packed_batch = dict()
    for data_source in data_sources_to_pack:
        packed_batch[data_source] = pack_batch_for_rnn(item_list=[e[data_source] for e in batch['slice']])
    return packed_batch


def pad_packed_sequence_for_rnn(packed_sequence: torch.nn.utils.rnn.PackedSequence) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of the `pack_batch_for_rnn` function.

    Parameters
    ----------
    packed_sequence: `torch.nn.utils.rnn.PackedSequence`, required
        The packed sequence to be unpacked.

    Returns
    -------
    `Tuple[torch.Tensor, torch.Tensor]`: The unpacked sequence and the sequence lengths.
    """
    unpacked_sequence, unpacked_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence, batch_first=True)
    return unpacked_sequence, unpacked_lengths


def pack_batch_for_rnn(item_list: List[torch.Tensor]) -> torch.nn.utils.rnn.PackedSequence:
    """
    Packs a batch of sequences for RNN.

    Parameters
    ----------
    item_list: `List[torch.Tensor]`, required
        The list of tensors, each of which is a sequence of shape `(sequence_length, rep_dim)`.

    Returns
    -------
    `torch.nn.utils.rnn.PackedSequence`: The packed sequence
    """
    # - preparing a placeholder for the sequence lengths
    device = item_list[0].device
    sequence_lengths = []

    # - preparing a placeholder for the output
    batch_tensor = []
    max_sequence_length = int(numpy.max([x.shape[0] for x in item_list]))
    rep_dim = item_list[0].shape[-1]
    for x in item_list:
        seq_len = x.shape[0]
        if seq_len > 0:
            batch_tensor.append(
                torch.cat((x, torch.zeros(max_sequence_length - seq_len, rep_dim).to(device)))
            )
            sequence_lengths.append(seq_len)
        else:
            batch_tensor.append(- torch.ones(max_sequence_length, rep_dim).to(device))
            sequence_lengths.append(1)

    batch_tensor = torch.stack(batch_tensor)
    sequence_lengths = torch.LongTensor(sequence_lengths)
    packed_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(batch_tensor, sequence_lengths, batch_first=True, enforce_sorted=False)
    return packed_padded_sequence


