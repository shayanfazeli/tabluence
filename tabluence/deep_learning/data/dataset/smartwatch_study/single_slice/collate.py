from typing import List, Dict, Any


def single_slice_collate_fn(batch: List[Any]) -> Dict[str, Any]:
    """
    Slice collate function

    Parameters
    ----------
    batch : `List[Any]`, required
        List of items in the batch
    """
    output = dict()
    output['meta'] = [e['meta'] for e in batch]
    output['slice'] = [e['slice'] for e in batch]
    return output
