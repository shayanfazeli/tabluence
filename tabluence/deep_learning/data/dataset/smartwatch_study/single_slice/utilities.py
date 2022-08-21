from typing import Dict, Any
import hashlib
import json


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """
    MD5 hash of a dictionary.
    From [this source](https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html)

    Parameters
    ----------
    dictionary : `Dict[str, Any]`, required
        The dictionary to hash

    Returns
    -------
    `str`: The MD5 hash of the given dict.
    """
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()