from typing import List, Dict, Any, Callable
import math
import numpy
from datetime import datetime


def stress_poke_function_gaussian_1(stress_label: Dict[str, Any]) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """
    Parameters
    ----------
    stress_label: `Dict[str, Any]`, required
        A stress labels, which is in the following format:
        Example:
        ```
        {
            "subject_id": "id",
            "stress_type": "induced",
            "perceived_rate": "low",
            "stress_rate": "none",
            "stress_description": "stress_description",
            "probe_datetime": datetime(2021, 3, 18, 12, 30, 0, tzinfo=timezone.utc)
        },
        ```

    Returns
    ----------
    `Callable[[numpy.ndarray], numpy.ndarray]`
        The mathematical function corresponding to this individual stress indication.
    """
    if isinstance(stress_label['probe_timestamp'], tuple):
        mean = numpy.floor(numpy.mean(stress_label['probe_timestamp']))
        std = 30 * 60 * min(1, (stress_label['probe_timestamp'][1] - stress_label['probe_timestamp'][0]) / (30 * 60))
    else:
        mean = stress_label['probe_timestamp']
        std = 30 * 60  # 30-min in seconds

    perceived_rate = stress_label['perceived_rate']
    if perceived_rate == 'high':
        severity_coefficient = 3.0
    elif perceived_rate == 'medium':
        severity_coefficient = 2.0
    else:
        severity_coefficient = 1.0

    return lambda x: severity_coefficient * numpy.exp(-((x - mean) ** 2) / (2 * (std ** 2))) #* (1. / numpy.sqrt(2 * math.pi * std))


def get_subject_general_stress_function(subject_id: str, stress_labels: List[Dict[str, Any]]):
    """
    Parameters
    ----------
    subject_id: `str`, required
        The subject id.

    stress_labels: `List[Dict[str, Any]]`, required
        List of stress labels, each element of which is in the following format:
        Example:
        ```
        {
            "subject_id": "id",
            "stress_type": "induced",
            "perceived_rate": "low",
            "stress_rate": "none",
            "stress_description": "stress_description",
            "probe_datetime": datetime(2021, 3, 18, 12, 30, 0, tzinfo=timezone.utc)
        },
        ```

    Returns
    ----------
    `Callable[[numpy.ndarray], numpy.ndarray]`
        The mathematical function corresponding to the general stress indication for a subject.

    __Remark__: It is caller's responsibility to ensure that the stress_labels includes only the stress types
    that are relevant. Every provided stress label will be used.
    """
    output = [lambda x: numpy.zeros(x.shape[0])]
    for stress_label in stress_labels:
        if stress_label['subject_id'] == subject_id:
            output.append(stress_poke_function_gaussian_1(stress_label))

    return lambda x: numpy.sum(numpy.array([f(x) for f in output]), axis=0)


def get_subject_specific_stress_function(
        subject_id: str,
        stress_labels: List[Dict[str, Any]],
        stress_types: List[str] = [
            "physical_duress",
            "social_duress",
            "work_duress"
        ]
) -> Callable[[numpy.ndarray], numpy.ndarray]:
    """
    Parameters
    ----------
    subject_id: `str`, required
        The subject id.

    stress_labels: `List[Dict[str, Any]]`, required
        List of stress labels, each element of which is in the following format:
        Example:
        ```
        {
            "subject_id": "id",
            "stress_type": "induced",
            "perceived_rate": "low",
            "stress_rate": "none",
            "stress_description": "stress_description",
            "probe_datetime": datetime(2021, 3, 18, 12, 30, 0, tzinfo=timezone.utc)
        },
        ```

    stress_types: `List[str]`, optional (default=[
        "physical_duress",
        "social_duress",
        "work_duress"])
        The stress types to be considered.

    Returns
    ----------
    `Callable[[numpy.ndarray], numpy.ndarray]`
        The mathematical function corresponding to the fine-grained specifics stress indication for a subject.
    """
    output = {stress_type: get_subject_general_stress_function(
        subject_id,
        [e for e in stress_labels if e['stress_type'] == stress_type]
    ) for stress_type in stress_types}

    return lambda x: {t: output[t](x) for t in output.keys()}
