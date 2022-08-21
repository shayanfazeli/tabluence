import datetime
import numpy
from typing import Dict, Any, Callable


def stress_poke_function_gaussian_1(stress_label: Dict[str, Any]) -> Callable[[float], float]:
    """
    Given a stress label, it looks through the `probe_timestamp` and returns a function that
    models the stress around the time of the probe.
    Please note that currently, the perceived rate values are hard coded (3 for high, 1 for low, 2 for medium).

    The model is a gaussian, with std being 30-minutes for single timestamp probes, and
    30-minutes * (period / 30-minutes) for timespans (determined by two timestamps) for double timestamp probes.

    Parameters
    ----------
    stress_label: `Dict[str, Any]`, str
        The stress label to be used for the function.
        An instance:

        ```
        {
            "subject_id": "subject_id",
            "stress_type": "induced",
            "stress_type_2": "induced",
            "perceived_rate": "low",
            "stress_rate": "none",
            "stress_description": "description",
            "probe_datetime": datetime(2021, 3, 8, 12, 30, 0, tzinfo=timezone.utc)
        }
        ```

    Returns
    -------
    `Callable[[float], float]`: the mathematical lambda function modeling the impact of this label.
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
