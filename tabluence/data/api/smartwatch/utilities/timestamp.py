from datetime import datetime, tzinfo, timezone
from typing import Union
import dateutil.parser as datetime_parser


def get_utc_timestamp_from_naive_datetime(dt: Union[datetime, str]) -> float:
    """
    Parameters
    ----------
    dt: `datetime`, required
        The datetime object for the smartwatch study. One critical thing to note
        here is that, we IGNORE the timestamp associated with this datetime. The idea is,
        we treat everything as UTC (our smartwatch api was designed to generate startTimeInSeconds+
        startTimeOffsetInSeconds, the sum of which would represent `local` time BUT in UTC timezone.

    Returns
    -------
    `float`:
        The timestamp of the given datetime object.
    """
    if isinstance(dt, str):
        dt = datetime_parser.parse(dt)
    return dt.replace(tzinfo=timezone.utc).timestamp()


def get_utc_date_from_utc_timestamp(utc_timestamp: float) -> datetime:
    """
    Parameters
    ----------
    utc_timestamp: `float`, required
        This is the UTC timestamp. In the context of our smartwatch study, for instance,
        the timezone is irrelevant and it basically means that if you treat the timestamp
        as UTC timestamp, resulting parsed datetime will indicate the local time.

    Returns
    -------
    `datetime`:
        The datetime object of the given UTC timestamp with `tzinfo=timezone.utc`.
    """
    return datetime.utcfromtimestamp(utc_timestamp).replace(tzinfo=timezone.utc)
