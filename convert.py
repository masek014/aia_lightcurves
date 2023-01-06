from datetime import datetime, timezone, timedelta


DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'


def str_to_epoch(s):
    """
    Convert a date-time string to epoch time.
    The string must be formatted as
    '%Y-%m-%dT%H:%M:%S.%f'.

    Parameters
    ----------
    s : str
        A string containing the date and time.
    
    Returns
    -------
    A float of the epoch time corresponding
    to the provided string.
    """
    
    dt = datetime.strptime(s, DATETIME_FORMAT+'.%f').replace(tzinfo=timezone.utc)
    
    return datetime_to_epoch(dt)


def epoch_to_str(t):
    """
    Convert an epoch time to a string format.
    The output format is formatted as
    '%Y-%m-%dT%H:%M:%S.%f'.

    Parameters
    ----------
    t : float
        A timestamp in epoch time.
    
    Returns
    -------
    A string corresponding to the input time.
    """
    return epoch_to_datetime(t).strftime(DATETIME_FORMAT)


def datetime_to_epoch(dt):
    """
    Convert a datetime object to an epoch timestamp.

    Parameters
    ----------
    dt : datetime object
        The input datetime object. This must be
        timezone aware.
    
    Returns
    -------
    A float of the epoch time corresponding
    to the provided string.
    """
    
    epoch =  datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds()


def epoch_to_datetime(t):
    """
    Convert an epoch time to a datetime object.

    Parameters
    ----------
    t : float
        A timestamp in epoch time.
    
    Returns
    -------
    A datetime object corresponding to the input time.
    It is timezone aware in UTC.
    """

    return datetime.fromtimestamp(t, tz=timezone.utc)