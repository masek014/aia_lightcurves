#!/bin/python3

from plotting import process_observation


def test_observation():
    """
    A test observation. Use this to make sure
    everything is working properly

    Returns
    -------
    d : dict
        Dictionary containing the processed information.
    """

    test_obs = {
        'start_time': '2018-09-09T12:00:00.0',
        'end_time': '2018-09-09T12:01:00.0',
        'map_time': '2018-09-09T12:01:00.0',
        'wavelengths': [171],
        'center': (350, 75),
        'radius': 200,
        'N': 15,
        'name': 'test'
    }

    d = process_observation(test_obs)

    return d


if __name__ == '__main__':
    test_observation()