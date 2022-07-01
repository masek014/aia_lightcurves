#!/bin/python3
"""
This is the main script showing how to processing observations.
Also see the example.ipynb notebook.
"""

import plotting
from test import test_observation


def main():

    obs_20190412 = {
        'start_time': '2019-04-12T17:10:00.0',
        'end_time': '2019-04-12T18:50:00.0',
        'map_time': '2019-04-12T18:00:33.0',
        'wavelengths': [171, 211],
        'center': (-220, 330),
        'radius': 50,
        'N': 51,
        'name': 'reg1'
    }

    obs_20190412_2 = {
        'start_time': '2019-04-12T17:15:00.0',
        'end_time': '2019-04-12T18:45:00.0',
        'map_time': '2019-04-12T18:00:00.0',
        'wavelengths': [94, 131, 171, 193, 211, 304, 335, 1600, 1700],
        'center': (-220, 330),
        'radius': 50,
        'N': 51,
        'name': 'reg1'
    }

    obs_20210429 = {
        'start_time': '2021-04-29T20:42:00.0',
        'end_time': '2021-04-30T06:00:00.0',
        'map_time': '2021-04-29T23:30:00.0',
        'wavelengths': [171],
        'center': (0, 0),
        'radius': 125,
        'N': 51,
        'name': 'reg1'
    }

    obs_20210429_2 = {
        'start_time': '2021-04-29T17:00:00.0',
        'end_time': '2021-04-29T20:00:00.0',
        'map_time': '2021-04-29T18:30:00.0',
        'wavelengths': [171, 211],
        'center': (800, -320),
        'radius': 100,
        'N': 51,
        'name': 'reg1'
    }

    obs_20210429_3 = {
        'start_time': '2021-04-29T17:00:00.0',
        'end_time': '2021-04-29T20:00:00.0',
        'map_time': '2021-04-29T18:30:00.0',
        'wavelengths': [171, 211],
        'center': (-430, 450),
        'radius': 80,
        'N': 51,
        'name': 'reg2'
    }

    obs_20211008 = {
        'start_time': '2021-10-08T05:00:10.0', # for some reason, 5:00:00-5:00:09 is invalid (pool is closed)
        'end_time': '2021-10-08T07:00:00.0',
        'map_time': '2021-10-08T06:00:00.0',
        'wavelengths': [171, 211],
        'center': (-400, 200),
        'radius': 125,
        'N': 51,
        'name': 'reg1'
    }

    obs = [obs_20190412_2]
    dicts = []
    for o in obs:
        dicts.append(plotting.process_observation(o))
    # d = test_observation()


if __name__ == '__main__':
    main()
