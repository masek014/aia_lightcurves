#!/bin/python3
"""
This is the main script showing how to processing observations.
Also see the example.ipynb notebook.
"""

from aia_lightcurves.plotting import process_observation


def main():

    obs_20190412_1 = {
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

    obs = [obs_20190412_1, obs_20190412_2]
    dicts = []
    for o in obs:
        dicts.append(process_observation(o))


if __name__ == '__main__':
    main()
