#!/bin/python3
"""
This is the main script showing how to processing observations.
Also see the example.ipynb notebook.
"""

import plotting


def main():

    obs_20190412 = {
        'start_time': '2019-04-12T17:10:00',
        'end_time': '2019-04-12T18:50:00',
        'map_time': '2019-04-12T18:00:33',
        'wavelengths': [171, 211],
        'center': (-220, 330),
        'radius': 50,
        'N': 51,
        'name': 'reg1'
    }

    obs_20211008 = {
        'start_time': '2021-10-08T05:00:10', # for some reason, 5:00:00-5:00:09 is invalid (pool is closed)
        'end_time': '2021-10-08T07:00:00',
        'map_time': '2021-10-08T06:00:00',
        'wavelengths': [171, 211],
        'center': (-400, 200),
        'radius': 125,
        'N': 51,
        'name': 'reg1'
    }

    obs = [obs_20190412, obs_20211008]
    dicts = []
    for o in obs:
        dicts.append(plotting.process_observation(o))


if __name__ == '__main__':
    main()