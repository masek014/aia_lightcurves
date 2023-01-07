import astropy.units as u
from aia_lightcurves.net import aia_requests as air

# guard with this `if` statement for multiprocessing
if __name__ == '__main__':
    air.cfg.read_timeout = 10 << u.s
    air.cfg.debug = True
    air.timed_test()
