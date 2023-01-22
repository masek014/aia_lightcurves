import datetime
import logging

import astropy.units as u
import astropy.time
from aia_lightcurves import file_io

def test():
    # Do not configure logging if you don't want debug messages to print.
    logfile = 'parallel_example.log'
    logging.basicConfig(filename=logfile, level=logging.INFO)

    print('logging example output to', logfile)
    print(f'you can view it with `tail -f {logfile}`')
    with open (logfile, 'w') as f:
        start = astropy.time.Time('2019-04-03T17:40:00.0')
        end = astropy.time.Time('2019-04-03T17:55:00.0')
        wav = 171 << u.Angstrom

        run_start = datetime.datetime.now()
        downloaded = file_io.download_fits_parallel(
            start_time=start,
            end_time=end,
            wavelengths=[wav]
        )
        run_end = datetime.datetime.now()
        logging.info(f'this took {run_end - run_start}')

        if all(d.success for d in downloaded):
            logging.info('all downloads successful')
        else:
            logging.info('failed:')
            for d in downloaded:
                if not d.success: print(d)

if __name__ == '__main__': test()