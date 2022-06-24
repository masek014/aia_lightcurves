import os
import sys
import multiprocessing
import numpy as np
import astropy.units as u

from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a


FITS_DIR = os.getcwd() + '/fits/'
LIGHTCURVES_DIR = os.getcwd() + '/lightcurves/'
IMAGES_DIR = os.getcwd() + '/images/'
TIME_STR_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
sys.path.append(os.getcwd())


def make_directories():
    """
    Create the necessary directories for storing the FITS files,
    the lightcurve CSVs, and the images. The created directories
    are: ./fits, ./lightcurves, and ./images.
    """

    dirs = [FITS_DIR, LIGHTCURVES_DIR, IMAGES_DIR]

    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


def get_fits_files_parallel(start_time, end_time, wavelengths, b_save_files=True, b_show_progress=False):

    # Divide up the start and end time into 10 second segments.
    start_dt = datetime.strptime(start_time, TIME_STR_FORMAT)
    end_dt = datetime.strptime(end_time, TIME_STR_FORMAT)
    cur_time = start_dt
    times_list = [] # List of time strings for Fido.
    while cur_time < end_dt:
        cur_timestr = datetime.strftime(cur_time, TIME_STR_FORMAT)
        times_list.append(cur_timestr)
        cur_time = cur_time + timedelta(seconds=10)
    times_list.append(end_time)
    
    files = []
    pool = multiprocessing.Pool(processes=4)
    jobs = []
    for i in range(len(times_list)-1):
        # Parallelize here.
        # f = lambda : files.append(get_fits_files(times_list[i], times_list[i+1], wavelengths, b_save_files, b_show_progress))
        p = multiprocessing.Process(target=get_fits_files,
            args=(times_list[i], times_list[i+1], wavelengths, b_save_files, b_show_progress))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
    pool.close()
    pool.join()

    import glob
    files_dir = FITS_DIR + start_time.split('T')[0].replace('-', '') + '/'
    files = glob.glob(files_dir + '*')
    files.sort()

    return files


def get_fits_files(start_time, end_time, wavelengths, b_save_files=True, b_show_progress=False):
    """
    Use Fido to get the FITS files for the specified time interval and wavelength.
    This method may take considerable time to run because it uses Fido.

    Parameters
    ----------
    start_time : str
        The start time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    end_time : str
        The end time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    wavelength : list on int
        The wavelengths of interest.
    b_save_file : bool
        Specify whether to save the downloaded image data.
    b_show_progress : bool
        Specify whether to show the download progress.
    
    Returns
    -------
    files : list of str
        A list containing all of the downloaded file paths.
    """

    if not isinstance(wavelengths, list):
        wavelengths = [wavelengths]

    files = []
    for wavelength in wavelengths:
        result = Fido.search(a.Time(start_time, end_time), 
                            a.Instrument('aia'), a.Wavelength(wavelength*u.angstrom), 
                            a.Sample(12 * u.second))
        if b_save_files:
            for res in result[0]:
                date = res[0].value.split(' ')[0].replace('-', '')
                path = FITS_DIR + date + '/'
                files += Fido.fetch(res, site='ROB', path=path, progress=b_show_progress)
        else:
            files += Fido.fetch(result, site='ROB', progress=b_show_progress)

    # Retry downloading any failed files.
    files = Fido.fetch(files) # TODO: Determine if we want this or not

    return files


def read_lightcurves(save_path):
    """
    Read lightcurve data from the specified CSV file.

    Parameters
    ----------
    save_path : str
        Path to the input CSV file.
    
    Returns
    -------
    lightcurve : tuple
        Contains the data from the input file.
        In the format (times, data) where times
        and data are both np.ndarray.
    """

    print('Reading lightcurve data from \'' + save_path + '\'')

    dat = np.genfromtxt(save_path, delimiter=',',
        # dtype=[('time', np.dtype('U100')), ('lc', np.float64), ('lc_bc', np.float64), ('lc_de', np.float64)])
        dtype=[('time', np.float64), ('lc', np.float64)])

    lightcurve = (dat['time'], dat['lc'])
    # lightcurve_bc = (dat['time'], dat['lc_bc'])
    # lightcurve_detrended = (dat['time'], dat['lc_de'])
    
    return lightcurve #, lightcurve_bc, lightcurve_detrended


def save_lightcurves(arrays, save_path):
    """
    Save lightcurve data to the specified CSV file.

    Parameters
    ----------
    arrays : tuple
        Contains the data to save to the file.
        In the format (times, data) where times
        and data are both np.ndarray.
    save_path : str
        Path to the input CSV file.
    """

    print('Saving lightcurve data to \'' + save_path + '\'')
    a = np.vstack(arrays).T
    try:
        np.savetxt(save_path, a, delimiter=',', fmt='%s')#, '%.18e', '%.18e', '%.18e'])
    except Exception as e:
        print('Exception while saving lightcurves to file.')
        print(e)
        os.remove(save_path)