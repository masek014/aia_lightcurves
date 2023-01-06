import os
import sys
import glob
import parse
import requests
import http.client
import numpy as np
import multiprocessing
import astropy.units as u

from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a


MAX_DOWNLOAD_ATTEMPTS = 3
DATA_DIR = os.getcwd() + '/data/'
FITS_DIR_FORMAT = DATA_DIR + '{date}/fits/'
LIGHTCURVES_DIR_FORMAT = DATA_DIR + '{date}/lightcurves/'
IMAGES_DIR_FORMAT = DATA_DIR + '{date}/images/'
TIME_STR_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
FITS_FILE_NAME_FORMAT = 'aia_{date}_{time}_{wavelength:04d}_image_lev1.fits'
sys.path.append(os.getcwd())


def make_directories(date):
    """
    Create the necessary directories for storing the FITS files,
    the lightcurve CSVs, and the images. The created directories
    are: ./fits, ./lightcurves, and ./images.
    """

    dirs = [
        DATA_DIR,
        DATA_DIR + date + '/',
        FITS_DIR_FORMAT.format(date=date),
        LIGHTCURVES_DIR_FORMAT.format(date=date),
        IMAGES_DIR_FORMAT.format(date=date)
    ]

    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


def download_fits_parallel(start_time, end_time, wavelength, b_save_files=True, b_show_progress=False):
    """
    An attempt at parallelizing Fido downloads.
    WIP.
    This hasn't been too successful so far.
    """

    result = Fido.search(a.Time(start_time, end_time), 
        a.Instrument('aia'), a.Wavelength(wavelength*u.angstrom), 
        a.Sample(12 * u.second))

    if b_save_files:
        path = DATA_DIR + start_time.split('T')[0].replace('-', '') + '/'
    else:
        path = None
    
    jobs = []
    files = [] # TODO: Need to put results from fetch into here
    if b_save_files:
        for res in result[0]:
            date = res[0].value.split(' ')[0].replace('-', '')
            path = FITS_DIR_FORMAT.format(date=date)
            p = multiprocessing.Process(target=Fido.fetch,
                args=(res,), 
                kwargs={'max_conn':1, 'site':'ROB', 'path':path, 'progress':b_show_progress})
            jobs.append(p)
            p.start()
    else:
        p = multiprocessing.Process(target=Fido.fetch,
                args=(result,), kwargs={'max_conn':1, 'site':'ROB', 'progress':b_show_progress})
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

    # TODO: Determine if we want this or not
    # Retry downloading any failed files.
    # https://docs.sunpy.org/en/stable/guide/acquiring_data/fido.html#retrying-downloads
    # print('Error fits:', files.errors)
    # files = Fido.fetch(files)

    if path is not None:
        files = glob.glob(f'{path}*_{wavelength:04d}_*')
        files.sort()

    return files


def download_fits(start_time, end_time, wavelengths, b_save_files=True, b_show_progress=False):
    """
    
    Use Fido to get the FITS files for the specified time interval and wavelength.
    This method may take considerable time to run because it serially uses Fido.

    This is the old version, but it works perfectly fine. It does not feature the exception handling that the new
    version has, which could be useful if internet connection is spotty.

    Parameters
    ----------
    start_time : str
        The start time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    end_time : str
        The end time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    wavelength : list or int
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
                path = FITS_DIR_FORMAT.format(date=date)
                files += Fido.fetch(res, site='ROB', path=path, progress=b_show_progress)
        else:
            files += Fido.fetch(result, site='ROB', progress=b_show_progress)

    return files


def download_fits2(start_time, end_time, wavelength, b_save_files=True, b_show_progress=False):
    """
    The files are downloaded through Fido one at a time.
    This allows the handling of connection errors/timeouts so that
    the download can be reattempted rather than crashing the
    entire process.

    Use Fido to get the FITS files for the specified time interval and wavelength.
    This method may take considerable time to run because it serially uses Fido.

    Parameters
    ----------
    start_time : str
        The start time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    end_time : str
        The end time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    wavelength : list or int
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

    # Divide up the start and end time into 12 second segments.
    start_dt = datetime.strptime(start_time, TIME_STR_FORMAT)
    end_dt = datetime.strptime(end_time, TIME_STR_FORMAT)
    times_list = [] # List of time strings for Fido
    cur_time = start_dt

    while cur_time < end_dt:
        cur_timestr = datetime.strftime(cur_time, TIME_STR_FORMAT)
        times_list.append(cur_timestr)
        cur_time = cur_time + timedelta(seconds=12)
    times_list.append(end_time)

    files = []
    for i in range(len(times_list)-1):
        attempt_number = 1
        while attempt_number <= MAX_DOWNLOAD_ATTEMPTS:
            print(f'Attempting data download for time {times_list[i]} - {times_list[i+1]}.')
            try:
                files += download_fits(times_list[i], times_list[i+1], wavelength, b_save_files, b_show_progress)
                print('Download successful.')
                break
            except (http.client.RemoteDisconnected, requests.exceptions.ConnectionError) as e:
                print(f'Exception on attempt number {attempt_number}/{MAX_DOWNLOAD_ATTEMPTS}.\n{e}')
                attempt_number += 1
                continue
        if attempt_number > MAX_DOWNLOAD_ATTEMPTS:
            print('Exceeded maximum number of attempts. Exitting.')
            sys.exit(1)

    return files


def redownload_file(file_path):
    """
    Redownload the given file. This is intended to be used to
    redownload corrupted FITS files.

    Parameters
    ----------
    file_path : str
        Full path to the file to redownload.
    
    Returns
    -------
    f : str
        Full path to the redownloaded file.
    """
    
    # Obtain the file details.
    file_name = file_path.split('/')[-1]
    parsed = parse.parse(FITS_FILE_NAME_FORMAT, file_name)

    # Format the datetime string
    date = parsed['date'][:4] + '-' + parsed['date'][4:]
    date = date[:7] + '-' + date[7:]
    time = parsed['time'][:2] + ':' + parsed['time'][2:]
    time = time[:5] + ':' + time[5:] + '.0'
    
    # Get the start and end time.
    dt = datetime.strptime(date+'T'+time, TIME_STR_FORMAT)
    dt_start = dt - timedelta(seconds=5)
    dt_end = dt + timedelta(seconds=5)
    start_time = datetime.strftime(dt_start, TIME_STR_FORMAT)
    end_time = datetime.strftime(dt_end, TIME_STR_FORMAT)

    print('Deleting file:', file_path)
    os.remove(file_path) # Delete the existing file

    print('Redownloading file.')
    f = download_fits(start_time, end_time, parsed['wavelength'])

    return f


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

    print(f'Reading lightcurve data from \'{save_path}\'')

    dat = np.genfromtxt(save_path, delimiter=',',
        dtype=[('time', np.float64), ('lc', np.float64)])

    lightcurve = (dat['time'], dat['lc'])
    
    return lightcurve


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

    print(f'Saving lightcurve data to \'{save_path}\'')
    a = np.vstack(arrays).T
    np.savetxt(save_path, a, delimiter=',', fmt='%s')