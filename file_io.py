import os
import sys

import numpy as np
import astropy.time
import astropy.units as u
from sunpy.net import Fido, attrs as a

from .net import aia_requests as air


MAX_DOWNLOAD_ATTEMPTS = 3
DATA_DIR = os.getcwd() + '/data/'
FITS_DIR_FORMAT = DATA_DIR + '{date}/fits/'
LIGHTCURVES_DIR_FORMAT = DATA_DIR + '{date}/lightcurves/'
IMAGES_DIR_FORMAT = DATA_DIR + '{date}/images/'
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


@u.quantity_input
def download_fits_parallel(
    start_time: astropy.time.Time,
    end_time: astropy.time.Time,
    wavelengths: list[u.Angstrom],
    num_simultaneous_connections: int=5,
    num_retries_for_failed: int=10
) -> list[air.DownloadResult]:
    ''' download fits in parallel using raw HTTP requests + XML '''

    date = start_time.strftime(air.DATE_FMT)
    make_directories(date=date)
    out = FITS_DIR_FORMAT.format(date=date)
    files = air.download_aia_between(
        start=start_time,
        end=end_time,
        wavelengths=wavelengths,
        fits_out_dir=out,
        num_jobs=num_simultaneous_connections,
        attempts=num_retries_for_failed
    )

    return files


def download_fits(
    start_time: str,
    end_time: str,
    wavelengths: int | list[int],
    b_save_files: bool = True,
    b_show_progress: bool = False
) -> list[str]:
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
        result = Fido.search(
            a.Time(start_time, end_time), 
            a.Instrument('aia'), a.Wavelength(wavelength*u.angstrom), 
            a.Sample(12 * u.second)
        )
        if b_save_files:
            for res in result[0]:
                date = res[0].value.split(' ')[0].replace('-', '')
                path = FITS_DIR_FORMAT.format(date=date)
                files += Fido.fetch(res, site='ROB', path=path, progress=b_show_progress)
        else:
            files += Fido.fetch(result, site='ROB', progress=b_show_progress)

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

    print(f'Reading lightcurve data from \'{save_path}\'')

    data = np.genfromtxt(
        save_path,
        delimiter=',',
        dtype=[('time', np.float64), ('lc', np.float64)]
    )

    lightcurve = (data['time'], data['lc'])
    
    return lightcurve


LcArrays = tuple[np.ndarray | u.Quantity, np.ndarray | u.Quantity]
def save_lightcurves(arrays: LcArrays, save_path: str):
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
    cleaned = []
    for a in arrays:
        if isinstance(a[0], u.Quantity):
            cleaned.append([v.value for v in a])
        else:
            cleaned.append(a)
    a = np.array(cleaned).T
    np.savetxt(save_path, a, delimiter=',', fmt='%s')
