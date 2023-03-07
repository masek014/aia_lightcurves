import os
import warnings

import numpy as np
import astropy.time
import astropy.units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from sunpy.net import Fido, attrs as a

from .net import aia_requests as air


MAX_DOWNLOAD_ATTEMPTS = 3
data_dir = os.getcwd() + '/data/'
fits_dir_format = data_dir + '{date}/fits/'
lightcurves_dir_format = data_dir + '{date}/lightcurves/'
images_dir_format = data_dir + '{date}/images/'


FILTER_CADENCES = {
      94 * u.Angstrom : 12 * u.second,
     131 * u.Angstrom : 12 * u.second,
     171 * u.Angstrom : 12 * u.second,
     193 * u.Angstrom : 12 * u.second,
     211 * u.Angstrom : 12 * u.second,
     304 * u.Angstrom : 12 * u.second,
     335 * u.Angstrom : 12 * u.second,
    1600 * u.Angstrom : 24 * u.second,
    1700 * u.Angstrom : 24 * u.second,
    4500 * u.Angstrom : 3600 * u.second,
}


def _update_dirs():

    global fits_dir_format, lightcurves_dir_format, images_dir_format

    fits_dir_format = data_dir + '{date}/fits/'
    lightcurves_dir_format = data_dir + '{date}/lightcurves/'
    images_dir_format = data_dir + '{date}/images/'


def set_data_dir(new_dir: str):
    """
    Sets the output data directory containing the
    fits, images, and lightcurves files.
    TODO: Should we do this or have a config file?
    """

    global data_dir

    if new_dir[-1] != '/':
        new_dir += '/'

    data_dir = new_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    _update_dirs()


def make_directories(date: str):
    """
    Create the necessary directories for storing the FITS files,
    the lightcurve CSVs, and the images. The created directories
    are: ./fits, ./lightcurves, and ./images.
    """

    dirs = [
        data_dir,
        f'{data_dir}{date}/',
        fits_dir_format.format(date=date),
        lightcurves_dir_format.format(date=date),
        images_dir_format.format(date=date)
    ]

    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


def gather_local_files(
    fits_dir: str,
    time_range: tuple[astropy.time.Time],
    wavelength: u.Quantity,
) -> list[str]:
    """
    Checks in_dir for AIA fits files that fall within the specified time_range.
    Returns a list of file paths sorted by time.
    """

    # Catch the astropy fits warnings so we know which
    # file caused it since astropy doesn't tell us...
    warnings.filterwarnings('error')

    times = [] # Used for sorting the file names
    paths = []
    dir_files = [f for f in os.listdir(fits_dir) if os.path.isfile(os.path.join(fits_dir, f))]
    for f in dir_files:
        try:
            path = os.path.join(fits_dir, f)
            with fits.open(path, output_verify='warn') as hdu:
                hdr = hdu[1].header
                obs_time = astropy.time.Time(hdr['date-obs'], scale='utc', format='isot')
                same_time = (obs_time >= time_range[0]) and (obs_time <= time_range[1])
                same_wavelength = (wavelength == hdr['wavelnth'] * u.Unit(hdr['waveunit']))
                if same_time and same_wavelength:
                    times.append(obs_time)
                    paths.append(path)
        except OSError as e: # Catch empty or corrupted fits files and non-fits files
            print(f'OSError with file {f}: {e}')
        except AstropyUserWarning as e:
            print(f'AstropyUserWarning with file {f}: {e}')

    paths = [f for _, f in sorted(zip(times, paths))]

    warnings.resetwarnings()
    
    return paths


def validate_local_files(
    fits_dir: str,
    time_range: tuple[astropy.time.Time],
    wavelength: u.angstrom
) -> tuple[list[str], bool]:
    """
    Returns a list of all available local files and a boolean specifying
    whether all files are available (True) or if some are missing (False).

    Check if all files within the specified wavelengths are available locally
    at fits_dir. This method avoids connecting to the SDO servers since
    they're unreliable, and we don't want to prevent the use of local files
    based on whether a remote server is offline.

    The found number of files is compared against minimum_expected.
    It's a minimum since we cannot determine if we require a +1
    if there are no local files for a given wavelength.
    """

    duration = (time_range[1] - time_range[0]).to(u.second)
    cadence = FILTER_CADENCES[wavelength]
    image_units = duration.value / cadence.value
    minimum_expected = int(image_units)
    local_files = gather_local_files(fits_dir, time_range, wavelength)
    
    # Correct for possible +1 file depending on
    # alignment of time_range with the image cadence.
    if local_files:
        with fits.open(local_files[0]) as hdu:
            f_start = astropy.time.Time(hdu[1].header['date-obs'], scale='utc', format='isot')
        front_diff = (f_start - time_range[0]).to(u.second)
        front_diff = (front_diff.value % cadence.value) * u.second
        if front_diff < (image_units % 1) * cadence:
            minimum_expected += 1
    
    b_satisfied = minimum_expected==len(local_files)

    return local_files, b_satisfied


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
    out = fits_dir_format.format(date=date)
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
                path = fits_dir_format.format(date=date)
                files += Fido.fetch(res, site='ROB', path=path, progress=b_show_progress)
        else:
            files += Fido.fetch(result, site='ROB', progress=b_show_progress)

    return files


def read_lightcurves(save_path: str):
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