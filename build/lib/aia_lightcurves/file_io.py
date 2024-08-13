import astropy.time
import astropy.units as u
import numpy as np
import os
import warnings

from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from pathlib import Path
from sunpy.net import Fido, attrs as a

from . import calibrate
from .net import aia_requests as air


MAX_DOWNLOAD_ATTEMPTS = 10
data_dir = os.getcwd() + '/data/'
fits_dir_format = data_dir + '{date}/fits/'
lightcurves_dir_format = data_dir + '{date}/lightcurves/'
images_dir_format = data_dir + '{date}/images/'


FILTER_CADENCES = {
    94 * u.Angstrom: 12 * u.second,
    131 * u.Angstrom: 12 * u.second,
    171 * u.Angstrom: 12 * u.second,
    193 * u.Angstrom: 12 * u.second,
    211 * u.Angstrom: 12 * u.second,
    304 * u.Angstrom: 12 * u.second,
    335 * u.Angstrom: 12 * u.second,
    1600 * u.Angstrom: 24 * u.second,
    1700 * u.Angstrom: 24 * u.second,
    4500 * u.Angstrom: 3600 * u.second,
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
        fits_dir_format.format(date=date),
        lightcurves_dir_format.format(date=date),
        images_dir_format.format(date=date)
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _gather_local_files_helper(
    path: str,
    time_range: tuple[astropy.time.Time, astropy.time.Time],
    wavelength: u.Quantity,
    level: float
) -> astropy.time.Time | None:
    # Catch the astropy fits warnings so we know which
    # file caused it since astropy doesn't tell us...
    # NB: not thread safe
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        with fits.open(path, output_verify='warn') as hdu:
            hdr = hdu[1].header
            obs_time = astropy.time.Time(
                hdr['DATE-OBS'], scale='utc', format='isot')
            same_time = (obs_time >= time_range[0]) and (
                obs_time <= time_range[1])
            same_wavelength = (
                wavelength == hdr['WAVELNTH'] * u.Unit(hdr['WAVEUNIT']))
            same_level = (
                level == float(hdr['LVL_NUM']))
            if (same_time and same_wavelength) and same_level:
                return obs_time


def gather_local_files(
    fits_dir: str,
    time_range: tuple[astropy.time.Time, astropy.time.Time],
    wavelength: u.Quantity,
    level: float
) -> list[str]:
    """
    Checks fits_dir for AIA fits files that fall within the specified time_range.
    Returns a list of file paths sorted by time.
    """
    times = [] # Used for sorting the file names
    paths = []
    dir_files = [f for f in os.listdir(
        fits_dir) if os.path.isfile(os.path.join(fits_dir, f))]
    fits_dir = os.path.abspath(fits_dir)
    fits_paths = [os.path.join(fits_dir, f) for f in dir_files]

    for p in fits_paths:
        try:
            t = _gather_local_files_helper(
                path=p,
                time_range=time_range,
                wavelength=wavelength,
                level=level
            )
            if t is not None:
                times.append(t)
                paths.append(p)
        except OSError as e:  # Catch empty or corrupted fits files and non-fits files
            print(f'OSError with file {p}: {e}')
        except AstropyUserWarning as e:
            print(f'AstropyUserWarning with file {p}: {e}')

    paths = [f for _, f in sorted(zip(times, paths))]
    return paths


def validate_local_files(
    fits_dir: str,
    time_range: tuple[astropy.time.Time],
    wavelength: u.angstrom,
    level: float
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
    local_files = gather_local_files(fits_dir, time_range, wavelength, level)

    # Correct for possible +1 file depending on
    # alignment of time_range with the image cadence.
    if local_files:
        with fits.open(local_files[0]) as hdu:
            f_start = astropy.time.Time(
                hdu[1].header['date-obs'], scale='utc', format='isot')
        front_diff = (f_start - time_range[0]).to(u.second)
        front_diff = (front_diff.value % cadence.value) * u.second
        if front_diff < (image_units % 1) * cadence:
            minimum_expected += 1

    b_satisfied = (minimum_expected == len(local_files))

    return local_files, b_satisfied


def identify_missing_l1p5(l1_files: list[str], l1p5_files: list[str]) -> list[str]:
    """
    Finds the level 1 files without a corresponding level 1.5 file.
    """

    def time_from_fits(file: str) -> astropy.time.Time:

        with fits.open(file) as hdu:
            hdr = hdu[1].header
            time = astropy.time.Time(
                hdr['DATE-OBS'], scale='utc', format='isot'
            )

        return time
    
    pairs = {}
    for file in l1_files:
        time = time_from_fits(file)
        pairs[time] = {'L1': file, 'L1.5': None}
    
    for file in l1p5_files:
        time = time_from_fits(file)
        pairs[time]['L1.5'] = file
    
    l1_with_missing_l1p5 = []
    for time, pair in pairs.items():
        if pair['L1.5'] is None:
            l1_with_missing_l1p5.append(pair['L1'])

    return l1_with_missing_l1p5


def obtain_files(
    time_range: tuple[astropy.time.Time, astropy.time.Time],
    wavelengths: list[u.Quantity],
    num_simultaneous_connections: int = 1,
    num_retries_for_failed: int = 10,
    level: float = 1.5
) -> list:
    """
    General purpose function for obtaining the desired files.
    If all files are available locally, the list of those
    files is returned. Otherwise, it will download the missing
    files and return the **full** list of files (local+downloaded).
    """
    
    date = time_range[0].strftime(air.DATE_FMT)
    make_directories(date=date)
    fits_dir = fits_dir_format.format(date=date)
    all_files = []
    for wavelength in wavelengths:

        l1p5_files, l1p5_satisfied = validate_local_files(
            fits_dir, time_range, wavelength, 1.5
        )

        l1_files, l1_satisfied = validate_local_files(
            fits_dir, time_range, wavelength, 1
        )
        new_l1_files = []

        if l1p5_satisfied and level == 1.5:
            all_files += l1p5_files
            continue
        elif l1_satisfied:
            lone_l1_companions = identify_missing_l1p5(l1_files, l1p5_files)
        else:
            results = download_fits_parallel(
                *time_range,
                [wavelength],
                num_simultaneous_connections,
                num_retries_for_failed
            )
            new_l1_files = [r.file for r in results]
            lone_l1_companions = [r.file for r in results]
        
        if lone_l1_companions:
            print('level 1 files with missing level 1.5 companion:')
            for file in lone_l1_companions:
                print('\t', file)

        new_l1p5_files = []
        if level == 1.5:

            for l1_file in lone_l1_companions:
                if 'lev1' in l1_file:
                    l1p5_file = l1_file.replace('lev1', 'lev1.5')
                else:
                    l1p5_file = f'{Path(l1_file).stem}_lev1.5.{Path(l1_file).suffix}'
                calibrate.level1_to_1p5(l1_file, l1p5_file, True, True)
                new_l1p5_files.append(l1p5_file)

            if l1p5_files:
                print('found l1.5 files:')
                for file in l1p5_files:
                    print('\t', file)
            if new_l1p5_files:
                print('new l1.5 files:')
                for file in new_l1p5_files:
                    print('\t', file)
        
        if level == 1:
            all_files += l1_files
            all_files += new_l1_files
        elif level == 1.5:
            all_files += l1p5_files
            all_files += new_l1p5_files

    return all_files


@u.quantity_input
def download_fits_parallel(
    start_time: astropy.time.Time,
    end_time: astropy.time.Time,
    wavelengths: list[u.Angstrom],
    num_simultaneous_connections: int = 1,
    num_retries_for_failed: int = 10
) -> list[air.DownloadResult]:
    """
    download fits in parallel using raw HTTP requests + XML
    **Only returns the files that were downloaded, i.e. does
    *not* return any files that may also be present locally.**
    """

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


def download_fits_old(
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
            a.Instrument('aia'), a.Wavelength(wavelength), 
            a.Sample(12 * u.second)
        )
        if b_save_files:
            for res in result[0]:
                date = res[0].value.split(' ')[0].replace('-', '')
                path = fits_dir_format.format(date=date)
                files += Fido.fetch(res, site='ROB', path=path,
                                    progress=b_show_progress)
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
