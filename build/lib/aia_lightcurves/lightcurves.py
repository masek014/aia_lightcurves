import astropy.time
import astropy.units as u
import numpy as np
import sunpy

from . import file_io, plotting, boxcar
from .data_classes import Lightcurve, RegionCanister


def sort_by_time(dat: Lightcurve) -> Lightcurve:
    indices = np.argsort(dat.t)
    apply_sort = lambda a: np.array(a)[indices]
    return Lightcurve(
        t=astropy.time.Time(apply_sort(dat.t)),
        y=apply_sort(dat.y),
        exposure_times=apply_sort([et.value for et in dat.exposure_times]) << u.s
    )


def download_and_make_lightcurve(
    start_time: str | astropy.time.Time,
    end_time: str | astropy.time.Time,
    wavelength: int | u.Quantity,
    region_can: RegionCanister
) -> Lightcurve:
    """
    Download proper FITS files given time range and wavelength into directory structure.
    Then, construct lightcurve for the specified region.
    Currently, the intensities are **not** normalized in any way;
    the lightcurve scale is arbitrary.
    Downloads are retried `file_io.MAX_DOWNLOAD_ATTEMPTS` times.

    Parameters
    ----------
    start_time : str | astropy.time.Time
        The start time of the observation.
    end_time : str | astropy.time.Time
        The end time of the observation.
    wavelength : int | u.Angstrom
        The wavelength of interest.

    Returns
    -------
    Lightcurve NamedTuple
        t : list[astropy.time.Time]
            The times for each data point.
        y : list[float]
            The lightcurve values at each time.
        exposure_times : list[u.Quantity]
            Exposure time of each point.
    """

    print('Generating light curve data.')

    files = file_io.download_fits_parallel(
        start_time=astropy.time.Time(start_time),
        end_time=astropy.time.Time(end_time),
        wavelengths=[wavelength << u.Angstrom],
        num_simultaneous_connections=8,
        num_retries_for_failed=file_io.MAX_DOWNLOAD_ATTEMPTS,
        print_debug_messages=False,
    )

    if any(not f.success for f in files):
        maxx = file_io.MAX_DOWNLOAD_ATTEMPTS
        raise RuntimeError(
            f'Failed to download some files after {maxx} tries. Quit.'
        )

    return make_lightcurve([f.file for f in files], region_can)


def make_lightcurve(
    files: list[file_io.air.DownloadResult],
    region_can: RegionCanister,
    time_format: str='unix'
) -> Lightcurve:
    """
    Construct lightcurve for the specified region.

    Parameters
    ----------
    files : list[str]
        The FITS files to load into `sunpy.map.Map`
    region_kwargs :
        blah
    time_format : str
        Format for the times, compliant with
        astropy.time.Time objects.

    Returns
    -------
    Lightcurve NamedTuple
        t : list[astropy.time.Time]
            The times for each data point.
        y : list[float]
            The lightcurve values at each time.
        exposure_times : list[u.Quantity]
            Exposure time of each point.
    """
    times = []
    intensities = []
    exposure_times = []
    
    for f in files:
        map_obj = sunpy.map.Map(f)
        region = region_can.construct_given_map(map_=map_obj)
        submap_obj = plotting.make_submap(map_obj, region)
        reg_data = plotting.get_region_data(submap_obj, region)

        # map_time = astropy.time.Time(map_obj.date, format=time_format)
        intensity = np.sum(reg_data) # TODO: Normalize or convert?
        times.append(map_obj.date)
        intensities.append(intensity)
        exposure_times.append(map_obj.exposure_time)

    times = astropy.time.Time(astropy.time.Time(times), format=time_format)
    lc = Lightcurve(t=times, y=intensities, exposure_times=exposure_times)
    return sort_by_time(lc)


def make_averaged_lightcurve(
    lightcurve: Lightcurve,
    N: int
) -> Lightcurve:

    averaged = Lightcurve(
        lightcurve.t,
        boxcar.boxcar_average(lightcurve.y, N),
        lightcurve.exposure_times
    )

    return averaged


def make_detrended_lightcurve(
    lightcurve: Lightcurve,
    averaged: Lightcurve
) -> Lightcurve:

    detrended = Lightcurve(
        lightcurve.t,
        lightcurve.y - averaged.y,
        lightcurve.exposure_times
    )

    return detrended