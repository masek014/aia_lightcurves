import aiapy.calibrate
import sunpy.map
import urllib.error

from astropy.io import fits


def level1_to_1p5(
    level1_file: str,
    level1p5_file: str,
    overwrite_existing: bool = False,
    remove_scaling_keywords: bool = True,
    max_attempts: int = 5
):
    """
    Uses aiapy to perform the following corrections on a level 1 AIA file to
    prepare it into a level 1.5 file:
    1. Pointing correction
    2. Registration (account for satellite roll angle and pixel scale)

    Source: https://aiapy.readthedocs.io/en/stable/preparing_data.html
    """
    
    map_ = sunpy.map.Map(level1_file)

    attempt = 0
    successful = False
    while not successful and attempt < max_attempts:
        try:
            map_pointing_corrected = aiapy.calibrate.update_pointing(map_)
            map_registered = aiapy.calibrate.register(map_pointing_corrected)
            successful = True
        except urllib.error.URLError as e:
            print(f'[attempt {attempt} / 5] Timeout while performing correction to level 1.5\n{e}')
            attempt += 1
    if not successful:
        raise RuntimeError(f'Could not correct level 1 file to level 1.5. Too many timeouts: {level1_file}')
    
    if remove_scaling_keywords:
        if 'BSCALE' in map_registered.meta:
            del(map_registered.meta['BSCALE'])
        if 'BZERO' in map_registered.meta:
            del(map_registered.meta['BZERO'])
    
    map_registered.save(
        level1p5_file,
        hdu_type=fits.CompImageHDU,
        overwrite=overwrite_existing
    )


# ... add the other (optional) calibration functions too?