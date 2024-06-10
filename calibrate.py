import aiapy.calibrate
import sunpy.map

from astropy.io import fits


def level1_to_1p5(
    level1_file: str,
    level1p5_file: str,
    overwrite_existing: bool = False,
    remove_scaling_keywords: bool = True
):
    """
    Uses aiapy to perform the following corrections on a level 1 AIA file to
    prepare it into a level 1.5 file:
    1. Pointing correction
    2. Registration (account for satellite roll angle and pixel scale)

    Source: https://aiapy.readthedocs.io/en/stable/preparing_data.html
    """
    
    map_ = sunpy.map.Map(level1_file)
    map_pointing_corrected = aiapy.calibrate.update_pointing(map_)
    map_registered = aiapy.calibrate.register(map_pointing_corrected)
    
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