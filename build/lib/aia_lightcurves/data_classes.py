import astropy.time
import astropy.units as u
import regions
import sunpy.map
import typing

from astropy.coordinates import SkyCoord
from dataclasses import dataclass


class Lightcurve(typing.NamedTuple):
    t: list[astropy.time.Time]
    y: list[float]
    exposure_times: list[u.Quantity]


class RegionCanister:
    @u.quantity_input
    def __init__(
        self,
        kind: regions.SkyRegion | None=None,
        center: tuple[u.arcsec, u.arcsec] | None=None,
        constructor_kwargs: dict[str, u.arcsec] | None=None
    ):
        self.kind = kind
        self.center = center
        self.constructor_kwargs = constructor_kwargs

    def construct_given_map(self, map_: sunpy.map.Map):
        return self.kind(
            SkyCoord(*self.center, frame=map_.coordinate_frame),
            **self.constructor_kwargs
        )

    def __bool__(self):
        return (
            (self.kind is not None) and
            (self.center is not None) and
            (self.constructor_kwargs is not None)
        )

    @u.quantity_input
    def half_width(self) -> u.arcsec:
        kw = self.constructor_kwargs
        if 'radius' in kw:
            return kw['radius']
        if 'width' in kw:
            return max(kw['width'], kw['height']) / 2
        if 'outer_width' in kw:
            return max(kw['outer_width'], kw['outer_height']) / 2

        raise ValueError("Specified SkyRegion does not have a well-defined half-width")
