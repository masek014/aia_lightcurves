import typing
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
import astropy.time
import astropy.units as u
import regions
import sunpy.map


class Lightcurve(typing.NamedTuple):
    t: list[astropy.time.Time]
    y: list[float]
    exposure_times: list[u.Quantity]


class RegionCanister(typing.NamedTuple):
    kind: regions.SkyRegion | None=None
    center: tuple[u.Quantity, u.Quantity] | None=None
    constructor_kwargs: dict[str, object] | None=None

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