import astropy.units as u
import astropy.time
import matplotlib.pyplot as plt
import numpy as np
import regions
import typing

from . import plotting, lightcurves, file_io, boxcar
from .data_classes import RegionCanister


class Observation():


    def __init__(
        self,
        start_time: str | astropy.time.Time,
        end_time: str | astropy.time.Time,
        wavelengths: u.Quantity | typing.Iterable[u.Quantity],
        boxcar_width: int | None = None,
        out_dir: str = None
    ):
        """
        Parameters
        ----------
        start_time : str or astropy.time.Time
            The start time of the observation, parasable by astropy.time.Time.
        end_time : str or astropy.time.Time
            The end time of the observation, parasable by astropy.time.Time.
        wavelengths : u.Quantity or an interable of u.Quantity
            The AIA wavelength(s) of interest.
        boxcar_width : int or None
            The width of the boxcar average performed on the lightcurve.
            If no width is specified, no average lightcurve is computed.
        out_dir : str
            Specify the output data directory. If None, the default in
            file_io is used.
        """

        self.start = astropy.time.Time(start_time)
        self.end = astropy.time.Time(end_time)

        self.wavelengths = tuple(np.atleast_1d(wavelengths).flatten())
        self.boxcar_width = boxcar_width

        self.data = {}
        # just need region metadata until drawing or making the lightcurve
        self.region_can = None

        if out_dir is not None:
            file_io.set_data_dir(out_dir)


    def set_region(
        self,
        RegionClass: regions.SkyRegion,
        center: tuple[u.Quantity, u.Quantity],
        name: str = '',
        **kwargs
    ):
        """
        Set the region of interest. The RegionClass parameter can be any
        SkyRegion subclass from the regions package, and kwargs are any keyword
        arguments associated with the specific RegionClass (e.g. radius).

        Parameters
        ----------
        RegionClass : SkyRegion
            The desired region type.
        center : tuple of u.Quantity
            The (x,y) pair defining the center of the region.
        name : str
            A label for the region. This is used for distinguishing regions if
            multiple regions are used during the same observation.
        kwargs
            The remaining keyword arguments used to define the region.
        """

        # Use this so the coordinate_frame doesn't need to ever be specified until the region is actually created
        self.region_can = RegionCanister(
            kind=RegionClass,
            center=center,
            constructor_kwargs=kwargs
        )
        self.name = name


    def process(
        self,
        b_plot_overview: bool = True,
        b_plot_summary: bool = True
    ):
        """
        Generate the lightcurve and plots for the defined observation.

        Parameters
        ----------
        b_plot_overview : bool
            Specifies whether the overview plots containing the maps and
            lightcurves should be generated and saved.
        b_plot_summary : bool
            Specifies whether the summary plots containing the lightcurve, normalized exposure-normalized lightcurve, and
            exposure time plots should be generated and saved.
        b_save_lightcurves : bool
            Specifies whether the lightcurve data should be saved to files.
        """

        self._check_region()

        for wavelength in self.wavelengths:

            files = self.data[wavelength]['files']
            lightcurve = lightcurves.make_lightcurve(files, self.region_can)
            self.data[wavelength]['lightcurve'] = lightcurve

            if b_plot_overview:
                self._plot_overview(wavelength)
            
            if b_plot_summary:
                self._plot_summary(wavelength)

            csv_path = self._build_lightcurve_path(wavelength)
            file_io.save_lightcurves((lightcurve[0], lightcurve[1]), csv_path)


    def preprocess(self, image_ref_time: str | astropy.time.Time = None):
        """
        Prepares the data for creating the maps and lightcurves.
        The AIA FITS files are downloaded.
        """

        # TODO: Should we keep the adjust_n call?
        if self.boxcar_width is not None:
            seconds = (self.end - self.start).to(u.second).value
            num_frames = seconds // 12
            self.boxcar_width = boxcar.adjust_n(num_frames, self.boxcar_width)

        if image_ref_time is not None:
            diff = ( astropy.time.Time(image_ref_time) - self.start ).to(u.s)
            ref_index = int( diff.value / 12 )
        else:
            ref_index = 0

        for wavelength in self.wavelengths:
            files = file_io.obtain_files(
                time_range=(self.start, self.end),
                wavelengths=[wavelength],
                num_simultaneous_connections=1,
                num_retries_for_failed=file_io.MAX_DOWNLOAD_ATTEMPTS,
            )
            reference_map = plotting.sunpy.map.Map(files[ref_index])
            self.data[wavelength] = dict(
                files=files,
                map=reference_map,
                lightcurve=None
            )


    def _check_region(self):
        """
        Determines whether the region is present for continuing the analysis.
        An AttributeError is raised if no region has been assigned.
        """
        if not self.region_can:
            raise AttributeError(
                    'Cannot continue without a region. '
                    'Specify the region with the \'set_region\' method.'
                )


    def _build_path_root(self, dir_format, wavelength):
        """
        Generates the root of a file save path.
        """

        start_date = self.start.strftime(file_io.air.DATE_FMT)
        start_time = self.start.strftime(file_io.air.TIME_FMT)
        end_time = self.end.strftime(file_io.air.TIME_FMT)
        images_dir = dir_format.format(date=start_date)

        label = self.name
        if self.name != '':
            label = label + '_'

        return (
            f'{images_dir}{label}{start_time}-'
            f'{end_time}_{wavelength}_'
            f'N{self.boxcar_width}'
        )


    def _build_figure_path(self, wavelength, plot_type):
        """
        Generates the path for the saved plot.
        """

        root = self._build_path_root(file_io.images_dir_format, wavelength)
        fig_path = f'{root}_{plot_type}.png'

        return fig_path


    def _build_lightcurve_path(self, wavelength):
        """
        Generates the path for the saved lightcurve data.
        """

        root = self._build_path_root(file_io.lightcurves_dir_format, wavelength)
        lc_path = f'{root}_lc.csv'

        return lc_path


    def _plot_overview(self, wavelength):
        """
        Generates the overview plots for the given wavelength.
        """

        fn = self.data[wavelength]['files'][0]
        m = self.data[wavelength]['map']
        
        # TODO: Make the map_kwargs and lc_kwargs parameters.
        fig, gs_maps, gs_lc = plotting.plot_overview(
            m,
            self.region_can.construct_given_map(map_=m),
            self.data[wavelength]['lightcurve'],
            self.boxcar_width,
            # map_kwargs
        )

        fig_path = self._build_figure_path(wavelength, 'overview')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_summary(self, wavelength):

        lightcurve = self.data[wavelength]['lightcurve']
        d = plotting.summary_lightcurves(lightcurve)
        fig_path = self._build_figure_path(wavelength, 'summary')
        d['fig'].set_layout_engine('constrained')
        d['fig'].savefig(fig_path, dpi=300)
        plt.close()