import pathlib
import regions
import sunpy.map
import astropy.time
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import matplotlib.gridspec as gridspec
import numpy as np
import numpy.typing

from datetime import datetime, timedelta
from sunpy.net import attrs as a
from astropy.coordinates import SkyCoord

from . import boxcar, file_io, lightcurves as lc
from .data_classes import Lightcurve, RegionCanister


MAP_X_LABEL = 'X (arcseconds)'
MAP_Y_LABEL = 'Y (arcseconds)'
AIA_COLORMAP = {
    94  : 'red',
    131 : 'red',
    171 : 'cyan',
    193 : 'cyan',
    211 : 'cyan',
    304 : 'cyan',
    335 : 'red',
    1600: 'red',
    1700: 'cyan'
} # Sets compatible region colors for each AIA filter


def apply_style(style_sheet: str):
    p = pathlib.Path(__file__)
    plt.style.use(p.parent / f'styles/{style_sheet}')


def apply_colorbar(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes,
    width: float = 0.005,
    **kwargs
) -> tuple[matplotlib.axes, matplotlib.colorbar.Colorbar]:
    """
    Adds a custom colorbar to the figure.
    A new axes object is created to house the colorbar.
    Inspired by: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to which to colorbar will be added.
    ax : matplotlib.axes
        The axes to which the colorbar is applied.
    width : float
        The colorbar width as a fraction of the plot width.
    kwargs
        Keyword arguments to the matplotlib.colorbar.ColorbarBase method.
    Returns
    -------
    cbax : matplotlib axes
        The axes containing the colorbar.
    cb : matplotlib colorbar
        The newly created colorbar.
    """

    default_kwargs = {
        'spacing': 'uniform'
    }
    kwargs = {**default_kwargs, **kwargs}

    cbax = fig.add_axes([
        ax.get_position().x1+0.005, # Set spacing between the plot and cb
        ax.get_position().y0, # Set bottom of cb to bottom of the plot
        width, # Set the width of the cb as a fraction of the plot width
        ax.get_position().height]) # Set the height of cb to the plot height
    cb = matplotlib.colorbar.ColorbarBase(cbax, **kwargs)
    cbax.tick_params(which='both', axis='y', direction='out')

    return cbax, cb


# TODO: Do we need this?
def get_map(
    wavelength: u.Quantity,
    map_time: astropy.time.Time
) -> sunpy.map.Map:
    """
    Query and return the most recent AIA
    image that Sunpy.Fido can get.
    
    Parameters
    ----------
    wavelength : astropy.units.Quantity
        The desired AIA wavelength.
    map_time : astropy.time.Time
        Date and time of desired image formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.

    Returns
    -------
    aia_map : sunpy.map.Map
    """

    past_time = map_time - timedelta(minutes=10)
    info = ( a.Instrument('aia') & a.Wavelength(wavelength) )
    result = file_io.Fido.search(a.Time(past_time, map_time), info)
    file_download = file_io.Fido.fetch(
        result[0, -1],
        site='ROB',
        progress=False
    )

    aia_map = sunpy.map.Map(file_download[-1])
    
    return aia_map


def plot_map(map_obj, fig=None, ax=None, **cb_kwargs):
    """
    Plot the provided sunpy.map.Map.
    If fig and ax are provided, the map will be added to them.
    Otherwise, new fig and ax objects will be created.

    Parameters
    ----------
    map_obj : sunpy.map.Map
        The map of interest.
    fig : matplotlib figure object
        If a figure object is provided, the provided lightcurve data
        will be added to the existing fig object.
    ax : matplotlib axes object
        If an axes object is provided, the provided lightcurve data
        will be added to the existing ax object.
    cb_kwargs
        Keywork parameters for the colorbar (norm, cmap, etc.).

    Returns
    -------
    fig : matplotlib figure object
        The created figure object if fig was not provded,
        or the updated figure object if it was provided.
    ax : matplotlib axes object
        The created axes object if ax was not provded,
        or the updated axes object if it was provided.
    """

    
    default_kwargs = {
        'cmap': map_obj.cmap.reversed(),
        'norm': matplotlib.colors.PowerNorm(0, map_obj.max())
    }
    cb_kwargs = {**default_kwargs, **cb_kwargs}
    
    if fig is None:
        apply_style('map.mplstyle')
        fig, ax = plt.subplots(
            figsize=(12,12),
            subplot_kw={'projection':map_obj}
        )

    map_obj.plot(axes=ax, **cb_kwargs)
    map_obj.draw_limb(color='black')

    ax.set(xlabel=MAP_X_LABEL, ylabel=MAP_Y_LABEL)
    ax.grid(False)
    cbax, cb = apply_colorbar(fig, ax, 0.01, label='Intensity', **cb_kwargs)

    return fig, ax


def make_submap(
    map_obj: sunpy.map.Map,
    region: regions.SkyRegion
) -> sunpy.map.Map:
    """
    Creates a submap around the provided region based on map_obj.

    Parameters
    ----------
    obs_map : sunpy.map.Map
        The original map that the submap will be based on.
    region : regions.SkyRegion
        The region around which the submap will be made.

    Returns
    -------
    submap_obj : sunpy.map.Map
        The submap around the specified region.
    """

    bl, tr = get_subregion_corners(map_obj, region)
    submap_obj = map_obj.submap(bottom_left=bl, top_right=tr)

    return submap_obj


def add_region(
    map_obj: sunpy.map.Map,
    ax: plt.axes,
    region: regions.SkyRegion,
    **kwargs
):
    """
    Adds the provided region to ax.
    
    Parameters
    ----------
    map_obj : sunpy.map.Map
        The input map that the region is based on.
    ax : matplotlib axes object
        The axes on which the region will be drawn.
    region : regions.SkyRegion
        The region of interest to be drawn.
    **kwargs
        Style parameters of the plotted region.
    """

    default_kwargs = {
        'color': AIA_COLORMAP[map_obj._meta['wavelnth']],
        'linestyle': 'dashed',
        'linewidth': 2
    }
    kwargs = {**default_kwargs, **kwargs}

    region_pixel = region.to_pixel(map_obj.wcs)
    region_pixel.plot(ax=ax, **kwargs)


def get_subregion_corners(
    map_obj: sunpy.map.Map,
    region: regions.SkyRegion
) -> tuple[SkyCoord, SkyCoord]:
    """
    Obtain the bottom left and top right coordinates of the subregion
    specified by the provided coordinates.

    Parameters
    ----------
    map_obj : sunpy.map.Map
        The input map that the region is based on.
    region: regions.SkyRegion
    
    Returns
    -------
    bl : SkyCoord
        The bottom left point of the subregion.
    tr : SkyCoord
        The top right point of the subregion.
    """

    mask = region.to_pixel(map_obj.wcs).to_mask()
    xmin, xmax = mask.bbox.ixmin, mask.bbox.ixmax
    ymin, ymax = mask.bbox.iymin, mask.bbox.iymax

    bl = regions.PixCoord(xmin, ymin).to_sky(map_obj.wcs)
    tr = regions.PixCoord(xmax, ymax).to_sky(map_obj.wcs)

    return bl, tr


def get_region_data(
    map_obj: sunpy.map.Map,
    region: regions.SkyRegion,
    fill_value: float = 0,
    b_full_size: bool = False
) -> np.ndarray:
    """
    Get the map data contained within the provided region.

    Parameters
    ----------
    map_obj : sunpy.map.Map
        The map containing the region of interest.
    region : regions.SkyRegion
        The bounding region.
    fill_value : float
        The default null value in indices outside the region.
    b_full_size : bool
        Specifies whether the returned array, region_data,
        is the same shape as the input array, data.
        The default is False since it is wasteful in memory.

    Returns
    -------
    region_data : np.ndarray
        An array containing only the pixel information within
        the provided reg.
    """

    map_data = map_obj.data
    reg_mask = (region.to_pixel(map_obj.wcs)).to_mask()
    xmin, xmax = reg_mask.bbox.ixmin, reg_mask.bbox.ixmax
    ymin, ymax = reg_mask.bbox.iymin, reg_mask.bbox.iymax
    region_data = np.where(reg_mask.data==1, map_data[ymin:ymax, xmin:xmax], fill_value)

    if b_full_size:
        a = np.full(
            shape=map_data.shape,
            fill_value=fill_value,
            dtype=region_data.dtype
        )
        a[ymin:ymax, xmin:xmax] = region_data
        region_data = a

    return region_data


def plot_lightcurve(
    lightcurve: Lightcurve,
    fig: matplotlib.figure.Figure | None= None,
    ax: matplotlib.axes.Axes | None= None,
    **plot_kwargs
) -> tuple[matplotlib.figure.Figure, matplotlib.axes]:
    """
    A general method for plotting lightcurve data.
    If fig and ax are provided, then the lightcurve data
    is added onto the provided fig and ax objects.

    Parameters
    ----------
    lightcurve : tuple | Lightcurve
        Contains the lightcurve data in the format (times, data, [exposure_times]),
        where times and data are in a format (i.e. list or np.ndarray)
        that is compatible with matplotlib.axes plotting. The elements
        in times should be of type str.
    fig : matplotlib.figure.Figure | None
        If a figure object is provided, the provided lightcurve data
        will be added to the existing fig object.
    ax : matplotlib.axes | None
        If an axes object is provided, the provided lightcurve data
        will be added to the existing ax object.
    **plot_kwargs
        Style parameters of the plotted lightcurve.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object if fig was not provded,
        or the updated figure object if it was provided.
    ax : matplotlib.axes
        The created axes object if ax was not provded,
        or the updated axes object if it was provided.
    """

    default_kwargs = {
        'linestyle': 'dashed',
        'linewidth': 1,
        'marker': 'o',
        'markersize': 2,
        'color': 'black'
    }
    plot_kwargs = {**default_kwargs, **plot_kwargs}

    times_converted = [t.datetime for t in lightcurve[0]]
    lightcurve_converted = (times_converted, lightcurve[1])

    if fig is None and ax is None:
        apply_style('lightcurve.mplstyle')
        fig, ax = plt.subplots()

    _ = ax.plot(*lightcurve_converted, **plot_kwargs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))

    return (fig or plt.gcf()), ax


def make_overview_gridspec(
) -> tuple[matplotlib.figure.Figure, matplotlib.gridspec, matplotlib.gridspec]:
    """
    Prepares the gridspec objects for the plot_overview method.
    """

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    gs_maps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.4)
    gs_lc = gs[1].subgridspec(2, 1, hspace=0)

    return fig, gs_maps, gs_lc


def plot_overview(
    map_obj: sunpy.map.Map,
    region: regions.SkyRegion,
    lightcurve: Lightcurve,
    boxcar_width: int = None,
    map_kwargs: dict = {},
    lc_kwargs: dict = {},
) -> tuple[matplotlib.figure.Figure, matplotlib.gridspec, matplotlib.gridspec]:
    """
    Make an overview plot of the specified plot, region, and lightcurve.
    # TODO: Giving the map, region, and lightcurve parameters is kinda silly...
    # TODO: Also maybe add other kwargs options for the averaged and detrended plots?
    # TODO: Add the same datetime formatter used in the summary plots?

    Parameters
    ----------
    map_obj : sunpy.map.Map
        The map to plotted.
    region : regions.SkyRegion
        The region of interest.
    lightcurve : Lightcurve
        The lightcurve corresponding to the region.
    boxcar_width : int
        The width over which the lightcurve will be averaged.
        If no width is provided, no averaging is performed.
    map_kwargs : dict
        Style parameters for the maps.
    lc_kwargs : dict
        Style parameters for the lightcurve.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the maps and plots.
    gs_maps: matplotlib.gridspec
        The gridspec object containing the map axes.
    gs_lc: matplotlib.gridspec
        The gridspec object containing the lightcurve axes.
    """

    wavelength = map_obj._meta["wavelnth"]
    map_time = map_obj.date.strftime("%Y-%m-%d %H:%M:%S")

    fig, gs_maps, gs_lc = make_overview_gridspec()
    fig.suptitle(f'AIA {wavelength} {map_time}', fontsize=20)

    apply_style('map.mplstyle')
    map_ax = fig.add_subplot(gs_maps[0,0], projection=map_obj)
    plot_map(map_obj, fig, map_ax, **map_kwargs)
    add_region(map_obj, map_ax, region)
    map_ax.set_title('Full disk')

    submap_ = make_submap(map_obj, region)
    submap_ax = fig.add_subplot(gs_maps[0,1], projection=submap_)
    plot_map(submap_, fig, submap_ax, **map_kwargs)
    add_region(submap_, submap_ax, region)
    submap_ax.set_title('Selected Region')

    apply_style('lightcurve.mplstyle')
    lightcurve_ax = fig.add_subplot(gs_lc[0,:])

    plot_lightcurve(
        lightcurve,
        fig,
        lightcurve_ax,
        label='Lightcurve',
        **lc_kwargs
    )
    lightcurve_ax.set(
        xlabel='Time',
        ylabel='Intensity',
        title='Region lightcurve'
    )

    if boxcar_width is not None:

        averaged = lc.make_averaged_lightcurve(lightcurve, boxcar_width)
        plot_lightcurve(
            averaged, fig, lightcurve_ax,
            color='steelblue', alpha=0.75, linestyle='dotted',
            label=f'Boxcar, $N=${boxcar_width}'
        )
        plt.setp(lightcurve_ax.get_xticklabels(), visible=False)
        lightcurve_ax.legend()

        detrended_ax = fig.add_subplot(gs_lc[1,:])
        detrended = lc.make_detrended_lightcurve(lightcurve, averaged)
        plot_lightcurve(detrended, fig, detrended_ax, color='purple', label='Detrended')
        
        detrended_ax.set(ylabel='Residual')
        detrended_ax.axhline(y=0, color='gray', linestyle='dotted', linewidth=0.6)
        [xmin, xmax, ymin, ymax] = lightcurve_ax.axis()
        detrended_ax.set_xlim(left=xmin, right=xmax)        

    return fig, gs_maps, gs_lc


def summary_lightcurves(dat: Lightcurve) -> dict[str, object]:
    
    apply_style('summary.mplstyle')

    sorted_dat = lc.sort_by_time(dat)
    dtimes = (sorted_dat.t).datetime

    conv = mdates.ConciseDateConverter()
    orig = munits.registry[datetime]
    munits.registry[datetime] = conv

    fig, (raw_ax, norm_ax, exp_ax) = plt.subplots(
        ncols=1, nrows=3,
        layout='constrained',
        figsize=(18, 10),
        sharex=True
    )

    shared_kw = dict(markersize=8, marker='.', lw=1)
    raw_ax.plot(dtimes, sorted_dat.y, color='red', **shared_kw)
    raw_ax.set(ylabel='Unnormalized light curve [arb]')

    ets = np.array([et.value for et in sorted_dat.exposure_times])
    exp_ax.scatter(dtimes, ets, color='blue', s=2)
    exp_ax.set(ylabel='Exposure time per frame [s]')

    norm_ax.plot(dtimes, np.array(sorted_dat.y) / np.array(ets), color='blue', **shared_kw)
    norm_ax.set(ylabel='Normalized light curve [arb / sec]')

    munits.registry[datetime] = orig
    return {
        'fig': fig,
        'raw_ax': raw_ax,
        'norm_ax': norm_ax,
        'exp_ax': exp_ax
    }


InsetRet = dict[str, matplotlib.figure.Figure | matplotlib.axes.Axes]
def plot_inset_region(
     fits_path: str | pathlib.Path,
     region_can: RegionCanister,
     fig: matplotlib.figure.Figure | None=None,
     reg_kw: dict[str, object] | None=None,
     inset_position: numpy.typing.ArrayLike | None=None
) -> InsetRet:

     m = sunpy.map.Map(fits_path)

     pad_mult = 1.5
     hw = region_can.half_width()
     bottom_left = SkyCoord(
          *(region_can.center - hw*pad_mult),
          frame=m.coordinate_frame
     )
     top_right = SkyCoord(
          *(region_can.center + hw*pad_mult) << u.arcsec,
          frame=m.coordinate_frame
     )

     subm = m.submap(bottom_left=bottom_left, top_right=top_right)

     fig = fig or plt.gcf()
     ax = fig.add_subplot(projection=m)
     m.plot(axes=ax)
     main_tick_col = 'gray'
     ax.coords.frame.set_color(main_tick_col)
     for crd in ax.coords:
          crd.tick_params(color=main_tick_col)

     pos = inset_position or [0.35, 0.35, 0.3, 0.3]
     axins = ax.inset_axes(pos, projection=subm)
     subm.plot(annotate=False, axes=axins, title=False)

     skc = SkyCoord(*region_can.center, frame=m.coordinate_frame)
     reg = region_can.kind(
          skc,
          **region_can.constructor_kwargs
     )
     default = dict(lw=2, color='lightblue', ls='dashed')
     reg_kw = default | (reg_kw or dict())
     add_region(map_obj=m, ax=ax, region=reg, **reg_kw)
     add_region(map_obj=subm, ax=axins, region=reg, **reg_kw)

     indic_col = (1, 1, 1, 0.8)
     x0, y0 = bottom_left.to_pixel(m.wcs)
     x1, y1 = top_right.to_pixel(m.wcs)
     ax.indicate_inset(
          bounds=(x0, y0, x1 - x0, y1 - y0),
          inset_ax=axins,
          edgecolor=indic_col,
          linewidth=2,
          alpha=indic_col[-1]
     )

     for crd in axins.coords:
          crd.set_ticks_visible(False)
          crd.set_ticklabel_visible(False)
     axins.set(
          title=' ', xlabel=' ', ylabel=' ',
     )
     axins.coords.frame.set_color(indic_col)
     axins.grid(False)

     return dict(fig=fig, ax=ax, axins=axins)