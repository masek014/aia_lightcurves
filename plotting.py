import sunpy.map
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from sunpy.net import attrs as a
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

from . import convert, file_io, utils
np = utils.np

plt.rcParams['font.size'] = 8


MAP_X_LABEL = 'X (arcseconds)'
MAP_Y_LABEL = 'Y (arcseconds)'
COLORMAP = {
    94:'red', 131:'red', 171:'cyan', 193:'cyan', 211:'cyan',
    304:'cyan', 335:'red', 1600:'red', 1700:'cyan'
} # Sets compatible region colors for each AIA filter


def add_minor_ticks(ax):
    """
    Adds minor ticks to the plot on the provided axes.

    Parameters
    ----------
    ax : matplotlib axes object
        The axes to which ticks will be added.
    """

    (ax.coords[0]).display_minor_ticks(True)
    (ax.coords[0]).set_minor_frequency(5)
    (ax.coords[1]).display_minor_ticks(True)
    (ax.coords[1]).set_minor_frequency(5)
    ax.tick_params(which='minor', length=1.5)


def apply_colorbar(fig, ax, width=0.005, **kwargs):
    """
    Adds a colorbar to the map plot.
    A new axes object is created to house the colorbar.
    Parameters
    ----------
    fig : matplotlib figure
        The figure to which to colorbar will be added.
    ax : matplotlib axes
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

    # Create a custom colorbar.
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    cbax = fig.add_axes([
        ax.get_position().x1+0.005, # Set spacing between the plot and cb
        ax.get_position().y0, # Set bottom of cb to bottom of the plot
        width, # Set the width of the cb as a fraction of the plot width
        ax.get_position().height]) # Set the height of cb to the plot height
    cb = matplotlib.colorbar.ColorbarBase(cbax, **kwargs)

    return cbax, cb


def get_map(aia_wavelength, map_time):
    """
    Query and return the most recent AIA
    image that Sunpy.Fido can get.
    
    Parameters
    ----------
    aia_wavelength : int
        The desired AIA wavelength.
    map_time : str
        Date and time of desired image formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.

    Returns
    -------
    Sunpy generic map.
    """

    DATETIME_FORMAT = convert.DATETIME_FORMAT
    
    info = (a.Instrument('aia') & a.Wavelength(aia_wavelength*u.angstrom))
    look_back = {'minutes': 30}
    
    map_dt = convert.datetime.strptime(map_time, DATETIME_FORMAT+'.%f')
    past = map_dt - convert.timedelta(**look_back)
    past_date = past.strftime(DATETIME_FORMAT)
    startt = str(past_date)
    endt = str(map_time)

    result = file_io.Fido.search(a.Time(startt, endt), info)
    file_download = file_io.Fido.fetch(result[0, -1], site='ROB', progress=False)

    data_map = sunpy.map.Map(file_download[-1])
    
    return data_map


def get_brightest_skycoord(map_obj):
    """
    Return the coordinate of the brightest coordinate as a SkyCoord.

    Parameters
    ----------
    map_obj : Sunpy map
        The input map.
    
    Returns
    -------
    max_coords : SkyCoord
        The coordinate of the brightest point on the map.
        In units of arcseconds.
    """
    
    max_coords = utils.np.where(map_obj._data==utils.np.max(map_obj._data)) # row, column
    max_coords = (max_coords[1][0], max_coords[0][0]) # column, row (x, y)
    max_coords = map_obj.wcs.pixel_to_world(*max_coords)

    return max_coords


def plot_map(map_obj, fig=None, ax=None, title='', **cb_kwargs):
    """
    Plot the provided Sunpy map.
    If fig and ax are provided, the map will be added to them.
    Otherwise, new fig and ax objects will be created.

    Parameters
    ----------
    map_obj : Sunpy map
        The map of interest.
    fig : matplotlib figure object
        If a figure object is provided, the provided lightcurve data
        will be added to the existing fig object.
    ax : matplotlib axes object
        If an axes object is provided, the provided lightcurve data
        will be added to the existing ax object.
    title : str
        The title string.
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

    defaultKwargs = {
        'cmap': map_obj.cmap.reversed(),
        'norm': matplotlib.colors.LogNorm(10, map_obj.max())
    }
    cb_kwargs = {**defaultKwargs, **cb_kwargs}

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,12), subplot_kw={'projection':map_obj})

    map_obj.plot(axes=ax, title=title, **cb_kwargs)
    map_obj.draw_limb(color='black')

    ax.tick_params(which='major', direction='in')
    ax.grid(False)
    ax.set(xlabel=MAP_X_LABEL, ylabel=MAP_Y_LABEL)
    add_minor_ticks(ax)
    cbax, cb = apply_colorbar(fig, ax, 0.01, label='Intensity', **cb_kwargs)

    return fig, ax


def make_submap(obs_map, center, radius):
    """
    Creates a submap around the provided circular region based on obs_map.
    Center is a tuple of (x,y), in arcseconds
    radius is the radius in arcseconds.

    Parameters
    ----------
    obs_map : Sunpy map
        The original map that the submap will be beased on.
    center : tuple
        The center point coordinates of the circular region.
        In units of arcseconds.
    radius : float
        The radius of the circular region.
        In units of arcseconds.

    Returns
    -------
    obs_submap : Sunpy map
        The submap around the specified region.
    """

    bl, tr = get_subregion_corners(obs_map, center, radius)
    obs_submap = obs_map.submap(bottom_left=bl, top_right=tr)

    return obs_submap


def add_region(map_obj, ax, center, radius, **kwargs):
    """
    Adds a circular region to the provided axes.
    
    Parameters
    ----------
    map_obj : Sunpy map
        The input map that the region is based on.
    ax : matplotlib axes object
        The axes on which the region will be drawn.
    center : tuple
        Coordinates for the region center, (x,y), in arcseconds.
    radius : float
        The radius of the circular region.
    **kwargs
        Style parameters of the plotted region.

    Returns
    -------
    reg : CircleSkyRegion
        Region object for the specified area.
    """

    defaultKwargs = {'color': 'red', 'linestyle': 'dashed', 'linewidth': 1}
    kwargs = {**defaultKwargs, **kwargs}

    center_skycoord = SkyCoord(*center, unit='arcsecond', frame=map_obj.coordinate_frame)
    reg = CircleSkyRegion(center_skycoord, radius*u.arcsecond)
    reg_pixel = reg.to_pixel(map_obj.wcs)
    reg_pixel.plot(ax=ax, **kwargs)

    return reg


def get_subregion_corners(map_obj, center, radius):
    """
    Obtain the bottom left and top right coordinates of the subregion
    specified by the provided coordinates.

    Parameters
    ----------
    map_obj : Sunpy map
        The input map that the region is based on.
    center : tuple
        Coordinates for the region center, (x,y), in arcseconds.
    radius : float
        The radius of the circular region.
    
    Returns
    -------
    bl : SkyCoord
        The bottom left point of the subregion.
    tr : SkyCoord
        The top right point of the subregion.
    """

    bl_x, bl_y = center[0] - radius, center[1] - radius
    bl = SkyCoord(bl_x*u.arcsecond, bl_y*u.arcsecond, frame=map_obj.coordinate_frame)

    tr_x, tr_y = center[0] + radius, center[1] + radius
    tr = SkyCoord(tr_x*u.arcsecond, tr_y*u.arcsecond, frame=map_obj.coordinate_frame)

    return bl, tr


def get_region_data(map_obj, reg, fill_val=0, b_full_size=False):
    """
    Get the map data contained within the provided region.

    Parameters
    ----------
    map_obj : Sunpy map
        The map containing the region of interest.
    reg : PixelRegion
        The bounding region.
    fill_val : float
        The default null value in indices outside the region.
    b_full_size : bool
        Specifies whether the returned array, reg_data,
        is the same shape as the input array, data.
        The default is False since it is wasteful in memory.

    Returns
    -------
    reg_data : np.ndarray
        An array containing only the pixel information within
        the provided reg.
    """

    data = map_obj.data
    reg_mask = (reg.to_pixel(map_obj.wcs)).to_mask()
    xmin, xmax = reg_mask.bbox.ixmin, reg_mask.bbox.ixmax
    ymin, ymax = reg_mask.bbox.iymin, reg_mask.bbox.iymax
    reg_data = np.where(reg_mask.data==1, data[ymin:ymax, xmin:xmax], fill_val)

    if b_full_size:
        a = np.full(shape=data.shape, fill_value=fill_val, dtype=reg_data.dtype)
        a[ymin:ymax, xmin:xmax] = reg_data
        reg_data = a

    return reg_data


def make_lightcurve(start_time, end_time, wavelength, center, radius):
    """
    Obtain the lightcurve data for the specified parameters
    for the given region. Currently, the intensities are
    **not** normalized in any way. The units/scale is
    arbitrary. There is a built-in procedure in the event
    that a FITS file is corrupted. It will delete the
    corrupted file and attempt to redownload it.

    Parameters
    ----------
    start_time : str
        The start time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    end_time : str
        The end time of the observation. Formatted as
        '%Y-%m-%dT%H:%M:%S.%f'.
    wavelength : int
        The wavelength of interest.
    center : tuple
        Coordinates for the region center, (x,y), in arcseconds.
    radius : float
        The radius of the circular region.

    Returns
    -------
    times : list
        The times for each data point.
    intensities : list
        The lightcurve values at each time.
    """

    print('Generating light curve data.')

    files = file_io.download_fits2(start_time, end_time, wavelength)
    # files = file_io.download_fits_parallel(start_time, end_time, wavelength)
    times, intensities = [], []

    # Download the files and handle potential file corruption.
    # Redownload the file if it is corrupted.
    for f in files:
        attempt_number = 1
        while attempt_number <= file_io.MAX_DOWNLOAD_ATTEMPTS:
            try:
                m = sunpy.map.Map(f)
                break
            except IOError as e: # File corruption, attempt redownload.
                print(e)
                print(f'FITS read error on attempt {attempt_number}.\
                     Deleting file and retrying download.')
                f = file_io.redownload_file(f)
                continue
        # This is only True if the except block is executed for all attempts.
        if attempt_number > file_io.MAX_DOWNLOAD_ATTEMPTS:
            print('Exceeded maximum number of attempts. Exitting.')
            file_io.sys.exit(1)

        bl, tr = get_subregion_corners(m, center, radius)
        subm = m.submap(bottom_left=bl, top_right=tr)
        reg = CircleSkyRegion(SkyCoord(*center, unit='arcsecond',
            frame=subm.coordinate_frame), radius*u.arcsecond)
        reg_data = get_region_data(subm, reg)

        time_str = m.date.value#.split('.')[0] # Remove the milliseconds
        times.append(convert.str_to_epoch(time_str))
        intensities.append(np.sum(reg_data))

    return times, intensities


def plot_lightcurve(lightcurve, fig=None, ax=None, xlabel='', ylabel='', title='', **plot_kwargs):
    """
    A general method for plotting lightcurve data.
    If fig and ax are provided, then the lightcurve data
    is added onto the provided fig and ax objects.

    Parameters
    ----------
    lightcurve : tuple
        Contains the lightcurve data in the format (times, data),
        where times and data are in a format (i.e. list or np.ndarray)
        that is compatible with matplotlib.axes plotting. The elements
        in times should be of type str.
    fig : matplotlib figure object
        If a figure object is provided, the provided lightcurve data
        will be added to the existing fig object.
    ax : matplotlib axes object
        If an axes object is provided, the provided lightcurve data
        will be added to the existing ax object.
    xlabel : str
        The x-axis label.
    ylabel : str
        The y-axis label.
    title : str
        The title string.
    **plot_kwargs
        Style parameters of the plotted lightcurve.

    Returns
    -------
    fig : matplotlib figure object
        The created figure object if fig was not provded,
        or the updated figure object if it was provided.
    ax : matplotlib axes object
        The created axes object if ax was not provded,
        or the updated axes object if it was provided.
    """

    defaultKwargs = {
        'linestyle':'dashed', 'linewidth':0.6, 'marker':'o',
        'markersize':2, 'color':'black'
    }
    plot_kwargs = {**defaultKwargs, **plot_kwargs}

    # Convert the time data to datetime objects.
    converted_datetimes = list(map(convert.epoch_to_datetime, lightcurve[0]))
    # norm_intensities = lightcurve[1]/np.sum(lightcurve[1]) # TODO: Decide normalization

    lightcurve_converted = (converted_datetimes, lightcurve[1])

    if fig is None:
        fig, ax = plt.subplots(figsize=(12,4))
    
    line = ax.plot(*lightcurve_converted, **plot_kwargs)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
    ax.tick_params(which='major', direction='in')
    ax.tick_params(which='minor', direction='in')
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    return fig, ax


def process_observation(obs):
    """
    This method compiles all functionality into one location to take
    advantage of all of the custom methods in this package.

    Example of an observation dictionary:
    obs = {
        'start_time': '2019-04-03T17:40:00.0', # The observation start time
        'end_time': '2019-04-03T17:55:00.0',   # The observation end time
        'map_time': '2019-04-03T18:00:33.0',   # The time used for plotting the maps
        'wavelengths': [171],                  # The wavelengths to examine
        'center': (-220, 330),                 # The center of the region of interest (arcseconds)
        'radius': 50,                          # The radius of the region of interest (arcseconds)
        'N': 21,                               # The boxcar width
        'name': 'reg1'                         # The region name to delineate files
    }

    Parameters
    ----------
    obs : dict
        A dictionary containing all of the observation information.
    
    Returns
    -------
    dicts : list of dicts
        Each dict in the list contains processed data for each
        wavelength examined in the observation.
        ex: {'obs_map': obs_map, 'obs_submap': obs_submap, 'lightcurve': lightcurve, 'fig': fig}
    """

    start_date = obs['start_time'].split('T')[0].replace('-', '')
    end_date = obs['end_time'].split('T')[0].replace('-', '')
    file_io.make_directories(start_date)
    file_io.make_directories(end_date)
    lightcurves_dir = file_io.LIGHTCURVES_DIR_FORMAT.format(date=start_date)
    images_dir = file_io.IMAGES_DIR_FORMAT.format(date=start_date)
    print(f'Generating plots for {obs["start_time"].replace("T", " ")} through {obs["end_time"].replace("T", " ")}')
    
    startt = obs['start_time'].replace('-','').replace(':','')
    endt = obs['end_time'].replace('-','').replace(':','')
    dicts = []

    for wavelength in obs['wavelengths']:
        
        print('Processing data for wavelength ' + str(wavelength))

        # Adjust the value of the boxcar width to accommodate the available data points.
        r = file_io.Fido.search(a.Time(obs['start_time'], obs['end_time']),
            a.Instrument('aia'), a.Wavelength(wavelength*u.angstrom), a.Sample(12 * u.second))
        obs['N'] = utils.adjust_n(len(r[0]), obs['N'])

        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        gs_maps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.3)
        gs_lc = gs[1].subgridspec(2, 1, hspace=0)

        # Make the full map.
        obs_map = get_map(wavelength, obs['map_time'])
        map_ax = fig.add_subplot(gs_maps[0,0], projection=obs_map)
        fig, map_ax = plot_map(obs_map, fig, map_ax, title='Full disk')
        reg = add_region(obs_map, map_ax, obs['center'], obs['radius'],
            label='Region', color=COLORMAP[wavelength])

        # Make the submap around the region of interest.
        obs_submap = make_submap(obs_map, obs['center'], obs['radius'])
        submap_ax = fig.add_subplot(gs_maps[0,1], projection=obs_submap)
        fig, submap_ax = plot_map(obs_submap, fig, submap_ax,
            title=f'Selected region: {obs["center"]}, r={obs["radius"]}')
        submap_reg = add_region(obs_submap, submap_ax, obs['center'], obs['radius'],
            alpha=1, label='Region', color=COLORMAP[wavelength])
        
        csv_file = f'{lightcurves_dir}lc_{startt}-{endt}_{wavelength}_N{obs["N"]}_{obs["name"]}.csv'
        if file_io.os.path.exists(csv_file):
            lightcurve = file_io.read_lightcurves(csv_file)
        else:
            # Then make the lightcurve.
            lightcurve = make_lightcurve(obs['start_time'], obs['end_time'], wavelength, obs['center'], obs['radius'])
        
        bc, N = utils.boxcar_average(lightcurve[1], obs['N'])
        lightcurve_bc = (lightcurve[0], bc)
        lightcurve_detrended = (lightcurve[0], lightcurve[1] - lightcurve_bc[1])

        # Plot lightcurve with boxcar average.
        lightcurve_ax = fig.add_subplot(gs_lc[0,:])
        fig, lightcurve_ax = plot_lightcurve(lightcurve, fig, lightcurve_ax,
            xlabel='', ylabel='Intensity', title='Region lightcurve',
            markersize=1, linestyle='solid', label='Lightcurve')
        plot_lightcurve(lightcurve_bc, fig, lightcurve_ax,
            xlabel='', ylabel='Intensity', title='Region lightcurve',
            markersize=1, color='steelblue', alpha=0.5, linestyle='dotted', label=f'Boxcar, $N=${obs["N"]}')
        plt.setp(lightcurve_ax.get_xticklabels(), visible=False)
        lightcurve_ax.legend(prop={'size': 6})

        # Plot detrended curve.
        detrended_ax = fig.add_subplot(gs_lc[1,:])
        fig, detrended_ax = plot_lightcurve(lightcurve_detrended, fig, detrended_ax,
            xlabel='Time', ylabel='Residual', title='',
            markersize=1, color='purple', label='Detrended')
        detrended_ax.axhline(y=0, color='gray', linestyle='dotted', linewidth=0.6, label='Zero')
        detrended_ax.tick_params(which='major', bottom=True, top=True)
        detrended_ax.tick_params(which='minor', bottom=True, top=True)
        [xmin, xmax, ymin, ymax] = lightcurve_ax.axis()
        detrended_ax.set_xlim(left=xmin, right=xmax)

        file_io.save_lightcurves((lightcurve[0], lightcurve[1]), csv_file)

        title = 'AIA ' + str(int(obs_map.wavelength.value)) + f' {obs_map.date}'.replace('T', ' ')
        fig.suptitle(title)

        fig_file = f'{images_dir}plots_{startt}-{endt}_{wavelength}_N{obs["N"]}_{obs["name"]}.png'
        plt.savefig(fig_file, bbox_inches='tight', dpi=300)
        print(f'Saved plots to \'{fig_file}\'')

        # Package the generated products into a dictionary.
        d = {'obs_map': obs_map, 'obs_submap': obs_submap, 'lightcurve': lightcurve, 'fig': fig}
        dicts.append(d)

    return dicts
