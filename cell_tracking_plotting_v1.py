# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:03:45 2021.

@author: Karlan
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

def plot_cell_z_movements(cell_df, nrows = 3, ncols = 4, plot_percentage = True, hist_xlim = False, hist_bins = 50,
                          return_fig = False, return_axes = False, width_ratios = [1,4,1,1], experiment_name = 'Undefined'):
    """
    Plot summary of cell movements in z.
    
    Makes plots for each compartment, summaryzing the number of cells in the compartment, 
    their movement compared to starting point and comparison of beginning and end compartments

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.
    nrows : int, optional
        Number of rows. The default is 3.
    ncols : int, optional
        Number of columns. The default is 4.
    plot_percentage : boolean, optional
        Plot cell numbers as percentages instead of absolute values. The default is True.
    hist_xlim : boolean or list, optional
        If True, sets x-limits based on the maximum of each row.
        If list, sets x-limits based on the supplied values. The default is False.
    hist_bins : int, optional
        How many bins to use for histogram. The default is 50.
    return_fig : boolean, optional
        Return figure object. The default is False.
    return_axes : boolean, optional
        Return axes object. The default is False.
    width_ratios : list, optional
        Specify column widths. The default is [1,4,1,1].
    experiment_name : str, optional
        Name of the current experiment. The default is 'Undefined'.

    Returns
    -------
    Depending on parameters, either returns nothing or figure and/or axes.

    """
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (25, 10), gridspec_kw = {'width_ratios': width_ratios, 'wspace': 0.2, 'hspace':0.3})
    for row, loc in enumerate(['crypt','villi','ambiguous']):
        tmp_tracks = {'up':{}, 'down':{}, 'same':{}}
        for track in set(cell_df['track_reindexed']):
            tmp_df = cell_df[cell_df['track_reindexed']==track]
            if not tmp_df.sort_values(by = 't')['z_class'].iloc[0] == loc:
                continue
            start = tmp_df.sort_values(by = 't')['z'].iloc[0]
            end = tmp_df.sort_values(by = 't')['z'].iloc[-1]
            if start == end:
                tmp_tracks['same'][track] = end-start
            elif end > start:
                tmp_tracks['up'][track] = end-start
            else:
                tmp_tracks['down'][track] = end-start

        height = [len(tmp_tracks['up']), len(tmp_tracks['down']), len(tmp_tracks['same'])]
        if plot_percentage:
            total = np.sum(height)
            height = len(tmp_tracks['up'])/total*100, len(tmp_tracks['down'])/total*100, len(tmp_tracks['same'])/total*100
        axes[row,0].bar(x = ['Higher', 'Lower', 'Same'], height = height)

        values = []
        for key in tmp_tracks.keys():
            values.append(sorted(list(tmp_tracks[key].values())))

        c_list = ['dodgerblue', 'g', 'orange']
        for ix, val in enumerate(values):
            axes[row,1].axvline(np.mean(val), ls = ':', c = c_list[ix])
            axes[row,1].axvline(np.median(val), ls = '-', c = c_list[ix])
            axes[row,1].hist(val, bins = hist_bins, color = c_list[ix])

        z_levels_end = {'crypt':[], 'villi':[],'ambiguous':[]}
        z_levels_start = {'crypt':[], 'villi':[],'ambiguous':[]}
        for direction in ['up','down','same']:
            for track in tmp_tracks[direction]:
                tmp_df = cell_df[cell_df['track_reindexed']==track]
                z_end = tmp_df.sort_values(by = 't')['z_class'].iloc[-1]
                z_start = tmp_df.sort_values(by = 't')['z_class'].iloc[0]
                z_levels_end[z_end].append(track)
                z_levels_start[z_start].append(track)

        height_start = [len(z_levels_start['crypt']), len(z_levels_start['villi']), len(z_levels_start['ambiguous'])]
        height_end = [len(z_levels_end['crypt']), len(z_levels_end['villi']), len(z_levels_end['ambiguous'])]
        if plot_percentage:
            total = np.sum(height_start)
            height_start = height_start/total*100
            height_end = height_end/total*100
        axes[row,2].bar(x = ['Crypt', 'Villus', 'Ambiguous'], height = height_start)
        axes[row,3].bar(x = ['Crypt', 'Villus', 'Ambiguous'], height = height_end, alpha = 0.5)

        # Set y-labels
        percentage_text = ' (%)' if plot_percentage else ''

        axes[row,0].set_ylabel('{}, n = {}\n\nNumber of cells{}'.format(loc.capitalize(), np.sum([len(x) for x in z_levels_start.values()]), percentage_text))
        axes[row, 1].set_ylabel('Number of cells (per bin)')
        axes[row, 2].set_ylabel('Number of cells{}'.format(percentage_text))
        axes[row, 3].set_ylabel('Number of cells{}'.format(percentage_text))

        # Set x-labels
        axes[row, 1].set_xlabel('Per cell change in z-level as compared to starting point')

    # Set titles
    axes[0,0].set_title('Exp: {}\nMovement of cells'.format(experiment_name))
    axes[0,1].set_title('Histogram of cell\'s change in z-level\nvertical lines show per-population mean (dashed) or median (solid)')
    axes[0,2].set_title('Starting location')
    axes[0,3].set_title('Final location')

    if hist_xlim:
        if type(hist_xlim) == bool:
            hist_xlim = [0,0]
            for row in np.arange(nrows):
                tmp_lim = axes[row, 1].get_xlim()
                if tmp_lim[0]<hist_xlim[0]:
                    hist_xlim[0] = tmp_lim[0]
                if tmp_lim[1]>hist_xlim[1]:
                    hist_xlim[1] = tmp_lim[1]

        for row in np.arange(nrows):
            axes[row, 1].set_xlim(hist_xlim)

    if plot_percentage:
        for row in np.arange(nrows):
            for col in [0,2,3]:
                axes[row, col].set_ylim([0,105])
                
    if all([return_fig, return_axes]): 
        return fig, axes
    elif return_fig: 
        return fig
    elif return_axes:
        return axes