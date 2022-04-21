# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:03:18 2021.

@author: Karl Annusver
"""


import os
import read_roi
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
from IPython.display import clear_output

import warnings
warnings.simplefilter('ignore', np.RankWarning)


def retreive_trackmate_settings(path):
    """
    Retrieve image information supplied to TrackMate.
    
    Get image size, pixel scaling and z-step from the TrackMate .xml file.

    Parameters
    ----------
    path : str
        Path to the folder with experiment files.

    Returns
    -------
    corrected_dimensions : list
        Image shape in pixels [x, y].
    xy_scaling : float
        Pixel scaling information.
    z_step : float
        Z-step size.

    """
    #tree = ET.parse(os.path.join(path, [x for x in os.listdir(path) if '.xml' in x][0]))
    tree = ET.parse(path)
    root = tree.getroot()
    settings_pos = np.where([child.tag=='Settings' for child in root])[0][0]
    imagedata_pos = np.where([child.tag=='ImageData' for child in root[settings_pos]])[0][0]
    settings = root[settings_pos][imagedata_pos].attrib

    corrected_dimensions = [int(settings['width']), int(settings['height'])]
    xy_scaling = float(settings['pixelwidth'])
    z_step = float(settings['voxeldepth'])
    z_slices = int(settings['nslices'])

    print('Information retreived from TrackMate xml file:')
    print('Image Size:\t', corrected_dimensions)
    print('xy_scaling:\t', xy_scaling)
    print('z_step:\t\t', z_step)
    print('z_slices:\t\t', z_slices)
    
    return corrected_dimensions, xy_scaling, z_step, z_slices

def get_file_paths_tracking_ld(path, exp_name):
    """
    Select filenames for tracking and live/dead ROIs.
    
    Wrapper to call and unpack the get_file_paths() function

    Parameters
    ----------
    path : str
        Main directory path with experiment folders.
    exp_name : str
        Experiment name.

    Returns
    -------
    tracked_cells_path : str
        Path to tracking .csv file.
    ld_path : str
        Path to live/dead ROI file.
        
    """
    filenames = get_file_paths(path, exp_name, title = 'Select csv and roi files')
    tracked_cells_path = [x for x in filenames if '.csv' in x][0]
    ld_path = [x for x in filenames if 'Roi' in x][0]
    print('Tracking file path:\t', tracked_cells_path)
    print('Live\\Dead Roi path:\t', ld_path)
    print('')
    
    return tracked_cells_path, ld_path


def get_file_paths_xml(path, exp_name):
    """
    Select filenames for tracking and live/dead ROIs.
    
    Wrapper to call and unpack the get_file_paths() function

    Parameters
    ----------
    path : str
        Main directory path with experiment folders.
    exp_name : str
        Experiment name.

    Returns
    -------
    xml_file : str
        Path to tracking .xml file.
        
    """
    xml_file = get_file_paths(path, exp_name, title = 'Select the tracking .xml file')[0]
    print('Xml file path:', xml_file)
    
    return xml_file



def get_file_paths(path, exp_name, title = None):
    """
    User input dialog to select files.
    
    Interactive dialog to prompt the user to select files.
    Returns a list of the paths to selected files

    Parameters
    ----------
    path : str
        Main directory path with experiment folders.
    exp_name : str
        Experiment name.
    title : str, optional
        Title for the file selection window.

    Returns
    -------
    filenames : str
        List of paths to selected filenames

    """
    root = Tk()
    root.withdraw() #We don't actually want to show a GUI window
    root.update_idletasks()
    root.overrideredirect(True)
    root.geometry('0x200+200+200')
    root.deiconify()
    root.lift()
    root.focus_force()
    filenames = filedialog.askopenfilenames(initialdir = os.path.join(path, exp_name), 
                                            parent = root, title=title) # show an "Open" dialog box and return the path to the selected file
    # Get rid of the top-level instance
    root.destroy()
    
    return filenames

def correct_measurements(cell_df, image_pixel_size = [512,512], minutes_per_frame = 2, xy_scaling = 1.405, z_step = 2.5,
                         imaging_start = 0, correct_scaling = True, default_xy_scaling = 1.405, default_z_step = 2.5, z_slices = None):
    """
    Adjust measurements and correct them for downstream usage.
    
    If needed, pixel scaling is corrected to what it should be
    Replace the DataFrame index with the "ID" column.
    Make measurements more usable and understandable in python.
    Correct for image size (in case of drift correction) and timepoint step (minutes)
    Add easy-to-use columns from the original x,y,z,t column names
    Reindex track numbering

    Parameters
    ----------
    cell_df : pandas.DataFrame
        DataFrame containing results from tracking.
    image_pixel_size : list of int, optional
        Shape of the image. The default is [512,512].
        Used to correctly set the y-coordinates, because the raw output from ImageJ has y-coordinates start from top!
    minutes_per_frame : int; float, optional
        Number of minutes between each imaging frame. The default is 2.
    xy_scaling : float, optional
        um/px scaling for the imaging. Should be 1.405 um/px in all images. Needed to correctly calculate xy positions. The default is 1.405.
    z_step : float, optional
        um/z-slice scaling for the imaging. Should be 2.5um/slice. The default is 2.5.
    imaging_start : int, optional
        Timepoint when imaging started.
    correct_scaling : boolean, optional
        Whether to run pixel scale correction or not.
    default_xy_scaling : int, optional
        The xy_scaling value (um/px) that the image should have. The default is 1.405.
    default_z_step : int, optional
        The z_step value (um/step) that the image should have. The default is 2.05.
    z_slices : int
        Number of z-slices in the tracking (obtained from the xml file or checking the image)

    Returns
    -------
    cell_df : pandas.DataFrame
        Corrected DataFrame.
    If pixel correction is done, updates and returns the variables "xy_scaling" and "z_step" to their default values
        
    """
    if correct_scaling:
        cell_df, xy_scaling, z_step = correct_pixel_scaling(cell_df, xy_scaling, z_step, image_pixel_size = image_pixel_size, z_slices = z_slices,
                                                            default_xy_scaling = default_xy_scaling, default_z_step = default_z_step)
        
    cell_df = cell_df.reset_index().set_index('ID')
    
    print('Adding columns \'x\',\'y\',\'z\'')
    cell_df['x'] = cell_df['POSITION_X'].values
    cell_df['y'] = (image_pixel_size[1]*xy_scaling)-cell_df['POSITION_Y'].values
    cell_df['z'] = cell_df['POSITION_Z'].values
    print('Saving timelapse minutes in column \'t_raw\' and adjusted start time in column \'t\'')
    if imaging_start%2!=0: print(f'Changing start time to {imaging_start+1} to make it even and fit with other datasets')
    cell_df['t_raw'] = cell_df['FRAME'] * minutes_per_frame
    
    #Correct for imaging start time after seeding
    #If start time is an odd number, add 1 to it to make it even
    #This prevents crazy fluctuations when different experiments are later combined into shared analysis
    cell_df['t'] = np.array(cell_df['FRAME'] * minutes_per_frame) + (imaging_start if imaging_start%2==0 else imaging_start+1)
    
    #Filter out identified cells that do not belong to a track - they weren't tracked
    cell_df = cell_df[cell_df['TRACK_ID'] != 'None']
    
    #cell_df['track_id'] = [int(x) for x in cell_df['TRACK_ID']]
    
    #Give tracks new indexes in linear order
    track_reindex_dict = {x:ix for ix, x in enumerate(sorted(set(cell_df['TRACK_ID'])))}
    cell_df['track_reindexed'] = [track_reindex_dict[track] for track in cell_df['TRACK_ID']]
    
    return cell_df, xy_scaling, z_step

def check_xy_pixel_scaling(cell_df, image_pixel_size = [512, 512]):
    """
    Check that pixel scaling has actually happened.
    
    Compares the maximum x and y position to the image pixel size - 
    if xy-scaling has occured, then the maximum values will be greater than the pixel sizes.
    If any of the maximum x or y values are still smaller than image pixel size, then return True to rescale coordinates

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.
    image_pixel_size : list, optional
        Image pixel sizes [x, y]. The default is [512, 512].

    Returns
    -------
    bool
        True if image should be rescaled, False if scaling has been done.

    """
    max_x = max(cell_df['POSITION_X'])
    max_y = max(cell_df['POSITION_Y'])
    if any([max_x < image_pixel_size[0], max_y < image_pixel_size[1]]):
        return True
    else: return False

def check_z_pixel_scaling(cell_df, z_slices):
    """
    Check if the maximum z-level is higher than number of z-slices in the image.
    
    If the maximum z_level is lower than the number of slices, then z-level hasn't been scaled to um.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.
    z_slices : int
        Number of z-slices in the tracking (obtained from the xml file or checking the image)

    Returns
    -------
    bool
        True if z-level should be rescaled, False if scaling has been done.

    """
    return max(cell_df['POSITION_Z']) < z_slices

def correct_pixel_scaling(cell_df, xy_scaling, z_step, z_slices,
                          default_xy_scaling = 1.405, default_z_step = 2.5, image_pixel_size = [512, 512]):
    """
    Correct for the difference in pixel scaling.
    
    First checks if the image has been scaled by comparing image shape in pixels and z-slices to x, y, and z-coordinates.
    Then checks if the pixel scaling parameters "xy_scaling" and "z_step" differ from what they should be in the images.
    If there is a difference, it corrects for the difference in scaling and updates the global variables so that they can be used later
    

    Parameters
    ----------
    cell_df : pandas.DataFrame
        DataFrame containing results from tracking.
    xy_scaling: float
        um/px scaling for the imaging set.
    z_step: float
        um/z-slice scaling for the imaging set.
    z_slices : int
        Number of z-slices in the tracking (obtained from the xml file or checking the image)
    default_xy_scaling : int, optional
        The xy_scaling value (um/px) that the image should have.. The default is 1.405.
    default_z_step : int, optional
        The z_step value (um/step) that the image should have. The default is 2.5.
    image_pixel_size : list, optional
        Image pixel sizes [x, y]. The default is [512, 512].

    Returns
    -------
    Updates and returns the working DataFrame, xy_scaling and z_step.

    """
    #Check xy coordinates
    if check_xy_pixel_scaling(cell_df, image_pixel_size = image_pixel_size):
        print(f'Image pixel scaling looks incorrect! Rescaling coordinates. Correction factor: {default_xy_scaling}')
        for col in ['POSITION_X','POSITION_Y']:
            cell_df[col] = cell_df[col]*default_xy_scaling
        xy_scaling = default_xy_scaling
    
    elif xy_scaling != default_xy_scaling:
        print(f'Correcting x- and y-scaling. Correction factor: {default_xy_scaling/xy_scaling}')
        for col in ['POSITION_X','POSITION_Y']:
            cell_df[col] = cell_df[col]*default_xy_scaling/xy_scaling
        xy_scaling = default_xy_scaling
    
    #Check z-coordinates
    if check_z_pixel_scaling(cell_df, z_slices):
        print(f'Z-scaling looks incorrect! Rescaling z-levels. Correction factor: {default_z_step}')
        cell_df['POSITION_Z'] = cell_df['POSITION_Z']*default_z_step
        z_step = default_z_step
        
    elif z_step != default_z_step:
        print(f'Correcting z-scaling. Correction factor: {default_z_step/z_step}')
        cell_df['POSITION_Z'] = cell_df['POSITION_Z']*default_z_step/z_step
        z_step = default_z_step
    
    return cell_df, xy_scaling, z_step

def filter_tracks(cell_df, fraction_of_max = 4, inplace = False):
    """
    Remove tracks that are present in too few timepoints.
    
    This was useful in the beginning where we included more tracks

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Input DataFrame.
    fraction_of_max : int, optional
        Lowest timepoint to include, depending on how it relates to the longest timepoint. 
        Set to 1 to select everything. The default is 4.
    inplace : boolean, optional
        Whether to change the values in the input DataFrame. The default is False.

    Returns
    -------
    pandas.DataFrame
        Returns the filtered DataFrame if inplace = True.

    """
    tracks_to_keep = cell_df['TRACK_ID'].value_counts() >= cell_df['TRACK_ID'].value_counts().max()//fraction_of_max
    
    if inplace:
        cell_df = cell_df[tracks_to_keep[cell_df['TRACK_ID']].values]
    else:
        tmp = cell_df.copy(deep = True)
        return tmp[tracks_to_keep[tmp['TRACK_ID']].values]
    
    
def classify_z(cell_df, dmz, z_step = 2.5):
    """
    Classify each cell based on it's location in z.
    
    Adds values to the input DataFrame
    Adds column z_crypt showing cell position relative to the crypt mouth

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.
    dmz : list
        List with 2 elements, where first elemnet gives the top of the crypt and second the bottom of the villus
    z_step : float, optional
        Step size in z (in um). Used to convert the z-levels given in dmz to microns used in the DataFrame.
        The default is 2.5.

    Returns
    -------
    None.
    Adds the following columns
    'z_class' : Location of the cell - crypt, ambiguous or villi
    'z_colors' : sets a color to the compartments (orange, darkblue, green) 

    """
    z_class, z_colors = [], []

    for ums in cell_df['z'].values:
        if ums < dmz[0]*z_step:
            z_class.append('crypt')
            z_colors.append('orange')
        elif ums > dmz[1]*z_step:
            z_class.append('villi')
            z_colors.append('green')
        else:
            z_class.append('ambiguous')
            z_colors.append('darkblue')

    cell_df['z_class'] = z_class
    cell_df['z_colors'] = z_colors
    
    cell_df['z_crypt'] = cell_df['z']-dmz[0]*z_step


# Functions to calculate movement for cells
def calculate_cell_movement(cell_df, no_movement_interval = 2.5, track_index = 'track_reindexed',
                            deg = 20, fit_measures = ['distance_z', 'z', 'x', 'y', 'difference_from_origin_z', 'distance_xy', 
                                                      'distance_xyz', 'z_crypt', 'cumulative_distance_xy', 
                                                      'cumulative_distance_xyz','cumulative_distance_z']):
    """
    Get cell movement descriptors.
    
    Wrapper function to call other cell-movement functions.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.
    no_movement_interval : float, optional
        Distance in z to consider for marking that cell has not moved between beginning and ending points. The default is 2.5.
    track : str, optional
        Name to use for track indexing
    deg : int, optional
        Degrees to use for polynomial fit. The default is 20.
    fit_measures : list, optional
        List of column names to include in fitting. By default calculates polynomial fits for all the newly generated informative columns. 
        The default is ['distance_z', 'z', 'x', 'y', 'difference_from_origin_z', 'distance_xy', 'distance_xyz', 'z_crypt', 'cumulative_distance_xy', 'cumulative_distance_xyz','cumulative_distance_z'].

    Returns
    -------
    Updates the working DataFrame with the following columns:
    "distance_{}" for xy, xyz and z
    "origin_{}" and "difference_from_origin_{}" for x,y,z and t, 
        coming from "calculate_movement_origin()" function call.
    "end_{}" for x,y,z and t - final coordinates for each cell

    """
    print('Finding cell origins...')
    calculate_movement_origin(cell_df, track_index = track_index)
    print('Finding cell endpoints...')
    calculate_movement_end(cell_df, track_index = track_index)
    print('Finding origin and ending compartments...')
    find_terminal_locations(cell_df, no_movement_interval = no_movement_interval, track_index = track_index)
    print('Calculating travelled distances...')
    calculate_distances(cell_df, track_index = track_index)
    print('Calculating cumulative changes for distances...')
    calculate_cumulative_distances(cell_df, track_index = track_index)
    calculate_fitted_values(cell_df, deg = deg, fit_measures = fit_measures, track_index = track_index)


def calculate_movement_origin(cell_df, track_index = 'track_reindexed'):
    """
    Calculate movement for each cell, starting from the first timepoint they were identified.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        working DataFrame.

    Returns
    -------
    Updates the existing DataFrame with the following columns:
    'origin_{}' for x, y, z and t - initial position for the cell
    'difference_from_origin_{}' for x, y, z and t - how the new position differs from the origin position

    """
    tracks = set(cell_df[track_index]) #Get a set of all tracks in the dataframe
        
    keys = ['x','y','z','t']
    new_columns = ['origin', 'difference_from_origin']
    for col in new_columns:
        for key in keys:
            cell_df[f'{col}_{key}'] = np.nan
    
    # Find minimum t for every track
    track_min_t = {track: cell_df[cell_df[track_index]==track]['t'].min() 
                   for track in tracks}

    origin_df = pd.DataFrame(index = tracks, columns = keys)
    
    #Find the origin for x,y,z - their values at the first timepoint for each track
    for track in tracks:
        tmp_val = cell_df[(cell_df[track_index]==track) & (cell_df['t']==track_min_t[track])][keys].values
        #Some tracks have multiple coordinates for same timeframe, due to 2 tracks being merged into 1
        origin_df.loc[track] = tmp_val[0] #So take only the first value (doesn't affect single cells)

    for key in keys:
        cell_df[f'origin_{key}'] = origin_df[key].loc[cell_df[track_index]].values
        cell_df[f'difference_from_origin_{key}'] = (cell_df[key] - cell_df['origin_{}'.format(key)]).astype(float).values
        
def calculate_movement_end(cell_df, track_index = 'track_reindexed'):
    """
    Find the x,y,z,t coordinates for the last position for each track.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        working DataFrame.

    Returns
    -------
    Updates the existing DataFrame with the following columns:
    'end_{}' for x, y, z and t - final position for the cell
    
    """
    tracks = set(cell_df[track_index]) #Get a set of all tracks in the dataframe

    keys = ['x','y','z','t']
    
    for key in keys:
        cell_df[f'end_{key}'] = np.nan
    
    # Find maximum t for every track
    track_max_t = {track: cell_df[cell_df[track_index]==track]['t'].max() 
                   for track in tracks}
    
    end_df = pd.DataFrame(index = tracks, columns = keys)
    
    #Find the end for x,y,z - their values at the last timepoint for each track
    for track in tracks:
        tmp_val = cell_df[(cell_df[track_index]==track) & (cell_df['t']==track_max_t[track])][keys].values
        #Some tracks have multiple coordinates for same timeframe, due to 2 tracks being merged into 1
        end_df.loc[track] = tmp_val[0] #So take only the first value (doesn't affect single cells)
    
    for key in keys:
        cell_df[f'end_{key}'] = end_df[key].loc[cell_df[track_index]].values

        
def calculate_distances(cell_df, track_index = 'track_reindexed'):
    """
    Calculate distance travelled between each timepoint in xy, z and xyz.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.

    Returns
    -------
    Updates the working DataFrame with the columns for distance travelled:
    "distance_{}" for xy, xyz and z.

    """
    distance_dict = {}
    
    for track in set(cell_df[track_index]):
        tmp = cell_df[cell_df[track_index]==track].sort_values(by = 't')
    
        for ix, x in enumerate(tmp.index):
            if ix == 0:
                distance_dict[x] = (0,0,0)
            else:
                x1 = tmp['x'].iloc[ix]
                x2 = tmp['x'].iloc[ix-1]
                xd = x1-x2
    
                y1 = tmp['y'].iloc[ix]
                y2 = tmp['y'].iloc[ix-1]
                yd = y1-y2
    
                z1 = tmp['z'].iloc[ix]
                z2 = tmp['z'].iloc[ix-1]
                zd = z1-z2
    
                dist_xy_tmp = xd*xd + yd*yd
                dist_xyz_tmp = dist_xy_tmp + zd*zd
                dist_z_tmp = zd*zd
    
                dist_xy = sqrt(dist_xy_tmp)
                dist_xyz = sqrt(dist_xyz_tmp)
                dist_z = sqrt(dist_z_tmp)
    
                distance_dict[x] = (dist_xy, dist_xyz, dist_z)
    
    distance_xy = [distance_dict[cell][0] for cell in cell_df.index]
    distance_xyz = [distance_dict[cell][1] for cell in cell_df.index]
    distance_z = [distance_dict[cell][2] for cell in cell_df.index]
    
    cell_df['distance_xy'] = distance_xy
    cell_df['distance_xyz'] = distance_xyz
    cell_df['distance_z'] = distance_z
    

def calculate_cumulative_distances(cell_df, track_index = 'track_reindexed'):
    """
    For each track calculate the cumulative distance travelled over timelapse.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.

    Returns
    -------
    Updates DataFrame with cumulative distance measuremnets:
    "cumulative_distance_{}" for xy, xyz and z

    """
    keys = ['distance_xy', 'distance_xyz', 'distance_z']
    tracks = set(cell_df[track_index])
    
    for key in keys:
        cell_df[f'cumulative_{key}'] = np.nan
    
    for track in tracks:
        tmp = cell_df[cell_df[track_index]==track].sort_values(by = 't')
        for key in keys:
            cell_df.loc[tmp.index, f'cumulative_{key}'] = tmp[key].cumsum().values
    
def calculate_fitted_values(cell_df, deg = 20, track_index = 'track_reindexed', 
                            fit_measures = ['distance_z', 'z', 'x', 'y', 'difference_from_origin_z', 'distance_xy', 
                                                               'distance_xyz', 'z_crypt', 'cumulative_distance_xy', 
                                                               'cumulative_distance_xyz','cumulative_distance_z']):
    """
    Calculate fitted values for the selected parameters.
    
    Uses numpy polyfit for fitting to reduce the effect of random movements and make plots nicer.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.
    deg : int, optional
        Degrees to use for polynomial fit. The default is 20.
    fit_measures : list, optional
        List of column names to include in fitting. By default calculates polynomial fits for all the newly generated informative columns. 
        The default is ['distance_z', 'z', 'x', 'y', 'difference_from_origin_z', 'distance_xy', 'distance_xyz', 'z_crypt', 'cumulative_distance_xy', 'cumulative_distance_xyz','cumulative_distance_z'].

    Returns
    -------
    Updates the DataFrame with the fitted values

    """
    tracks = sorted(set(cell_df[track_index]))
    
    for measure in fit_measures:
        cell_df['{}_fit'.format(measure)] = np.nan

    for track in tracks:
        tmp = cell_df[cell_df[track_index]==track]
        t = tmp['t'].values
        for measure in fit_measures:
            z = tmp[measure].values
            z_fit = np.poly1d(np.polyfit(x = t, y = z, deg = deg))(t)
            cell_df.loc[tmp.index, '{}_fit'.format(measure)] = z_fit
        
        print('Calculating fitted values for the following columns:')
        print(*fit_measures, sep = ', ')
        print(f'Tracks: {track}/{len(tracks)-1}')
        clear_output(wait = True)
    
def find_terminal_locations(cell_df, no_movement_interval = 2.5, track_index = 'track_reindexed'):
    """
    Find the starting and ending compartment for each track.

    Parameters
    ----------
    cell_df : pandas.DataFrame
        Working DataFrame.
    no_movement_interval : float, optional
        Distance in z to consider for marking that cell has not moved between beginning and ending points. The default is 2.5.
        
    Returns
    -------
    Updates the DataFrame with 'origin_class' and 'end_class' columns.

    """
    tracks = sorted(set(cell_df[track_index]))

    cell_df['origin_class'] = np.nan
    cell_df['end_class'] = np.nan

    min_dict = {}
    max_dict = {}
    for track in tracks:
        tmp = cell_df[cell_df[track_index]==track]
        min_dict[track] = tmp[tmp['t']==tmp['t'].min()]['z_class'].values[0]
        max_dict[track] = tmp[tmp['t']==tmp['t'].max()]['z_class'].values[0]

    cell_df['origin_class'] = [min_dict[track] for track in cell_df[track_index]]
    cell_df['end_class'] = [max_dict[track] for track in cell_df[track_index]]
    cell_df['Change_z'] = ['Lower' if x < -no_movement_interval else 'Higher' if x > no_movement_interval else 'Same' for x in cell_df['end_z']-cell_df['origin_z']]


def live_dead_import_plot_classify(ld_path, cell_df, cell_radius = 10, s = 5, corrected_dimensions = [512, 512], xy_drift = None, xy_scaling = 1.405, return_ax = False):
    """
    Import live/dead cell coordinates, plot results and classify cells accordingly.

    Parameters
    ----------
    ld_path : str
        Path to the live/dead ROI file.
    cell_df : pandas.DataFrame
        Working DataFrame.
    cell_radius : int, optional
        Radius for each cell to match them with nearby ROIs. The default is 10.
    s : int, optional
        Size for the plotted ROI dots, only visual. The default is 5.
    corrected_dimensions : list, optional
        Image pixel size. The default is [512, 512].
    xy_drift : list, optional
        Drift in pixels for x and y coordinates (x, y), to correct for drift between timelapse and live/dead imaging. The default is None.
    xy_scaling: float, optional
        um/px scaling for the imaging. Should be 1.405 um/px in all images. Needed to correctly calculate xy positions. The default is 1.405.
    return_ax : boolean, optional
        Whether to return the plotted axis. The default is False.

    Returns
    -------
    Updates working DataFrame with live/dead classification column "Live_dead".

    """
    print('Importing coordinates...')
    ld_df = import_live_dead_coordinates(ld_path, corrected_dimensions = corrected_dimensions, xy_drift = xy_drift)
    print('Plotting')
    patches = plot_live_dead_cells(ld_df, cell_df, cell_radius = cell_radius, s = s, return_ax = False)
    print('Classifying...')
    classify_dead_cells_v2(ld_df, cell_df, patches)
    
    
def import_live_dead_coordinates(path, corrected_dimensions = [512, 512], xy_drift = None, xy_scaling = 1.405):
    """
    Import live/dead coordinates from ROI file.
    
    Correct for image size and drift.

    Parameters
    ----------
    path : str
        Path to the live/dead ROI file.
    corrected_dimensions : list, optional
        Image pixel size. The default is [512, 512].
    xy_drift : TYPE, optional
        Drift in pixels for x and y coordinates (x, y), to correct for drift between timelapse and live/dead imaging. The default is None.
    xy_scaling: float, optional
        um/px scaling for the imaging. Should be 1.405 um/px in all images. Needed to correctly calculate xy positions. The default is 1.405.

    Returns
    -------
    None.

    """
    ld_df = pd.DataFrame.from_dict(read_roi.read_roi_zip(path)).T[['x','y']]
    
    ld_df['x'] = [x[-1]*xy_scaling for x in ld_df['x']]
    ld_df['y'] = [(corrected_dimensions[1]-y[-1])*xy_scaling for y in ld_df['y']]
    if xy_drift:
        ld_df['x'] = [xi + xy_drift[0] for xi in ld_df['x']]
        ld_df['y'] = [yi - xy_drift[1] for yi in ld_df['y']]
    ld_df = ld_df.drop_duplicates(subset=['x','y'])
    ld_df['name'] = [[] for x in ld_df.index]
    
    return ld_df
    
def plot_live_dead_cells(ld_df, cell_df, cell_radius = 10, s = 5, figsize = (10,10), return_ax = False):
    """
    Plot the imported coordinates on the cells in the last timelapse position.

    Parameters
    ----------
    ld_df : pandas.DataFrame
        DataFrame containing dead-cell coordinates.
    cell_df : pandas.DataFrame
        Working DataFrame.
    cell_radius : int, optional
        Radius for each cell to match them with nearby ROIs. The default is 10.
    s : int, optional
        Size for the plotted ROI dots, only visual. The default is 5.
    figsize : tuple, optional
        Figure size for plotting. The default is (10,10).
    return_ax : boolean, optional
        Whether to return the plotted axis. The default is False.

    Returns
    -------
    matplotlib.PatchCollection
        ROI patches for cell classification.
    matplotlib axis, optional
        Figure axis

    """
    tmp = cell_df[cell_df['t']==cell_df['end_t']]
    
    # Make a circular patch out of every cell to later find if the patch contains the marked dead ROI spot
    patches = {}
    for cell in tmp.index:
        polygon = mpl.patches.Circle(tmp[['x','y']].loc[cell], radius = cell_radius)
        patches[cell] = polygon
    
    fig, ax = plt.subplots(figsize = figsize)
    p = mpl.collections.PatchCollection(patches.values())
    ax.add_collection(p)
    
    ax.scatter(tmp['x'], tmp['y'], color = 'lightgrey', s = s)
    ax.scatter(ld_df['x'], ld_df['y'], color = 'red', s = cell_radius)

    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', markerfacecolor='blue', 
                                        label='Tracked cells', markersize = 15),
                       mpl.lines.Line2D([0], [0], marker='o', markerfacecolor='r', 
                                        label='Dead cells', markersize = 15)]
    ax.legend(handles=legend_elements)

    
    plt.show()
    if return_ax:
        return patches, ax
    else:
        return patches
    

def classify_dead_cells_v2(ld_df, cell_df, patches):
    """
    Classify cells according to their live/dead annotation.

    Parameters
    ----------
    ld_df : pandas.DataFrame
        DataFrame containing dead-cell coordinates.
    cell_df : pandas.DataFrame
        Working DataFrame.
    patches : matplotlib.PatchCollection
        ROI patches for cell classification.

    Returns
    -------
    Updates working DataFrame with live/dead classification column "Live_dead".

    """
    tmp = cell_df[cell_df['t']==cell_df['end_t']]
    
    #Find cells that have been marked as dead
    for name, patch in patches.items():
        for cell in ld_df.index:
            if patch.contains(ld_df[['x','y']].loc[cell])[0]:
                ld_df['name'].loc[cell].append(name)
    
    #Get track name of the cell.
    #In case of 2 cells overlaping, find which center is closer
    #If nothing overlaps, drop cell.
    ld_df['closest_track'] = np.nan
    for cell in ld_df.index:
        identified_cells = ld_df['name'].loc[cell]
        if len(identified_cells)>1:
            lowest_dist = 10e2
            cell_name = ''
            b = ld_df[['x','y']].loc[cell]
            for tracked_cell in identified_cells:
                a = tmp[['x','y']].loc[tracked_cell]
                dist = ((a[0]-b[0])**2)+((a[1]-b[1])**2)
                if dist < lowest_dist:
                    lowest_dist = dist
                    cell_name = tracked_cell
            ld_df['closest_track'].loc[cell] = tmp['track_reindexed'].loc[cell_name]
        elif len(identified_cells) == 0:
            ld_df['closest_track'].loc[cell] = np.nan
        else:
            ld_df['closest_track'].loc[cell] = tmp['track_reindexed'].loc[identified_cells[0]]

    ld_df = ld_df.dropna(subset=['closest_track'])
    
    cell_df['Live_dead'] = np.where(cell_df['track_reindexed'].isin(ld_df['closest_track']), 'Dead', 'Alive')
    