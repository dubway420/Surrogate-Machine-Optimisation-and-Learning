import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from core_parse import CoreInstance as CoreInstance
from tools import cases_list
from random import seed
from random import randint
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns


def turn_off_graph_decorations(axis):
    axis.xaxis.set_ticks_position('none')
    axis.yaxis.set_ticks_position('none')
    axis.set_xticklabels([])
    axis.set_yticklabels([])


##################################################
# ############### USER INPUTS ####################
##################################################

# This seed is used to generate numbers to select cases
seed(3)

# The number of cases to compare
no_cases = 3

frames_of_interest = [48, 55, 65, 68]
results_of_interest = [1, 2]

# Make sure this points to the folder which contains all of the cases
case_root = 'D:/parmec_results/'
##################################################

# This should point to an intact case
case_intact = 'C:/Users/Huw/PycharmProjects/Results/intact_core'

# Generates a core instance of the intact case
instance = CoreInstance(case_intact)

# Gets the xy coordinates of the fuel channels
channel_coord_list_fuel = instance.get_brick_xyz_positions('xy', channel_type='fuel')

##################################################

# Makes a list of all the cases in the root directory
total_cases = cases_list(case_root)

cases = []
instances = []

cracks_channel_specific = []

# Iterates through all of the cases select
for i in range(no_cases):
    # for each case, generates a number between 0 and the count of cases
    case_no = randint(0, len(total_cases) - 1)
    print(case_no)

    # Gets the base name of the case and appends it to a list
    case = total_cases[case_no].split('/')[-1]
    cases.append(case)

    # Generates an instance of core parse for each case
    path = case_root + case + "/" + case
    inst = CoreInstance(path)

    # Gets the number of cracks surrounding each channel
    cracks_channel_specific.append(inst.channel_specific_cracks()[1])
    instances.append(inst)

# Gets the channel coordinates for each channel. This is common across all instances so can take [0]
channel_coord_list_inter = instances[0].get_brick_xyz_positions('xy', channel_type='inter')

##################################################
# ######### CRACK CONCENTRATION PLOT #############
##################################################

# Creates the figure
fig = plt.figure(figsize=(9, 8))

# The maximum number of cracks that surround any channel. Used for bounding.
max_counts = np.amax(cracks_channel_specific)

# Generates the plot grid for each case
counts_grid = AxesGrid(fig, 111,
                       nrows_ncols=(1, len(cases)),
                       axes_pad=0.12,
                       cbar_mode='single',
                       cbar_location='bottom',
                       cbar_pad=0.2
                       )

# Iterates through each plot setting the parameters
for i, ax in enumerate(counts_grid):
    # sets the title as the case
    column_title = cases[i]
    ax.title.set_text(column_title)

    # Turns off the labels and ticks of each axis
    turn_off_graph_decorations(ax)

    # Generates the plot
    im = ax.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1],
                    marker='o', c=cracks_channel_specific[i], cmap='nipy_spectral', label='inter',
                    s=20)
    im.set_clim(0, max_counts)

# Creates the colorbar
cbar = ax.cax.colorbar(im)
cbar = counts_grid.cbar_axes[0].colorbar(im)

# Saves the figure
filenm = "./Comparing_three_cases/concentration_of_cracks.png"
plt.savefig(filenm)

##################################################
# ################ RESULT PLOT ###################
##################################################

channel_type = 'inter'

# This gets the acceleration time history. Used for labelling.
earthquake_acceleration = (pd.read_csv('time_history.csv').values[:, 2])

# Iterates through the user defined output types of interest, generating a separate figure for each
for result in results_of_interest:
    results = []

    fig = plt.figure(figsize=(12, 12))

    # Iterate through the frames of interest, generating a row for each
    for frame_no, frame in enumerate(frames_of_interest):

        # Gets the earthquake acceleration at the given time-step
        accel = earthquake_acceleration[frame]

        # Iterate through instances (cases) generating an array of length 321 i.e. the result for each channel
        for instance_no, instance in enumerate(instances):
            results.append(instance.get_result_at_time(time_index=frame, result_columns=str(result)))

    # Multiplies the results by 1000 to convert from meters to millimeters
    results = np.multiply(results, 1000)

    # Gets the min and max value result across all cases and frames.
    # Used for setting a common range across all plots and the colorbar
    min_result = np.amin(results)
    max_result = np.amax(results)

    # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(len(frames_of_interest), len(cases)),
                    axes_pad=0.12,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.2
                    )

    frame_no = 0

    # Iterates through all plots in the grid, assigning the results to the plot
    for i, ax in enumerate(grid):

        # If the iterator is in the first row, assigns the title as the case
        if i < len(cases):
            column_title = cases[i]
            ax.title.set_text(column_title)

        # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
        # acceleration
        if (i % 3) == 0:
            frame = frames_of_interest[frame_no]
            row_title = "Frame: " + str(frame) + " - Acc: " + str(round(earthquake_acceleration[frame], 2))
            ax.set_ylabel(row_title)
            frame_no += 1

        # Turns off the other graph markings
        turn_off_graph_decorations(ax)

        # Plots the results for the case/frame
        im = ax.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1],
                        marker='o', c=results[i], cmap='nipy_spectral', label='inter',
                        s=20)
        im.set_clim(min_result, max_result)

    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)

    # Saves the file
    filenm = "./Comparing_three_cases/results" + str(result) + ".png"
    plt.savefig(filenm)

    ########################################################################################
    # ######################## result vs concentration #####################################
    ########################################################################################

    fig = plt.figure(figsize=(24, 12))

    # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
    grid_compare = AxesGrid(fig, 111,
                            nrows_ncols=(len(frames_of_interest), len(cases)),
                            axes_pad=0.12,
                            cbar_pad=0.2
                            )

    frame_no = 0

    pearson_coeffs = [[], [], []]
    # Iterates through all plots in the grid, assigning the results to the plot
    for i, ax in enumerate(grid_compare):

        # If the iterator is in the first row, assigns the title as the case
        if i < len(cases):
            column_title = cases[i]
            ax.title.set_text(column_title)

        # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
        # acceleration
        if (i % 3) == 0:
            frame = frames_of_interest[frame_no]
            row_title = "Frame: " + str(frame) + " - Acc: " + str(round(earthquake_acceleration[frame], 2))
            ax.set_ylabel(row_title)
            frame_no += 1

        # Plots the results for the case/frame

        x = cracks_channel_specific[(i % 3)]
        y = results[i]
        pearson_coeffs[(i % 3)].append(pearsonr(x, y)[0])
        im = ax.scatter(x, y, marker='o')
        ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 2))(np.unique(x)))

    # Saves the file
    filenm = "./Comparing_three_cases/correlation" + str(result) + ".png"
    plt.savefig(filenm)
    print(np.mean(pearson_coeffs, axis=0))
