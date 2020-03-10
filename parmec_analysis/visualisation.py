import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import seaborn as sns
import parmec_analysis.core_parse as core
from parmec_analysis.utils import cases_list, ReLu, ReLu_all
from random import seed
from random import randint
import pandas as pd
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import pickle

inclusive_layers = [100, 6, 4, 2]

layer_colors = ['royalblue', 'lime', 'yellow', 'red']


def turn_off_graph_decorations(axis):
    axis.xaxis.set_ticks_position('none')
    axis.yaxis.set_ticks_position('none')
    axis.set_xticklabels([])
    axis.set_yticklabels([])


def features_and_labels_single_frame(path_string, time=50, result="1", x_type='positions'):
    """ Gets the features and labels from the folder of results"""

    cases_total = cases_list(path_string)

    X, Y = [], []

    for case in cases_total:

        try:
            instance = core.Parse(case)

            X.append(instance.linear_crack_array_1d(array_type=x_type))
            Y.append(instance.get_result_at_time(time, result_columns=str(result)))

        except FileNotFoundError:
            print("case error: ", case.split('/')[-1])

    return X, Y


##################################################
# ############### USER INPUTS ####################
##################################################

# This seed is used to generate numbers to select cases
seed(2)
#
frames_of_interest = [48, 55, 65, 68]
results_of_interest = [1, 2]
#
# Make sure this points to the folder which contains all of the cases
case_root = 'D:/parmec_results/'

# This should point to an intact case
case_intact = 'C:/Users/Huw/PycharmProjects/parmec_agr_ml_surrogate/intact_core_rb'

# Generates a core instance of the intact case
instance_intact = core.Parse(case_intact)

# Number of levels
inter_levels = instance_intact.inter_levels

inter_channels = instance_intact.interstitial_channels

# # Gets the channel coordinates for each channel. This is common across all instances so can take [0]
channel_coord_list_inter = instance_intact.get_brick_xyz_positions('xy', channel_type='inter')

# Gets the xy coordinates of the fuel channels
channel_coord_list_fuel = instance_intact.get_brick_xyz_positions('xy', channel_type='fuel')

# Makes a list of all the cases in the root directory
total_cases = cases_list(case_root)

for i, case in enumerate(total_cases):
    print(i, case)

cases = []
instances = []

cracks_channel_specific = []
cracks_level_specific = []
cracks_region_specific = []

case_chosen = [2, 0, 1]

# The number of cases to compare
no_cases = len(case_chosen)

# Iterates through all of the cases select
for i in range(no_cases):
    # for each case, generates a number between 0 and the count of cases
    # case_no = randint(0, len(total_cases) - 1)

    case_no = case_chosen[i]

    # Gets the base name of the case and appends it to a list
    case = total_cases[case_no].split('/')[-1]
    cases.append(case)

    # Generates an instance of core parse for each case
    path = case_root + case + "/" + case
    inst = core.Parse(path)

    cracks_per_layer = []
    for size in inclusive_layers:
        cracks_per_layer.append(inst.get_cracks_per_layer(channel="161", array_type="positions",
                                                          size=size, channel_type="interstitial")[-1])

    cracks_region_specific.append(cracks_per_layer)

    # Gets the number of cracks surrounding each channel
    cracks_channel_specific.append(inst.channel_specific_cracks()[1])

    # Gets the cracks per level

    # TODO MAKE THIS INTO A FUNCTION OF ITS OWN
    # cracks_per_level = [inst.get_cracks_per_level(array_type="positions", quiet=True)]
    # for size in inclusive_layers:
    #     cracks_per_level.append(inst.get_cracks_per_level(channel="161", array_type="positions", quiet=True,
    #                                                       size=size, channel_type="interstitial"))

    # cracks_level_specific.append(cracks_per_level)
    #
    instances.append(inst)

cracks_region_specific_max = np.amax(np.array(cracks_region_specific))
cracks_region_specific_norm = (cracks_region_specific / np.amax(cracks_region_specific)) * 0.5


layer_no_of_channels = np.zeros(instance_intact.interstitial_channels)

for i in range(instance_intact.interstitial_channels):
    channel_no = i + 1
    layer_no_of_channels[i] = instance_intact.layers_from_centre(channel_no, channel_type='inter')

##################################################
# ############# 3D CRACK PLOT ####################
##################################################

# brick_coord_list_fuel = instance.get_brick_xyz_positions('xyz', channel_type="fuel", channels_only=0)
# X = instances[0].linear_crack_array_1d(array_type="pos")
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# print(np.unique(X))
#
# ax.title.set_text(cases[0])
# ax.scatter(brick_coord_list_fuel[0], brick_coord_list_fuel[1], brick_coord_list_fuel[2], c=X, cmap='gnuplot')
# plt.legend()
# ax.view_init(45, 45)
# plt.show()

##################################################
# ######### CRACK CONCENTRATION PLOT #############
##################################################

# Creates the figure
fig = plt.figure(figsize=(10, 9))

# The maximum number of cracks that surround any channel. Used for bounding.
max_counts = np.amax(cracks_channel_specific)

print("Max counts: ", max_counts)

# Generates the plot grid for each case
counts_grid = AxesGrid(fig, 111,
                       nrows_ncols=(1, len(cases)),
                       axes_pad=0.2,
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
                    marker='o', c=cracks_channel_specific[i], cmap='OrRd', label='inter',
                    s=100)
    im.set_clim(0, max_counts)

# Creates the colorbar
cbar = ax.cax.colorbar(im)
cbar = counts_grid.cbar_axes[0].colorbar(im)

# Saves the figure
filenm = "./Comparing_three_cases/concentration_of_cracks.png"
# plt.savefig(filenm)
plt.show()

##################################################
# ############ CRACK AREA PLOT ###################
##################################################


# fig = plt.figure(figsize=(27, 9))
# axes = [fig.add_subplot(1, no_cases, (c + 1)) for c in range(0, no_cases)]
#
# y_labels = False
#
# for case, ax, case_region_values in zip(cases, axes, cracks_region_specific):
#
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     ax.title.set_text(case)
#
#     square = plt.Rectangle((0, 0), 1, 1, color='royalblue')
#     ax.add_artist(square)
#
#     for region_cracks, colour in zip(case_region_values, layer_colors):
#         region_cracks_norm = (region_cracks / cracks_region_specific_max) * 0.4
#         circle = plt.Circle((0.5, 0.5), region_cracks_norm, color=colour)
#         ax.add_artist(circle)
#         ax.annotate(str(region_cracks), xy=(region_cracks_norm + 0.5, 0.49), size=25)
#
# fig.show()

##################################################
# ######### CRACKS PER LEVEL PLOT ################
##################################################
#
# ncols = 3
#
# # create the plots
# fig = plt.figure(figsize=(25, 9))
# axes = [fig.add_subplot(1, ncols, (c + 1)) for c in range(0, ncols)]
#
# y_labels = False
#
# # add some data
# for case, ax, case_levels in zip(cases, axes, cracks_region_specific):

#     if y_labels: ax.set_yticks([])
#     else: ax.set_ylabel('Core Level')
#     y_labels = True
#
#     for i, region in enumerate(case_levels):
#         ax.barh(np.arange(len(region))+1, region, color=layer_colors[i])
#
#     ax.set_title(case)
#
# plt.show()

##################################################
# ############ EARTHQUAKE PLOT ###################
##################################################

# # This gets the acceleration time history. Used for labelling.
# earthquake_acceleration = (pd.read_csv('time_history.csv').values[:150, 2])
#
# for frame in frames_of_interest:
#     plt.plot([frame, frame], [np.amin(earthquake_acceleration), np.amax(earthquake_acceleration)], color='silver')
#
# plt.plot(earthquake_acceleration)
# plt.xlabel("Time Frame (n)")
# plt.ylabel("Ground Acceleration (m/$s^2$)")
#
#
#
# plt.show()

##################################################
# ################ RESULT PLOT ###################
##################################################

# channel_type = 'inter'
# # Iterates through the user defined output types of interest, generating a separate figure for each
# for result in results_of_interest:
#     results = []
#
#     fig = plt.figure(figsize=(12, 12))
#
#     # Iterate through the frames of interest, generating a row for each
#     for frame_no, frame in enumerate(frames_of_interest):
#
#         # Gets the earthquake acceleration at the given time-step
#         accel = earthquake_acceleration[frame]
#
#         # Iterate through instances (cases) generating an array of length 321 i.e. the result for each channel
#         for instance_no, instance in enumerate(instances):
#             results.append(instance.get_result_at_time(time_index=frame, result_columns=str(result), result_type="sum"))
#
#     # Multiplies the results by 1000 to convert from meters to millimeters
#     results = np.multiply(results, 1000)
#
#     # Gets the min and max value result across all cases and frames.
#     # Used for setting a common range across all plots and the colorbar
#     min_result = np.amin(results)
#     max_result = np.amax(results)
#
#     # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
#     grid = AxesGrid(fig, 111,
#                     nrows_ncols=(len(frames_of_interest), len(cases)),
#                     axes_pad=0.12,
#                     cbar_mode='single',
#                     cbar_location='right',
#                     cbar_pad=0.2
#                     )
#
#     frame_no = 0
#
#     # Iterates through all plots in the grid, assigning the results to the plot
#     for i, ax in enumerate(grid):
#
#         # If the iterator is in the first row, assigns the title as the case
#         if i < len(cases):
#             column_title = cases[i]
#             ax.title.set_text(column_title)
#
#         # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
#         # acceleration
#         if (i % 3) == 0:
#             frame = frames_of_interest[frame_no]
#             row_title = "Frame: " + str(frame) + " - Acc: " + str(round(earthquake_acceleration[frame], 2))
#             ax.set_ylabel(row_title)
#             frame_no += 1
#
#         # Turns off the other graph markings
#         turn_off_graph_decorations(ax)
#
#         # Plots the results for the case/frame
#         im = ax.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1],
#                         marker='o', c=results[i], cmap='seismic', label='inter',
#                         s=50)
#         im.set_clim(min_result, max_result)
#
#     cbar = ax.cax.colorbar(im)
#     cbar = grid.cbar_axes[0].colorbar(im)
#
#     # Saves the file
#     filenm = "./Comparing_three_cases/results" + str(result) + ".png"
#     plt.savefig(filenm)
# plt.show()
#
#     ########################################################################################
#     # ############################### Regional Results #####################################
#     ########################################################################################


# channel_type = 'inter'
# # Iterates through the user defined output types of interest, generating a separate figure for each
# for result in results_of_interest:
#     results = []
#
#     fig = plt.figure(figsize=(12, 12))
#
#     # Iterate through the frames of interest, generating a row for each
#     for frame_no, frame in enumerate(frames_of_interest):
#
#         # Gets the earthquake acceleration at the given time-step
#         accel = earthquake_acceleration[frame]
#
#         # Iterate through instances (cases) generating an array of length 321 i.e. the result for each channel
#         for instance_no, instance in enumerate(instances):
#             instance_result = instance.get_result_at_time(time_index=frame, result_columns=str(result),
#                                                           result_type="relu")
#
#             graph_results = np.zeros(len(inclusive_layers))
#
#             for i, channel_result in enumerate(instance_result):
#                 channel_layer = layer_no_of_channels[i]
#
#                 for j, region in enumerate(inclusive_layers):
#                     if channel_layer < region: graph_results[j] += channel_result
#
#             results.append(graph_results)
#
#     results_max = np.amax(np.array(results))
#
#     # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
#     grid = AxesGrid(fig, 111,
#                     nrows_ncols=(len(frames_of_interest), len(cases)),
#                     # axes_pad=0.12,
#                     # cbar_mode='single',
#                     # cbar_location='right',
#                     # cbar_pad=0.2
#                     )
#
#     frame_no = 0
#
#     # Iterates through all plots in the grid, assigning the results to the plot
#     for i, ax in enumerate(grid):
#
#         # If the iterator is in the first row, assigns the title as the case
#         if i < len(cases):
#             column_title = cases[i]
#             ax.title.set_text(column_title)
#
#         # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
#         # acceleration
#         if (i % 3) == 0:
#             frame = frames_of_interest[frame_no]
#             row_title = "Frame: " + str(frame) + " - Acc: " + str(round(earthquake_acceleration[frame], 2))
#             ax.set_ylabel(row_title)
#             frame_no += 1
#
#         # Turns off the other graph markings
#         turn_off_graph_decorations(ax)
#
#         result_i = results[i]
#         for region_result, colour in zip(result_i, layer_colors):
#             region_results_norm = (region_result / results_max) * 0.4
#             circle = plt.Circle((0.5, 0.5), region_results_norm, color=colour)
#             ax.add_artist(circle)
#             # ax.annotate(str(round(region_result, 2)), xy=(region_results_norm + 0.5, 0.49), size=25)
#
#     filenm = "./Comparing_three_cases/region_results" + str(result) + ".png"
#     plt.savefig(filenm)

#     ###############################################################################
# #   ############################## RESULTS PER LEVEL PLOT #########################
#     ###############################################################################

channel_type = 'inter'
# Iterates through the user defined output types of interest, generating a separate figure for each
# for result in results_of_interest:
#     results = []
#
#     fig = plt.figure(figsize=(12, 3))
#
#     # Iterate through the frames of interest, generating a row for each
#     for frame_no, frame in enumerate(frames_of_interest):
#
#         # Gets the earthquake acceleration at the given time-step
#         accel = earthquake_acceleration[frame]
#
#         # Iterate through instances (cases) generating an array of length 321 i.e. the result for each channel
#         for instance_no, instance in enumerate(instances):
#
#             # 2D array - channels : levels
#             instance_result = instance.get_result_at_time(time_index=frame, result_columns=str(result),
#                                                           result_type="all")
#
#             # Generate a 2D np array - no_regions : no_levels
#             graph_results = np.zeros([len(inclusive_layers), inter_levels])
#
#             # cycle through channel results
#             for channel_layer, channel_result in zip(layer_no_of_channels, instance_result):
#
#                 # cycle through region boundaries i.e. the layer the bounds each region
#                 for i, region in enumerate(inclusive_layers):
#
#                     # cycle through core interstitial levels
#                     for j in range(inter_levels):
#                         level_result = channel_result[j]
#
#                         if channel_layer < region: graph_results[i, j] += level_result
#
#             results.append(graph_results)
#
#     # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
#     fig1, axs = plt.subplots(len(frames_of_interest), no_cases)
#
#     frame_no = 0
#
#     # Iterates through all plots in the grid, assigning the results to the plot
#     for i, ax in enumerate(axs.flat):
#
#         # If the iterator is in the first row, assigns the title as the case
#         if i < len(cases):
#             column_title = cases[i]
#             ax.title.set_text(column_title)
#
#         # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
#         # acceleration
#         if (i % 3) == 0:
#             frame = frames_of_interest[frame_no]
#             row_title = "Frame: " + str(frame)
#             ax.set_ylabel(row_title)
#             frame_no += 1
#
#         # Turns off the other graph markings
#         turn_off_graph_decorations(ax)
#
#         result_i = results[i]
#         for region_result, colour in zip(result_i, layer_colors):
#             ax.barh(np.arange(len(region_result)) + 1, region_result, color=colour)
#
#     filenm = "./Comparing_three_cases/level_results" + str(result) + ".png"
#     plt.savefig(filenm)
# plt.show()

#     ########################################################################################
#     # ######################## result vs concentration #####################################
#     ########################################################################################
#
# fig = plt.figure(figsize=(24, 12))
#
# # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
# grid_compare = AxesGrid(fig, 111,
#                         nrows_ncols=(len(frames_of_interest), len(cases)),
#                         axes_pad=0.12,
#                         cbar_pad=0.2
#                         )
#
# frame_no = 0
#
# pearson_coeffs = [[], [], []]
# # Iterates through all plots in the grid, assigning the results to the plot
# for i, ax in enumerate(grid_compare):
#
#     # If the iterator is in the first row, assigns the title as the case
#     if i < len(cases):
#         column_title = cases[i]
#         ax.title.set_text(column_title)
#
#     # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
#     # acceleration
#     if (i % 3) == 0:
#         frame = frames_of_interest[frame_no]
#         row_title = "Frame: " + str(frame) + " - Acc: " + str(round(earthquake_acceleration[frame], 2))
#         ax.set_ylabel(row_title)
#         frame_no += 1
#
#     # Plots the results for the case/frame
#
#     x = cracks_channel_specific[(i % 3)]
#     y = results[i]
#     pearson_coeffs[(i % 3)].append(pearsonr(x, y)[0])
#     im = ax.scatter(x, y, marker='o')
#     ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 2))(np.unique(x)))
#
# # Saves the file
# filenm = "./Comparing_three_cases/correlation" + str(result) + ".png"
# # plt.show()
# plt.savefig(filenm)
# print(np.mean(pearson_coeffs, axis=0))


#     ########################################################################################
#     # ######################## ALL CASE 2D TOP DOWN COMPOSITE ##############################
#     ########################################################################################


# labels = []
#
# for result in results_of_interest:
#
#     result_labels = []
#     for frame in frames_of_interest:
#
#         features, frame_labels = features_and_labels_single_frame(case_root, result=str(result), time=frame)
#         result_labels.append(frame_labels)
#
#     labels.append(result_labels)

# with open('full_result.pkl', 'wb') as f:
#     pickle.dump([features, labels], f)

# with open('full_result.pkl', 'rb') as f:
#     features, labels = pickle.load(f)
#
# fig = plt.figure(figsize=(24, 12))
#
# labels_composite = np.sum(np.array(labels), axis=3)
#
# sizes = [40, 15]
#
# for labels1, size in zip(labels_composite, sizes):
#
#     # sns.set(color_codes=True)
#     #
#     # Y_central_top = labels[:, 140: 143]
#     # Y_central_mid = labels[:, 159: 162]
#     # Y_central_bot = labels[:, 178: 181]
#     #
#     # Y_central_channels = np.concatenate((Y_central_bot, Y_central_mid, Y_central_top))
#
#     colours = ['c', 'm', 'y', 'orangered']
#
#     for frame, frame_label, colour in zip(frames_of_interest, labels1, colours):
#
#         lab = "Frame: " + str(frame)
#         label = ReLu_all(frame_label)
#         sns.distplot(label, label=lab, color=colour)
#
#     plt.xlabel("Displacement Value (mm)")
#     plt.legend(prop={'size': size})
#     plt.show()

#
# # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
#
# grid = AxesGrid(fig, 111,
#                 nrows_ncols=(len(frames_of_interest), 1),
#                 axes_pad=0.12,
#                 cbar_mode='single',
#                 cbar_location='right',
#                 cbar_pad=0.2
#                 )
#
# for result_labels in labels_composite:
#
#     min_result = np.amin(result_labels)
#     max_result = np.amax(result_labels)
#
#     for composite_result_frame, ax in zip(result_labels, grid):
#         im = ax.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1], marker='o', c=composite_result_frame,
#                         cmap='seismic', label='inter', s=50)
#
#         im.set_clim(min_result, max_result)
#
#         # Creates the colorbar
#         cbar = ax.cax.colorbar(im)
#         cbar = grid.cbar_axes[0].colorbar(im)
#
#     fig.show()

# composite_result = np.sum(np.array(labels_1_48), axis=0)
#
# plt.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1], marker='o', c=composite_result, cmap='seismic',
#             label='inter', s=50)
#
# plt.show()

# full_results = np.zeros([len(results_of_interest), len(frames_of_interest), len(cases_list(case_root)), inter_channels])
#
# for result in results_of_interest:
#
#     for i, frame in enumerate(frames_of_interest):
#         features, full_results[result-1, i] = features_and_labels_single_frame(case_root, result=str(result), time=frame)
#


# with open('full_result.pkl', 'rb') as f:
#     features, full_results = pickle.load(f)

#     ########################################################################################
#     # ######################## ALL CASE COMPOSITE ###########################################
#     ########################################################################################


#     ########################################################################################
#     # ######################## ALL CASE HISTOGRAM ###########################################
#     ########################################################################################


# with open('features_labels.pkl', 'rb') as f:
#     features, labels1 = pickle.load(f)


# labels = np.array(labels)

# Y_central = labels[:, 160]
#
# sns.set(color_codes=True)
#
# Y_central_top = labels[:, 140: 143]
# Y_central_mid = labels[:, 159: 162]
# Y_central_bot = labels[:, 178: 181]
#
# Y_central_channels = np.concatenate((Y_central_bot, Y_central_mid, Y_central_top))
#
# sns.distplot(Y_central.flatten(), label="Central Channel Only")
# sns.distplot(Y_central_channels.flatten(), label="9 Central Channels")
#
# # plt.ylabel("Frequency")
# plt.xlabel("Displacement value (mm)")
# plt.legend()
# plt.show()

########################################################################################
# ######################### VARIABLE CRACK FRACTION ####################################
########################################################################################

# case_folders = ["D:/variable_crack_percentage/seed_5004",
#                 "D:/variable_crack_percentage/seed_5008"
#                 "D:/variable_crack_percentage/seed_5015"]
#
# no_cases = len(case_folders)
#
# # Creates the figure
# fig = plt.figure(figsize=(10, 9))
#
# crack_fractions = [5, 10, 20]
#
# no_fractions = [crack_fractions]
#
# # Generates the plot grid for each case
# counts_grid = AxesGrid(fig, 111,
#                        nrows_ncols=(no_fractions, no_cases),
#                        axes_pad=0.2,
#                        cbar_mode='single',
#                        cbar_location='bottom',
#                        cbar_pad=0.2
#                        )
#
# for fraction in crack_fractions:
#
#     fraction_percentage = "P" + str(fraction)
#     print(fraction_percentage)