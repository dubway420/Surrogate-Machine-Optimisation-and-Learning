import matplotlib as mpl

# Agg backend will render without X server on a compute node in batch
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import seaborn as sns
import parmec_analysis.reactor_case as core
from parmec_analysis.utils import convert_case_to_channel_result, is_in, plot_names_title
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


def model_comparison(model_names, training_loss, validation_loss, errors_training, errors_validation):
    x = np.arange(len(model_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar((x - width / 2), training_loss, width, yerr=errors_training, capsize=5, label='Training')
    rects2 = ax.bar((x + width / 2), validation_loss, width, yerr=errors_validation, capsize=5, label='Validation')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Squared Error')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    fig.tight_layout()

    plt.savefig("plots/" + "comparing_models.png")


class CoreView:

    def __init__(self, model, dataset, features, labels, iteration, plot_no_cases=3, cases_seed=4235, convert_to="sum"):
        # Items used for visualisation
        self.instance_intact = core.Parse('intact_core_rb')
        self.channel_coord_list_inter = self.instance_intact.get_brick_xyz_positions('xy', channel_type='inter')

        # Generated at start of training process
        self.iteration = iteration

        # Objects used by dataset
        self.model = model
        self.dataset = dataset
        self.features = features
        self.labels = labels

        # Model callbacks
        self.epochs_with_data = []
        self.epoch_predictions = []

        # Generate the numbers of the cases to be plotted
        seed(cases_seed)

        self.cases_to_plot_train = [randint(0, len(labels.training_set()) - 1) for _ in range(plot_no_cases)]
        self.cases_to_plot_val = [randint(0, len(labels.validation_set()) - 1) for _ in range(plot_no_cases)]

        # get the id of each chosen case
        self.cases_to_plot_id = [[self.dataset.training_instances()[i].get_id() for i in self.cases_to_plot_train]]
        self.cases_to_plot_id.append([self.dataset.validation_instances()[i].get_id() for i in self.cases_to_plot_val])

        self.plot_no_cases = plot_no_cases

        # The features of the cases to plot
        self.plot_features_train = np.array([features.training_set()[i] for i in self.cases_to_plot_train])
        self.plot_features_val = np.array([features.validation_set()[i] for i in self.cases_to_plot_val])

        # Values used to generate plots. These will be updated every time update data is accessed
        self.number_of_plots = plot_no_cases

        self.convert_to = convert_to

        if not is_in(labels.type, 'all'):
            # if per channel results are being returned e.g. sum, avg, max etc.
            self.plot_results = [[labels.training_set()[i] for i in self.cases_to_plot_train]]
            self.plot_results.append([labels.validation_set()[i] for i in self.cases_to_plot_val])
        else:
            # if it's an 'all' result then it can't be easily visualised in a 2D setting. Convert to a channel-wise
            # variable instead
            self.plot_results = [[convert_case_to_channel_result(labels.training_set()[i], convert_to,
                                                                 labels.number_channels,
                                                                 labels.number_levels)
                                  for i in self.cases_to_plot_train]]

            self.plot_results.append([convert_case_to_channel_result(labels.validation_set()[i], convert_to,
                                                                     labels.number_channels,
                                                                     labels.number_levels)
                                      for i in self.cases_to_plot_val])

        self.plot_results_min = [0, 0]
        self.plot_results_max = [0, 0]

        # update the min and max for the training and validation results
        for i in range(2):
            abs_max = np.amax(np.absolute(self.plot_results[i]))
            self.plot_results_min[i] = np.amin(abs_max * -1)
            self.plot_results_max[i] = np.amax(abs_max)

        self.subtitle = ""
        self.main_title = ""
        self.file_name = ""

    def update_data(self, epoch, model, plot=True):
        """ Update the predictions"""

        self.main_title = model.name + " (Instances: " + str(len(self.features.training_set())) + "/" + \
                          str(len(self.features.validation_set())) + " - Iteration: " + \
                          str(self.iteration) + ")"

        # # plot titles and filename
        sub_title, file_name = plot_names_title(model, self.dataset, self.features, self.labels, self.iteration)

        self.subtitle = sub_title
        self.file_name = "CoreView_" + file_name

        self.epochs_with_data.append(epoch)

        # update the model
        self.model = model

        epoch_results_train = model.predict(self.plot_features_train)
        epoch_results_val = model.predict(self.plot_features_val)

        for case_train, case_val in zip(epoch_results_train, epoch_results_val):

            if not is_in(self.labels.type, 'all'):
                self.plot_results[0].append(case_train)
                self.plot_results[1].append(case_val)
            else:
                self.plot_results[0].append(convert_case_to_channel_result(case_train, self.convert_to,
                                                                           self.labels.number_channels,
                                                                           self.labels.number_levels))
                self.plot_results[1].append(convert_case_to_channel_result(case_val, self.convert_to,
                                                                           self.labels.number_channels,
                                                                           self.labels.number_levels))

        # update the min and max for the training and validation results
        for i in range(2):
            abs_max = np.amax(np.absolute(self.plot_results[i]))
            self.plot_results_min[i] = np.amin(abs_max * -1)
            self.plot_results_max[i] = np.amax(abs_max)

        if plot:
            self.plot()

    def plot(self):
        fig = plt.figure(figsize=(18, 18))

        # Generates the plot grid for each case
        counts_grid = AxesGrid(fig, 111,
                               # There's a row for each epoch data plus one for the ground truth labels
                               # There's a column for each case to be plotted
                               nrows_ncols=(len(self.epochs_with_data) + 1, self.plot_no_cases),
                               axes_pad=0.3,
                               cbar_mode='single',
                               cbar_location='bottom',
                               cbar_pad=0.2
                               )

        train_val = ("Training", "Validation")

        # Repeat the plotting for training and validation
        for i, stage in enumerate(train_val):

            # Iterates through all plots in the grid, assigning the results to the plot
            for j, ax in enumerate(counts_grid):

                # If the iterator is in the first row, assigns the case id as title
                if j < self.plot_no_cases:
                    column_title = self.cases_to_plot_id[i][j]
                    ax.title.set_text(column_title)

                # If the iterator is in the first column, assigns a row title which includes
                # the frame number and earthquake acceleration
                if (j % self.plot_no_cases) == 0:
                    row_number = int(j / self.plot_no_cases)

                    if row_number == 0:
                        row_title = "Ground Truth"
                    else:
                        row_title = "Epoch: " + str(self.epochs_with_data[(row_number - 1)])
                    ax.set_ylabel(row_title)

                # Turns off the other graph markings
                turn_off_graph_decorations(ax)

                # Plots the results for the case/frame
                im = ax.scatter(self.channel_coord_list_inter[0], self.channel_coord_list_inter[1],
                                marker='o', c=self.plot_results[i][j], cmap='seismic', label='inter',
                                s=30)
                im.set_clim(self.plot_results_min[i], self.plot_results_max[i])

            # cbar = ax.cax.colorbar(im)
            counts_grid.cbar_axes[0].colorbar(im)

            plt.suptitle(self.main_title)
            plt.title(self.subtitle)

            # Saves the figure

            file_name = "plots/" + stage + "_" + self.file_name
            plt.savefig(file_name)
            # plt.show()


class TrainingHistoryRealTime:

    def __init__(self, dataset, features, labels, iteration, loss_function, plot_from=10):
        self.main_title = ""

        self.dataset = dataset
        self.features = features
        self.labels = labels
        self.iteration = iteration
        self.model = None

        self.subtitle = ""
        self.file_name = ""

        self.loss_function = loss_function
        self.losses = []
        self.val_losses = []
        self.plot_from = plot_from

    def update_data(self, logs, model, plot=True):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.model = model

        if plot and len(self.losses) >= self.plot_from:
            self.plotting()

    def plotting(self):
        self.main_title = self.model.name + " (Instances: " + str(len(self.features.training_set())) + "/" + \
                          str(len(self.features.validation_set())) + " - Iteration: " + \
                          str(self.iteration) + ")"

        subtitle, file_name = plot_names_title(self.model, self.dataset, self.features, self.labels, self.iteration)

        self.subtitle = subtitle
        self.file_name = "plots/RThistory_" + file_name

        # generate plot data

        training_losses_plot = self.losses[self.plot_from:]
        validation_losses_plot = self.val_losses[self.plot_from:]

        epochs_range = np.arange(len(training_losses_plot)) + self.plot_from

        last_result_line_training = [training_losses_plot[-1] for _ in training_losses_plot]
        last_result_line_validation = [validation_losses_plot[-1] for _ in validation_losses_plot]

        # create training plots

        plt.plot(epochs_range, training_losses_plot, label='Training')

        plt.plot(epochs_range, last_result_line_training, 'k:')

        # create validation plots

        plt.plot(epochs_range, validation_losses_plot, label='Validation')

        plt.plot(epochs_range, last_result_line_validation, 'k:')

        # Titles and annotation

        plt.suptitle(self.main_title, fontsize=16, y=1.005)
        plt.title(self.subtitle, fontsize=10, pad=5)

        plt.xlabel("Epoch")
        plt.ylabel(self.loss_function.__name__)
        plt.legend(loc='upper right')

        plt.savefig(self.file_name, bbox_inches="tight", pad_inches=0.25)
        plt.close()

        ##################################################

# ############### USER INPUTS ####################
##################################################
#
# # This seed is used to generate numbers to select cases
# seed(2)
# #
# frames_of_interest = [48, 55, 65, 68]
# results_of_interest = [1, 2]
# #
# # Make sure this points to the folder which contains all of the cases
# case_root = '/media/huw/Disk1/parmec_results/'
#
# # This should point to an intact case
# # case_intact = 'intact_core_rb'
# #
# # # Generates a core instance of the intact case
# # instance_intact = core.Parse(case_intact)
#
# # Number of levels
# inter_levels = instance_intact.inter_levels
#
# inter_channels = instance_intact.interstitial_channels
#
# # # Gets the channel coordinates for each channel. This is common across all instances so can take [0]
# channel_coord_list_inter = instance_intact.get_brick_xyz_positions('xy', channel_type='inter')
#
# # Gets the xy coordinates of the fuel channels
# channel_coord_list_fuel = instance_intact.get_brick_xyz_positions('xy', channel_type='fuel')
#
# # Makes a list of all the cases in the root directory
# total_cases = cases_list(case_root)
#
# cases = []
# instances = []
#
# cracks_channel_specific = []
# cracks_level_specific = []
# cracks_region_specific = []
#
# case_chosen = [2, 0, 1]
#
# # # This gets the acceleration time history. Used for labelling.
# earthquake_acceleration = (pd.read_csv('time_history.csv').values[:150, 2])
#
# # The number of cases to compare
# no_cases = len(case_chosen)
#
# # Iterates through all of the cases select
# for i in range(no_cases):
#     # for each case, generates a number between 0 and the count of cases
#     case_no = randint(0, len(total_cases) - 1)
#
#     # Gets the base name of the case and appends it to a list
#     case = total_cases[case_no].split('/')[-1]
#     cases.append(case)
#
#     # Generates an instance of core parse for each case
#     path = case_root + case + "/" + case
#     inst = core.Parse(path)
#     instances.append(inst)
#
#     #     cracks_per_layer = []
#     #     for size in inclusive_layers:
#     #         cracks_per_layer.append(inst.get_cracks_per_layer(channel="161", array_type="positions",
#     #                                                           size=size, channel_type="interstitial")[-1])
#     #
#     #     cracks_region_specific.append(cracks_per_layer)
#     #
#     # Gets the number of cracks surrounding each channel
#     cracks_channel_specific.append(inst.channel_specific_cracks()[1])
#
# # Gets the cracks per level
#
# # TODO MAKE THIS INTO A FUNCTION OF ITS OWN
# # cracks_per_level = [inst.get_cracks_per_level(array_type="positions", quiet=True)]
# # for size in inclusive_layers:
# #     cracks_per_level.append(inst.get_cracks_per_level(channel="161", array_type="positions", quiet=True,
# #                                                       size=size, channel_type="interstitial"))
#
# # cracks_level_specific.append(cracks_per_level)
# #
# #
# #
# # cracks_region_specific_max = np.amax(np.array(cracks_region_specific))
# # cracks_region_specific_norm = (cracks_region_specific / np.amax(cracks_region_specific)) * 0.5
# #
# #
# # layer_no_of_channels = np.zeros(instance_intact.interstitial_channels)
# #
# # for i in range(instance_intact.interstitial_channels):
# #     channel_no = i + 1
# #     layer_no_of_channels[i] = instance_intact.layers_from_centre(channel_no, channel_type='inter')
#
# ##################################################
# # ############# 3D CRACK PLOT ####################
# ##################################################
#
# # brick_coord_list_fuel = instance.get_brick_xyz_positions('xyz', channel_type="fuel", channels_only=0)
# # X = instances[0].linear_crack_array_1d(array_type="pos")
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# #
# # print(np.unique(X))
# #
# # ax.title.set_text(cases[0])
# # ax.scatter(brick_coord_list_fuel[0], brick_coord_list_fuel[1], brick_coord_list_fuel[2], c=X, cmap='gnuplot')
# # plt.legend()
# # ax.view_init(45, 45)
# # plt.show()
#
# ##################################################
# # ######### CRACK CONCENTRATION PLOT #############
# ##################################################
#
# # # Creates the figure
# # fig = plt.figure(figsize=(10, 9))
# #
# # # The maximum number of cracks that surround any channel. Used for bounding.
# # max_counts = np.amax(cracks_channel_specific)
# #
# # print("Max counts: ", max_counts)
# #
# # # Generates the plot grid for each case
# # counts_grid = AxesGrid(fig, 111,
# #                        nrows_ncols=(1, len(cases)),
# #                        axes_pad=0.2,
# #                        cbar_mode='single',
# #                        cbar_location='bottom',
# #                        cbar_pad=0.2
# #                        )
# #
# # # Iterates through each plot setting the parameters
# # for i, ax in enumerate(counts_grid):
# #     # sets the title as the case
# #     column_title = cases[i]
# #     ax.title.set_text(column_title)
# #
# #     # Turns off the labels and ticks of each axis
# #     turn_off_graph_decorations(ax)
# #
# #     # Generates the plot
# #     im = ax.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1],
# #                     marker='o', c=cracks_channel_specific[i], cmap='OrRd', label='inter',
# #                     s=100)
# #     im.set_clim(0, max_counts)
# #
# # # Creates the colorbar
# # cbar = ax.cax.colorbar(im)
# # cbar = counts_grid.cbar_axes[0].colorbar(im)
# #
# # # Saves the figure
# # filenm = "./Comparing_three_cases/concentration_of_cracks.png"
# # # plt.savefig(filenm)
# # plt.show()
#
# ##################################################
# # ############ CRACK AREA PLOT ###################
# ##################################################
#
#
# # fig = plt.figure(figsize=(27, 9))
# # axes = [fig.add_subplot(1, no_cases, (c + 1)) for c in range(0, no_cases)]
# #
# # y_labels = False
# #
# # for case, ax, case_region_values in zip(cases, axes, cracks_region_specific):
# #
# #     ax.set_xticks([])
# #     ax.set_yticks([])
# #
# #     ax.title.set_text(case)
# #
# #     square = plt.Rectangle((0, 0), 1, 1, color='royalblue')
# #     ax.add_artist(square)
# #
# #     for region_cracks, colour in zip(case_region_values, layer_colors):
# #         region_cracks_norm = (region_cracks / cracks_region_specific_max) * 0.4
# #         circle = plt.Circle((0.5, 0.5), region_cracks_norm, color=colour)
# #         ax.add_artist(circle)
# #         ax.annotate(str(region_cracks), xy=(region_cracks_norm + 0.5, 0.49), size=25)
# #
# # fig.show()
#
# ##################################################
# # ######### CRACKS PER LEVEL PLOT ################
# ##################################################
# #
# # ncols = 3
# #
# # # create the plots
# # fig = plt.figure(figsize=(25, 9))
# # axes = [fig.add_subplot(1, ncols, (c + 1)) for c in range(0, ncols)]
# #
# # y_labels = False
# #
# # # add some data
# # for case, ax, case_levels in zip(cases, axes, cracks_region_specific):
#
# #     if y_labels: ax.set_yticks([])
# #     else: ax.set_ylabel('Core Level')
# #     y_labels = True
# #
# #     for i, region in enumerate(case_levels):
# #         ax.barh(np.arange(len(region))+1, region, color=layer_colors[i])
# #
# #     ax.set_title(case)
# #
# # plt.show()
#
# ##################################################
# # ############ EARTHQUAKE PLOT ###################
# ##################################################
#
# #
# # for frame in frames_of_interest:
# #     plt.plot([frame, frame], [np.amin(earthquake_acceleration), np.amax(earthquake_acceleration)], color='silver')
# #
# # plt.plot(earthquake_acceleration)
# # plt.xlabel("Time Frame (n)")
# # plt.ylabel("Ground Acceleration (m/$s^2$)")
# #
# #
# #
# # plt.show()
#
# ##################################################
# # ################ RESULT PLOT ###################
# ##################################################
#
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
#             results.append(instance.get_result_at_time(time_index=frame, result_columns=str(result),
#                                                        result_type="max absolute"))
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
#                         marker='o', c=results[i], cmap='Reds', label='inter',
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
# #
# #     ########################################################################################
# #     # ############################### Regional Results #####################################
# #     ########################################################################################
#
#
# # channel_type = 'inter'
# # # Iterates through the user defined output types of interest, generating a separate figure for each
# # for result in results_of_interest:
# #     results = []
# #
# #     fig = plt.figure(figsize=(12, 12))
# #
# #     # Iterate through the frames of interest, generating a row for each
# #     for frame_no, frame in enumerate(frames_of_interest):
# #
# #         # Gets the earthquake acceleration at the given time-step
# #         accel = earthquake_acceleration[frame]
# #
# #         # Iterate through instances (cases) generating an array of length 321 i.e. the result for each channel
# #         for instance_no, instance in enumerate(instances):
# #             instance_result = instance.get_result_at_time(time_index=frame, result_columns=str(result),
# #                                                           result_type="relu")
# #
# #             graph_results = np.zeros(len(inclusive_layers))
# #
# #             for i, channel_result in enumerate(instance_result):
# #                 channel_layer = layer_no_of_channels[i]
# #
# #                 for j, region in enumerate(inclusive_layers):
# #                     if channel_layer < region: graph_results[j] += channel_result
# #
# #             results.append(graph_results)
# #
# #     results_max = np.amax(np.array(results))
# #
# #     # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
# #     grid = AxesGrid(fig, 111,
# #                     nrows_ncols=(len(frames_of_interest), len(cases)),
# #                     # axes_pad=0.12,
# #                     # cbar_mode='single',
# #                     # cbar_location='right',
# #                     # cbar_pad=0.2
# #                     )
# #
# #     frame_no = 0
# #
# #     # Iterates through all plots in the grid, assigning the results to the plot
# #     for i, ax in enumerate(grid):
# #
# #         # If the iterator is in the first row, assigns the title as the case
# #         if i < len(cases):
# #             column_title = cases[i]
# #             ax.title.set_text(column_title)
# #
# #         # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
# #         # acceleration
# #         if (i % 3) == 0:
# #             frame = frames_of_interest[frame_no]
# #             row_title = "Frame: " + str(frame) + " - Acc: " + str(round(earthquake_acceleration[frame], 2))
# #             ax.set_ylabel(row_title)
# #             frame_no += 1
# #
# #         # Turns off the other graph markings
# #         turn_off_graph_decorations(ax)
# #
# #         result_i = results[i]
# #         for region_result, colour in zip(result_i, layer_colors):
# #             region_results_norm = (region_result / results_max) * 0.4
# #             circle = plt.Circle((0.5, 0.5), region_results_norm, color=colour)
# #             ax.add_artist(circle)
# #             # ax.annotate(str(round(region_result, 2)), xy=(region_results_norm + 0.5, 0.49), size=25)
# #
# #     filenm = "./Comparing_three_cases/region_results" + str(result) + ".png"
# #     plt.savefig(filenm)
#
# #     ###############################################################################
# # #   ############################## RESULTS PER LEVEL PLOT #########################
# #     ###############################################################################
#
# channel_type = 'inter'
# # Iterates through the user defined output types of interest, generating a separate figure for each
# # for result in results_of_interest:
# #     results = []
# #
# #     fig = plt.figure(figsize=(12, 3))
# #
# #     # Iterate through the frames of interest, generating a row for each
# #     for frame_no, frame in enumerate(frames_of_interest):
# #
# #         # Gets the earthquake acceleration at the given time-step
# #         accel = earthquake_acceleration[frame]
# #
# #         # Iterate through instances (cases) generating an array of length 321 i.e. the result for each channel
# #         for instance_no, instance in enumerate(instances):
# #
# #             # 2D array - channels : levels
# #             instance_result = instance.get_result_at_time(time_index=frame, result_columns=str(result),
# #                                                           result_type="all")
# #
# #             # Generate a 2D np array - no_regions : no_levels
# #             graph_results = np.zeros([len(inclusive_layers), inter_levels])
# #
# #             # cycle through channel results
# #             for channel_layer, channel_result in zip(layer_no_of_channels, instance_result):
# #
# #                 # cycle through region boundaries i.e. the layer the bounds each region
# #                 for i, region in enumerate(inclusive_layers):
# #
# #                     # cycle through core interstitial levels
# #                     for j in range(inter_levels):
# #                         level_result = channel_result[j]
# #
# #                         if channel_layer < region: graph_results[i, j] += level_result
# #
# #             results.append(graph_results)
# #
# #     # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
# #     fig1, axs = plt.subplots(len(frames_of_interest), no_cases)
# #
# #     frame_no = 0
# #
# #     # Iterates through all plots in the grid, assigning the results to the plot
# #     for i, ax in enumerate(axs.flat):
# #
# #         # If the iterator is in the first row, assigns the title as the case
# #         if i < len(cases):
# #             column_title = cases[i]
# #             ax.title.set_text(column_title)
# #
# #         # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
# #         # acceleration
# #         if (i % 3) == 0:
# #             frame = frames_of_interest[frame_no]
# #             row_title = "Frame: " + str(frame)
# #             ax.set_ylabel(row_title)
# #             frame_no += 1
# #
# #         # Turns off the other graph markings
# #         turn_off_graph_decorations(ax)
# #
# #         result_i = results[i]
# #         for region_result, colour in zip(result_i, layer_colors):
# #             ax.barh(np.arange(len(region_result)) + 1, region_result, color=colour)
# #
# #     filenm = "./Comparing_three_cases/level_results" + str(result) + ".png"
# #     plt.savefig(filenm)
# # plt.show()
#
# #     ########################################################################################
# #     # ######################## result vs concentration #####################################
# #     ########################################################################################
# #
# # fig = plt.figure(figsize=(24, 12))
# #
# # # Creates a plot grid. Number of rows is number of frames of interest, number of columns is cases of interest
# # grid_compare = AxesGrid(fig, 111,
# #                         nrows_ncols=(len(frames_of_interest), len(cases)),
# #                         axes_pad=0.12,
# #                         cbar_pad=0.2
# #                         )
# #
# # frame_no = 0
# #
# # pearson_coeffs = [[], [], []]
# # # Iterates through all plots in the grid, assigning the results to the plot
# # for i, ax in enumerate(grid_compare):
# #
# #     # If the iterator is in the first row, assigns the title as the case
# #     if i < len(cases):
# #         column_title = cases[i]
# #         ax.title.set_text(column_title)
# #
# #     # If the iterator is in the first column, assigns a row title which includes the frame number and earthquake
# #     # acceleration
# #     if (i % 3) == 0:
# #         frame = frames_of_interest[frame_no]
# #         row_title = "Frame: " + str(frame) + " - Acc: " + str(round(earthquake_acceleration[frame], 2))
# #         ax.set_ylabel(row_title)
# #         frame_no += 1
# #
# #     # Plots the results for the case/frame
# #
# #     x = cracks_channel_specific[(i % 3)]
# #     y = results[i]
# #     pearson_coeffs[(i % 3)].append(pearsonr(x, y)[0])
# #     im = ax.scatter(x, y, marker='o')
# #     ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 2))(np.unique(x)))
# #
# # # Saves the file
# # filenm = "./Comparing_three_cases/correlation" + str(result) + ".png"
# # # plt.show()
# # plt.savefig(filenm)
# # print(np.mean(pearson_coeffs, axis=0))
#
#
# #     ########################################################################################
# #     # ######################## ALL CASE 2D TOP DOWN COMPOSITE ##############################
# #     ########################################################################################
#
#
# labels = []
#
# for result in results_of_interest:
#
#     result_labels = []
#     for frame in frames_of_interest:
#         features, frame_labels = features_and_labels_single_frame(case_root, result=str(result), time=frame)
#         result_labels.append(frame_labels)
#
#     labels.append(result_labels)
#
# with open('full_result_.pkl', 'wb') as f:
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

# case_folders = ["/media/huw/Disk1/variable_crack_percentage/seed_5004/",
#                 "/media/huw/Disk1/variable_crack_percentage/seed_5008/",
#                 "/media/huw/Disk1/variable_crack_percentage/seed_5015/"
#                 ]
#
# cases = np.array([cases_list(case_folder) for case_folder in case_folders])
#
# crack_fractions = [5, 10, 20, 40]
#
# no_seeds = len(case_folders)
# no_fractions = len(crack_fractions)
#
# result_column = 1
# frame = 48
# #
# # channel_cracks = np.zeros([no_seeds, no_fractions, inter_channels])
#
# channel_results = np.zeros([no_seeds, no_fractions, inter_channels])
#
# for i, seed_cases in enumerate(cases):
#
#     for case_path in seed_cases:
#
#         for j, fraction in enumerate(crack_fractions):
#             search_term = "P" + str(fraction)
#             if is_in(case_path, search_term):
#                 instance = core.Parse(case_path)
#                 channel_results[i, j] = instance.get_result_at_time(time_index=frame, result_columns=str(result_column),
#                                                                     result_type="absolute max")

#                 channel_cracks[i, j] = instance.channel_specific_cracks()[1]
#
# np.save('variable_fraction', channel_cracks)

# channel_cracks = np.load("variable_fraction.npy")
#
# # # Creates the figure
# fig = plt.figure(figsize=(10, 9))
#
# # # Generates the plot grid for each case
# grid = AxesGrid(fig, 111,
#                 nrows_ncols=(no_fractions, no_seeds),
#                 axes_pad=0.12,
#                 cbar_mode='single',
#                 cbar_location='right',
#                 cbar_pad=0.2
#                 )
#
# max_counts = np.amax(channel_cracks)

# for i, ax in enumerate(grid):
#     row = int(i / no_seeds)
#     column = i % no_seeds
#
#     if row == 0:
#         column_title = case_folders[i].split('/')[-2]
#         ax.title.set_text(column_title)
#
#     if column == 0:
#         fraction = crack_fractions[row]
#         y_label = str(fraction) + "% cracked"
#         ax.set_ylabel(y_label)
#
#     # Turns off the labels and ticks of each axis
#     turn_off_graph_decorations(ax)
#
#     # Generates the plot
#     im = ax.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1],
#                     marker='o', c=channel_cracks[column, row], cmap='OrRd', label='inter',
#                     s=30)
#     im.set_clim(0, max_counts)
#
#     # Creates the colorbar
#     cbar = ax.cax.colorbar(im)
#     cbar = grid.cbar_axes[0].colorbar(im)

# plt.show()

# RESULTS - TODO MOVE THE GRAPHING TO A FUNCTION OF ITS OWN

# max_result = np.amax(channel_results)
# min_result = np.amin(channel_results)
#
# for i, ax in enumerate(grid):
#     row = int(i / no_seeds)
#     column = i % no_seeds
#
#     if row == 0:
#         column_title = case_folders[i].split('/')[-2]
#         ax.title.set_text(column_title)
#
#     if column == 0:
#         fraction = crack_fractions[row]
#         y_label = str(fraction) + "% cracked"
#         ax.set_ylabel(y_label)
#
#     # Turns off the labels and ticks of each axis
#     turn_off_graph_decorations(ax)
#
#     # Generates the plot
#     im = ax.scatter(channel_coord_list_inter[0], channel_coord_list_inter[1],
#                     marker='o', c=channel_results[column, row], cmap='seismic', label='inter',
#                     s=30)
#     im.set_clim(min_result, max_result)
#
#     # Creates the colorbar
#     cbar = ax.cax.colorbar(im)
#     cbar = grid.cbar_axes[0].colorbar(im)
#
# plt.show()
