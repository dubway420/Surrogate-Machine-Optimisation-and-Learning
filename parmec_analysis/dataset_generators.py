from parmec_analysis.utils import cases_list, is_in, split_separators, features_and_or_labels
from parmec_analysis import reactor_case
import numpy as np
import warnings
import pickle
import os
from sys import getsizeof


def save_features(features, folder):
    # If there's more than one feature
    if isinstance(features, list):

        for i, array in enumerate(features):
            file_nm = folder + "/features_" + str(i)
            print(file_nm)
            # np.save(file_nm, array)

    # Numpy array i.e. only one feature
    else:
        file_nm = folder + "/features"
        print(file_nm)


def requested_feature_types(feature_types_string):
    """ Takes a string which the user inputs to request feature types (i.e. '1-d' or '2D')
        Returns an integer that represents the number of the feature requested"""

    # Remove any irrelevant terms from x_request
    for split_separator in split_separators:
        feature_types_string = feature_types_string.replace(split_separator, "")

    # Cycles through each possible feature type. Returns the int number of the feature requested
    for i in range(1, 4):
        if is_in(feature_types_string, (str(i) + "d")): return i

    # If it finds nothing, returns 0 (i.e. error)
    return 0


def array_shape(core_instance, channels, levels, channel_type='fuel'):
    """ Based on an instance of a whole core and strings that contain the channels and levels required, returns the
    dimensions of the output array i.e. number of channels and levels"""

    # Work out the number of channels
    min_channel, max_channel = core_instance.parse_channel_argument(channels, channel_type)
    no_channels = max_channel - min_channel

    min_level, max_level = core_instance.parse_level_argument(levels, channel_type)
    no_levels = max_level - min_level

    return no_channels, no_levels


class DatasetSingleFrame:

    def __init__(self, path_string, name="dataset"):

        # This is the list of cases detected in the directory
        cases_list_path = cases_list(path_string)

        # First, check if the directory is empty
        if len(cases_list_path) == 0:
            message = "Directory " + path_string + " contains no data. Please check and try again."
            warnings.warn(message, stacklevel=3)

        else:

            # If the directory contains folders, continue
            self.cases_list = cases_list_path

            # The user can choose to name the dataset
            self.name = name

            # This is the path to the dataset files
            self.dataset_path = path_string

            # List of cracked core instances. Left bank at this point
            self.cases_instances = None

    def load_dataset_instances(self, number_of_cases='all', save_local=True):

        if not hasattr(self, 'cases_list'):
            message = "You have not yet instantiated this dataset object properly. Please check and try again."
            warnings.warn(message, stacklevel=3)
            return

        # Handles if user specifies number of cases by string
        if isinstance(number_of_cases, str):

            if number_of_cases.isnumeric():
                # cases_list_strings = cases_list_strings[:int(number_of_cases)]
                int_number_of_cases = int(number_of_cases)
            else:
                int_number_of_cases = len(self.cases_list)

        elif isinstance(number_of_cases, int):
            # cases_list_strings = cases_list_strings[:number_of_cases]
            int_number_of_cases = number_of_cases
        else:
            message = "You have not provided a valid value for number_of_cases. Please check and try again."
            warnings.warn(message, stacklevel=3)
            return

        # Checks if a local dataset exists
        file_name = self.name + "_cases.pkl"
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                cases_from_file, instances_from_file = pickle.load(f)

            # If the user requests a number of cases fewer or equal to the number on file take a slice of the file array
            if int_number_of_cases <= len(cases_from_file):
                print('loading cases from file...')
                self.cases_list = cases_from_file[:int_number_of_cases]
                self.cases_instances = instances_from_file[:int_number_of_cases]
                return

        cases_list_strings = self.cases_list[:int_number_of_cases]

        # TODO dynamic growth of data-set i.e. if the user specifies a larger dataset than is saved, it uses existing
        # values loading from file

        # Instantiate all the cases. Skip the ones with problems. Make a list of the instances of all the rest.
        cases_instances = []
        cases_list_updated = []
        for case in cases_list_strings:

            try:

                # Try and instantiate the case
                cases_instances.append(reactor_case.Parse(case))
                cases_list_updated.append(case)

            except FileNotFoundError:
                message = "Warning: Case " + case + " was not found. Skipping this case"
                warnings.warn(message, stacklevel=2)

        # Update cases list with ones which have successfully instantiated
        self.cases_list = cases_list_updated

        # Load list of reactor core instances into memory
        self.cases_instances = cases_instances

        # Save instances to local file if requested
        if save_local:
            file_name = self.name + "_cases.pkl"
            print(" Saving core instances to file " + file_name + "...")
            with open(file_name, 'wb') as f:
                pickle.dump([cases_list_updated, cases_instances], f)

    def features(self, requested_feature='1d', channels='all', array_type='pos', levels='all', neural_network=True):

        # Checks if dataset has been loaded
        if self.cases_instances is None:
            message = "You have not yet loaded the dataset. Please call the load_dataset_instances function before."
            warnings.warn(message, stacklevel=3)
            return

        cases_instances = self.cases_instances

        # Take a single instance to get representative variables
        example_instance = cases_instances[0]

        no_fuel_channels, no_fuel_levels = array_shape(example_instance, channels, levels)

        # The total number of rows/columns which are padding
        double_padding = example_instance.padding * 2

        rows_columns = [example_instance.core_rows + double_padding, example_instance.core_columns + double_padding]

        def features_1d():

            array_length = no_fuel_channels * no_fuel_levels
            X_1d = np.zeros([len(cases_instances), array_length])

            for i, instance in enumerate(cases_instances):
                X_1d[i] = instance.crack_array_1d(channels=channels, array_type=array_type, levels=levels)

            # If the user requires the array to be configured for neural networks frameworks, then the array is reshaped
            if neural_network:
                return X_1d.reshape([X_1d.shape[0], X_1d.shape[1], 1])

            # Else just return the normal numpy array
            else:
                return X_1d

        def features_2d():

            # Get the 1d array
            X_1d = features_1d()

            # If the user requires the array to be configured for neural networks frameworks, then the array is reshaped
            if neural_network:
                return X_1d.reshape([X_1d.shape[0], no_fuel_levels, no_fuel_channels, 1])

            # Else just return the normal numpy array
            else:
                return X_1d.reshape([X_1d.shape[0], no_fuel_levels, no_fuel_channels])

        def features_3d():

            # This array contains the data taken directly from the core instances
            X_3d_inst = np.zeros([len(cases_instances), no_fuel_levels, rows_columns[0], rows_columns[1]])

            for i, instance in enumerate(cases_instances):
                X_3d_inst[i] = instance.channel_array_argument_wrapper(array_type=array_type, levels=levels,
                                                                       quiet=True)

            # Reshape the 3D array so as to to have a more appropriate 'view' onto the core for ML
            X_3d_rs = np.zeros([len(cases_instances), rows_columns[0], rows_columns[1], no_fuel_levels])

            # Go through each level of the old array and reassign its location
            for level in range(X_3d_inst.shape[1]):
                feature_slice = X_3d_inst[:, level, :, :]
                X_3d_rs[:, :, :, level] = feature_slice

            return X_3d_rs

        def features_2d_flat():
            # todo rename 2d flat
            pass

        # A switch for which feature array is required i.e. 1d, 2d or 3d
        feature_feature_int = requested_feature_types(requested_feature)

        if feature_feature_int == 0:
            message = "You have specified an invalid feature type. Please check and try again."
            warnings.warn(message, stacklevel=3)

        elif feature_feature_int == 1:
            return features_1d()
        elif feature_feature_int == 2:
            return features_2d()
        elif feature_feature_int == 3:
            return features_3d()

    def labels(self, channels='all', levels='all', result_time=50, result_column="1", result_type="max", flat=True):

        # Checks if dataset has been loaded
        if self.cases_instances is None:
            message = "You have not yet loaded the dataset. Please call the load_dataset_instances function before."
            warnings.warn(message, stacklevel=3)
            return

        cases_instances = self.cases_instances

        # Take a single instance to get representative variables
        example_instance = cases_instances[0]

        array_dims = [len(cases_instances)]

        number_inter_channels, number_inter_levels = array_shape(example_instance, channels, levels, 'labels')

        # If the result for all bricks is being returned
        if is_in(result_type, 'all'):
            # Flat 1D
            if flat:
                array_dims.append(number_inter_channels * number_inter_levels)

            # 2d
            else:
                array_dims.append(number_inter_channels)
                array_dims.append(number_inter_levels)

        # Level specific results
        else:
            array_dims.append(number_inter_channels)

        Y = np.zeros(array_dims)

        for i, instance in enumerate(cases_instances):
            Y[i] = instance.get_result_at_time(result_time, result_columns=str(result_column),
                                                   result_type=result_type, flat=flat)

        return Y

        #
        #
        #
        #
        #

    # def __init__(self, path_string, features_labels='both',  # General terms
    #              channels='all', levels='all',  # Data slice
    #              x_request='1d', x_type='positions',  # Feature terms
    #              result_time=50, result_column="1", result_type="max",  # Label terms
    #              flat_y=False):  # Only relevant if result_type  is "all"
    #     """ Gets the features and labels from the folder of results"""
    #
    #     instance = None
    #     return_X = None
    #     X_1d_np = None
    #
    #     # A switch for whether the user wants features, labels or both.
    #     self.features_labels_bool = features_and_or_labels(features_labels)
    #
    #     # Output lists
    #     X_1d, X_3d, Y = [], [], []
    #
    #     # A switch for which feature arrays are required i.e. 1d, 2d or 3d, or any combination
    #
    #     # Remove any irrelevant terms from x_request
    #     for split_separator in split_separators:
    #         x_request = x_request.replace(split_separator, "")
    #
    #     # search the request string for each of the three dimensions
    #     features_requested = [is_in(x_request, str(i + 1) + 'd') for i in range(3)]
    #
    #     cases = cases_list(path_string)[0:10]
    #
    #     self.case_list = cases
    #
    #     for case in cases:
    #
    #         try:
    #             instance = reactor_case.Parse(case)
    #
    #             # Features
    #             if return_items[0]:
    #
    #                 # If either the 1d or 2d feature is requested return the 1d array (both array types are dependent on 1d)
    #                 if features_requested[0] or features_requested[1]:
    #                     X_1d.append(instance.crack_array_1d(channels=channels, array_type=x_type, levels=levels))
    #
    #                 if features_requested[2]:
    #                     X_3d.append(instance.channel_array_argument_wrapper(array_type=x_type, levels=levels,
    #                                                                         quiet=True))
    #
    #             if return_items[1]:
    #                 Y.append(instance.get_result_at_time(result_time, result_columns=str(result_column),
    #                                                      result_type=result_type, flat=flat_y))
    #
    #         # If there's any problems with the files in a case, it completely skips to the next
    #         except FileNotFoundError:
    #             message = "Warning: Case " + case + " was not found. Skipping this case"
    #             warnings.warn(message)
    #
    #     # Work out the number of channels
    #     min_channel, max_channel = instance.parse_channel_argument(channels)
    #     no_fuel_channels = max_channel - min_channel
    #
    #     min_level, max_level = instance.parse_level_argument(levels)
    #     no_fuel_levels = max_level - min_level
    #
    #     # If features are requested, generate a list of them
    #     if return_items[0]:
    #         X = []
    #
    #         # Generate 1d numpy array. Only do this if 1d or 2d feature requested
    #         if features_requested[0] or features_requested[1]:
    #             # Convert 1D array to numpy and then reshape to single channel
    #             X_1d_np = np.array(X_1d)
    #             X_1d_np = X_1d_np.reshape([X_1d_np.shape[0], X_1d_np.shape[1]])
    #
    #         if features_requested[0]: X.append(X_1d_np)
    #         if features_requested[1]: X.append(X_1d_np.reshape(X_1d_np.shape[0], no_fuel_levels, no_fuel_channels))
    #
    #         # Changes the view onto the core array TODO make an option to change the view
    #         if features_requested[2]:
    #             X_3d_np = np.array(X_3d)
    #             X_3d_rs = np.zeros([X_3d_np.shape[0], X_3d_np.shape[2], X_3d_np.shape[3], X_3d_np.shape[1]])
    #
    #             for level in range(X_3d_np.shape[1]):
    #                 feature_slice = X_3d_np[:, level, :, :]
    #                 X_3d_rs[:, :, :, level] = feature_slice
    #
    #             X.append(X_3d_rs)
    #
    #         # Finalise the feature variable
    #         if len(X) == 1:
    #             return_X = X[0]
    #         else:
    #             return_X = X
    #
    #     # If both features and labels are requested:
    #     if return_items[0] and return_items[1]:
    #         return return_X, np.array(Y)
    #
    #     # Features only:
    #     elif return_items[0]:
    #         return return_X
    #
    #     # Labels only
    #     elif return_items[1]:
    #         return np.array(Y)
    #
    #     # If neither was requested, warn the user
    #     else:
    #         warnings.warn("Warning: You have not requested any output. Please specificy something in the ",
    #                       stacklevel=2)

#
# def dataset_storage(command, features_labels='both', folder='dataset_storage', overwrite=False):
#     """ Extracts the dataset and stores it. Returns the path of each saved file."""
#
#     # Folder validation: if the folder exists
#     if os.path.exists(folder) and os.path.isdir(folder):
#         print("Directory", folder + ' exists')
#
#         # If folder isn't empty
#         if len(os.listdir(folder)) > 0 and not overwrite:
#             warnings.warn("Folder for storage of files is not empty and override is NOT enabled. Exiting sub...",
#                           stacklevel=2)
#             return
#
#         # If folder isn't empty but overwrite enabled
#         elif len(os.listdir(folder)) > 0 and overwrite:
#             warnings.warn("Folder for storage of files is not empty and override is ENABLED. Overwriting files...",
#                           stacklevel=2)
#
#         # If folder is empty.
#         elif len(os.listdir(folder)) == 0:
#             print("Folder for storage of files is empty.")
#
#     else:
#         print("Directory", folder + ' does not exist. Creating...')
#         os.mkdir(folder)
#
#     # Determines if the user wants features, labels, or both
#     return_items = return_vector(features_labels)
#
#     # If both features and labels are set to false, exit the sub
#     if not return_items[0] and not return_items[1]:
#         warnings.warn("You have specified neither data metric (features or labels). Please check and try again...",
#                       stacklevel=2)
#         return
#
#     # Run the command to extract the dataset
#     data = command
#
#     # If both features and labels are requested
#     if return_items[0] and return_items[1]:
#         features = data[0]
#         labels = data[1]
#
#         save_features(features, folder)
#
#         filenm = folder + '/labels'
#         print(filenm)
#         # np.save(filenm, labels)
#
#     # Features only
#     elif return_items[0]:
#         save_features(data, folder)
#
#     elif return_items[1]:
#
#         filenm = folder + '/labels'
#         print(filenm)
#         # np.save(filenm, labels)
#
#     else:
#         warnings.warn("Warning: You have not requested any output. Please specificy something in the ",
#                       stacklevel=2)
