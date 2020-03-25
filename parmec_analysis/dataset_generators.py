from parmec_analysis.utils import cases_list, is_in, split_separators
from parmec_analysis import reactor_case
import numpy as np
import warnings
import pickle
import os


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


def min_max_channels_levels(core_instance, channels, levels, channel_type='fuel'):
    """ returns the minimum and maximum channel and level based on user defined input strings"""

    # Work out the number of channels
    min_channel, max_channel = core_instance.parse_channel_argument(channels, channel_type)

    min_level, max_level = core_instance.parse_level_argument(levels, channel_type)

    return (min_channel, max_channel), (min_level, max_level)


def get_number(input_val):
    """ converts an input to an int. If it's char input, return None"""

    # Handles if user specifies number of cases by string
    if isinstance(input_val, str):

        if input_val.isnumeric():
            return int(input_val)
        else:
            pass

    elif isinstance(input_val, int):
        return input_val

    return None


def check_directory(path):
    """ check a directory for validity """

    if not os.path.exists(path):
        return False
    elif cases_list(path) == 0:
        return False
    else:
        return True


def load_data_from_file(file_name):
    """ attempts to load data from a given file name. If file doesn't exist, returns False"""

    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            cases_list_file, instances_file = pickle.load(f)
            return cases_list_file, instances_file
    else:
        return False


class DatasetSingleFrame:

    def __init__(self, path_string="", name="dataset", number_of_cases='all', save_local=True):

        # The user can choose to name the dataset
        self.name = name

        cases_from_path = []
        cases_from_file, instances_from_file = [], []

        # check the directory
        if check_directory(path_string):
            cases_from_path = cases_list(path_string)

        # get the requested number of cases
        number_of_cases_requested = get_number(number_of_cases)

        # a file name associated with the dataset
        file_name = name + "_cases.pkl"

        #################################################################################################
        ################## LOADING FROM LOCAL DISK ######################################################
        #################################################################################################
        # this section determines if there is locally stored data i.e. if this dataset has been loaded
        # previously. If there is evidence of a

        # Attempt to load data from the local disk
        data_from_file = load_data_from_file(file_name)

        # if data exists, continue attempting to load
        if data_from_file:

            print('Case were found in file: ' + file_name + '...')

            cases_from_file = data_from_file[0]
            instances_from_file = data_from_file[1]

            # if whole dataset requested and cases from file are >= to number at path
            if not number_of_cases_requested and len(cases_from_file) >= len(cases_from_path):
                # Load entire locally stored dataset
                print('Loading cases from file: ' + file_name + '...')
                self.cases_list = cases_from_file
                self.core_instances = instances_from_file
                return

            # If the user requests a number of cases less than the number of cases on file
            elif number_of_cases_requested <= len(cases_from_file):
                print('Loading cases from file: ' + file_name + '...')
                self.cases_list = cases_from_file[:number_of_cases_requested]
                self.core_instances = instances_from_file[:number_of_cases_requested]
                return

            # Continue to data extraction below
            else:
                print('Insufficient data found in file: ' + file_name + '. Expanding...')
                pass

        # check if data was actually found before continuing
        if len(cases_from_file) == 0 and len(cases_from_path) == 0:
            message = "Directory " + path_string + " contains no data and no local data found." \
                                                   " Please check and try again."
            warnings.warn(message, stacklevel=3)
            return

        #################################################################################################
        ################## LOADING FROM PATH ############################################################
        #################################################################################################
        # if the program is running for the first time, this section runs first. It iterates through
        # the data set loading data from a saved file if it exists (loaded in previous section) or
        # from the path. If the number of successfully loaded cases equals the number required, it ends.

        cases_list_updated = []
        core_instances_updated = []

        # This iterator increases every time a successful case is joined to the list
        i = 0

        for case in cases_from_path:

            try:

                # Loads case from file if it exists
                if i < len(cases_from_file):
                    cases_list_updated.append(cases_from_file[i])
                    core_instances_updated.append(instances_from_file[i])
                else:
                    # Try and instantiate the case
                    core_instance = reactor_case.Parse(case)
                    cases_list_updated.append(case)
                    core_instances_updated.append(core_instance)

                i += 1

                # if the number of successful cases is reached, end the
                if i == number_of_cases_requested:
                    break

            # skips the case if a problem is detected
            except FileNotFoundError:
                message = "Warning: Case " + case + " was not found. Skipping this case"
                warnings.warn(message, stacklevel=2)

        print("Successfully obtained", i, "cases.")

        self.cases_list = cases_list_updated
        self.core_instances = core_instances_updated

        # Save instances to local file if requested and number loaded
        # greater than the number existing on file
        if save_local and len(cases_list_updated) > len(cases_from_file):
            print("Saving core instances to file " + file_name + "...")
            with open(file_name, 'wb') as f:
                pickle.dump([cases_list_updated, core_instances_updated], f)


class Features:

    def __init__(self, dataset, channels, levels, array_type):
        self.dataset = dataset
        self.number_instances = len(dataset.cases_list)

        example_instance = dataset.core_instances[0]

        # The total number of rows/columns which are padding
        double_padding = example_instance.padding * 2

        self.core_rows = example_instance.core_rows + double_padding
        self.core_columns = example_instance.core_columns + double_padding

        channels_range, levels_range = min_max_channels_levels(example_instance, channels, levels)

        self.channels_range = channels_range
        self.levels_range = levels_range

        self.number_channels = channels_range[1] - channels_range[0]
        self.number_levels = levels_range[1] - levels_range[0]

        self.array_type = array_type


class Features1D(Features):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', extra_dimension=False):
        super().__init__(dataset, channels, levels, array_type)

        self.values, self.feature_shape = self.generate_array(dataset, channels, levels, array_type, extra_dimension)
        self.extra_dimension = extra_dimension

    def generate_array(self, dataset, channels, levels, array_type, extra_dimension):

        instance_array_length = self.number_channels * self.number_levels
        X_1d = np.zeros([len(dataset.cases_list), instance_array_length])

        for i, instance in enumerate(dataset.core_instances):
            X_1d[i] = instance.crack_array_1d(channels=channels, levels=levels, array_type=array_type)

        # If the user requires the array to be configured for convolutional networks, then the array is reshaped
        if extra_dimension:
            X_shape = [X_1d.shape[0], X_1d.shape[1], 1]
            return X_1d.reshape(X_shape), X_shape[1:]

        # Else just return the normal numpy array
        else:
            return X_1d, X_1d.shape[1]


class Features2D(Features1D):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', extra_dimension=False):
        Features.__init__(self, dataset, channels, levels, array_type)

        self.values, self.feature_shape = self.generate_array(dataset, channels, levels, array_type, extra_dimension)
        self.extra_dimension = extra_dimension

    def generate_array(self, dataset, channels, levels, array_type, extra_dimension):
        # Get the 1d feature array
        values_1d, _ = Features1D.generate_array(self, dataset, channels, levels, array_type, extra_dimension)

        X_shape = [values_1d.shape[0], self.number_levels, self.number_channels]

        # If the user requires the array to be configured for convolutional networks, then the array is reshaped
        if extra_dimension:
            X_shape.append(1)

        return values_1d.reshape(X_shape), X_shape[1:]


class Features3D(Features):

    def __init__(self, dataset, levels='all', array_type='positions only'):
        super().__init__(dataset, channels='all', levels=levels, array_type=array_type)

        self.values, self.feature_shape = self.generate_array(dataset, levels, array_type)

    def generate_array(self, dataset, levels, array_type):

        # This array contains the data taken directly from the core instances
        X_3d_inst = np.zeros([len(dataset.cases_list), self.number_levels, self.core_rows, self.core_columns])

        for i, instance in enumerate(dataset.core_instances):
            X_3d_inst[i] = instance.channel_array_argument_wrapper(array_type=array_type, levels=levels,
                                                                   quiet=True)

        # Reshape the 3D array so as to to have a more appropriate 'view' onto the core for ML
        X_shape = [len(dataset.cases_list), self.core_rows, self.core_columns, self.number_levels]
        X_3d_rs = np.zeros(X_shape)

        # Go through each level of the old array and reassign its location
        for level in range(X_3d_inst.shape[1]):
            feature_slice = X_3d_inst[:, level, :, :]
            X_3d_rs[:, :, :, level] = feature_slice

        return X_3d_rs, X_shape[1:]


class FeaturesFlat(Features3D):

    pass

# def labels(self, channels='all', levels='all', result_time=50, result_column="1", result_type="max", flat=True,
#            load_from_file=False):
#
#     # self.label_attributes['Label Channels'] = channels
#     # self.label_attributes['Label Levels'] = levels
#     # self.label_attributes['Result Frame:'] = result_time
#     # self.label_attributes['Result Column:'] = result_column
#     # self.label_attributes['Result Type:'] = result_type
#
#     # Checks if dataset has been loaded
#     if self.cases_instances is None:
#         message = "You have not yet loaded the dataset. Please call the load_dataset_instances function before."
#         warnings.warn(message, stacklevel=3)
#         return
#
#     cases_instances = self.cases_instances
#     no_cases = self.number_of_cases_requested
#
#     # Take a single instance to get representative variables
#     example_instance = cases_instances[0]
#
#     array_dims = [no_cases]
#
#     # Work out the number of channels
#     min_channel, max_channel = example_instance.parse_channel_argument(channels, 'label')
#     number_inter_channels = max_channel - min_channel
#
#     min_level, max_level = example_instance.parse_level_argument(levels, 'label')
#     number_inter_levels = max_level - min_level
#
#     # save metrics
#     # self.number_label_channels = number_inter_channels
#     # self.number_label_levels = number_inter_levels
#
#     # Check if label dataset exists
#     if flat:
#         flat_string = 'T'
#     else:
#         flat_string = 'F'
#
#     # Generates a filename based on the user settings
#     file_name = "Y_" + self.name + "_" + str(number_inter_channels) + "_" + str(number_inter_levels) + "_" + \
#                 str(result_time) + "_" + str(result_column) + "_" + result_type + "_" + flat_string + ".npy"
#
#     # Checks if a file with similar settings was generated previously
#     if os.path.exists(file_name):
#         if load_from_file:
#             Y_loaded = np.load(file_name)
#             if no_cases <= Y_loaded.shape[0]:
#                 print("Labels dataset was found on disk. Loading...")
#                 if len(Y_loaded.shape[1:]) == 1:
#                     self.label_shape = Y_loaded.shape[1]
#                 else:
#                     self.label_shape = Y_loaded.shape[1:]
#                 return Y_loaded[:no_cases]
#         else:
#             print("Labels dataset was found on disk. However, load_from_disk parameter is set to False. If you wish"
#                   " to load this dataset, please set load_from_disk parameter to True")
#             return
#
#     # If the result for all bricks is being returned
#     if is_in(result_type, 'all'):
#         # Flat 1D
#         if flat:
#             array_dims.append(number_inter_channels * number_inter_levels)
#
#         # 2d
#         else:
#             array_dims.append(number_inter_channels)
#             array_dims.append(number_inter_levels)
#
#     # Level specific results
#     else:
#         array_dims.append(number_inter_channels)
#
#     Y = np.zeros(array_dims)
#
#     for i, instance in enumerate(cases_instances):
#
#         instance_result = instance.get_result_at_time(result_time, result_columns=str(result_column),
#                                                result_type=result_type, flat=flat)
#
#         if is_in(channels, 'all') and is_in(levels, 'all'):
#             Y[i] = instance_result
#
#         else:
#             # TODO FIX THIS. MOVE CHANNEL/LEVEL RESIZE OF ARRAY INSIDE get_result_at_time function
#             instance_result_cl = instance_result.reshape([instance.interstitial_channels, instance.inter_levels])
#             # print(min_channel, max_channel, min_level, max_level)
#             Y[i] = instance_result_cl[min_channel-1:max_channel+1, min_level:max_level].flatten()
#
#     if len(array_dims[1:]) == 1:
#         self.label_shape = array_dims[1]
#     else:
#         self.label_shape = array_dims[1:]
#     print("Saving labels to file: " + file_name)
#     np.save(file_name, Y)
#     return Y
#
#
