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

            # List of cracked core instances. Left blank at this point
            self.cases_instances = None

            # Number of instances requested by the user. Left blank at this point
            self.number_of_cases_requested = None

    def load_dataset_instances(self, number_of_cases='all', save_local=True):

        if not hasattr(self, 'cases_list'):
            message = "You have not yet instantiated this dataset object properly. Please check and try again."
            warnings.warn(message, stacklevel=3)
            return

        #############################################################
        ################### Number of Cases #########################
        #############################################################

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

        self.number_of_cases_requested = int_number_of_cases

        #############################################################
        loading = False
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
            else:
                print(" Expanding dataset from file...")
                loading = True

        cases_list_strings = self.cases_list[:int_number_of_cases]

        # TODO dynamic growth of data-set i.e. if the user specifies a larger dataset than is saved, it uses existing

        i = 0

        # Instantiate all the cases. Skip the ones with problems. Make a list of the instances of all the rest.
        cases_instances = []
        cases_list_updated = []
        for case in cases_list_strings:

            try:

                # Loads case from file if it exists
                if loading and i < len(cases_from_file):
                    cases_list_updated.append(cases_from_file[i])
                    cases_instances.append(instances_from_file[i])
                else:
                    # Try and instantiate the case
                    cases_list_updated.append(case)
                    cases_instances.append(reactor_case.Parse(case))

                i = i + 1

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

    def labels(self, channels='all', levels='all', result_time=50, result_column="1", result_type="max", flat=True,
               load_from_file=False):

        # Checks if dataset has been loaded
        if self.cases_instances is None:
            message = "You have not yet loaded the dataset. Please call the load_dataset_instances function before."
            warnings.warn(message, stacklevel=3)
            return

        cases_instances = self.cases_instances
        no_cases = len(cases_instances)

        # Take a single instance to get representative variables
        example_instance = cases_instances[0]

        array_dims = [no_cases]

        number_inter_channels, number_inter_levels = array_shape(example_instance, channels, levels, 'labels')

        # Check if label dataset exists
        if flat: flat_string = 'T'
        else: flat_string = 'F'

        # Generates a filename based on the user settings
        file_name = "Y_" + self.name + "_" + str(number_inter_channels) + "_" + str(number_inter_levels) + "_" + \
                    str(result_time) + "_" + str(result_column) + "_" + flat_string + ".npy"

        no_cases = self.number_of_cases_requested

        # Checks if a file with similar settings was generated previously
        if os.path.exists(file_name):
            if load_from_file:
                Y_loaded = np.load(file_name)
                if no_cases <= Y_loaded.shape[0]:
                    print("Labels dataset was found on disk. Loading...")
                    return Y_loaded[:no_cases]
            else:
                print("Labels dataset was found on disk. However, load_from_disk parameter is set to False. If you wish"
                      " to load this dataset, please set load_from_disk parameter to True")
                return

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

        print("Saving labels to file: " + file_name)
        np.save(file_name, Y)
        return Y

        #
        #
        #
        #
        #
