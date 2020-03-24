from parmec_analysis.utils import cases_list, is_in, split_separators
from parmec_analysis import reactor_case
import numpy as np
import warnings
import pickle
import os
from sys import getsizeof


# TODO make sub classes of features and labels

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

        # The user can choose to name the dataset
        self.name = name

        try:
            # This is the list of cases detected in the directory
            cases_list_path = cases_list(path_string)
        except FileNotFoundError:
            print("Can not access results directory.")
            if self.case_exists():
                file_name = self.name + "_cases.pkl"
                print("Cases found saved in local file. These will be loaded but be aware the cases list can lot be "
                      "expanded without access to the original path.")
                with open(file_name, 'rb') as f:
                    cases_list_path, _ = pickle.load(f)

            else:
                print("Exiting...")
                return

        # First, check if the directory is empty
        if len(cases_list_path) == 0:
            message = "Directory " + path_string + " contains no data. Please check and try again."
            warnings.warn(message, stacklevel=3)

        else:

            # If the directory contains folders, continue
            self.cases_list = cases_list_path

            # This is the path to the dataset files
            self.dataset_path = path_string

            # List of cracked core instances. Left blank at this point
            self.cases_instances = None

            # Number of instances requested by the user. Left blank at this point
            self.number_of_cases_requested = None

            # Size and shape of last called feature array
            self.feature_shape = None

            # Size and shape of last called label array
            self.label_shape = None

            # These will be lists of attributes regarding the features and labels
            # Will be used for setting graph titles and such
            self.feature_attributes = {}
            self.label_attributes = {}

            # Size metrics for feature and label arrays
            self.number_feature_channels = None
            self.number_feature_levels = None

            self.number_label_channels = None
            self.number_label_levels = None

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
        loading = self.case_exists()
        # Checks if a local dataset exists
        file_name = self.name + "_cases.pkl"
        if loading:
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

    def features(self, requested_feature='1d', channels='all', levels='all', array_type='pos', extra_dimension=True):

        # Checks if dataset has been loaded
        if self.cases_instances is None:
            message = "You have not yet loaded the dataset. Please call the load_dataset_instances function before."
            warnings.warn(message, stacklevel=3)
            return

        self.feature_attributes['Feature Channels'] = channels
        self.feature_attributes['Feature Levels'] = levels
        self.feature_attributes['Feature Type'] = array_type

        cases_instances = self.cases_instances

        # Take a single instance to get representative variables
        example_instance = cases_instances[0]

        no_fuel_channels, no_fuel_levels = array_shape(example_instance, channels, levels)

        # Save metrics
        self.number_feature_channels = no_fuel_channels
        self.number_feature_levels = no_fuel_levels

        # The total number of rows/columns which are padding
        double_padding = example_instance.padding * 2

        rows_columns = [example_instance.core_rows + double_padding, example_instance.core_columns + double_padding]

        def features_1d():

            array_length = no_fuel_channels * no_fuel_levels
            X_1d = np.zeros([len(cases_instances), array_length])

            for i, instance in enumerate(cases_instances):
                X_1d[i] = instance.crack_array_1d(channels=channels, array_type=array_type, levels=levels)

            # If the user requires the array to be configured for convolutional networks, then the array is reshaped
            if extra_dimension:
                X_shape = [X_1d.shape[0], X_1d.shape[1], 1]
                self.feature_shape = X_shape[1:]
                return X_1d.reshape(X_shape)

            # Else just return the normal numpy array
            else:
                self.feature_shape = X_1d.shape[1]
                return X_1d

        def features_2d():

            # Get the 1d array
            X_1d = features_1d()

            X_shape = [X_1d.shape[0], no_fuel_levels, no_fuel_channels]

            # If the user requires the array to be configured for convolutional networks, then the array is reshaped
            if extra_dimension:
                X_shape.append(1)

            self.feature_shape = X_shape[1:]
            return X_1d.reshape(X_shape)

        def features_3d():

            # This array contains the data taken directly from the core instances
            X_3d_inst = np.zeros([len(cases_instances), no_fuel_levels, rows_columns[0], rows_columns[1]])

            for i, instance in enumerate(cases_instances):
                X_3d_inst[i] = instance.channel_array_argument_wrapper(array_type=array_type, levels=levels,
                                                                       quiet=True)

            # Reshape the 3D array so as to to have a more appropriate 'view' onto the core for ML
            X_shape = [len(cases_instances), rows_columns[0], rows_columns[1], no_fuel_levels]
            X_3d_rs = np.zeros(X_shape)

            # Go through each level of the old array and reassign its location
            for level in range(X_3d_inst.shape[1]):
                feature_slice = X_3d_inst[:, level, :, :]
                X_3d_rs[:, :, :, level] = feature_slice

            self.feature_shape = X_shape[1:]
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

        self.label_attributes['Label Channels'] = channels
        self.label_attributes['Label Levels'] = levels
        self.label_attributes['Result Frame:'] = result_time
        self.label_attributes['Result Column:'] = result_column
        self.label_attributes['Result Type:'] = result_type

        # Checks if dataset has been loaded
        if self.cases_instances is None:
            message = "You have not yet loaded the dataset. Please call the load_dataset_instances function before."
            warnings.warn(message, stacklevel=3)
            return

        cases_instances = self.cases_instances
        no_cases = self.number_of_cases_requested

        # Take a single instance to get representative variables
        example_instance = cases_instances[0]

        array_dims = [no_cases]

        # Work out the number of channels
        min_channel, max_channel = example_instance.parse_channel_argument(channels, 'label')
        number_inter_channels = max_channel - min_channel

        min_level, max_level = example_instance.parse_level_argument(levels, 'label')
        number_inter_levels = max_level - min_level

        # save metrics
        self.number_label_channels = number_inter_channels
        self.number_label_levels = number_inter_levels

        # Check if label dataset exists
        if flat:
            flat_string = 'T'
        else:
            flat_string = 'F'

        # Generates a filename based on the user settings
        file_name = "Y_" + self.name + "_" + str(number_inter_channels) + "_" + str(number_inter_levels) + "_" + \
                    str(result_time) + "_" + str(result_column) + "_" + result_type + "_" + flat_string + ".npy"

        # Checks if a file with similar settings was generated previously
        if os.path.exists(file_name):
            if load_from_file:
                Y_loaded = np.load(file_name)
                if no_cases <= Y_loaded.shape[0]:
                    print("Labels dataset was found on disk. Loading...")
                    if len(Y_loaded.shape[1:]) == 1:
                        self.label_shape = Y_loaded.shape[1]
                    else:
                        self.label_shape = Y_loaded.shape[1:]
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

            instance_result = instance.get_result_at_time(result_time, result_columns=str(result_column),
                                                   result_type=result_type, flat=flat)

            if is_in(channels, 'all') and is_in(levels, 'all'):
                Y[i] = instance_result

            else:
                # TODO FIX THIS. MOVE CHANNEL/LEVEL RESIZE OF ARRAY INSIDE get_result_at_time function
                instance_result_cl = instance_result.reshape([instance.interstitial_channels, instance.inter_levels])
                # print(min_channel, max_channel, min_level, max_level)
                Y[i] = instance_result_cl[min_channel-1:max_channel+1, min_level:max_level].flatten()

        if len(array_dims[1:]) == 1:
            self.label_shape = array_dims[1]
        else:
            self.label_shape = array_dims[1:]
        print("Saving labels to file: " + file_name)
        np.save(file_name, Y)
        return Y

    def case_exists(self):
        file_name = self.name + "_cases.pkl"
        return os.path.exists(file_name)
