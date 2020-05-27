from parmec_analysis.utils import cases_list, is_in, split_separators, function_switch, get_number, check_directory
from parmec_analysis.utils import load_data_from_file
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
        if is_in(feature_types_string, (str(i) + "d")):
            return i

    # If it finds nothing, returns 0 (i.e. error)
    return 0


def min_max_channels_levels(core_instance, channels, levels, channel_type='fuel'):
    """ returns the minimum and maximum channel and level based on user defined input strings"""

    # Work out the number of channels
    min_channel, max_channel = core_instance.parse_channel_argument(channels, channel_type)

    min_level, max_level = core_instance.parse_level_argument(levels, channel_type)

    return (min_channel, max_channel), (min_level, max_level)


class DatasetSingleFrame:

    def __init__(self, path_string="", name="dataset", number_of_cases='all', validation_split=0.2, save_local=True):

        # The user can choose to name the dataset
        self.name = name
        self.validation_split = validation_split

        # These variables will be defined in the assign_attributes method

        self.cases_list = None
        self.core_instances = None
        self.number_instances = None
        self.split_number = None

        self.shuffled = False
        self.shuffle_seed = 0

        self.rolled = False
        self.rolled_by_increment = 0

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
        # ################# LOADING FROM LOCAL DISK ######################################################
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
                self.assign_attributes(cases_from_file, instances_from_file)
                return

            # If the user requests a number of cases less than the number of cases on file
            elif number_of_cases_requested <= len(cases_from_file):
                print('Loading cases from file: ' + file_name + '...')
                self.assign_attributes(cases_from_file[:number_of_cases_requested],
                                       instances_from_file[:number_of_cases_requested])
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
        # ################# LOADING FROM PATH ############################################################
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

        self.assign_attributes(cases_list_updated, core_instances_updated)

        # Save instances to local file if requested and number loaded
        # greater than the number existing on file
        if save_local and len(cases_list_updated) > len(cases_from_file):
            print("Saving core instances to file " + file_name + "...")
            with open(file_name, 'wb') as f:
                pickle.dump([cases_list_updated, core_instances_updated], f)

    def assign_attributes(self, case_list, core_instances):

        self.cases_list = case_list
        self.core_instances = core_instances
        self.number_instances = len(case_list)
        self.split_number = int(self.number_instances * (1 - self.validation_split))

    def training_cases(self):
        return self.cases_list[:self.split_number]

    def validation_cases(self):
        return self.cases_list[self.split_number:]

    def training_instances(self):
        return self.core_instances[:self.split_number]

    def validation_instances(self):
        return self.core_instances[self.split_number:]

    def shuffle(self, seed=12):
        """ shuffle the cases and instances lists. For use in generating new """

        np.random.seed(seed)
        cases_list_pre = self.cases_list
        np.random.shuffle(cases_list_pre)
        self.cases_list = cases_list_pre

        np.random.seed(seed)
        core_instances_pre = self.core_instances
        np.random.shuffle(core_instances_pre)
        self.core_instances = core_instances_pre

        self.shuffled = True
        self.shuffle_seed = seed

    def roll(self, increment=1, quiet=False):
        """ rolls the data by a multiple of a fraction. Useful for folding and validation"""

        fraction = self.validation_split

        if increment > int(1 / fraction) and not quiet:
            message = "You have specified an increment number (" + str(increment) + ") which is greater than 1 / " \
                                                                                    "fraction. The data-set will " \
                                                                                    "simply wrap."
            warnings.warn(message, stacklevel=3)

        # The number of instances that is represented by the fraction. This could be the validation fraction for eg.
        number_per_fraction = int(self.number_instances * fraction)

        roll_by_number = number_per_fraction * int(increment)

        cases_rolled = np.roll(self.cases_list, roll_by_number, axis=0)
        instances_rolled = np.roll(self.core_instances, roll_by_number, axis=0)

        self.cases_list = cases_rolled
        self.core_instances = instances_rolled

        self.rolled = True
        self.rolled_by_increment = increment

    def summary(self):

        summary_text = [
            "Dataset: " + self.name,
            "Total number of instances: " + str(self.number_instances),
            str(self.split_number) + " Training",
            str(self.number_instances - self.split_number) + " Validation",
            "Validation split: " + str(self.validation_split)

        ]

        if self.shuffled:
            shuffle_text = "Shuffled using seed: " + str(self.shuffle_seed)
            summary_text.append(shuffle_text)

        if self.rolled:
            roll_text = "Rolled by " + str(self.rolled_by_increment) + " increments"
            summary_text.append(roll_text)

        return summary_text


class Features:
    """ An abstract class to be inherited by one of the options below"""

    def __init__(self, dataset, channels, levels, array_type, channel_type='fuel', load_from_file=True):
        # self.feature_mode = None

        self.dataset = dataset
        self.number_instances = len(dataset.cases_list)

        example_instance = dataset.core_instances[0]

        self.example_instance = example_instance

        # TODO delete later
        self.example_instance.inter_rows = 19
        self.example_instance.inter_columns = 19

        # The total number of rows/columns which are padding
        double_padding = example_instance.padding * 2

        self.core_rows = example_instance.core_rows + double_padding
        self.core_columns = example_instance.core_columns + double_padding

        channels_range, levels_range = min_max_channels_levels(example_instance, channels, levels, channel_type)

        self.channels_range = channels_range
        self.levels_range = levels_range

        self.number_channels = channels_range[1] - channels_range[0]
        self.number_levels = levels_range[1] - levels_range[0]

        self.array_type = array_type

        # self.values, self.feature_shape = None, None
        self.values = None

        # Handles loading from file

        X_loaded = self.load_features_from_file()
        if len(X_loaded) > 0:

            if not load_from_file:
                print("Features dataset was found on disk. However, load_from_disk parameter is set to False. "
                      "If you wish to load this dataset, please set load_from_disk parameter to True")
                return

            number_loaded = len(X_loaded)
            if number_loaded >= self.number_instances:
                self.values = X_loaded[:self.number_instances]

                return
            else:
                print("There are too few instances on locally found dataset. Loading features from "
                      "cases located at path...")

    def generate_filename(self):
        """ Generates a filename based on the user settings """

        file_name = "X_" + self.feature_mode + "_" + self.dataset.name + "_C" + str(self.channels_range[0]) + "_" \
 
        file_name += str(self.channels_range[1])

        file_name += "_L" + str(self.levels_range[0]) + "_" + str(self.levels_range[1]) + "_T"

        if is_in(self.array_type, 'pos'):

            file_name += "pos"

        else:

            file_name += 'ori'

        if self.extra_dimension:
            file_name += "_ED"

        file_name += ".npy"

        return file_name

    def load_features_from_file(self):
        """ Determines if labels storage file exists and if so loads data"""

        file_name = self.generate_filename()

        if not os.path.exists(file_name):
            return []

        print("Features dataset was found on file:", file_name, "Loading...")
        return np.load(file_name)

    def save(self):

        file_name = self.generate_filename()

        print("Saving features to file:", file_name)
        np.save(file_name, self.values)

    def training_set(self):
        return self.values[:self.dataset.split_number]

    def validation_set(self):
        return self.values[self.dataset.split_number:]

    def summary(self):
        summary_text = [
            "Feature mode: " + self.feature_mode,
            "Input shape: " + str(self.feature_shape),
            "Input channels range: " + str(self.channels_range),
            "Input levels range: " + str(self.levels_range),
            "Input array type: " + self.array_type

        ]

        return summary_text


class Features1D(Features):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', extra_dimension=False):

        self.feature_mode = "1D_flat"

        self.extra_dimension = extra_dimension

        super().__init__(dataset, channels, levels, array_type)

        # the super class tries to load features from files. If it fails, the features are loaded from the dataset
        if self.values is None:
            self.values = self.generate_array(dataset, channels, levels, array_type, extra_dimension)

        if self.extra_dimension:
            self.feature_shape = self.values.shape[1:]
        else:
            self.feature_shape = self.values.shape[1]

        self.save()

    def generate_array(self, dataset, channels, levels, array_type, extra_dimension):

        instance_array_length = self.number_channels * self.number_levels
        X_1d = np.zeros([len(dataset.cases_list), instance_array_length])

        for i, instance in enumerate(dataset.core_instances):
            X_1d[i] = instance.crack_array_1d(channels=channels, levels=levels, array_type=array_type)

        # If the user requires the array to be configured for convolutional networks, then the array is reshaped
        if extra_dimension:
            X_shape = [X_1d.shape[0], X_1d.shape[1], 1]
            return X_1d.reshape(X_shape)

        # Else just return the normal numpy array
        else:
            return X_1d


class Features2D(Features1D):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', extra_dimension=False):

        self.feature_mode = "2D_multi"
        self.extra_dimension = extra_dimension

        Features.__init__(self, dataset, channels, levels, array_type)

        # the super class tries to load features from files. If it fails, the features are loaded from the dataset
        if self.values is None:
            self.values = self.generate_array(dataset, channels, levels, array_type, extra_dimension)

        self.feature_shape = self.values.shape[1:]

        self.save()

    def generate_array(self, dataset, channels, levels, array_type, extra_dimension):

        # Get the 1d feature array
        values_1d = Features1D.generate_array(self, dataset, channels, levels, array_type, extra_dimension)

        X_shape = [values_1d.shape[0], self.number_levels, self.number_channels]

        # If the user requires the array to be configured for convolutional networks, then the array is reshaped
        if extra_dimension:
            X_shape.append(1)

        return values_1d.reshape(X_shape)


class Features3D(Features):

    def __init__(self, dataset, levels='all', array_type='positions only'):

        self.feature_mode = "3D_multi"
        self.extra_dimension = False

        super().__init__(dataset, channels='all', levels=levels, array_type=array_type)

        # the super class tries to load features from files. If it fails, the features are loaded from the dataset
        if self.values is None:
            self.values = self.generate_array(dataset, levels, array_type)

        self.feature_shape = self.values.shape[1:]

        self.save()

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

        return X_3d_rs


class FeaturesFlat(Features3D):
    pass


class FeaturesConcentration1D(Features):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', extra_dimension=False):

        self.feature_mode = "Concentration1D"

        self.extra_dimension = extra_dimension

        super().__init__(dataset, channels, levels, array_type, 'inter')

        # the super class tries to load features from files. If it fails, the features are loaded from the dataset
        if self.values is None:
            self.values = self.generate_array(dataset, channels, levels, array_type, extra_dimension)

        if extra_dimension:
            self.feature_shape = self.values.shape[1:]
        else:
            self.feature_shape = self.values.shape[1]


        self.save()

    def generate_array(self, dataset, channels, levels, array_type, extra_dimension):

        # TODO make options for channels, array type etc.

        instance_array_length = self.number_channels
        X_1d = np.zeros([len(dataset.cases_list), instance_array_length])

        for i, instance in enumerate(dataset.core_instances):
            X_1d[i] = instance.channel_specific_cracks(levels)[1]

        # If the user requires the array to be configured for convolutional networks, then the array is reshaped
        if extra_dimension:
            X_shape = [X_1d.shape[0], X_1d.shape[1], 1]
            return X_1d.reshape(X_shape)

        # Else just return the normal numpy array
        else:
            return X_1d


class FeaturesConcentration2D(FeaturesConcentration1D):

    def __init__(self, dataset, channels='all', levels='all', array_type='positions only', extra_dimension=False):
        self.feature_mode = "Concentration2D"
        self.extra_dimension = extra_dimension

        Features.__init__(self, dataset, channels, levels, array_type, 'inter')

        # the super class tries to load features from files. If it fails, the features are loaded from the dataset
        if self.values is None:
            self.values = self.generate_array(dataset, channels, levels, array_type, extra_dimension)

        self.feature_shape = self.values.shape[1:]

        self.save()

    def generate_array(self, dataset, channels, levels, array_type, extra_dimension):
        # Get the 1d feature array

        values_1d = FeaturesConcentration1D.generate_array(self, dataset, channels, levels, array_type, extra_dimension)

        array_shape = [len(dataset.cases_list),
                       self.example_instance.inter_rows,
                       self.example_instance.inter_columns]

        if extra_dimension:
            array_shape.append(1)

        values_2d = np.zeros(array_shape)

        inter_rows = self.example_instance.inter_rows
        inter_columns = self.example_instance.inter_columns

        for instance_no in range(values_2d.shape[0]):

            print("\n ======== \n")

            i = 0
            instance_vals = values_1d[instance_no]

            for row in range(inter_rows):
                column_offset = self.example_instance.first_columns_row_interstitial[row]
                row_values = instance_vals[i:(i + (inter_columns - (column_offset * 2)))]
                i += len(row_values)

                values_2d[instance_no, row, column_offset:(inter_columns - column_offset)] = row_values

        return values_2d


class Labels:

    def __init__(self, dataset, channels='all', levels='all', result_time=50, result_column="1", result_type="max",
                 flat=True, load_from_file=True):
        self.dataset = dataset
        self.number_instances = len(dataset.cases_list)

        example_instance = dataset.core_instances[0]

        channels_range, levels_range = min_max_channels_levels(example_instance, channels, levels, 'inter')

        self.channels_range = channels_range
        self.levels_range = levels_range

        self.number_channels = channels_range[1] - channels_range[0]
        self.number_levels = levels_range[1] - levels_range[0]

        self.time = result_time

        self.column = result_column

        self.type = result_type

        self.flat = flat

        ##############################################################
        # ######## Attempt to load data from file #####################
        ##############################################################

        Y_loaded = self.load_labels_from_file()
        if len(Y_loaded) > 0:

            if not load_from_file:
                print("Labels dataset was found on disk. However, load_from_disk parameter is set to False. If you wish"
                      " to load this dataset, please set load_from_disk parameter to True")
                return

            number_loaded = len(Y_loaded)
            if number_loaded >= self.number_instances:
                self.values = Y_loaded[:self.number_instances]

                if len(Y_loaded.shape) > 2:
                    self.label_shape = Y_loaded.shape[1:]
                else:
                    self.label_shape = Y_loaded.shape[1]

                return
            else:
                print("Expanding label set with cases located at path...")
        else:
            number_loaded = 0

        ##############################################################
        # ######## Load data from path ################################
        ##############################################################

        # Load data from path
        array_dims = [self.number_instances]

        # If the result for all bricks is being returned
        if is_in(result_type, 'all'):
            # Flat 1D
            if flat:
                array_dims.append(self.number_channels * self.number_levels)

            # 2d
            else:
                array_dims.append(self.number_channels)
                array_dims.append(self.number_levels)

        # channel specific results (max, sum, average etc.)
        else:
            array_dims.append(self.number_channels)

        Y = np.zeros(array_dims)

        for i, instance in enumerate(dataset.core_instances):

            if i < number_loaded:
                Y[i] = Y_loaded[i]
            else:

                if is_in(channels, 'all') and is_in(levels, 'all'):
                    Y[i] = instance.get_result_at_time(result_time, result_columns=str(result_column),
                                                       result_type=result_type, flat=flat)

                else:
                    # Get the instance result - note that result_type is set to 'all and falt to false
                    instance_result = instance.get_result_at_time(result_time, result_columns=str(result_column),
                                                                  result_type='all', flat=False)

                    # the slice of the instance result array corresponding to the channels and levels required
                    instance_result_slice = instance_result[self.channels_range[0]:self.channels_range[1],
                                            self.levels_range[0]:self.levels_range[1]]

                    # if the result type is all
                    if is_in(result_type, 'all'):
                        if flat:
                            Y[i] = instance_result_slice.flatten()
                        else:
                            Y[i] = instance_result_slice

                    # channel specific result. One result per channel requested
                    else:

                        command = function_switch(result_type)

                        # iterate through each channel
                        for c, channel in enumerate(instance_result_slice):
                            Y[i, c] = command(channel)

        self.values = Y

        if flat:
            self.label_shape = Y.shape[1]
        else:
            self.label_shape = Y.shape[1:]

        file_name = self.generate_filename()
        print("Saving labels to file:", file_name)
        np.save(file_name, Y)

    def generate_filename(self):
        """ Generates a filename based on the user settings """

        # Check if label dataset exists
        if self.flat:
            flat_string = 'T'
        else:
            flat_string = 'F'

        file_name = "Y_" + self.dataset.name + "_C" + str(self.channels_range[0]) + "_" + str(self.channels_range[1])

        file_name += "_L" + str(self.levels_range[0]) + "_" + str(self.levels_range[1]) + "_T" + str(self.time)

        file_name += "_R" + str(self.column) + "_" + str(self.type) + "_" + flat_string + ".npy"

        return file_name

    def load_labels_from_file(self):
        """ Determines if labels storage file exists and if so loads data"""

        file_name = self.generate_filename()

        if not os.path.exists(file_name):
            return []

        print("Labels dataset was found on file:", file_name, "Loading...")
        return np.load(file_name)

    def training_set(self):
        return self.values[:self.dataset.split_number]

    def validation_set(self):
        return self.values[self.dataset.split_number:]

    def summary(self):

        summary_text = [
            "Output shape: " + str(self.label_shape),
            "Output channels range: " + str(self.channels_range),
            "Output levels range: " + str(self.levels_range),
            "Output time-frame: " + str(self.time),

            "Output result column: " + str(self.column),

            "Output result type: " + str(self.type),

        ]

        return summary_text