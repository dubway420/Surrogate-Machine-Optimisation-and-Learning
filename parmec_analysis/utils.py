import os
import inspect
import numpy as np
import re as remove
import pandas as pd
import pickle
import time
import sys

# Global Variables
split_separators = ["to", "To", "TO", ":", "-", "/", "_", ",", " "]


def get_id_from_filename(filename):
    """ Generates an ID tag from the name of a file specified """

    return os.path.splitext(os.path.basename(filename))[0]


def is_in(string, search_term, *args):
    """ This method tells the user if a search term is contained within another string.
    It's an extension of the 'in' qualifier, but does so for all capitalisations"""

    # Converts both input arguments to lowercase
    string_lowercase = string.lower()

    # If there are any optional arguments, convert those to lower case too
    terms_lowercase = [search_term.lower()]
    for term in args:
        if isinstance(term, str):
            terms_lowercase.append(term.lower())
        else:
            print("skipping term: ", term, "which is not a valid string")

    # checks if the lowercase search_string is in each string
    # If it is, return true, else false

    for search_term_lowercase in terms_lowercase:
        if search_term_lowercase in string_lowercase:
            return True

    return False


def directories_in_path(path):
    """For a given path, returns a list of directories contained within it"""

    cases = []

    for item in os.listdir(path):
        if is_in(item, '.') is False:
            cases.append(item)

    return cases


def cases_list(path_string):
    """ For a given directory, returns a list of instance cases, including the original path """

    cases = directories_in_path(path_string)

    case_list = []

    for base in cases:
        case_list.append(path_string + base + '/' + base)

    return case_list


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
            loaded = pickle.load(f)
            return loaded
    else:
        return False


def split_by_column_uniques(input_array, column_no):
    """ Recursively splits a 2D array by a list of columns"""

    # First, sort by the column number required
    sorted_array = input_array[input_array[:, column_no[0]].argsort()]

    # If there are no more columns to split by, return the array
    if len(column_no) == 1:
        return sorted_array

    uniques, indices, counts = np.unique(sorted_array[:, column_no[0]], return_index=True, return_counts=True)

    split_array = []

    # Cycle through each unique value in the column
    for index, count in zip(indices, counts):
        # takes a segment of the data array corresponding that contains
        core_slice = sorted_array[index:(index + count)]

        # Recursively split again by the next column of interest
        split_array.append(split_by_column_uniques(core_slice, column_no[1:]))

    return split_array


def string_splitter(string):
    """splits a string containing two numbers into two integers"""

    # Iterates through possible separators to see if they are contained in the input string
    for splitter in split_separators:

        # if it finds one of the separators in the string, it splits the string
        if splitter in string:
            components = string.split(splitter)

            try:

                # If the input spring can be split into exactly two parts, then the two parts are returned
                if len(components) == 2:
                    extracted_values = int(remove.sub("[^0-9]", "", components[0])), int(remove.sub("[^0-9]", "",
                                                                                                    components[1]))

                    # Ensures the returned values are in order
                    if extracted_values[1] >= extracted_values[0]:
                        return extracted_values
                    else:
                        return extracted_values[1], extracted_values[0]

            # if there's any problems with the input values to the above code, just let it continue to the default
            # output
            except ValueError:
                continue

    # If none of the split separators are found in the input string, sends a warning and returns default (0, 0)
    print('Warning: could not split string \"', string, '\" into two equal parts. Returning default output (0, 0)')
    return 0, 0


def features_and_or_labels(features_labels):
    """ Takes the user defined string 'features_labels' and returns a list of booleans dep"""
    vector = [False, False]

    if is_in(features_labels, 'both', 'all'):
        vector = [True, True]
    if is_in(features_labels, 'feat'):
        vector[0] = True
    if is_in(features_labels, 'lab'):
        vector[1] = True

    return vector


def retrieve_name(var):
    """ For a given variable, var, returns the variable name"""
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def read_output_file(filename, pandas=False):
    """ Wrapper which reads the parmec output file and returns a data structure"""

    if is_in(filename, '.csv'):
        return_array = pd.read_csv(filename, header=0, index_col=False)
    elif is_in(filename, '.xl'):
        return_array = pd.read_excel(filename, header=None, index_col=False)
    else:
        print("ERROR: Invalid input file type specified. No output array generated.")
        return None

    if pandas:
        return return_array
    else:
        return return_array.values


def index_array_fuel(case, ext='.csv'):
    """ Sorts an array by the values in a given column, then splits into sub arrays based on unique
    values in column """

    # The path to the zeroth file of the case - sorting is performed on this file because no movement has occurred yet
    zeroth_file = case + '.0' + ext

    # Recursively split and sort array - this effectively creates sub arrays for each core row, and then a further
    # sub array within that for each channel
    # Note: if there is cracking, the cracked bricks form their own 'channels' as they no longer
    # occupy the same x/y coordinate as the base channel
    split_array = split_by_column_uniques(read_output_file(zeroth_file), [-2, -3, -1])

    channels_core = []

    # Each item is effectively a row of channels within the core
    for item in split_array:

        channels_row = []

        # Each stack is either a fuel/interstitial channel (intact) or a stack of cracked bricks
        for stack in item:

            # if the stack has 13 or more items, then it is sorted into the list of interstitial bricks
            # it only stores the indices of each brick (column 6)
            if len(stack) == 7:
                channels_row.append(stack[:, 6])

        if len(channels_row) > 0:
            channels_core.append(channels_row)

    # Reverse the order of the indices - this reverses the order of the rows (which are switched from bottom to
    # top to visa-versa) but keeps the order of the columns (which go from left to right).
    reversed_core_arrays = channels_core[::-1]

    # ============================================================================================================
    # Flattening the arrays so that each sub array represents a channel
    flat_array = []

    # Iterate through each row, then through each channel in the row, appending that channel's dataset to the inner
    # list. Each element in this array then corresponds to a channel
    for row in reversed_core_arrays:
        for channel in row:
            flat_array.append(channel.astype(int))

    return flat_array


def index_array_interstitial(case, ext='.csv'):
    """ Sorts an array by the values in a given column, then splits into sub arrays based on unique
    values in column """

    # The path to the zeroth file of the case - sorting is performed on this file because no movement has occurred yet
    zeroth_file = case + '.0' + ext

    # Recursively split and sort array - this effectively creates sub arrays for each core row, and then a further
    # sub array within that for each channel
    # Note: if there is cracking, the cracked bricks form their own 'channels' as they no longer
    # occupy the same x/y coordinate as the base channel
    split_array = split_by_column_uniques(read_output_file(zeroth_file), [-2, -3, -1])

    channels_core = []

    # Each item is effectively a row of channels within the core
    for item in split_array:

        channels_row = []

        # Each stack is either a fuel/interstitial channel (intact) or a stack of cracked bricks
        for stack in item:

            # if the stack has 13 or more items, then it is sorted into the list of interstitial bricks
            # it only stores the indices of each brick (column 6)
            if len(stack) >= 13:
                channels_row.append(stack[:, 6])

        if len(channels_row) > 0:
            channels_core.append(channels_row)

    # Reverse the order of the indices - this reverses the order of the rows (which are switched from bottom to
    # top to visa-versa) but keeps the order of the columns (which go from left to right).
    reversed_core_arrays = channels_core[::-1]

    # ============================================================================================================
    # Flattening the arrays so that each sub array represents a channel
    flat_array = []

    # Iterate through each row, then through each channel in the row, appending that channel's dataset to the inner
    # list. Each element in this array then corresponds to a channel
    for row in reversed_core_arrays:
        for channel in row:
            flat_array.append(channel.astype(int))

    return flat_array


################################################################
# ################ DATA EXTRACTION FUNCTIONS ###################
################################################################

def absolute_sum(array):
    """ return the sum of the absolute of every value in an array"""

    absolute_array = np.absolute(array)
    return sum(absolute_array)


def max_absolute(array):
    """ return the greatest value in absolute terms"""

    absolute_array = np.absolute(array)
    return max(absolute_array)


def floor_zero_sum(array):
    return sum(floor_zero_all(array))


def floor_zero_all(array):
    return np.maximum(array, 0)


def return_all(array):
    return np.array(array).flatten()


def function_switch(result_type):
    if result_type.isnumeric():
        command = return_numeric_function(result_type)
    elif is_in(result_type, "max"):
        command = np.max
    elif is_in(result_type, "min"):
        command = np.min
    elif is_in(result_type, "sum"):
        command = np.sum
    elif is_in(result_type, "mean"):
        command = np.mean
    elif is_in(result_type, "med"):
        command = np.median
    elif is_in(result_type, "abs") and is_in(result_type, "sum"):
        command = absolute_sum
    elif is_in(result_type, "floor zero sum"):
        command = floor_zero_sum
    elif is_in(result_type, "floor zero all"):
        command = floor_zero_all
    elif is_in(result_type, "abs") and is_in(result_type, "max"):
        command = max_absolute
    elif is_in(result_type, "all"):
        command = return_all
    # TODO min vs max function
    else:
        command = np.sum

    return command


def return_numeric_function(n):
    """ This function returns a function which extracts the value n from an array """

    int_n = int(n)

    def level(*args):
        for arg in args:
            return arg[int_n - 1]

    return level


##################################
# ##### label set manipulation ####
##################################

def convert_all_to_channel_result(Y, result_type, no_channels, no_levels):
    """ This converts a flat array that contains a value for all bricks to one where there's a value per channel """

    # The command to use to convert the array to
    command = function_switch(result_type)

    # Initialise array to be returned
    Y_converted = np.zeros([Y.shape[0], no_channels])

    for i, case in enumerate(Y):

        case_reshaped = case.reshape(no_channels, no_levels)
        for c, channel in enumerate(case_reshaped):
            Y_converted[i, c] = command(channel)

    return Y_converted


def convert_case_to_channel_result(y, result_type, no_channels, no_levels):
    # The command to use to convert the array to
    command = function_switch(result_type)

    y_converted = np.zeros(no_channels)
    y_reshaped = y.reshape(no_channels, no_levels)

    for c, channel in enumerate(y_reshaped):
        y_converted[c] = command(channel)

    return y_converted


def plot_names_title(experiment, iteration):
    # TODO replace everything that can be with experiment

    # Unpack objects used by dataset
    dataset = experiment.dataset
    features = experiment.features
    labels = experiment.labels

    line = ""
    file_name = dataset.name + "_" + experiment.name

    line += "Feature Channels/Levels: " + str(features.channels_range[0] + 1) + "-" + \
            str(features.channels_range[1]) + ", " + str(features.levels_range[0] + 1) + "-" + \
            str(features.levels_range[1]) + ", Array Type: " + str(features.array_type)

    line += "\n"

    file_name += "_C" + str(features.channels_range[0]) + "_" + str(features.channels_range[1]) + "_" \
                 + "L" + str(features.levels_range[0]) + "_" + str(features.levels_range[1]) + "_"

    # Label data

    line += "Label Channels/Levels: " + str(labels.channels_range[0] + 1) + "-" + \
            str(labels.channels_range[1]) + ", " + str(labels.levels_range[0] + 1) + "-" + \
            str(labels.levels_range[1]) + ", Time: " + str(labels.time) + ", Column: " + \
            str(labels.column) + ", Result Type: " + str(labels.type)

    file_name += str(labels.channels_range[0]) + "_" + str(labels.channels_range[1]) + "_" + \
                 str(labels.levels_range[0]) + "_" + str(labels.levels_range[1]) + "_" + \
                 str(labels.time) + "_" + str(labels.column) + "_" + str(labels.type) + \
                 "I" + str(iteration)

    file_name += ".png"

    return line, file_name


def load_results(trial_name):
    loaded = load_data_from_file(trial_name)

    # check if a test with the test_name has been done. If so, the file should exist -
    if loaded:

        results_dict = loaded

    # - if not, create a dictionary
    else:
        results_dict = {}

    return results_dict


def row_offset_channels(row):
    row_offset = 0
    channels_in_row = 19

    if row == 3 or row == 15:
        row_offset = 1
        channels_in_row = 17

    if row == 2 or row == 16:
        row_offset = 2
        channels_in_row = 15

    if row == 1 or row == 17:
        row_offset = 3
        channels_in_row = 13

    if row == 0 or row == 18:
        row_offset = 4
        channels_in_row = 11

    return row_offset, channels_in_row


def result3d(result):
    assert len(result.shape) > 1, "Ensure that get_result_at_time method is called with the flat=False argument"

    result_3d = np.zeros([result.shape[1], 19, 19])

    for row in range(19):

        result_index = 0

        row_offset, channels_in_row = row_offset_channels(row)

        for channel in range(channels_in_row):
            column = channel + row_offset
            result_3d[:, row, column] = result[result_index]

            result_index += 1

    return result_3d


def result1d(result_3d):
    pass


# MACHINE LEARNING STUFF *********************

def experiment_assignment_validation(experiments, experiment_number):
    """ Check that it has selected a valid experiment number and assign experiment if so.
    :param experiments: a list of objects of the experiments class
    :param experiment_number: an experiment number. The purpose of this function to ensure that the number is between
    0 and len(experiments) -1
    :return: an experiment object
    """

    try:

        experiment_selected = experiments[int(experiment_number)]

    except IndexError:

        print("You have entered an experiment number,", experiment_number, ", which is invalid. Please enter an integer"
                                                                           " corresponding to one of the following: ")

        for i, exp in enumerate(experiments):
            print(i, exp.name)
        sys.exit()

    return experiment_selected


def folder_validation(trial_name):
    cwd = os.getcwd()

    try:

        os.mkdir(cwd + "/" + trial_name)

    except FileExistsError:

        pass


def experiment_iteration(exp_name, trial_name, ext=".ind"):
    results_file_name = trial_name + ext

    results_dict = load_results(results_file_name)

    if exp_name in results_dict:

        exp_i = len(results_dict[exp_name])

    else:
        exp_i = 0
        results_dict[exp_name] = []

    results_dict[exp_name].append([])

    with open(results_file_name, 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return exp_i


def save_results(exp_name, trial_name, exp_i, exp_result, ext=".ind", attempts=0):
    results_file_name = trial_name + ext

    assigned = False

    while attempts < 4 and not assigned:

        try:
            results_dict = load_results(results_file_name)

            results_dict[exp_name][exp_i] = exp_result

            with open(results_file_name, 'wb') as f:
                pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            assigned = True

        except KeyError:

            attempts += 1

            print("Error whilst saving results. Will wait 30 second then try again. "
                  "It is possible the repository is in use.")
            print("Attempts made:", attempts)
            time.sleep(30)
            save_results(exp_name, trial_name, exp_i, exp_result, attempts=attempts)
