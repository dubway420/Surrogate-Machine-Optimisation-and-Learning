import os
import numpy as np
import re as remove
import pandas as pd

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

        if len(channels_row) > 0: channels_core.append(channels_row)

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

        if len(channels_row) > 0: channels_core.append(channels_row)

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
