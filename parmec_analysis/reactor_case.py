# -*- coding: utf-8 -*-

# Created by Huw Rhys Jones - huw.jones@postgrad.manchester.ac.uk - August 2019
# 
# *******************CoreInstance*******************
# 
# This class creates an object representing a single whole core crack pattern as generated by the spreadsheet cracked_
# core_generator_MODIFIED (1)as created by Atkins and modified by Huw Rhys Jones. The CoreInstance class takes the
# output spreadsheets from (1) and generates an representative object. The object can be used to transform the 3D
# dimensional array representing the whole core crack pattern from (1) into various other forms.
# For example, if the user is interested in the local crack pattern adjacent to a particular channel,
# this can be obtained via the .get_channel_crack_array_fuel
# method. 
# 
# The class object of (1) can be coupled with the corresponding output of parmec to give matching inputs outputs. 
#
# The intention of this class is that it makes the production and training of machine learning models easier. 
#
# ====================================================================
# TODO
# 1.) Cracks orientated towards channel in question
#
# ====================================================================
# Changelog 
# 28/08/19 - fixing tls.is_in subroutine, add positions functionality to cracks per level, complete distance method
# 2/09/19 - Completed method returning cracks per layer
# 4/09/19 - Started creating methods for extracting output/label data

import numpy as np
import pandas as pd
import math
from parmec_analysis import utils as utils


# TODO split parse into FeaturesCase and LabelsCase


# TODO Convert to FullCase which will inherit the functions of FeaturesCase and LabelsCAse
class Parse:
    """ Instance representing the entire core, including crack positions in 3D array and post earthquake
    array of displacements"""

    # ==========================================================
    # -------------------------SET UP---------------------------
    # ==========================================================

    def __init__(self, case, fuel_levels=7, inter_levels=13, core_rows=18, core_columns=18,
                 fuel_channels=284, interstitial_channels=321, padding=2, augmentation=None):
        """Assign ID. Initialise or extract data arrays """

        # The number of levels represents the vertical stack height in the core i.e. the number of bricks stacked in a
        # channel. These have traditionally been called 'layers' in AGR terminology. However, to avoid confusion with
        # the machine learning term 'layers' (as in layers of neural network) this term has been renamed 'levels'.
        # Additionally, the bricks delineating a channel are also now referred to as a 'layer'.

        # Core dimensions
        # =============
        self.core_rows = core_rows
        self.core_columns = core_columns

        self.inter_rows = core_rows + 1
        self.inter_columns = core_columns + 1

        self.core_levels = fuel_levels
        self.inter_levels = inter_levels

        # The number of fuel channels in the core. This isn't simply the product of row * columns because of peripheral
        # channels.
        self.fuel_channels = fuel_channels

        # The number of interstitial channels in the core. There is one extra column and row than there are for fuel
        # channels.
        self.interstitial_channels = interstitial_channels
        # =============

        # Uncracked bricks are represented by a -1, with cracked bricks being represented by an integer 1 -4 depending
        # on their orientation. Reflector and exterior channels are represented by a 0 as default. The depth of the
        # layer of padding with reflector/exterior bricks is 2 by default, but can be specified by the user.

        self.padding = padding

        self.augmentation = augmentation

        # The first number of each row
        self.first_numbers_row_fuel = [1, 11, 23, 37, 53, 71, 89, 107, 125, 143, 161, 179, 197, 215, 233, 249, 263, 275]
        self.first_numbers_row_interstitial = [1, 12, 25, 40, 57, 76, 95, 114, 133, 152, 171, 190, 209, 228, 247, 266,
                                               283, 298, 311]

        # The first column of each row
        self.first_columns_row_fuel = [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
        self.first_columns_row_interstitial = [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]

        # print("\n====")
        # print("Generating core instance")

        self.case = case

        # Set id and generate cor array of cracking
        # =============
        self.id = None
        # If there's no crack file it just sets a default id and blank crack array. If there is a crack
        # file then it extracts it from the Excel sheet
        if case == "":
            print("No crack file detected. Assigning default values.")
            self.set_id("DEFAULT_ID")
            self.crack_array = self.base_crack_array()
        else:
            self.set_id(utils.get_id_from_filename(case))

            try:
                self.set_crack_array()
            except FileNotFoundError:
                self.crack_array = self.base_crack_array()

        # =============
        # Results stuff

        # print("Calculating indices of channel items from results file")

        self.fuel_indices = utils.index_array_fuel(case)  # TODO WARNING - THIS ONLY CURRENTLY WORKS FOR INTACT CASES
        self.results_indices = utils.index_array_interstitial(case)

        self.results_3D_array = None

        if augmentation:
            self.apply_augmentation()

        # =============

        # print("Initialisation complete")
        # print("====\n")

    def set_id(self, id_value):
        """ set the id of the core  """

        self.id = id_value
        # print("Setting id as " + self.id)

    def get_id(self):
        """ set the id of the core  """

        return self.id

    def base_crack_array(self):
        return np.zeros(
            shape=(self.core_levels, self.core_rows + (self.padding * 2), self.core_columns + (self.padding * 2)))

    def set_crack_array(self):
        """ Assign the 3D crack array """

        array = self.base_crack_array()

        filename = self.case + '.xlsx'

        # cycles through all core levels, extracting the 2D array and assigning it to the 3D array
        for i in range(0, self.core_levels):
            array[i][:] = np.pad(np.array(pd.read_excel(filename, sheet_name=i, header=None, index_col=False,
                                                        engine='openpyxl')),
                                 [(self.padding, self.padding), (self.padding, self.padding)], mode='constant',
                                 constant_values=0)

        # print("Writing cracked core pattern to array")
        self.crack_array = array

    # ==========================================================
    # ---------------------FEATURE EXTRACTION-------------------
    # ==========================================================

    def get_crack_array(self, array_type="orientations", levels='all', print_mode=False):
        """ Returns the crack array. Can be of type 'orientations' or 'positions' """

        # The array is simply stored as self.crack array.
        # This subroutine just determines if the user wants the array as containing
        # orientations (base array) or just positions.

        # The user can specify the type of array needed via a string argument
        # ('orientations' or 'positions'). Alternatively, the user can specify
        # their desired type by integer (0 or 1 for orientations and positions, respectively)

        # If the user specifies 'orientations' then the base array is returned.
        # If 'positions' specified, then all integers in the array >1 are changed to 1

        # Type
        # --------------------------------------------------
        # Handles if user specifies output type by string

        # As default generates an array of all zeros the size of the core
        array = self.base_crack_array()

        if isinstance(array_type, str):

            # Can handle slight misspellings or capitalisation

            # Orientations - just returns the base array
            if utils.is_in(array_type, "ori"):
                array = self.crack_array

            # Positions - makes the maximum of the array 1
            elif utils.is_in(array_type, "pos"):
                array = np.where(self.crack_array > 1, 1, self.crack_array)
            else:
                print("WARNING: ambiguous output type specified. Defaulting to orientations.")
                array = self.crack_array

        # Handles if user specifies output type by integer
        elif isinstance(array_type, int):

            # Orientations
            if array_type == 0:
                array = self.crack_array

            # Positions
            elif array_type > 0:
                array = np.where(self.crack_array > 1, 1, self.crack_array)

            else:
                print("WARNING: ambiguous output type specified. Defaulting to orientations.")
                array = self.crack_array

        # --------------------------------------------------

        # Interprets the 'levels' argument
        min_level, max_level = self.parse_level_argument(levels)

        if print_mode:
            return array[min_level:max_level].astype(int)

        return array[min_level:max_level]

    def get_channel_crack_array(self, channel_no, array_size=2, array_type='orientations', levels='all',
                                channel_type='fuel'):
        """ Returns a sub array centered around a user defined channel."""

        pad = self.padding
        if array_size > pad:
            print("WARNING: The requested array size (" + str(array_size) + ") exceeds the padding (" + str(
                pad) + ") that surrounds the core array.")
            print("This configuration risks the channel array going outside the bounds of the parent array.")
            print("Consider increasing the padding or reducing the array size.\n")

        # Get the array representing the whole core
        core_array = self.get_crack_array(array_type)

        # For the given channel number (1 - 284), returns the 2D coordinates (row, column) of it in the whole core array
        row_origin, column_origin = self.channel_coordinates(channel_no, channel_type)

        min_level, max_level = self.parse_level_argument(levels)

        # TODO come up with something more elegant than this.
        fudge = 0
        if array_size == 0: fudge = 1

        if utils.is_in(channel_type, "fuel"):
            row_start = row_origin - array_size
            row_end = row_origin + array_size + 1
            column_start = column_origin - array_size
            column_end = column_origin + array_size + 1
        else:
            row_start = max(row_origin - array_size, 0)
            row_end = row_origin + array_size + fudge

            column_start = max(column_origin - array_size, 0)
            column_end = column_origin + array_size + fudge

        channel_crack_array = core_array[min_level:max_level, row_start:row_end,
                              column_start:column_end]

        return channel_crack_array

    def get_cracks_per_level(self, channel='core', size=2, array_type='orientations', levels='all', fraction=False,
                             channel_type='fuel', quiet=False):
        """ Returns an array stating the number of cracks in each level. If the array is of the orientations type,
        the array is 2D - the first dimension specifies crack orientation, the second specifies level"""

        array = self.channel_array_argument_wrapper(channel, size, array_type, levels, channel_type, quiet)
        cracks_count = []

        # Cycle through levels of the core and take the 2D array
        for level in array:

            # Initialise array
            counts_per_orientation = [0, 0, 0, 0]

            # 2D array of orientations and the counts of it
            orientations, counts = np.unique(level, return_counts=True)

            # For each pair orientation/count
            for orientation, count in zip(orientations, counts):

                # Excludes -1 (uncracked) and 0 (peripheral)
                if orientation > 0:
                    # assigns the number of counts to the correct orientation
                    counts_per_orientation[int(orientation) - 1] = count

            # Append the layer array
            cracks_count.append(counts_per_orientation)

        if np.amax(array) > 1:
            return np.array(cracks_count)
        else:
            return np.array(cracks_count)[:, 0]

    def get_cracks_per_layer(self, channel='core', size=2, array_type='orientations', levels='all',
                             channel_type='fuel', inclusive=False):

        cracks_per_layer = []

        for i in range(0, size + 1):

            cracks_total = np.sum(self.get_cracks_per_level(channel, i, array_type, levels, False, channel_type,
                                                            quiet=True), axis=0)

            if i == 0:
                cracks_per_layer.append(cracks_total)
            else:
                if not inclusive:
                    cracks_per_layer.append(np.subtract(cracks_total, cracks_per_layer[i - 1]))
                else:
                    cracks_per_layer.append(cracks_total)

        return np.array(cracks_per_layer)

    def crack_array_1d(self, channels='all', array_type='orientations', levels='all'):

        # Gets the instance crack array
        crack_array_3d = self.channel_array_argument_wrapper(array_type=array_type, levels=levels, quiet=True)

        # Removes all padding from the array
        crack_array_no_padding = crack_array_3d[crack_array_3d != 0]

        # If all channels are requested, simply return the full array
        if utils.is_in(channels, "all") or utils.is_in(channels, "core"):
            return crack_array_no_padding

        # If not, try and work out which channels are required

        # Get the channel range. It's fuel array type because interstitial bricks don't crack
        min_channel, max_channel = self.parse_channel_argument(channels, 'fuel')
        number_of_channels = max_channel - min_channel

        # Get the range of levels
        min_level, max_level = self.parse_level_argument(levels)
        number_of_levels = max_level - min_level

        # Create a new array the size
        output_array_size = number_of_levels * number_of_channels
        crack_array_user = np.zeros(output_array_size)

        for i in range(number_of_levels):
            input_start = (i * self.fuel_channels) + min_channel
            input_end = (i * self.fuel_channels) + max_channel

            output_start = i * number_of_channels
            output_end = output_start + number_of_channels

            crack_array_user[output_start:output_end] = crack_array_no_padding[input_start:input_end]

        return crack_array_user

    def channel_specific_cracks(self, levels="all"):
        """ Return the number of cracks local to each channel"""

        channel_type = 'inter'

        counts_local, counts_adjacent, counts_outer = [], [], []

        # Iterate through channels
        for i in range(1, self.last_channel(channel_type=channel_type) + 1):
            local, adjacent, outer = self.get_cracks_per_layer(str(i), array_type='pos', channel_type=channel_type,
                                                               inclusive=True, levels=levels)

            counts_local.append(local)
            counts_adjacent.append(adjacent)
            counts_outer.append(outer)

        return counts_local, counts_adjacent, counts_outer

    # ==========================================================
    # ----------------------PROCESSING--------------------------
    # ==========================================================

    def get_result_at_time(self, time_index=0, ext='.csv', result_columns='all', result_type="max", flat=False):
        """ Results results array for a particular time frame"""

        case_path = self.case

        no_interstitial_channels = self.interstitial_channels
        no_interstitial_levels = self.inter_levels

        # These are the indices of interstitial bricks from the results array
        indices = self.results_indices

        time_file = case_path + '.' + str(time_index) + ext

        # Get the unsorted array corresponding to the specified user time_frame
        time_array_base = utils.read_output_file(time_file)

        # ==============================================
        number_of_output_metrics = len(time_array_base[0])

        if utils.is_in(result_columns, "all"):
            min_column = 0
            max_column = number_of_output_metrics

        elif result_columns.isnumeric():

            min_column = int(result_columns) - 1
            max_column = int(result_columns)

        else:
            components = utils.string_splitter(result_columns)
            min_column = components[0] - 1
            max_column = components[1]

        # Handles if a column out of range is specified
        if max_column > number_of_output_metrics:
            print('Error: requested data column:', max_column, ", is out of range. Defaulting to last column: ",
                  number_of_output_metrics)
            min_column = number_of_output_metrics - 1
            max_column = number_of_output_metrics

        elif min_column < 0:
            print('Error: requested data column:', min_column + 1, ", is out of range. Defaulting to first column.")
            min_column = 0
            max_column = 1

        # ==============================================

        command = utils.function_switch(result_type)

        time_array_user_columns = time_array_base[:, min_column:max_column]

        # ==============================================
        # Sorting the results file into sub arrays, each corresponding to a channel

        # Give numpy array length of interstitial channels
        result_array_dims = [no_interstitial_channels]

        # If all of the results in a channel are requested, then the array is made 2D
        if utils.is_in(result_type, "all"):
            result_array_dims.append(no_interstitial_levels)

        results_at_time_channel_sorted = np.zeros(result_array_dims)

        # For each sub array in the indices array, extract the corresponding bricks
        for i, channel in enumerate(indices):
            # TODO - if there's more than one column, takes the by column function

            channel_result = command(time_array_user_columns[channel][0:self.inter_levels])

            # # # One of the channels has more than 13 bricks
            # if utils.is_in(result_type, "all"):
            #     channel_result = channel_result[0:13]

            results_at_time_channel_sorted[i] = channel_result

        if flat:
            return results_at_time_channel_sorted.flatten()

        return results_at_time_channel_sorted

    def get_fuel_result_at_time(self, time_index=0, ext='.csv', result_columns='all', result_type="max"):
        """ Results results array for a particular time frame"""

        case_path = self.case
        indices = self.fuel_indices

        time_file = case_path + '.' + str(time_index) + ext

        # Get the unsorted array corresponding to the specified user time_frame
        time_array_base = utils.read_output_file(time_file)

        # ==============================================
        number_of_output_metrics = len(time_array_base[0])

        if utils.is_in(result_columns, "all"):
            min_column = 0
            max_column = number_of_output_metrics

        elif result_columns.isnumeric():

            min_column = int(result_columns) - 1
            max_column = int(result_columns)

        else:
            components = utils.string_splitter(result_columns)
            min_column = components[0] - 1
            max_column = components[1]

        # Handles if a column out of range is specified
        if max_column > number_of_output_metrics:
            print('Error: requested data column:', max_column, ", is out of range. Defaulting to last column: ",
                  number_of_output_metrics)
            min_column = number_of_output_metrics - 1
            max_column = number_of_output_metrics

        elif min_column < 0:
            print('Error: requested data column:', min_column + 1, ", is out of range. Defaulting to first column.")
            min_column = 0
            max_column = 1

        # ==============================================

        if utils.is_in(result_type, "max"):
            command = np.max
        elif utils.is_in(result_type, "min"):
            command = np.min
        elif utils.is_in(result_type, "sum"):
            command = np.sum
        elif utils.is_in(result_type, "mean"):
            command = np.mean
        elif utils.is_in(result_type, "med"):
            command = np.median
        elif utils.is_in(result_type, "abs") and utils.is_in(result_type, "sum"):
            command = utils.absolute_sum
        elif utils.is_in(result_type, "all"):
            command = utils.return_all
        # TODO min vs max function
        else:
            command = np.sum

        time_array_user_columns = time_array_base[:, min_column:max_column]

        # ==============================================
        # Sorting the results file into sub arrays, each corresponding to a channel
        time_array_sorted = []

        # For each sub array in the indices array, extract the corresponding bricks
        for channel in indices:
            # TODO - if there's more than one column, takes the by column function

            channel_result = command(time_array_user_columns[channel])
            time_array_sorted.append(channel_result)

        return time_array_sorted

    def result_time_history(self, result="1", result_type="max", time_steps=271):
        """Gets a result time history for one case i.e. one result at each time"""

        time_history = []

        for time in range(1, time_steps + 1):
            time_history.append(self.get_result_at_time(time, result_type=result_type, result_columns=str(result)))

        return time_history

    def get_brick_xyz_positions(self, include='xyz', channels_only=1, channel_type='interstitial'):

        # Allows the user to request a value for each channel only or every brick
        if channels_only == 1:
            result_type = "max"
        else:
            result_type = "all"

        # Fuel or interstitial
        if utils.is_in(channel_type, 'inter'):
            command = self.get_result_at_time
        else:
            command = self.get_fuel_result_at_time

        return_array = []

        for letter in include:
            result_column = str(ord(letter) - 85)
            return_array.append(np.array(command(time_index=0, result_columns=result_column, result_type=result_type)))

        return return_array

    def channel_coordinates(self, number, channel_type="fuel"):
        """ For a given channel integer, returns a tuple containing the row and column coordinates of its location"""

        # Identify requested channel type (fuel or interstitial) then assign the correct variables
        # Variables include the number of channels of that type and the first channel number on each row

        # if fuel
        if utils.is_in(channel_type, "fuel"):
            number_of_channels = self.fuel_channels
            first_numbers_row = self.first_numbers_row_fuel
            first_columns_row = self.first_columns_row_fuel

        # if interstitial
        elif utils.is_in(channel_type, "inter"):
            number_of_channels = self.interstitial_channels
            first_numbers_row = self.first_numbers_row_interstitial
            first_columns_row = self.first_columns_row_interstitial

        # if channel type not detected, default to fuel
        else:
            print("ambiguous type - returning fuel")
            number_of_channels = self.fuel_channels
            first_numbers_row = self.first_numbers_row_fuel
            first_columns_row = self.first_columns_row_fuel

        # Checks if number is out of range
        if number > number_of_channels or number < 1:
            print("ERROR: Channel number specified is out of range (1 - " + str(self.fuel_channels) + ")")
            return 0, 0

        # Padding on either side of each axis
        pad = self.padding

        # Cycles through each row number
        for i, row_number in enumerate(first_numbers_row):

            # If the sub gets to the final row
            if i == len(first_numbers_row) - 1:

                row = i + pad
                column = number - row_number + first_columns_row[i] + pad
                return row, column

            # Checks if the channel number is between the first number of the current
            # row and the first number of the next
            elif first_numbers_row[i] <= number < first_numbers_row[i + 1]:

                row = i + pad
                column = number - row_number + first_columns_row[i] + pad
                return row, column

    def channel_array_argument_wrapper(self, channel='core', size=2, array_type='orientations', levels='all',
                                       channel_type="fuel", quiet=False, ):
        """Wrapper for crack array subroutines. Takes the channel argument and returns the correct array."""

        # Handles if the user enters a string as input argument
        if isinstance(channel, str):

            # If the string is a number, convert it to an int
            if channel.isnumeric():

                channel = int(channel)

            # Otherwise, returns whole core array
            else:
                if not quiet: print("Returning array for whole core.")
                return self.get_crack_array(array_type, levels)

        # Is the number in the range of channels?
        if 0 < int(channel) <= self.last_channel(channel_type):
            if not quiet: print("Producing cracks per level array for channel " + str(channel) + ".")
            return self.get_channel_crack_array(int(channel), size, array_type, levels, channel_type)

        # If the channel specified is out of range, it just defaults to whole core
        else:
            if not quiet: print("specified channel is outside of range of channels (1 - " + str(
                self.last_channel(channel_type)) + "). Returning array for whole core.")
            return self.get_crack_array(array_type, levels)

    def last_channel(self, channel_type='fuel'):
        """ Returns the channel number of the last channel """

        # Depending on whether fuel or interstitial type is specified, sets the last channel i.e. the number of channels
        # of that type (284 for fuel, 321 interstitial by default)
        if utils.is_in(channel_type, "fuel"):
            return self.fuel_channels
        else:
            return self.interstitial_channels

    def top_level(self, channel_type='fuel'):
        """ Returns the level number of the top level """

        # Depending on whether fuel or interstitial type is specified, sets the top level i.e. the number of levels
        # of that type (7 for fuel, 13 interstitial by default)
        if utils.is_in(channel_type, "fuel"):
            return self.core_levels
        else:
            return self.inter_levels

    def parse_level_argument(self, levels, channel_type='fuel'):
        """ Takes a string or integer as input and returns a tuple (min_level, max_level)
        stating the range of levels required by the user"""

        # Gets the highest level depending
        top_level = self.top_level(channel_type)

        # Handles if user specifies levels by string
        if isinstance(levels, str):

            # if the string represents a number
            if levels.isnumeric():
                components = int(levels) - 1, int(levels)

            # Can handle slight misspellings or capitalisation
            # DEFAULT return all levels
            elif utils.is_in(levels, "all"):
                return 0, top_level

            # Top level
            elif utils.is_in(levels, "top"):
                return top_level - 1, top_level

            # Bottom level
            elif utils.is_in(levels, "bot"):
                components = 0, 1

            # If the user specifies something else, it tries to work out what it is

            else:

                split_items = utils.string_splitter(levels)
                components = split_items[0] - 1, split_items[1]

            # check if component[0] (min_level) is in acceptable range
            if 0 <= int(components[0]) < top_level:
                min_level = int(components[0])
            else:
                print("WARNING: Specified lower bound level number (" + str(components[0]).strip()
                      + ") is outside of the acceptable range. Setting to default (1)")
                min_level = 0

            # check if component[1] (max_level) greater than min level and lower than the
            # maximum number of levels
            if min_level < int(components[1]) <= top_level:
                max_level = int(components[1])
            else:
                print("WARNING: Specified upper bound level number (" + str(components[1]).strip()
                      + ") is outside of the acceptable range. Setting to default (" +
                      str(top_level) + ")")
                max_level = top_level

        # Handles if user specifies level type by integer
        # In this case, the min and max level is the same - it is assumed the user is interested in just a single level
        elif isinstance(levels, int):

            # Checks that level is in the region 1 - 7
            if levels < 1 or levels > top_level:
                print("Invalid levels argument specified (" + str(levels) + "). Setting levels to default (1 - " + str(
                    top_level) + ").")
                min_level = 0
                max_level = top_level
            else:
                min_level = levels - 1
                max_level = levels

        # If it can't figure out what the user wants, it just returns all levels
        else:
            print("Invalid levels argument specified (" + str(levels) + "). Setting levels to default (1 - " + str(
                top_level) + ").")
            min_level = 0
            max_level = top_level

        return min_level, max_level

    def parse_channel_argument(self, channels, channel_type="fuel"):
        """ Takes a string or integer as input and returns a tuple (min_channel, max_channel)
        stating the range of levels required by the user"""

        # TODO the value returned for min_channel is not zero based. Also see 334 and 335.

        # Depending on whether fuel or interstitial type is specified, sets the last channel i.e. the number of channels
        # of that type (284 for fuel, 321 interstitial by default)
        last_channel = self.last_channel(channel_type)

        # Handles if user specifies levels by string
        if isinstance(channels, str):

            if channels.isnumeric():
                min_channel = int(channels) - 1
                max_channel = int(channels)

            # Can handle slight misspellings or capitalisation
            # DEFAULT return all channels
            elif utils.is_in(channels, "all") or utils.is_in(channels, "core"):
                min_channel = 0
                max_channel = last_channel

            # Last channel
            elif utils.is_in(channels, "last"):
                min_channel = last_channel - 1
                max_channel = last_channel

            # First channel
            elif utils.is_in(channels, "first"):
                min_channel = 0
                max_channel = 1

            # If the user specifies something else, it tries to work out what it is

            else:

                components = utils.string_splitter(channels)

                # check if component[0] (min_channel) is in acceptable range
                if 1 <= int(components[0]) <= last_channel:
                    min_channel = int(components[0]) - 1
                else:
                    print("WARNING: Specified lower bound channel number (" + str(components[0]).strip()
                          + ") is outside of the acceptable range. Setting to lower bound default.")
                    min_channel = 0

                # check if component[1] (max_channel) greater than min level and lower than the
                # maximum number of levels
                if min_channel <= int(components[1]) <= last_channel:
                    max_channel = int(components[1])
                else:
                    print("WARNING: Specified upper bound level number (" + str(components[1]).strip()
                          + ") is outside of the acceptable range. Setting to default (" +
                          str(last_channel) + ")")
                    max_channel = last_channel

        # Handles if user specifies level type by integer
        # In this case, the min and max level is the same - it is assumed the user is interested in just a single level
        elif isinstance(channels, int):

            # Checks that level is in the range 1 - last channel (284/321 for fuel/interstitial, respectively)
            if channels < 1 or channels > last_channel:
                print("Invalid levels argument specified (" + str(channels) + "). Setting levels to default (1 - " +
                      str(last_channel) + ").")
                min_channel = 0
                max_channel = last_channel
            else:
                min_channel = channels - 1
                max_channel = channels

        # If it can't figure out what the user wants, it just returns all levels
        else:
            print("Invalid levels argument specified (" + str(channels) + "). Setting levels to default (1 - " + str(
                last_channel) + ").")
            min_channel = 0
            max_channel = last_channel

        # Added + 1 to max channel otherwise iterator will miss last channel
        return min_channel, max_channel

    def distance_from_centre(self, channel_no, channel_type="fuel", rows="", columns=""):
        """ Calculates the planar distance from the centre of the core given the channel coordinates """

        square_sum = 0

        # these are the row/column coordinates  of the channel in question
        numbers = self.channel_coordinates(channel_no, channel_type)

        # Gets the dimensions of the core from the object, unless specified otherwise
        if utils.is_in(channel_type, "fuel"):

            if rows == "": rows = self.core_rows - 1
            if columns == "": columns = self.core_columns - 1

        else:

            if rows == "": rows = self.core_rows
            if columns == "": columns = self.core_columns

        core_dimensions = [rows, columns]

        # Sum of the squares of the distance of each coordinate
        for number, dimension in zip(numbers, core_dimensions):
            square_sum += (round(abs(number - self.padding - dimension / 2) + 0.5)) ** 2

        return math.sqrt(square_sum)

    def layers_from_centre(self, channel_no, channel_type="fuel", rows="", columns=""):
        """ Calculates the planar distance from the centre of the core given the channel coordinates """

        square_sum = 0

        # these are the row/column coordinates  of the channel in question
        numbers = self.channel_coordinates(channel_no, channel_type)

        # Gets the dimensions of the core from the object, unless specified otherwise
        if utils.is_in(channel_type, "fuel"):

            if rows == "": rows = self.core_rows - 1
            if columns == "": columns = self.core_columns - 1

        else:

            if rows == "": rows = self.core_rows
            if columns == "": columns = self.core_columns

        core_dimensions = [rows, columns]

        distances = []

        # Sum of the squares of the distance of each coordinate
        for number, dimension in zip(numbers, core_dimensions):
            distances.append(int(abs(number - self.padding - dimension / 2)))

        return max(distances)

    def results_2D(self):

        base_indices = self.results_indices
        indices_np = np.zeros([self.inter_rows, self.inter_columns, self.inter_levels])

        first_columns_row = self.first_columns_row_interstitial

        number_columns = self.inter_columns

        channel = 0

        for row, column_offset in enumerate(first_columns_row):

            number_columns_this_row = number_columns - (2 * column_offset)
            for column in range(column_offset, column_offset + number_columns_this_row):
                channel_value = np.array(base_indices[channel][0:13])
                indices_np[row, column] = channel_value
                channel += 1

        self.results_3D_array = indices_np

    def apply_augmentation(self, augmentation=None):

        if not augmentation:
            augmentation = self.augmentation

        self.results_2D()
        cracks = self.crack_array

        start, num = augmentation.split("_")

        if utils.is_in(start, "flip"):

            # Flip about the vertical axis
            if num == "1":
                cracks_rotated = np.rot90(cracks, -1, (1, 2))
                cracks_rotated_flipped = np.fliplr(cracks_rotated)
                self.crack_array = np.rot90(cracks_rotated_flipped, 1, (1, 2))

                indices = self.results_3D_array

                indices_rot = np.rot90(np.fliplr(indices), -1, (1, 0))

                self.reassign_indices(np.rot90(indices_rot, 1, (1, 0)))

            # Flip about the axis 45° from vertical
            if num == "2":
                pass

            # Flip about the horizontal axis
            if num == "3":
                self.crack_array = np.fliplr(cracks)
                indices_flipped = np.flipud(self.results_3D_array)
                self.reassign_indices(indices_flipped)

            # Flip about the axis 135° from vertical
            if num == "4":
                pass

        elif utils.is_in(start, "rot"):

            rotations = int(num)

            self.crack_array = np.rot90(cracks, -rotations, (1, 2))

            indices_rotated = np.rot90(self.results_3D_array, rotations, (1, 0))

            indices_new = []

            for row in indices_rotated:
                for channel in row:
                    if sum(channel) > 0:
                        indices_new.append(channel.astype(int))

            self.results_indices = indices_new

        else:
            print("Please choose at least one valid augmentation method.")

    def reassign_indices(self, indices_3d):

        indices_new = []

        for row in indices_3d:
            for channel in row:
                if sum(channel) > 0:
                    indices_new.append(channel.astype(int))

        self.results_indices = indices_new
