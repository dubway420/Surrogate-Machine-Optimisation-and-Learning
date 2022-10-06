from os import listdir
import numpy as np


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


def find_nearest(array, value):
    array = np.asarray(array)
    diffs = (np.abs(array - value))
    return np.unravel_index(np.argmin(diffs), diffs.shape)


def value_parse(val):

    if val.isnumeric():
        return int(val)
    elif is_in(val, "_"):
        return val.split("_")
    else:
        print("Invalid value:", val)
        return None


def augmentation_handler(variables):
    flip = None
    rotate = None

    for variable in variables:

        try:
            type = variable.split("=")[0]
            value = variable.split("=")[1]

            if is_in(type, "flip"):
                flip = value_parse(value)

            elif is_in(type, "rot"):
                rotate = value_parse(value)

            else:
                print("Invalid augmentation type:", type)

        except IndexError:
            print("Invalid entry:", variable)
            print("augmentations need to be specified in the form <augmentation>=<value1>_<value2>_...\n")

    return flip, rotate


def experiment_number_finder(search_name, directory):
    """ Find the experiment number of the experiment with the given name in the given directory 
    
    search_name (string): this is the name of the experiment you are trying to find

    directory (string): this is the folder that contains the files to search through. This is usually 
    the parent file of the trial

    Returns: experiment number (integer), this is the experiment number of the experiment you are 
    trying to find. Taken from the file name i.e. if the search_name was found in experiment3.py, this
    method will return 3.

    """



    # Get the list of files in the directory
    path = directory + "/experiment_input_files"
    files = listdir(path)


    # Find the experiment number
    # Do this by iterating through the files in the experiment_input_files directory
    for file in files:

        # If the file has "experiment" in the title, then it is a experimentX.py files
        if is_in(file, "experiment"): 

            # Get just the number of the experiment i.e. remove the string "experiment" and the 
            # extension ".py"
            experiment_number = file.replace("experiment", "").replace(".py", "")

            # Read all lines of the experimentX.py file until the experiment name label is found
            with open(path + "/" + file, "r") as f:
                lines = f.readlines()

                for line in lines:

                    # This is the identifying line
                    if is_in(line, "experiment_name = "):

                        # Get just the name of the experiment and stop searching through this 
                        # particular file 
                        experiment_name = line.split("=")[1].strip()
                        break

            # If the file carries the correct experiment name, return the experiment number            
            if is_in(experiment_name, search_name):
                return experiment_number     
