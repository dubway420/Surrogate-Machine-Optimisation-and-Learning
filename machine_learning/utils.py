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
