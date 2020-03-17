from parmec_analysis.utils import cases_list, is_in, split_separators
from parmec_analysis import reactor_case
import numpy as np
import warnings


def features_and_labels_single_frame(path_string, return_request='both',  # General terms
                                     x_request='1d', x_type='positions',  # Feature terms
                                     channels='all', levels='all',  # Data slice
                                     result_time=50, result_column="1", result_type="max",  # Label terms
                                     flat_y=False):  # Only relevant
    # if result_type
    # is "all"
    """ Gets the features and labels from the folder of results"""

    instance = None
    return_X = None

    # A switch for whether the user wants features, labels or both.

    return_vector = [False, False]

    if is_in(return_request, 'both', 'all'):
        return_vector = [True, True]
    elif is_in(return_request, 'feat'):
        return_vector[0] = True
    elif is_in(return_request, 'lab'):
        return_vector[1] = True

    # Output lists
    X_1d, X_3d, Y = [], [], []

    # A switch for which feature arrays are required i.e. 1d, 2d or 3d, or any combination

    # Remove any irrelevant terms from x_request
    for split_separator in split_separators:
        x_request = x_request.replace(split_separator, "")

    # search the request string for each of the three dimensions
    features_requested = [is_in(x_request, str(i + 1) + 'd') for i in range(3)]

    cases = cases_list(path_string)[0:10]

    for case in cases:

        try:
            instance = reactor_case.Parse(case)

            # Features
            if return_vector[0]:

                # If either the 1d or 2d feature is requested return the 1d array (both array types are dependent on 1d)
                if features_requested[0] or features_requested[1]:
                    X_1d.append(instance.crack_array_1d(channels=channels, array_type=x_type, levels=levels))

                if features_requested[2]:
                    X_3d.append(instance.channel_array_argument_wrapper(array_type=x_type, levels=levels,
                                                                        quiet=True))

            if return_vector[1]:
                Y.append(instance.get_result_at_time(result_time, result_columns=str(result_column),
                                                     result_type=result_type, flat=flat_y))

        # If there's any problems with the files in a case, it completely skips to the next
        except FileNotFoundError:
            message = "Warning: Case " + case + " was not found. Skipping this case"
            warnings.warn(message)

    no_fuel_channels = instance.fuel_channels
    no_fuel_levels = instance.core_levels

    # If features are requested, generate a list of them
    if return_vector[0]:
        X = []

        # Convert 1D array to numpy and then reshape to single channel
        X_1d_np = np.array(X_1d)
        X_1d_np = X_1d_np.reshape([X_1d_np.shape[0], X_1d_np.shape[1], 1])

        if features_requested[0]: X.append(X_1d_np)
        if features_requested[1]: X.append(X_1d_np.reshape(X_1d_np.shape[0], no_fuel_channels, no_fuel_levels, 1))
        if features_requested[2]: X.append(np.array(X_3d))

        # Finalise the feature variable
        if len(X) == 1:
            return_X = X[0]
        else:
            return_X = X

    # If both features and labels are requested:
    if return_vector[0] and return_vector[1]:
        return return_X, np.array(Y)

    # Features only:
    elif return_vector[0]:
        return return_X

    # Labels only
    elif return_vector[1]:
        return np.array(Y)

    # If nethier was requested, warn the user
    else:
        warnings.warn("Warning: You have not requested any output. Please specificy something in the ")


