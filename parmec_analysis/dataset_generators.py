from parmec_analysis.utils import cases_list
from parmec_analysis import reactor_case
import numpy as np


def features_and_labels_single_frame(path_string, x_type='positions', result_time=50, result_column="1",
                                     result_type='max', return_vector=None, flat_y=False):
    """ Gets the features and labels from the folder of results"""

    if return_vector is None:
        return_vector = [True, True]

    cases = cases_list(path_string)

    X, Y = [], []

    for case in cases:
        instance = reactor_case.Parse(case)

        if return_vector[0]:
            X.append(instance.linear_crack_array_1d(array_type=x_type))
        if return_vector[1]:
            Y.append(instance.get_result_at_time(result_time, result_columns=str(result_column),
                                                 result_type=result_type, flat=flat_y))

    Xnp = np.array(X)
    Ynp = np.array(Y)

    return Xnp.reshape(Xnp.shape[0], Xnp.shape[1], 1), Y
