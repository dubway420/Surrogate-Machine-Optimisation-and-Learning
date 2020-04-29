from parmec_analysis.utils import load_results
from parmec_analysis.visualisation import model_comparison
import numpy as np

results_file_name = "second_test.ind"
results_dict = load_results(results_file_name)

names = list(results_dict.keys())

averages_training = np.zeros(len(names))
averages_validation = np.zeros(len(names))

errors_training = np.zeros([2, len(names)])
errors_validation = np.zeros([2, len(names)])

results = results_dict.values()

for i, result in enumerate(results):
    np_array = np.array(result)  # convert to np array

    names[i] = names[i] + " (" + str(len(np_array)) + ")"

    training_loss = np_array[:, 0]
    validation_loss = np_array[:, 1]

    training_mean = np.mean(training_loss)
    validation_mean = np.mean(validation_loss)

    averages_training[i] = training_mean
    averages_validation[i] = validation_mean

    errors_training[0:2, i] = np.abs(np.array([np.min(training_loss), np.max(training_loss)]) - training_mean)

    errors_validation[0:2, i] = np.abs(np.array([np.min(validation_loss), np.max(validation_loss)]) - validation_mean)

model_comparison(names, averages_training, averages_validation,
                 errors_training, errors_validation, 'Mean Squared Error')
