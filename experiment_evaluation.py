from parmec_analysis.utils import load_results
from parmec_analysis.visualisation import model_comparison
import numpy as np
import sys


trial_name = str(sys.argv[-1])
results_file_name = trial_name + ".ind"
results_dict = load_results(results_file_name)

names = list(results_dict.keys())

averages_training = np.zeros(len(names))
averages_validation = np.zeros(len(names))

errors_training = np.zeros([2, len(names)])
errors_validation = np.zeros([2, len(names)])

results = results_dict.values()

experiment_training_losses = []
experiment_validation_losses = []

for i, result in enumerate(results):
    np_array = np.array(result)  # convert to np array

    names[i] = names[i] + " (" + str(len(np_array)) + ")"

    training_loss = np_array[:, 0]
    validation_loss = np_array[:, 1]

    experiment_training_losses.append(training_loss)
    experiment_validation_losses.append(validation_loss)

    training_mean = np.mean(training_loss)
    validation_mean = np.mean(validation_loss)

    averages_training[i] = training_mean
    averages_validation[i] = validation_mean

    errors_training[0:2, i] = np.abs(np.array([np.min(training_loss), np.max(training_loss)]) - training_mean)

    errors_validation[0:2, i] = np.abs(np.array([np.min(validation_loss), np.max(validation_loss)]) - validation_mean)


experiment_validation_losses_min = np.min(experiment_validation_losses, axis=1)

sort_args = np.argsort(experiment_validation_losses_min)

result_mins_sorted = experiment_validation_losses_min[sort_args]

names_np = np.array(names)
names_sorted = names_np[sort_args]

summary_file = trial_name + ".txt"

f = open(summary_file, "a")

prev_result = 0

for name, result in zip(names_sorted, result_mins_sorted):

    if prev_result != 0:
        diff = result - prev_result
        percent = (diff/prev_result)*100
        string = name + " " + str(result) + " +" + str(diff) + " (" + str(percent) + "%)" + "\n"
    else:
        string = name + " " + str(result) + "\n"

    f.write(string)

    prev_result = result

model_comparison(names, averages_training, averages_validation,
                 errors_training, errors_validation, 'Loss', trial_name)
