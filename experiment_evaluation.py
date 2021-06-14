from parmec_analysis.utils import load_results
from parmec_analysis.visualisation import model_comparison
import numpy as np
import sys
from datetime import datetime

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

experiment_validation_losses_min = np.zeros(len(names))
experiment_validation_losses_std = np.zeros(len(names))

for i, exp in enumerate(experiment_validation_losses):
    experiment_validation_losses_min[i] = np.min(exp)
    experiment_validation_losses_std[i] = np.std(exp)

# experiment_validation_losses_min = np.min(experiment_validation_losses, axis=1)

sort_args = np.argsort(experiment_validation_losses_min)

result_mins_sorted = experiment_validation_losses_min[sort_args]
result_means_sorted = averages_validation[sort_args]
result_std_sorted = experiment_validation_losses_std[sort_args]

names_np = np.array(names)
names_sorted = names_np[sort_args]

summary_file = trial_name + ".txt"

f = open(summary_file, "a")

header = "Summary of trial: " + trial_name + "\n"
subhead = "Trial completed: " + datetime.now().strftime("%A %d. %B %Y - %H:%M")

f.write("\n----------------------------\n")

f.write(header)
f.write(subhead)

f.write("\n----------------------------\n")

prev_result = 0

for name, result, mean, std in zip(names_sorted, result_mins_sorted, result_means_sorted, result_std_sorted):


    if prev_result != 0:
        diff = result - prev_result
        percent = (diff / prev_result) * 100
        string = name + " " + str(round(result, 6)) + " +" + str(round(diff, 6)) + " (" + str(round(percent, 3)) + "%)"
    else:
        string = name + " " + str(round(result, 6))

    extras = " [m: " + str(round(mean, 6)) + ", std: " + str(round(std, 6)) + "] \n"

    f.write((string + extras))

    prev_result = result

model_comparison(names, averages_training, averages_validation,
                 errors_training, errors_validation, 'Loss', trial_name)
