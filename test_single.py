import pickle
from tensorflow.keras import models
from machine_learning.dataset_generators import DatasetSingleFrame, Cracks1D, Cracks3D, CracksPlanar, Displacements, augmentation_string
from machine_learning.callbacks import correlation_foursquare, histogram_foursquare
from tensorflow.keras.losses import mean_squared_error as mse
import numpy as np
from os import listdir, mkdir
from machine_learning.utils import is_in, augmentation_handler, experiment_number_finder
import sys
from openpyxl import load_workbook

variables = sys.argv[1:]

path = variables[0]

# If the last character of path is a slash remove it
if path[-1] == "/":
    path = path[:-1]


name = path.split("/")[-2] + "_" + path.split("/")[-1]

experiment_name = path.split("/")[-1]

trial_name = path.split("/")[-2]

exp_number = experiment_number_finder(experiment_name, trial_name)

package = trial_name + ".experiment_input_files.experiment" + exp_number

experiment = getattr(__import__(package, fromlist=["experiment"]), "experiment")(trial_name)

dataset = experiment.dataset
inputs = experiment.features
labels = experiment.labels
# dataset = DatasetSingleFrame(name="test_set")

# #inputs = CracksPlanar(dataset, extra_dimension=True, levels="5-7")
# inputs = Cracks3D(dataset, array_type="Positions", levels="5-7")
# #inputs = Cracks3D(dataset, array_type="Orient", levels="5-7")
# #inputs = Cracks1D(dataset, array_type="Positions", levels="5-7")

# # Labels
# channels_labels = "160"
# levels_labels = '12'
# result_type = 'all'
# result_time = 48
# result_column = 1

# no_instances = 'all'

# labels = Displacements(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
#                        # levels=levels_labels, unit="millimeters")
#                        levels=levels_labels)

transformer_name = labels.generate_filename(ext=".tfr")

transformer_name_dataset = transformer_name.replace("test_set", "dataset")


if len(variables) >= 2:

    flip, rotate = augmentation_handler(variables[1:])

    aug_name = augmentation_string(flip, rotate)

    aug_name += ".tfr"

    transformer_name_dataset = transformer_name_dataset.replace(".tfr", aug_name)

# Remove any existing transformation
labels.inverse_transform()

print("Transformer name: ", transformer_name_dataset)


with open(transformer_name_dataset, 'rb') as f:
    transformer = pickle.load(f)
    
    # min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))
labels.transform(transformer, fit=False, save=False)


# if name == "":
    # name = path.split("/")[-2] + "_" + path.split("/")[-1]

folder_name = "TEST_" + name


files = listdir(path)

model_files = []
model_losses = []

mod_file_found = False
# loop trough all files in the folder and check if any of them are .mod files
for file in files:
    if is_in(file, ".mod"):
        mod_file_found = True
        model_files.append(file)

if not mod_file_found:
    print("No .mod file found in the folder")
    sys.exit()         


    
try:
    mkdir(folder_name)

except FileExistsError:
    print("Please be aware that a folder already exists by the name " + folder_name)
    print("Continuing...\n")

for i, model_name in enumerate(model_files):

    model_path = path + "/" + model_name

    model = models.load_model(model_path, compile=False)

    predictions = model.predict(inputs.values)

    error = mse(predictions, labels.values)

    mse_iteration = np.mean(error)
    message = str(i) + ": " + model_name + ", MSE: " + str(mse_iteration)
    print(message)

    model_losses.append(mse_iteration)

    save_name = folder_name + "/" + model_name.split('.')[0]

    correlation_foursquare(("100"), 56, labels.values, [1, predictions], save_name)
    histogram_foursquare(("100"), 56, labels.values, [1, predictions], save_name)

    del model, predictions, error


mean_error = round(np.mean(model_losses), 4)

print("\n ------------------ \n ")
print("Mean error:", mean_error)

best_error = np.min(model_losses)
best_i = np.argmin(model_losses)
best_model = model_files[best_i]

message = "Best result for model: " + best_model + " (" + str(best_i) + ")"
print(message)
print("Result: ", round(best_error, 4))


filename = folder_name + "/results"
np.save(filename, model_losses)

wb = load_workbook('results_summary.xlsx')

ws = wb.active

ws.append([name, best_error, mean_error, best_model])

# Save the file
wb.save("results_summary.xlsx")

# with open("results_summary.txt", "a") as f:
#     f.write(name + "\n")
#     f.write(message + "\n")
#     f.write("Best: " + str(best_error) + "\n")
#     f.write("Mean: " + str(mean_error) + "\n")
#     f.write("\n-----------------\n")
#     f.close()
