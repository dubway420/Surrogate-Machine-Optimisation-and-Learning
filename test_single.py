from tensorflow.keras import models
from machine_learning.dataset_generators import DatasetSingleFrame, Cracks3D, Displacements
from machine_learning.callbacks import correlation_foursquare, histogram_foursquare
from tensorflow.keras.losses import mean_squared_error as mse
import numpy as np
from sklearn import preprocessing as pre

from os import listdir, mkdir
from parmec_analysis.utils import is_in

dataset = DatasetSingleFrame(name="test_set")

inputs = Cracks3D(dataset, array_type="Positions", levels="5-7")

print(inputs.summary())

# Labels
channels_labels = "160"
levels_labels = '12'
result_type = 'all'
result_time = 48
result_column = 1

no_instances = 'all'

labels = Displacements(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
                       # levels=levels_labels, unit="millimeters")
                       levels=levels_labels)

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))
labels.transform(min_max_scaler)


#########################################################
# Replace this with the path to your download of the experiment
#########################################################
path = "C:/Users/Huw/Documents/MobaXterm/home/AugDown/VaryingAugRot2/levels5_7_simple_DO2"

#########################################################
# Replace this with the name of the folder you want to save your test results to
#########################################################
folder_name = "VaryingAugRot2"

try:
    mkdir(folder_name)

except FileExistsError:
    print("Please be aware that a folder already exists by the name " + folder_name)
    print("Continuing...\n")

files = listdir(path)

model_files = []
model_losses = []

for file in files:

    if is_in(file, ".mod"):
        model_files.append(file)

for i, model_name in enumerate(model_files):

    model_path = path + "/" + model_name

    model = models.load_model(model_path)

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

print("/n ------------------ /n ")
print("Mean error:", mean_error)

best_error = np.min(model_losses)
best_i = np.argmin(model_losses)
best_model = model_files[best_i]

message = "Best result for model: " + best_model + " (" + str(best_i) + ")"
print(message)
print("Result: ", round(best_error, 4))


filename = folder_name + "/results"
np.save(filename, model_losses)
