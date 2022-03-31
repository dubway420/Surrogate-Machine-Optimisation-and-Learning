from tensorflow.keras import models
from machine_learning.dataset_generators import DatasetSingleFrame, Cracks3D, Displacements
from machine_learning.callbacks import correlation_foursquare, histogram_foursquare
from tensorflow.keras.losses import mean_squared_error as mse
import numpy as np
from sklearn import preprocessing as pre
import warnings

warnings.filterwarnings('ignore')

from os import listdir, mkdir
from parmec_analysis.utils import is_in

dataset = DatasetSingleFrame(name="test_set")

inputs = Cracks3D(dataset, levels='3-7', array_type="Positions")

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

fold_losses = []

fold_best_model = []

trial_name = "Paper1"

mkdir(trial_name)

for i in range(0, 8):

    roll = "Roll" + str(i)

    path = "C:/Users/Huw/Documents/PhD/Paper1Models/" + roll

    path += "/thinning_256to16_DOp4_tanh_softmax_FC256_lvls_3_7_nopadding"

    folder_name = trial_name + "/" + roll + "/"

    mkdir(folder_name)

    files = listdir(path)

    model_files = []
    model_losses = []

    for file in files:

        if is_in(file, ".mod"):
            model_files.append(file)

    print("Fold: ", i)

    for j, model_name in enumerate(model_files):
        model_path = path + "/" + model_name

        model = models.load_model(model_path)

        predictions = model.predict(inputs.values)

        error = mse(predictions, labels.values)

        mse_iteration = np.mean(error)
        message = str(j) + ": " + model_name + ", MSE: " + str(mse_iteration)
        print(message)

        model_losses.append(mse_iteration)

        save_name = folder_name + model_name.split('.')[0]
        correlation_foursquare(("100"), 56, labels.values, [1, predictions], save_name)
        histogram_foursquare(("100"), 56, labels.values, [1, predictions], save_name)

        del model, predictions, error

    fold_losses.append(model_losses)

    mean_error = round(np.mean(model_losses), 4)

    print("\n ------------------ \n ")
    print("Mean error:", mean_error)

    best_error = np.min(model_losses)
    best_j = np.argmin(model_losses)
    best_model = model_files[best_j]

    message = "Best result for model: " + best_model + " (" + str(best_j) + ")"
    print(message)
    print("Result: ", round(best_error, 4))

    fold_best_model.append(best_model)

    print("\n ------------------ \n ")

print("\n ########################################## \n")
print("\n ########################################## \n")

best_fold_losses = []
losses_flat = []

for fold in fold_losses:
    best_fold_losses.append(np.min(fold))
    losses_flat.extend(fold)

best_loss_overall = np.min(best_fold_losses)

best_fold = np.argmin(best_fold_losses)

losses_for_best_fold = fold_losses[best_fold]

best_model_no = np.argmin(losses_for_best_fold)

best_model = fold_best_model[best_fold]

print("Best result overall: ", str(best_loss_overall))
message = "This was for fold: " + str(best_fold) + " model: " + best_model + " (" + str(best_model_no) + ")"
print(message)

print("\n---\n")

print("Mean Loss: ", np.mean(losses_flat))


