from tensorflow.keras import models
from machine_learning.dataset_generators import DatasetSingleFrame, Cracks3D, Displacements
from machine_learning.callbacks import correlation_foursquare
from tensorflow.keras.losses import mean_squared_error as mse
import numpy as np
from sklearn import preprocessing as pre

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

for i in range(6, 7):

    roll = "Roll" + str(i)

    path = "C:/Users/Huw/Documents/PhD/Paper1Models/" + roll

    path += "/thinning_256to16_DOp4_tanh_softmax_FC256_lvls_3_7_nopadding"

    folder_name = "Paper1Results" + str(i) + "/"

    mkdir(folder_name)

    files = listdir(path)

    model_files = []

    for file in files:

        if is_in(file, ".mod"):
            model_files.append(file)

    for model_name in model_files:
        print(model_name)

        model_path = path + "/" + model_name

        model = models.load_model(model_path)

        predictions = model.predict(inputs.values)
        print(predictions)

        error = mse(predictions, labels.values)

        print(model_name, " MSE: ", np.mean(error))

        save_name = folder_name + model_name.split('.')[0]
        correlation_foursquare(("100"), 56, labels.values, [1, predictions], save_name)

        del model, predictions, error
