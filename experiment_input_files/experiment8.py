from machine_learning.experiment import Experiment
from experiment_input_files.trial_common_parameters import parameters
from machine_learning.models import RegressionModels as RegMods
from machine_learning.dataset_generators import Features3D as Features
from machine_learning.callbacks import LossHistory as History, lr_scheduler
#from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing as pre


callbacks = [History]

dataset = parameters.dataset

features = Features(parameters.dataset, array_type="Positions")

labels = parameters.labels

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

#features.transform(min_max_scaler)
#labels.transform(min_max_scaler)

model = RegMods.convolutional_neural_network_2d(features.feature_shape, labels.label_shape, activation="linear",
                                                layers=(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 128, 256), padding="same", kernel_shape=3)

experiment = Experiment(parameters, "CNN_3D_38x32_128_256", model, dataset, features, labels,
                        callbacks)