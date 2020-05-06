from machine_learning.experiment import Experiment
from experiment_input_files.trial_common_parameters import parameters
from machine_learning.models import RegressionModels as RegMods
from machine_learning.dataset_generators import Features1D as Features

dataset = parameters.dataset

features = Features(parameters.dataset, extra_dimension=False)

labels = parameters.labels

model = RegMods.multi_layer_perceptron(features.feature_shape, labels.label_shape, activation="relu",
                                       layers=(32, 16, 8))

experiment = Experiment(parameters, "MLP_32_16_8_relu", model, dataset, features, labels)
