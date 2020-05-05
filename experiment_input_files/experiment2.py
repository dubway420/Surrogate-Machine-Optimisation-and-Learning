from machine_learning.experiment import Experiment
from machine_learning.model_training import run_experiment
from experiment_input_files.trial_common_parameters import parameters
from machine_learning.models import RegressionModels as RegMods
from machine_learning.dataset_generators import Features1D as Features

features = Features(parameters.dataset, extra_dimension=False)

labels = parameters.labels

model = RegMods.multi_layer_perceptron(features.feature_shape, labels.label_shape, activation="linear",
                                       layers=(32, 16, 8))

experiment = Experiment(parameters, "MLP_32_16_8_linear", model, parameters.dataset, features, labels)
