from machine_learning.experiment import Experiment
# from experiment_input_files.trial_common_parameters import parameters
from machine_learning.models import RegressionModels as RegMods
from machine_learning.dataset_generators import CracksPlanar as Features
from machine_learning.callbacks import TrainingProgress as History, lr_scheduler
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing as pre
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense


def experiment(trial):

    package = trial + ".experiment_input_files.trial_common_parameters"

    parameters = getattr(__import__(package, fromlist=["parameters"]), "parameters")

    experiment_name = "VGG19_DOp3"

    callbacks = [History]

    dataset = parameters.dataset

    features = Features(dataset, one_hot=True)

    labels = parameters.labels

    min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

    # features.transform(min_max_scaler)
    labels.transform(min_max_scaler)

    extra_layers = (Dropout(0.3),)
    model = RegMods.vgg_model((88, 44, 3), 1, final_bias=True, extra_layers=extra_layers)

    return Experiment(parameters, experiment_name, model, dataset, features, labels,
                            callbacks)
