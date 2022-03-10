from machine_learning.experiment import Experiment
# from experiment_input_files.trial_common_parameters import parameters
from machine_learning.models import RegressionModels as RegMods
from machine_learning.dataset_generators import Cracks3D as Features
from machine_learning.callbacks import TrainingProgress as History, lr_scheduler
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing as pre
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense


def experiment(trial):
    package = trial + ".experiment_input_files.trial_common_parameters"

    parameters = getattr(__import__(package, fromlist=["parameters"]), "parameters")

    experiment_name = "thinning_256to16_DOp4_tanh_softmax_FC256_lvls_3_7_nopadding"

    callbacks = [History]

    dataset = parameters.dataset

    features = Features(parameters.dataset, levels='3-7', array_type="Positions")

    labels = parameters.labels

    min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

    # features.transform(min_max_scaler)
    labels.transform(min_max_scaler)

    batch_size = 32

    model = RegMods.convolutional_neural_network_2d(features.feature_shape, labels.label_shape, activation=(
        "tanh", "softplus", "tanh", "softplus", "tanh", "softplus", "tanh"),
                                                    layers=(256, 64, 32, 16, 16, 64, 256),
                                                    dropout=0.4, padding="same",
                                                    kernel_shape=3)

    return Experiment(parameters, experiment_name, model, dataset, features, labels, batch_size,
                      callbacks)