from machine_learning.experiment import Experiment
from experiment_input_files.trial_common_parameters import parameters
from machine_learning.models import RegressionModels as RegMods
from machine_learning.dataset_generators import Features3D as Features
from machine_learning.callbacks import LossHistory as History, lr_scheduler
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn import preprocessing as pre
import os
import ntpath

checkpoint_filepath = os.path.splitext(ntpath.basename(__file__))[0] + ".{epoch:02d}-{val_loss:.2f}.hdf5"


############################################
# Model Checkpoint
############################################

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

############################################

#lrate = LearningRateScheduler(lr_scheduler, verbose=1)
callbacks = [History, model_checkpoint_callback]

dataset = parameters.dataset

features = Features(parameters.dataset, array_type="Positions")

labels = parameters.labels

min_max_scaler = pre.MinMaxScaler(feature_range=(0, 1))

#features.transform(min_max_scaler)
#labels.transform(min_max_scaler)

model = RegMods.convolutional_neural_network_2d(features.feature_shape, labels.label_shape, activation="linear",
                                                layers=(64, 256, 384, 384, 256, 256, 256), padding="same", kernel_shape=[11, 5, 3, 3, 3])

experiment = Experiment(parameters, "3D_64_256_2x384_256_256_256_ks11_5_3_3_3", model, dataset, features, labels,
                        callbacks)
