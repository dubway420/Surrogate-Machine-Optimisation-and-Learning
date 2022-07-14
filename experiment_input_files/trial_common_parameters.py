from machine_learning.dataset_generators import DatasetSingleFrame, Displacements as Labels
from machine_learning.trial import TrialParameters
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from machine_learning.losses import adjusted_mse as loss
import numpy as np

###############################################################################
# ######## FILL THIS IS WITH PARAMETERS COMMON TO THE ENTIRE TRIAL ############
###############################################################################

trial_name = "CustomLossMeanAlpha7"
dataset_path = '~/training_data/cases_48_only/'

# Model parameters
epochs = 1500
opt = Nadam(0.00005)

plot_every_n_epochs = int(epochs / 2)  # by default, prints 4 times during training

# Features - Note this is only for decoration/record: these variables affect nothing
channels_features = 'all'
levels_features = 'all'
array_type = 'Positions'

# Labels
channels_labels = "160"
levels_labels = '12'
result_type = 'all'
result_time = 48
result_column = 1

dataset = DatasetSingleFrame(validation_split=0.1)

#dataset.augment(flip=(3, ), rotate=(), retain_validation=True)

#dataset.roll(5)

save_model = True

labels = Labels(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
                #levels=levels_labels, unit="millimeters")
                levels=levels_labels)
############################

mean = np.mean(labels.values)

loss = loss(mean, alpha=7)

parameters = TrialParameters(trial_name, dataset, labels, epochs, opt, loss, plot_every_n_epochs, save_model)
