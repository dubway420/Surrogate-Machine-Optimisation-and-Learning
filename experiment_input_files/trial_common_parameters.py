from machine_learning.dataset_generators import DatasetSingleFrame, Displacements as Labels
from machine_learning.trial import TrialParameters
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from machine_learning.losses import huber_loss as loss

###############################################################################
# ######## FILL THIS IS WITH PARAMETERS COMMON TO THE ENTIRE TRIAL ############
###############################################################################

trial_name = "batch_test"
dataset_path = '~/training_data/cases_48_only/'

# Model parameters
epochs = 1500
opt = Nadam(0.00005)
loss = loss
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

#dataset.augment(retain_validation=True)

#dataset.roll(9)

save_model = True

labels = Labels(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
                #levels=levels_labels, unit="millimeters")
                levels=levels_labels)
############################

parameters = TrialParameters(trial_name, dataset, labels, epochs, opt, loss, plot_every_n_epochs, save_model)
