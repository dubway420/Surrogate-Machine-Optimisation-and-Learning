from machine_learning.dataset_generators import DatasetSingleFrame, Labels
from machine_learning.trial import TrialParameters
from keras.optimizers import Adam
from keras.losses import mean_squared_error as loss

###############################################################################
# ######## FILL THIS IS WITH PARAMETERS COMMON TO THE ENTIRE TRIAL ############
###############################################################################

trial_name = "kirstie"
dataset_path = '~/localscratch/'

# Model parameters
epochs = 50000
opt = Adam(lr=1e-3)
loss = loss
plot_every_n_epochs = int(epochs / 4)  # by default, prints 4 times during training

# Features
channels_features = 'all'
levels_features = 'all'
array_type = 'Positions Only'

# Labels
channels_labels = 'all'
levels_labels = 'all'
result_type = 'all'
result_time = 48
result_column = 1

no_instances = 'all'

dataset = DatasetSingleFrame(dataset_path, number_of_cases=no_instances)

labels = Labels(dataset, result_time=result_time, result_type=result_type)

############################

parameters = TrialParameters(trial_name, dataset, labels, epochs, opt, loss, plot_every_n_epochs)