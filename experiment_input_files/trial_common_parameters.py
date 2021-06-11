from machine_learning.dataset_generators import DatasetSingleFrame, Displacements as Labels
from machine_learning.trial import TrialParameters
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.losses import mean_squared_error as loss

###############################################################################
# ######## FILL THIS IS WITH PARAMETERS COMMON TO THE ENTIRE TRIAL ############
###############################################################################

trial_name = "Reanalysis1"
dataset_path = '~/'

# Model parameters
epochs = 300
opt = Nadam(0.0001)
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

no_instances = 'all'

dataset = DatasetSingleFrame(dataset_path, number_of_cases=no_instances)

# '

labels = Labels(dataset, channels=channels_labels, result_time=result_time, result_type=result_type,
                #levels=levels_labels, unit="millimeters")
                levels=levels_labels)   
############################

parameters = TrialParameters(trial_name, dataset, labels, epochs, opt, loss, plot_every_n_epochs)
