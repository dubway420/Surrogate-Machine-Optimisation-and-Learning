from keras.callbacks import Callback
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from keras.losses import mean_absolute_percentage_error
from parmec_analysis.utils import load_results
from parmec_analysis.visualisation import CoreView, TrainingHistoryRealTime
from experiment_setup import list_of_experiments as experiments
import sys
import os
import pickle

# START HERE ****************************************
#####################################################
# Model parameters ##################################
#####################################################

epochs = 10001
opt = Adam(lr=1e-3, decay=1e-3 / 200)
loss = mean_absolute_percentage_error
plot_every_n_epochs = int(epochs / 4)  # by default, prints 4 times during training

#####################################################

NUMCORES = int(os.getenv("NSLOTS", 1))

sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
                                        allow_soft_placement=True,
                                        device_count={'CPU': NUMCORES}))

# Set the Keras TF session
K.set_session(sess)


class LossHistory(Callback):
    def __init__(self, loss_function, trial, experiment, iteration, print_every_n_epochs=None):
        super().__init__()
        self.losses = []
        self.val_losses = []

        self.loss_function = loss_function
        self.trial = trial
        self.experiment = experiment
        self.iteration = iteration

        if not print_every_n_epochs:
            print_every_n_epochs = 1000

        self.plot_every_n_epochs = print_every_n_epochs

        self.view = CoreView(trial, iteration, experiment)
        self.train_history = TrainingHistoryRealTime(trial, iteration, experiment, loss_function, 500)

    def on_epoch_end(self, epoch, logs={}):

        self.train_history.update_data(logs, self.model, plot=False)

        if epoch > 0 and (epoch % self.plot_every_n_epochs) == 0:
            self.view.update_data(epoch, self.model)
            self.train_history.plotting()


experiment_number = sys.argv[-2]

trial_name = sys.argv[-1]

try:

    # TODO Move experiments to another file also

    experiment_selected = experiments[int(experiment_number)]

except IndexError:

    print("You have entered an experiment number,", experiment_number, ", which is invalid. Please enter an integer"
                                                                       " corresponding to one of the following: ")

    for i, exp in enumerate(experiments):
        print(i, exp.name)
    sys.exit()

results_file_name = trial_name + ".ind"
results_dict = load_results(results_file_name)

exp_name = experiment_selected.name

if exp_name in results_dict:
    exp_i = len(results_dict[exp_name])
else:
    exp_i = 0

history = LossHistory(loss, trial_name, experiment_selected,  exp_i, plot_every_n_epochs)

model_i = experiment_selected.model(experiment_selected.features.feature_shape, experiment_selected.labels.label_shape)

model_i.compile(loss=loss.__name__, optimizer=opt)

model_fit = model_i.fit(experiment_selected.features.training_set(), experiment_selected.labels.training_set(),
                        validation_data=(experiment_selected.features.validation_set(),
                                         experiment_selected.labels.validation_set()),
                        epochs=epochs, verbose=0, callbacks=[history])

losses = (model_fit.history['loss'][-1], model_fit.history['val_loss'][-1])

if exp_i == 0:
    results_dict[exp_name] = [losses]

else:
    results_dict[exp_name].append(losses)

with open(results_file_name, 'wb') as f:
    pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
